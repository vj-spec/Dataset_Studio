"""Unified local compute scheduling and authenticated worker monitoring services."""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
import secrets
import socket
import ssl
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

from PySide6.QtCore import QObject, QThread, Signal

from config import app_state_file
from experiment_database import ExperimentDatabase
from hardware import detect_hardware


MODES = ("Controller", "Worker", "Hybrid")
MAX_MESSAGE_BYTES = 16 * 1024 * 1024
MAX_REQUEST_BYTES = 1024 * 1024


def utc_now():
    return datetime.now(timezone.utc)


def _worker_secret_hash(value: str) -> bytes:
    return hashlib.sha256(value.encode("utf-8")).digest()


@dataclass
class ResourceProfile:
    max_concurrent_cpu_jobs: int = 1
    max_concurrent_gpu_jobs: int = 1
    cpu_threads_per_job: int = 4
    ram_safety_percent: int = 20
    minimum_free_disk_gb: float = 10.0
    gpu_memory_safety_mb: int = 2048
    monitoring_interval_seconds: int = 3

    @classmethod
    def from_dict(cls, values):
        allowed = cls.__dataclass_fields__
        return cls(**{key: values[key] for key in allowed if key in values})

    def validate(self):
        if not 1 <= self.max_concurrent_cpu_jobs <= 32: raise ValueError("Concurrent CPU job limit must be between 1 and 32.")
        if not 1 <= self.max_concurrent_gpu_jobs <= 16: raise ValueError("Concurrent GPU job limit must be between 1 and 16.")
        if not 1 <= self.cpu_threads_per_job <= 256: raise ValueError("CPU threads per job must be between 1 and 256.")
        if not 0 <= self.ram_safety_percent <= 90: raise ValueError("RAM safety threshold must be between 0 and 90 percent.")
        if self.minimum_free_disk_gb < 0: raise ValueError("Minimum free disk space cannot be negative.")
        if self.gpu_memory_safety_mb < 0: raise ValueError("GPU memory margin cannot be negative.")
        if not 1 <= self.monitoring_interval_seconds <= 60: raise ValueError("Monitoring interval must be between 1 and 60 seconds.")
        return self


class DPAPICredentialStore:
    """Current-user Windows DPAPI credential store; fails closed elsewhere."""
    ENTROPY = b"Dataset Studio worker credential v1"

    class _Blob(ctypes.Structure):
        _fields_ = [("cbData", ctypes.c_ulong), ("pbData", ctypes.POINTER(ctypes.c_ubyte))]

    @classmethod
    def _crypt(cls, data: bytes, protect: bool):
        if os.name != "nt":
            raise RuntimeError("Secure worker credential storage is available only through Windows DPAPI on this platform.")
        crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        input_buffer = ctypes.create_string_buffer(data)
        entropy_buffer = ctypes.create_string_buffer(cls.ENTROPY)
        source = cls._Blob(len(data), ctypes.cast(input_buffer, ctypes.POINTER(ctypes.c_ubyte)))
        entropy = cls._Blob(len(cls.ENTROPY), ctypes.cast(entropy_buffer, ctypes.POINTER(ctypes.c_ubyte)))
        output = cls._Blob()
        if protect:
            ok = crypt32.CryptProtectData(ctypes.byref(source), "Dataset Studio", ctypes.byref(entropy), None, None, 0, ctypes.byref(output))
        else:
            ok = crypt32.CryptUnprotectData(ctypes.byref(source), None, ctypes.byref(entropy), None, None, 0, ctypes.byref(output))
        if not ok: raise ctypes.WinError(ctypes.get_last_error())
        try: return ctypes.string_at(output.pbData, output.cbData)
        finally: kernel32.LocalFree(output.pbData)

    def __init__(self, root: Path | None = None):
        self.root = root or app_state_file("worker_credentials")
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, worker_id):
        return self.root / (hashlib.sha256(worker_id.encode("utf-8")).hexdigest() + ".bin")

    def save(self, worker_id: str, credential: str):
        encrypted = self._crypt(credential.encode("utf-8"), True)
        path = self._path(worker_id); temporary = path.with_suffix(".tmp")
        temporary.write_bytes(encrypted); os.replace(temporary, path)

    def load(self, worker_id: str):
        path = self._path(worker_id)
        if not path.is_file(): return None
        return self._crypt(path.read_bytes(), False).decode("utf-8")

    def delete(self, worker_id: str):
        self._path(worker_id).unlink(missing_ok=True)


class HardwareMonitorWorker(QThread):
    snapshot = Signal(object)
    failed = Signal(str)
    def __init__(self, interval_seconds=3, parent=None):
        super().__init__(parent); self.interval_seconds = max(1, int(interval_seconds)); self._stop = threading.Event()
    def stop(self): self._stop.set()
    def run(self):
        while not self._stop.is_set():
            try:
                data = detect_hardware()
                data["gpus"] = [gpu.__dict__ if hasattr(gpu, "__dict__") else gpu for gpu in data.get("gpus", [])]
                self.snapshot.emit(data)
            except Exception as exc: self.failed.emit(str(exc))
            self._stop.wait(self.interval_seconds)


class LocalJobScheduler(QObject):
    """Priority queue and reservations for local training workers; remote jobs are not dispatched."""
    job_changed = Signal(object)
    scheduling_message = Signal(str)

    def __init__(self, database: ExperimentDatabase, profile: ResourceProfile | None = None,
                 hardware_provider=detect_hardware, parent=None):
        super().__init__(parent); self.database = database; self.profile = profile or ResourceProfile()
        self.hardware_provider = hardware_provider; self._workers = {}; self._states = {}; self._launchers = {}
        self._lock = threading.RLock(); self._timer = None
        self.profile.validate()
        self.reconcile_after_restart()

    def reconcile_after_restart(self):
        for job in self.database.list_compute_jobs():
            if job["state"] == "queued":
                self.database.set_compute_job_state(job["id"], "failed", error="Queued job requires manual resubmission after application restart.")
                self.database.release_compute_resources(job["id"])

    def start_pump(self, interval_ms=500):
        from PySide6.QtCore import QTimer
        if self._timer is None:
            self._timer = QTimer(self); self._timer.setInterval(max(100, int(interval_ms))); self._timer.timeout.connect(self.pump); self._timer.start()

    def stop_pump(self):
        if self._timer: self._timer.stop()

    def active_count(self):
        return sum(1 for job in self.database.list_compute_jobs() if job["state"] in ("preparing", "running", "validating"))

    def submit(self, *, architecture: str, requirements: dict, preferred_device="auto", priority=0,
               launcher=None, experiment_id=None, dedupe_key=None):
        job = {"id": uuid.uuid4().hex, "experiment_id": experiment_id, "architecture": architecture,
               "requirements": dict(requirements), "preferred_device": preferred_device, "priority": int(priority),
               "dedupe_key": dedupe_key}
        record = self.database.create_compute_job(job)
        self._launchers[record["id"]] = launcher
        self.job_changed.emit(record); self.pump()
        return record["id"]

    def set_profile(self, profile: ResourceProfile):
        profile.validate()
        if self.active_count() > 0:
            raise RuntimeError("Resource settings cannot change while local jobs are active.")
        self.profile = profile

    def _device_kind(self, requirements, hardware):
        device = str(requirements.get("device", "auto")).lower()
        return "gpu" if device.startswith("cuda") or (device == "auto" and hardware.get("torch_cuda_available")) else "cpu"

    def _can_reserve(self, job, hardware):
        req = job["requirements"]
        reservations = self.database.compute_reservations()
        kind = self._device_kind(req, hardware)
        same_kind = [row for row in reservations if row["resources"].get("device_kind") == kind]
        limit = self.profile.max_concurrent_gpu_jobs if kind == "gpu" else self.profile.max_concurrent_cpu_jobs
        if len(same_kind) >= limit: return False, f"Waiting for an available {kind.upper()} slot."
        disk_free = (hardware.get("disk") or {}).get("free_gb")
        min_disk = max(self.profile.minimum_free_disk_gb, float(req.get("minimum_free_disk_gb", 0)))
        if disk_free is not None and disk_free < min_disk: return False, f"Requires {min_disk:g} GB free disk space."
        if kind == "cpu":
            memory = hardware.get("memory") or {}; available = memory.get("available_mb"); total = memory.get("total_mb")
            reserved = sum(int(row["resources"].get("ram_mb", 0)) for row in reservations if row["resources"].get("device_kind") == "cpu")
            requested = max(0, int(req.get("ram_mb", 0)))
            safety = int((total or 0) * self.profile.ram_safety_percent / 100)
            if available is not None and available - reserved - requested < safety: return False, "Waiting for safe available RAM."
            if available is None and reserved: return False, "Available RAM is unknown; only one CPU job is allowed at a time."
            return True, ""
        gpus = hardware.get("gpus") or []
        device_text = str(req.get("device", "auto")).lower()
        if not gpus or not hardware.get("torch_cuda_available"): return False, "CUDA is unavailable on this computer."
        selected = 0
        if device_text.startswith("cuda:"):
            try: selected = int(device_text.split(":", 1)[1])
            except ValueError: return False, "GPU device selection is invalid."
        if selected >= len(gpus): return False, "Selected GPU was not detected."
        gpu = gpus[selected]
        active_on_gpu = [row for row in reservations if row["resources"].get("device_kind") == "gpu" and row["resources"].get("gpu_index") == selected]
        if len(active_on_gpu) >= self.profile.max_concurrent_gpu_jobs: return False, "Waiting for the selected GPU to become available."
        available_mb = gpu.get("memory_free_mb")
        if available_mb is None:
            if active_on_gpu: return False, "GPU memory is unknown; only one job per GPU is allowed."
            return True, ""
        reserved_mb = sum(int(row["resources"].get("gpu_memory_mb", 0)) for row in active_on_gpu)
        requested_mb = max(0, int(req.get("gpu_memory_mb", 0)))
        if available_mb - reserved_mb - requested_mb < self.profile.gpu_memory_safety_mb:
            return False, "Waiting for sufficient free GPU memory including the safety margin."
        return True, ""

    def pump(self):
        with self._lock:
            hardware = self.hardware_provider()
            queued = [job for job in self.database.list_compute_jobs() if job["state"] == "queued" and job["origin"] == "local"]
            for job in queued:
                launcher = self._launchers.get(job["id"])
                if launcher is None: continue
                allowed, reason = self._can_reserve(job, hardware)
                if not allowed:
                    self.scheduling_message.emit(f"{job['id'][:8]}: {reason}"); continue
                resources = dict(job["requirements"])
                resources["device_kind"] = self._device_kind(job["requirements"], hardware)
                if resources["device_kind"] == "gpu":
                    raw_device = str(resources.get("device", "auto")); resources["gpu_index"] = int(raw_device.split(":", 1)[1]) if raw_device.startswith("cuda:") else 0
                else:
                    resources["cpu_threads"] = int(resources.get("cpu_threads", self.profile.cpu_threads_per_job))
                if not self.database.reserve_compute_resources(job["id"], resources): continue
                self.database.set_compute_job_state(job["id"], "preparing")
                try:
                    worker = launcher(self.database.get_compute_job(job["id"]))
                    if worker is None: raise RuntimeError("Local job launcher returned no worker.")
                    self._workers[job["id"]] = worker; self._states[job["id"]] = "preparing"
                    self._connect_worker(job["id"], worker)
                    worker.start()
                    self.job_changed.emit(self.database.get_compute_job(job["id"]))
                except Exception as exc:
                    self.database.set_compute_job_state(job["id"], "failed", error=str(exc))
                    self.database.release_compute_resources(job["id"])
                    self._workers.pop(job["id"], None); self._states.pop(job["id"], None)
                    self.job_changed.emit(self.database.get_compute_job(job["id"]))

    def _connect_worker(self, job_id, worker):
        lifecycle = getattr(worker, "lifecycle", None)
        if lifecycle is not None: lifecycle.connect(lambda state, jid=job_id: self._set_job_state(jid, str(state)))
        success = getattr(worker, "finished_ok", None)
        failure = getattr(worker, "failed", None)
        if success is not None: success.connect(lambda _path, jid=job_id, handle=worker: self._terminal_state(jid, handle, "completed"))
        if failure is not None: failure.connect(lambda error, jid=job_id: self._set_job_state(jid, "failed", str(error)))
        worker.finished.connect(lambda jid=job_id, handle=worker: self._worker_finished(jid, handle))

    def _set_job_state(self, job_id, state, error=""):
        if state not in {"preparing", "running", "validating", "failed", "cancelled", "completed"}: return
        with self._lock:
            self._states[job_id] = state
            record = self.database.set_compute_job_state(job_id, state, error=error)
            self.job_changed.emit(record)

    def _terminal_state(self, job_id, worker, normal_state):
        state = "cancelled" if getattr(worker, "_stop_requested", threading.Event()).is_set() else normal_state
        self._set_job_state(job_id, state)

    def _worker_finished(self, job_id, worker):
        with self._lock:
            state = self._states.get(job_id, "failed")
            if state not in {"completed", "failed", "cancelled"}:
                state = "cancelled" if getattr(worker, "_stop_requested", threading.Event()).is_set() else "failed"
                self.database.set_compute_job_state(job_id, state, error="Worker exited without a terminal result." if state == "failed" else "")
            self.database.release_compute_resources(job_id)
            self._workers.pop(job_id, None); self._states.pop(job_id, None)
            self.job_changed.emit(self.database.get_compute_job(job_id))
        self.pump()

    def cancel(self, job_id: str):
        job = self.database.get_compute_job(job_id)
        if not job: raise KeyError(job_id)
        if job["state"] == "queued":
            self.database.set_compute_job_state(job_id, "cancelled")
            self._launchers.pop(job_id, None); self.job_changed.emit(self.database.get_compute_job(job_id)); return
        worker = self._workers.get(job_id)
        if worker is None: return
        worker.request_stop()

    def worker_for_job(self, job_id): return self._workers.get(job_id)


class ComputeEvents(QObject):
    worker_changed = Signal(str)
    error = Signal(str)
    connected = Signal(str)
    remote_message = Signal(object)


class ComputeControllerServer(threading.Thread):
    """TLS-protected newline JSON endpoint for monitoring and validated remote messages."""
    def __init__(self, database: ExperimentDatabase, host: str, port: int, cert_file: str, key_file: str,
                 heartbeat_event: Callable = None, error_event: Callable = None,
                 remote_handler: Callable = None, command_provider: Callable = None):
        super().__init__(name="dataset-studio-controller", daemon=True)
        self.database, self.host, self.port = database, host, int(port)
        self.cert_file, self.key_file = cert_file, key_file
        self.heartbeat_event, self.error_event = heartbeat_event, error_event
        self.remote_handler = remote_handler
        self.command_provider = command_provider
        self._stop_event = threading.Event(); self._listener = None; self._client_sockets = set(); self._lock = threading.Lock()

    @staticmethod
    def _read(stream):
        raw = stream.readline(MAX_REQUEST_BYTES + 1)
        if not raw or len(raw) > MAX_REQUEST_BYTES or not raw.endswith(b"\n"):
            raise ValueError("Worker request is empty or exceeds the message size limit.")
        value = json.loads(raw.decode("utf-8"))
        if not isinstance(value, dict): raise ValueError("Worker request must be a JSON object.")
        return value

    @staticmethod
    def _send(stream, value):
        raw = (json.dumps(value, ensure_ascii=False) + "\n").encode("utf-8")
        if len(raw) > MAX_MESSAGE_BYTES: raise ValueError("Controller response exceeded the message size limit.")
        stream.write(raw); stream.flush()

    def run(self):
        try:
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.minimum_version = ssl.TLSVersion.TLSv1_2
            context.load_cert_chain(self.cert_file, self.key_file)
            listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind((self.host, self.port)); listener.listen(32); listener.settimeout(.5)
            self._listener = listener
            while not self._stop_event.is_set():
                try: client, address = listener.accept()
                except socket.timeout: continue
                except OSError:
                    if self._stop_event.is_set(): break
                    raise
                with self._lock: self._client_sockets.add(client)
                threading.Thread(target=self._serve_client, args=(client, address, context), daemon=True).start()
        except Exception as exc:
            if self.error_event: self.error_event(str(exc))
        finally:
            try:
                if self._listener: self._listener.close()
            except OSError: pass

    def _serve_client(self, raw_socket, address, context):
        worker_id = connection_id = None
        try:
            with context.wrap_socket(raw_socket, server_side=True) as connection:
                with connection.makefile("rwb") as stream:
                    hello = self._read(stream)
                    kind = hello.get("type")
                    if len(json.dumps(hello.get("hardware", {}), default=str)) > 256_000 or len(json.dumps(hello.get("software", {}), default=str)) > 64_000:
                        raise ValueError("Worker hardware/software report exceeded its size limit.")
                    if kind == "enroll":
                        token = hello.get("token", "")
                        if not isinstance(token, str) or len(token) > 128: raise PermissionError("Enrollment token is invalid.")
                        worker_id = str(hello.get("worker_id", "")); name = str(hello.get("name", ""))[:80]
                        if len(worker_id) > 64 or not worker_id or not name: raise ValueError("Worker identity is invalid.")
                        credential = secrets.token_urlsafe(48)
                        self.database.enroll_worker(_worker_secret_hash(token), worker_id, name,
                            f"{address[0]}:{address[1]}", credential, hello.get("hardware", {}), hello.get("software", {}))
                        connection_id = uuid.uuid4().hex
                        self._send(stream, {"type": "enrolled", "worker_id": worker_id, "credential": credential, "approved": False})
                    elif kind == "authenticate":
                        worker_id = str(hello.get("worker_id", "")); credential = hello.get("credential", "")
                        worker = self.database.authenticate_worker(worker_id, credential if isinstance(credential, str) else "")
                        if worker is None or not worker["enabled"]: raise PermissionError("Worker authentication failed or this worker is disabled/revoked.")
                        connection_id = uuid.uuid4().hex
                        self.database.record_worker_heartbeat(worker_id, connection_id, hello.get("hardware", {}), hello.get("software", {}))
                        worker = self.database.get_worker(worker_id)
                        self._send(stream, {"type": "authenticated", "worker_id": worker_id, "approved": bool(worker["approved"])})
                    else:
                        raise PermissionError("Unknown worker connection type.")
                    if self.heartbeat_event: self.heartbeat_event(worker_id)
                    last_update = 0.0
                    while not self._stop_event.is_set():
                        try: heartbeat = self._read(stream)
                        except (ConnectionResetError, OSError, ValueError): break
                        if heartbeat.get("type") == "disconnect": break
                        if heartbeat.get("type") != "heartbeat": raise PermissionError("Only heartbeat messages are accepted after authentication.")
                        if time.monotonic() - last_update < .5: raise PermissionError("Heartbeat rate limit exceeded.")
                        last_update = time.monotonic()
                        if not self.database.record_worker_heartbeat(worker_id, connection_id,
                                heartbeat.get("hardware", {}), heartbeat.get("software", {})):
                            raise PermissionError("Worker is no longer enabled.")
                        current = self.database.get_worker(worker_id)
                        reply = {"type": "heartbeat_ack", "approved": bool(current["approved"]), "time": datetime.now(timezone.utc).isoformat()}
                        remote_messages = heartbeat.get("remote", [])
                        if remote_messages:
                            if (not isinstance(remote_messages, list) or len(remote_messages) > 32
                                    or any(not isinstance(item, dict) for item in remote_messages)
                                    or len(json.dumps(remote_messages, default=str)) > MAX_REQUEST_BYTES - 4096):
                                raise ValueError("Remote worker updates must be a bounded list of JSON objects.")
                            if not current["approved"] or not current["enabled"]:
                                raise PermissionError("Worker approval is required for remote job messages.")
                            if self.remote_handler:
                                if any(item.get("type") in {"poll", "controller_poll"} for item in remote_messages):
                                    raise PermissionError("Workers cannot issue controller polling operations.")
                                reply["remote_responses"] = [response for response in
                                    (self.remote_handler(worker_id, item) for item in remote_messages) if response is not None]
                        if self.command_provider and current["approved"] and current["enabled"]:
                            command = self.command_provider(worker_id)
                            if command is not None:
                                if not isinstance(command, dict) or len(json.dumps(command, default=str)) > MAX_MESSAGE_BYTES // 2:
                                    raise ValueError("Controller remote command is invalid or too large.")
                                reply["command"] = command
                        self._send(stream, reply)
                        if self.heartbeat_event: self.heartbeat_event(worker_id)
        except Exception as exc:
            if self.error_event and not self._stop_event.is_set(): self.error_event(str(exc))
        finally:
            if worker_id and connection_id: self.database.disconnect_worker(worker_id, connection_id)
            with self._lock: self._client_sockets.discard(raw_socket)
            if worker_id and self.heartbeat_event: self.heartbeat_event(worker_id)

    def stop(self):
        self._stop_event.set()
        if self._listener:
            try: self._listener.close()
            except OSError: pass
        with self._lock:
            for connection in tuple(self._client_sockets):
                try: connection.shutdown(socket.SHUT_RDWR); connection.close()
                except OSError: pass
        if self.is_alive(): self.join(timeout=1)


class ComputeWorkerClient(threading.Thread):
    """Outbound TLS worker channel with optional application-owned remote messages."""
    def __init__(self, *, worker_id, name, host, port, ca_file, database, credential_store,
                 enrollment_token=None, heartbeat_seconds=3, events: ComputeEvents | None = None):
        super().__init__(name="dataset-studio-worker-client", daemon=True)
        self.worker_id, self.name = worker_id, name
        self.host, self.port, self.ca_file = host, int(port), ca_file or None
        self.database, self.credential_store, self.enrollment_token = database, credential_store, enrollment_token
        self.heartbeat_seconds, self.events = max(2, int(heartbeat_seconds)), events
        self._stop_event = threading.Event(); self._wake_event = threading.Event(); self._socket = None
        self._remote_lock = threading.Lock(); self._remote_outbox = []

    def send_remote_message(self, message: dict):
        """Queue a small structured status event for the next authenticated heartbeat."""
        if not isinstance(message, dict) or len(json.dumps(message, default=str)) > 768 * 1024:
            raise ValueError("Remote status must be a bounded JSON object.")
        with self._remote_lock:
            if message.get("type") == "progress" and not message.get("request_id"):
                for index in range(len(self._remote_outbox) - 1, -1, -1):
                    previous = self._remote_outbox[index]
                    if previous.get("type") == "progress" and previous.get("job_id") == message.get("job_id"):
                        self._remote_outbox[index] = message
                        return
            if len(self._remote_outbox) >= 32:
                removable = next((index for index, item in enumerate(self._remote_outbox)
                                  if item.get("type") == "progress" and not item.get("request_id")), None)
                if removable is None:
                    raise BufferError("Remote message queue is full; controller communication is unavailable.")
                self._remote_outbox.pop(removable)
            self._remote_outbox.append(message)
        self._wake_event.set()

    def _take_remote_messages(self):
        with self._remote_lock:
            pending, self._remote_outbox = self._remote_outbox, []
        return pending

    def _send(self, stream, payload):
        raw = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
        if len(raw) > MAX_REQUEST_BYTES: raise ValueError("Worker request exceeds the size limit.")
        stream.write(raw); stream.flush()

    def _receive(self, stream):
        raw = stream.readline(MAX_MESSAGE_BYTES + 1)
        if not raw or len(raw) > MAX_MESSAGE_BYTES or not raw.endswith(b"\n"): raise ConnectionError("Controller response was invalid.")
        message = json.loads(raw.decode("utf-8"))
        if not isinstance(message, dict): raise ValueError("Controller response must be an object.")
        if message.get("type") == "error": raise PermissionError(str(message.get("message", "Controller rejected request.")))
        return message

    def run(self):
        credential = None
        try: credential = self.credential_store.load(self.worker_id)
        except Exception as exc:
            if self.events: self.events.error.emit(f"Secure credential storage is unavailable: {exc}")
            return
        if credential is None and not self.enrollment_token:
            if self.events: self.events.error.emit("A one-time enrollment token is required for initial connection.")
            return
        while not self._stop_event.is_set():
            try:
                context = ssl.create_default_context(cafile=self.ca_file)
                context.minimum_version = ssl.TLSVersion.TLSv1_2
                sock = socket.create_connection((self.host, self.port), timeout=8)
                sock.settimeout(self.heartbeat_seconds + 10)
                tls = context.wrap_socket(sock, server_hostname=self.host)
                self._socket = tls
                with tls, tls.makefile("rwb") as stream:
                    snapshot = detect_hardware(); snapshot["gpus"] = [gpu.__dict__ for gpu in snapshot["gpus"]]
                    versions = snapshot.get("training_framework_versions", {})
                    if credential is None:
                        self._send(stream, {"type": "enroll", "worker_id": self.worker_id, "name": self.name,
                            "token": self.enrollment_token, "hardware": snapshot, "software": versions})
                        response = self._receive(stream)
                        if response.get("type") != "enrolled": raise PermissionError("Controller did not complete enrollment.")
                        credential = str(response.get("credential", ""))
                        if not credential: raise PermissionError("Controller returned an empty credential.")
                        self.credential_store.save(self.worker_id, credential); self.enrollment_token = None
                    else:
                        self._send(stream, {"type": "authenticate", "worker_id": self.worker_id, "credential": credential,
                                            "hardware": snapshot, "software": versions})
                        response = self._receive(stream)
                        if response.get("type") != "authenticated": raise PermissionError("Controller authentication failed.")
                    if self.events: self.events.connected.emit("approved" if response.get("approved") else "pending approval")
                    while not self._stop_event.is_set():
                        snapshot = detect_hardware(); snapshot["gpus"] = [gpu.__dict__ for gpu in snapshot["gpus"]]
                        self._send(stream, {"type": "heartbeat", "hardware": snapshot,
                                            "software": snapshot.get("training_framework_versions", {}),
                                            "remote": self._take_remote_messages()})
                        ack = self._receive(stream)
                        if ack.get("type") != "heartbeat_ack": raise ConnectionError("Controller did not acknowledge worker heartbeat.")
                        for message in ack.get("remote_responses", []):
                            if self.events: self.events.remote_message.emit(message)
                        command = ack.get("command")
                        if command is not None:
                            if not ack.get("approved") or not isinstance(command, dict):
                                raise PermissionError("Controller sent remote work without worker approval.")
                            if self.events: self.events.remote_message.emit({"type": "command", "payload": command})
                        if self.events: self.events.connected.emit("approved" if ack.get("approved") else "pending approval")
                        self._wake_event.clear()
                        self._wake_event.wait(self.heartbeat_seconds)
                    try: self._send(stream, {"type": "disconnect"})
                    except OSError: pass
            except Exception as exc:
                if self.events and not self._stop_event.is_set(): self.events.error.emit(str(exc))
                self._stop_event.wait(5)
            finally:
                if self._socket:
                    try: self._socket.close()
                    except OSError: pass
                    self._socket = None

    def stop(self):
        self._stop_event.set(); self._wake_event.set()
        if self._socket:
            try: self._socket.shutdown(socket.SHUT_RDWR); self._socket.close()
            except OSError: pass
        if self.is_alive(): self.join(timeout=1)


class UnifiedComputeManager(QObject):
    mode_changed = Signal(str)
    def __init__(self, database: ExperimentDatabase, parent=None, credential_store=None):
        super().__init__(parent); self.database = database
        stored = database.get_compute_profile()
        profile_values = stored.get("resources", {})
        try: self.resource_profile = ResourceProfile.from_dict(profile_values).validate()
        except (TypeError, ValueError): self.resource_profile = ResourceProfile()
        self.mode = stored.get("mode", "Controller") if stored.get("mode") in MODES else "Controller"
        self.settings = stored
        self.scheduler = LocalJobScheduler(database, self.resource_profile, parent=self)
        self.scheduler.start_pump()
        self.credential_store = credential_store or DPAPICredentialStore()
        self.events = ComputeEvents()
        self.controller = None; self.worker_client = None
        self.remote_broker = None; self.remote_agent = None

    def save_settings(self, settings: dict, profile: ResourceProfile | None = None):
        next_mode = settings.get("mode", self.mode)
        if next_mode not in MODES: raise ValueError("Operating mode must be Controller, Worker, or Hybrid.")
        if next_mode != self.mode and self.scheduler.active_count():
            raise RuntimeError("Cannot change operating mode while local jobs are active; wait for them to stop safely.")
        if profile and asdict(profile) != asdict(self.scheduler.profile):
            self.scheduler.set_profile(profile)
        self.mode = next_mode; self.settings = {**self.settings, **settings, "mode": self.mode,
                                                "resources": asdict(self.scheduler.profile)}
        self.database.save_compute_profile(self.settings); self.mode_changed.emit(self.mode)
        if self.mode == "Worker" and self.controller: self.stop_controller()
        if self.mode == "Controller" and self.worker_client: self.stop_worker()
        return self.settings

    def create_enrollment_token(self, lifetime_minutes=15):
        if self.mode not in ("Controller", "Hybrid"): raise RuntimeError("Controller capabilities are not enabled in the selected mode.")
        token = secrets.token_urlsafe(32); expires = (utc_now() + timedelta(minutes=max(1, min(int(lifetime_minutes), 1440)))).isoformat()
        self.database.add_enrollment_token(_worker_secret_hash(token), expires)
        return token, expires

    def start_controller(self):
        if self.mode not in ("Controller", "Hybrid"): raise RuntimeError("Switch to Controller or Hybrid mode first.")
        if self.controller and self.controller.is_alive(): return
        host = str(self.settings.get("listen_host", "0.0.0.0")); port = int(self.settings.get("listen_port", 8765))
        cert, key = self.settings.get("tls_cert", ""), self.settings.get("tls_key", "")
        if not cert or not key: raise ValueError("Select a TLS certificate and private key before starting the controller service.")
        from distributed_training import RemoteJobBroker
        self.remote_broker = RemoteJobBroker(self.database, app_state_file("remote_dataset_transfers"))
        self.controller = ComputeControllerServer(self.database, host, port, cert, key,
            lambda worker_id: self.events.worker_changed.emit(worker_id), lambda error: self.events.error.emit(error),
            remote_handler=self.remote_broker.handle, command_provider=self.remote_broker.offer_next)
        self.controller.start()

    def submit_remote_job(self, config, dataset_dir, *, worker_id=None, priority=0, experiment_id=None,
                          ram_mb=0, minimum_free_disk_gb=0):
        if self.mode not in ("Controller", "Hybrid"):
            raise RuntimeError("Remote job submission is available in Controller or Hybrid mode.")
        from distributed_training import RemoteJobBroker
        if self.remote_broker is None:
            self.remote_broker = RemoteJobBroker(self.database, app_state_file("remote_dataset_transfers"))
        return self.remote_broker.submit(config, dataset_dir, worker_id=worker_id, priority=priority,
            experiment_id=experiment_id, ram_mb=ram_mb, minimum_free_disk_gb=minimum_free_disk_gb)

    def cancel_remote_job(self, job_id: str):
        result = self.database.request_remote_job_cancel(job_id)
        if result == "cancelled":
            job = self.database.get_compute_job(job_id)
            if job and job.get("experiment_id"):
                experiment = self.database.get_experiment(job["experiment_id"])
                metadata = dict(experiment.get("metadata", {})) if experiment else {}
                metadata.update(experiment_id=job["experiment_id"], status="cancelled", end_time=job.get("ended_at"))
                self.database.upsert_experiment(metadata)
        return result

    def stop_controller(self):
        if self.controller: self.controller.stop(); self.controller = None

    def start_worker(self, *, worker_id, name, host, port, ca_file="", enrollment_token=None):
        if self.mode not in ("Worker", "Hybrid"): raise RuntimeError("Switch to Worker or Hybrid mode first.")
        self.stop_worker()
        self.worker_client = ComputeWorkerClient(worker_id=worker_id, name=name, host=host, port=port, ca_file=ca_file,
            database=self.database, credential_store=self.credential_store, enrollment_token=enrollment_token, events=self.events)
        from distributed_training import RemoteWorkerAgent
        self.remote_agent = RemoteWorkerAgent(self.worker_client, self.events,
            cache_root=app_state_file("worker_dataset_cache"), output_root=app_state_file("remote_training_runs"))
        self.worker_client.start()

    def stop_worker(self):
        if self.worker_client: self.worker_client.stop(); self.worker_client = None
        if self.remote_agent:
            try: self.events.remote_message.disconnect(self.remote_agent._on_message)
            except (RuntimeError, TypeError): pass
            self.remote_agent = None

    def close(self):
        self.scheduler.stop_pump(); self.stop_worker(); self.stop_controller()
        for job in self.database.list_compute_jobs():
            if job["state"] in ("preparing", "running", "validating"):
                self.scheduler.cancel(job["id"])
