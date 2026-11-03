import hashlib
import json
import os
import socket
import tempfile
import threading
import time
import unittest
import queue
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import QApplication, QStackedWidget, QVBoxLayout, QWidget, QPushButton

import hardware
from compute_manager import (ComputeControllerServer, ComputeEvents, ComputeWorkerClient,
    DPAPICredentialStore, LocalJobScheduler, ResourceProfile, UnifiedComputeManager)
from compute_studio import ComputeCenterPanel, _EnrollmentTokenDialog
from experiment_database import ExperimentDatabase


class FakeCredentialStore:
    def __init__(self): self.values = {}
    def save(self, key, value): self.values[key] = value
    def load(self, key): return self.values.get(key)
    def delete(self, key): self.values.pop(key, None)


class FakePipeSocket:
    def __init__(self, incoming, outgoing): self.incoming, self.outgoing = incoming, outgoing; self.closed = False
    def settimeout(self, _timeout): pass
    def makefile(self, _mode): return FakePipeStream(self)
    def shutdown(self, _how): self.close()
    def close(self):
        if not self.closed:
            self.closed = True; self.outgoing.put(None)
    def __enter__(self): return self
    def __exit__(self, *_exc): self.close()


class FakePipeStream:
    def __init__(self, sock): self.sock = sock; self.closed = False
    def write(self, raw): self.sock.outgoing.put(bytes(raw)); return len(raw)
    def flush(self): pass
    def readline(self, _limit=-1):
        try: value = self.sock.incoming.get(timeout=3)
        except queue.Empty: return b""
        return b"" if value is None else value
    def close(self): self.closed = True
    def __enter__(self): return self
    def __exit__(self, *_exc): self.close()


class FakeHardware:
    def __init__(self, cuda=False, gpu_memory=None):
        self.cuda = cuda; self.gpu_memory = gpu_memory
    def __call__(self):
        return {"torch_cuda_available": self.cuda,
            "cpu": {"logical_cores": 8}, "memory": {"total_mb": 32000, "available_mb": 24000},
            "disk": {"free_gb": 200},
            "gpus": ([{"name": "Mock GPU", "memory_mb": 16000, "memory_free_mb": self.gpu_memory,
                       "memory_used_mb": None, "torch_cuda": True}] if self.cuda else [])}


class BlockingWorker(QThread):
    lifecycle = Signal(str)
    finished_ok = Signal(str)
    failed = Signal(str)
    def __init__(self, started, release, counters):
        super().__init__(); self.started_event = started; self.release = release; self.counters = counters
        self._stop_requested = threading.Event()
    def request_stop(self): self._stop_requested.set(); self.release.set()
    def run(self):
        with self.counters["lock"]:
            self.counters["active"] += 1; self.counters["max"] = max(self.counters["max"], self.counters["active"])
        self.started_event.set(); self.lifecycle.emit("running"); self.release.wait(5)
        self.finished_ok.emit("done")
        with self.counters["lock"]: self.counters["active"] -= 1


class ComputeManagerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent.parent)
        self.root = Path(self.temp.name); self.db = ExperimentDatabase(self.root / "compute.sqlite3")

    def tearDown(self): self.temp.cleanup()

    def pump_events_until(self, predicate, timeout=4):
        end = time.monotonic() + timeout
        while time.monotonic() < end and not predicate():
            self.app.processEvents(); time.sleep(.02)
        self.app.processEvents()
        return predicate()

    def test_cpu_only_hardware_detection_reports_resources_without_cuda(self):
        with patch("hardware._run_nvidia_smi", return_value=[]):
            data = hardware.detect_hardware()
        self.assertIn("cpu", data); self.assertIn("memory", data); self.assertIn("disk", data)
        self.assertEqual(data["gpus"], [])
        self.assertFalse(data["torch_cuda_available"])
        self.assertIn("YOLO", data["supported_training_architectures"])

    def test_mock_gpu_discovery_reports_vram_and_cuda_compatibility(self):
        gpu = hardware.GPUInfo(0, "Mock RTX", 12288, driver="550", torch_cuda=True,
                               memory_used_mb=2048, memory_free_mb=10240, utilization_percent=22.0)
        with patch("hardware._run_nvidia_smi", return_value=[gpu]), patch("torch.cuda.is_available", return_value=True):
            data = hardware.detect_hardware()
        self.assertEqual(data["gpus"][0].memory_free_mb, 10240)
        self.assertTrue(data["torch_cuda_compatible"])

    def test_resource_profile_validation_and_persistence(self):
        profile = ResourceProfile(2, 1, 6, 25, 15, 3072, 4).validate()
        manager = UnifiedComputeManager(self.db, credential_store=FakeCredentialStore())
        manager.save_settings({"mode": "Hybrid"}, profile)
        self.assertEqual(self.db.get_compute_profile()["mode"], "Hybrid")
        self.assertEqual(ResourceProfile.from_dict(self.db.get_compute_profile()["resources"]).cpu_threads_per_job, 6)
        manager.close()
        with self.assertRaises(ValueError): ResourceProfile(max_concurrent_cpu_jobs=0).validate()

    def test_mode_switch_is_persisted_and_blocked_while_job_is_active(self):
        manager = UnifiedComputeManager(self.db, credential_store=FakeCredentialStore())
        manager.save_settings({"mode": "Worker"})
        self.assertEqual(manager.mode, "Worker")
        job = self.db.create_compute_job({"id": "active", "architecture": "YOLO", "requirements": {}, "priority": 0})
        self.db.reserve_compute_resources(job["id"], {"device_kind": "cpu"})
        with self.assertRaisesRegex(RuntimeError, "active"):
            manager.save_settings({"mode": "Hybrid"})
        manager.close()

    def test_worker_enrollment_authentication_approval_heartbeat_and_revocation(self):
        token = "single-use-enrollment-token"; credential = "worker-secret-high-entropy"
        expiry = (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()
        token_hash = hashlib.sha256(token.encode()).digest(); self.db.add_enrollment_token(token_hash, expiry)
        worker = self.db.enroll_worker(token_hash, "worker-1", "Node One", "127.0.0.1:55", credential,
                                       {"cpu": {"logical_cores": 8}}, {"ultralytics": "8.4"})
        self.assertEqual(worker["status"], "pending"); self.assertFalse(worker["approved"])
        self.assertIsNotNone(self.db.authenticate_worker("worker-1", credential))
        self.assertIsNone(self.db.authenticate_worker("worker-1", "wrong"))
        with self.assertRaises(PermissionError): self.db.enroll_worker(token_hash, "worker-2", "Node Two", "", "another", {}, {})
        self.db.set_worker_approved("worker-1", True)
        self.assertTrue(self.db.record_worker_heartbeat("worker-1", "session-a", {"memory": {"available_mb": 900}}, {"torch": "cpu"}))
        self.assertEqual(self.db.get_worker("worker-1")["status"], "online")
        self.assertEqual(len(self.db.worker_heartbeat_history("worker-1")), 1)
        self.db.disconnect_worker("worker-1", "session-a")
        self.assertEqual(self.db.get_worker("worker-1")["status"], "offline")
        self.db.revoke_worker("worker-1")
        self.assertIsNone(self.db.authenticate_worker("worker-1", credential))
        with self.db._connect() as conn:
            raw = conn.execute("SELECT credential_hash FROM workers WHERE id='worker-1'").fetchone()[0]
        self.assertNotEqual(raw, credential.encode())

    def test_worker_disable_and_revoke_remove_heartbeat_access(self):
        token_hash = hashlib.sha256(b"token").digest()
        self.db.add_enrollment_token(token_hash, (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat())
        self.db.enroll_worker(token_hash, "node", "Node", "", "secret", {}, {})
        self.db.set_worker_enabled("node", False)
        self.assertFalse(self.db.record_worker_heartbeat("node", "session", {}, {}))
        self.db.set_worker_enabled("node", True); self.db.set_worker_approved("node", True)
        self.assertTrue(self.db.record_worker_heartbeat("node", "session", {}, {}))
        self.db.revoke_worker("node")
        self.assertFalse(self.db.record_worker_heartbeat("node", "session", {}, {}))

    def test_token_revocation_consumption_status_and_secret_redaction(self):
        manager = UnifiedComputeManager(self.db, credential_store=FakeCredentialStore())
        token, _expires = manager.create_enrollment_token()
        token_id = hashlib.sha256(token.encode()).hexdigest()[:12]
        self.assertEqual(next(row for row in self.db.list_enrollment_tokens() if row["token_id"] == token_id)["status"], "unused")
        self.assertTrue(self.db.revoke_enrollment_token(token))
        self.assertEqual(next(row for row in self.db.list_enrollment_tokens() if row["token_id"] == token_id)["status"], "revoked")
        with self.assertRaisesRegex(PermissionError, "invalid, expired, or already used"):
            self.db.enroll_worker(hashlib.sha256(token.encode()).digest(), "revoked-node", "Node", "", "secret", {}, {})
        events = self.db.list_security_events()
        self.assertTrue(any(event["category"] == "enrollment_token_created" for event in events))
        self.assertNotIn(token, json.dumps(events))
        manager.close()

    def test_expired_enrollment_token_is_rejected(self):
        token = "expired-token-value"
        token_hash = hashlib.sha256(token.encode()).digest()
        self.db.add_enrollment_token(token_hash, (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat())
        with self.assertRaisesRegex(PermissionError, "invalid, expired, or already used"):
            self.db.enroll_worker(token_hash, "expired-node", "Expired", "", "secret", {}, {})

    def test_unapproved_worker_cannot_send_remote_job_messages(self):
        token_hash = hashlib.sha256(b"pending-worker").digest()
        self.db.add_enrollment_token(token_hash, (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat())
        self.db.enroll_worker(token_hash, "pending", "Pending", "", "secret", {}, {})
        worker_to_controller = queue.Queue(); controller_to_worker = queue.Queue()
        raw = FakePipeSocket(worker_to_controller, controller_to_worker)
        worker_to_controller.put((json.dumps({"type": "authenticate", "worker_id": "pending", "credential": "secret", "hardware": {}, "software": {}}) + "\n").encode())
        worker_to_controller.put((json.dumps({"type": "heartbeat", "hardware": {}, "software": {},
            "remote": [{"type": "progress", "job_id": "forged"}]}) + "\n").encode())
        handled = []; errors = []
        class ControllerTLS:
            def wrap_socket(self, sock, **_kwargs): return sock
        server = ComputeControllerServer(self.db, "127.0.0.1", 0, "unused", "unused",
            error_event=errors.append, remote_handler=lambda worker, message: handled.append(message))
        thread = threading.Thread(target=server._serve_client, args=(raw, ("127.0.0.1", 2), ControllerTLS()), daemon=True)
        thread.start(); thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertEqual(handled, [])
        self.assertTrue(any("approval is required" in error for error in errors))

    def test_worker_session_is_unique_and_stale_connection_can_reconnect(self):
        token_hash = hashlib.sha256(b"single-session").digest()
        self.db.add_enrollment_token(token_hash, (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat())
        self.db.enroll_worker(token_hash, "single", "Single", "", "secret", {}, {})
        self.db.set_worker_approved("single", True)
        self.assertTrue(self.db.record_worker_heartbeat("single", "session-a", {"memory": {"available_mb": 1000}}, {}))
        self.assertFalse(self.db.record_worker_heartbeat("single", "session-b", {"memory": {"available_mb": 900}}, {}))
        old = (datetime.now(timezone.utc) - timedelta(seconds=90)).isoformat()
        with self.db._connect() as conn: conn.execute("UPDATE workers SET last_seen=? WHERE id='single'", (old,))
        self.assertEqual(self.db.expire_stale_workers(30), ["single"])
        self.assertTrue(self.db.record_worker_heartbeat("single", "session-b", {"memory": {"available_mb": 900}}, {}))

    def test_token_dialog_copy_and_expiry_clears_token(self):
        manager = UnifiedComputeManager(self.db, credential_store=FakeCredentialStore())
        token, expires = manager.create_enrollment_token(lifetime_minutes=1)
        dialog = _EnrollmentTokenDialog(manager, token, expires)
        dialog.copy_token()
        self.assertEqual(QApplication.clipboard().text(), token)
        dialog.expires_at = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
        dialog.update_status()
        self.assertEqual(dialog.token, "")
        self.assertEqual(dialog.token_edit.text(), "")
        dialog.timer.stop(); dialog.close(); manager.close()

    def test_compute_center_layout_remains_visible_at_common_window_sizes(self):
        manager = UnifiedComputeManager(self.db, credential_store=FakeCredentialStore())
        panel = ComputeCenterPanel(manager)
        for width, height in ((960, 640), (1280, 800), (1650, 950)):
            panel.resize(width, height); panel.show(); panel.tabs.setCurrentIndex(2)
            self.app.processEvents()
            self.assertGreater(panel.tabs.width(), 0)
            self.assertGreater(panel.tabs.height(), 0)
            self.assertGreater(panel.jobs_table.width(), 0)
            self.assertGreater(panel.jobs_table.height(), 0)
        panel.stop(); manager.close(); panel.close(); self.app.processEvents()

    def test_scheduler_runs_concurrent_cpu_jobs_and_releases_resources(self):
        started = [threading.Event(), threading.Event()]; release = threading.Event()
        counts = {"active": 0, "max": 0, "lock": threading.Lock()}; workers = []
        scheduler = LocalJobScheduler(self.db, ResourceProfile(max_concurrent_cpu_jobs=2), FakeHardware())
        def launch(_job):
            worker = BlockingWorker(started[len(workers)], release, counts); workers.append(worker); return worker
        first = scheduler.submit(architecture="YOLO", requirements={"device": "cpu"}, launcher=launch)
        second = scheduler.submit(architecture="RT-DETR", requirements={"device": "cpu"}, launcher=launch)
        self.assertTrue(started[0].wait(2)); self.assertTrue(started[1].wait(2))
        self.assertEqual(counts["max"], 2)
        release.set()
        for worker in workers: worker.wait(3000)
        self.assertTrue(self.pump_events_until(lambda: not self.db.compute_reservations()))
        self.assertEqual(self.db.get_compute_job(first)["state"], "completed")
        self.assertEqual(self.db.get_compute_job(second)["state"], "completed")

    def test_scheduler_runs_two_independent_cpu_jobs_sequentially(self):
        started = [threading.Event(), threading.Event()]
        release = [threading.Event(), threading.Event()]
        counts = {"active": 0, "max": 0, "lock": threading.Lock()}; workers = []
        scheduler = LocalJobScheduler(self.db, ResourceProfile(max_concurrent_cpu_jobs=1), FakeHardware())
        def launch(_job):
            index = len(workers)
            worker = BlockingWorker(started[index], release[index], counts); workers.append(worker); return worker
        first = scheduler.submit(architecture="YOLO", requirements={"device": "cpu"}, launcher=launch)
        second = scheduler.submit(architecture="YOLO", requirements={"device": "cpu"}, launcher=launch)
        self.assertTrue(started[0].wait(2)); self.assertFalse(started[1].is_set())
        release[0].set(); workers[0].wait(3000)
        self.assertTrue(self.pump_events_until(lambda: started[1].is_set()))
        release[1].set(); workers[1].wait(3000)
        self.assertTrue(self.pump_events_until(lambda: not self.db.compute_reservations()))
        self.assertEqual(counts["max"], 1)
        self.assertEqual(self.db.get_compute_job(first)["state"], "completed")
        self.assertEqual(self.db.get_compute_job(second)["state"], "completed")

    def test_duplicate_submission_and_unknown_gpu_memory_are_conservative(self):
        started = threading.Event(); release = threading.Event(); counts = {"active": 0, "max": 0, "lock": threading.Lock()}; workers = []
        scheduler = LocalJobScheduler(self.db, ResourceProfile(max_concurrent_gpu_jobs=2), FakeHardware(cuda=True, gpu_memory=None))
        def launch(_job):
            worker = BlockingWorker(started, release, counts); workers.append(worker); return worker
        first = scheduler.submit(architecture="YOLO", requirements={"device": "cuda:0"}, launcher=launch, dedupe_key="same-output")
        self.assertTrue(started.wait(2))
        with self.assertRaisesRegex(ValueError, "active job"):
            scheduler.submit(architecture="YOLO", requirements={"device": "cuda:0"}, launcher=launch, dedupe_key="same-output")
        second = scheduler.submit(architecture="RT-DETR", requirements={"device": "cuda:0"}, launcher=launch)
        self.assertEqual(self.db.get_compute_job(second)["state"], "queued")
        release.set(); workers[0].wait(3000)
        self.assertTrue(self.pump_events_until(lambda: len(workers) == 2))
        workers[-1].request_stop(); workers[-1].wait(3000)
        self.assertTrue(self.pump_events_until(lambda: not self.db.compute_reservations()))
        self.assertIn(self.db.get_compute_job(first)["state"], ("completed", "cancelled"))

    def test_restart_reconciles_interrupted_jobs_as_failed_not_completed(self):
        job = self.db.create_compute_job({"id": "lost", "architecture": "YOLO", "requirements": {}, "priority": 0})
        self.db.reserve_compute_resources(job["id"], {"device_kind": "cpu"})
        self.db.set_compute_job_state(job["id"], "running")
        restarted = ExperimentDatabase(self.root / "compute.sqlite3")
        saved = restarted.get_compute_job("lost")
        self.assertEqual(saved["state"], "failed"); self.assertIn("Interrupted", saved["error"])
        self.assertEqual(restarted.compute_reservations(), [])

    def test_compute_center_ui_smoke_and_secure_storage_fails_closed_outside_windows(self):
        manager = UnifiedComputeManager(self.db, credential_store=FakeCredentialStore())
        panel = ComputeCenterPanel(manager)
        self.assertEqual(panel.mode.currentText(), "Controller")
        self.assertGreaterEqual(panel.tabs.count(), 4)
        self.assertIn("Controller listener:", panel.resource_summary.text())
        self.assertNotIn("Remote execution disabled", panel.resource_summary.text())
        self.assertTrue(any(button.text() == "Archive selected" for button in panel.findChildren(QPushButton)))
        self.assertTrue(any(button.text() == "Remove revoked worker…" for button in panel.findChildren(QPushButton)))
        panel.stop(); manager.close(); panel.close()
        if os.name != "nt":
            with self.assertRaisesRegex(RuntimeError, "DPAPI"):
                DPAPICredentialStore(self.root / "credentials").save("node", "secret")

    def test_guided_remote_training_resolves_main_window_through_stacked_widget(self):
        manager = UnifiedComputeManager(self.db, credential_store=FakeCredentialStore())

        class MainWindow(QWidget):
            def __init__(self):
                super().__init__()
                self.manager = type("DatasetOwner", (), {"info": object()})()
                self._training_config = lambda: object()
                layout = QVBoxLayout(self)
                self.stack = QStackedWidget(self)
                layout.addWidget(self.stack)

        main_window = MainWindow()
        panel = ComputeCenterPanel(manager, main_window.stack)
        main_window.stack.addWidget(panel)

        self.assertIs(panel.parentWidget(), main_window.stack)
        self.assertFalse(hasattr(panel.parentWidget(), "manager"))
        self.assertIs(panel._resolve_training_owner(), main_window)

        panel.stop(); manager.close(); panel.close(); main_window.close(); self.app.processEvents()

    def test_mocked_tls_worker_connection_performs_enrollment_and_heartbeats(self):
        controller_to_worker = queue.Queue(); worker_to_controller = queue.Queue()
        left = FakePipeSocket(worker_to_controller, controller_to_worker)
        right = FakePipeSocket(controller_to_worker, worker_to_controller)
        class IdentityTLS:
            def wrap_socket(self, sock, **_kwargs): return sock
        class ControllerTLS:
            def wrap_socket(self, sock, **_kwargs): return sock
        token = "test-enrollment-token"; token_hash = hashlib.sha256(token.encode()).digest()
        self.db.add_enrollment_token(token_hash, (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat())
        server = ComputeControllerServer(self.db, "127.0.0.1", 0, "unused", "unused")
        server_context = ControllerTLS()
        server_thread = threading.Thread(target=server._serve_client, args=(left, ("127.0.0.1", 4444), server_context), daemon=True)
        server_thread.start()
        events = ComputeEvents(); connected = threading.Event(); events.connected.connect(lambda _value: connected.set())
        credentials = FakeCredentialStore()
        client = ComputeWorkerClient(worker_id="net-node", name="Net Node", host="controller.local", port=8765,
            ca_file="unused-ca.pem", database=self.db, credential_store=credentials, enrollment_token=token,
            heartbeat_seconds=2, events=events)
        with patch("compute_manager.ssl.create_default_context", return_value=IdentityTLS()), \
             patch("compute_manager.socket.create_connection", return_value=right), \
             patch("compute_manager.detect_hardware", return_value={"gpus": [], "memory": {"available_mb": 512}, "training_framework_versions": {"torch": "cpu"}}):
            client.start()
            self.assertTrue(self.pump_events_until(connected.is_set, 3))
            self.assertTrue(self.pump_events_until(lambda: credentials.load("net-node") is not None))
            self.db.set_worker_approved("net-node", True)
            self.assertTrue(self.pump_events_until(lambda: self.db.get_worker("net-node")["status"] == "online"))
            client.stop()
        server_thread.join(2)
        self.assertTrue(self.pump_events_until(lambda: self.db.get_worker("net-node")["status"] == "offline"))
        self.assertGreaterEqual(len(self.db.worker_heartbeat_history("net-node")), 1)

    def test_worker_reconnects_after_temporary_controller_network_error(self):
        token_hash = hashlib.sha256(b"enroll-again").digest()
        self.db.add_enrollment_token(token_hash, (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat())
        self.db.enroll_worker(token_hash, "reconnect-node", "Reconnect Node", "", "secret", {}, {})
        self.db.set_worker_approved("reconnect-node", True)
        controller_to_worker = queue.Queue(); worker_to_controller = queue.Queue()
        left = FakePipeSocket(worker_to_controller, controller_to_worker)
        right = FakePipeSocket(controller_to_worker, worker_to_controller)
        class IdentityTLS:
            def wrap_socket(self, sock, **_kwargs): return sock
        class FastEvent:
            def __init__(self): self.event=threading.Event()
            def set(self): self.event.set()
            def is_set(self): return self.event.is_set()
            def wait(self, _timeout=None): return self.event.wait(.02)
        controller=ComputeControllerServer(self.db, "127.0.0.1", 0, "unused", "unused")
        server_thread=threading.Thread(target=controller._serve_client, args=(left, ("127.0.0.1", 9), IdentityTLS()), daemon=True)
        server_thread.start()
        events=ComputeEvents(); connected=threading.Event(); errors=[]
        events.connected.connect(lambda _status: connected.set()); events.error.connect(errors.append)
        credentials=FakeCredentialStore(); credentials.save("reconnect-node", "secret")
        client=ComputeWorkerClient(worker_id="reconnect-node", name="Reconnect Node", host="controller.local", port=8765,
            ca_file="unused", database=self.db, credential_store=credentials, events=events)
        client._stop_event=FastEvent()
        attempts=[]
        def connect(*_args, **_kwargs):
            attempts.append(1)
            if len(attempts) == 1: raise OSError("temporary network interruption")
            return right
        with patch("compute_manager.ssl.create_default_context", return_value=IdentityTLS()), \
             patch("compute_manager.socket.create_connection", side_effect=connect), \
             patch("compute_manager.detect_hardware", return_value={"gpus": [], "memory": {}, "training_framework_versions": {"torch": "2.15", "ultralytics": "8.4"}}):
            client.start()
            self.assertTrue(self.pump_events_until(connected.is_set, 4))
            self.assertGreaterEqual(len(attempts), 2)
            client.stop()
        server_thread.join(2)
        self.assertTrue(errors)
        self.assertGreaterEqual(len(self.db.worker_heartbeat_history("reconnect-node")), 1)

    def test_approved_worker_remote_messages_are_bounded_and_polled_over_tls(self):
        token_hash = hashlib.sha256(b"enroll").digest()
        self.db.add_enrollment_token(token_hash, (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat())
        self.db.enroll_worker(token_hash, "remote-node", "Remote Node", "", "secret", {}, {})
        self.db.set_worker_approved("remote-node", True)
        worker_to_controller = queue.Queue(); controller_to_worker = queue.Queue()
        raw = FakePipeSocket(worker_to_controller, controller_to_worker)

        def send(value): worker_to_controller.put((json.dumps(value) + "\n").encode())
        send({"type": "authenticate", "worker_id": "remote-node", "credential": "secret", "hardware": {}, "software": {}})
        send({"type": "heartbeat", "hardware": {}, "software": {}, "remote": [{"type": "progress", "job_id": "j1", "epoch": 2}]})
        send({"type": "disconnect"})
        handled = []
        def handler(worker_id, message):
            handled.append((worker_id, message))
            return {"accepted": True}
        class ControllerTLS:
            def wrap_socket(self, sock, **_kwargs): return sock
        server = ComputeControllerServer(self.db, "127.0.0.1", 0, "unused", "unused", remote_handler=handler,
            command_provider=lambda _worker_id: {"job_id": "j2"})
        thread = threading.Thread(target=server._serve_client, args=(raw, ("127.0.0.1", 1), ControllerTLS()), daemon=True)
        thread.start(); thread.join(3)
        self.assertFalse(thread.is_alive())
        replies = []
        while not controller_to_worker.empty():
            item = controller_to_worker.get_nowait()
            if item is not None: replies.append(json.loads(item.decode()))
        self.assertEqual(replies[0]["type"], "authenticated")
        self.assertEqual(replies[1]["remote_responses"], [{"accepted": True}])
        self.assertEqual(replies[1]["command"], {"job_id": "j2"})
        self.assertEqual(handled[0], ("remote-node", {"type": "progress", "job_id": "j1", "epoch": 2}))


if __name__ == "__main__": unittest.main()
