"""SQLite persistence for experiments, metrics, artifacts and model records."""
from __future__ import annotations

import json
import hashlib
import hmac
import sqlite3
import uuid
import secrets
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from config import app_state_file


SCHEMA_VERSION = 5


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


class ExperimentDatabase:
    """Small repository layer. Files stay on disk; only metadata is stored here."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path is not None else app_state_file("experiments.sqlite3")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(str(self.path), timeout=15)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=15000")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self):
        with self._connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS schema_info(version INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS experiments(
                    id TEXT PRIMARY KEY, name TEXT NOT NULL, description TEXT NOT NULL DEFAULT '',
                    architecture TEXT NOT NULL, adapter TEXT NOT NULL DEFAULT '', model_name TEXT NOT NULL DEFAULT '',
                    initial_weights TEXT NOT NULL DEFAULT '', dataset_ref TEXT NOT NULL DEFAULT '',
                    classes_json TEXT NOT NULL DEFAULT '[]', class_mapping_json TEXT NOT NULL DEFAULT '[]',
                    config_json TEXT NOT NULL DEFAULT '{}', start_time TEXT, end_time TEXT,
                    duration_seconds REAL, status TEXT NOT NULL, hardware_json TEXT NOT NULL DEFAULT '{}',
                    versions_json TEXT NOT NULL DEFAULT '{}', output_dir TEXT NOT NULL DEFAULT '',
                    best_checkpoint TEXT NOT NULL DEFAULT '', last_checkpoint TEXT NOT NULL DEFAULT '',
                    error TEXT NOT NULL DEFAULT '', metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_experiments_status ON experiments(status);
                CREATE INDEX IF NOT EXISTS idx_experiments_created ON experiments(created_at DESC);
                CREATE TABLE IF NOT EXISTS deleted_experiments(
                    experiment_id TEXT PRIMARY KEY, deleted_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS experiment_metrics(
                    experiment_id TEXT NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
                    epoch INTEGER NOT NULL, raw_json TEXT NOT NULL DEFAULT '{}',
                    normalized_json TEXT NOT NULL DEFAULT '{}', recorded_at TEXT NOT NULL,
                    PRIMARY KEY(experiment_id, epoch)
                );
                CREATE TABLE IF NOT EXISTS experiment_artifacts(
                    id TEXT PRIMARY KEY, experiment_id TEXT NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
                    kind TEXT NOT NULL, path TEXT NOT NULL, metadata_json TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_artifacts_experiment ON experiment_artifacts(experiment_id);
                CREATE TABLE IF NOT EXISTS model_registry(
                    id TEXT PRIMARY KEY, name TEXT NOT NULL, version TEXT NOT NULL, architecture TEXT NOT NULL,
                    checkpoint_path TEXT NOT NULL, source_experiment_id TEXT REFERENCES experiments(id) ON DELETE SET NULL,
                    dataset_ref TEXT NOT NULL DEFAULT '', classes_json TEXT NOT NULL DEFAULT '[]',
                    input_dimensions_json TEXT NOT NULL DEFAULT '{}', metrics_json TEXT NOT NULL DEFAULT '{}',
                    description TEXT NOT NULL DEFAULT '', notes TEXT NOT NULL DEFAULT '', preferred INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL, UNIQUE(name, version)
                );
                CREATE TABLE IF NOT EXISTS evaluations(
                    id TEXT PRIMARY KEY, experiment_id TEXT REFERENCES experiments(id) ON DELETE SET NULL,
                    model_id TEXT REFERENCES model_registry(id) ON DELETE SET NULL,
                    architecture TEXT NOT NULL, checkpoint_path TEXT NOT NULL, dataset_ref TEXT NOT NULL,
                    metrics_json TEXT NOT NULL DEFAULT '{}', per_class_json TEXT NOT NULL DEFAULT '{}',
                    confusion_matrix_path TEXT NOT NULL DEFAULT '', settings_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS benchmarks(
                    id TEXT PRIMARY KEY, experiment_id TEXT REFERENCES experiments(id) ON DELETE SET NULL,
                    model_id TEXT REFERENCES model_registry(id) ON DELETE SET NULL,
                    checkpoint_path TEXT NOT NULL, device TEXT NOT NULL, input_size INTEGER NOT NULL,
                    batch_size INTEGER NOT NULL, warmup_iterations INTEGER NOT NULL, measured_iterations INTEGER NOT NULL,
                    average_latency_ms REAL, throughput_fps REAL, settings_json TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS compute_profiles(
                    profile_id TEXT PRIMARY KEY, profile_json TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS workers(
                    id TEXT PRIMARY KEY, name TEXT NOT NULL, endpoint TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'offline', approved INTEGER NOT NULL DEFAULT 0,
                    enabled INTEGER NOT NULL DEFAULT 1, credential_salt BLOB NOT NULL, credential_hash BLOB NOT NULL,
                    hardware_json TEXT NOT NULL DEFAULT '{}', software_json TEXT NOT NULL DEFAULT '{}',
                    last_seen TEXT, created_at TEXT NOT NULL, revoked_at TEXT, connection_id TEXT,
                    removed_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_workers_status ON workers(status,approved,enabled);
                CREATE TABLE IF NOT EXISTS enrollment_tokens(
                    token_hash BLOB PRIMARY KEY, expires_at TEXT NOT NULL, consumed_at TEXT, created_at TEXT NOT NULL,
                    revoked_at TEXT
                );
                CREATE TABLE IF NOT EXISTS worker_heartbeats(
                    id INTEGER PRIMARY KEY AUTOINCREMENT, worker_id TEXT NOT NULL REFERENCES workers(id) ON DELETE CASCADE,
                    received_at TEXT NOT NULL, hardware_json TEXT NOT NULL DEFAULT '{}', software_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS idx_heartbeats_worker_time ON worker_heartbeats(worker_id,received_at DESC);
                CREATE TABLE IF NOT EXISTS compute_jobs(
                    id TEXT PRIMARY KEY, experiment_id TEXT REFERENCES experiments(id) ON DELETE SET NULL,
                    architecture TEXT NOT NULL, requirements_json TEXT NOT NULL DEFAULT '{}', preferred_device TEXT NOT NULL DEFAULT 'auto',
                    worker_id TEXT REFERENCES workers(id) ON DELETE SET NULL, priority INTEGER NOT NULL DEFAULT 0,
                    state TEXT NOT NULL, created_at TEXT NOT NULL, started_at TEXT, ended_at TEXT,
                    error TEXT NOT NULL DEFAULT '', origin TEXT NOT NULL DEFAULT 'local',
                    dedupe_key TEXT, archived INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_compute_jobs_queue ON compute_jobs(state,priority DESC,created_at);
                CREATE UNIQUE INDEX IF NOT EXISTS idx_compute_jobs_active_dedupe ON compute_jobs(dedupe_key)
                    WHERE dedupe_key IS NOT NULL AND state IN ('queued','preparing','running','validating');
                CREATE UNIQUE INDEX IF NOT EXISTS idx_compute_jobs_active_experiment ON compute_jobs(experiment_id)
                    WHERE experiment_id IS NOT NULL AND state IN ('queued','preparing','running','validating');
                CREATE TABLE IF NOT EXISTS compute_reservations(
                    job_id TEXT PRIMARY KEY REFERENCES compute_jobs(id) ON DELETE CASCADE,
                    worker_id TEXT REFERENCES workers(id) ON DELETE SET NULL, resources_json TEXT NOT NULL,
                    reserved_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS remote_job_updates(
                    job_id TEXT NOT NULL REFERENCES compute_jobs(id) ON DELETE CASCADE,
                    sequence INTEGER NOT NULL, worker_id TEXT NOT NULL REFERENCES workers(id) ON DELETE CASCADE,
                    state TEXT NOT NULL, progress_json TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL,
                    PRIMARY KEY(job_id,sequence)
                );
                CREATE TABLE IF NOT EXISTS security_events(
                    event_id TEXT PRIMARY KEY, category TEXT NOT NULL, subject_type TEXT NOT NULL DEFAULT '',
                    subject_id TEXT NOT NULL DEFAULT '', details_json TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_security_events_created ON security_events(created_at DESC);
            """)
            token_columns = {row[1] for row in db.execute("PRAGMA table_info(enrollment_tokens)")}
            if "revoked_at" not in token_columns:
                db.execute("ALTER TABLE enrollment_tokens ADD COLUMN revoked_at TEXT")
            job_columns = {row[1] for row in db.execute("PRAGMA table_info(compute_jobs)")}
            if "archived" not in job_columns:
                db.execute("ALTER TABLE compute_jobs ADD COLUMN archived INTEGER NOT NULL DEFAULT 0")
            worker_columns = {row[1] for row in db.execute("PRAGMA table_info(workers)")}
            if "removed_at" not in worker_columns:
                db.execute("ALTER TABLE workers ADD COLUMN removed_at TEXT")
            version = db.execute("SELECT version FROM schema_info LIMIT 1").fetchone()
            if version is None:
                db.execute("INSERT INTO schema_info(version) VALUES (?)", (SCHEMA_VERSION,))
            elif int(version[0]) > SCHEMA_VERSION:
                raise RuntimeError("Experiment database was created by a newer application version.")
            elif int(version[0]) < SCHEMA_VERSION:
                db.execute("UPDATE schema_info SET version=?", (SCHEMA_VERSION,))
            # A process crash may leave an active row. Reconcile it as interrupted,
            # never as successfully completed.
            db.execute("UPDATE compute_jobs SET state='failed',ended_at=?,error='Interrupted by application restart.' WHERE state IN ('preparing','running','validating')", (_now(),))
            db.execute("DELETE FROM compute_reservations WHERE job_id IN (SELECT id FROM compute_jobs WHERE state='failed' AND error='Interrupted by application restart.')")

    @staticmethod
    def _decode(row):
        if row is None:
            return None
        item = dict(row)
        for key in list(item):
            if key.endswith("_json"):
                try:
                    item[key[:-5]] = json.loads(item.pop(key))
                except (TypeError, ValueError):
                    item[key[:-5]] = {}
        return item

    def upsert_experiment(self, metadata: dict, description: str | None = None):
        eid = str(metadata["experiment_id"])
        cfg = metadata.get("training_configuration") or {}
        dataset = metadata.get("dataset") or {}
        checkpoints = metadata.get("checkpoints") or {}
        now = _now()
        with self._connect() as db:
            previous = db.execute("SELECT description, created_at FROM experiments WHERE id=?", (eid,)).fetchone()
            db.execute("""INSERT INTO experiments(
                id,name,description,architecture,adapter,model_name,initial_weights,dataset_ref,classes_json,class_mapping_json,
                config_json,start_time,end_time,duration_seconds,status,hardware_json,versions_json,output_dir,best_checkpoint,
                last_checkpoint,error,metadata_json,created_at,updated_at)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET name=excluded.name, architecture=excluded.architecture,
                adapter=excluded.adapter,model_name=excluded.model_name,initial_weights=excluded.initial_weights,
                dataset_ref=excluded.dataset_ref,classes_json=excluded.classes_json,class_mapping_json=excluded.class_mapping_json,
                config_json=excluded.config_json,start_time=excluded.start_time,end_time=excluded.end_time,
                duration_seconds=excluded.duration_seconds,status=excluded.status,hardware_json=excluded.hardware_json,
                versions_json=excluded.versions_json,output_dir=excluded.output_dir,best_checkpoint=excluded.best_checkpoint,
                last_checkpoint=excluded.last_checkpoint,error=excluded.error,metadata_json=excluded.metadata_json,updated_at=excluded.updated_at""",
                (eid, metadata.get("experiment_name") or cfg.get("experiment_name") or eid,
                 description if description is not None else (previous["description"] if previous else metadata.get("description", "")),
                 metadata.get("architecture", "unknown"), metadata.get("adapter", ""), metadata.get("model_version", ""),
                 metadata.get("initial_weights", ""), dataset.get("yaml", ""), _json(dataset.get("classes", [])),
                 _json(metadata.get("class_mapping", [])), _json(cfg), metadata.get("start_time"), metadata.get("end_time"),
                 metadata.get("duration_seconds"), metadata.get("status", "created"), _json(metadata.get("hardware", {})),
                 _json(metadata.get("framework_versions", {})), cfg.get("output_directory", ""), checkpoints.get("best", ""),
                 checkpoints.get("last", ""), metadata.get("error", ""), _json(metadata),
                 previous["created_at"] if previous else now, now))
            for row in metadata.get("metrics", []):
                try:
                    epoch = int(row.get("epoch"))
                except (TypeError, ValueError):
                    continue
                normalized = normalize_metrics(row)
                db.execute("""INSERT INTO experiment_metrics(experiment_id,epoch,raw_json,normalized_json,recorded_at)
                    VALUES(?,?,?,?,?) ON CONFLICT(experiment_id,epoch) DO UPDATE SET
                    raw_json=excluded.raw_json,normalized_json=excluded.normalized_json,recorded_at=excluded.recorded_at""",
                    (eid, epoch, _json(row), _json(normalized), now))
            for path in metadata.get("artifacts", []):
                db.execute("INSERT INTO experiment_artifacts(id,experiment_id,kind,path,metadata_json,created_at) VALUES(?,?,?,?,?,?)",
                           (uuid.uuid5(uuid.NAMESPACE_URL, f"{eid}:{path}").hex, eid, _artifact_kind(path), str(path), "{}", now))
            for label, path in checkpoints.items():
                if path:
                    db.execute("INSERT OR IGNORE INTO experiment_artifacts(id,experiment_id,kind,path,metadata_json,created_at) VALUES(?,?,?,?,?,?)",
                               (uuid.uuid5(uuid.NAMESPACE_URL, f"{eid}:{label}:{path}").hex, eid, f"checkpoint:{label}", str(path), "{}", now))
        return self.get_experiment(eid)

    def get_experiment(self, experiment_id: str):
        with self._connect() as db:
            return self._decode(db.execute("SELECT * FROM experiments WHERE id=?", (experiment_id,)).fetchone())

    def is_experiment_deleted(self, experiment_id: str) -> bool:
        with self._connect() as db:
            return db.execute("SELECT 1 FROM deleted_experiments WHERE experiment_id=?", (experiment_id,)).fetchone() is not None

    def list_experiments(self, query: str = "", status: str = "All"):
        sql = "SELECT * FROM experiments WHERE 1=1"; params = []
        if query.strip():
            like = f"%{query.strip()}%"; sql += " AND (name LIKE ? OR architecture LIKE ? OR model_name LIKE ? OR dataset_ref LIKE ?)"; params.extend([like] * 4)
        if status and status != "All": sql += " AND status=?"; params.append(status.lower())
        sql += " ORDER BY created_at DESC"
        with self._connect() as db:
            return [self._decode(row) for row in db.execute(sql, params).fetchall()]

    def metrics(self, experiment_id: str):
        with self._connect() as db:
            rows = db.execute("SELECT * FROM experiment_metrics WHERE experiment_id=? ORDER BY epoch", (experiment_id,)).fetchall()
        return [{"epoch": row["epoch"], "raw": json.loads(row["raw_json"]), "normalized": json.loads(row["normalized_json"])} for row in rows]

    def artifacts(self, experiment_id: str):
        with self._connect() as db:
            return [self._decode(row) for row in db.execute("SELECT * FROM experiment_artifacts WHERE experiment_id=? ORDER BY kind,path", (experiment_id,)).fetchall()]

    def set_description(self, experiment_id: str, description: str):
        with self._connect() as db:
            db.execute("UPDATE experiments SET description=?,updated_at=? WHERE id=?", (description, _now(), experiment_id))
        return self.get_experiment(experiment_id)

    def delete_experiment_metadata(self, experiment_id: str):
        """Remove DB metadata only; never touches output files or checkpoints."""
        with self._connect() as db:
            db.execute("INSERT OR REPLACE INTO deleted_experiments(experiment_id,deleted_at) VALUES(?,?)", (experiment_id, _now()))
            db.execute("DELETE FROM experiments WHERE id=?", (experiment_id,))

    def register_model(self, *, name: str, version: str, architecture: str, checkpoint_path: str,
                       source_experiment_id: str | None = None, dataset_ref: str = "", classes=None,
                       input_dimensions=None, metrics=None, description="", notes=""):
        checkpoint = Path(checkpoint_path).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
        if not name.strip() or not version.strip():
            raise ValueError("Model name and version are required.")
        model_id = uuid.uuid4().hex
        with self._connect() as db:
            db.execute("""INSERT INTO model_registry(id,name,version,architecture,checkpoint_path,source_experiment_id,
                dataset_ref,classes_json,input_dimensions_json,metrics_json,description,notes,created_at)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (model_id, name.strip(), version.strip(), architecture, str(checkpoint), source_experiment_id,
                 dataset_ref, _json(classes or []), _json(input_dimensions or {}), _json(metrics or {}), description, notes, _now()))
        return self.get_model(model_id)

    def get_model(self, model_id: str):
        with self._connect() as db:
            return self._decode(db.execute("SELECT * FROM model_registry WHERE id=?", (model_id,)).fetchone())

    def list_models(self, query: str = ""):
        sql = "SELECT * FROM model_registry"; params = []
        if query.strip():
            like = f"%{query.strip()}%"; sql += " WHERE name LIKE ? OR architecture LIKE ? OR checkpoint_path LIKE ?"; params = [like] * 3
        sql += " ORDER BY preferred DESC, created_at DESC"
        with self._connect() as db:
            return [self._decode(row) for row in db.execute(sql, params).fetchall()]

    def latest_evaluation(self, *, experiment_id=None, model_id=None):
        if not experiment_id and not model_id:
            return None
        column, value = ("experiment_id", experiment_id) if experiment_id else ("model_id", model_id)
        with self._connect() as db:
            row = db.execute(f"SELECT * FROM evaluations WHERE {column}=? ORDER BY created_at DESC LIMIT 1", (value,)).fetchone()
        return self._decode(row)

    def latest_benchmark(self, *, experiment_id=None, model_id=None):
        if not experiment_id and not model_id:
            return None
        column, value = ("experiment_id", experiment_id) if experiment_id else ("model_id", model_id)
        with self._connect() as db:
            row = db.execute(f"SELECT * FROM benchmarks WHERE {column}=? ORDER BY created_at DESC LIMIT 1", (value,)).fetchone()
        return self._decode(row)

    def set_preferred_model(self, model_id: str):
        with self._connect() as db:
            db.execute("UPDATE model_registry SET preferred=0")
            if db.execute("UPDATE model_registry SET preferred=1 WHERE id=?", (model_id,)).rowcount != 1:
                raise KeyError(model_id)
        return self.get_model(model_id)

    def save_evaluation(self, *, architecture: str, checkpoint_path: str, dataset_ref: str, metrics: dict,
                        per_class=None, confusion_matrix_path="", settings=None, experiment_id=None, model_id=None):
        record_id = uuid.uuid4().hex
        with self._connect() as db:
            db.execute("INSERT INTO evaluations VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                       (record_id, experiment_id, model_id, architecture, checkpoint_path, dataset_ref, _json(metrics),
                        _json(per_class or {}), confusion_matrix_path, _json(settings or {}), _now()))
        return record_id

    def save_benchmark(self, *, checkpoint_path: str, device: str, input_size: int, batch_size: int,
                       warmup_iterations: int, measured_iterations: int, average_latency_ms: float | None,
                       throughput_fps: float | None, settings=None, experiment_id=None, model_id=None):
        record_id = uuid.uuid4().hex
        with self._connect() as db:
            db.execute("INSERT INTO benchmarks VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                       (record_id, experiment_id, model_id, checkpoint_path, device, input_size, batch_size,
                        warmup_iterations, measured_iterations, average_latency_ms, throughput_fps, _json(settings or {}), _now()))
        return record_id

    # Compute Center persistence. Credentials are always stored as salted hashes.
    def get_compute_profile(self):
        with self._connect() as db:
            row = db.execute("SELECT profile_json FROM compute_profiles WHERE profile_id='default'").fetchone()
        return json.loads(row[0]) if row else {}

    def save_compute_profile(self, profile: dict):
        with self._connect() as db:
            db.execute("INSERT INTO compute_profiles(profile_id,profile_json,updated_at) VALUES('default',?,?) "
                       "ON CONFLICT(profile_id) DO UPDATE SET profile_json=excluded.profile_json,updated_at=excluded.updated_at",
                       (_json(profile), _now()))

    def add_enrollment_token(self, token_hash: bytes, expires_at: str):
        with self._connect() as db:
            db.execute("INSERT INTO enrollment_tokens(token_hash,expires_at,created_at) VALUES(?,?,?)",
                       (token_hash, expires_at, _now()))

    def list_enrollment_tokens(self, limit=100):
        with self._connect() as db:
            rows = db.execute("SELECT token_hash,expires_at,consumed_at,created_at,revoked_at FROM enrollment_tokens ORDER BY created_at DESC LIMIT ?",
                              (max(1, min(int(limit), 1000)),)).fetchall()
        return [{"token_id": bytes(row[0]).hex()[:12], "expires_at": row[1], "consumed_at": row[2],
                 "created_at": row[3], "revoked_at": row[4],
                 "status": ("consumed" if row[2] else "revoked" if row[4] else "expired" if row[1] <= _now() else "unused")}
                for row in rows]

    def revoke_enrollment_token(self, token: str):
        token_hash = hashlib.sha256(token.encode("utf-8")).digest()
        with self._connect() as db:
            changed = db.execute("UPDATE enrollment_tokens SET revoked_at=? WHERE token_hash=? AND consumed_at IS NULL AND revoked_at IS NULL",
                                 (_now(), token_hash)).rowcount
        if changed:
            self.record_security_event("enrollment_token_revoked", "enrollment_token", token_hash.hex()[:12])
        return bool(changed)

    def revoke_unused_enrollment_tokens(self):
        with self._connect() as db:
            changed = db.execute("UPDATE enrollment_tokens SET revoked_at=? WHERE consumed_at IS NULL AND revoked_at IS NULL AND expires_at>?",
                                 (_now(), _now())).rowcount
        if changed:
            self.record_security_event("unused_enrollment_tokens_revoked", "enrollment_token", details={"count": changed})
        return changed

    def record_security_event(self, category: str, subject_type="", subject_id="", details=None):
        """Persist structured security metadata while redacting credential-like fields."""
        def scrub(value):
            if isinstance(value, dict):
                return {str(key)[:80]: ("[redacted]" if any(term in str(key).lower()
                        for term in ("token", "credential", "secret", "private_key", "password"))
                        else scrub(item)) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [scrub(item) for item in value[:100]]
            if isinstance(value, (str, int, float, bool)) or value is None:
                return value if not isinstance(value, str) else value[:1000]
            return str(value)[:1000]
        with self._connect() as db:
            db.execute("INSERT INTO security_events(event_id,category,subject_type,subject_id,details_json,created_at) VALUES(?,?,?,?,?,?)",
                       (uuid.uuid4().hex, str(category)[:100], str(subject_type)[:80], str(subject_id)[:160],
                        _json(scrub(details or {})), _now()))

    def list_security_events(self, limit=200):
        with self._connect() as db:
            rows = db.execute("SELECT * FROM security_events ORDER BY created_at DESC LIMIT ?",
                              (max(1, min(int(limit), 5000)),)).fetchall()
        return [{"event_id": row["event_id"], "category": row["category"],
                 "subject_type": row["subject_type"], "subject_id": row["subject_id"],
                 "details": json.loads(row["details_json"]), "created_at": row["created_at"]} for row in rows]

    def enroll_worker(self, token_hash: bytes, worker_id: str, name: str, endpoint: str, credential: str,
                      hardware: dict, software: dict):
        salt = secrets.token_bytes(16)
        credential_hash = hashlib.pbkdf2_hmac("sha256", credential.encode("utf-8"), salt, 240_000)
        now = _now()
        with self._connect() as db:
            token = db.execute("SELECT expires_at,consumed_at,revoked_at FROM enrollment_tokens WHERE token_hash=?", (token_hash,)).fetchone()
            if token is None or token[1] is not None or token[2] is not None or token[0] <= now:
                raise PermissionError("Enrollment token is invalid, expired, or already used.")
            if db.execute("UPDATE enrollment_tokens SET consumed_at=? WHERE token_hash=? AND consumed_at IS NULL", (now, token_hash)).rowcount != 1:
                raise PermissionError("Enrollment token was already consumed.")
            db.execute("INSERT INTO workers(id,name,endpoint,status,approved,enabled,credential_salt,credential_hash,hardware_json,software_json,last_seen,created_at) "
                       "VALUES(?,?,?,'pending',0,1,?,?,?,?,?,?)",
                       (worker_id, name, endpoint, salt, credential_hash, _json(hardware), _json(software), now, now))
        self.record_security_event("worker_enrolled", "worker", worker_id, {"name": name, "endpoint": endpoint})
        return self.get_worker(worker_id)

    def authenticate_worker(self, worker_id: str, credential: str):
        with self._connect() as db:
            row = db.execute("SELECT credential_salt,credential_hash,revoked_at FROM workers WHERE id=?", (worker_id,)).fetchone()
        if row is None or row[2] is not None:
            return None
        candidate = hashlib.pbkdf2_hmac("sha256", credential.encode("utf-8"), row[0], 240_000)
        if not hmac.compare_digest(candidate, row[1]):
            return None
        return self.get_worker(worker_id)

    def get_worker(self, worker_id: str):
        with self._connect() as db:
            row = db.execute("""SELECT id,name,endpoint,status,approved,enabled,hardware_json,software_json,
                last_seen,created_at,revoked_at,connection_id FROM workers WHERE id=?""", (worker_id,)).fetchone()
        return self._decode(row)

    def list_workers(self, *, include_removed=False):
        with self._connect() as db:
            query = """SELECT id,name,endpoint,status,approved,enabled,hardware_json,software_json,
                last_seen,created_at,revoked_at,connection_id,removed_at FROM workers"""
            if not include_removed:
                query += " WHERE removed_at IS NULL"
            rows = db.execute(query + " ORDER BY name,id").fetchall()
        return [self._decode(row) for row in rows]

    def set_worker_approved(self, worker_id: str, approved: bool):
        with self._connect() as db:
            if db.execute("UPDATE workers SET approved=? WHERE id=? AND revoked_at IS NULL", (int(approved), worker_id)).rowcount != 1:
                raise KeyError(worker_id)
            db.execute("UPDATE workers SET status=CASE WHEN ?=0 THEN 'pending' WHEN connection_id IS NOT NULL THEN 'online' ELSE 'offline' END WHERE id=?",
                       (int(approved), worker_id))
        self.record_security_event("worker_approved" if approved else "worker_approval_removed", "worker", worker_id)
        return self.get_worker(worker_id)

    def set_worker_enabled(self, worker_id: str, enabled: bool):
        with self._connect() as db:
            if db.execute("UPDATE workers SET enabled=?,status=CASE WHEN ?=0 THEN 'disabled' ELSE 'offline' END WHERE id=? AND revoked_at IS NULL",
                          (int(enabled), int(enabled), worker_id)).rowcount != 1:
                raise KeyError(worker_id)
        self.record_security_event("worker_enabled" if enabled else "worker_disabled", "worker", worker_id)
        return self.get_worker(worker_id)

    def revoke_worker(self, worker_id: str):
        with self._connect() as db:
            if db.execute("UPDATE workers SET revoked_at=?,status='revoked',connection_id=NULL WHERE id=? AND removed_at IS NULL", (_now(), worker_id)).rowcount != 1:
                raise KeyError(worker_id)
        self.record_security_event("worker_revoked", "worker", worker_id)

    def remove_revoked_worker(self, worker_id: str):
        """Hide a revoked registration without deleting its audit/job history."""
        with self._connect() as db:
            worker = db.execute("SELECT revoked_at,connection_id,removed_at FROM workers WHERE id=?", (worker_id,)).fetchone()
            if worker is None or worker[2] is not None:
                raise KeyError(worker_id)
            if worker[0] is None:
                raise ValueError("Revoke the worker credentials before removing this worker.")
            if worker[1] is not None:
                raise ValueError("Worker is still connected. Wait for it to disconnect before removing it.")
            active = db.execute("SELECT id,state FROM compute_jobs WHERE worker_id=? AND state IN ('queued','preparing','running','validating') LIMIT 1", (worker_id,)).fetchone()
            if active:
                raise ValueError(f"Worker still has active job {active[0]} ({active[1]}). Stop and reconcile that job before removal.")
            db.execute("UPDATE workers SET removed_at=? WHERE id=?", (_now(), worker_id))
        self.record_security_event("worker_record_removed", "worker", worker_id,
                                   {"preserved_job_history": True, "preserved_audit_history": True})

    def record_worker_heartbeat(self, worker_id: str, connection_id: str, hardware: dict, software: dict):
        now = _now()
        with self._connect() as db:
            row = db.execute("SELECT approved,enabled,revoked_at,connection_id FROM workers WHERE id=?", (worker_id,)).fetchone()
            if row is None or not row[1] or row[2] is not None:
                return False
            if row[3] is not None and row[3] != connection_id:
                return False
            state = "online" if row[0] else "pending"
            db.execute("UPDATE workers SET status=?,hardware_json=?,software_json=?,last_seen=?,connection_id=? WHERE id=?",
                       (state, _json(hardware), _json(software), now, connection_id, worker_id))
            db.execute("INSERT INTO worker_heartbeats(worker_id,received_at,hardware_json,software_json) VALUES(?,?,?,?)",
                       (worker_id, now, _json(hardware), _json(software)))
        return True

    def expire_stale_workers(self, timeout_seconds=30):
        cutoff = (datetime.now(timezone.utc) - timedelta(seconds=max(5, int(timeout_seconds)))).isoformat()
        with self._connect() as db:
            rows = db.execute("SELECT id FROM workers WHERE connection_id IS NOT NULL AND last_seen<? AND revoked_at IS NULL", (cutoff,)).fetchall()
            if rows:
                db.execute("UPDATE workers SET status=CASE WHEN approved=1 THEN 'offline' ELSE 'pending' END,connection_id=NULL WHERE connection_id IS NOT NULL AND last_seen<? AND revoked_at IS NULL", (cutoff,))
        for row in rows:
            self.record_security_event("worker_heartbeat_timeout", "worker", row[0], {"timeout_seconds": timeout_seconds})
        return [row[0] for row in rows]

    def disconnect_worker(self, worker_id: str, connection_id: str):
        with self._connect() as db:
            db.execute("UPDATE workers SET status=CASE WHEN approved=1 THEN 'offline' ELSE 'pending' END,connection_id=NULL "
                       "WHERE id=? AND connection_id=? AND revoked_at IS NULL", (worker_id, connection_id))

    def worker_heartbeat_history(self, worker_id: str, limit: int = 100):
        with self._connect() as db:
            rows = db.execute("SELECT * FROM worker_heartbeats WHERE worker_id=? ORDER BY received_at DESC LIMIT ?",
                              (worker_id, max(1, min(int(limit), 1000)))).fetchall()
        return [self._decode(row) for row in rows]

    def create_compute_job(self, job: dict):
        now = _now()
        with self._connect() as db:
            try:
                origin = job.get("origin", "local")
                if origin not in {"local", "remote"}: raise ValueError("Compute job origin must be local or remote.")
                db.execute("INSERT INTO compute_jobs(id,experiment_id,architecture,requirements_json,preferred_device,worker_id,priority,state,created_at,origin,dedupe_key) "
                           "VALUES(?,?,?,?,?,NULL,?,'queued',?,?,?)",
                           (job["id"], job.get("experiment_id"), job["architecture"], _json(job.get("requirements", {})),
                            job.get("preferred_device", "auto"), int(job.get("priority", 0)), now, origin, job.get("dedupe_key")))
            except sqlite3.IntegrityError as exc:
                raise ValueError("An active job already exists for this experiment or output target.") from exc
        return self.get_compute_job(job["id"])

    def get_compute_job(self, job_id: str):
        with self._connect() as db:
            row = db.execute("""SELECT j.*,w.name AS worker_name FROM compute_jobs j LEFT JOIN workers w ON w.id=j.worker_id WHERE j.id=?""", (job_id,)).fetchone()
        return self._decode(row)

    def list_compute_jobs(self):
        with self._connect() as db:
            rows = db.execute("SELECT j.*,w.name AS worker_name FROM compute_jobs j LEFT JOIN workers w ON w.id=j.worker_id ORDER BY j.archived,CASE j.state WHEN 'running' THEN 0 WHEN 'preparing' THEN 1 WHEN 'queued' THEN 2 ELSE 3 END,j.priority DESC,j.created_at").fetchall()
        return [self._decode(row) for row in rows]

    def archive_compute_jobs(self, job_ids):
        job_ids = list(dict.fromkeys(str(job_id) for job_id in job_ids))
        if not job_ids: return 0
        with self._connect() as db:
            placeholders = ",".join("?" for _ in job_ids)
            active = db.execute(f"SELECT id FROM compute_jobs WHERE id IN ({placeholders}) AND state IN ('queued','preparing','running','validating')", job_ids).fetchall()
            if active: raise ValueError("Active jobs cannot be archived. Cancel or stop them and wait for a terminal state.")
            changed = db.execute(f"UPDATE compute_jobs SET archived=1 WHERE id IN ({placeholders}) AND state IN ('completed','failed','cancelled')", job_ids).rowcount
        if changed: self.record_security_event("job_history_archived", "compute_job", details={"job_count": changed})
        return changed

    def delete_compute_jobs(self, job_ids):
        job_ids = list(dict.fromkeys(str(job_id) for job_id in job_ids))
        if not job_ids: return 0
        with self._connect() as db:
            placeholders = ",".join("?" for _ in job_ids)
            rows = db.execute(f"SELECT id,state,origin,worker_id FROM compute_jobs WHERE id IN ({placeholders})", job_ids).fetchall()
            active = [row["id"] for row in rows if row["state"] in {"queued", "preparing", "running", "validating"}]
            if active: raise ValueError("Active jobs cannot be deleted. Cancel or stop them and wait for a terminal state.")
            for row in rows:
                if row["origin"] == "remote" and row["worker_id"] and row["state"] != "cancelled":
                    terminal = db.execute("SELECT state FROM remote_job_updates WHERE job_id=? ORDER BY sequence DESC LIMIT 1", (row["id"],)).fetchone()
                    if terminal is None or terminal[0] not in {"completed", "failed", "cancelled"}:
                        raise ValueError("Remote job has no worker-confirmed terminal update; verify the worker stopped before deleting its history.")
            deleted = db.execute(f"DELETE FROM compute_jobs WHERE id IN ({placeholders}) AND state IN ('completed','failed','cancelled')", job_ids).rowcount
        if deleted: self.record_security_event("job_history_deleted", "compute_job", details={"job_count": deleted})
        return deleted

    def clear_terminal_compute_jobs(self, *, archived_only=False):
        with self._connect() as db:
            predicate = "state IN ('completed','failed','cancelled')" + (" AND archived=1" if archived_only else "")
            rows = db.execute(f"SELECT id FROM compute_jobs WHERE {predicate}").fetchall()
        return self.delete_compute_jobs([row[0] for row in rows])

    def set_compute_job_state(self, job_id: str, state: str, *, error: str = "", worker_id: str | None = None):
        allowed = {"queued", "preparing", "running", "validating", "completed", "failed", "cancelled"}
        if state not in allowed: raise ValueError(f"Invalid compute job state: {state}")
        now = _now()
        with self._connect() as db:
            db.execute("UPDATE compute_jobs SET state=?,error=?,worker_id=COALESCE(?,worker_id), "
                       "started_at=CASE WHEN ?='preparing' AND started_at IS NULL THEN ? ELSE started_at END, "
                       "ended_at=CASE WHEN ? IN ('completed','failed','cancelled') THEN ? ELSE ended_at END WHERE id=?",
                       (state,error,worker_id,state,now,state,now,job_id))
        return self.get_compute_job(job_id)

    def update_compute_job_experiment(self, job_id: str, experiment_id: str):
        with self._connect() as db:
            db.execute("UPDATE compute_jobs SET experiment_id=? WHERE id=?", (experiment_id, job_id))

    def reserve_compute_resources(self, job_id: str, resources: dict, worker_id: str | None = None):
        with self._connect() as db:
            job = db.execute("SELECT state FROM compute_jobs WHERE id=?", (job_id,)).fetchone()
            if job is None or job[0] != "queued": return False
            db.execute("INSERT INTO compute_reservations(job_id,worker_id,resources_json,reserved_at) VALUES(?,?,?,?)",
                       (job_id, worker_id, _json(resources), _now()))
            db.execute("UPDATE compute_jobs SET state='preparing',worker_id=? WHERE id=?", (worker_id, job_id))
        return True

    def compute_reservations(self):
        with self._connect() as db:
            rows = db.execute("SELECT * FROM compute_reservations").fetchall()
        return [self._decode(row) for row in rows]

    def release_compute_resources(self, job_id: str):
        with self._connect() as db: db.execute("DELETE FROM compute_reservations WHERE job_id=?", (job_id,))

    def record_remote_job_update(self, job_id: str, worker_id: str, sequence: int, state: str, progress: dict):
        """Persist an authenticated worker update only for the assigned job and sequence."""
        allowed = {"preparing", "running", "validating", "completed", "failed", "cancelled"}
        if state not in allowed: raise ValueError(f"Invalid remote job state: {state}")
        if not isinstance(progress, dict) or len(_json(progress).encode("utf-8")) > 64_000:
            raise ValueError("Remote progress must be a bounded object.")
        with self._connect() as db:
            row = db.execute("SELECT worker_id,state FROM compute_jobs WHERE id=? AND origin='remote'", (job_id,)).fetchone()
            if row is None: raise KeyError(job_id)
            if row["worker_id"] != worker_id: raise PermissionError("This worker is not assigned to the remote job.")
            if row["state"] in {"completed", "failed", "cancelled"}:
                result = self.get_compute_job(job_id); result["_remote_update_accepted"] = False; return result
            last = db.execute("SELECT MAX(sequence) FROM remote_job_updates WHERE job_id=?", (job_id,)).fetchone()[0]
            if int(sequence) <= (last or 0):
                result = self.get_compute_job(job_id); result["_remote_update_accepted"] = False; return result
            allowed_next = {
                "preparing": {"preparing", "running", "failed", "cancelled"},
                "running": {"running", "validating", "failed", "cancelled"},
                "validating": {"running", "validating", "completed", "failed", "cancelled"},
            }
            if state not in allowed_next.get(row["state"], set()):
                raise ValueError(f"Remote job cannot move from {row['state']} to {state}.")
            db.execute("INSERT INTO remote_job_updates(job_id,sequence,worker_id,state,progress_json,created_at) VALUES(?,?,?,?,?,?)",
                       (job_id, int(sequence), worker_id, state, _json(progress), _now()))
            now = _now()
            db.execute("UPDATE compute_jobs SET state=?,error=?,started_at=CASE WHEN ? IN ('preparing','running') AND started_at IS NULL THEN ? ELSE started_at END,"
                       "ended_at=CASE WHEN ? IN ('completed','failed','cancelled') THEN ? ELSE ended_at END WHERE id=?",
                       (state, str(progress.get("error", "")) if state == "failed" else "", state, now, state, now, job_id))
            if state in {"completed", "failed", "cancelled"}:
                db.execute("DELETE FROM compute_reservations WHERE job_id=?", (job_id,))
        result = self.get_compute_job(job_id); result["_remote_update_accepted"] = True; return result

    def remote_job_updates(self, job_id: str):
        with self._connect() as db:
            rows = db.execute("SELECT * FROM remote_job_updates WHERE job_id=? ORDER BY sequence", (job_id,)).fetchall()
        return [{"job_id": row["job_id"], "sequence": row["sequence"], "worker_id": row["worker_id"],
                 "state": row["state"], "progress": json.loads(row["progress_json"]), "created_at": row["created_at"]}
                for row in rows]

    def request_remote_job_cancel(self, job_id: str):
        event_needed = False
        with self._connect() as db:
            row = db.execute("SELECT state,origin,requirements_json FROM compute_jobs WHERE id=?", (job_id,)).fetchone()
            if row is None: raise KeyError(job_id)
            if row["origin"] != "remote": raise ValueError("Job is not a remote job.")
            if row["state"] == "queued":
                db.execute("UPDATE compute_jobs SET state='cancelled',ended_at=? WHERE id=?", (_now(), job_id))
                result = "cancelled"; event_needed = True
            elif row["state"] in {"completed", "failed", "cancelled"}:
                result = row["state"]
            else:
                requirements = json.loads(row["requirements_json"] or "{}")
                requirements["cancel_requested"] = True
                db.execute("UPDATE compute_jobs SET requirements_json=? WHERE id=?", (_json(requirements), job_id))
                result = "requested"; event_needed = True
        if event_needed:
            self.record_security_event("remote_job_cancellation_requested", "compute_job", job_id, {"state": result})
        return result


_METRIC_ALIASES = {
    "precision": ("metrics/precision(b)", "metrics/precision", "precision", "box(p)"),
    "recall": ("metrics/recall(b)", "metrics/recall", "recall", "box(r)"),
    "map50": ("metrics/map50(b)", "metrics/map50", "map50", "map@50"),
    "map50_95": ("metrics/map50-95(b)", "metrics/map50-95", "map50-95", "map@50-95"),
    "train_loss": ("train/box_loss", "train/cls_loss", "train/dfl_loss", "train/giou_loss", "train/l1_loss"),
    "val_loss": ("val/box_loss", "val/cls_loss", "val/dfl_loss", "val/giou_loss", "val/l1_loss"),
    "learning_rate": ("lr/pg0", "lr/pg1", "lr/pg2", "learning_rate", "lr"),
}


def normalize_metrics(row: dict) -> dict:
    """Extract only known present values, retaining raw metrics separately."""
    lowered = {str(key).strip().casefold(): value for key, value in row.items()}
    result = {}
    for canonical, aliases in _METRIC_ALIASES.items():
        found = []
        for alias in aliases:
            value = lowered.get(alias.casefold())
            try:
                if value not in (None, ""):
                    number = float(value)
                    if number == number and abs(number) != float("inf"):
                        found.append((alias, number))
            except (TypeError, ValueError):
                pass
        if not found:
            continue
        if canonical in ("train_loss", "val_loss"):
            result[canonical] = {name: value for name, value in found}
        elif canonical == "learning_rate":
            result[canonical] = {name: value for name, value in found}
        else:
            result[canonical] = found[0][1]
    if "precision" in result and "recall" in result and result["precision"] + result["recall"] > 0:
        result["f1"] = 2 * result["precision"] * result["recall"] / (result["precision"] + result["recall"])
    for key, value in row.items():
        if key not in ("epoch", "epoch_count", "time"):
            try:
                result.setdefault("architecture_specific", {})[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
    return result


def _artifact_kind(path):
    suffix = Path(path).suffix.lower()
    return "checkpoint" if suffix in (".pt", ".pth", ".onnx", ".engine") else "artifact"
