import hashlib
import sqlite3
import tempfile
import unittest
from pathlib import Path

from experiment_database import ExperimentDatabase, SCHEMA_VERSION


class ComputeJobHistoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent.parent)
        self.root = Path(self.temp.name); self.db = ExperimentDatabase(self.root / "experiments.sqlite3")

    def tearDown(self): self.temp.cleanup()

    def _job(self, job_id="job-1", experiment_id=None, origin="local"):
        return self.db.create_compute_job({"id": job_id, "experiment_id": experiment_id,
            "architecture": "YOLO", "requirements": {}, "origin": origin})

    def test_archive_and_delete_terminal_job_preserves_experiment_and_files(self):
        experiment_id = "experiment-kept"
        self.db.upsert_experiment({"experiment_id": experiment_id, "experiment_name": "Keep me",
            "architecture": "YOLO", "status": "completed", "checkpoints": {"best": str(self.root / "best.pt")}})
        checkpoint = self.root / "best.pt"; checkpoint.write_bytes(b"model")
        self._job(experiment_id=experiment_id)
        self.db.set_compute_job_state("job-1", "completed")
        self.assertEqual(self.db.archive_compute_jobs(["job-1"]), 1)
        self.assertTrue(self.db.get_compute_job("job-1")["archived"])
        self.assertEqual(self.db.delete_compute_jobs(["job-1"]), 1)
        self.assertIsNone(self.db.get_compute_job("job-1"))
        self.assertIsNotNone(self.db.get_experiment(experiment_id))
        self.assertEqual(checkpoint.read_bytes(), b"model")

    def test_active_jobs_cannot_be_archived_or_deleted(self):
        self._job()
        self.db.reserve_compute_resources("job-1", {"device_kind": "cpu"})
        with self.assertRaisesRegex(ValueError, "Active jobs"):
            self.db.archive_compute_jobs(["job-1"])
        with self.assertRaisesRegex(ValueError, "Active jobs"):
            self.db.delete_compute_jobs(["job-1"])
        self.assertIsNotNone(self.db.get_compute_job("job-1"))

    def test_remote_job_without_worker_confirmed_terminal_state_cannot_be_deleted(self):
        token = b"registration"; token_hash = hashlib.sha256(token).digest()
        from datetime import datetime, timedelta, timezone
        self.db.add_enrollment_token(token_hash, (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat())
        self.db.enroll_worker(token_hash, "worker", "Worker", "", "credential", {}, {})
        job = self.db.create_compute_job({"id": "remote-job", "architecture": "YOLO", "requirements": {}, "origin": "remote"})
        self.db.reserve_compute_resources(job["id"], {"device_kind": "remote"}, "worker")
        self.db.set_compute_job_state(job["id"], "failed", error="Controller lost contact")
        with self.assertRaisesRegex(ValueError, "worker-confirmed terminal"):
            self.db.delete_compute_jobs([job["id"]])

    def _worker(self, worker_id="worker-remove"):
        token_hash = hashlib.sha256((worker_id + "-token").encode()).digest()
        from datetime import datetime, timedelta, timezone
        self.db.add_enrollment_token(token_hash, (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat())
        self.db.enroll_worker(token_hash, worker_id, "Revoked worker", "", "credential", {}, {})
        self.db.revoke_worker(worker_id)
        return worker_id

    def test_remove_revoked_worker_hides_registration_and_preserves_job_and_audit_history(self):
        worker_id = self._worker()
        job = self._job("history-job", origin="remote")
        self.db.reserve_compute_resources(job["id"], {"device_kind": "remote"}, worker_id)
        self.db.set_compute_job_state(job["id"], "failed", error="Worker was revoked")
        self.db.remove_revoked_worker(worker_id)
        self.assertNotIn(worker_id, [worker["id"] for worker in self.db.list_workers()])
        self.assertEqual(self.db.get_compute_job(job["id"])["worker_id"], worker_id)
        self.assertIn(worker_id, [worker["id"] for worker in self.db.list_workers(include_removed=True)])
        self.assertTrue(any(event["category"] == "worker_record_removed" for event in self.db.list_security_events()))

    def test_remove_worker_requires_revocation_and_blocks_active_remote_jobs(self):
        worker_id = self._worker("worker-active")
        job = self._job("active-remote", origin="remote")
        self.db.reserve_compute_resources(job["id"], {"device_kind": "remote"}, worker_id)
        with self.assertRaisesRegex(ValueError, "active job"):
            self.db.remove_revoked_worker(worker_id)
        self.db.set_compute_job_state(job["id"], "cancelled")
        self.db.remove_revoked_worker(worker_id)
        self.assertNotIn(worker_id, [worker["id"] for worker in self.db.list_workers()])

    def test_nonrevoked_worker_cannot_be_removed(self):
        token_hash = hashlib.sha256(b"not-revoked-token").digest()
        from datetime import datetime, timedelta, timezone
        self.db.add_enrollment_token(token_hash, (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat())
        self.db.enroll_worker(token_hash, "worker-not-revoked", "Worker", "", "credential", {}, {})
        with self.assertRaisesRegex(ValueError, "Revoke"):
            self.db.remove_revoked_worker("worker-not-revoked")

    def test_schema_v3_database_migrates_without_losing_records(self):
        self._job()
        self.db.set_compute_job_state("job-1", "failed", error="keep failure")
        conn = sqlite3.connect(self.db.path)
        try:
            conn.execute("ALTER TABLE enrollment_tokens DROP COLUMN revoked_at")
            conn.execute("ALTER TABLE compute_jobs DROP COLUMN archived")
            conn.execute("ALTER TABLE workers DROP COLUMN removed_at")
            conn.execute("UPDATE schema_info SET version=3")
            conn.commit()
        finally:
            conn.close()
        migrated = ExperimentDatabase(self.db.path)
        self.assertEqual(SCHEMA_VERSION, 5)
        self.assertEqual(migrated.get_compute_job("job-1")["error"], "keep failure")
        self.assertFalse(migrated.get_compute_job("job-1")["archived"])
        with migrated._connect() as conn:
            self.assertIn("revoked_at", [row[1] for row in conn.execute("PRAGMA table_info(enrollment_tokens)")])
            self.assertIn("removed_at", [row[1] for row in conn.execute("PRAGMA table_info(workers)")])

    def test_security_event_log_redacts_secret_fields_and_keeps_audit_metadata(self):
        self.db.record_security_event("test_auth", "worker", "worker-1",
            {"peer_ip": "10.0.0.4", "token": "never-store-this", "credential": "also-secret"})
        event = self.db.list_security_events(1)[0]
        self.assertEqual(event["category"], "test_auth")
        self.assertEqual(event["details"]["peer_ip"], "10.0.0.4")
        self.assertEqual(event["details"]["token"], "[redacted]")
        self.assertEqual(event["details"]["credential"], "[redacted]")
        with self.db._connect() as conn:
            stored = conn.execute("SELECT details_json FROM security_events").fetchone()[0]
        self.assertNotIn("never-store-this", stored)


if __name__ == "__main__": unittest.main()
