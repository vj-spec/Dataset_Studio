import tempfile
import unittest
import socket
import ssl
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from compute_security import (ControllerCertificateManager, build_enrollment_package,
    save_enrollment_package, validate_enrollment_package)


class ComputeSecurityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent.parent)
        self.root = Path(self.temp.name)
        self.protector = lambda raw, protect: raw[::-1]

    def tearDown(self): self.temp.cleanup()

    def test_certificate_generation_sans_validation_reuse_and_rotation(self):
        manager = ControllerCertificateManager(self.root / "tls", protector=self.protector)
        generated = manager.generate_or_reuse(hostnames=["controller.local"], addresses=["192.168.7.20"])
        self.assertFalse(generated["reused"])
        cert = x509.load_pem_x509_certificate(generated["certificate"].read_bytes())
        san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
        self.assertIn("controller.local", san.get_values_for_type(x509.DNSName))
        self.assertIn("192.168.7.20", [str(value) for value in san.get_values_for_type(x509.IPAddress)])
        self.assertTrue(generated["private_key"].read_bytes().startswith(b"-----BEGIN") is False)
        manager.validate(generated["ca_certificate"], generated["certificate"], generated["private_key"])
        reused = manager.generate_or_reuse(hostnames=["controller.local"], addresses=["192.168.7.20"])
        self.assertTrue(reused["reused"])
        renewed = manager.generate_or_reuse(force_rotate=True, hostnames=["controller.local"], addresses=["192.168.7.20"])
        self.assertFalse(renewed["reused"])
        self.assertEqual(generated["ca_certificate"].read_bytes(), renewed["ca_certificate"].read_bytes())

    def test_certificate_rejects_mismatched_private_key(self):
        manager = ControllerCertificateManager(self.root / "tls", protector=self.protector)
        files = manager.generate_or_reuse(hostnames=["controller.local"], addresses=["127.0.0.1"])
        other_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        encrypted = self.protector(other_key.private_bytes(serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8, serialization.NoEncryption()), True)
        wrong_path = self.root / "wrong.dpapi"; wrong_path.write_bytes(encrypted)
        with self.assertRaisesRegex(ValueError, "do not match"):
            manager.validate(files["ca_certificate"], files["certificate"], wrong_path)

    def test_tls_rejects_hostname_mismatch_and_untrusted_ca(self):
        manager = ControllerCertificateManager(self.root / "tls", protector=self.protector)
        files = manager.generate_or_reuse(hostnames=["controller.local"], addresses=["127.0.0.1"])
        other_manager = ControllerCertificateManager(self.root / "other-tls", protector=self.protector)
        other_files = other_manager.generate_or_reuse(hostnames=["other.local"], addresses=["127.0.0.1"])
        with manager.runtime_private_key(files["private_key"]) as key_path:
            server_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            server_context.minimum_version = ssl.TLSVersion.TLSv1_2
            server_context.load_cert_chain(str(files["certificate"]), str(key_path))

            def assert_rejected(ca_path, hostname):
                listener = socket.socket(); listener.bind(("127.0.0.1", 0)); listener.listen(1)
                port = listener.getsockname()[1]; server_errors = []
                def serve():
                    try:
                        raw, _ = listener.accept()
                        with server_context.wrap_socket(raw, server_side=True): pass
                    except (OSError, ssl.SSLError) as exc: server_errors.append(exc)
                    finally: listener.close()
                thread = threading.Thread(target=serve, daemon=True); thread.start()
                context = ssl.create_default_context(cafile=str(ca_path)); context.minimum_version = ssl.TLSVersion.TLSv1_2
                raw = socket.create_connection(("127.0.0.1", port), timeout=3)
                with self.assertRaises(ssl.SSLCertVerificationError): context.wrap_socket(raw, server_hostname=hostname)
                thread.join(3)

            assert_rejected(files["ca_certificate"], "not-the-controller.local")
            assert_rejected(other_files["ca_certificate"], "controller.local")

    def test_enrollment_package_contains_only_public_ca_and_expiring_token(self):
        manager = ControllerCertificateManager(self.root / "tls", protector=self.protector)
        files = manager.generate_or_reuse(hostnames=["controller.local"], addresses=["127.0.0.1"])
        package = build_enrollment_package(host="controller.local", port=8765,
            ca_certificate=files["ca_certificate"].read_text(encoding="ascii"), token="one-time-secret-" + "x" * 32,
            expires_at=(datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat())
        self.assertNotIn("private_key", package)
        self.assertNotIn("worker_credentials", package)
        self.assertEqual(validate_enrollment_package(package)["controller_host"], "controller.local")
        path = save_enrollment_package(self.root / "enroll.json", package)
        self.assertTrue(path.is_file())
        package["expires_at"] = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
        with self.assertRaisesRegex(ValueError, "expired"):
            validate_enrollment_package(package)

    def test_enrollment_package_rejects_injected_private_material(self):
        package = {"format": "dataset-studio-enrollment", "version": 1,
            "controller_host": "localhost", "controller_port": 8765,
            "ca_certificate_pem": "bad", "token": "token",
            "expires_at": (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat(),
            "ca_private_key": "secret"}
        with self.assertRaises(ValueError): validate_enrollment_package(package)


if __name__ == "__main__": unittest.main()
