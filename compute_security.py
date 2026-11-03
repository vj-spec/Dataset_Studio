"""Certificate and enrollment-package helpers for the Compute Center."""
from __future__ import annotations

import ipaddress
import json
import os
import re
import socket
import tempfile
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from cryptography import x509
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID


def _utc_now():
    return datetime.now(timezone.utc)


def controller_sans(hostnames=None, addresses=None):
    """Return unique valid DNS and IP identities for this computer."""
    names = set(hostnames or ())
    ips = set(addresses or ())
    names.update({socket.gethostname(), socket.getfqdn(), "localhost"})
    try:
        for item in socket.getaddrinfo(socket.gethostname(), None):
            ips.add(item[4][0])
    except OSError:
        pass
    try:
        import psutil
        for interface in psutil.net_if_addrs().values():
            for address in interface:
                if address.family in (socket.AF_INET, socket.AF_INET6):
                    ips.add(address.address.split("%", 1)[0])
    except (ImportError, OSError):
        pass
    dns = sorted({name.rstrip(".") for name in names if name and len(name) <= 253 and
                  all(label and len(label) <= 63 for label in name.rstrip(".").split("."))})
    parsed_ips = set()
    for value in ips:
        try:
            parsed = ipaddress.ip_address(value)
            if not parsed.is_unspecified and not parsed.is_multicast:
                parsed_ips.add(parsed)
        except ValueError:
            continue
    return [x509.DNSName(name) for name in dns] + [x509.IPAddress(addr) for addr in sorted(parsed_ips, key=str)]


class ControllerCertificateManager:
    """Generate and validate an app-local CA and TLS server certificate.

    Private keys are protected at rest with Windows DPAPI. A short-lived PEM
    is materialized only because Python's SSLContext loader accepts file paths.
    """
    def __init__(self, root: str | Path, protector=None):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._protector = protector

    def _protect(self, raw: bytes) -> bytes:
        if self._protector:
            return self._protector(raw, True)
        from compute_manager import DPAPICredentialStore
        return DPAPICredentialStore._crypt(raw, True, b"Dataset Studio controller TLS key v1")

    def _unprotect(self, raw: bytes) -> bytes:
        if self._protector:
            return self._protector(raw, False)
        from compute_manager import DPAPICredentialStore
        return DPAPICredentialStore._crypt(raw, False, b"Dataset Studio controller TLS key v1")

    @staticmethod
    def _load_cert(path):
        return x509.load_pem_x509_certificate(Path(path).read_bytes())

    def _load_key(self, path):
        return serialization.load_pem_private_key(self._unprotect(Path(path).read_bytes()), password=None)

    @staticmethod
    def _key_matches(cert, key):
        public_a = cert.public_key().public_bytes(serialization.Encoding.DER,
                                                   serialization.PublicFormat.SubjectPublicKeyInfo)
        public_b = key.public_key().public_bytes(serialization.Encoding.DER,
                                                  serialization.PublicFormat.SubjectPublicKeyInfo)
        return public_a == public_b

    def generate_or_reuse(self, *, force_rotate=False, rotate_ca=False, hostnames=None, addresses=None,
                          ca_valid_days=3650, server_valid_days=397):
        ca_cert_path = self.root / "controller-ca.pem"
        ca_key_path = self.root / "controller-ca-key.dpapi"
        cert_path = self.root / "controller-cert.pem"
        key_path = self.root / "controller-key.dpapi"
        ca_key = ca_cert = None
        if not rotate_ca and ca_cert_path.is_file() and ca_key_path.is_file():
            try:
                ca_cert = self._load_cert(ca_cert_path)
                ca_key = self._load_key(ca_key_path)
                if not self._key_matches(ca_cert, ca_key) or not ca_cert.extensions.get_extension_for_class(x509.BasicConstraints).value.ca:
                    ca_cert = ca_key = None
                if ca_cert is not None:
                    if ca_cert.issuer != ca_cert.subject or ca_cert.not_valid_before_utc > _utc_now():
                        ca_cert = ca_key = None
                    else:
                        ca_cert.public_key().verify(ca_cert.signature, ca_cert.tbs_certificate_bytes,
                            ca_cert.signature_algorithm_parameters or padding.PKCS1v15(), ca_cert.signature_hash_algorithm)
            except (ValueError, OSError, TypeError, x509.ExtensionNotFound, InvalidSignature):
                ca_cert = ca_key = None

        now = _utc_now()
        if ca_key is None or ca_cert is None or ca_cert.not_valid_after_utc <= now:
            ca_key = rsa.generate_private_key(public_exponent=65537, key_size=3072)
            ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Dataset Studio Local Compute CA")])
            ca_cert = (x509.CertificateBuilder().subject_name(ca_name).issuer_name(ca_name)
                .public_key(ca_key.public_key()).serial_number(x509.random_serial_number())
                .not_valid_before(now - timedelta(minutes=5)).not_valid_after(now + timedelta(days=ca_valid_days))
                .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
                .add_extension(x509.KeyUsage(digital_signature=True, content_commitment=False, key_encipherment=False,
                    data_encipherment=False, key_agreement=False, key_cert_sign=True, crl_sign=True,
                    encipher_only=False, decipher_only=False), critical=True)
                .sign(ca_key, hashes.SHA256()))
            self._atomic_write(ca_cert_path, ca_cert.public_bytes(serialization.Encoding.PEM))
            self._atomic_write(ca_key_path, self._protect(ca_key.private_bytes(serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8, serialization.NoEncryption())))
        ca_name = ca_cert.subject

        expected_names = controller_sans(hostnames, addresses)
        expected_values = {(type(item), item.value) for item in expected_names}
        if not force_rotate and cert_path.is_file() and key_path.is_file():
            try:
                validated = self.validate(ca_cert_path, cert_path, key_path)
                existing_sans = validated["certificate"].extensions.get_extension_for_class(x509.SubjectAlternativeName).value
                present = {(type(item), item.value) for item in existing_sans}
                if expected_values.issubset(present):
                    return {"ca_certificate": ca_cert_path, "certificate": cert_path,
                            "private_key": key_path, "reused": True}
            except (ValueError, OSError, TypeError, x509.ExtensionNotFound):
                pass

        server_key = rsa.generate_private_key(public_exponent=65537, key_size=3072)
        server_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, socket.getfqdn()[:64])])
        san_entries = expected_names
        server_cert = (x509.CertificateBuilder().subject_name(server_name).issuer_name(ca_name)
            .public_key(server_key.public_key()).serial_number(x509.random_serial_number())
            .not_valid_before(now - timedelta(minutes=5)).not_valid_after(now + timedelta(days=server_valid_days))
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .add_extension(x509.SubjectAlternativeName(san_entries), critical=False)
            .add_extension(x509.KeyUsage(digital_signature=True, content_commitment=False, key_encipherment=True,
                data_encipherment=False, key_agreement=False, key_cert_sign=False, crl_sign=False,
                encipher_only=False, decipher_only=False), critical=True)
            .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
            .sign(ca_key, hashes.SHA256()))
        self._atomic_write(cert_path, server_cert.public_bytes(serialization.Encoding.PEM))
        self._atomic_write(key_path, self._protect(server_key.private_bytes(serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8, serialization.NoEncryption())))
        self.validate(ca_cert_path, cert_path, key_path)
        return {"ca_certificate": ca_cert_path, "certificate": cert_path,
                "private_key": key_path, "reused": False}

    @staticmethod
    def _atomic_write(path: Path, contents: bytes):
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_bytes(contents)
        os.replace(temp, path)

    def validate(self, ca_certificate, certificate, private_key, *, key_is_encrypted=True):
        ca = self._load_cert(ca_certificate)
        cert = self._load_cert(certificate)
        key = self._load_key(private_key) if key_is_encrypted else serialization.load_pem_private_key(Path(private_key).read_bytes(), password=None)
        now = _utc_now()
        if cert.not_valid_before_utc > now or cert.not_valid_after_utc <= now:
            raise ValueError("Controller TLS certificate is not currently valid.")
        if ca.not_valid_before_utc > now or ca.not_valid_after_utc <= now:
            raise ValueError("Controller CA certificate is not currently valid.")
        if not self._key_matches(cert, key):
            raise ValueError("Controller certificate and private key do not match.")
        try:
            constraints = ca.extensions.get_extension_for_class(x509.BasicConstraints).value
            if not constraints.ca:
                raise ValueError("Configured trust certificate is not a CA certificate.")
            cert.extensions.get_extension_for_class(x509.SubjectAlternativeName)
            cert.extensions.get_extension_for_class(x509.ExtendedKeyUsage)
        except x509.ExtensionNotFound as exc:
            raise ValueError("Controller certificate is missing required TLS extensions.") from exc
        if cert.issuer != ca.subject:
            raise ValueError("Controller certificate was not issued by the configured CA.")
        try:
            ca.public_key().verify(cert.signature, cert.tbs_certificate_bytes,
                cert.signature_algorithm_parameters or padding.PKCS1v15(), cert.signature_hash_algorithm)
        except InvalidSignature as exc:
            raise ValueError("Controller certificate signature does not match the configured CA.") from exc
        return {"ca": ca, "certificate": cert, "private_key": key}

    @contextmanager
    def runtime_private_key(self, encrypted_key_path):
        pem = self._unprotect(Path(encrypted_key_path).read_bytes())
        fd, filename = tempfile.mkstemp(prefix="dataset-studio-tls-", suffix=".pem", dir=self.root)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(pem); stream.flush(); os.fsync(stream.fileno())
            yield Path(filename)
        finally:
            try: Path(filename).unlink(missing_ok=True)
            except OSError: pass


def build_enrollment_package(*, host: str, port: int, ca_certificate: str,
                             token: str, expires_at: str) -> dict:
    if not host or not 1 <= int(port) <= 65535 or not token:
        raise ValueError("Controller address, port, and one-time token are required.")
    return {"format": "dataset-studio-enrollment", "version": 1,
        "controller_host": str(host).strip(), "controller_port": int(port),
        "ca_certificate_pem": str(ca_certificate), "token": token, "expires_at": expires_at}


def validate_enrollment_package(package: dict):
    if not isinstance(package, dict) or package.get("format") != "dataset-studio-enrollment" or package.get("version") != 1:
        raise ValueError("This is not a supported Dataset Studio enrollment package.")
    allowed = {"format", "version", "controller_host", "controller_port", "ca_certificate_pem", "token", "expires_at"}
    if set(package) - allowed:
        raise ValueError("Enrollment package contains unsupported fields, possibly including private material.")
    host, token, ca_pem = package.get("controller_host"), package.get("token"), package.get("ca_certificate_pem")
    try:
        port = int(package.get("controller_port"))
        expires = datetime.fromisoformat(str(package.get("expires_at")).replace("Z", "+00:00"))
        if not isinstance(ca_pem, str) or len(ca_pem) > 1024 * 1024:
            raise ValueError("CA certificate is too large.")
        ca = x509.load_pem_x509_certificate(ca_pem.encode("ascii"))
        constraints = ca.extensions.get_extension_for_class(x509.BasicConstraints).value
        if not constraints.ca or ca.not_valid_before_utc > _utc_now() or ca.not_valid_after_utc <= _utc_now():
            raise ValueError("Enrollment package CA is not a currently valid CA certificate.")
        try:
            ca.public_key().verify(ca.signature, ca.tbs_certificate_bytes,
                ca.signature_algorithm_parameters or padding.PKCS1v15(), ca.signature_hash_algorithm)
        except InvalidSignature as exc:
            raise ValueError("Enrollment package CA signature is invalid.") from exc
    except Exception as exc:
        raise ValueError("Enrollment package fields or public CA certificate are invalid.") from exc
    if (not isinstance(host, str) or not host.strip() or len(host) > 253 or any(ch.isspace() for ch in host)
            or not isinstance(token, str) or not re.fullmatch(r"[A-Za-z0-9_-]{32,128}", token)
            or not 1 <= port <= 65535):
        raise ValueError("Enrollment package endpoint or token is invalid.")
    if expires.tzinfo is None or expires <= _utc_now():
        raise ValueError("Enrollment package token has expired.")
    return {"controller_host": str(host), "controller_port": port,
            "ca_certificate_pem": str(ca_pem), "token": token,
            "expires_at": expires.astimezone(timezone.utc).isoformat()}


def save_enrollment_package(path, package):
    validated = validate_enrollment_package(package)
    target = Path(path).expanduser().resolve()
    temp = target.with_suffix(target.suffix + ".tmp")
    temp.write_text(json.dumps(validated, indent=2), encoding="utf-8")
    os.replace(temp, target)
    return target
