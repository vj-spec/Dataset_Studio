"""Compute Center views embedded in the existing Dataset Studio window."""
from __future__ import annotations

import json
import hashlib
import socket
from datetime import datetime, timezone
import uuid
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, QThread, Signal
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox, QLabel,
    QPushButton, QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox, QTableWidget, QTableWidgetItem,
    QFileDialog, QMessageBox, QInputDialog, QSplitter, QTabWidget, QDialog, QStackedWidget,
    QProgressBar, QCheckBox, QApplication, QScrollArea, QHeaderView)

from config import app_state_file
from compute_manager import HardwareMonitorWorker, ResourceProfile, UnifiedComputeManager
from distributed_training import DatasetCache, remote_job_suitability
from compute_security import (ControllerCertificateManager, build_enrollment_package,
    save_enrollment_package, validate_enrollment_package)


class _RemoteSubmitThread(QThread):
    completed = Signal(str)
    failed = Signal(str)
    def __init__(self, compute, config, dataset_root, worker_id, priority, ram_mb=0, gpu_memory_mb=0, parent=None):
        super().__init__(parent); self.compute=compute; self.config=config; self.dataset_root=dataset_root
        self.worker_id=worker_id; self.priority=priority; self.ram_mb=ram_mb; self.gpu_memory_mb=gpu_memory_mb
    def run(self):
        try: self.completed.emit(self.compute.submit_remote_job(self.config, self.dataset_root,
            worker_id=self.worker_id, priority=self.priority, ram_mb=self.ram_mb,
            gpu_memory_mb=self.gpu_memory_mb))
        except Exception as exc: self.failed.emit(str(exc))


class _EnrollmentTokenDialog(QDialog):
    def __init__(self, compute, token, expires_at, on_export=None, parent=None):
        super().__init__(parent); self.compute=compute; self.token=token; self.expires_at=expires_at; self.on_export=on_export
        self.setWindowTitle("Worker enrollment token"); self.setMinimumWidth(460)
        layout=QVBoxLayout(self)
        layout.addWidget(QLabel("Share this one-use token only through a trusted private channel. It expires automatically."))
        self.token_edit=QLineEdit(token); self.token_edit.setEchoMode(QLineEdit.Password); self.token_edit.setReadOnly(True)
        layout.addWidget(self.token_edit)
        self.show_token=QCheckBox("Show token"); self.show_token.toggled.connect(lambda show: self.token_edit.setEchoMode(QLineEdit.Normal if show else QLineEdit.Password)); layout.addWidget(self.show_token)
        self.status=QLabel(); self.status.setWordWrap(True); layout.addWidget(self.status)
        actions=QVBoxLayout(); first_row=QHBoxLayout()
        copy=QPushButton("Copy token"); copy.clicked.connect(self.copy_token); first_row.addWidget(copy)
        another=QPushButton("Generate another token"); another.clicked.connect(self.generate_another); first_row.addWidget(another)
        actions.addLayout(first_row); second_row=QHBoxLayout()
        revoke=QPushButton("Revoke unused tokens"); revoke.clicked.connect(self.revoke_unused); second_row.addWidget(revoke)
        if on_export:
            package=QPushButton("Export worker package…"); package.clicked.connect(lambda: on_export(self.token, self.expires_at)); second_row.addWidget(package)
        actions.addLayout(second_row); layout.addLayout(actions)
        close=QPushButton("Close"); close.clicked.connect(self.accept); layout.addWidget(close, alignment=Qt.AlignRight)
        self.timer=QTimer(self); self.timer.timeout.connect(self.update_status); self.timer.start(500); self.update_status()

    def copy_token(self):
        if self.token: QApplication.clipboard().setText(self.token)

    def generate_another(self):
        if self.token: self.compute.database.revoke_enrollment_token(self.token)
        self.token, self.expires_at=self.compute.create_enrollment_token()
        self.token_edit.setText(self.token); self.token_edit.setEchoMode(QLineEdit.Password); self.show_token.setChecked(False)
        self.update_status()

    def revoke_unused(self):
        count=self.compute.database.revoke_unused_enrollment_tokens()
        self.token=""; self.token_edit.clear(); self.status.setText(f"Revoked {count} unused token(s). Generate another token to enroll a worker.")

    def update_status(self):
        if not self.token:
            self.status.setText("No active token is available."); return
        token_id=hashlib.sha256(self.token.encode("utf-8")).hexdigest()[:12]
        rows=self.compute.database.list_enrollment_tokens()
        record=next((row for row in rows if row["token_id"] == token_id), None)
        try:
            expires=datetime.fromisoformat(self.expires_at.replace("Z", "+00:00"))
            seconds=max(0, int((expires-datetime.now(timezone.utc)).total_seconds()))
        except (ValueError, TypeError, AttributeError): seconds=0
        state=record["status"] if record else "unknown"
        if state != "unused" or seconds <= 0:
            self.token=""; self.token_edit.clear()
            self.status.setText(f"Token status: {state}. Token cleared from memory.")
            return
        self.status.setText(f"Status: unused · Expires in {seconds // 60} min {seconds % 60:02d} sec")


class ComputeCenterPanel(QWidget):
    def __init__(self, manager: UnifiedComputeManager, parent=None):
        super().__init__(parent); self.compute = manager; self.database = manager.database
        self._hardware = {}; self._closing = False; self._identity = self._load_identity(); self._build()
        self.compute.events.worker_changed.connect(lambda _worker: self.refresh_workers())
        self.compute.events.error.connect(self._network_error)
        self.compute.events.connected.connect(self._worker_connection_status)
        self.compute.mode_changed.connect(self._mode_changed)
        self.compute.scheduler.job_changed.connect(lambda _job: self.refresh_jobs())
        self.compute.scheduler.scheduling_message.connect(self._schedule_message)
        self.monitor = HardwareMonitorWorker(self.compute.resource_profile.monitoring_interval_seconds, self)
        self.monitor.snapshot.connect(self._update_hardware); self.monitor.failed.connect(self._network_error); self.monitor.start()
        self.refresh_workers(); self.refresh_jobs()
        self.timer = QTimer(self); self.timer.timeout.connect(self._refresh_stale_workers); self.timer.start(3000)
        self._mode_changed(self.compute.mode)

    def _mode_changed(self, mode):
        self.mode.setCurrentText(mode)
        self.worker_status.setText(f"Operating mode: {mode}. Start or connect network services explicitly from this page.")

    @staticmethod
    def _load_identity():
        path = app_state_file("compute_worker_config.json")
        try: return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError): return {}

    def _save_identity(self):
        values = {"worker_id": self.worker_id.text().strip(), "name": self.worker_name.text().strip(),
                  "controller_host": self.controller_host.text().strip(), "controller_port": self.controller_port.value(),
                  "ca_file": self.ca_file.text().strip()}
        if not values["worker_id"]: values["worker_id"] = uuid.uuid4().hex
        self.worker_id.setText(values["worker_id"]); self._identity = values
        path = app_state_file("compute_worker_config.json"); temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(values, indent=2), encoding="utf-8"); temporary.replace(path)

    def _build(self):
        root = QVBoxLayout(self)
        top = QHBoxLayout(); self.mode = QComboBox(); self.mode.addItems(["Controller", "Worker", "Hybrid"]); self.mode.setCurrentText(self.compute.mode)
        apply_mode = QPushButton("Apply operating mode"); apply_mode.clicked.connect(self.apply_mode)
        self.overview = QLabel("Discovering local compute resources…"); self.overview.setWordWrap(True)
        top.addWidget(QLabel("Operating mode:")); top.addWidget(self.mode); top.addWidget(apply_mode); top.addWidget(self.overview, 1); root.addLayout(top)
        self.tabs = QTabWidget(); root.addWidget(self.tabs, 1)
        self.tabs.addTab(self._hardware_tab(), "Overview & Hardware")
        self.tabs.addTab(self._workers_tab(), "Worker Manager")
        self.tabs.addTab(self._jobs_tab(), "Job Queue")
        self.tabs.addTab(self._settings_tab(), "Compute Settings")

    def _hardware_tab(self):
        pane = QWidget(); layout = QVBoxLayout(pane)
        self.hardware_summary = QLabel("Collecting CPU, memory, disk and GPU status in background…"); self.hardware_summary.setWordWrap(True)
        layout.addWidget(self.hardware_summary)
        self.resource_summary = QLabel(""); self.resource_summary.setWordWrap(True); layout.addWidget(self.resource_summary)
        self.gpu_table = QTableWidget(0, 6); self.gpu_table.setHorizontalHeaderLabels(["GPU", "Vendor", "VRAM used / total MB", "Free MB", "GPU %", "CUDA"])
        self.gpu_table.setEditTriggers(QTableWidget.NoEditTriggers); layout.addWidget(self.gpu_table, 1)
        return pane

    def _workers_tab(self):
        pane = QWidget(); outer = QVBoxLayout(pane); scroll = QScrollArea(); scroll.setWidgetResizable(True)
        content = QWidget(); layout = QVBoxLayout(content); scroll.setWidget(content); outer.addWidget(scroll)
        controller = QGroupBox("Controller service — outbound worker connections use TLS")
        form = QFormLayout(controller)
        self.listen_host = QLineEdit(str(self.compute.settings.get("listen_host", "127.0.0.1")))
        self.listen_port = QSpinBox(); self.listen_port.setRange(1, 65535); self.listen_port.setValue(int(self.compute.settings.get("listen_port", 8765)))
        self.tls_cert = QLineEdit(str(self.compute.settings.get("tls_cert", "")))
        cert_row = QHBoxLayout(); cert_row.addWidget(self.tls_cert, 1); cert_browse = QPushButton("Browse…"); cert_browse.clicked.connect(lambda: self._browse_file(self.tls_cert, "TLS certificate")); cert_row.addWidget(cert_browse)
        self.tls_key = QLineEdit(str(self.compute.settings.get("tls_key", "")))
        key_row = QHBoxLayout(); key_row.addWidget(self.tls_key, 1); key_browse = QPushButton("Browse…"); key_browse.clicked.connect(lambda: self._browse_file(self.tls_key, "TLS private key")); key_row.addWidget(key_browse)
        self.tls_ca = QLineEdit(str(self.compute.settings.get("tls_ca", "")))
        ca_row = QHBoxLayout(); ca_row.addWidget(self.tls_ca, 1); ca_browse = QPushButton("Browse CA…")
        ca_browse.clicked.connect(lambda: self._browse_file(self.tls_ca, "Controller CA certificate")); ca_row.addWidget(ca_browse)
        form.addRow("Bind address:", self.listen_host); form.addRow("Port:", self.listen_port)
        form.addRow("Certificate:", cert_row); form.addRow("Private key or DPAPI key:", key_row); form.addRow("Public CA certificate:", ca_row)
        setup_tls = QPushButton("Set up or reuse controller certificates…"); setup_tls.clicked.connect(self.setup_controller_certificates)
        rotate_tls = QPushButton("Renew controller certificate…"); rotate_tls.clicked.connect(lambda: self.setup_controller_certificates(force_rotate=True))
        rotate_ca = QPushButton("Rotate CA…"); rotate_ca.clicked.connect(lambda: self.setup_controller_certificates(force_rotate=True, rotate_ca=True))
        tls_buttons = QVBoxLayout(); tls_row = QHBoxLayout(); tls_row.addWidget(setup_tls); tls_row.addWidget(rotate_tls)
        tls_buttons.addLayout(tls_row); tls_buttons.addWidget(rotate_ca); form.addRow(tls_buttons)
        controller_buttons = QHBoxLayout(); start = QPushButton("Start secure controller"); start.clicked.connect(self.start_controller)
        stop = QPushButton("Stop controller"); stop.clicked.connect(self.stop_controller); token = QPushButton("Create one-time enrollment token…"); token.clicked.connect(self.create_token)
        controller_buttons.addWidget(start); controller_buttons.addWidget(stop); controller_buttons.addWidget(token); form.addRow(controller_buttons)
        export_package = QPushButton("Export worker enrollment package…"); export_package.clicked.connect(self.export_worker_package)
        form.addRow(export_package)
        layout.addWidget(controller)
        worker = QGroupBox("Worker service — outbound connection, remote training, and status")
        wf = QFormLayout(worker)
        self.worker_name = QLineEdit(str(self._identity.get("name", "Dataset Studio Worker")))
        self.worker_id = QLineEdit(str(self._identity.get("worker_id", ""))); self.worker_id.setReadOnly(True)
        self.controller_host = QLineEdit(str(self._identity.get("controller_host", "")))
        self.controller_port = QSpinBox(); self.controller_port.setRange(1, 65535); self.controller_port.setValue(int(self._identity.get("controller_port", 8765)))
        self.ca_file = QLineEdit(str(self._identity.get("ca_file", "")))
        ca_row = QHBoxLayout(); ca_row.addWidget(self.ca_file, 1); ca_browse = QPushButton("Browse CA/certificate…"); ca_browse.clicked.connect(lambda: self._browse_file(self.ca_file, "Controller CA/certificate")); ca_row.addWidget(ca_browse)
        self.enrollment = QLineEdit(); self.enrollment.setPlaceholderText("One-time token for first enrollment; kept in memory only")
        self.enrollment.setEchoMode(QLineEdit.Password)
        self.show_enrollment_token = QCheckBox("Show enrollment token")
        self.show_enrollment_token.toggled.connect(lambda show: self.enrollment.setEchoMode(QLineEdit.Normal if show else QLineEdit.Password))
        self.worker_status = QLabel("Disconnected"); self.worker_status.setWordWrap(True)
        wf.addRow("Worker name:", self.worker_name); wf.addRow("Persistent worker ID:", self.worker_id)
        wf.addRow("Controller host:", self.controller_host); wf.addRow("Port:", self.controller_port)
        wf.addRow("Controller trust certificate:", ca_row); wf.addRow("Enrollment token:", self.enrollment); wf.addRow(self.show_enrollment_token)
        worker_buttons = QHBoxLayout(); connect = QPushButton("Connect worker securely"); connect.clicked.connect(self.connect_worker)
        disconnect = QPushButton("Disconnect worker"); disconnect.clicked.connect(self.disconnect_worker)
        import_package = QPushButton("Join existing Controller…"); import_package.clicked.connect(self.import_worker_package)
        worker_buttons.addWidget(import_package); worker_buttons.addWidget(connect); worker_buttons.addWidget(disconnect); wf.addRow(worker_buttons); wf.addRow("Connection:", self.worker_status)
        layout.addWidget(worker)
        cache_row = QHBoxLayout(); cache_cleanup = QPushButton("Clear worker dataset cache…")
        cache_cleanup.clicked.connect(self.clear_worker_dataset_cache); cache_row.addWidget(cache_cleanup); cache_row.addStretch(1); layout.addLayout(cache_row)
        security = QPushButton("View security events…"); security.clicked.connect(self.view_security_events); layout.addWidget(security)
        layout.addWidget(QLabel("Approved workers can receive validated YOLO and RT-DETR jobs. Dataset files are staged by content hash; received checkpoints are verified before being attached to experiment history."))
        self.worker_table = QTableWidget(0, 11); self.worker_table.setHorizontalHeaderLabels(["Name", "ID", "Status", "Approval", "Enabled", "CPU / RAM / Disk", "GPU / VRAM", "Architectures", "PyTorch / Ultralytics", "Active jobs", "Last heartbeat"])
        self.worker_table.setSelectionBehavior(QTableWidget.SelectRows); self.worker_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.worker_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded); self.worker_table.setMinimumHeight(180)
        self.worker_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        layout.addWidget(self.worker_table, 1)
        actions = QHBoxLayout()
        for label, slot in (("Approve selected", self.approve_worker), ("Disable / enable", self.toggle_worker),
                            ("Revoke credentials…", self.revoke_worker), ("Remove revoked worker…", self.remove_revoked_worker)):
            button = QPushButton(label); button.clicked.connect(slot); actions.addWidget(button)
        layout.addLayout(actions)
        return pane

    def _jobs_tab(self):
        pane = QWidget(); layout = QVBoxLayout(pane)
        history = QHBoxLayout(); self.jobs_summary = QLabel("No compute jobs submitted.")
        self.job_history_filter = QComboBox(); self.job_history_filter.addItem("Current history", False); self.job_history_filter.addItem("Archived history", True)
        self.job_history_filter.currentIndexChanged.connect(self.refresh_jobs)
        history.addWidget(self.jobs_summary, 1); history.addWidget(QLabel("View:")); history.addWidget(self.job_history_filter); layout.addLayout(history)
        self.jobs_table = QTableWidget(0, 11); self.jobs_table.setHorizontalHeaderLabels(["Job ID", "Experiment", "Architecture", "Device", "State", "Priority", "Worker", "Progress", "Latest checkpoint", "Created", "Archived"])
        self.jobs_table.setSelectionBehavior(QTableWidget.SelectRows); self.jobs_table.setSelectionMode(QTableWidget.ExtendedSelection)
        self.jobs_table.setEditTriggers(QTableWidget.NoEditTriggers); self.jobs_table.setWordWrap(False)
        self.jobs_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.jobs_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.jobs_table.horizontalHeader().setSectionResizeMode(7, QHeaderView.Stretch)
        layout.addWidget(self.jobs_table, 1)
        actions = QHBoxLayout(); remote = QPushButton("Guided remote training…"); remote.clicked.connect(self.guided_remote_training)
        cancel = QPushButton("Cancel / stop selected job safely"); cancel.clicked.connect(self.cancel_job); refresh = QPushButton("Refresh"); refresh.clicked.connect(self.refresh_jobs)
        actions.addWidget(remote); actions.addWidget(cancel); actions.addWidget(refresh); layout.addLayout(actions)
        history_actions = QHBoxLayout()
        details = QPushButton("Inspect job / logs…"); details.clicked.connect(self.inspect_selected_job); history_actions.addWidget(details)
        archive = QPushButton("Archive selected"); archive.clicked.connect(self.archive_selected_jobs); history_actions.addWidget(archive)
        delete = QPushButton("Delete selected…"); delete.clicked.connect(self.delete_selected_jobs); history_actions.addWidget(delete)
        clear = QPushButton("Clear terminal jobs…"); clear.clicked.connect(self.clear_terminal_jobs); history_actions.addWidget(clear)
        layout.addLayout(history_actions)
        return pane

    def guided_remote_training(self):
        if self.compute.mode not in ("Controller", "Hybrid"):
            QMessageBox.information(self, "Controller mode required", "Switch this Dataset Studio installation to Controller or Hybrid mode to submit remote jobs.")
            return
        try:
            studio = self._resolve_training_owner()
            manager = getattr(studio, "manager", None)
            if manager is None:
                raise ValueError("The main Dataset Studio window has no dataset manager.")
            info = getattr(manager, "info", None)
            if info is None:
                raise ValueError("Open a prepared training dataset first.")
            root_value = getattr(info, "root", None)
            if not root_value:
                raise ValueError("The prepared dataset has no available folder. Reopen or prepare the dataset, then try again.")
            dataset_root = Path(root_value).expanduser()
            if not dataset_root.is_dir():
                raise ValueError("The prepared dataset folder is unavailable. Reopen or prepare the dataset, then try again.")
            if not callable(getattr(studio, "_training_config", None)):
                raise ValueError("The main Dataset Studio window cannot provide the training configuration.")
            config = studio._training_config()
            if config is None:
                raise ValueError("Training configuration is unavailable. Complete the Train panel settings first.")
            dataset_yaml = Path(getattr(config, "dataset_yaml", "")).expanduser()
            if not dataset_yaml.is_file():
                raise ValueError("The prepared dataset YAML is unavailable. Prepare or reopen the dataset, then try again.")
            dataset_root = str(dataset_root)
        except Exception as exc:
            QMessageBox.warning(self, "Training setup is incomplete", str(exc)); return

        dialog = QDialog(self); dialog.setWindowTitle("Guided Remote Training"); dialog.setMinimumSize(540, 380); dialog.resize(740, 560)
        root = QVBoxLayout(dialog); steps = QStackedWidget(); scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(steps); root.addWidget(scroll, 1)
        def page(title, description):
            pane = QWidget(); layout = QVBoxLayout(pane); heading = QLabel(f"<h2>{title}</h2><p>{description}</p>")
            heading.setWordWrap(True); layout.addWidget(heading); return pane, layout

        pane, layout = page("1–3 · Dataset and training setup",
            "The currently open prepared dataset and settings in the Train panel will be used. This step identifies the images, labels, model family, and training options that the worker needs. If something is wrong, close this guide, correct it in Train, and start again.")
        layout.addWidget(QLabel(f"Dataset folder: {dataset_root}\nDataset YAML: {config.dataset_yaml}\nArchitecture: {config.architecture}\nModel: {config.model_version}\nEpochs: {config.epochs} · Image size: {config.image_size} · Device: {config.device}"))
        steps.addWidget(pane)

        pane, layout = page("4–5 · Choose a worker",
            "Automatic assignment chooses an online, approved worker that reports the required architecture, device, software, memory, and storage. You can pin this job to one worker. An unsuitable worker is skipped, and its reason is shown below.")
        self.remote_worker_choice = QComboBox(); self.remote_worker_choice.addItem("Automatic compatible worker", "")
        self.remote_ram_requirement = QSpinBox(); self.remote_ram_requirement.setRange(0, 1_048_576)
        self.remote_ram_requirement.setSuffix(" MB additional"); self.remote_ram_requirement.setToolTip("Extra RAM to reserve for this training job. The controller also keeps the configured RAM safety reserve free.")
        self.remote_gpu_requirement = QSpinBox(); self.remote_gpu_requirement.setRange(0, 1_048_576)
        self.remote_gpu_requirement.setSuffix(" MB additional"); self.remote_gpu_requirement.setToolTip("Extra VRAM to reserve. The controller also keeps the configured GPU safety margin free.")
        compatibility = []
        for worker in self.database.list_workers():
            if worker.get("status") == "online" and worker.get("approved") and worker.get("enabled"):
                suitable, reason = self._selected_worker_suitability(worker, config)
                self.remote_worker_choice.addItem(worker["name"], worker["id"])
                compatibility.append(f"{worker['name']}: {'Suitable' if suitable else 'Not suitable'} — {reason}")
        self.remote_compatibility = QLabel("\n".join(compatibility) if compatibility else "No approved worker is online. You may queue the job for an approved compatible worker to connect.")
        self.remote_compatibility.setWordWrap(True)
        self.remote_worker_choice.currentIndexChanged.connect(lambda _i: self._show_worker_reason(config))
        self.remote_ram_requirement.valueChanged.connect(lambda _v: self._show_worker_reason(config))
        self.remote_gpu_requirement.valueChanged.connect(lambda _v: self._show_worker_reason(config))
        layout.addWidget(QLabel("Worker selection:")); layout.addWidget(self.remote_worker_choice)
        layout.addWidget(QLabel("Additional RAM required by this job (set 0 if unknown):")); layout.addWidget(self.remote_ram_requirement)
        layout.addWidget(QLabel("Additional GPU memory required by this job (set 0 if unknown):")); layout.addWidget(self.remote_gpu_requirement)
        layout.addWidget(self.remote_compatibility)
        steps.addWidget(pane)

        pane, layout = page("6 · Dataset staging",
            "Dataset Studio will hash every dataset file, make a streaming archive, and send only missing data to the chosen worker. The worker verifies every file and stores it in its content-addressed cache. The source dataset stays in place. If transfer fails, the job reports the error and you can retry after checking the network and free disk space.")
        layout.addWidget(QLabel("The transfer begins after submission. Larger datasets take longer and require enough free space on both computers. Exact unchanged dataset versions can be reused from the worker cache."))
        steps.addWidget(pane)

        pane, layout = page("7–10 · Review, submit, monitor, retrieve",
            "Review the job before submitting. The controller keeps its experiment record. The worker runs the existing isolated YOLO or RT-DETR adapter, sends progress, and returns verified best.pt, last.pt, and available training logs. You can follow the job in Compute Center and its experiment in Experiments. A disconnected worker keeps its local files; the job is not automatically duplicated or retried.")
        self.remote_review = QLabel(); self.remote_review.setWordWrap(True); layout.addWidget(self.remote_review)
        self.remote_busy = QProgressBar(); self.remote_busy.setRange(0, 0); self.remote_busy.hide(); layout.addWidget(self.remote_busy)
        steps.addWidget(pane)

        controls = QHBoxLayout(); back = QPushButton("Back"); next_button = QPushButton("Next"); submit = QPushButton("Submit remote job"); close = QPushButton("Close")
        controls.addWidget(back); controls.addWidget(next_button); controls.addStretch(1); controls.addWidget(submit); controls.addWidget(close); root.addLayout(controls)
        submit.hide(); back.clicked.connect(lambda: steps.setCurrentIndex(max(0, steps.currentIndex()-1)))
        next_button.clicked.connect(lambda: steps.setCurrentIndex(min(steps.count()-1, steps.currentIndex()+1)))
        next_button.setDefault(True); submit.setDefault(True)
        def update_buttons(index):
            back.setEnabled(index > 0); next_button.setVisible(index < steps.count()-1); submit.setVisible(index == steps.count()-1)
            self.remote_review.setText(f"Experiment: {config.experiment_name}\nArchitecture: {config.architecture}\nDataset: {dataset_root}\nWorker: {self.remote_worker_choice.currentText()}\nDevice: {config.device}\nPriority: {self.priority.value()}")
        steps.currentChanged.connect(update_buttons); update_buttons(0)
        close.clicked.connect(dialog.reject)
        def submit_job():
            selected = self.remote_worker_choice.currentData() or None
            if selected:
                worker = self.database.get_worker(selected)
                suitable, reason = self._selected_worker_suitability(worker or {}, config)
                if not suitable:
                    QMessageBox.warning(dialog, "Worker is not suitable", reason); return
            submit.setEnabled(False); back.setEnabled(False); next_button.setEnabled(False); self.remote_busy.show()
            self._remote_submit_thread = _RemoteSubmitThread(self.compute, config, dataset_root, selected,
                self.priority.value(), self.remote_ram_requirement.value(), self.remote_gpu_requirement.value(), self)
            self._remote_submit_thread.completed.connect(lambda job_id: (self.refresh_jobs(), QMessageBox.information(dialog,
                "Remote job queued", f"Job {job_id[:12]} was saved to the controller queue. It will start when a compatible approved worker polls in."), dialog.accept()))
            self._remote_submit_thread.failed.connect(lambda error: (self.remote_busy.hide(), submit.setEnabled(True),
                QMessageBox.critical(dialog, "Remote job could not be submitted", error)))
            self._remote_submit_thread.start()
        submit.clicked.connect(submit_job)
        dialog.exec()

    def _resolve_training_owner(self):
        """Return the top-level Dataset Studio window that owns training state.

        Compute Center can be nested under tabs, stacked widgets, or other
        containers, so its immediate parent is not a reliable owner.
        """
        window = self.window()
        if window is self or not callable(getattr(window, "_training_config", None)):
            raise ValueError("Could not find the main Dataset Studio window that owns training settings.")
        if not hasattr(window, "manager"):
            raise ValueError("The main Dataset Studio window does not expose its dataset manager.")
        return window

    def _show_worker_reason(self, config):
        worker_id = self.remote_worker_choice.currentData()
        if not worker_id:
            self.remote_compatibility.setText("Automatic assignment checks each approved online worker and selects the first compatible worker when it connects.")
            return
        worker = self.database.get_worker(worker_id)
        suitable, reason = self._selected_worker_suitability(worker or {}, config)
        self.remote_compatibility.setText(f"{worker.get('name') if worker else worker_id}: {'Suitable' if suitable else 'Not suitable'} — {reason}")

    def _selected_worker_suitability(self, worker, config):
        requirements = {"ram_mb": self.remote_ram_requirement.value(),
            "gpu_memory_mb": self.remote_gpu_requirement.value(),
            "ram_safety_percent": self.compute.resource_profile.ram_safety_percent,
            "gpu_memory_safety_mb": self.compute.resource_profile.gpu_memory_safety_mb}
        record = {"architecture": config.architecture, "preferred_device": config.device,
                  "requirements": requirements}
        return remote_job_suitability(worker, record, active_jobs=self._worker_active_count(worker.get("id", "")))

    def _worker_active_count(self, worker_id):
        return sum(job["origin"] == "remote" and job["worker_id"] == worker_id and
            job["state"] in {"preparing", "running", "validating"} for job in self.database.list_compute_jobs())

    def clear_worker_dataset_cache(self):
        if self.compute.remote_agent and self.compute.remote_agent._active:
            QMessageBox.information(self, "Worker job is active", "Wait for the remote training job to stop before clearing its dataset cache.")
            return
        cache = DatasetCache(app_state_file("worker_dataset_cache")); rows = cache.cached_datasets()
        if not rows:
            QMessageBox.information(self, "Worker cache is empty", "There are no content-addressed dataset cache folders to clear.")
            return
        total = sum(row["bytes"] for row in rows) / (1024 ** 3)
        answer = QMessageBox.question(self, "Clear verified worker cache?",
            f"Remove {len(rows)} content-addressed dataset cache folder(s) ({total:.2f} GB) from this worker? This includes any cache folders that fail integrity checks. Original dataset folders, model files, training outputs, and checkpoints will not be changed.")
        if answer != QMessageBox.Yes: return
        removed = sum(cache.cleanup(row["dataset_id"]) for row in rows)
        QMessageBox.information(self, "Worker cache cleared", f"Removed {removed} cached dataset version(s).")

    def _settings_tab(self):
        pane = QWidget(); outer = QVBoxLayout(pane); scroll = QScrollArea(); scroll.setWidgetResizable(True)
        content = QWidget(); layout = QVBoxLayout(content); scroll.setWidget(content); outer.addWidget(scroll)
        group = QGroupBox("Conservative local resource profile"); form = QFormLayout(group)
        profile = self.compute.resource_profile
        self.cpu_jobs = QSpinBox(); self.cpu_jobs.setRange(1, 32); self.cpu_jobs.setValue(profile.max_concurrent_cpu_jobs)
        self.gpu_jobs = QSpinBox(); self.gpu_jobs.setRange(1, 16); self.gpu_jobs.setValue(profile.max_concurrent_gpu_jobs)
        self.cpu_threads = QSpinBox(); self.cpu_threads.setRange(1, 256); self.cpu_threads.setValue(profile.cpu_threads_per_job)
        self.ram_safety = QSpinBox(); self.ram_safety.setRange(0, 90); self.ram_safety.setSuffix("% of total RAM"); self.ram_safety.setValue(profile.ram_safety_percent)
        self.free_disk = QDoubleSpinBox(); self.free_disk.setRange(0, 100000); self.free_disk.setSuffix(" GB"); self.free_disk.setValue(profile.minimum_free_disk_gb)
        self.gpu_margin = QSpinBox(); self.gpu_margin.setRange(0, 1_000_000); self.gpu_margin.setSuffix(" MB"); self.gpu_margin.setValue(profile.gpu_memory_safety_mb)
        self.monitor_interval = QSpinBox(); self.monitor_interval.setRange(1, 60); self.monitor_interval.setSuffix(" sec"); self.monitor_interval.setValue(profile.monitoring_interval_seconds)
        form.addRow("Maximum concurrent CPU jobs:", self.cpu_jobs); form.addRow("Maximum concurrent GPU jobs:", self.gpu_jobs)
        form.addRow("CPU threads per job:", self.cpu_threads); form.addRow("RAM safety reserve:", self.ram_safety)
        form.addRow("Minimum free disk:", self.free_disk); form.addRow("GPU memory safety margin:", self.gpu_margin)
        form.addRow("Resource monitor interval:", self.monitor_interval)
        save = QPushButton("Save compute settings"); save.clicked.connect(self.save_settings); form.addRow(save)
        layout.addWidget(group)
        self.priority = QSpinBox(); self.priority.setRange(-100, 100); self.priority.setValue(0)
        layout.addWidget(QLabel("Priority for new Training Studio jobs (-100 to 100; higher priority is dispatched first).")); layout.addWidget(self.priority)
        layout.addWidget(QLabel("Local resource profiles control local jobs. Remote assignment checks each worker's reported RAM, disk, GPU, framework versions, supported architecture, and active load. Remote jobs use one active training slot per worker."))
        layout.addStretch(1); return pane

    def _browse_file(self, field, title):
        path, _ = QFileDialog.getOpenFileName(self, title, field.text(), "All files (*.*)")
        if path: field.setText(path)

    def save_settings(self):
        try:
            profile = ResourceProfile(self.cpu_jobs.value(), self.gpu_jobs.value(), self.cpu_threads.value(),
                self.ram_safety.value(), self.free_disk.value(), self.gpu_margin.value(), self.monitor_interval.value()).validate()
            settings = {"mode": self.mode.currentText(), "listen_host": self.listen_host.text().strip(),
                        "listen_port": self.listen_port.value(), "tls_cert": self.tls_cert.text().strip(),
                        "tls_key": self.tls_key.text().strip(), "tls_ca": self.tls_ca.text().strip()}
            self.compute.save_settings(settings, profile)
            self.monitor.interval_seconds = profile.monitoring_interval_seconds
            self.refresh_overview()
        except Exception as exc: QMessageBox.warning(self, "Compute settings not applied", str(exc))

    def apply_mode(self):
        self.save_settings()

    def start_controller(self):
        self.save_settings()
        try: self.compute.start_controller(); self.worker_status.setText("Controller TLS listener is starting.")
        except Exception as exc: QMessageBox.warning(self, "Could not start controller", str(exc))

    def stop_controller(self): self.compute.stop_controller(); self.worker_status.setText("Controller stopped.")

    def setup_controller_certificates(self, force_rotate=False, rotate_ca=False):
        if self.compute.controller and self.compute.controller.is_alive():
            QMessageBox.warning(self, "Stop controller first", "Stop the controller listener before creating or renewing its certificates.")
            return
        if rotate_ca and QMessageBox.warning(self, "Rotate the local CA?",
            "This replaces the trust authority. Every worker must import a new enrollment package and enroll again. Continue?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        try:
            manager = ControllerCertificateManager(app_state_file("controller_tls"))
            files = manager.generate_or_reuse(force_rotate=force_rotate, rotate_ca=rotate_ca)
            self.tls_ca.setText(str(files["ca_certificate"])); self.tls_cert.setText(str(files["certificate"]))
            self.tls_key.setText(str(files["private_key"]))
            values = {"mode": self.mode.currentText(), "listen_host": self.listen_host.text().strip(),
                "listen_port": self.listen_port.value(), "tls_cert": str(files["certificate"]),
                "tls_key": str(files["private_key"]), "tls_ca": str(files["ca_certificate"])}
            self.compute.save_settings(values)
            self.database.record_security_event("controller_certificate_rotated" if force_rotate or rotate_ca else
                "controller_certificate_reused" if files["reused"] else "controller_certificate_created", "controller")
            QMessageBox.information(self, "Controller certificate ready",
                f"Controller certificate and key were {'reused' if files['reused'] else 'created'}. Private keys are DPAPI-protected at rest.\n\n"
                f"Certificate: {files['certificate']}\nPublic CA: {files['ca_certificate']}\n"
                "Use the computer's reachable LAN name/IP in worker packages. The certificate SANs include detected hostnames and interface addresses.")
        except Exception as exc:
            QMessageBox.warning(self, "Controller certificate setup failed", str(exc))

    def _save_enrollment_package(self, token, expires):
        host, ok = QInputDialog.getText(self, "Controller address for worker", "Enter the controller hostname or LAN IP reachable by the worker:", text=socket.gethostname())
        if not ok or not host.strip(): return
        ca_path = Path(self.tls_ca.text().strip())
        if not ca_path.is_file():
            QMessageBox.warning(self, "Controller CA unavailable", "Set up the controller certificates before exporting a worker package.")
            return
        package = build_enrollment_package(host=host, port=self.listen_port.value(),
            ca_certificate=ca_path.read_text(encoding="ascii"), token=token, expires_at=expires)
        path, _ = QFileDialog.getSaveFileName(self, "Save worker enrollment package", "dataset-studio-enrollment.json", "Enrollment package (*.json)")
        if not path: return
        try:
            save_enrollment_package(path, package)
            QMessageBox.information(self, "Enrollment package saved", "The package contains the public CA certificate and a one-use token. Transfer it only through a trusted private channel; it contains no private keys or worker credentials.")
        except Exception as exc: QMessageBox.warning(self, "Could not save enrollment package", str(exc))

    def export_worker_package(self):
        self.save_settings()
        try:
            token, expires = self.compute.create_enrollment_token()
            dialog = _EnrollmentTokenDialog(self.compute, token, expires, self._save_enrollment_package, self)
            dialog.exec()
            dialog.token = ""; dialog.token_edit.clear()
        except Exception as exc: QMessageBox.warning(self, "Could not create enrollment package", str(exc))

    def import_worker_package(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import worker enrollment package", "", "Enrollment package (*.json)")
        if not path: return
        try:
            package = validate_enrollment_package(json.loads(Path(path).read_text(encoding="utf-8")))
            ca_path = app_state_file("controller_trust_ca.pem")
            temp = ca_path.with_suffix(".tmp"); temp.write_text(package["ca_certificate_pem"], encoding="ascii"); temp.replace(ca_path)
            self.controller_host.setText(package["controller_host"]); self.controller_port.setValue(package["controller_port"])
            self.ca_file.setText(str(ca_path)); self.enrollment.setText(package["token"])
            self._enrollment_package_token_hash = hashlib.sha256(package["token"].encode("utf-8")).hexdigest()
            self._enrollment_package_expiry = package["expires_at"]
            self._identity.update(controller_host=package["controller_host"], controller_port=package["controller_port"], ca_file=str(ca_path))
            self.worker_status.setText(f"Package imported; token expires at {package['expires_at']}. Review settings, then connect.")
        except Exception as exc:
            QMessageBox.warning(self, "Enrollment package rejected", str(exc))

    def create_token(self):
        self.save_settings()
        try:
            token, expires = self.compute.create_enrollment_token()
            dialog = _EnrollmentTokenDialog(self.compute, token, expires, self._save_enrollment_package, self)
            dialog.exec(); dialog.token = ""; dialog.token_edit.clear()
        except Exception as exc: QMessageBox.warning(self, "Could not create enrollment token", str(exc))

    def connect_worker(self):
        if not self.controller_host.text().strip(): QMessageBox.warning(self, "Controller required", "Enter the controller host."); return
        self._save_identity()
        token = self.enrollment.text().strip() or None
        expiry = getattr(self, "_enrollment_package_expiry", None) if token and hashlib.sha256(token.encode("utf-8")).hexdigest() == getattr(self, "_enrollment_package_token_hash", None) else None
        try:
            self.compute.start_worker(worker_id=self.worker_id.text(), name=self.worker_name.text().strip(),
                host=self.controller_host.text().strip(), port=self.controller_port.value(), ca_file=self.ca_file.text().strip(),
                enrollment_token=token, enrollment_expires_at=expiry)
            self.enrollment.clear(); self.worker_status.setText("Connecting over validated TLS…")
            self._enrollment_package_expiry = None; self._enrollment_package_token_hash = None
        except Exception as exc: QMessageBox.warning(self, "Could not start worker connection", str(exc))

    def disconnect_worker(self): self.compute.stop_worker(); self.worker_status.setText("Worker disconnected.")

    def _update_hardware(self, hardware):
        if self._closing: return
        self._hardware = hardware
        cpu = hardware.get("cpu", {}); memory = hardware.get("memory", {}); disk = hardware.get("disk", {})
        gpus = hardware.get("gpus", [])
        self.hardware_summary.setText(
            f"{hardware.get('platform')} · Python {hardware.get('python')}\n"
            f"CPU: {cpu.get('name') or 'Unknown'} · {cpu.get('physical_cores') or '?'} physical / {cpu.get('logical_cores') or '?'} logical cores · {cpu.get('utilization_percent') or 0:.0f}%\n"
            f"RAM: {memory.get('available_mb') if memory.get('available_mb') is not None else '?'} MB available / {memory.get('total_mb') if memory.get('total_mb') is not None else '?'} MB · Disk free: {disk.get('free_gb') if disk.get('free_gb') is not None else '?'} GB\n"
            f"PyTorch {hardware.get('torch_version') or 'unavailable'} · CUDA {hardware.get('torch_cuda_version') or 'unavailable'} · CUDA usable: {hardware.get('torch_cuda_available')} · Architectures: {', '.join(hardware.get('supported_training_architectures', []))}")
        self.gpu_table.setRowCount(len(gpus))
        for row, gpu in enumerate(gpus):
            values = [gpu.get("name", "GPU"), gpu.get("vendor", "Unknown"),
                      f"{gpu.get('memory_used_mb') if gpu.get('memory_used_mb') is not None else '?'} / {gpu.get('memory_mb') or '?'}",
                      gpu.get("memory_free_mb") if gpu.get("memory_free_mb") is not None else "Unknown",
                      gpu.get("utilization_percent") if gpu.get("utilization_percent") is not None else "Unknown",
                      "Yes" if gpu.get("torch_cuda") else "No"]
            for col, value in enumerate(values): self.gpu_table.setItem(row, col, QTableWidgetItem(str(value)))
        self.gpu_table.resizeColumnsToContents(); self.refresh_overview()

    def refresh_overview(self):
        jobs = self.database.list_compute_jobs(); active = sum(j["state"] in ("preparing", "running", "validating") for j in jobs)
        queued = sum(j["state"] == "queued" for j in jobs)
        workers = self.database.list_workers(); online = sum(w["status"] == "online" for w in workers)
        mem = self._hardware.get("memory", {}); cpu = self._hardware.get("cpu", {})
        local_active = sum(j["origin"] == "local" and j["state"] in ("preparing", "running", "validating") for j in jobs)
        remote_active = sum(j["origin"] == "remote" and j["state"] in ("preparing", "running", "validating") for j in jobs)
        approved_online = [w for w in workers if w["status"] == "online" and w["approved"] and w["enabled"]]
        listener_running = bool(self.compute.controller and self.compute.controller.is_alive())
        remote_mode = self.compute.mode in ("Controller", "Hybrid")
        capability = "Available (YOLO, RT-DETR)" if remote_mode else "Controller mode required"
        scheduling = f"{len(approved_online)} approved worker(s) connected; eligibility is checked per job"
        self.overview.setText(f"Mode: {self.compute.mode} · Local active: {local_active} · Remote active: {remote_active} · Queued: {queued} · Workers online: {online}/{len(workers)} · CPU {cpu.get('utilization_percent') or 0:.0f}% · Local RAM available: {mem.get('available_mb') if mem.get('available_mb') is not None else 'Unknown'} MB")
        self.resource_summary.setText(f"Controller listener: {'Running' if listener_running else 'Stopped'}\n"
            f"Worker authorization: {len(approved_online)} online, approved and enabled / {len(workers)} registered\n"
            f"Remote execution capability: {capability}\nScheduling eligibility: {scheduling}\n"
            f"Local reservations: {len(self.database.compute_reservations())} · CPU slots {self.compute.resource_profile.max_concurrent_cpu_jobs} · GPU slots {self.compute.resource_profile.max_concurrent_gpu_jobs} · RAM safety reserve {self.compute.resource_profile.ram_safety_percent}% · GPU safety margin {self.compute.resource_profile.gpu_memory_safety_mb} MB")

    def refresh_workers(self):
        self.database.expire_stale_workers()
        workers = self.database.list_workers(); self.worker_table.setRowCount(len(workers))
        for row, worker in enumerate(workers):
            cpu = worker.get("hardware", {}).get("cpu", {}) if isinstance(worker.get("hardware"), dict) else {}
            mem = worker.get("hardware", {}).get("memory", {}) if isinstance(worker.get("hardware"), dict) else {}
            hardware = worker.get("hardware", {}) if isinstance(worker.get("hardware"), dict) else {}
            disk = hardware.get("disk", {})
            gpu_names = ", ".join(f"{g.get('name', 'GPU')} ({g.get('memory_free_mb') if g.get('memory_free_mb') is not None else '?'} MB free)" for g in hardware.get("gpus", []))
            hardware = worker.get("hardware", {}) if isinstance(worker.get("hardware"), dict) else {}
            software = worker.get("software", {}) if isinstance(worker.get("software"), dict) else {}
            versions = f"{software.get('torch', '?')} / {software.get('ultralytics', '?')}"
            active = sum(job["origin"] == "remote" and job["worker_id"] == worker["id"] and
                job["state"] in {"preparing", "running", "validating"} for job in self.database.list_compute_jobs())
            values = [worker["name"], worker["id"], worker["status"], "Approved" if worker["approved"] else "Pending",
                      "Enabled" if worker["enabled"] else "Disabled",
                      f"{cpu.get('logical_cores') or '?'} threads · {mem.get('available_mb') or '?'} / {mem.get('total_mb') or '?'} MB RAM · {disk.get('free_gb') if disk.get('free_gb') is not None else '?'} GB free",
                      gpu_names or "CPU only", ", ".join(hardware.get("supported_training_architectures", [])) or "Unknown",
                      versions, active, worker["last_seen"] or "Never"]
            for col, value in enumerate(values):
                item = QTableWidgetItem(str(value)); item.setData(Qt.UserRole, worker["id"]); self.worker_table.setItem(row, col, item)
        self.worker_table.resizeColumnsToContents(); self.refresh_overview()

    def _selected_worker(self):
        item = self.worker_table.item(self.worker_table.currentRow(), 0) if self.worker_table.currentRow() >= 0 else None
        return self.database.get_worker(item.data(Qt.UserRole)) if item else None

    def approve_worker(self):
        worker = self._selected_worker()
        if not worker: return
        try: self.database.set_worker_approved(worker["id"], True); self.refresh_workers()
        except Exception as exc: QMessageBox.warning(self, "Worker approval failed", str(exc))

    def toggle_worker(self):
        worker = self._selected_worker()
        if not worker: return
        try: self.database.set_worker_enabled(worker["id"], not bool(worker["enabled"])); self.refresh_workers()
        except Exception as exc: QMessageBox.warning(self, "Worker update failed", str(exc))

    def revoke_worker(self):
        worker = self._selected_worker()
        if not worker: return
        if QMessageBox.question(self, "Revoke worker credentials?", f"Revoke {worker['name']}'s authentication credential? The worker will be disconnected and must be enrolled again.") != QMessageBox.Yes: return
        try: self.database.revoke_worker(worker["id"]); self.refresh_workers()
        except Exception as exc: QMessageBox.warning(self, "Worker revocation failed", str(exc))

    def remove_revoked_worker(self):
        worker = self._selected_worker()
        if not worker: return
        if worker.get("revoked_at") is None:
            QMessageBox.information(self, "Revoke worker first", "Use ‘Revoke credentials…’ before removing a worker registration.")
            return
        if QMessageBox.question(self, "Remove revoked worker?",
                f"Remove {worker['name']} from Worker Manager? Its job, experiment, and security-event history will be retained.") != QMessageBox.Yes:
            return
        try:
            self.database.remove_revoked_worker(worker["id"])
            self.refresh_workers()
        except Exception as exc:
            QMessageBox.warning(self, "Worker removal failed", str(exc))

    def view_security_events(self):
        events = self.database.list_security_events(500)
        dialog = QDialog(self); dialog.setWindowTitle("Compute Center security events"); dialog.setMinimumSize(600, 380); dialog.resize(900, 560)
        layout = QVBoxLayout(dialog); table = QTableWidget(len(events), 5)
        table.setHorizontalHeaderLabels(["Time", "Event", "Subject", "Subject ID", "Details"])
        table.setEditTriggers(QTableWidget.NoEditTriggers); table.setWordWrap(False)
        table.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
        for row, event in enumerate(events):
            values = [event["created_at"], event["category"], event["subject_type"], event["subject_id"], json.dumps(event["details"], ensure_ascii=False)]
            for col, value in enumerate(values): table.setItem(row, col, QTableWidgetItem(str(value)))
        layout.addWidget(table, 1); close = QPushButton("Close"); close.clicked.connect(dialog.accept); layout.addWidget(close, alignment=Qt.AlignRight)
        dialog.exec()

    def refresh_jobs(self):
        archived_view = bool(self.job_history_filter.currentData())
        jobs = [job for job in self.database.list_compute_jobs() if bool(job.get("archived")) == archived_view]
        self.jobs_table.setRowCount(len(jobs))
        for row, job in enumerate(jobs):
            updates = self.database.remote_job_updates(job["id"]) if job["origin"] == "remote" else []
            progress = updates[-1]["progress"] if updates else {}
            progress_text = (f"Epoch {progress['epoch']}/{progress.get('total_epochs', '?')}" if "epoch" in progress else
                f"Dataset transfer {progress['transfer_bytes'] / max(1, progress.get('transfer_total_bytes', 1)):.0%}" if "transfer_bytes" in progress else
                f"Model transfer {progress['model_transfer_bytes'] / max(1, progress.get('model_transfer_total', 1)):.0%}" if "model_transfer_bytes" in progress else
                progress.get("message", "Waiting for compatible worker · see worker compatibility details") if job["origin"] == "remote" else "—")
            worker_diagnostics = []
            if job["origin"] == "remote" and job["state"] == "queued":
                requested_worker = job.get("requirements", {}).get("requested_worker_id")
                for worker in self.database.list_workers():
                    if requested_worker and worker["id"] != requested_worker:
                        suitable, reason = False, f"Job is pinned to worker {requested_worker}."
                    else:
                        suitable, reason = remote_job_suitability(worker, job,
                            active_jobs=self._worker_active_count(worker["id"]))
                    worker_diagnostics.append((worker, suitable, reason))
                compatible = next((worker for worker, suitable, _reason in worker_diagnostics if suitable), None)
                if compatible:
                    progress_text = f"Waiting for assignment · compatible: {compatible['name']}"
                elif worker_diagnostics:
                    diagnostic = next((entry for entry in worker_diagnostics if entry[0]["id"] == requested_worker), worker_diagnostics[0])
                    progress_text = f"Waiting · {diagnostic[2]}"
                else:
                    progress_text = "Waiting · no worker is registered."
            experiment = self.database.get_experiment(job["experiment_id"]) if job["experiment_id"] else None
            checkpoints = experiment or {}
            checkpoint = checkpoints.get("best_checkpoint") or checkpoints.get("last_checkpoint") or "—"
            values = [job["id"][:8], job["experiment_id"] or "—", job["architecture"], job["preferred_device"],
                      job["state"], job["priority"], job["worker_name"] or ("Waiting for worker" if job["origin"] == "remote" else "Local"),
                      progress_text, checkpoint, job["created_at"], "Yes" if job.get("archived") else "No"]
            for col, value in enumerate(values):
                item = QTableWidgetItem(str(value)); item.setData(Qt.UserRole, job["id"]); self.jobs_table.setItem(row, col, item)
                if col == 7 and worker_diagnostics:
                    item.setToolTip("\n".join(f"{worker['name']}: {'Suitable' if suitable else reason}" for worker, suitable, reason in worker_diagnostics))
        self.jobs_table.resizeColumnsToContents(); self.jobs_summary.setText(f"{len(jobs)} job record(s) · {len(self.database.compute_reservations())} active resource reservation(s)")
        self.refresh_overview()

    def _selected_job_ids(self):
        rows = sorted({index.row() for index in self.jobs_table.selectionModel().selectedRows()})
        return [self.jobs_table.item(row, 0).data(Qt.UserRole) for row in rows if self.jobs_table.item(row, 0)]

    def inspect_selected_job(self):
        job_ids = self._selected_job_ids()
        if not job_ids: return
        job = self.database.get_compute_job(job_ids[0])
        if not job: return
        updates = self.database.remote_job_updates(job["id"]) if job["origin"] == "remote" else []
        lines = [f"Job: {job['id']}", f"Experiment: {job.get('experiment_id') or '—'}", f"Architecture: {job['architecture']}",
            f"Device: {job['preferred_device']}", f"State: {job['state']}", f"Worker: {job.get('worker_name') or '—'}",
            f"Created: {job['created_at']}", f"Started: {job.get('started_at') or '—'}", f"Ended: {job.get('ended_at') or '—'}",
            f"Failure / status detail: {job.get('error') or 'No failure recorded.'}"]
        if updates:
            lines.append("\nRecent remote updates:")
            lines.extend(f"{row['created_at']} · {row['state']} · {json.dumps(row['progress'], ensure_ascii=False)}" for row in updates[-20:])
        QMessageBox.information(self, "Job details and logs", "\n".join(lines))

    def archive_selected_jobs(self):
        job_ids = self._selected_job_ids()
        if not job_ids: return
        if QMessageBox.question(self, "Archive selected jobs?", f"Archive {len(job_ids)} terminal job record(s)? Experiment history and all files will be preserved.") != QMessageBox.Yes: return
        try:
            count = self.database.archive_compute_jobs(job_ids)
            self.refresh_jobs(); self.jobs_summary.setText(f"Archived {count} job record(s). Experiments and files were preserved.")
        except Exception as exc: QMessageBox.warning(self, "Could not archive selected jobs", str(exc))

    def delete_selected_jobs(self):
        job_ids = self._selected_job_ids()
        if not job_ids: return
        details = [self.database.get_compute_job(job_id) for job_id in job_ids]
        failure = "\n".join(f"{job['id'][:8]}: {job['error']}" for job in details if job and job.get("error"))
        message = f"Delete {len(job_ids)} terminal compute-job record(s)? Experiment records, checkpoints, logs, datasets, and model files will be preserved."
        if failure: message += "\n\nFailure details:\n" + failure
        if QMessageBox.question(self, "Delete compute-job history?", message) != QMessageBox.Yes: return
        try:
            count = self.database.delete_compute_jobs(job_ids)
            self.refresh_jobs(); self.jobs_summary.setText(f"Deleted {count} compute-job record(s). Experiment history and files were preserved.")
        except Exception as exc: QMessageBox.warning(self, "Could not delete selected jobs", str(exc))

    def clear_terminal_jobs(self):
        archived_view = bool(self.job_history_filter.currentData())
        candidates = [job["id"] for job in self.database.list_compute_jobs()
            if job["state"] in {"completed", "failed", "cancelled"} and bool(job.get("archived")) == archived_view]
        if not candidates: return
        if QMessageBox.question(self, "Clear terminal job history?",
            f"Delete {len(candidates)} terminal job record(s) from {'archived' if archived_view else 'current'} history? Experiment records and all files will be preserved.") != QMessageBox.Yes: return
        try:
            count = self.database.delete_compute_jobs(candidates)
            self.refresh_jobs(); self.jobs_summary.setText(f"Deleted {count} terminal job record(s); experiment records and files were preserved.")
        except Exception as exc: QMessageBox.warning(self, "Could not clear terminal jobs", str(exc))

    def cancel_job(self):
        item = self.jobs_table.item(self.jobs_table.currentRow(), 0) if self.jobs_table.currentRow() >= 0 else None
        if not item: return
        try:
            job_id = item.data(Qt.UserRole); job = self.database.get_compute_job(job_id)
            if job and job["origin"] == "remote":
                result = self.compute.cancel_remote_job(job_id)
                self.jobs_summary.setText("Remote job cancelled." if result == "cancelled" else
                    "Cancellation requested. The worker will stop at its next safe checkpoint." if result == "requested" else f"Remote job is already {result}.")
            else:
                self.compute.scheduler.cancel(job_id)
            self.refresh_jobs()
        except Exception as exc: QMessageBox.warning(self, "Could not stop compute job", str(exc))

    def _refresh_stale_workers(self):
        self.database.expire_stale_workers()
        self.refresh_workers(); self.refresh_jobs()

    def _worker_connection_status(self, status):
        execution = "remote worker agent ready" if self.compute.remote_agent else "no local remote worker agent"
        self.worker_status.setText(f"TLS connection established · authorization: {status} · capability: {execution}")
    def _network_error(self, error): self.worker_status.setText(f"Network / monitor status: {error}")
    def _schedule_message(self, message): self.jobs_summary.setText(message)

    def stop(self):
        self._closing = True
        self.timer.stop()
        try:
            self.monitor.snapshot.disconnect(self._update_hardware)
            self.monitor.failed.disconnect(self._network_error)
        except RuntimeError:
            pass
        self.monitor.stop(); self.monitor.wait(6000)
