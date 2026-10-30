from __future__ import annotations

import subprocess
import sys
import threading
from pathlib import Path

from PySide6.QtCore import QThread, Signal

class TrainingWorker(QThread):
    message = Signal(str)
    progress = Signal(str)
    epoch_progress = Signal(int, int, str)
    finished_ok = Signal(str)
    failed = Signal(str)

    def __init__(self, model_name, data_yaml, device, epochs, imgsz, batch, project_dir, resume=False):
        super().__init__()
        self.model_name = model_name
        self.data_yaml = str(data_yaml)
        self.device = device
        self.epochs = int(epochs)
        self.imgsz = int(imgsz)
        self.batch = int(batch)
        self.project_dir = str(project_dir)
        self.resume = bool(resume)
        self._stop_requested = threading.Event()
        self._trainer = None

    def request_stop(self):
        """Ask Ultralytics to stop at its next safe epoch boundary."""
        self._stop_requested.set()
        if self._trainer is not None:
            self._trainer.stop = True

    def run(self):
        try:
            from ultralytics import YOLO

            self.message.emit(f"Loading model: {self.model_name}")
            model = YOLO(self.model_name)

            def report_batch(trainer):
                batch = int(getattr(trainer, "batch_i", -1)) + 1
                batches = int(getattr(trainer, "nb", 0) or 0)
                if batch == 1 or batch % 5 == 0 or (batches and batch == batches):
                    epoch = int(getattr(trainer, "epoch", 0)) + 1
                    text = f"Epoch {epoch}/{self.epochs} — batch {batch}/{batches or '?'}"
                    losses = getattr(trainer, "tloss", None)
                    if losses is not None:
                        try:
                            values = losses.detach().cpu().tolist()
                            text += " | loss " + ", ".join(f"{float(v):.4f}" for v in values)
                        except Exception:
                            pass
                    self.progress.emit(text)

            def on_pretrain(trainer):
                self.message.emit("Preparing dataset and training environment…")

            def on_train_start(trainer):
                self._trainer = trainer
                if self._stop_requested.is_set():
                    trainer.stop = True

            def on_train_epoch_end(trainer):
                if self._stop_requested.is_set():
                    trainer.stop = True

            model.add_callback("on_train_batch_end", report_batch)
            model.add_callback("on_pretrain_routine_start", on_pretrain)
            model.add_callback("on_train_start", on_train_start)
            model.add_callback("on_train_epoch_end", on_train_epoch_end)

            def report_epoch(trainer):
                row = {str(k): float(v) for k, v in getattr(trainer, "metrics", {}).items()
                       if isinstance(v, (int, float))}
                epoch = trainer.epoch + 1
                detail = ", ".join(f"{k}={v:.4f}" for k, v in row.items())
                self.progress.emit(f"{epoch}/{trainer.epochs} | {detail}")
                self.epoch_progress.emit(epoch, trainer.epochs, detail)

            model.add_callback("on_fit_epoch_end", report_epoch)

            self.message.emit(
                f"Training on {self.device} | epochs={self.epochs} | imgsz={self.imgsz} | batch={self.batch}"
            )
            if self.resume:
                self.message.emit("Resuming training from the last saved checkpoint…")
                results = model.train(resume=True)
            else:
                results = model.train(
                    data=self.data_yaml,
                    epochs=self.epochs,
                    imgsz=self.imgsz,
                    batch=self.batch,
                    device=self.device,
                    project=self.project_dir,
                    name="train",
                    exist_ok=True,
                )

            save_dir = getattr(results, "save_dir", None)
            if save_dir:
                self.finished_ok.emit(str(save_dir))
            else:
                self.finished_ok.emit("Training finished.")
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")

def install_cuda_pytorch(callback=None):
    """
    Installs a CUDA-enabled PyTorch wheel into the CURRENT Python environment.
    This does not install the NVIDIA driver or system CUDA Toolkit.
    """
    cmd = [
        sys.executable, "-m", "pip", "install", "--upgrade",
        "torch==2.13.0", "torchvision==0.28.0",
        "--index-url", "https://download.pytorch.org/whl/cu132",
    ]
    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
