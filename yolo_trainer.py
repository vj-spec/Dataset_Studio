from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from PySide6.QtCore import QThread, Signal

class TrainingWorker(QThread):
    message = Signal(str)
    progress = Signal(str)
    epoch_progress = Signal(int, int, str)
    finished_ok = Signal(str)
    failed = Signal(str)

    def __init__(self, model_name, data_yaml, device, epochs, imgsz, batch, project_dir):
        super().__init__()
        self.model_name = model_name
        self.data_yaml = str(data_yaml)
        self.device = device
        self.epochs = int(epochs)
        self.imgsz = int(imgsz)
        self.batch = int(batch)
        self.project_dir = str(project_dir)

    def run(self):
        try:
            from ultralytics import YOLO

            self.message.emit(f"Loading model: {self.model_name}")
            model = YOLO(self.model_name)

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
