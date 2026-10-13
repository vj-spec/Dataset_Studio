from __future__ import annotations

import platform
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional

@dataclass
class GPUInfo:
    index: int
    name: str
    memory_mb: Optional[int]
    driver: str = ""
    torch_cuda: bool = False

def _run_nvidia_smi():
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode != 0:
            return []
        rows = []
        for line in result.stdout.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                try:
                    idx = int(parts[0])
                except ValueError:
                    continue
                mem = None
                if len(parts) >= 3:
                    try:
                        mem = int(float(parts[2]))
                    except ValueError:
                        pass
                rows.append(GPUInfo(
                    index=idx,
                    name=parts[1],
                    memory_mb=mem,
                    driver=parts[3] if len(parts) >= 4 else "",
                ))
        return rows
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return []

def detect_hardware():
    gpus = _run_nvidia_smi()
    torch_version = None
    torch_cuda_version = None
    torch_cuda_available = False

    try:
        import torch
        torch_version = torch.__version__
        torch_cuda_version = getattr(torch.version, "cuda", None)
        torch_cuda_available = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            for gpu in gpus:
                try:
                    gpu.torch_cuda = True
                except Exception:
                    pass
            # nvidia-smi can be unavailable in unusual environments.
            if not gpus:
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    gpus.append(GPUInfo(
                        index=i,
                        name=props.name,
                        memory_mb=int(props.total_memory / (1024 * 1024)),
                        torch_cuda=True,
                    ))
    except Exception:
        pass

    return {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "gpus": gpus,
        "nvidia_gpu_detected": bool(gpus),
        "torch_version": torch_version,
        "torch_cuda_version": torch_cuda_version,
        "torch_cuda_available": torch_cuda_available,
    }

def cuda_install_command():
    # We deliberately do not guess a CUDA Toolkit version. PyTorch wheels
    # bundle the CUDA runtime; the NVIDIA driver is installed separately.
    return (
        f'"{sys.executable}" -m pip install --upgrade '
        '"torch" "torchvision" --index-url https://download.pytorch.org/whl/cu132'
    )

def cpu_install_command():
    return (
        f'"{sys.executable}" -m pip install --upgrade '
        '"torch" "torchvision" --index-url https://download.pytorch.org/whl/cpu'
    )
