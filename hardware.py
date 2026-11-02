from __future__ import annotations

import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

@dataclass
class GPUInfo:
    index: int
    name: str
    memory_mb: Optional[int]
    driver: str = ""
    torch_cuda: bool = False
    vendor: str = "NVIDIA"
    memory_used_mb: Optional[int] = None
    memory_free_mb: Optional[int] = None
    utilization_percent: Optional[float] = None

def _run_nvidia_smi():
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,driver_version",
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
                def number(index, cast=int):
                    try: return cast(float(parts[index]))
                    except (ValueError, IndexError): return None
                mem = number(2)
                rows.append(GPUInfo(
                    index=idx,
                    name=parts[1],
                    memory_mb=mem,
                    driver=parts[6] if len(parts) >= 7 else "",
                    vendor="NVIDIA", memory_used_mb=number(3), memory_free_mb=number(4),
                    utilization_percent=number(5, float),
                ))
        return rows
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return []

def detect_hardware():
    gpus = _run_nvidia_smi()
    torch_version = None
    torch_cuda_version = None
    torch_cuda_available = False
    cpu_info = {"name": platform.processor() or platform.machine(), "physical_cores": None,
                "logical_cores": None, "utilization_percent": None}
    memory = {"total_mb": None, "available_mb": None, "used_mb": None}
    disk = {"path": str(Path.home()), "free_gb": None, "total_gb": None}
    try:
        import psutil
        cpu_info.update(physical_cores=psutil.cpu_count(logical=False), logical_cores=psutil.cpu_count(logical=True),
                        utilization_percent=psutil.cpu_percent(interval=None))
        vm = psutil.virtual_memory()
        memory.update(total_mb=int(vm.total / (1024 * 1024)), available_mb=int(vm.available / (1024 * 1024)),
                      used_mb=int(vm.used / (1024 * 1024)))
        usage = psutil.disk_usage(str(Path.home().anchor or Path.home()))
        disk.update(free_gb=round(usage.free / (1024 ** 3), 2), total_gb=round(usage.total / (1024 ** 3), 2))
    except Exception:
        pass

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
                        vendor="NVIDIA" if "nvidia" in str(props.name).lower() else "Unknown",
                    ))
    except Exception:
        pass

    return {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu": cpu_info,
        "memory": memory,
        "disk": disk,
        "gpus": gpus,
        "nvidia_gpu_detected": bool(gpus),
        "torch_version": torch_version,
        "torch_cuda_version": torch_cuda_version,
        "torch_cuda_available": torch_cuda_available,
        "torch_cuda_compatible": bool(torch_cuda_available),
        "training_framework_versions": _framework_versions(),
        "supported_training_architectures": ["YOLO", "RT-DETR"],
    }


def _framework_versions():
    versions = {}
    try:
        from importlib.metadata import version, PackageNotFoundError
        for package in ("ultralytics", "torch"):
            try: versions[package] = version(package)
            except PackageNotFoundError: pass
    except Exception:
        pass
    return versions

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
