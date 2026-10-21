"""Hardware and execution utilities optimized for NVIDIA DGX (CUDA), Apple Silicon (MPS), and PySpark."""
import os
from pathlib import Path
import platform
import sys
from typing import Dict, Optional, Tuple

import torch


def get_platform_info() -> Dict[str, str]:
    """Return diagnostic information about the host system."""
    return {
        "os": platform.system(),
        "os_release": platform.release(),
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "cuda_available": str(torch.cuda.is_available()),
        "cuda_device_count": str(torch.cuda.device_count()) if torch.cuda.is_available() else "0",
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else ("Apple Silicon" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) else "CPU"),
        "bf16_supported": str(torch.cuda.is_bf16_supported()) if torch.cuda.is_available() else "False",
    }


def resolve_device(requested: str = "auto") -> torch.device:
    """Resolve the optimal compute device (CUDA on DGX, MPS on Mac, or CPU fallback)."""
    req = requested.lower().strip()
    if req not in ("auto", "cuda", "mps", "cpu"):
        # Handle specific cuda index like 'cuda:0', 'cuda:1'
        if req.startswith("cuda:"):
            return torch.device(req)
        raise ValueError(f"Unsupported device: {requested}. Choose 'auto', 'cuda', 'mps', or 'cpu'.")

    if req == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    
    if req == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available on this system.")
        return torch.device("cuda")
        
    if req == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("MPS requested but Apple Silicon MPS is not available.")
        return torch.device("mps")
        
    return torch.device("cpu")


def resolve_dtype(device: torch.device, requested: Optional[str] = None) -> torch.dtype:
    """Resolve optimal tensor precision (bfloat16 for DGX Ampere/Hopper, float16, or float32)."""
    if requested:
        req = requested.lower().strip()
        if req in ("bf16", "bfloat16"):
            return torch.bfloat16
        if req in ("fp16", "float16"):
            return torch.float16
        if req in ("fp32", "float32"):
            return torch.float32
        raise ValueError(f"Unknown dtype request: {requested}")

    if device.type == "cuda":
        # NVIDIA DGX (A100 / H100 / RTX 6000 Ada) natively excels at bfloat16
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    elif device.type == "mps":
        return torch.float16
    return torch.float32


def optimize_cuda_env():
    """Apply DGX-specific CUDA environment optimizations."""
    if torch.cuda.is_available():
        # Enable Tensor Core TF32 matrix multiplications on Ampere+ architectures
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        
        # Optimize memory allocation caching to avoid fragmentation
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def clear_gpu_memory(device: Optional[torch.device] = None):
    """Release cached GPU memory across PyTorch allocators."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        if hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()


def safe_cross_platform_path(path: Path) -> str:
    """Ensure paths are cleanly normalized across Windows and POSIX systems."""
    p = Path(path).resolve()
    return str(p)
