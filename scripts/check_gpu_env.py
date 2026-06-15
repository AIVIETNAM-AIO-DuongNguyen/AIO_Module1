"""Verify that a GPU environment is usable for training.

Run this on the GPU server after installing requirements:

    python scripts/check_gpu_env.py

It reports the PyTorch and CUDA versions, lists visible GPUs, and runs a small
matrix multiplication on the first GPU to confirm compute actually works.
Exit code 0 means a usable CUDA device was found and exercised; exit code 1
means no usable GPU is available.
"""

import sys


def main() -> int:
    try:
        import torch
    except ImportError:
        print("PyTorch is not installed. Activate the virtual environment and "
              "run: pip install -r requirements.txt")
        return 1

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("No CUDA device detected. This environment is CPU-only.")
        return 1

    print(f"CUDA runtime version: {torch.version.cuda}")
    device_count = torch.cuda.device_count()
    print(f"Visible GPUs: {device_count}")
    for index in range(device_count):
        print(f"  [{index}] {torch.cuda.get_device_name(index)}")

    device = torch.device("cuda:0")
    left = torch.randn(512, 512, device=device)
    right = torch.randn(512, 512, device=device)
    result = left @ right
    torch.cuda.synchronize()
    print(f"Matrix multiply on {device} succeeded; result shape {tuple(result.shape)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
