"""Read-only environment report. Works on Windows and Linux.

Usage: python scripts/check_env.py
"""
from __future__ import annotations

import importlib.metadata as md
import platform
import shutil
import subprocess
import sys
from pathlib import Path

PACKAGES = ("torch", "torchvision", "ultralytics", "transformers", "opencv-python-headless", "numpy", "shapely", "pyyaml")


def main() -> None:
    print(f"python   : {sys.version.split()[0]} ({platform.platform()})")
    for name in PACKAGES:
        try:
            print(f"{name:<9}: {md.version(name)}")
        except md.PackageNotFoundError:
            print(f"{name:<9}: NOT INSTALLED")
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            print(f"gpu      : {props.name}, {props.total_memory / 2**30:.1f} GiB")
        else:
            print("gpu      : torch installed, CUDA not available")
    except ImportError:
        pass
    if shutil.which("nvidia-smi"):
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
        print(out.stdout.strip())
    else:
        print("nvidia-smi: not found")
    free = shutil.disk_usage(Path.cwd()).free / 2**30
    print(f"disk free: {free:.1f} GiB")


if __name__ == "__main__":
    main()
