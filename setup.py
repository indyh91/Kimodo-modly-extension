#!/usr/bin/env python3
"""Managed setup for the Kimodo runtime-core extension."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Modly executes extension entrypoints via runpy/importlib, which may not put
# this directory on sys.path. Make sibling modules importable without relying
# on the caller's working directory or modifying any global Python settings.
_EXTENSION_ROOT = Path(__file__).resolve().parent
if str(_EXTENSION_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXTENSION_ROOT))

from host_compat import EXPERIMENTAL, UNSUPPORTED, python_tag_for_python, resolve_host_compatibility


KIMODO_GIT_URL = "git+https://github.com/nv-tlabs/kimodo.git"
SETUP_SUMMARY_FILE = ".kimodo-runtime-core-setup.json"
KIMODO_RUNTIME_DEPENDENCIES = [
    "hydra-core>=1.3",
    "omegaconf>=2.3",
    "numpy>=1.23",
    "scipy>=1.10",
    "transformers==5.1.0",
    "urllib3>=2.6.3",
    "boto3",
    "peft>=0.18",
    "einops>=0.7",
    "tqdm>=4.0",
    "packaging>=21.0",
    "pydantic>=2.0",
    "filelock>=3.20.3",
    "trimesh>=3.21.7",
    "pillow>=9.0",
    "av>=16.1.0",
    "bvhio",
]


def managed_bin_dir(venv_dir: Path) -> Path:
    if platform.system() == "Windows":
        return venv_dir / "Scripts"
    return venv_dir / "bin"


def venv_python(venv_dir: Path) -> Path:
    executable = "python.exe" if platform.system() == "Windows" else "python"
    return managed_bin_dir(venv_dir) / executable


def venv_pip(venv_dir: Path) -> Path:
    executable = "pip.exe" if platform.system() == "Windows" else "pip"
    return managed_bin_dir(venv_dir) / executable


def run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    subprocess.run(command, check=True, env=env)


def pip_install(python: Path, *args: str, env: dict[str, str] | None = None) -> None:
    run([str(python), "-m", "pip", "install", *args], env=env)


def install_torch(python: Path, compatibility) -> None:
    if compatibility.wheel_urls:
        run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--retries",
                "10",
                "--timeout",
                "120",
                "--no-cache-dir",
                "--extra-index-url",
                str(compatibility.torch_index),
                *compatibility.wheel_urls,
            ]
        )
        return

    pip_install(python, *compatibility.torch_packages, "--index-url", str(compatibility.torch_index))


def verify_runtime(python: Path) -> None:
    run([
        str(python),
        "-c",
        "import kimodo, numpy, trimesh; print('kimodo-runtime-core-ready')",
    ])
    run([str(python), "-m", "kimodo.scripts.generate", "--help"])


def write_summary(ext_dir: Path, summary: dict) -> None:
    path = ext_dir / SETUP_SUMMARY_FILE
    path.write_text(f"{json.dumps(summary, indent=2)}\n", encoding="utf-8")


def setup(python_exe: str, ext_dir: Path, gpu_sm: int, cuda_version: int = 0) -> None:
    target_python_tag = python_tag_for_python(python_exe)
    facts, compatibility = resolve_host_compatibility(
        python_tag=target_python_tag,
        gpu_sm=gpu_sm,
        cuda_version=cuda_version,
    )
    if compatibility.state == UNSUPPORTED:
        raise RuntimeError(compatibility.reason)

    venv_dir = ext_dir / "venv"
    run([python_exe, "-m", "venv", str(venv_dir)])
    ext_python = venv_python(venv_dir)

    pip_install(ext_python, "--upgrade", "pip", "setuptools", "wheel")
    install_torch(ext_python, compatibility)
    pip_install(ext_python, *KIMODO_RUNTIME_DEPENDENCIES)

    install_env = os.environ.copy()
    install_env["SKIP_MOTION_CORRECTION_IN_SETUP"] = "1"
    pip_install(ext_python, "--no-deps", KIMODO_GIT_URL, env=install_env)

    verify_runtime(ext_python)

    summary = {
        "status": "ready",
        "phase": "kimodo-host-compat",
        "supported_platform": facts.platform_key,
        "python": str(ext_python),
        "gpu_sm": facts.gpu_sm,
        "cuda_version": facts.cuda_version,
        "torch_packages": list(compatibility.torch_packages),
        "torch_index": compatibility.torch_index,
        "wheel_urls": list(compatibility.wheel_urls),
        "host_facts": facts.as_dict(),
        "host_compat": compatibility.as_dict(),
        "kimodo_install": {
            "source": KIMODO_GIT_URL,
            "motion_correction": "skipped",
            "dependency_profile": "runtime-core-minimal"
        },
        "warning": compatibility.reason if compatibility.state == EXPERIMENTAL else None,
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }
    write_summary(ext_dir, summary)


if __name__ == "__main__":
    if len(sys.argv) >= 4:
        setup(
            python_exe=sys.argv[1],
            ext_dir=Path(sys.argv[2]),
            gpu_sm=int(sys.argv[3]),
            cuda_version=int(sys.argv[4]) if len(sys.argv) >= 5 else 0,
        )
    elif len(sys.argv) == 2:
        payload = json.loads(sys.argv[1])
        setup(
            python_exe=payload["python_exe"],
            ext_dir=Path(payload["ext_dir"]),
            gpu_sm=int(payload.get("gpu_sm", 0)),
            cuda_version=int(payload.get("cuda_version", 0)),
        )
    else:
        print("Usage: python setup.py <python_exe> <ext_dir> <gpu_sm> [cuda_version]")
        print('   or: python setup.py \'{"python_exe":"...","ext_dir":"...","gpu_sm":89,"cuda_version":124}\'')
        sys.exit(1)
