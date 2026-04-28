#!/usr/bin/env python3
"""Shared host compatibility policy for the Kimodo extension."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import platform
import re
import subprocess
import sys
from pathlib import Path


SUPPORTED = "supported"
EXPERIMENTAL = "experimental"
UNSUPPORTED = "unsupported"

X86_CU124_LANE = "linux-x86_64-cu124"
X86_CU128_LANE = "linux-x86_64-cu128"
ARM64_CU124_LANE = "linux-arm64-cu124"
ARM64_CU128_LANE = "linux-arm64-cu128"

X86_CU124_PACKAGES = ("torch==2.6.0", "torchvision==0.21.0")
X86_CU128_PACKAGES = ("torch==2.7.0", "torchvision==0.22.0")

CU124_INDEX = "https://download.pytorch.org/whl/cu124"
CU128_INDEX = "https://download.pytorch.org/whl/cu128"

ARM64_CU124_WHEELS = {
    "cp310": {
        "torch": "https://download-r2.pytorch.org/whl/cu124/torch-2.5.1-cp310-cp310-linux_aarch64.whl#sha256=d468d0eddc188aa3c1e417ec24ce615c48c0c3f592b0354d9d3b99837ef5faa6",
        "torchvision": "https://download-r2.pytorch.org/whl/cu124/torchvision-0.20.1-cp310-cp310-linux_aarch64.whl#sha256=38765e53653f93e529e329755992ddbea81091aacedb61ed053f6a14efb289e5",
    },
    "cp311": {
        "torch": "https://download-r2.pytorch.org/whl/cu124/torch-2.5.1-cp311-cp311-linux_aarch64.whl#sha256=e080353c245b752cd84122e4656261eee6d4323a37cfb7d13e0fffd847bae1a3",
        "torchvision": "https://download-r2.pytorch.org/whl/cu124/torchvision-0.20.1-cp311-cp311-linux_aarch64.whl#sha256=2c5350a08abe005a16c316ae961207a409d0e35df86240db5f77ec41345c82f3",
    },
    "cp312": {
        "torch": "https://download-r2.pytorch.org/whl/cu124/torch-2.5.1-cp312-cp312-linux_aarch64.whl#sha256=302041d457ee169fd925b53da283c13365c6de75c6bb3e84130774b10e2fbb39",
        "torchvision": "https://download-r2.pytorch.org/whl/cu124/torchvision-0.20.1-cp312-cp312-linux_aarch64.whl#sha256=3e3289e53d0cb5d1b7f55b3f5912f46a08293c6791585ba2fc32c12cded9f9af",
    },
}

ARM64_CU128_WHEELS = {
    "cp310": {
        "torch": "https://download-r2.pytorch.org/whl/cu128/torch-2.7.0%2Bcu128-cp310-cp310-manylinux_2_28_aarch64.whl#sha256=b1f0cdd0720ad60536deb5baa427b782fd920dd4fcf72e244d32974caafa3b9e",
        "torchvision": "https://download-r2.pytorch.org/whl/cu128/torchvision-0.22.0-cp310-cp310-manylinux_2_28_aarch64.whl#sha256=566224d7b4f00bc6366bed1d62f834ca80f8e57fe41e10e4a5636bfa3ffb984e",
    },
    "cp311": {
        "torch": "https://download-r2.pytorch.org/whl/cu128/torch-2.7.0%2Bcu128-cp311-cp311-manylinux_2_28_aarch64.whl#sha256=47c895bcab508769d129d717a4b916b10225ae3855723aeec8dff8efe5346207",
        "torchvision": "https://download-r2.pytorch.org/whl/cu128/torchvision-0.22.0-cp311-cp311-manylinux_2_28_aarch64.whl#sha256=6be714bcdd8849549571f6acfaa2dfa9e00676f042bda517432745fb116f7904",
    },
    "cp312": {
        "torch": "https://download-r2.pytorch.org/whl/cu128/torch-2.7.0%2Bcu128-cp312-cp312-manylinux_2_28_aarch64.whl#sha256=6bba7dca5d9a729f1e8e9befb98055498e551efaf5ed034824c168b560afc1ac",
        "torchvision": "https://download-r2.pytorch.org/whl/cu128/torchvision-0.22.0-cp312-cp312-manylinux_2_28_aarch64.whl#sha256=6e9752b48c1cdd7f6428bcd30c3d198b30ecea348d16afb651f95035e5252506",
    },
    "cp313": {
        "torch": "https://download-r2.pytorch.org/whl/cu128/torch-2.7.0%2Bcu128-cp313-cp313-manylinux_2_28_aarch64.whl#sha256=633f35e8b1b1f640ef5f8a98dbd84f19b548222ce7ba8f017fe47ce6badc106a",
        "torchvision": "https://download-r2.pytorch.org/whl/cu128/torchvision-0.22.0-cp313-cp313-manylinux_2_28_aarch64.whl#sha256=e4d4d5a14225875d9bf8c5221d43d8be97786adc498659493799bdeff52c54cf",
    },
}


@dataclass(frozen=True)
class HostFacts:
    system: str
    machine: str
    platform_key: str
    python_tag: str
    gpu_sm: int
    cuda_version: int
    gpu_name: str | None = None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class HostCompatibility:
    state: str
    platform_key: str
    lane: str | None
    reason: str
    torch_index: str | None
    torch_packages: tuple[str, ...]
    wheel_urls: tuple[str, ...]
    notes: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "state": self.state,
            "platform_key": self.platform_key,
            "lane": self.lane,
            "reason": self.reason,
            "torch_index": self.torch_index,
            "torch_packages": list(self.torch_packages),
            "wheel_urls": list(self.wheel_urls),
            "notes": list(self.notes),
        }


def normalize_machine(raw: str) -> str:
    value = raw.strip().lower()
    if value in {"amd64", "x64"}:
        return "x86_64"
    if value == "aarch64":
        return "arm64"
    return value


def platform_key(system: str | None = None, machine: str | None = None) -> str:
    current_system = (system or platform.system()).strip().lower()
    current_machine = normalize_machine(machine or platform.machine())
    return f"{current_system}/{current_machine}"


def current_python_tag() -> str:
    return f"cp{sys.version_info.major}{sys.version_info.minor}"


def python_tag_for_python(python_exe: str | Path) -> str:
    output = subprocess.check_output(
        [str(python_exe), "-c", "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')"],
        text=True,
    ).strip()
    if not output:
        raise RuntimeError("Unable to resolve Python tag for target interpreter.")
    return output


def parse_compute_cap(raw: str) -> int:
    cleaned = raw.strip()
    if not cleaned:
        return 0
    if "." in cleaned:
        major, minor = cleaned.split(".", 1)
        return int(major) * 10 + int(minor)
    return int(cleaned)


def parse_cuda_version(raw: str) -> int:
    cleaned = raw.strip()
    if not cleaned:
        return 0
    if "." in cleaned:
        major, minor = cleaned.split(".", 1)
        return int(major) * 10 + int(minor)
    digits = re.sub(r"\D", "", cleaned)
    return int(digits) if digits else 0


def _command_output(command: list[str]) -> str | None:
    try:
        result = subprocess.run(command, capture_output=True, check=True, text=True)
    except Exception:
        return None
    output = result.stdout.strip()
    return output or None


def detect_gpu_sm() -> int:
    output = _command_output(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"])
    if not output:
        return 0
    first_line = output.splitlines()[0].strip()
    try:
        return parse_compute_cap(first_line)
    except Exception:
        return 0


def detect_cuda_version() -> int:
    output = _command_output(["nvidia-smi"])
    if output:
        match = re.search(r"CUDA Version:\s*([0-9]+\.[0-9]+)", output)
        if match:
            return parse_cuda_version(match.group(1))

    output = _command_output(["nvcc", "--version"])
    if output:
        match = re.search(r"release\s+([0-9]+\.[0-9]+)", output)
        if match:
            return parse_cuda_version(match.group(1))
    return 0


def detect_gpu_name() -> str | None:
    output = _command_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
    if not output:
        return None
    return output.splitlines()[0].strip() or None


def build_host_facts(
    *,
    system: str | None = None,
    machine: str | None = None,
    python_tag: str | None = None,
    gpu_sm: int | None = None,
    cuda_version: int | None = None,
    gpu_name: str | None = None,
) -> HostFacts:
    resolved_system = (system or platform.system()).strip().lower()
    resolved_machine = normalize_machine(machine or platform.machine())
    resolved_python_tag = python_tag or current_python_tag()
    resolved_gpu_sm = detect_gpu_sm() if gpu_sm in (None, 0) else gpu_sm
    resolved_cuda_version = detect_cuda_version() if cuda_version in (None, 0) else cuda_version
    resolved_gpu_name = gpu_name if gpu_name is not None else detect_gpu_name()
    return HostFacts(
        system=resolved_system,
        machine=resolved_machine,
        platform_key=f"{resolved_system}/{resolved_machine}",
        python_tag=resolved_python_tag,
        gpu_sm=resolved_gpu_sm,
        cuda_version=resolved_cuda_version,
        gpu_name=resolved_gpu_name,
    )


def resolve_host_compatibility(**kwargs) -> tuple[HostFacts, HostCompatibility]:
    facts = build_host_facts(**kwargs)
    return facts, classify_host(facts)


def classify_host(facts: HostFacts) -> HostCompatibility:
    if facts.platform_key == "linux/x86_64":
        return _classify_linux_x86_64(facts)
    if facts.platform_key == "linux/arm64":
        return _classify_linux_arm64(facts)
    if facts.platform_key == "windows/x86_64":
        return HostCompatibility(
            state=UNSUPPORTED,
            platform_key=facts.platform_key,
            lane=None,
            reason="Windows x86_64 remains outside the validated runtime-core and host-compat scope for Kimodo in this repository.",
            torch_index=None,
            torch_packages=(),
            wheel_urls=(),
            notes=("Upstream Kimodo mentions Windows, but this extension has no validated Windows runtime path yet.",),
        )
    return HostCompatibility(
        state=UNSUPPORTED,
        platform_key=facts.platform_key,
        lane=None,
        reason=(
            "Kimodo host compatibility currently models only Linux NVIDIA hosts for this extension. "
            f"Received unsupported platform {facts.platform_key}."
        ),
        torch_index=None,
        torch_packages=(),
        wheel_urls=(),
        notes=(),
    )


def _classify_linux_x86_64(facts: HostFacts) -> HostCompatibility:
    if facts.gpu_sm >= 100 or facts.cuda_version >= 128:
        return HostCompatibility(
            state=SUPPORTED,
            platform_key=facts.platform_key,
            lane=X86_CU128_LANE,
            reason="Linux x86_64 + NVIDIA Blackwell/CUDA 12.8+ is a supported Kimodo runtime lane.",
            torch_index=CU128_INDEX,
            torch_packages=X86_CU128_PACKAGES,
            wheel_urls=(),
            notes=(
                "Uses the cu128 x86_64 PyTorch lane.",
                "Postprocess remains disabled in runtime-core regardless of host lane.",
            ),
        )

    if facts.gpu_sm >= 70 and (facts.cuda_version == 0 or facts.cuda_version >= 124):
        reason = "Linux x86_64 + NVIDIA SM 70+ is a supported Kimodo runtime lane."
        if facts.cuda_version == 0:
            reason = "Linux x86_64 + NVIDIA SM 70+ is treated as supported; CUDA version was not detected so cu124 is selected conservatively."
        return HostCompatibility(
            state=SUPPORTED,
            platform_key=facts.platform_key,
            lane=X86_CU124_LANE,
            reason=reason,
            torch_index=CU124_INDEX,
            torch_packages=X86_CU124_PACKAGES,
            wheel_urls=(),
            notes=(
                "Uses the cu124 x86_64 PyTorch lane.",
                "Postprocess remains disabled in runtime-core regardless of host lane.",
            ),
        )

    if facts.gpu_sm >= 70:
        return HostCompatibility(
            state=UNSUPPORTED,
            platform_key=facts.platform_key,
            lane=None,
            reason=(
                "Linux x86_64 host has a usable NVIDIA GPU, but the detected CUDA driver/toolkit is below the modeled cu124 baseline for this extension. "
                f"Received gpu_sm={facts.gpu_sm} cuda_version={facts.cuda_version or 'unknown'}."
            ),
            torch_index=None,
            torch_packages=(),
            wheel_urls=(),
            notes=(),
        )

    return HostCompatibility(
        state=UNSUPPORTED,
        platform_key=facts.platform_key,
        lane=None,
        reason=(
            "Kimodo runtime-core currently targets NVIDIA CUDA hosts with GPU SM >= 70. "
            f"Received gpu_sm={facts.gpu_sm or 'unknown'} on {facts.platform_key}."
        ),
        torch_index=None,
        torch_packages=(),
        wheel_urls=(),
        notes=(),
    )


def _classify_linux_arm64(facts: HostFacts) -> HostCompatibility:
    if facts.gpu_sm >= 100 or facts.cuda_version >= 128:
        wheel_urls = _arm64_wheels(ARM64_CU128_WHEELS, facts.python_tag)
        if wheel_urls is None:
            return _unsupported_missing_arm64_wheels(facts, ARM64_CU128_LANE, ARM64_CU128_WHEELS)
        return HostCompatibility(
            state=EXPERIMENTAL,
            platform_key=facts.platform_key,
            lane=ARM64_CU128_LANE,
            reason=(
                "Linux arm64 + NVIDIA Blackwell/CUDA 12.8+ is allowed through the experimental Kimodo host-compat lane. "
                "This is best-effort support, not parity or SLA."
            ),
            torch_index=CU128_INDEX,
            torch_packages=(),
            wheel_urls=wheel_urls,
            notes=(
                "Uses pinned ARM64 cu128 PyTorch wheels.",
                "Experimental means limited validation, explicit diagnostics, and no host-wide parity promise.",
                "Postprocess remains disabled in runtime-core regardless of host lane.",
            ),
        )

    if facts.gpu_sm >= 70 and (facts.cuda_version == 0 or facts.cuda_version >= 124):
        wheel_urls = _arm64_wheels(ARM64_CU124_WHEELS, facts.python_tag)
        if wheel_urls is None:
            return _unsupported_missing_arm64_wheels(facts, ARM64_CU124_LANE, ARM64_CU124_WHEELS)
        reason = (
            "Linux arm64 + NVIDIA SM 70+ is allowed through the experimental Kimodo host-compat lane. "
            "This is best-effort support, not parity or SLA."
        )
        if facts.cuda_version == 0:
            reason = (
                "Linux arm64 + NVIDIA SM 70+ is treated as experimental; CUDA version was not detected so the cu124 ARM64 lane is selected conservatively."
            )
        return HostCompatibility(
            state=EXPERIMENTAL,
            platform_key=facts.platform_key,
            lane=ARM64_CU124_LANE,
            reason=reason,
            torch_index=CU124_INDEX,
            torch_packages=(),
            wheel_urls=wheel_urls,
            notes=(
                "Uses pinned ARM64 cu124 PyTorch wheels.",
                "Experimental means limited validation, explicit diagnostics, and no host-wide parity promise.",
                "Postprocess remains disabled in runtime-core regardless of host lane.",
            ),
        )

    if facts.gpu_sm >= 70:
        return HostCompatibility(
            state=UNSUPPORTED,
            platform_key=facts.platform_key,
            lane=None,
            reason=(
                "Linux arm64 host has a usable NVIDIA GPU, but the detected CUDA driver/toolkit is below the modeled cu124 baseline for the experimental lane. "
                f"Received gpu_sm={facts.gpu_sm} cuda_version={facts.cuda_version or 'unknown'}."
            ),
            torch_index=None,
            torch_packages=(),
            wheel_urls=(),
            notes=(),
        )

    return HostCompatibility(
        state=UNSUPPORTED,
        platform_key=facts.platform_key,
        lane=None,
        reason=(
            "Linux arm64 compatibility for Kimodo is only modeled for NVIDIA CUDA hosts with GPU SM >= 70. "
            f"Received gpu_sm={facts.gpu_sm or 'unknown'} on {facts.platform_key}."
        ),
        torch_index=None,
        torch_packages=(),
        wheel_urls=(),
        notes=(),
    )


def _arm64_wheels(wheel_map: dict[str, dict[str, str]], python_tag: str) -> tuple[str, ...] | None:
    urls = wheel_map.get(python_tag)
    if urls is None:
        return None
    return (urls["torch"], urls["torchvision"])


def _unsupported_missing_arm64_wheels(
    facts: HostFacts,
    lane: str,
    wheel_map: dict[str, dict[str, str]],
) -> HostCompatibility:
    supported_tags = ", ".join(sorted(wheel_map))
    return HostCompatibility(
        state=UNSUPPORTED,
        platform_key=facts.platform_key,
        lane=None,
        reason=(
            f"The {lane} experimental lane has no pinned ARM64 PyTorch wheels for Python tag {facts.python_tag}. "
            f"Supported tags: {supported_tags}."
        ),
        torch_index=None,
        torch_packages=(),
        wheel_urls=(),
        notes=(),
    )
