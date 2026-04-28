"""Metadata helpers for retarget/export attempts."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from retarget_errors import RetargetError, legacy_error_aliases


def retarget_failure_metadata(error: RetargetError, *, diagnostics: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "retarget_status": "failed",
        "retarget_error_code": error.code,
        "retarget_error_aliases": list(legacy_error_aliases(error.code)),
        "retarget_error_message": error.message,
        "diagnostics": diagnostics or {},
    }


def retarget_success_metadata(
    output_path: Path,
    *,
    rig: Any,
    clip: Any,
    warnings: list[str] | tuple[str, ...],
    diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    mapping = getattr(clip, "mapping", None)
    return {
        "retarget_status": "success",
        "animated_glb": str(output_path),
        "clip_name": getattr(clip, "name", "Kimodo Motion"),
        "duration": getattr(clip, "duration", None),
        "classification": getattr(rig, "classification", "unknown"),
        "source_kind": getattr(rig, "source_kind", "unknown_mesh_source"),
        "mapping_confidence": getattr(mapping, "confidence", None),
        "warnings": list(warnings),
        "diagnostics": diagnostics or {},
    }


def serializable(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    return value
