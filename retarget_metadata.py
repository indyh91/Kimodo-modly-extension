"""Metadata helpers for retarget/export attempts."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from retarget_errors import RetargetError, legacy_error_aliases


def retarget_failure_metadata(error: RetargetError, *, diagnostics: dict[str, Any] | None = None) -> dict[str, Any]:
    merged_diagnostics = dict(diagnostics or {})
    merged_diagnostics.update(getattr(error, "diagnostics", {}) or {})
    payload = {
        "retarget_status": "failed",
        "retarget_error_code": error.code,
        "retarget_error_aliases": list(legacy_error_aliases(error.code)),
        "retarget_error_message": error.message,
        "diagnostics": merged_diagnostics,
    }
    if merged_diagnostics.get("humanoid_contract"):
        payload["humanoid_contract"] = merged_diagnostics["humanoid_contract"]
    if merged_diagnostics.get("retarget_calibration"):
        payload["retarget_calibration"] = merged_diagnostics["retarget_calibration"]
    return payload


def retarget_success_metadata(
    output_path: Path,
    *,
    rig: Any,
    clip: Any,
    warnings: list[str] | tuple[str, ...],
    diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    mapping = getattr(clip, "mapping", None)
    humanoid_contract = _humanoid_contract_metadata(rig)
    merged_diagnostics = dict(diagnostics or {})
    if humanoid_contract:
        merged_diagnostics["humanoid_contract"] = humanoid_contract
    calibration = getattr(clip, "calibration", {}) or {}
    if calibration:
        merged_diagnostics["retarget_calibration"] = calibration
    top_level_calibration = calibration if calibration.get("version") else None
    return {
        "retarget_status": "success",
        "retarget_error_code": None,
        "retarget_error_aliases": [],
        "retarget_error_message": None,
        "animated_glb": str(output_path),
        "clip_name": getattr(clip, "name", "Kimodo Motion"),
        "duration": getattr(clip, "duration", None),
        "classification": getattr(rig, "classification", "unknown"),
        "source_kind": getattr(rig, "source_kind", "unknown_mesh_source"),
        "mapping_confidence": getattr(mapping, "confidence", None),
        "warnings": list(warnings),
        "diagnostics": merged_diagnostics,
        "humanoid_contract": humanoid_contract,
        "retarget_calibration": top_level_calibration,
    }


def _humanoid_contract_metadata(rig: Any) -> dict[str, Any] | None:
    path = getattr(rig, "humanoid_contract_path", None)
    if path is None:
        return None
    return {
        "path": str(path),
        "schema": getattr(rig, "humanoid_contract_schema", None),
        "roles": [serializable(role) for role in getattr(rig, "humanoid_roles", ())],
        "chains": [serializable(chain) for chain in getattr(rig, "humanoid_chains", ())],
        "quality": list(getattr(rig, "humanoid_quality", ())),
        "rest": getattr(rig, "humanoid_rest", {}) or {},
        "basis_hints": getattr(rig, "humanoid_basis_hints", {}) or {},
        "diagnostics": list(getattr(rig, "humanoid_diagnostics", ())),
    }


def serializable(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    return value
