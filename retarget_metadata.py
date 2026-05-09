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
    visual_quality = _visual_quality_metadata(clip)
    if visual_quality is not None:
        merged_diagnostics["visual_quality"] = visual_quality
    top_level_calibration = calibration if calibration.get("version") else None
    merged_warnings = _merge_success_warnings(warnings, visual_quality)
    visual_quality_status = visual_quality.get("status") if isinstance(visual_quality, dict) else None
    foot_stabilization = visual_quality.get("foot_stabilization") if isinstance(visual_quality, dict) else None
    stabilization_status = foot_stabilization.get("status") if isinstance(foot_stabilization, dict) else "not_evaluated"
    return {
        "runtime_status": "success",
        "contract_status": getattr(rig, "contract_status", "contract_missing_or_untrusted"),
        "stabilization_status": stabilization_status or "not_evaluated",
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
        "warnings": merged_warnings,
        "visual_quality_status": visual_quality_status,
        "diagnostics": merged_diagnostics,
        "humanoid_contract": humanoid_contract,
        "retarget_calibration": top_level_calibration,
    }


def _visual_quality_metadata(clip: Any) -> dict[str, Any] | None:
    raw = getattr(clip, "visual_quality", None)
    if not isinstance(raw, dict):
        calibration = getattr(clip, "calibration", {}) or {}
        raw = calibration.get("visual_quality") if isinstance(calibration, dict) else None
    if not isinstance(raw, dict):
        return None
    payload = dict(raw)
    payload.setdefault("version", "kimodo.visual_quality.v1")
    payload.setdefault("blocking", False)
    payload.setdefault("limitations", ["Diagnostics do not prove visual success."])
    payload.setdefault("warnings", [])
    return payload


def _merge_success_warnings(warnings: list[str] | tuple[str, ...], visual_quality: dict[str, Any] | None) -> list[str]:
    merged = list(warnings)
    if isinstance(visual_quality, dict):
        status = visual_quality.get("status")
        visual_warnings = [str(warning) for warning in visual_quality.get("warnings", []) if isinstance(warning, str)]
        if status in {"warning", "not_evaluated"} or visual_warnings:
            merged.append("visual_quality_warning")
        merged.extend(visual_warnings)
    return list(dict.fromkeys(merged))


def _humanoid_contract_metadata(rig: Any) -> dict[str, Any] | None:
    path = getattr(rig, "humanoid_contract_path", None)
    if path is None:
        return None
    return {
        "path": str(path),
        "schema": getattr(rig, "humanoid_contract_schema", None),
        "status": getattr(rig, "contract_status", "contract_missing_or_untrusted"),
        "trusted": getattr(rig, "contract_status", "contract_missing_or_untrusted") == "trusted",
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
