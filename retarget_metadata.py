"""Metadata helpers for retarget/export attempts."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from retarget_errors import RetargetError, legacy_error_aliases


def retarget_failure_metadata(error: RetargetError, *, diagnostics: dict[str, Any] | None = None) -> dict[str, Any]:
    merged_diagnostics = _metadata_safe_copy(diagnostics or {})
    merged_diagnostics.update(getattr(error, "diagnostics", {}) or {})
    payload = {
        "retarget_status": "failed",
        "animation_mapping_status": "failed",
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
    merged_diagnostics = _metadata_safe_copy(diagnostics or {})
    if humanoid_contract:
        merged_diagnostics["humanoid_contract"] = humanoid_contract
    animation_mapping_status = _animation_mapping_status(rig, mapping)
    raw_calibration = getattr(clip, "calibration", {}) or {}
    calibration = _metadata_safe_copy(raw_calibration) if isinstance(raw_calibration, dict) else {}
    locomotion_axis = _locomotion_axis_metadata(calibration)
    if locomotion_axis is not None:
        merged_diagnostics["locomotion_axis"] = locomotion_axis
    limb_swing_plane = _limb_swing_plane_metadata(raw_calibration)
    if limb_swing_plane is not None:
        merged_diagnostics["limb_swing_plane"] = limb_swing_plane
    if calibration and animation_mapping_status != "semantic_candidate":
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
        "animation_mapping_status": animation_mapping_status,
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


def _animation_mapping_status(rig: Any, mapping: Any) -> str:
    confidence = getattr(mapping, "confidence", None)
    warnings = tuple(getattr(mapping, "warnings", ()) or ())
    if confidence == "calibrated" or "humanoid_contract_sidecar_mapping" in warnings:
        return "trusted_contract"
    if confidence == "semantic_candidate" or "semantic_candidate_mapping" in warnings:
        return "semantic_candidate"
    if getattr(rig, "contract_status", "contract_missing_or_untrusted") == "trusted":
        return "trusted_contract"
    return "best_effort"


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


def _locomotion_axis_metadata(calibration: Any) -> dict[str, Any] | None:
    if not isinstance(calibration, dict):
        return None
    payload = calibration.get("locomotion_axis_alignment")
    if not isinstance(payload, dict):
        return None
    copied = _metadata_safe_copy(payload)
    return copied if isinstance(copied, dict) else None


def _limb_swing_plane_metadata(calibration: Any) -> dict[str, Any] | None:
    if not isinstance(calibration, dict):
        return None
    payload = calibration.get("limb_swing_plane")
    if not isinstance(payload, dict) or payload.get("schema") != "kimodo.limb_swing_plane.diagnostics.v1":
        return None

    feet = payload.get("feet") if isinstance(payload.get("feet"), dict) else {}
    root_hips_axis = payload.get("root_hips_axis") if isinstance(payload.get("root_hips_axis"), dict) else {}
    correction = payload.get("correction") if isinstance(payload.get("correction"), dict) else {}

    summary = {
        "schema": payload.get("schema"),
        "status": payload.get("status"),
        "classification": payload.get("classification"),
        "risk": payload.get("risk"),
        "source": payload.get("diagnostic_source") or payload.get("source"),
        "sample_count": payload.get("sample_count"),
        "root_advance_summary": _limb_root_summary(root_hips_axis),
        "per_foot_stats": _limb_per_foot_stats(feet),
        "midline_crossing": _limb_midline_crossing(feet),
        "validation_gates": {
            "basis": _metadata_safe_copy(payload.get("basis") if isinstance(payload.get("basis"), dict) else {}),
            "mirror": _metadata_safe_copy(payload.get("mirror_validation") if isinstance(payload.get("mirror_validation"), dict) else {}),
            "chain": _metadata_safe_copy(payload.get("chain_validation") if isinstance(payload.get("chain_validation"), dict) else {}),
        },
        "correction_eligibility": _metadata_safe_copy(payload.get("correction_eligibility") if isinstance(payload.get("correction_eligibility"), dict) else {}),
        "correction": {
            "status": correction.get("status") or correction.get("correction_status") or "skipped",
            "correction_status": correction.get("correction_status") or correction.get("status") or "skipped",
            "eligible": bool(correction.get("eligible")),
            "applied": bool(correction.get("applied")),
            "target_scope": correction.get("target_scope"),
            "applied_channels": _metadata_safe_copy(correction.get("applied_channels") if isinstance(correction.get("applied_channels"), list) else []),
            "recommended_channels": _metadata_safe_copy(correction.get("recommended_channels") if isinstance(correction.get("recommended_channels"), list) else []),
            "reason": correction.get("reason") or correction.get("skipped_reason"),
            "reasons": _metadata_safe_copy(correction.get("reasons") if isinstance(correction.get("reasons"), list) else []),
        },
        "absolute_foot_motion_insufficient": payload.get("absolute_foot_motion_insufficient"),
        "thresholds": _metadata_safe_copy(payload.get("thresholds") if isinstance(payload.get("thresholds"), dict) else {}),
        "confidence": payload.get("confidence"),
        "reasons": _metadata_safe_copy(payload.get("reasons") if isinstance(payload.get("reasons"), list) else []),
        "warnings": _metadata_safe_copy(payload.get("warnings") if isinstance(payload.get("warnings"), list) else []),
        "pre_export_foot_samples": _pre_export_foot_sample_summary(payload.get("pre_export_foot_samples")),
    }
    return summary


def _limb_root_summary(root_hips_axis: dict[str, Any]) -> dict[str, Any]:
    return {
        "aligned": root_hips_axis.get("aligned"),
        "signed_axis": root_hips_axis.get("signed_axis"),
        "forward_range": root_hips_axis.get("forward_range"),
        "lateral_drift_ratio": root_hips_axis.get("lateral_drift_ratio"),
    }


def _limb_per_foot_stats(feet: dict[str, Any]) -> dict[str, Any]:
    return {side: _limb_foot_stats(metrics) for side, metrics in feet.items() if side in {"left", "right"} and isinstance(metrics, dict)}


def _limb_foot_stats(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_count": metrics.get("sample_count"),
        "lateral_range": metrics.get("lateral_range"),
        "forward_range": metrics.get("forward_range"),
        "lateral_forward_ratio": metrics.get("lateral_forward_ratio"),
        "wrong_side_fraction": metrics.get("wrong_side_fraction"),
        "lateral_dominated": metrics.get("lateral_dominated"),
    }


def _limb_midline_crossing(feet: dict[str, Any]) -> dict[str, Any]:
    return {
        side: _metadata_safe_copy(metrics.get("midline_crossing") if isinstance(metrics.get("midline_crossing"), dict) else {})
        for side, metrics in feet.items()
        if side in {"left", "right"} and isinstance(metrics, dict)
    }


def _pre_export_foot_sample_summary(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    summary = {"status": value.get("status")}
    if value.get("reason") is not None:
        summary["reason"] = value.get("reason")
    samples = value.get("samples")
    if summary.get("status") == "success" and isinstance(samples, dict):
        summary["sample_counts"] = {side: len(positions) for side, positions in samples.items() if side in {"left", "right"} and isinstance(positions, (list, tuple))}
    return summary


def _metadata_safe_copy(value: Any, seen: set[int] | None = None) -> Any:
    if seen is None:
        seen = set()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    marker = id(value)
    if marker in seen:
        return None
    if isinstance(value, dict):
        if value.get("schema") == "kimodo.limb_swing_plane.diagnostics.v1":
            return _limb_swing_plane_metadata({"limb_swing_plane": value})
        seen.add(marker)
        copied = {}
        for key, item in value.items():
            key_text = str(key)
            if key_text == "evidence" and isinstance(item, dict) and item.get("schema") == "kimodo.locomotion_axis.diagnostics.v1":
                copied[key_text] = {
                    "schema": item.get("schema"),
                    "classification": item.get("classification"),
                    "root_advance_axis": (item.get("root_advance") or {}).get("signed_axis") if isinstance(item.get("root_advance"), dict) else None,
                    "contract_forward_axis": (item.get("contract_forward") or {}).get("signed_axis") if isinstance(item.get("contract_forward"), dict) else None,
                }
                continue
            copied[key_text] = _metadata_safe_copy(item, seen)
        return copied
    if isinstance(value, list):
        seen.add(marker)
        copied_list = [_metadata_safe_copy(item, seen) for item in value]
        return copied_list
    if isinstance(value, tuple):
        seen.add(marker)
        copied_tuple = tuple(_metadata_safe_copy(item, seen) for item in value)
        return copied_tuple
    return serializable(value)


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
