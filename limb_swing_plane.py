"""Pure limb swing-plane diagnostics for Kimodo retarget safety gates."""

from __future__ import annotations

import math
from typing import Any


SCHEMA = "kimodo.limb_swing_plane.diagnostics.v1"
CORRECTION_PLAN_SCHEMA = "kimodo.limb_swing_plane.correction_plan.v1"

_LOWER_BODY_ROLES = {
    "left_upper_leg",
    "left_lower_leg",
    "left_foot",
    "left_toes",
    "right_upper_leg",
    "right_lower_leg",
    "right_foot",
    "right_toes",
}

_DEFAULT_THRESHOLDS = {
    "min_samples": 3,
    "root_lateral_drift_ratio_max": 0.1,
    "lateral_dominance_ratio_min": 2.0,
    "midline_wrong_side_fraction_min": 0.05,
    "forward_epsilon": 1e-6,
}


def diagnose_limb_swing_plane(
    *,
    root_positions: Any,
    hips_positions: Any,
    left_foot_positions: Any,
    right_foot_positions: Any,
    basis: dict[str, Any] | None = None,
    mirror_evidence: dict[str, Any] | None = None,
    chain_evidence: dict[str, Any] | None = None,
    thresholds: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return JSON-safe hips-relative foot swing-plane diagnostics.

    Phase 1 deliberately stays pure: no filesystem access, no Modly backend calls,
    no rig mutation, and no inferred basis authority. The default contract treats
    +Z as forward and X as lateral, matching the current retarget contract.
    """

    resolved_thresholds = _resolve_thresholds(thresholds)
    basis_payload = _basis_payload(basis)
    mirror_validation = _validation_payload("mirror", mirror_evidence)
    chain_validation = _validation_payload("chain", chain_evidence)
    root = _coerce_positions(root_positions)
    hips = _coerce_positions(hips_positions)
    left = _coerce_positions(left_foot_positions)
    right = _coerce_positions(right_foot_positions)
    sample_count = min(len(root), len(hips), len(left), len(right))

    if sample_count < int(resolved_thresholds["min_samples"]):
        return _not_evaluated(resolved_thresholds, sample_count, basis_payload, mirror_validation, chain_validation)

    root = root[:sample_count]
    hips = hips[:sample_count]
    left = left[:sample_count]
    right = right[:sample_count]

    root_axis = _axis_metrics(root, resolved_thresholds)
    hips_axis = _axis_metrics(hips, resolved_thresholds)
    root_hips_axis = _root_hips_axis(root_axis, hips_axis, resolved_thresholds)

    feet = {
        "left": _foot_metrics("left", left, hips, resolved_thresholds),
        "right": _foot_metrics("right", right, hips, resolved_thresholds),
    }
    lateral_dominated_feet = [side for side, metrics in feet.items() if metrics["lateral_dominated"]]
    crossing_feet = [side for side, metrics in feet.items() if metrics["midline_crossing"]["detected"]]
    absolute_foot_motion_insufficient = bool(lateral_dominated_feet or crossing_feet)

    reasons: list[dict[str, Any]] = []
    warnings: list[str] = []
    if not root_hips_axis["aligned"]:
        reasons.append({"code": "root/not_forward_invariant"})
        warnings.append("root/not_forward_invariant")
    if lateral_dominated_feet:
        reasons.append({"code": "limb/lateral_dominated", "sides": lateral_dominated_feet})
        warnings.append("limb/lateral_dominated")
    if crossing_feet:
        reasons.append({"code": "limb/midline_crossing_detected", "sides": crossing_feet})
        warnings.append("limb/midline_crossing_detected")

    classification = _classification(root_hips_axis, lateral_dominated_feet, crossing_feet)
    correction_eligibility = _correction_eligibility(
        status="evaluated",
        basis_payload=basis_payload,
        mirror_validation=mirror_validation,
        chain_validation=chain_validation,
        diagnostic_reasons=reasons,
    )

    payload = {
        "schema": SCHEMA,
        "status": "evaluated",
        "classification": classification,
        "risk": classification,
        "sample_count": sample_count,
        "basis": basis_payload,
        "mirror_validation": mirror_validation,
        "chain_validation": chain_validation,
        "correction_eligibility": correction_eligibility,
        "correction": _correction_payload(),
        "root_hips_axis": root_hips_axis,
        "feet": feet,
        "absolute_foot_motion_insufficient": absolute_foot_motion_insufficient,
        "thresholds": resolved_thresholds,
        "confidence": _confidence(root_hips_axis, feet),
        "reasons": reasons,
        "warnings": warnings,
    }
    payload["correction"] = plan_limb_swing_plane_correction(payload)
    return payload


def _resolve_thresholds(thresholds: dict[str, Any] | None) -> dict[str, Any]:
    resolved = dict(_DEFAULT_THRESHOLDS)
    for key, value in (thresholds or {}).items():
        if key in resolved:
            resolved[key] = _clean_float(value)
    resolved["min_samples"] = int(resolved["min_samples"])
    return resolved


def _not_evaluated(
    thresholds: dict[str, Any],
    sample_count: int,
    basis_payload: dict[str, Any],
    mirror_validation: dict[str, Any],
    chain_validation: dict[str, Any],
) -> dict[str, Any]:
    reasons = [{"code": "samples/insufficient"}]
    correction_eligibility = _correction_eligibility(
        status="not_evaluated",
        basis_payload=basis_payload,
        mirror_validation=mirror_validation,
        chain_validation=chain_validation,
        diagnostic_reasons=reasons,
    )
    payload = {
        "schema": SCHEMA,
        "status": "not_evaluated",
        "classification": "not_evaluated",
        "risk": "not_evaluated",
        "sample_count": int(sample_count),
        "basis": basis_payload,
        "mirror_validation": mirror_validation,
        "chain_validation": chain_validation,
        "correction_eligibility": correction_eligibility,
        "correction": _correction_payload(),
        "root_hips_axis": None,
        "feet": {},
        "absolute_foot_motion_insufficient": True,
        "thresholds": thresholds,
        "confidence": 0.0,
        "reasons": reasons,
        "warnings": ["samples/insufficient"],
    }
    payload["correction"] = plan_limb_swing_plane_correction(payload)
    return payload


def plan_limb_swing_plane_correction(
    diagnostics: dict[str, Any] | None,
    *,
    channels: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None = None,
) -> dict[str, Any]:
    """Return a trusted-only correction plan without applying any correction.

    Phase 6 intentionally creates a fail-closed seam only. It never mutates
    rotations, translations, clips, channels, or runtime/export artifacts.
    """

    payload = diagnostics if isinstance(diagnostics, dict) else {}
    status = str(payload.get("status") or "not_evaluated")
    gate_summary = _correction_gate_summary(payload)
    reasons = _correction_gate_reasons(payload)
    warnings: list[str] = []

    if status != "evaluated":
        reasons.extend(_reason_strings(payload.get("reasons")))

    if not _has_limb_lateral_risk(payload):
        reasons.append("motion/non_lateral_or_no_limb_risk")

    reasons = _dedupe(reasons)
    eligible = not reasons
    target_channels = _lower_body_rotation_channels(channels) if eligible else []
    target_scope = "lower_body_rotation_channels" if eligible else None
    plan_status = "eligible_not_applied" if eligible else "skipped"

    return {
        "schema": CORRECTION_PLAN_SCHEMA,
        "status": plan_status,
        "correction_status": plan_status,
        "applied": False,
        "eligible": eligible,
        "policy": "trusted_basis_and_mirror_required",
        "target_scope": target_scope,
        "recommended_channels": target_channels,
        "applied_channels": [],
        "reasons": reasons,
        "warnings": warnings,
        "gate_summary": gate_summary,
    }


def _correction_gate_summary(diagnostics: dict[str, Any]) -> dict[str, Any]:
    return {
        "basis": _safe_gate_copy(diagnostics.get("basis")),
        "mirror": _safe_gate_copy(diagnostics.get("mirror_validation")),
        "chain": _safe_gate_copy(diagnostics.get("chain_validation")),
        "diagnostics": {
            "status": diagnostics.get("status"),
            "risk": diagnostics.get("risk") or diagnostics.get("classification"),
            "sample_count": diagnostics.get("sample_count"),
        },
    }


def _safe_gate_copy(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    copied: dict[str, Any] = {}
    for key, item in value.items():
        if isinstance(item, (str, int, float, bool)) or item is None:
            copied[str(key)] = item
        elif isinstance(item, list):
            copied[str(key)] = [entry for entry in item if isinstance(entry, (str, int, float, bool, dict))]
        elif isinstance(item, dict):
            copied[str(key)] = {str(nested_key): nested_value for nested_key, nested_value in item.items() if isinstance(nested_value, (str, int, float, bool, list, dict)) or nested_value is None}
    return copied


def _correction_gate_reasons(diagnostics: dict[str, Any]) -> list[str]:
    eligibility = diagnostics.get("correction_eligibility") if isinstance(diagnostics.get("correction_eligibility"), dict) else {}
    reasons = _reason_strings(eligibility.get("reasons"))
    if not reasons and eligibility.get("eligible") is False:
        reasons.append("eligibility/not_eligible")
    return reasons


def _has_limb_lateral_risk(diagnostics: dict[str, Any]) -> bool:
    risk = str(diagnostics.get("risk") or diagnostics.get("classification") or "")
    if risk in {"lateral_dominated_limb_swing", "ambiguous_limb_swing"}:
        return True
    feet = diagnostics.get("feet") if isinstance(diagnostics.get("feet"), dict) else {}
    for metrics in feet.values():
        if not isinstance(metrics, dict):
            continue
        crossing = metrics.get("midline_crossing") if isinstance(metrics.get("midline_crossing"), dict) else {}
        if metrics.get("lateral_dominated") is True or crossing.get("detected") is True:
            return True
    return False


def _lower_body_rotation_channels(channels: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None) -> list[str]:
    if not channels:
        return [f"{role}_rotation" for role in sorted(_LOWER_BODY_ROLES)]
    selected: list[str] = []
    for channel in channels:
        if not isinstance(channel, dict):
            continue
        role = str(channel.get("role") or "").strip()
        path = str(channel.get("path") or "").strip()
        if role not in _LOWER_BODY_ROLES or path != "rotation":
            continue
        name = str(channel.get("name") or channel.get("target_joint") or f"{role}_rotation")
        selected.append(name)
    return selected


def _basis_payload(basis: dict[str, Any] | None) -> dict[str, Any]:
    raw = basis or {}
    status = str(raw.get("status") or "missing").strip().lower()
    if not basis:
        status = "missing"
    trusted = status in {"trusted", "asserted"}
    if status == "missing":
        reason = "basis/missing"
    elif status == "inferred":
        reason = "basis/inferred_not_authorized"
    elif trusted:
        reason = None
    else:
        reason = "basis/untrusted_status"
    return {
        "source": raw.get("source"),
        "status": status,
        "forward": _normalized_axis(raw.get("forward")),
        "up": _normalized_axis(raw.get("up")),
        "handedness": _normalized_handedness(raw.get("handedness")),
        "trusted_for_correction": trusted,
        "reason": reason,
    }


def _validation_payload(kind: str, evidence: dict[str, Any] | None) -> dict[str, Any]:
    if not evidence:
        return {"status": "missing", "reasons": [f"{kind}/missing"], "evidence": {}}

    raw_status = str(evidence.get("status") or "not_evaluated").strip().lower()
    summary = _evidence_summary(evidence)
    reasons = _reason_strings(evidence.get("reasons"))

    if kind == "mirror":
        structural_reasons = _mirror_structural_reasons(summary)
    else:
        structural_reasons = _chain_structural_reasons(summary)
    reasons.extend(reason for reason in structural_reasons if reason not in reasons)

    if raw_status == "passed" and not reasons:
        status = "passed"
    elif raw_status == "missing":
        status = "missing"
        reasons.append(f"{kind}/missing")
    elif raw_status == "not_evaluated":
        status = "not_evaluated"
        reasons.append(f"{kind}/not_evaluated")
    else:
        status = "failed"
        reasons.append(f"{kind}/failed")

    return {"status": status, "reasons": _dedupe(reasons), "evidence": summary}


def _mirror_structural_reasons(summary: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if not summary["left_chain_roles"] or not summary["right_chain_roles"]:
        reasons.append("roles/missing_leg_chains")
    signs = summary["rest_side_signs"]
    if signs.get("left") is None or signs.get("right") is None:
        reasons.append("mirror/rest_sides_missing")
    elif not (signs["left"] < 0 and signs["right"] > 0):
        reasons.append("mirror/rest_sides_crossed")
    return reasons


def _chain_structural_reasons(summary: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if not summary["left_chain_roles"] or not summary["right_chain_roles"]:
        reasons.append("roles/missing_leg_chains")
    if summary.get("ordered") is False:
        reasons.append("chain/not_ordered")
    if summary.get("feet_present") is False or summary.get("toes_present") is False:
        reasons.append("roles/missing_feet_or_toes")
    return reasons


def _evidence_summary(evidence: dict[str, Any]) -> dict[str, Any]:
    return {
        "left_chain_roles": _string_list(evidence.get("left_chain_roles")),
        "right_chain_roles": _string_list(evidence.get("right_chain_roles")),
        "rest_side_signs": _rest_side_signs(evidence.get("rest_side_signs")),
        "symmetry_score": _optional_clean_float(evidence.get("symmetry_score")),
        "ordered": _optional_bool(evidence.get("ordered")),
        "feet_present": _optional_bool(evidence.get("feet_present")),
        "toes_present": _optional_bool(evidence.get("toes_present")),
    }


def _correction_eligibility(
    *,
    status: str,
    basis_payload: dict[str, Any],
    mirror_validation: dict[str, Any],
    chain_validation: dict[str, Any],
    diagnostic_reasons: list[dict[str, Any]],
) -> dict[str, Any]:
    reasons: list[str] = []
    basis_reason = basis_payload.get("reason")
    if not basis_payload.get("trusted_for_correction") and basis_reason:
        reasons.append(str(basis_reason))
    reasons.extend(_gate_reason("mirror", mirror_validation))
    reasons.extend(_gate_reason("chain", chain_validation))
    if status != "evaluated":
        reasons.extend(_reason_strings(diagnostic_reasons))
    return {
        "eligible": not reasons,
        "policy": "trusted_basis_and_passing_chain_mirror_required",
        "reasons": _dedupe(reasons),
    }


def _gate_reason(kind: str, validation: dict[str, Any]) -> list[str]:
    status = validation.get("status")
    if status == "passed":
        return []
    if status == "missing":
        return [f"{kind}/missing"]
    if status == "failed":
        return [f"{kind}/failed", *_reason_strings(validation.get("reasons"))]
    return [f"{kind}/not_evaluated", *_reason_strings(validation.get("reasons"))]


def _correction_payload() -> dict[str, Any]:
    return {
        "status": "skipped",
        "applied": False,
        "applied_channels": [],
        "recommended_channels": [],
        "reason": "correction/not_integrated_phase_2",
    }


def _normalized_axis(value: Any) -> str | None:
    if value is None:
        return None
    axis = str(value).strip().upper()
    return axis if axis in {"+X", "-X", "+Y", "-Y", "+Z", "-Z"} else None


def _normalized_handedness(value: Any) -> str | None:
    if value is None:
        return None
    handedness = str(value).strip().lower()
    return handedness if handedness in {"left", "right"} else None


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [str(item) for item in value if str(item)]


def _rest_side_signs(value: Any) -> dict[str, int | None]:
    if not isinstance(value, dict):
        return {"left": None, "right": None}
    return {"left": _side_sign(value.get("left")), "right": _side_sign(value.get("right"))}


def _side_sign(value: Any) -> int | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number < 0.0:
        return -1
    if number > 0.0:
        return 1
    return 0


def _optional_clean_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return _clean_float(value)
    except (TypeError, ValueError):
        return None


def _optional_bool(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _reason_strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    reasons: list[str] = []
    for item in value:
        if isinstance(item, dict):
            code = item.get("code")
            if code:
                reasons.append(str(code))
        elif item:
            reasons.append(str(item))
    return reasons


def _dedupe(values: list[str]) -> list[str]:
    deduped: list[str] = []
    for value in values:
        if value and value not in deduped:
            deduped.append(value)
    return deduped


def _coerce_positions(positions: Any) -> list[tuple[float, float, float]]:
    coerced: list[tuple[float, float, float]] = []
    if positions is None:
        return coerced
    for position in positions:
        vector = _vec3(position)
        if vector is not None:
            coerced.append(vector)
    return coerced


def _vec3(value: Any) -> tuple[float, float, float] | None:
    try:
        if len(value) < 3:
            return None
        vector = (float(value[0]), float(value[1]), float(value[2]))
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(component) for component in vector):
        return None
    return vector


def _axis_metrics(positions: list[tuple[float, float, float]], thresholds: dict[str, Any]) -> dict[str, Any]:
    start = positions[0]
    end = positions[-1]
    dx = end[0] - start[0]
    dz = end[2] - start[2]
    abs_x = abs(dx)
    abs_z = abs(dz)
    signed_axis = "+Z" if abs_z >= abs_x and dz >= 0.0 else "-Z" if abs_z >= abs_x else "+X" if dx >= 0.0 else "-X"
    forward_range = _range([position[2] for position in positions])
    lateral_range = _range([position[0] for position in positions])
    return {
        "signed_axis": signed_axis,
        "magnitude": _clean_float(math.hypot(dx, dz)),
        "forward_range": forward_range,
        "lateral_range": lateral_range,
        "lateral_drift_ratio": _ratio(lateral_range, forward_range, thresholds),
        "delta": [_clean_float(dx), 0.0, _clean_float(dz)],
    }


def _root_hips_axis(root_axis: dict[str, Any], hips_axis: dict[str, Any], thresholds: dict[str, Any]) -> dict[str, Any]:
    aligned = (
        root_axis["signed_axis"] == "+Z"
        and hips_axis["signed_axis"] == "+Z"
        and root_axis["lateral_drift_ratio"] <= thresholds["root_lateral_drift_ratio_max"]
        and hips_axis["lateral_drift_ratio"] <= thresholds["root_lateral_drift_ratio_max"]
    )
    return {
        "signed_axis": root_axis["signed_axis"],
        "magnitude": root_axis["magnitude"],
        "forward_range": root_axis["forward_range"],
        "lateral_range": root_axis["lateral_range"],
        "lateral_drift_ratio": root_axis["lateral_drift_ratio"],
        "hips_signed_axis": hips_axis["signed_axis"],
        "hips_forward_range": hips_axis["forward_range"],
        "hips_lateral_drift_ratio": hips_axis["lateral_drift_ratio"],
        "aligned": aligned,
    }


def _foot_metrics(
    side: str,
    foot_positions: list[tuple[float, float, float]],
    hips_positions: list[tuple[float, float, float]],
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    relative = [_subtract(foot, hips) for foot, hips in zip(foot_positions, hips_positions)]
    lateral_values = [position[0] for position in relative]
    forward_values = [position[2] for position in relative]
    lateral_range = _range(lateral_values)
    forward_range = _range(forward_values)
    lateral_forward_ratio = _ratio(lateral_range, forward_range, thresholds)
    expected_sign = -1 if side == "left" else 1
    wrong_indices = [index for index, lateral in enumerate(lateral_values) if lateral * expected_sign < 0.0]
    wrong_side_fraction = _clean_float(len(wrong_indices) / len(lateral_values)) if lateral_values else 0.0
    crossings = _midline_crossing(lateral_values, wrong_indices, wrong_side_fraction, thresholds)
    absolute_advance = _axis_metrics(foot_positions, thresholds)
    lateral_dominated = lateral_forward_ratio >= thresholds["lateral_dominance_ratio_min"]

    return {
        "sample_count": len(relative),
        "lateral_range": lateral_range,
        "forward_range": forward_range,
        "lateral_forward_ratio": lateral_forward_ratio,
        "leg_plane_ratio": lateral_forward_ratio,
        "swing_plane_ratio": lateral_forward_ratio,
        "lateral_dominated": lateral_dominated,
        "wrong_side_fraction": wrong_side_fraction,
        "midline_crossing": crossings,
        "absolute_advance": {
            "signed_axis": absolute_advance["signed_axis"],
            "magnitude": absolute_advance["magnitude"],
            "forward_range": absolute_advance["forward_range"],
        },
    }


def _midline_crossing(
    lateral_values: list[float],
    wrong_indices: list[int],
    wrong_side_fraction: float,
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    sign_changes = 0
    previous_sign = _sign(lateral_values[0]) if lateral_values else 0
    for value in lateral_values[1:]:
        current_sign = _sign(value)
        if previous_sign and current_sign and current_sign != previous_sign:
            sign_changes += 1
        if current_sign:
            previous_sign = current_sign
    detected = wrong_side_fraction >= thresholds["midline_wrong_side_fraction_min"] and sign_changes > 0
    return {
        "detected": detected,
        "count": int(sign_changes),
        "wrong_side_frame_count": len(wrong_indices),
        "frame_range": [wrong_indices[0], wrong_indices[-1]] if wrong_indices else None,
    }


def _classification(root_hips_axis: dict[str, Any], lateral_dominated_feet: list[str], crossing_feet: list[str]) -> str:
    if lateral_dominated_feet:
        return "lateral_dominated_limb_swing"
    if crossing_feet:
        return "ambiguous_limb_swing"
    if root_hips_axis["aligned"]:
        return "healthy_limb_swing"
    return "ambiguous_limb_swing"


def _confidence(root_hips_axis: dict[str, Any], feet: dict[str, dict[str, Any]]) -> float:
    if not root_hips_axis["aligned"]:
        return 0.5
    if any(foot["lateral_dominated"] or foot["midline_crossing"]["detected"] for foot in feet.values()):
        return 0.9
    return 0.95


def _subtract(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _range(values: list[float]) -> float:
    if not values:
        return 0.0
    return _clean_float(max(values) - min(values))


def _ratio(numerator: float, denominator: float, thresholds: dict[str, Any]) -> float:
    epsilon = float(thresholds["forward_epsilon"])
    if abs(denominator) <= epsilon:
        return _clean_float(numerator / epsilon) if numerator > epsilon else 0.0
    return _clean_float(numerator / denominator)


def _sign(value: float) -> int:
    if value > 0.0:
        return 1
    if value < 0.0:
        return -1
    return 0


def _clean_float(value: Any) -> float:
    number = float(value)
    if not math.isfinite(number):
        return 0.0
    return round(number, 6)
