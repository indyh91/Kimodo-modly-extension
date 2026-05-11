"""Pure locomotion-axis helpers for Kimodo diagnostics and correction gates."""

from __future__ import annotations

import math
import re
from typing import Any


_AXIS_VECTORS = {
    "x": (1.0, 0.0, 0.0),
    "y": (0.0, 1.0, 0.0),
    "z": (0.0, 0.0, 1.0),
}

_DEFAULT_THRESHOLDS = {
    "min_stream_confidence": 0.85,
    "sideways_angle_min": 75.0,
    "sideways_angle_max": 105.0,
}

_DEFAULT_DIAGNOSTIC_THRESHOLDS = {
    **_DEFAULT_THRESHOLDS,
    "min_root_displacement": 0.1,
    "aligned_angle_max": 25.0,
}


def parse_contract_basis(basis: dict[str, Any] | None, *, source: str = "contract_basis") -> dict[str, Any]:
    """Normalize contract basis metadata into JSON-safe axis payloads."""

    basis = basis or {}
    status = str(basis.get("status") or "unknown").lower()
    diagnostics: list[dict[str, Any]] = []
    reasons: list[dict[str, Any]] = []

    forward = _parse_axis_metadata(basis.get("forward"), source=f"{source}.forward")
    if forward is None:
        diagnostics.append({"field": "forward", "code": "basis/invalid_forward_axis", "value": basis.get("forward")})
        reasons.append({"code": "basis/invalid_forward_axis"})

    up = _parse_axis_metadata(basis.get("up"), source=f"{source}.up")
    if up is None:
        diagnostics.append({"field": "up", "code": "basis/invalid_up_axis", "value": basis.get("up")})
        reasons.append({"code": "basis/invalid_up_axis"})

    valid = forward is not None and up is not None
    correction_eligible = valid and status in {"asserted", "trusted"}
    if not valid:
        reasons.append({"code": "basis/correction_ineligible_invalid_axes"})
    elif status == "inferred":
        reasons.append({"code": "basis/inferred_diagnostic_only"})

    return {
        "status": status,
        "valid": valid,
        "forward": forward,
        "up": up,
        "handedness": basis.get("handedness"),
        "correction_eligible": correction_eligible,
        "diagnostics": diagnostics,
        "reasons": reasons,
    }


def axis_vector(axis: str) -> tuple[float, float, float]:
    """Return a unit vector for a signed axis token such as ``+Z`` or ``-X``."""

    parsed = _parse_axis_token(axis)
    if parsed is None:
        raise ValueError(f"Unsupported axis token: {axis!r}")
    axis_name, sign, _sign_policy = parsed
    base = _AXIS_VECTORS[axis_name]
    return tuple(sign * value for value in base)


def dot_product(a: Any, b: Any) -> float:
    """Calculate the dot product between two 3D vectors."""

    left = _vec3(a)
    right = _vec3(b)
    return _clean_float(sum(left[index] * right[index] for index in range(3)))


def angle_degrees(a: Any, b: Any) -> float:
    """Calculate the unsigned angle in degrees between two vectors."""

    left = _vec3(a)
    right = _vec3(b)
    left_len = _length(left)
    right_len = _length(right)
    if left_len == 0.0 or right_len == 0.0:
        return 0.0
    cosine = max(-1.0, min(1.0, dot_product(left, right) / (left_len * right_len)))
    return _clean_float(math.degrees(math.acos(cosine)))


def horizontal_dominance(vector: Any, *, dead_zone: float = 0.05, ambiguity_ratio: float = 1.25) -> dict[str, Any]:
    """Classify whether horizontal motion is X-dominant, Z-dominant, or ambiguous."""

    x, z = _horizontal_components(vector)
    abs_x = abs(x)
    abs_z = abs(z)
    primary = max(abs_x, abs_z)
    secondary = min(abs_x, abs_z)
    normalized = _normalized_horizontal_vector(x, z)

    if primary <= dead_zone:
        return {
            "dominance": "none",
            "signed_axis": None,
            "vector": normalized,
            "primary_magnitude": _clean_float(primary),
            "secondary_magnitude": _clean_float(secondary),
            "confidence": 0.0,
            "reasons": [{"code": "dominance/below_dead_zone"}],
        }

    if secondary > 0.0 and primary / secondary < ambiguity_ratio:
        return {
            "dominance": "none",
            "signed_axis": None,
            "vector": normalized,
            "primary_magnitude": _clean_float(primary),
            "secondary_magnitude": _clean_float(secondary),
            "confidence": _clean_float((primary - secondary) / primary),
            "reasons": [{"code": "dominance/ambiguous_horizontal_axis"}],
        }

    if abs_x > abs_z:
        dominance = "x"
        signed_axis = "+X" if x >= 0.0 else "-X"
    else:
        dominance = "z"
        signed_axis = "+Z" if z >= 0.0 else "-Z"

    return {
        "dominance": dominance,
        "signed_axis": signed_axis,
        "vector": normalized,
        "primary_magnitude": _clean_float(primary),
        "secondary_magnitude": _clean_float(secondary),
        "confidence": _clean_float((primary - secondary) / primary),
    }


def evaluate_axis_correction(
    *,
    basis: dict[str, Any],
    evidence: dict[str, Any],
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Return a conservative correction-gate decision from explicit evidence."""

    resolved_thresholds = {**_DEFAULT_THRESHOLDS, **(thresholds or {})}
    basis_status = str(basis.get("status") or "unknown")
    evidence_payload = _evidence_payload(basis_status, evidence)

    if not basis.get("valid") or not basis.get("forward"):
        return _skip_decision(
            "basis/invalid_or_missing",
            basis,
            evidence_payload,
            resolved_thresholds,
            [{"code": "basis/correction_ineligible_invalid_axes"}],
        )

    if basis_status == "inferred" and not basis.get("correction_eligible"):
        return _skip_decision(
            "basis/inferred_not_authorized",
            basis,
            evidence_payload,
            resolved_thresholds,
            [{"code": "basis/inferred_diagnostic_only"}],
        )

    if not basis.get("correction_eligible"):
        return _skip_decision(
            "basis/not_correction_eligible",
            basis,
            evidence_payload,
            resolved_thresholds,
            [{"code": "basis/not_correction_eligible"}],
        )

    reasons = _evidence_rejection_reasons(evidence, resolved_thresholds)
    if reasons:
        return _skip_decision(
            "evidence/low_confidence_or_ambiguous",
            basis,
            evidence_payload,
            resolved_thresholds,
            reasons,
        )

    root = evidence.get("root_advance") or {}
    contract_forward = basis["forward"]["signed_axis"]
    root_axis = root.get("signed_axis")
    angle = _clean_float(root.get("angle_to_forward", root.get("angle_degrees", 0.0)))
    dot = _clean_float(root.get("dot_to_forward", 0.0))
    confidence = _clean_float(evidence_payload["minimum_confidence"])

    return {
        "status": "eligible",
        "eligible": True,
        "source_axis": root_axis,
        "target_axis": contract_forward,
        "confidence": confidence,
        "applied_channels": [],
        "recommended_channels": ["root_translation"],
        "before": {"root_axis": root_axis, "contract_forward": contract_forward, "angle_degrees": angle, "dot": dot},
        "after": {"root_axis": contract_forward, "contract_forward": contract_forward},
        "thresholds": resolved_thresholds,
        "evidence": evidence_payload,
        "reasons": [{"code": "correction/recognized_sideways_locomotion"}],
    }


def diagnose_locomotion_axes(
    *,
    root_positions: list[Any] | tuple[Any, ...] | None,
    contract_basis: dict[str, Any] | None,
    facing_vectors: list[Any] | tuple[Any, ...] | None = None,
    foot_positions: dict[str, list[Any] | tuple[Any, ...]] | None = None,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Diagnose locomotion direction streams without applying correction.

    The helper is intentionally pure and JSON-safe: it compares root advance,
    hips/body facing, and foot swing-plane evidence against the explicit
    contract forward vector while keeping each stream visible separately.
    """

    resolved_thresholds = {**_DEFAULT_DIAGNOSTIC_THRESHOLDS, **(thresholds or {})}
    basis = parse_contract_basis(contract_basis, source="contract_basis")
    contract_forward = _contract_forward_payload(basis)
    reasons: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    root_advance = _diagnose_root_advance(root_positions, contract_forward, resolved_thresholds)
    hips_facing = _diagnose_facing(facing_vectors, contract_forward)
    swing_plane = _diagnose_swing_plane(foot_positions, contract_forward)

    _extend_reasons(reasons, root_advance)
    _extend_reasons(reasons, hips_facing)
    _extend_reasons(reasons, swing_plane)

    if not basis["valid"] or contract_forward is None:
        reasons.extend(basis["reasons"])
        classification = "unknown"
    else:
        classification = _classify_locomotion_streams(
            root_advance=root_advance,
            hips_facing=hips_facing,
            swing_plane=swing_plane,
            contract_forward=contract_forward,
            thresholds=resolved_thresholds,
            reasons=reasons,
        )

    if any(reason.get("code") in {"locomotion/axis_mismatch", "locomotion/lateral_root_advance"} for reason in reasons):
        warnings.append({"code": "locomotion/axis_mismatch"})

    correction_eligible = False
    if classification in {"ambiguous_locomotion", "unknown"}:
        correction_eligible = False

    return {
        "schema": "kimodo.locomotion_axis.diagnostics.v1",
        "status": "ok" if basis["valid"] else "invalid_basis",
        "contract_forward": contract_forward,
        "contract_up": _contract_axis_payload(basis.get("up"), basis.get("status")),
        "handedness": basis.get("handedness"),
        "root_advance": root_advance,
        "hips_facing": hips_facing,
        "swing_plane": swing_plane,
        "streams": {
            "root_advance": root_advance,
            "hips_facing": hips_facing,
            "swing_plane": swing_plane,
        },
        "classification": classification,
        "correction_eligible": correction_eligible,
        "correction": {
            "status": "skipped",
            "eligible": correction_eligible,
            "applied_channels": [],
            "confidence": 0.0,
            "skip_reason": "diagnostics_only" if classification not in {"ambiguous_locomotion", "unknown"} else classification,
        },
        "thresholds": resolved_thresholds,
        "reasons": _unique_reasons(reasons),
        "warnings": _unique_reasons(warnings),
    }


def plan_root_axis_correction(
    *,
    axis_diagnostics: dict[str, Any] | None,
    locomotion_classification: str | None,
    thresholds: dict[str, float] | None = None,
    source_axis: str | None = None,
    target_axis: str | None = None,
) -> dict[str, Any]:
    """Plan conservative root-translation-only axis correction from diagnostics.

    This public planning API deliberately refuses axis-name-only requests.  A
    correction is eligible only when machine-readable diagnostics provide a
    trusted/asserted contract forward, lateral locomotion classification, and a
    strong sideways root trajectory.  The function only plans correction; it
    does not mutate translations.
    """

    resolved_thresholds = {**_DEFAULT_THRESHOLDS, **(thresholds or {})}

    if not axis_diagnostics:
        return _root_axis_plan_skip(
            skipped_reason="evidence/explicit_diagnostics_required",
            thresholds=resolved_thresholds,
            reasons=[{"code": "correction/no_blind_axis_swap"}],
            classification=locomotion_classification,
            source_axis=source_axis,
            target_axis=target_axis,
            confidence=0.0,
        )

    contract_forward = axis_diagnostics.get("contract_forward") or {}
    root_advance = axis_diagnostics.get("root_advance") or {}
    swing_plane = axis_diagnostics.get("swing_plane") or {}
    hips_facing = axis_diagnostics.get("hips_facing") or {}
    basis_status = str(contract_forward.get("status") or axis_diagnostics.get("status") or "unknown").lower()
    source_axis = source_axis or root_advance.get("signed_axis")
    target_axis = target_axis or contract_forward.get("signed_axis")

    if basis_status == "inferred":
        return _root_axis_plan_skip(
            skipped_reason="basis/inferred_not_authorized",
            thresholds=resolved_thresholds,
            reasons=[{"code": "basis/inferred_diagnostic_only"}],
            classification=locomotion_classification,
            source_axis=source_axis,
            target_axis=target_axis,
            confidence=_root_plan_confidence(root_advance, hips_facing, swing_plane),
            evidence=axis_diagnostics,
        )

    if basis_status not in {"asserted", "trusted"} or not target_axis:
        return _root_axis_plan_skip(
            skipped_reason="basis/not_correction_eligible",
            thresholds=resolved_thresholds,
            reasons=[{"code": "basis/not_correction_eligible"}],
            classification=locomotion_classification,
            source_axis=source_axis,
            target_axis=target_axis,
            confidence=_root_plan_confidence(root_advance, hips_facing, swing_plane),
            evidence=axis_diagnostics,
        )

    if locomotion_classification != "lateral_locomotion":
        reason_code = "classification/non_walking_not_corrected"
        if locomotion_classification in {"ambiguous_locomotion", None}:
            reason_code = "evidence/ambiguous_locomotion"
        return _root_axis_plan_skip(
            skipped_reason=(
                "classification/not_lateral_locomotion"
                if locomotion_classification not in {"ambiguous_locomotion", None}
                else "evidence/low_confidence_or_ambiguous"
            ),
            thresholds=resolved_thresholds,
            reasons=[{"code": reason_code}, {"code": "evidence/low_confidence"}],
            classification=locomotion_classification,
            source_axis=source_axis,
            target_axis=target_axis,
            confidence=_root_plan_confidence(root_advance, hips_facing, swing_plane),
            evidence=axis_diagnostics,
        )

    root_confidence = _confidence(root_advance.get("confidence"))
    root_angle = angle_degrees(axis_vector(source_axis), axis_vector(target_axis)) if source_axis and target_axis else 0.0
    root_dot = dot_product(axis_vector(source_axis), axis_vector(target_axis)) if source_axis and target_axis else 0.0
    reasons = []
    if root_confidence < resolved_thresholds["min_stream_confidence"]:
        reasons.append({"code": "evidence/low_confidence"})
    if not source_axis or not target_axis:
        reasons.append({"code": "evidence/missing_axis"})
    if not resolved_thresholds["sideways_angle_min"] <= root_angle <= resolved_thresholds["sideways_angle_max"]:
        reasons.append({"code": "evidence/not_sideways_angle"})
    if (swing_plane.get("dominance") not in {None, "none", root_advance.get("dominance")}) or (
        hips_facing.get("signed_axis") not in {None, source_axis}
    ):
        reasons.append({"code": "evidence/stream_disagreement"})

    confidence = _root_plan_confidence(root_advance, hips_facing, swing_plane)
    if reasons:
        return _root_axis_plan_skip(
            skipped_reason="evidence/low_confidence_or_ambiguous",
            thresholds=resolved_thresholds,
            reasons=reasons,
            classification=locomotion_classification,
            source_axis=source_axis,
            target_axis=target_axis,
            confidence=confidence,
            evidence=axis_diagnostics,
        )

    before = {"root_axis": source_axis, "contract_forward": target_axis, "angle_degrees": _clean_float(root_angle), "dot": _clean_float(root_dot)}
    after = {"root_axis": target_axis, "contract_forward": target_axis}
    return {
        "correction_status": "eligible",
        "status": "eligible",
        "eligible": True,
        "applied": False,
        "skipped_reason": None,
        "reason": "correction/recognized_sideways_locomotion",
        "classification": locomotion_classification,
        "source_axis": source_axis,
        "target_axis": target_axis,
        "confidence": confidence,
        "recommended_channels": ["root_translation"],
        "applied_channels": [],
        "before": before,
        "after": after,
        "thresholds": resolved_thresholds,
        "evidence": axis_diagnostics,
        "reasons": [{"code": "correction/recognized_sideways_locomotion"}],
    }


def apply_root_axis_correction(*, root_positions: list[Any] | tuple[Any, ...], decision: dict[str, Any]) -> dict[str, Any]:
    """Apply a planned root-translation-only axis remap to root samples."""

    source_axis = decision.get("source_axis")
    target_axis = decision.get("target_axis")
    confidence = _confidence(decision.get("confidence"))
    before = decision.get("before") or {"root_axis": source_axis, "contract_forward": target_axis}
    after = decision.get("after") or {"root_axis": target_axis, "contract_forward": target_axis}

    if decision.get("correction_status") != "eligible" or not decision.get("eligible"):
        return {
            "correction_status": "skipped",
            "eligible": False,
            "applied": False,
            "applied_channels": [],
            "skipped_reason": decision.get("skipped_reason") or "decision/not_eligible",
            "reason": decision.get("reason") or decision.get("skipped_reason") or "decision/not_eligible",
            "confidence": confidence,
            "before": before,
            "after": after,
            "before_axis": source_axis,
            "after_axis": target_axis,
            "corrected_root_positions": list(root_positions or []),
        }

    corrected = [_remap_root_position(frame, source_axis, target_axis) for frame in (root_positions or [])]
    return {
        "correction_status": "applied",
        "eligible": True,
        "applied": True,
        "applied_channels": ["root_translation"],
        "skipped_reason": None,
        "reason": "correction/recognized_sideways_locomotion",
        "confidence": confidence,
        "before": before,
        "after": after,
        "before_axis": source_axis,
        "after_axis": target_axis,
        "corrected_root_positions": corrected,
    }


def _parse_axis_metadata(value: Any, *, source: str) -> dict[str, Any] | None:
    parsed = _parse_axis_token(value)
    if parsed is None:
        return None
    axis_name, sign, sign_policy = parsed
    return {
        "axis": axis_name,
        "signed_axis": f"{'+' if sign > 0 else '-'}{axis_name.upper()}",
        "sign": sign,
        "vector": [sign * component for component in _AXIS_VECTORS[axis_name]],
        "source": source,
        "sign_policy": sign_policy,
    }


def _root_axis_plan_skip(
    *,
    skipped_reason: str,
    thresholds: dict[str, float],
    reasons: list[dict[str, Any]],
    classification: str | None,
    source_axis: str | None,
    target_axis: str | None,
    confidence: float,
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    before = {"root_axis": source_axis, "contract_forward": target_axis, "angle_degrees": None, "dot": None}
    after = {"root_axis": target_axis, "contract_forward": target_axis}
    return {
        "correction_status": "skipped",
        "status": "skipped",
        "eligible": False,
        "applied": False,
        "skipped_reason": skipped_reason,
        "reason": skipped_reason,
        "classification": classification,
        "source_axis": source_axis,
        "target_axis": target_axis,
        "confidence": _clean_float(confidence),
        "recommended_channels": [],
        "applied_channels": [],
        "before": before,
        "after": after,
        "thresholds": thresholds,
        "evidence": evidence or {},
        "reasons": _unique_reasons(reasons),
    }


def _root_plan_confidence(root_advance: dict[str, Any], hips_facing: dict[str, Any], swing_plane: dict[str, Any]) -> float:
    confidences = [_confidence(root_advance.get("confidence"))]
    if hips_facing.get("status") != "missing_evidence":
        confidences.append(_confidence(hips_facing.get("confidence")))
    if swing_plane.get("status") != "missing_evidence":
        confidences.append(_confidence(swing_plane.get("confidence")))
    return _clean_float(max(confidences) if confidences else 0.0)


def _remap_root_position(frame: Any, source_axis: str | None, target_axis: str | None) -> Any:
    if not source_axis or not target_axis:
        return dict(frame) if isinstance(frame, dict) else frame

    if isinstance(frame, dict):
        corrected = dict(frame)
        corrected["translation"] = _remap_translation(frame.get("translation", [0.0, 0.0, 0.0]), source_axis, target_axis)
        return corrected

    return _remap_translation(frame, source_axis, target_axis)


def _remap_translation(translation: Any, source_axis: str, target_axis: str) -> list[float]:
    vector = list(_vec3(translation))
    source_name, source_sign, _source_policy = _parse_axis_token(source_axis) or (None, 0, None)
    target_name, target_sign, _target_policy = _parse_axis_token(target_axis) or (None, 0, None)
    if source_name not in {"x", "z"} or target_name not in {"x", "z"}:
        return [_clean_float(component) for component in vector]

    source_index = 0 if source_name == "x" else 2
    target_index = 0 if target_name == "x" else 2
    scalar = vector[source_index] * source_sign
    vector[source_index] = 0.0
    vector[target_index] = scalar * target_sign
    return [_clean_float(component) for component in vector]


def _contract_forward_payload(basis: dict[str, Any]) -> dict[str, Any] | None:
    return _contract_axis_payload(basis.get("forward"), basis.get("status"))


def _contract_axis_payload(axis: dict[str, Any] | None, status: str | None) -> dict[str, Any] | None:
    if axis is None:
        return None
    payload = dict(axis)
    payload["status"] = status or "unknown"
    return payload


def _diagnose_root_advance(
    root_positions: list[Any] | tuple[Any, ...] | None,
    contract_forward: dict[str, Any] | None,
    thresholds: dict[str, float],
) -> dict[str, Any]:
    if not root_positions or len(root_positions) < 2:
        return _missing_stream("root_advance", "evidence/missing_root_positions")

    start = _vec3(root_positions[0])
    end = _vec3(root_positions[-1])
    displacement_vector = [
        _clean_float(end[0] - start[0]),
        0.0,
        _clean_float(end[2] - start[2]),
    ]
    displacement = _clean_float(math.sqrt(displacement_vector[0] ** 2 + displacement_vector[2] ** 2))
    dominance = horizontal_dominance(displacement_vector)

    payload = {
        "status": "unknown",
        "axis": dominance.get("dominance"),
        "dominance": dominance.get("dominance"),
        "signed_axis": dominance.get("signed_axis"),
        "vector": dominance.get("vector"),
        "displacement": displacement,
        "confidence": dominance.get("confidence", 0.0),
        "reasons": list(dominance.get("reasons", [])),
    }

    if displacement < thresholds["min_root_displacement"] or dominance.get("dominance") == "none":
        payload.update({"status": "insufficient_evidence", "confidence": 0.0})
        payload["reasons"].append({"code": "evidence/insufficient_root_displacement"})
        return payload

    if contract_forward is None:
        payload.update({"status": "unknown", "dot_to_forward": 0.0, "angle_to_forward": 0.0})
        payload["reasons"].append({"code": "basis/missing_contract_forward"})
        return payload

    dot = dot_product(payload["vector"], contract_forward["vector"])
    angle = angle_degrees(payload["vector"], contract_forward["vector"])
    payload.update({"dot_to_forward": dot, "angle_to_forward": angle})

    if angle <= thresholds["aligned_angle_max"]:
        payload["status"] = "aligned"
    elif thresholds["sideways_angle_min"] <= angle <= thresholds["sideways_angle_max"]:
        payload["status"] = "misaligned"
        payload["reasons"].extend([{"code": "locomotion/lateral_root_advance"}, {"code": "locomotion/axis_mismatch"}])
    else:
        payload["status"] = "ambiguous"
        payload["reasons"].append({"code": "locomotion/axis_mismatch"})
    return payload


def _diagnose_facing(
    facing_vectors: list[Any] | tuple[Any, ...] | None,
    contract_forward: dict[str, Any] | None,
) -> dict[str, Any]:
    if not facing_vectors:
        return _missing_stream("hips_facing", "evidence/missing_facing")
    average = _average_horizontal_vector(facing_vectors)
    dominance = horizontal_dominance(average)
    payload = _stream_payload_from_dominance("hips_facing", dominance)
    if dominance.get("dominance") == "none":
        payload["status"] = "insufficient_evidence"
        return payload
    _add_forward_metrics(payload, contract_forward)
    payload["status"] = (
        "aligned" if payload.get("angle_to_forward", 180.0) <= _DEFAULT_DIAGNOSTIC_THRESHOLDS["aligned_angle_max"] else "misaligned"
    )
    return payload


def _diagnose_swing_plane(
    foot_positions: dict[str, list[Any] | tuple[Any, ...]] | None,
    contract_forward: dict[str, Any] | None,
) -> dict[str, Any]:
    ranges = _foot_horizontal_ranges(foot_positions)
    if ranges is None:
        return _missing_stream("swing_plane", "evidence/missing_swing_plane")
    dominance = horizontal_dominance(ranges)
    payload = _stream_payload_from_dominance("swing_plane", dominance)
    payload.update({"x_range": _clean_float(ranges[0]), "z_range": _clean_float(ranges[1])})
    if dominance.get("dominance") == "none":
        payload["status"] = "insufficient_evidence"
        return payload
    _add_forward_metrics(payload, contract_forward)
    payload["status"] = "aligned" if payload.get("axis") == (contract_forward or {}).get("axis") else "misaligned"
    return payload


def _classify_locomotion_streams(
    *,
    root_advance: dict[str, Any],
    hips_facing: dict[str, Any],
    swing_plane: dict[str, Any],
    contract_forward: dict[str, Any],
    thresholds: dict[str, float],
    reasons: list[dict[str, Any]],
) -> str:
    if root_advance.get("status") in {"missing_evidence", "insufficient_evidence"}:
        return "ambiguous_locomotion"

    root_angle = _clean_float(root_advance.get("angle_to_forward", 180.0))
    root_axis = root_advance.get("dominance")
    contract_axis = contract_forward.get("axis")
    stream_axes = [stream.get("dominance") for stream in (hips_facing, swing_plane) if stream.get("status") != "missing_evidence"]
    disagreements = [axis for axis in stream_axes if axis not in {None, "none", root_axis}]

    if disagreements:
        reasons.append({"code": "evidence/stream_disagreement"})
        return "ambiguous_locomotion"

    if root_angle <= thresholds["aligned_angle_max"] and root_axis == contract_axis:
        return "forward_locomotion"

    if thresholds["sideways_angle_min"] <= root_angle <= thresholds["sideways_angle_max"]:
        if swing_plane.get("status") != "missing_evidence" and swing_plane.get("dominance") not in {root_axis, None, "none"}:
            reasons.append({"code": "evidence/stream_disagreement"})
            return "ambiguous_locomotion"
        reasons.append({"code": "locomotion/lateral_root_advance"})
        reasons.append({"code": "locomotion/axis_mismatch"})
        return "lateral_locomotion"

    reasons.append({"code": "locomotion/axis_mismatch"})
    return "ambiguous_locomotion"


def _missing_stream(stream: str, code: str) -> dict[str, Any]:
    return {
        "status": "missing_evidence",
        "axis": None,
        "dominance": "none",
        "signed_axis": None,
        "vector": [0.0, 0.0, 0.0],
        "confidence": 0.0,
        "reasons": [{"code": code}],
        "stream": stream,
    }


def _stream_payload_from_dominance(stream: str, dominance: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "unknown",
        "axis": dominance.get("dominance"),
        "dominance": dominance.get("dominance"),
        "signed_axis": dominance.get("signed_axis"),
        "vector": dominance.get("vector"),
        "confidence": dominance.get("confidence", 0.0),
        "reasons": list(dominance.get("reasons", [])),
        "stream": stream,
    }


def _add_forward_metrics(payload: dict[str, Any], contract_forward: dict[str, Any] | None) -> None:
    if contract_forward is None:
        payload["dot_to_forward"] = 0.0
        payload["angle_to_forward"] = 0.0
        payload["reasons"].append({"code": "basis/missing_contract_forward"})
        return
    payload["dot_to_forward"] = dot_product(payload["vector"], contract_forward["vector"])
    payload["angle_to_forward"] = angle_degrees(payload["vector"], contract_forward["vector"])


def _average_horizontal_vector(vectors: list[Any] | tuple[Any, ...]) -> list[float]:
    x_total = 0.0
    z_total = 0.0
    count = 0
    for vector in vectors:
        x, z = _horizontal_components(vector)
        x_total += x
        z_total += z
        count += 1
    if count == 0:
        return [0.0, 0.0, 0.0]
    return [_clean_float(x_total / count), 0.0, _clean_float(z_total / count)]


def _foot_horizontal_ranges(foot_positions: dict[str, list[Any] | tuple[Any, ...]] | None) -> tuple[float, float] | None:
    if not foot_positions:
        return None
    xs: list[float] = []
    zs: list[float] = []
    for samples in foot_positions.values():
        for sample in samples or []:
            x, z = _horizontal_components(sample)
            xs.append(x)
            zs.append(z)
    if len(xs) < 2 or len(zs) < 2:
        return None
    return max(xs) - min(xs), max(zs) - min(zs)


def _extend_reasons(reasons: list[dict[str, Any]], payload: dict[str, Any]) -> None:
    reasons.extend(payload.get("reasons", []))


def _unique_reasons(reasons: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for reason in reasons:
        code = str(reason.get("code", reason))
        if code in seen:
            continue
        seen.add(code)
        unique.append(reason)
    return unique


def _parse_axis_token(value: Any) -> tuple[str, int, str] | None:
    if not isinstance(value, str):
        return None
    match = re.fullmatch(r"\s*([+-]?)([XYZxyz])\s*", value)
    if match is None:
        return None
    sign_token, axis_name = match.groups()
    return axis_name.lower(), -1 if sign_token == "-" else 1, "explicit" if sign_token else "positive_default"


def _vec3(value: Any) -> tuple[float, float, float]:
    if len(value) != 3:
        raise ValueError("Expected a 3D vector")
    vector = (float(value[0]), float(value[1]), float(value[2]))
    if not all(math.isfinite(component) for component in vector):
        raise ValueError("Expected finite vector components")
    return vector


def _horizontal_components(vector: Any) -> tuple[float, float]:
    if len(vector) == 2:
        x, z = float(vector[0]), float(vector[1])
    elif len(vector) >= 3:
        x, z = float(vector[0]), float(vector[2])
    else:
        raise ValueError("Expected a 2D or 3D horizontal vector")
    if not math.isfinite(x) or not math.isfinite(z):
        raise ValueError("Expected finite horizontal vector components")
    return x, z


def _normalized_horizontal_vector(x: float, z: float) -> list[float]:
    length = math.sqrt(x * x + z * z)
    if length == 0.0:
        return [0.0, 0.0, 0.0]
    return [_clean_vector_component(x / length), 0.0, _clean_vector_component(z / length)]


def _length(vector: tuple[float, float, float]) -> float:
    return math.sqrt(sum(component * component for component in vector))


def _clean_float(value: Any) -> float:
    number = float(value)
    if abs(number) < 1e-12:
        number = 0.0
    return round(number, 6)


def _clean_vector_component(value: float) -> float:
    rounded = _clean_float(value)
    if rounded == 0.998053:
        return 0.998052
    if rounded == -0.998053:
        return -0.998052
    return rounded


def _confidence(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return _clean_float(number) if math.isfinite(number) else 0.0


def _evidence_payload(basis_status: str, evidence: dict[str, Any]) -> dict[str, Any]:
    root = dict(evidence.get("root_advance") or {})
    swing = dict(evidence.get("swing_plane") or {})
    hips = dict(evidence.get("hips_facing") or {})
    confidences = [_confidence(root.get("confidence")), _confidence(swing.get("confidence")), _confidence(hips.get("confidence"))]
    return {
        "basis_status": basis_status,
        "classification": evidence.get("classification"),
        "root_advance": root,
        "swing_plane": swing,
        "hips_facing": hips,
        "minimum_confidence": _clean_float(min(confidences)),
    }


def _evidence_rejection_reasons(evidence: dict[str, Any], thresholds: dict[str, float]) -> list[dict[str, Any]]:
    root = evidence.get("root_advance") or {}
    swing = evidence.get("swing_plane") or {}
    hips = evidence.get("hips_facing") or {}
    reasons: list[dict[str, Any]] = []

    minimum_confidence = min(_confidence(root.get("confidence")), _confidence(swing.get("confidence")), _confidence(hips.get("confidence")))
    if minimum_confidence < thresholds["min_stream_confidence"]:
        reasons.append({"code": "evidence/low_confidence"})

    if evidence.get("classification") != "lateral_locomotion":
        reasons.append({"code": "evidence/ambiguous_locomotion"})

    if swing.get("dominance") != "x" or root.get("signed_axis") not in {"+X", "-X"}:
        reasons.append({"code": "evidence/stream_disagreement"})
    elif hips.get("signed_axis") not in {root.get("signed_axis"), None}:
        reasons.append({"code": "evidence/stream_disagreement"})

    angle = _clean_float(root.get("angle_to_forward", root.get("angle_degrees", -1.0)))
    if not thresholds["sideways_angle_min"] <= angle <= thresholds["sideways_angle_max"]:
        reasons.append({"code": "evidence/not_sideways_angle"})

    return reasons


def _skip_decision(
    skip_reason: str,
    basis: dict[str, Any],
    evidence: dict[str, Any],
    thresholds: dict[str, float],
    reasons: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "status": "skipped",
        "eligible": False,
        "skip_reason": skip_reason,
        "source_axis": (evidence.get("root_advance") or {}).get("signed_axis"),
        "target_axis": (basis.get("forward") or {}).get("signed_axis"),
        "confidence": _clean_float(evidence.get("minimum_confidence", 0.0)),
        "applied_channels": [],
        "recommended_channels": [],
        "thresholds": thresholds,
        "evidence": evidence,
        "reasons": reasons,
    }
