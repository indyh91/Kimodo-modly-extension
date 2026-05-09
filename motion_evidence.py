"""Motion-evidence classifier for stabilization decisions.

The classifier is intentionally kinematics-first: prompt text is preserved as a
weak diagnostic hint, never as the source of truth when motion evidence exists.
"""

from __future__ import annotations

import math
from typing import Any


def classify_motion_evidence(
    *,
    root_samples: Any = None,
    foot_samples: Any = None,
    foot_contacts: Any = None,
    dynamic_intensity: Any = None,
    trusted_leg_chains: bool = False,
    prompt_hint: str | None = None,
) -> dict[str, Any]:
    root = _vec3_series(root_samples)
    feet = _foot_series(foot_samples)
    contacts = _contact_series(foot_contacts)
    frame_count = _frame_count(root, feet, contacts)
    prompt = prompt_hint if isinstance(prompt_hint, str) else None
    evidence = {
        "root_displacement": _clean_float(_path_displacement(root)),
        "foot_velocities": _foot_velocity_summary(feet),
        "contact_windows": _contact_summary(contacts),
        "stance_consistency": "unknown",
        "dynamic_intensity": _optional_clean(dynamic_intensity),
        "trusted_leg_chains": bool(trusted_leg_chains),
        "frame_count": frame_count,
    }
    if not trusted_leg_chains:
        return _result(
            "ineligible",
            0.0,
            ["missing_trusted_leg_chains"],
            evidence,
            prompt,
            warnings=["motion_evidence/missing_trusted_leg_chains"],
        )
    if frame_count < 2 or not root or not _has_both_feet(feet):
        return _result("unknown", 0.0, ["insufficient_frame_domain"], evidence, prompt)

    dynamic = _float_or_none(dynamic_intensity)
    root_displacement = evidence["root_displacement"]
    foot_velocity_max = evidence["foot_velocities"]["max"]
    contact_pattern = evidence["contact_windows"]["pattern"]
    stance = _stance_consistency(root_displacement, foot_velocity_max, contact_pattern, dynamic)
    evidence["stance_consistency"] = stance

    if dynamic is not None and dynamic >= 0.8:
        reasons = ["high_dynamic_intensity"]
        if stance == "unstable":
            reasons.append("unstable_contact_windows")
        return _with_prompt_conflict(_result("dynamic", 0.9, reasons, evidence, prompt), prompt)

    if root_displacement >= 0.35 and contact_pattern == "alternating" and foot_velocity_max >= 0.08:
        return _with_prompt_conflict(
            _result("walking", 0.85, ["root_displacement_locomotive", "alternating_foot_contacts", "foot_velocity_locomotive"], evidence, prompt),
            prompt,
        )

    if root_displacement <= 0.12 and foot_velocity_max <= 0.05 and stance == "stable" and (dynamic is None or dynamic <= 0.25):
        return _with_prompt_conflict(_result("gesture_like", 0.75, ["stable_root", "stable_feet", "low_dynamic_intensity"], evidence, prompt), prompt)

    if stance == "unstable" or foot_velocity_max >= 0.45:
        return _with_prompt_conflict(_result("dynamic", 0.7, ["unstable_contact_windows"], evidence, prompt), prompt)

    return _with_prompt_conflict(_result("unknown", 0.2, ["ambiguous_motion_evidence"], evidence, prompt), prompt)


def _result(
    motion_class: str,
    confidence: float,
    reasons: list[str],
    evidence: dict[str, Any],
    prompt_hint: str | None,
    *,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "class": motion_class,
        "confidence": _clean_float(confidence),
        "reasons": reasons,
        "evidence": evidence,
        "prompt_hint": prompt_hint,
        "conflicts": [],
        "warnings": warnings or [],
    }


def _with_prompt_conflict(result: dict[str, Any], prompt_hint: str | None) -> dict[str, Any]:
    hint_class = _prompt_hint_class(prompt_hint)
    if hint_class is not None and not _hint_matches(result["class"], hint_class):
        result = {**result, "conflicts": [*result.get("conflicts", []), "prompt_conflicts_with_kinematics"]}
    return result


def _prompt_hint_class(prompt_hint: str | None) -> str | None:
    text = str(prompt_hint or "").lower()
    if any(token in text for token in ("dance", "dancing", "spin", "jump")):
        return "dynamic"
    if any(token in text for token in ("walk", "walking", "run", "running", "locomotion")):
        return "walking"
    if any(token in text for token in ("gesture", "wave", "idle", "stand", "static", "raising", "raisin")):
        return "gesture_like"
    return None


def _hint_matches(class_name: str, hint_class: str) -> bool:
    if class_name == hint_class:
        return True
    if class_name == "locomotive" and hint_class == "walking":
        return True
    return False


def _vec3_series(values: Any) -> tuple[tuple[float, float, float], ...]:
    series = []
    for sample in values or ():
        try:
            if len(sample) < 3:
                continue
            point = (float(sample[0]), float(sample[1]), float(sample[2]))
        except (TypeError, ValueError):
            continue
        if all(math.isfinite(v) for v in point):
            series.append(point)
    return tuple(series)


def _foot_series(foot_samples: Any) -> dict[str, tuple[tuple[float, float, float], ...]]:
    if not isinstance(foot_samples, dict):
        return {}
    return {side: _vec3_series(foot_samples.get(side)) for side in ("left", "right")}


def _contact_series(foot_contacts: Any) -> tuple[tuple[float, float], ...]:
    rows = []
    for row in foot_contacts or ():
        try:
            if len(row) < 2:
                continue
            pair = (float(row[0]), float(row[1]))
        except (TypeError, ValueError):
            continue
        if all(math.isfinite(v) for v in pair):
            rows.append(pair)
    return tuple(rows)


def _frame_count(root: tuple[Any, ...], feet: dict[str, tuple[Any, ...]], contacts: tuple[Any, ...]) -> int:
    counts = [len(root), len(feet.get("left", ())), len(feet.get("right", ())), len(contacts)]
    positive = [count for count in counts if count > 0]
    return min(positive) if positive else 0


def _has_both_feet(feet: dict[str, tuple[Any, ...]]) -> bool:
    return len(feet.get("left", ())) >= 2 and len(feet.get("right", ())) >= 2


def _path_displacement(samples: tuple[tuple[float, float, float], ...]) -> float:
    if len(samples) < 2:
        return 0.0
    return _distance(samples[0], samples[-1])


def _foot_velocity_summary(feet: dict[str, tuple[tuple[float, float, float], ...]]) -> dict[str, Any]:
    velocities = []
    per_side: dict[str, list[float]] = {}
    for side, samples in feet.items():
        side_velocities = [_distance(samples[index - 1], samples[index]) for index in range(1, len(samples))]
        per_side[side] = [_clean_float(value) for value in side_velocities]
        velocities.extend(side_velocities)
    return {
        "max": _clean_float(max(velocities, default=0.0)),
        "mean": _clean_float(sum(velocities) / len(velocities)) if velocities else 0.0,
        "per_side": per_side,
    }


def _contact_summary(contacts: tuple[tuple[float, float], ...]) -> dict[str, Any]:
    if not contacts:
        return {"pattern": "unknown", "frames": 0, "left_contact_frames": 0, "right_contact_frames": 0}
    left = sum(1 for row in contacts if row[0] >= 0.5)
    right = sum(1 for row in contacts if row[1] >= 0.5)
    both = sum(1 for row in contacts if row[0] >= 0.5 and row[1] >= 0.5)
    none = sum(1 for row in contacts if row[0] < 0.5 and row[1] < 0.5)
    alternating = sum(1 for row in contacts if (row[0] >= 0.5) != (row[1] >= 0.5))
    frames = len(contacts)
    if none / frames >= 0.25:
        pattern = "unstable"
    elif alternating / frames >= 0.5 and left and right:
        pattern = "alternating"
    elif both / frames >= 0.75:
        pattern = "stable"
    else:
        pattern = "mixed"
    return {
        "pattern": pattern,
        "frames": frames,
        "left_contact_frames": left,
        "right_contact_frames": right,
        "both_contact_frames": both,
        "no_contact_frames": none,
    }


def _stance_consistency(root_displacement: float, foot_velocity_max: float, contact_pattern: str, dynamic_intensity: float | None) -> str:
    if contact_pattern == "unstable" or (dynamic_intensity is not None and dynamic_intensity >= 0.8):
        return "unstable"
    if root_displacement <= 0.12 and foot_velocity_max <= 0.05 and contact_pattern in {"stable", "mixed", "unknown"}:
        return "stable"
    if contact_pattern == "alternating":
        return "alternating"
    return "mixed"


def _distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt(sum((float(b[index]) - float(a[index])) ** 2 for index in range(3)))


def _float_or_none(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _optional_clean(value: Any) -> float | None:
    parsed = _float_or_none(value)
    return _clean_float(parsed) if parsed is not None else None


def _clean_float(value: float) -> float:
    return round(float(value), 6)
