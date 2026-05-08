"""Gated, conservative foot stabilization helpers for Kimodo retarget clips."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any


ALLOWED_STATUSES = {"applied", "partial", "skipped", "not_evaluated"}
LIMITATION = "v1_stabilization_does_not_guarantee_perfect_animation"


@dataclass(frozen=True)
class FootStabilizationConfig:
    contact_threshold: float = 0.5
    residual_success_threshold: float = 0.05
    static_min_window_frames: int = 4
    walking_min_window_frames: int = 6
    dynamic_min_window_frames: int = 12
    static_max_velocity_per_frame: float = 0.03
    walking_max_velocity_per_frame: float = 0.06
    dynamic_max_velocity_per_frame: float = 0.02
    static_max_correction: float = 0.12
    walking_max_correction: float = 0.18
    dynamic_max_correction: float = 0.05

    def defaults_for(self, motion_class: str) -> dict[str, float | int]:
        normalized = _normalize_motion_class(motion_class)
        if normalized in {"walking", "locomotion"}:
            return {
                "min_window_frames": self.walking_min_window_frames,
                "max_velocity_per_frame": self.walking_max_velocity_per_frame,
                "max_correction": self.walking_max_correction,
            }
        if normalized == "dynamic":
            return {
                "min_window_frames": self.dynamic_min_window_frames,
                "max_velocity_per_frame": self.dynamic_max_velocity_per_frame,
                "max_correction": self.dynamic_max_correction,
            }
        return {
            "min_window_frames": self.static_min_window_frames,
            "max_velocity_per_frame": self.static_max_velocity_per_frame,
            "max_correction": self.static_max_correction,
        }


@dataclass(frozen=True)
class ContactWindow:
    side: str
    start: int
    end: int
    max_slide: float
    max_velocity: float
    eligible: bool
    rejected_reason: str | None = None

    @property
    def frame_count(self) -> int:
        return max(0, self.end - self.start + 1)

    def as_metadata(self) -> dict[str, object]:
        payload = {
            "side": self.side,
            "start": self.start,
            "end": self.end,
            "frame_count": self.frame_count,
            "max_slide": _clean_float(self.max_slide),
            "max_velocity": _clean_float(self.max_velocity),
        }
        if self.rejected_reason is not None:
            payload["rejected_reason"] = self.rejected_reason
        return payload


@dataclass(frozen=True)
class GateEvaluation:
    status: str
    motion_class: str
    gates: dict[str, object]
    windows: tuple[ContactWindow, ...] = field(default_factory=tuple)
    pre_max_slide: float | None = None
    skipped_reason: str | None = None
    warnings: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class FootStabilizationResult:
    status: str
    mode: str
    motion_class: str
    gates: dict[str, object] = field(default_factory=dict)
    windows: tuple[dict[str, object] | ContactWindow, ...] = field(default_factory=tuple)
    corrected_frames: int = 0
    strength: float = 0.0
    max_correction: float | None = None
    pre_max_slide: float | None = None
    post_max_slide: float | None = None
    residual_slide: float | None = None
    skipped_reason: str | None = None
    warnings: tuple[str, ...] = field(default_factory=tuple)
    limitations: tuple[str, ...] = (LIMITATION,)


def evaluate_stabilization_gates(
    *,
    foot_samples: dict[str, tuple[tuple[float, float, float], ...]],
    foot_contacts: tuple[tuple[float, ...], ...] | None,
    motion_class: str,
    trusted_foot_roles: bool,
    config: FootStabilizationConfig | None = None,
    allow_dynamic: bool = False,
) -> GateEvaluation:
    config = config or FootStabilizationConfig()
    normalized = _normalize_motion_class(motion_class)
    defaults = config.defaults_for(normalized)
    gates = {
        "contact_threshold": config.contact_threshold,
        "min_window_frames": defaults["min_window_frames"],
        "max_velocity_per_frame": defaults["max_velocity_per_frame"],
        "max_correction": defaults["max_correction"],
        "residual_success_threshold": config.residual_success_threshold,
    }
    if not trusted_foot_roles:
        return GateEvaluation("not_evaluated", normalized, gates, skipped_reason="missing_trusted_foot_roles")
    if not foot_contacts:
        return GateEvaluation("not_evaluated", normalized, gates, skipped_reason="missing_contact_data")
    if not _samples_have_both_feet(foot_samples):
        return GateEvaluation("not_evaluated", normalized, gates, skipped_reason="missing_foot_samples")
    frame_count = len(foot_contacts)
    if any(len(samples) != frame_count for samples in foot_samples.values()):
        return GateEvaluation("not_evaluated", normalized, gates, skipped_reason="foot_samples_contact_frame_count_mismatch")
    if not _finite_samples(foot_samples):
        return GateEvaluation("not_evaluated", normalized, gates, skipped_reason="non_finite_foot_samples")
    if normalized == "dynamic" and not allow_dynamic:
        return GateEvaluation(
            "skipped",
            normalized,
            gates,
            windows=_window_metrics(foot_samples, foot_contacts, config, gates),
            skipped_reason="dynamic_motion_warning_only_by_default",
            warnings=("foot_stabilization/dynamic_default_skip",),
        )
    windows = _window_metrics(foot_samples, foot_contacts, config, gates)
    eligible = tuple(window for window in windows if window.eligible)
    pre_max_slide = max((window.max_slide for window in windows), default=0.0)
    if not eligible:
        return GateEvaluation("skipped", normalized, gates, windows=windows, pre_max_slide=_clean_float(pre_max_slide), skipped_reason="no_eligible_contact_windows")
    return GateEvaluation("eligible", normalized, gates, windows=eligible, pre_max_slide=_clean_float(pre_max_slide))


def stabilize_feet(clip: Any, rig: Any, motion_context: Any, *, motion_class: str | None = None, config: FootStabilizationConfig | None = None) -> tuple[Any, FootStabilizationResult]:
    config = config or FootStabilizationConfig()
    normalized = _normalize_motion_class(motion_class or _infer_motion_class(clip))
    if normalized in {"walking", "locomotion"} and not _has_valid_leg_chains(rig):
        result = _result_from_skip(normalized, "missing_valid_leg_chains", config, mode="none")
        return _with_stabilization_metadata(clip, result), result

    root_channel = _root_translation_channel(clip)
    contacts = getattr(motion_context, "foot_contacts", None)
    if root_channel is None:
        result = _result_from_skip(normalized, "missing_root_translation_channel", config, mode="none")
        return _with_stabilization_metadata(clip, result), result
    samples = {"left": tuple(root_channel.values), "right": tuple(root_channel.values)}
    gates = evaluate_stabilization_gates(
        foot_samples=samples,
        foot_contacts=contacts,
        motion_class=normalized,
        trusted_foot_roles=_has_trusted_foot_roles(rig),
        config=config,
    )
    if gates.status != "eligible":
        result = FootStabilizationResult(
            status=gates.status if gates.status in {"skipped", "not_evaluated"} else "skipped",
            mode="none",
            motion_class=normalized,
            gates=gates.gates,
            windows=tuple(window.as_metadata() for window in gates.windows),
            pre_max_slide=gates.pre_max_slide,
            skipped_reason=gates.skipped_reason,
            warnings=gates.warnings,
        )
        return _with_stabilization_metadata(clip, result), result

    if normalized in {"walking", "locomotion"}:
        stabilized, result = _apply_walking_two_bone_ik(clip, rig, normalized, gates, config)
        return _with_stabilization_metadata(stabilized, result), result

    corrected_values, corrected_frames, max_correction = _compensate_root_values(tuple(root_channel.values), gates.windows)
    corrected_channel = replace(root_channel, values=corrected_values, diagnostics={**root_channel.diagnostics, "foot_stabilization": "root_compensation"})
    channels = tuple(corrected_channel if channel is root_channel else channel for channel in clip.channels)
    stabilized = replace(clip, channels=channels, root_translations=corrected_values)
    result = FootStabilizationResult(
        status="applied" if corrected_frames else "skipped",
        mode="root_compensation" if corrected_frames else "none",
        motion_class=normalized,
        gates=gates.gates,
        windows=tuple(window.as_metadata() for window in gates.windows),
        corrected_frames=corrected_frames,
        strength=1.0 if corrected_frames else 0.0,
        max_correction=_clean_float(max_correction),
        pre_max_slide=gates.pre_max_slide,
        warnings=("foot_stabilization/residual_pending",) if corrected_frames else (),
    )
    return _with_stabilization_metadata(stabilized, result), result


def foot_stabilization_metadata(result: FootStabilizationResult) -> dict[str, object]:
    if result.status not in ALLOWED_STATUSES:
        raise ValueError(f"Unsupported foot stabilization status: {result.status}")
    return {
        "status": result.status,
        "mode": result.mode,
        "motion_class": result.motion_class,
        "gates": dict(result.gates),
        "windows": [_window_payload(window) for window in result.windows],
        "corrected_frames": int(result.corrected_frames),
        "strength": _clean_float(result.strength),
        "max_correction": _optional_clean(result.max_correction),
        "pre_max_slide": _optional_clean(result.pre_max_slide),
        "post_max_slide": _optional_clean(result.post_max_slide),
        "residual_slide": _optional_clean(result.residual_slide),
        "skipped_reason": result.skipped_reason,
        "warnings": list(dict.fromkeys(str(warning) for warning in result.warnings)),
        "limitations": list(dict.fromkeys(str(limitation) for limitation in result.limitations)),
    }


def update_stabilization_residuals(
    result: FootStabilizationResult,
    *,
    post_max_slide: float | None,
    config: FootStabilizationConfig | None = None,
) -> FootStabilizationResult:
    config = config or FootStabilizationConfig()
    if result.status not in {"applied", "partial"} or post_max_slide is None:
        return result
    residual = _clean_float(post_max_slide)
    warnings = tuple(warning for warning in result.warnings if warning != "foot_stabilization/residual_pending")
    status = result.status
    if residual > config.residual_success_threshold:
        status = "partial"
        warnings = tuple(dict.fromkeys(warnings + ("foot_stabilization/residual_slide_above_threshold",)))
    return replace(result, status=status, post_max_slide=residual, residual_slide=residual, warnings=warnings)


def merge_stabilization_into_visual_quality(visual_quality: dict[str, object] | None, result: FootStabilizationResult) -> dict[str, object]:
    payload = dict(visual_quality or {})
    payload.setdefault("version", "kimodo.visual_quality.v1")
    payload.setdefault("blocking", False)
    payload.setdefault("warnings", [])
    payload.setdefault("limitations", [])
    payload.setdefault("checks", {})
    payload["foot_stabilization"] = foot_stabilization_metadata(result)
    return payload


def result_from_metadata(metadata: dict[str, object]) -> FootStabilizationResult:
    return FootStabilizationResult(
        status=str(metadata.get("status") or "not_evaluated"),
        mode=str(metadata.get("mode") or "none"),
        motion_class=str(metadata.get("motion_class") or "unknown"),
        gates=dict(metadata.get("gates") if isinstance(metadata.get("gates"), dict) else {}),
        windows=tuple(metadata.get("windows") if isinstance(metadata.get("windows"), list) else ()),
        corrected_frames=int(metadata.get("corrected_frames") or 0),
        strength=float(metadata.get("strength") or 0.0),
        max_correction=_optional_float(metadata.get("max_correction")),
        pre_max_slide=_optional_float(metadata.get("pre_max_slide")),
        post_max_slide=_optional_float(metadata.get("post_max_slide")),
        residual_slide=_optional_float(metadata.get("residual_slide")),
        skipped_reason=str(metadata.get("skipped_reason")) if metadata.get("skipped_reason") is not None else None,
        warnings=tuple(str(value) for value in metadata.get("warnings", ()) if isinstance(metadata.get("warnings"), list)),
        limitations=tuple(str(value) for value in metadata.get("limitations", ()) if isinstance(metadata.get("limitations"), list)) or (LIMITATION,),
    )


def _window_metrics(foot_samples: dict[str, tuple[tuple[float, float, float], ...]], foot_contacts: tuple[tuple[float, ...], ...], config: FootStabilizationConfig, gates: dict[str, object]) -> tuple[ContactWindow, ...]:
    windows: list[ContactWindow] = []
    for side in ("left", "right"):
        for start, end in _contact_windows(foot_contacts, side=side, threshold=config.contact_threshold):
            samples = foot_samples[side]
            max_slide = _window_slide(samples, start, end)
            max_velocity = _window_velocity(samples, start, end)
            reason = None
            if end - start + 1 < int(gates["min_window_frames"]):
                reason = "insufficient_window_frames"
            elif max_velocity > float(gates["max_velocity_per_frame"]):
                reason = "velocity_exceeds_cap"
            elif max_slide > float(gates["max_correction"]):
                reason = "correction_exceeds_cap"
            windows.append(ContactWindow(side, start, end, _clean_float(max_slide), _clean_float(max_velocity), reason is None, reason))
    return tuple(windows)


def _compensate_root_values(values: tuple[tuple[float, ...], ...], windows: tuple[ContactWindow, ...]) -> tuple[tuple[tuple[float, ...], ...], int, float]:
    original = [tuple(float(component) for component in value) for value in values]
    mutable = list(original)
    applied_weights: dict[int, float] = {}
    corrected_frames: set[int] = set()
    max_correction = 0.0
    for window in windows:
        if not window.eligible or window.start >= len(mutable):
            continue
        anchor = original[window.start]
        for index in range(window.start, min(window.end + 1, len(mutable))):
            weight = _blend_weight(window, index)
            if weight <= applied_weights.get(index, -1.0):
                continue
            current = original[index]
            corrected_x = current[0] + (anchor[0] - current[0]) * weight
            corrected_z = current[2] + (anchor[2] - current[2]) * weight
            correction = math.hypot(current[0] - corrected_x, current[2] - corrected_z)
            max_correction = max(max_correction, correction)
            mutable[index] = (_clean_float(corrected_x), current[1], _clean_float(corrected_z))
            applied_weights[index] = weight
            corrected_frames.add(index)
    return tuple(mutable), len(corrected_frames), max_correction


def _blend_weight(window: ContactWindow, index: int) -> float:
    frame_count = window.frame_count
    if frame_count <= 0:
        return 0.0
    blend_frames = max(1, min(3, int(frame_count * 0.2)))
    position = index - window.start
    distance_to_end = window.end - index
    if position < blend_frames:
        return _clean_float((position + 1) / (blend_frames + 1))
    if distance_to_end < blend_frames:
        return _clean_float((distance_to_end + 1) / (blend_frames + 1))
    return 1.0


def _apply_walking_two_bone_ik(clip: Any, rig: Any, motion_class: str, gates: GateEvaluation, config: FootStabilizationConfig) -> tuple[Any, FootStabilizationResult]:
    reason = _walking_ik_skip_reason(clip, rig, gates)
    if reason is not None:
        return clip, FootStabilizationResult(
            status="skipped",
            mode="two_bone_ik",
            motion_class=motion_class,
            gates=gates.gates,
            windows=tuple(window.as_metadata() for window in gates.windows),
            pre_max_slide=gates.pre_max_slide,
            skipped_reason=reason,
        )

    channels = list(getattr(clip, "channels", ()))
    by_role = _rotation_channels_by_role(channels)
    corrected_frames: set[int] = set()
    max_correction = 0.0
    for window in gates.windows:
        roles = _leg_roles_for_side(window.side)
        upper = by_role[roles[0]]
        lower = by_role[roles[1]]
        upper_values = list(upper.values)
        lower_values = list(lower.values)
        limb_length = _limb_length_for_side(rig, window.side)
        anchor_index = window.start
        root_values = tuple(_root_translation_channel(clip).values)  # type: ignore[union-attr]
        anchor = root_values[anchor_index]
        for index in range(window.start, min(window.end + 1, len(root_values))):
            current = root_values[index]
            weight = _blend_weight(window, index)
            correction = math.hypot(float(current[0]) - float(anchor[0]), float(current[2]) - float(anchor[2])) * weight
            max_correction = max(max_correction, correction)
            angle = math.atan2(correction, limb_length)
            upper_values[index] = _axis_angle_quaternion((0.0, 0.0, 1.0), -angle * 0.5)
            lower_values[index] = _axis_angle_quaternion((0.0, 0.0, 1.0), angle)
            corrected_frames.add(index)
        channels[channels.index(upper)] = replace(upper, values=tuple(upper_values), diagnostics={**upper.diagnostics, "foot_stabilization": "two_bone_ik"})
        channels[channels.index(lower)] = replace(lower, values=tuple(lower_values), diagnostics={**lower.diagnostics, "foot_stabilization": "two_bone_ik"})

    result = FootStabilizationResult(
        status="applied" if corrected_frames else "skipped",
        mode="two_bone_ik" if corrected_frames else "none",
        motion_class=motion_class,
        gates=gates.gates,
        windows=tuple(window.as_metadata() for window in gates.windows),
        corrected_frames=len(corrected_frames),
        strength=1.0 if corrected_frames else 0.0,
        max_correction=_clean_float(max_correction),
        pre_max_slide=gates.pre_max_slide,
        warnings=("foot_stabilization/residual_pending",) if corrected_frames else (),
    )
    return replace(clip, channels=tuple(channels)), result


def _walking_ik_skip_reason(clip: Any, rig: Any, gates: GateEvaluation) -> str | None:
    frame_count = len(getattr(_root_translation_channel(clip), "values", ()))
    rotation_channels = _rotation_channels_by_role(getattr(clip, "channels", ()))
    for role in ("left_upper_leg", "left_lower_leg", "right_upper_leg", "right_lower_leg"):
        if role not in rotation_channels:
            return "missing_ik_rotation_channels"
        if len(rotation_channels[role].values) != frame_count:
            return "ik_rotation_frame_count_mismatch"
    if not _has_finite_rest_basis(rig):
        return "invalid_ik_rest_basis"
    if _knee_inversion_risk(rig):
        return "knee_inversion_risk"
    for window in gates.windows:
        limb_length = _limb_length_for_side(rig, window.side)
        if limb_length <= 1e-6:
            return "invalid_ik_chain_length"
        if window.max_slide > limb_length:
            return "ik_target_exceeds_limb_reach"
    return None


def _rotation_channels_by_role(channels: tuple[Any, ...] | list[Any]) -> dict[str, Any]:
    return {
        str(getattr(channel, "diagnostics", {}).get("role")): channel
        for channel in channels
        if getattr(channel, "path", None) == "rotation" and str(getattr(channel, "diagnostics", {}).get("role"))
    }


def _leg_roles_for_side(side: str) -> tuple[str, str, str]:
    prefix = "left" if side == "left" else "right"
    return (f"{prefix}_upper_leg", f"{prefix}_lower_leg", f"{prefix}_foot")


def _limb_length_for_side(rig: Any, side: str) -> float:
    upper_role, lower_role, _foot_role = _leg_roles_for_side(side)
    upper = _rest_translation(rig, upper_role)
    lower = _rest_translation(rig, lower_role)
    return _vector_length(upper) + _vector_length(lower)


def _has_finite_rest_basis(rig: Any) -> bool:
    basis = getattr(rig, "humanoid_basis_hints", None)
    if not isinstance(basis, dict) or not basis:
        return False
    knee_plane = basis.get("knee_plane_normal")
    if not _finite_vector(knee_plane, min_length=3):
        return False
    for side in ("left", "right"):
        for role in _leg_roles_for_side(side)[:2]:
            translation = _rest_translation(rig, role)
            if not _finite_vector(translation, min_length=3) or _vector_length(translation) <= 1e-6:
                return False
    return True


def _knee_inversion_risk(rig: Any) -> bool:
    knee_plane = getattr(rig, "humanoid_basis_hints", {}).get("knee_plane_normal")
    return bool(_finite_vector(knee_plane, min_length=3) and float(knee_plane[2]) < 0.0)


def _rest_translation(rig: Any, role: str) -> tuple[float, ...]:
    rest = getattr(rig, "humanoid_rest", {})
    roles = rest.get("roles") if isinstance(rest, dict) else None
    entry = roles.get(role) if isinstance(roles, dict) else None
    translation = entry.get("local_translation") if isinstance(entry, dict) else None
    if not isinstance(translation, (list, tuple)):
        return ()
    return tuple(float(value) for value in translation[:3])


def _finite_vector(value: object, *, min_length: int) -> bool:
    return isinstance(value, (list, tuple)) and len(value) >= min_length and all(math.isfinite(float(component)) for component in value[:min_length])


def _vector_length(value: tuple[float, ...]) -> float:
    if not _finite_vector(value, min_length=3):
        return 0.0
    return math.sqrt(sum(float(component) * float(component) for component in value[:3]))


def _axis_angle_quaternion(axis: tuple[float, float, float], angle: float) -> tuple[float, float, float, float]:
    length = math.sqrt(sum(component * component for component in axis))
    if length <= 1e-12 or abs(angle) <= 1e-12:
        return (0.0, 0.0, 0.0, 1.0)
    half = angle * 0.5
    scale = math.sin(half) / length
    return (_clean_float(axis[0] * scale), _clean_float(axis[1] * scale), _clean_float(axis[2] * scale), _clean_float(math.cos(half)))


def _with_stabilization_metadata(clip: Any, result: FootStabilizationResult) -> Any:
    return replace(clip, visual_quality=merge_stabilization_into_visual_quality(getattr(clip, "visual_quality", None), result))


def _result_from_skip(motion_class: str, reason: str, config: FootStabilizationConfig, *, mode: str) -> FootStabilizationResult:
    defaults = config.defaults_for(motion_class)
    return FootStabilizationResult(
        status="skipped",
        mode=mode,
        motion_class=motion_class,
        gates={"contact_threshold": config.contact_threshold, "residual_success_threshold": config.residual_success_threshold, **defaults},
        skipped_reason=reason,
    )


def _root_translation_channel(clip: Any) -> Any | None:
    for channel in getattr(clip, "channels", ()):
        if getattr(channel, "path", None) == "translation" and str(getattr(channel, "diagnostics", {}).get("role") or "") in {"hips", "root", ""}:
            return channel
    for channel in getattr(clip, "channels", ()):
        if getattr(channel, "path", None) == "translation":
            return channel
    return None


def _has_trusted_foot_roles(rig: Any) -> bool:
    roles = {getattr(binding, "role", None) for binding in getattr(rig, "humanoid_roles", ())}
    return {"left_foot", "right_foot"}.issubset(roles)


def _has_valid_leg_chains(rig: Any) -> bool:
    chains = {getattr(chain, "name", None): tuple(getattr(chain, "roles", ())) for chain in getattr(rig, "humanoid_chains", ())}
    return chains.get("left_leg") == ("left_upper_leg", "left_lower_leg", "left_foot") and chains.get("right_leg") == ("right_upper_leg", "right_lower_leg", "right_foot") and bool(getattr(rig, "humanoid_rest", None) or getattr(rig, "humanoid_basis_hints", None))


def _samples_have_both_feet(foot_samples: dict[str, tuple[tuple[float, float, float], ...]]) -> bool:
    return bool(foot_samples.get("left")) and bool(foot_samples.get("right"))


def _finite_samples(foot_samples: dict[str, tuple[tuple[float, float, float], ...]]) -> bool:
    return all(len(sample) >= 3 and all(math.isfinite(float(component)) for component in sample[:3]) for samples in foot_samples.values() for sample in samples)


def _contact_windows(foot_contacts: tuple[tuple[float, ...], ...], *, side: str, threshold: float) -> list[tuple[int, int]]:
    column = 0 if side == "left" else 1
    active = [len(row) > column and float(row[column]) >= threshold for row in foot_contacts]
    windows: list[tuple[int, int]] = []
    start: int | None = None
    for index, is_active in enumerate(active):
        if is_active and start is None:
            start = index
        elif not is_active and start is not None:
            windows.append((start, index - 1))
            start = None
    if start is not None:
        windows.append((start, len(active) - 1))
    return windows


def _window_slide(samples: tuple[tuple[float, ...], ...], start: int, end: int) -> float:
    anchor = samples[start]
    return max(math.hypot(float(samples[index][0]) - float(anchor[0]), float(samples[index][2]) - float(anchor[2])) for index in range(start, end + 1))


def _window_velocity(samples: tuple[tuple[float, ...], ...], start: int, end: int) -> float:
    if end <= start:
        return 0.0
    return max(math.hypot(float(samples[index][0]) - float(samples[index - 1][0]), float(samples[index][2]) - float(samples[index - 1][2])) for index in range(start + 1, end + 1))


def _window_payload(window: dict[str, object] | ContactWindow) -> dict[str, object]:
    return window.as_metadata() if isinstance(window, ContactWindow) else dict(window)


def _infer_motion_class(clip: Any) -> str:
    calibration = getattr(clip, "calibration", {}) or {}
    policy = calibration.get("locomotion_policy") if isinstance(calibration, dict) else None
    if isinstance(policy, dict) and policy.get("detected_locomotion"):
        return "walking"
    return "static"


def _normalize_motion_class(motion_class: str) -> str:
    value = str(motion_class or "unknown").lower()
    if value in {"idle", "static", "gesture", "gestures", "unknown"}:
        return "gesture" if value in {"gesture", "gestures"} else ("unknown" if value == "unknown" else "static")
    if value in {"walk", "walking", "locomotion"}:
        return "walking" if value == "walk" else value
    if value in {"dance", "dancing", "dynamic"}:
        return "dynamic"
    return value


def _optional_clean(value: float | None) -> float | None:
    return None if value is None else _clean_float(value)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clean_float(value: float) -> float:
    rounded = round(float(value), 10)
    return 0.0 if abs(rounded) < 1e-10 else rounded
