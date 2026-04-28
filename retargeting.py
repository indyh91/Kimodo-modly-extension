"""Conservative BVH-to-target-rig mapping and keyframe seams."""

from __future__ import annotations

from dataclasses import dataclass, field

from kimodo_bvh import BvhMotion
from retarget_errors import MAPPING_INCOMPATIBLE, RetargetError
from rig_inspector import RigInspection


BVH_TO_TARGET_ALIASES = {
    "hips": ("hips", "hip", "pelvis", "root"),
    "spine": ("spine", "spine1", "chest"),
    "head": ("head", "neck"),
    "left_arm": ("leftarm", "leftupperarm", "lupperarm", "leftshoulder"),
    "right_arm": ("rightarm", "rightupperarm", "rupperarm", "rightshoulder"),
    "left_leg": ("leftupleg", "leftleg", "lthigh", "leftthigh"),
    "right_leg": ("rightupleg", "rightleg", "rthigh", "rightthigh"),
}


@dataclass(frozen=True)
class JointMapping:
    pairs: dict[str, str]
    confidence: str
    warnings: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class RetargetChannel:
    target_joint: str
    path: str
    times: tuple[float, ...]
    values: tuple[tuple[float, ...], ...]
    interpolation: str = "LINEAR"


@dataclass(frozen=True)
class RetargetClip:
    name: str
    duration: float
    frame_time: float
    root_translations: tuple[tuple[float, float, float], ...]
    channels: tuple[RetargetChannel, ...]
    mapping: JointMapping
    warnings: tuple[str, ...]


def build_joint_mapping(source: BvhMotion, target: RigInspection) -> JointMapping:
    source_lookup = {_normalize(name): name for name in source.joints}
    target_lookup = {_normalize(name): name for name in target.joint_names}
    pairs: dict[str, str] = {}
    missing: list[str] = []
    for semantic, aliases in BVH_TO_TARGET_ALIASES.items():
        source_name = _first_match(source_lookup, aliases)
        target_name = _first_match(target_lookup, aliases)
        if source_name and target_name:
            pairs[source_name] = target_name
        else:
            missing.append(semantic)

    required_missing = {"hips", "spine", "head"}.intersection(missing)
    if required_missing:
        raise RetargetError(MAPPING_INCOMPATIBLE, f"Required humanoid joints could not be mapped: {sorted(required_missing)}")
    confidence = "compatible" if len(pairs) >= 5 else "best_effort"
    warnings = tuple(f"missing_{semantic}_mapping" for semantic in missing)
    return JointMapping(pairs=pairs, confidence=confidence, warnings=warnings)


def build_retarget_clip(source: BvhMotion, target: RigInspection) -> RetargetClip:
    mapping = build_joint_mapping(source, target)
    root_channels = [idx for idx, (joint, channel) in enumerate(source.channel_order) if joint == source.root and channel.endswith("position")]
    translations = []
    for frame in source.frames:
        xyz = [0.0, 0.0, 0.0]
        for idx in root_channels:
            channel = source.channel_order[idx][1].lower()
            if channel.startswith("x"):
                xyz[0] = frame[idx]
            elif channel.startswith("y"):
                xyz[1] = frame[idx]
            elif channel.startswith("z"):
                xyz[2] = frame[idx]
        translations.append(tuple(xyz))
    times = tuple(index * source.frame_time for index in range(len(source.frames)))
    root_target = mapping.pairs.get(source.root)
    channels: tuple[RetargetChannel, ...] = ()
    if root_target is not None:
        channels = (
            RetargetChannel(
                target_joint=root_target,
                path="translation",
                times=times,
                values=tuple(translations),
            ),
        )
    warnings = mapping.warnings + target.warnings + ("axis_assumption_bvh_xyz_degrees", "root_motion_mode_preserve_translation")
    return RetargetClip(
        name="Kimodo Motion",
        duration=source.duration,
        frame_time=source.frame_time,
        root_translations=tuple(translations),
        channels=channels,
        mapping=mapping,
        warnings=warnings,
    )


def _first_match(lookup: dict[str, str], aliases: tuple[str, ...]) -> str | None:
    for alias in aliases:
        normalized = _normalize(alias)
        if normalized in lookup:
            return lookup[normalized]
    return None


def _normalize(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())
