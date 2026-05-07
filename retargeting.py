"""Conservative BVH-to-target-rig mapping and keyframe seams."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

from kimodo_bvh import BvhMotion
from retarget_errors import CALIBRATION_UNAVAILABLE, MAPPING_INCOMPATIBLE, RetargetError
from rig_inspector import HUMANOID_CONTRACT_SCHEMA, REQUIRED_HUMANOID_CHAINS, RigInspection


BVH_TO_TARGET_ALIASES = {
    "hips": ("hips", "hip", "pelvis", "root"),
    "spine": ("spine", "spine1", "chest"),
    "head": ("head", "neck"),
    "left_arm": ("leftarm", "leftupperarm", "lupperarm"),
    "right_arm": ("rightarm", "rightupperarm", "rupperarm"),
    "left_leg": ("leftupleg", "leftleg", "lthigh", "leftthigh"),
    "right_leg": ("rightupleg", "rightleg", "rthigh", "rightthigh"),
}

SOURCE_ROLE_ALIASES = {
    "hips": ("hips", "hip", "pelvis", "root"),
    "spine": ("spine", "spine1"),
    "chest": ("chest", "spine2", "spine3", "upperchest"),
    "neck": ("neck",),
    "head": ("head",),
    "left_shoulder": ("leftshoulder", "leftclavicle", "lshoulder", "lclavicle"),
    "left_upper_arm": ("leftupperarm", "leftarm", "lupperarm"),
    "left_forearm": ("leftforearm", "leftlowerarm", "llowerarm", "leftelbow"),
    "left_hand": ("lefthand", "lhand", "leftwrist"),
    "right_shoulder": ("rightshoulder", "rightclavicle", "rshoulder", "rclavicle"),
    "right_upper_arm": ("rightupperarm", "rightarm", "rupperarm"),
    "right_forearm": ("rightforearm", "rightlowerarm", "rlowerarm", "rightelbow"),
    "right_hand": ("righthand", "rhand", "rightwrist"),
    "left_upper_leg": ("leftupperleg", "leftupleg", "leftleg", "lthigh", "leftthigh"),
    "left_lower_leg": ("leftlowerleg", "leftleglower", "lshin", "leftshin", "leftcalf"),
    "left_foot": ("leftfoot", "lfoot", "leftankle"),
    "right_upper_leg": ("rightupperleg", "rightupleg", "rightleg", "rthigh", "rightthigh"),
    "right_lower_leg": ("rightlowerleg", "rightleglower", "rshin", "rightshin", "rightcalf"),
    "right_foot": ("rightfoot", "rfoot", "rightankle"),
}

OPTIONAL_HUMANOID_CHAINS = {
    "left_fingers": ("left_thumb", "left_index", "left_middle", "left_ring", "left_little"),
    "right_fingers": ("right_thumb", "right_index", "right_middle", "right_ring", "right_little"),
    "left_toes": ("left_toes",),
    "right_toes": ("right_toes",),
}

OPTIONAL_PROXIMAL_HUMANOID_CHAINS = {
    "left_arm_proximal": ("left_shoulder", "left_upper_arm", "left_forearm", "left_hand"),
    "right_arm_proximal": ("right_shoulder", "right_upper_arm", "right_forearm", "right_hand"),
}

TARGET_BASIS_CHILD_ROLES = {
    "hips": "spine",
    "left_shoulder": "left_upper_arm",
    "right_shoulder": "right_upper_arm",
    "left_upper_leg": "left_lower_leg",
    "right_upper_leg": "right_lower_leg",
    "left_lower_leg": "left_foot",
    "right_lower_leg": "right_foot",
    "left_upper_arm": "left_forearm",
    "right_upper_arm": "right_forearm",
}
TARGET_REST_BASIS_ROLL_WARNING = "target_basis_roll_inferred_not_visual_safe"

CALIBRATION_VERSION = "kimodo.retarget_calibration.v1"
COMPOSITION_ORDER = "target_rest * (target_basis * inverse(source_basis)) * (source_sample * inverse(source_first)) * inverse(target_basis * inverse(source_basis))"
FULL_MATRIX_COMPOSITION_ORDER = "target_rest * (inverse(target_basis) * source_basis) * (source_sample * inverse(source_first)) * inverse(inverse(target_basis) * source_basis)"
DEFAULT_TARGET_BASIS_COMPOSITION = "target_basis * inverse(source_basis)"
FULL_MATRIX_TARGET_BASIS_COMPOSITION = "inverse(target_basis) * source_basis"


@dataclass(frozen=True)
class ChainMapping:
    chain_name: str
    source_joints: tuple[str, ...]
    target_joints: tuple[str, ...]
    required: bool
    coverage: str
    roles: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class JointBasis:
    joint: str
    quaternion: tuple[float, float, float, float]
    source: str
    warnings: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class JointMapping:
    pairs: dict[str, str]
    confidence: str
    warnings: tuple[str, ...] = field(default_factory=tuple)
    chains: tuple[ChainMapping, ...] = field(default_factory=tuple)
    sidecar_provenance: dict[str, str] = field(default_factory=dict)
    coverage: dict[str, str] = field(default_factory=dict)
    basis_sources: dict[str, dict[str, object]] = field(default_factory=dict)


@dataclass(frozen=True)
class RetargetChannel:
    target_joint: str
    path: str
    times: tuple[float, ...]
    values: tuple[tuple[float, ...], ...]
    interpolation: str = "LINEAR"
    diagnostics: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class RetargetClip:
    name: str
    duration: float
    frame_time: float
    root_translations: tuple[tuple[float, float, float], ...]
    channels: tuple[RetargetChannel, ...]
    mapping: JointMapping
    warnings: tuple[str, ...]
    calibration: dict[str, object] = field(default_factory=dict)
    visual_quality: dict[str, object] | None = None


@dataclass(frozen=True)
class MotionContext:
    path: Path | None
    root_positions: tuple[tuple[float, ...], ...] | None = None
    smooth_root_pos: tuple[tuple[float, ...], ...] | None = None
    foot_contacts: tuple[tuple[float, ...], ...] | None = None
    source_keys: tuple[str, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def root_source(self) -> str:
        if _valid_root_positions(self.smooth_root_pos):
            return "smooth_root_pos"
        if _valid_root_positions(self.root_positions):
            return "root_positions"
        return "none"

    @property
    def root_samples(self) -> tuple[tuple[float, float, float], ...] | None:
        selected = self.smooth_root_pos if self.root_source == "smooth_root_pos" else self.root_positions
        if not _valid_root_positions(selected):
            return None
        return tuple((float(sample[0]), float(sample[1]), float(sample[2])) for sample in selected or ())

    @property
    def root_frame_count(self) -> int:
        samples = self.root_samples
        return len(samples) if samples is not None else 0

    @property
    def foot_contact_shape(self) -> tuple[int, ...]:
        return _array_shape(self.foot_contacts)

    @property
    def foot_contact_ratio(self) -> float:
        if not self.foot_contacts:
            return 0.0
        values = [float(value) for row in self.foot_contacts for value in row]
        return sum(values) / len(values) if values else 0.0

    def root_deltas(self, *, frame_count: int) -> tuple[tuple[float, float, float], ...]:
        samples = self.root_samples
        if samples is None or len(samples) != frame_count or not samples:
            return tuple((0.0, 0.0, 0.0) for _ in range(frame_count))
        origin = samples[0]
        return tuple((_clean_float(sample[0] - origin[0]), 0.0, _clean_float(sample[2] - origin[2])) for sample in samples)

    def locomotion_policy(self, *, frame_count: int, fallback_mode: str = "hips_suppressed_in_place") -> dict[str, object]:
        samples = self.root_samples
        warnings = list(self.warnings)
        source = self.root_source
        detected = False
        horizontal_range = 0.0
        displacement = (0.0, 0.0, 0.0)
        mode = fallback_mode
        if source == "none":
            warnings.extend(_invalid_root_position_warnings(self))
        elif samples is None or len(samples) != frame_count:
            warnings.append("root_positions_frame_count_mismatch")
        else:
            xs = [sample[0] for sample in samples]
            zs = [sample[2] for sample in samples]
            horizontal_range = math.hypot(max(xs) - min(xs), max(zs) - min(zs))
            displacement = (samples[-1][0] - samples[0][0], 0.0, samples[-1][2] - samples[0][2])
            horizontal_displacement = math.hypot(displacement[0], displacement[2])
            if max(horizontal_range, horizontal_displacement) >= 0.5:
                if max(horizontal_range, horizontal_displacement) <= 20.0:
                    detected = True
                    mode = "preserve_scaled_npz_root_motion"
                else:
                    warnings.append("root_displacement_sanity_clamped_as_invalid")
        return {
            "mode": mode,
            "source": source if source != "none" and not any(warning.startswith("root_positions_frame_count_mismatch") for warning in warnings) else (source if source != "none" else "none"),
            "detected_locomotion": detected,
            "root_horizontal_range": _clean_float(horizontal_range),
            "root_displacement": [_clean_float(value) for value in displacement],
            "foot_contact_shape": list(self.foot_contact_shape),
            "foot_contact_ratio": _clean_float(self.foot_contact_ratio),
            "warnings": _unique_list(warnings),
            "deferred": ["contact_aware_foot_locking"],
        }


def load_motion_context(path: Path | str | None) -> MotionContext | None:
    if path is None:
        return None
    npz_path = Path(path)
    try:
        import numpy as np

        with np.load(npz_path) as payload:
            keys = tuple(str(key) for key in payload.files)
            return MotionContext(
                path=npz_path,
                root_positions=_array_to_tuple(payload["root_positions"]) if "root_positions" in payload.files else None,
                smooth_root_pos=_array_to_tuple(payload["smooth_root_pos"]) if "smooth_root_pos" in payload.files else None,
                foot_contacts=_array_to_tuple(payload["foot_contacts"]) if "foot_contacts" in payload.files else None,
                source_keys=keys,
                warnings=(),
            )
    except Exception as exc:
        return MotionContext(path=npz_path, source_keys=(), warnings=(f"motion_context_load_failed:{type(exc).__name__}",))


def build_joint_mapping(source: BvhMotion, target: RigInspection) -> JointMapping:
    source_lookup = {_normalize(name): name for name in source.joints}
    calibrated_mapping = _build_calibrated_contract_mapping(source, source_lookup, target)
    if calibrated_mapping is not None:
        return calibrated_mapping
    contract_mapping = _build_humanoid_contract_mapping(source_lookup, target)
    if contract_mapping is not None:
        return contract_mapping
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
        anonymous_mapping = _build_anonymous_unirig_mapping(source_lookup, target)
        if anonymous_mapping is not None:
            return anonymous_mapping
        raise RetargetError(MAPPING_INCOMPATIBLE, f"Required humanoid joints could not be mapped: {sorted(required_missing)}")
    confidence = "compatible" if len(pairs) >= 5 else "best_effort"
    warnings = tuple(f"missing_{semantic}_mapping" for semantic in missing)
    return JointMapping(pairs=pairs, confidence=confidence, warnings=warnings)


def _build_humanoid_contract_mapping(source_lookup: dict[str, str], target: RigInspection) -> JointMapping | None:
    if not target.humanoid_roles:
        return None
    roles = {binding.role: binding.node_name for binding in target.humanoid_roles}
    semantic_roles = {
        "hips": ("hips",),
        "spine": ("spine",),
        "head": ("head", "neck"),
        "left_arm": ("left_upper_arm",),
        "right_arm": ("right_upper_arm",),
        "left_leg": ("left_upper_leg",),
        "right_leg": ("right_upper_leg",),
    }
    pairs: dict[str, str] = {}
    missing: list[str] = []
    for semantic, aliases in BVH_TO_TARGET_ALIASES.items():
        source_name = _first_match(source_lookup, aliases)
        target_name = next((roles[role] for role in semantic_roles[semantic] if role in roles), None)
        if source_name and target_name:
            pairs[source_name] = target_name
        elif not source_name:
            missing.append(semantic)
    required_missing = {"hips", "spine", "head"}.intersection(missing)
    if required_missing:
        raise RetargetError(MAPPING_INCOMPATIBLE, f"Required humanoid contract source joints could not be mapped: {sorted(required_missing)}")
    warnings = tuple(f"missing_{semantic}_source_mapping" for semantic in missing) + ("humanoid_contract_sidecar_mapping",)
    confidence = "compatible" if len(pairs) >= 5 else "best_effort"
    return JointMapping(pairs=pairs, confidence=confidence, warnings=warnings)


def build_retarget_clip(source: BvhMotion, target: RigInspection, motion_context: MotionContext | None = None) -> RetargetClip:
    mapping = build_joint_mapping(source, target)
    times = tuple(index * source.frame_time for index in range(len(source.frames)))
    root_target = _root_motion_target(source, mapping)
    suppressed_hips_source = _suppressed_hips_root_motion_source(source, mapping)
    root_motion_source = _root_motion_source_joint(source, mapping)
    root_motion_source_label = "hips_suppressed_in_place" if suppressed_hips_source is not None else _root_motion_source_label(source, root_motion_source)
    translations = _position_samples_for_joint(source, root_motion_source)
    fallback_mode = "hips_suppressed_in_place" if suppressed_hips_source is not None else "preserve_bvh_root_translation"
    locomotion_policy = _locomotion_policy_for_context(motion_context, frame_count=len(source.frames), fallback_mode=fallback_mode)
    if locomotion_policy["mode"] == "preserve_scaled_npz_root_motion" and motion_context is not None:
        translations = list(motion_context.root_deltas(frame_count=len(source.frames)))
        root_motion_source_label = str(locomotion_policy["source"])
    channels: list[RetargetChannel] = []
    if root_target is not None:
        diagnostics = {
            "root_motion_source": root_motion_source_label,
            "source_joint": root_motion_source,
            "target_joint": root_target,
        }
        if locomotion_policy["mode"] == "preserve_scaled_npz_root_motion":
            diagnostics["root_motion_mode"] = "preserve_scaled_npz_root_motion"
            if suppressed_hips_source is not None:
                diagnostics["suppressed_source_joint"] = suppressed_hips_source
        elif suppressed_hips_source is not None:
            diagnostics["suppressed_source_joint"] = suppressed_hips_source
            diagnostics["root_motion_mode"] = "in_place_default"
        channels.append(
            RetargetChannel(
                target_joint=root_target,
                path="translation",
                times=times,
                values=tuple(translations),
                diagnostics=diagnostics,
            ),
        )
    if mapping.confidence == "calibrated":
        channels.extend(_build_calibrated_rotation_channels(source, mapping, times))
    else:
        channels.extend(_build_rotation_channels(source, mapping, times))
    root_motion_warnings = [
        "axis_assumption_bvh_xyz_degrees",
        "rotation_order_assumption_bvh_channel_order_local_xyzw_quaternion",
        "root_motion_mode_preserve_translation",
    ]
    if locomotion_policy["mode"] == "preserve_scaled_npz_root_motion":
        root_motion_warnings.append("root_motion_mode_preserve_scaled_npz_root_motion")
    elif suppressed_hips_source is not None:
        root_motion_warnings.append("root_motion_mode_in_place_default")
        root_motion_warnings.append(f"root_motion_hips_detected_suppressed:{suppressed_hips_source}")
    root_motion_warnings.append(f"root_motion_source:{root_motion_source_label}")
    warnings = _unique_tuple(mapping.warnings + target.warnings + tuple(root_motion_warnings))
    calibration = _calibration_metadata(target, mapping) if mapping.confidence == "calibrated" else {}
    calibration["locomotion_policy"] = locomotion_policy
    contact_locking_diagnostics = _contact_locking_metadata_for_context(motion_context, frame_count=len(source.frames))
    if contact_locking_diagnostics is not None:
        calibration["contact_aware_foot_locking"] = contact_locking_diagnostics
    if calibration:
        calibration["root_motion_source"] = root_motion_source_label
        motion_quality_diagnostics = _motion_quality_diagnostics_metadata(channels)
        if motion_quality_diagnostics is not None:
            calibration["motion_quality_filter"] = motion_quality_diagnostics["motion_quality_filter"]
            calibration["motion_quality_diagnostics"] = motion_quality_diagnostics
    clip = RetargetClip(
        name="Kimodo Motion",
        duration=source.duration,
        frame_time=source.frame_time,
        root_translations=tuple(translations),
        channels=tuple(channels),
        mapping=mapping,
        warnings=warnings,
        calibration=calibration,
    )
    return RetargetClip(
        name=clip.name,
        duration=clip.duration,
        frame_time=clip.frame_time,
        root_translations=clip.root_translations,
        channels=clip.channels,
        mapping=clip.mapping,
        warnings=clip.warnings,
        calibration=clip.calibration,
        visual_quality=_safe_visual_quality_diagnostics(clip, motion_context=motion_context),
    )


def _build_calibrated_contract_mapping(source: BvhMotion, source_lookup: dict[str, str], target: RigInspection) -> JointMapping | None:
    if target.humanoid_contract_path is None and not target.humanoid_contract_schema:
        return None
    if target.humanoid_contract_schema != HUMANOID_CONTRACT_SCHEMA:
        raise RetargetError(CALIBRATION_UNAVAILABLE, f"Calibrated retargeting requires {HUMANOID_CONTRACT_SCHEMA} sidecar data.")
    if not target.humanoid_roles or not target.humanoid_chains:
        raise RetargetError(CALIBRATION_UNAVAILABLE, "Calibrated retargeting requires validated humanoid roles and chains.")
    if not target.humanoid_rest and not target.humanoid_basis_hints:
        raise RetargetError(
            CALIBRATION_UNAVAILABLE,
            "Calibrated retargeting requires target rest transforms or basis hints.",
            diagnostics={"retarget_calibration": {"fail_closed_reasons": ["missing_target_rest_or_basis_hints"]}},
        )

    role_to_target = {binding.role: binding.node_name for binding in target.humanoid_roles}
    chain_by_name = {chain.name: chain for chain in target.humanoid_chains}
    missing: list[str] = []
    pairs: dict[str, str] = {}
    chain_mappings: list[ChainMapping] = []
    basis_sources: dict[str, dict[str, str]] = {}
    bridge_warnings: list[str] = []
    for chain_name, required_roles in REQUIRED_HUMANOID_CHAINS.items():
        chain = chain_by_name.get(chain_name)
        if chain is None:
            missing.append(f"missing_target_chain:{chain_name}")
            continue
        ordered_roles = tuple(role for role in chain.roles if role in required_roles)
        if ordered_roles != tuple(required_roles):
            missing.append(f"unusable_target_chain:{chain_name}")
            continue
        source_joints: list[str] = []
        target_joints: list[str] = []
        for role in required_roles:
            source_joint, source_warnings = _source_joint_for_role(source, source_lookup, role)
            bridge_warnings.extend(source_warnings)
            target_joint = role_to_target.get(role)
            if source_joint is None:
                missing.append(f"missing_source_role:{chain_name}:{role}")
                continue
            if target_joint is None:
                missing.append(f"missing_target_role:{chain_name}:{role}")
                continue
            source_basis = _source_joint_basis(source, source_joint)
            target_basis = _target_joint_basis(target, role, target_joint)
            if source_basis is None:
                missing.append(f"missing_source_basis:{chain_name}:{role}:{source_joint}")
                continue
            if target_basis is None:
                missing.append(f"missing_target_basis:{chain_name}:{role}:{target_joint}")
                continue
            bridge_warnings.extend(source_basis.warnings)
            bridge_warnings.extend(target_basis.warnings)
            source_joints.append(source_joint)
            target_joints.append(target_joint)
            pairs[source_joint] = target_joint
            basis_sources[target_joint] = {
                "source": source_basis.source,
                "target": target_basis.source,
                "role": role,
                "source_quaternion": source_basis.quaternion,
                "target_quaternion": target_basis.quaternion,
            }
        chain_mappings.append(
            ChainMapping(
                chain_name=chain_name,
                source_joints=tuple(source_joints),
                target_joints=tuple(target_joints),
                required=True,
                coverage=f"{len(source_joints)}/{len(required_roles)}",
                roles=tuple(required_roles),
            )
        )

    if missing:
        raise RetargetError(
            CALIBRATION_UNAVAILABLE,
            "Calibrated retargeting missing required humanoid data: " + ", ".join(missing),
            diagnostics={"retarget_calibration": {"fail_closed_reasons": missing, "fallback_widened": False}},
        )

    optional_mappings, optional_warnings = _build_optional_proximal_chain_mappings(
        source,
        source_lookup,
        target,
        role_to_target,
        chain_by_name,
        pairs,
        basis_sources,
    )
    chain_mappings.extend(optional_mappings)
    warnings = _optional_chain_warnings(source, source_lookup, target, role_to_target, chain_by_name)
    warnings = _unique_tuple(tuple(bridge_warnings) + optional_warnings + warnings)
    warnings = ("humanoid_contract_calibrated_mapping",) + warnings
    return JointMapping(
        pairs=pairs,
        confidence="calibrated",
        warnings=warnings,
        chains=tuple(chain_mappings),
        sidecar_provenance={
            "schema": str(target.humanoid_contract_schema),
            "path": str(target.humanoid_contract_path),
            "quality": ";".join(target.humanoid_quality),
        },
        coverage={mapping.chain_name: mapping.coverage for mapping in chain_mappings},
        basis_sources=basis_sources,
    )


def _source_joint_basis(source: BvhMotion, joint_name: str) -> JointBasis | None:
    joint = source.joints.get(joint_name)
    if joint is None:
        return None
    vector = None
    for child_name in joint.children:
        child = source.joints.get(child_name)
        if child is not None and _vector_length(child.offset) > 1e-9:
            vector = child.offset
            break
    if vector is None and _vector_length(joint.offset) > 1e-9:
        vector = joint.offset
    if vector is None:
        return JointBasis(joint=joint_name, quaternion=(0.0, 0.0, 0.0, 1.0), source="source_hierarchy_offsets:identity")
    return JointBasis(joint=joint_name, quaternion=_quaternion_from_y_axis_to_vector(vector), source="source_hierarchy_offsets")


def _source_joint_for_role(source: BvhMotion, source_lookup: dict[str, str], role: str) -> tuple[str | None, tuple[str, ...]]:
    direct = _first_match(source_lookup, SOURCE_ROLE_ALIASES.get(role, (role,)))
    if direct is not None:
        return direct, ()
    if role != "neck":
        return None, ()
    bridged = _bridge_numbered_neck_source(source, source_lookup)
    if bridged is None:
        return None, ()
    return bridged, (f"source_neck_role_bridged:{bridged}",)


def _bridge_numbered_neck_source(source: BvhMotion, source_lookup: dict[str, str]) -> str | None:
    """Bridge the known BVH shape that names neck joints Neck1/Neck2.

    This is deliberately narrow: it only accepts numbered neck joints that are
    actually on the ancestor chain between a recognized Head and a recognized
    chest/upper-spine joint. It does not map arbitrary head/spine joints to neck.
    """

    head = _first_match(source_lookup, ("head",))
    if head is None:
        return None
    torso_anchor = _first_match(source_lookup, ("chest", "spine2", "spine3", "upperchest", "spine"))
    if torso_anchor is None:
        return None
    chain_from_head: list[str] = []
    current = source.joints.get(head)
    while current is not None and current.parent is not None:
        parent_name = current.parent
        if parent_name == torso_anchor:
            numbered_necks = [name for name in reversed(chain_from_head) if _is_numbered_neck(name)]
            return numbered_necks[0] if numbered_necks else None
        chain_from_head.append(parent_name)
        current = source.joints.get(parent_name)
    return None


def _is_numbered_neck(name: str) -> bool:
    normalized = _normalize(name)
    return normalized.startswith("neck") and normalized[4:].isdigit()


def _target_joint_basis(target: RigInspection, role: str, joint_name: str) -> JointBasis | None:
    matrix_basis = _rest_matrix_basis_for_role(target.humanoid_rest, role, joint_name)
    if matrix_basis is not None:
        quaternion, source = matrix_basis
        return JointBasis(joint=joint_name, quaternion=quaternion, source=source)
    hint = _basis_hint_for_role(target.humanoid_basis_hints, role)
    if hint is not None:
        return JointBasis(joint=joint_name, quaternion=hint[0], source=hint[1])
    child_role = _target_basis_child_role(role)
    child_joint_name = _target_joint_for_role(target, child_role) if child_role is not None else None
    rest_basis = _rest_vector_for_role(
        target.humanoid_rest,
        role,
        joint_name,
        child_role=child_role,
        child_joint_name=child_joint_name,
    )
    if rest_basis is not None:
        rest_vector, source = rest_basis
        return JointBasis(
            joint=joint_name,
            quaternion=_quaternion_from_y_axis_to_vector(rest_vector),
            source=source,
            warnings=(TARGET_REST_BASIS_ROLL_WARNING,),
        )
    if target.humanoid_rest:
        return JointBasis(
            joint=joint_name,
            quaternion=(0.0, 0.0, 0.0, 1.0),
            source="target_rest_transform:identity_fallback",
            warnings=(f"target_basis_child_vector_unavailable:{role}", TARGET_REST_BASIS_ROLL_WARNING),
        )
    return None


def _target_basis_child_role(role: str) -> str | None:
    return TARGET_BASIS_CHILD_ROLES.get(role)


def _target_joint_for_role(target: RigInspection, role: str | None) -> str | None:
    if role is None:
        return None
    for binding in target.humanoid_roles:
        if binding.role == role:
            return binding.node_name
    return None


def _optional_chain_warnings(
    source: BvhMotion,
    source_lookup: dict[str, str],
    target: RigInspection,
    role_to_target: dict[str, str],
    chain_by_name: dict[str, object],
) -> tuple[str, ...]:
    warnings: list[str] = []
    for chain_name, optional_roles in OPTIONAL_HUMANOID_CHAINS.items():
        chain = chain_by_name.get(chain_name)
        if chain is None:
            warnings.append(f"optional_chain_unavailable:{chain_name}")
            continue
        chain_roles = tuple(getattr(chain, "roles", ()))
        present_roles = tuple(role for role in chain_roles if role in optional_roles)
        if not present_roles:
            warnings.append(f"optional_chain_unusable:{chain_name}")
            continue
        covered = 0
        for role in present_roles:
            source_joint, _source_warnings = _source_joint_for_role(source, source_lookup, role)
            target_joint = role_to_target.get(role)
            if source_joint is None:
                warnings.append(f"optional_missing_source_role:{chain_name}:{role}")
                continue
            if target_joint is None:
                warnings.append(f"optional_missing_target_role:{chain_name}:{role}")
                continue
            if _source_joint_basis(source, source_joint) is None:
                warnings.append(f"optional_missing_source_basis:{chain_name}:{role}:{source_joint}")
                continue
            if _target_joint_basis(target, role, target_joint) is None:
                warnings.append(f"optional_missing_target_basis:{chain_name}:{role}:{target_joint}")
                continue
            covered += 1
        if covered != len(present_roles):
            warnings.append(f"optional_chain_coverage:{chain_name}:{covered}/{len(present_roles)}")
    return _unique_tuple(tuple(warnings))


def _build_optional_proximal_chain_mappings(
    source: BvhMotion,
    source_lookup: dict[str, str],
    target: RigInspection,
    role_to_target: dict[str, str],
    chain_by_name: dict[str, object],
    pairs: dict[str, str],
    basis_sources: dict[str, dict[str, str]],
) -> tuple[list[ChainMapping], tuple[str, ...]]:
    mappings: list[ChainMapping] = []
    warnings: list[str] = []
    for chain_name, optional_roles in OPTIONAL_PROXIMAL_HUMANOID_CHAINS.items():
        chain = chain_by_name.get(chain_name)
        if chain is None:
            warnings.append(f"optional_chain_unavailable:{chain_name}")
            continue
        chain_roles = tuple(getattr(chain, "roles", ()))
        present_roles = tuple(role for role in chain_roles if role in optional_roles)
        if present_roles != optional_roles:
            warnings.append(f"optional_chain_unusable:{chain_name}")
            continue
        source_joints: list[str] = []
        target_joints: list[str] = []
        for role in optional_roles:
            source_joint, source_warnings = _source_joint_for_role(source, source_lookup, role)
            warnings.extend(source_warnings)
            target_joint = role_to_target.get(role)
            if source_joint is None:
                warnings.append(f"optional_missing_source_role:{chain_name}:{role}")
                continue
            if target_joint is None:
                warnings.append(f"optional_missing_target_role:{chain_name}:{role}")
                continue
            source_basis = _source_joint_basis(source, source_joint)
            target_basis = _target_joint_basis(target, role, target_joint)
            if source_basis is None:
                warnings.append(f"optional_missing_source_basis:{chain_name}:{role}:{source_joint}")
                continue
            if target_basis is None:
                warnings.append(f"optional_missing_target_basis:{chain_name}:{role}:{target_joint}")
                continue
            warnings.extend(source_basis.warnings)
            warnings.extend(target_basis.warnings)
            source_joints.append(source_joint)
            target_joints.append(target_joint)
            pairs.setdefault(source_joint, target_joint)
            basis_sources.setdefault(
                target_joint,
                {
                    "source": source_basis.source,
                    "target": target_basis.source,
                    "role": role,
                    "source_quaternion": source_basis.quaternion,
                    "target_quaternion": target_basis.quaternion,
                },
            )
        coverage = f"{len(source_joints)}/{len(optional_roles)}"
        if len(source_joints) == len(optional_roles):
            mappings.append(
                ChainMapping(
                    chain_name=chain_name,
                    source_joints=tuple(source_joints),
                    target_joints=tuple(target_joints),
                    required=False,
                    coverage=coverage,
                    roles=optional_roles,
                )
            )
        else:
            warnings.append(f"optional_chain_coverage:{chain_name}:{coverage}")
    return mappings, _unique_tuple(tuple(warnings))


def _rest_vector_for_role(
    raw_rest: object,
    role: str,
    joint_name: str,
    *,
    child_role: str | None = None,
    child_joint_name: str | None = None,
) -> tuple[tuple[float, float, float], str] | None:
    if not isinstance(raw_rest, dict):
        return None

    if child_role is not None:
        child_vector = _rest_vector_for_keys(raw_rest, child_role, child_joint_name)
        if child_vector is not None:
            return child_vector, f"target_rest_child_transform:{role}->{child_role}"

    own_vector = _rest_vector_for_keys(raw_rest, role, joint_name)
    if own_vector is not None:
        return own_vector, f"target_rest_transform:{role}"
    return None


def _rest_matrix_basis_for_role(raw_rest: object, role: str, joint_name: str) -> tuple[tuple[float, float, float, float], str] | None:
    if not isinstance(raw_rest, dict):
        return None
    for section_name in ("roles", "joints", "nodes", "transforms"):
        section = raw_rest.get(section_name)
        if not isinstance(section, dict):
            continue
        basis = _rest_matrix_basis_from_entry(section.get(role), role)
        if basis is None:
            basis = _rest_matrix_basis_from_entry(section.get(joint_name), role)
        if basis is not None:
            return basis
    basis = _rest_matrix_basis_from_entry(raw_rest.get(role), role)
    if basis is None:
        basis = _rest_matrix_basis_from_entry(raw_rest.get(joint_name), role)
    return basis


def _rest_matrix_basis_from_entry(entry: object, role: str) -> tuple[tuple[float, float, float, float], str] | None:
    if not isinstance(entry, dict):
        return None
    for key in ("rest_local", "rest_world"):
        matrix = _parse_matrix4(entry.get(key))
        if matrix is None:
            continue
        quaternion = _quaternion_from_matrix3(matrix)
        if quaternion is not None:
            return quaternion, f"target_rest_matrix:{key}:{role}"
    return None


def _parse_matrix4(value: object) -> tuple[tuple[float, float, float, float], ...] | None:
    try:
        if isinstance(value, (list, tuple)) and len(value) == 4 and all(isinstance(row, (list, tuple)) and len(row) == 4 for row in value):
            return tuple(tuple(float(component) for component in row) for row in value)
        if isinstance(value, (list, tuple)) and len(value) == 16:
            components = [float(component) for component in value]
            return tuple(tuple(components[index:index + 4]) for index in range(0, 16, 4))
    except (TypeError, ValueError):
        return None
    return None


def _quaternion_from_matrix3(matrix: tuple[tuple[float, float, float, float], ...]) -> tuple[float, float, float, float] | None:
    rows = tuple(tuple(row[column] for column in range(3)) for row in matrix[:3])
    if not _is_orthonormal_matrix3(rows):
        return None
    m00, m01, m02 = rows[0]
    m10, m11, m12 = rows[1]
    m20, m21, m22 = rows[2]
    trace = m00 + m11 + m22
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (m21 - m12) / scale
        y = (m02 - m20) / scale
        z = (m10 - m01) / scale
    elif m00 > m11 and m00 > m22:
        scale = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (m21 - m12) / scale
        x = 0.25 * scale
        y = (m01 + m10) / scale
        z = (m02 + m20) / scale
    elif m11 > m22:
        scale = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (m02 - m20) / scale
        x = (m01 + m10) / scale
        y = 0.25 * scale
        z = (m12 + m21) / scale
    else:
        scale = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (m10 - m01) / scale
        x = (m02 + m20) / scale
        y = (m12 + m21) / scale
        z = 0.25 * scale
    return _normalize_quaternion_xyzw((x, y, z, w))


def _is_orthonormal_matrix3(rows: tuple[tuple[float, float, float], ...]) -> bool:
    columns = tuple(tuple(rows[row][column] for row in range(3)) for column in range(3))
    for column in columns:
        if abs(_vector_length(column) - 1.0) > 1e-4:
            return False
    for left_index in range(3):
        for right_index in range(left_index + 1, 3):
            if abs(_dot(columns[left_index], columns[right_index])) > 1e-4:
                return False
    determinant = _dot(columns[0], _cross(columns[1], columns[2]))
    return abs(determinant - 1.0) <= 1e-4


def _rest_vector_for_keys(raw_rest: dict[str, object], role: str, joint_name: str | None) -> tuple[float, float, float] | None:
    for section_name in ("roles", "joints", "nodes", "transforms"):
        section = raw_rest.get(section_name)
        if not isinstance(section, dict):
            continue
        vector = _rest_vector_from_entry(section.get(role))
        if vector is None and joint_name is not None:
            vector = _rest_vector_from_entry(section.get(joint_name))
        if vector is not None:
            return vector
    vector = _rest_vector_from_entry(raw_rest.get(role))
    if vector is None and joint_name is not None:
        vector = _rest_vector_from_entry(raw_rest.get(joint_name))
    return vector


def _rest_vector_from_entry(entry: object) -> tuple[float, float, float] | None:
    if isinstance(entry, dict):
        for key in ("child_offset", "offset", "local_translation", "translation"):
            vector = _parse_vector3(entry.get(key))
            if vector is not None and _vector_length(vector) > 1e-9:
                return vector
    return _parse_vector3(entry)


def _parse_vector3(value: object) -> tuple[float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    try:
        return tuple(float(component) for component in value)
    except (TypeError, ValueError):
        return None


def _basis_hint_for_role(raw_hints: object, role: str) -> tuple[tuple[float, float, float, float], str] | None:
    if not isinstance(raw_hints, dict):
        return None
    role_hint = raw_hints.get(role)
    if isinstance(raw_hints.get("roles"), dict):
        role_hint = raw_hints["roles"].get(role, role_hint)
    if isinstance(role_hint, dict):
        quaternion = role_hint.get("basis_quaternion") or role_hint.get("quaternion")
        parsed = _parse_quaternion(quaternion)
        if parsed is not None:
            return parsed, f"target_basis_hint:{role}"
    parsed = _parse_quaternion(raw_hints.get("basis_quaternion") or raw_hints.get("quaternion"))
    if parsed is not None:
        return parsed, "target_basis_hint:global_quaternion"
    if raw_hints.get("up_axis") in {"X", "Y", "Z", "x", "y", "z"}:
        return (0.0, 0.0, 0.0, 1.0), f"target_basis_hint:up_axis:{raw_hints.get('up_axis')}"
    return None


def _parse_quaternion(value: object) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        return _normalize_quaternion_xyzw(tuple(float(component) for component in value))
    except (TypeError, ValueError):
        return None


def _build_calibrated_rotation_channels(source: BvhMotion, mapping: JointMapping, times: tuple[float, ...]) -> tuple[RetargetChannel, ...]:
    rotation_indices = _rotation_indices(source)
    target_to_role = {target_joint: values.get("role") for target_joint, values in mapping.basis_sources.items()}
    channels: list[RetargetChannel] = []
    for source_joint, target_joint in mapping.pairs.items():
        joint_rotation_indices = rotation_indices.get(source_joint)
        if not joint_rotation_indices:
            continue
        role = str(target_to_role.get(target_joint) or "")
        source_basis = _source_basis_quaternion_from_mapping(mapping, target_joint)
        target_basis = _target_basis_quaternion_from_mapping(mapping, target_joint)
        if _target_basis_uses_rest_matrix(mapping, target_joint):
            calibration = _multiply_quaternion_xyzw(_inverse_quaternion_xyzw(target_basis), source_basis)
            composition_order = FULL_MATRIX_COMPOSITION_ORDER
            target_basis_composition = FULL_MATRIX_TARGET_BASIS_COMPOSITION
        else:
            calibration = _multiply_quaternion_xyzw(target_basis, _inverse_quaternion_xyzw(source_basis))
            composition_order = COMPOSITION_ORDER
            target_basis_composition = DEFAULT_TARGET_BASIS_COMPOSITION
        samples = [_bvh_euler_degrees_to_quaternion_xyzw(tuple((channel, frame[index]) for index, channel in joint_rotation_indices)) for frame in source.frames]
        first_inverse = _inverse_quaternion_xyzw(samples[0]) if samples else (0.0, 0.0, 0.0, 1.0)
        values = []
        for sample in samples:
            source_delta = _multiply_quaternion_xyzw(sample, first_inverse)
            target_delta = _multiply_quaternion_xyzw(_multiply_quaternion_xyzw(calibration, source_delta), _inverse_quaternion_xyzw(calibration))
            values.append(_normalize_quaternion_xyzw(target_delta))
        channels.append(
            _apply_motion_quality_filter(
                RetargetChannel(
                    target_joint=target_joint,
                    path="rotation",
                    times=times,
                    values=tuple(values),
                    diagnostics={
                        "source_joint": source_joint,
                        "target_joint": target_joint,
                        "role": role,
                        "composition_order": composition_order,
                        "target_basis_composition": target_basis_composition,
                        "calibration_quaternion": calibration,
                    },
                )
            )
        )
    return tuple(channels)


def foot_trajectory_diagnostics(
    *,
    source_feet: dict[str, tuple[tuple[float, float, float], ...]],
    retarget_feet: dict[str, tuple[tuple[float, float, float], ...]],
) -> dict[str, object]:
    """Report forward/lateral dominance for source and retarget foot paths.

    Coordinates intentionally stay in the project's existing horizontal convention:
    X is lateral and Z is forward. The function is pure so diagnostic tests can
    exercise it without requiring GLB export or runtime assets.
    """

    source = {side: _trajectory_axis_metrics(samples) for side, samples in source_feet.items()}
    retarget = {side: _trajectory_axis_metrics(samples) for side, samples in retarget_feet.items()}
    sides = sorted(set(source) & set(retarget))
    mismatched_sides = [side for side in sides if source[side]["dominant_axis"] != retarget[side]["dominant_axis"]]
    return {
        "coordinate_convention": {"lateral_axis": "x", "forward_axis": "z"},
        "source": source,
        "retarget": retarget,
        "alignment": "mismatch" if mismatched_sides else "aligned",
        "mismatched_sides": mismatched_sides,
    }


def stance_foot_slide_diagnostics(
    foot_samples: dict[str, tuple[tuple[float, float, float], ...]],
    foot_contacts: tuple[tuple[float, ...], ...] | None,
    *,
    contact_threshold: float = 0.5,
) -> dict[str, object]:
    """Measure horizontal stance-foot slide over contact windows."""

    shape = _array_shape(foot_contacts)
    ratio = _contact_ratio(foot_contacts)
    result: dict[str, object] = {
        "contact_shape": list(shape),
        "contact_ratio": _clean_float(ratio),
        "contact_threshold": contact_threshold,
    }
    max_slide = 0.0
    total_windows = 0
    for side in ("left", "right"):
        samples = foot_samples.get(side, ())
        windows = _contact_windows(foot_contacts, side=side, frame_count=len(samples), contact_threshold=contact_threshold)
        window_metrics = [_stance_window_slide(samples, start, end) for start, end in windows]
        side_max = max(window_metrics) if window_metrics else 0.0
        max_slide = max(max_slide, side_max)
        total_windows += len(windows)
        result[side] = {
            "windows": [[start, end] for start, end in windows],
            "window_count": len(windows),
            "max_slide": _clean_float(side_max),
            "window_slides": [_clean_float(value) for value in window_metrics],
        }
    result["summary"] = {
        "window_count": total_windows,
        "max_slide": _clean_float(max_slide),
    }
    return result


def apply_contact_aware_foot_locking(
    foot_samples: dict[str, tuple[tuple[float, float, float], ...]],
    foot_contacts: tuple[tuple[float, ...], ...] | None,
    *,
    contact_threshold: float = 0.5,
) -> tuple[dict[str, tuple[tuple[float, float, float], ...]], dict[str, object]]:
    """Lock stance windows to their first foot sample and report pre/post slide.

    This is the conservative phase-2 primitive: it operates on explicit target or
    proxy foot trajectories and does not pretend to solve full world-space IK.
    """

    pre = stance_foot_slide_diagnostics(foot_samples, foot_contacts, contact_threshold=contact_threshold)
    locked: dict[str, tuple[tuple[float, float, float], ...]] = {}
    applied = False
    for side, samples in foot_samples.items():
        mutable = [tuple(float(component) for component in sample) for sample in samples]
        for start, end in _contact_windows(foot_contacts, side=side, frame_count=len(samples), contact_threshold=contact_threshold):
            if 0 <= start < len(mutable):
                anchor = mutable[start]
                for index in range(start, min(end + 1, len(mutable))):
                    if mutable[index] != anchor:
                        applied = True
                    mutable[index] = anchor
        locked[side] = tuple(mutable)
    post = stance_foot_slide_diagnostics(locked, foot_contacts, contact_threshold=contact_threshold)
    diagnostics = {
        "status": "applied" if applied else "not_applied",
        "algorithm": "stance_window_anchor_lock",
        "pre": pre,
        "post": post,
        "deferred": ["world_space_leg_ik"],
    }
    return locked, diagnostics


VISUAL_QUALITY_VERSION = "kimodo.visual_quality.v1"
VISUAL_QUALITY_LIMITATION = "Diagnostics do not prove visual success."


def visual_quality_diagnostics(
    clip: RetargetClip,
    *,
    motion_context: MotionContext | None = None,
    foot_contacts: tuple[tuple[float, ...], ...] | None = None,
    world_foot_samples: dict[str, tuple[tuple[float, float, float], ...]] | None = None,
    source_euler_degrees: dict[str, tuple[float, ...]] | None = None,
    foot_slide_threshold: float = 0.05,
) -> dict[str, object]:
    """Compute warning-only visual-quality diagnostics for a retarget clip.

    This function is intentionally diagnostics-only. It never mutates animation
    samples, never applies IK/foot locking, and its output is additive metadata.
    """

    contacts = foot_contacts if foot_contacts is not None else (motion_context.foot_contacts if motion_context is not None else None)
    warnings: list[str] = []
    limitations = [VISUAL_QUALITY_LIMITATION]
    angular_checks = _angular_velocity_checks(clip)
    for check in angular_checks:
        if int(check["spike_count"] or 0) <= 0:
            continue
        code = _angular_warning_code(str(check.get("role") or ""))
        if code:
            warnings.append(code)
    foot_contact = _visual_foot_contact_check(
        contacts,
        world_foot_samples=world_foot_samples,
        slide_threshold=foot_slide_threshold,
    )
    for code in foot_contact.get("warnings", []):
        warnings.append(str(code))
    euler_advisories = _euler_wrap_advisories(source_euler_degrees or {}, angular_checks)
    namespaced_warnings = _unique_list(warnings)
    status = "warning" if any(code not in {"visual_quality/foot_contact_proxy_only", "visual_quality/foot_contact_not_evaluated"} for code in namespaced_warnings) else "not_evaluated"
    if not namespaced_warnings and angular_checks and foot_contact.get("confidence") == "measured":
        status = "ok"
    checks: dict[str, object] = {
        "angular_velocity": angular_checks,
        "foot_contact": {key: value for key, value in foot_contact.items() if key != "warnings"},
    }
    if euler_advisories:
        checks["euler_wrap_advisory"] = euler_advisories
    return {
        "version": VISUAL_QUALITY_VERSION,
        "blocking": False,
        "status": status,
        "warnings": namespaced_warnings,
        "limitations": limitations,
        "summary": {
            "warning_count": len(namespaced_warnings),
            "angular_channel_count": len(angular_checks),
            "foot_contact_confidence": foot_contact.get("confidence"),
        },
        "checks": checks,
    }


def _safe_visual_quality_diagnostics(clip: RetargetClip, *, motion_context: MotionContext | None) -> dict[str, object]:
    try:
        return visual_quality_diagnostics(clip, motion_context=motion_context)
    except Exception as exc:  # noqa: BLE001 - diagnostics must never block export
        return {
            "version": VISUAL_QUALITY_VERSION,
            "blocking": False,
            "status": "not_evaluated",
            "warnings": ["visual_quality/metrics_not_evaluated"],
            "limitations": [VISUAL_QUALITY_LIMITATION],
            "summary": {"warning_count": 1, "foot_contact_confidence": "not_evaluated"},
            "checks": {
                "angular_velocity": [],
                "foot_contact": {"status": "not_evaluated", "confidence": "not_evaluated", "reason": "metrics_exception"},
            },
            "reason": f"{type(exc).__name__}: {exc}",
        }


def _angular_velocity_checks(clip: RetargetClip) -> list[dict[str, object]]:
    checks: list[dict[str, object]] = []
    frame_time = float(getattr(clip, "frame_time", 0.0) or 0.0)
    if frame_time <= 0.0:
        return checks
    for channel in clip.channels:
        if channel.path != "rotation" or len(channel.values) < 2:
            continue
        role = str(channel.diagnostics.get("role") or "")
        threshold = _visual_angular_threshold_for_role(role)
        if threshold is None:
            continue
        velocities = _quaternion_angular_velocities_deg_s(channel.values, frame_time)
        spike_frames = [index + 1 for index, value in enumerate(velocities) if value > threshold]
        checks.append(
            {
                "role": role,
                "target_joint": channel.target_joint,
                "unit": "deg/s",
                "threshold_deg_s": _clean_float(threshold),
                "max_deg_s": _clean_float(max(velocities) if velocities else 0.0),
                "p95_deg_s": _clean_float(_percentile95(velocities)),
                "spike_count": len(spike_frames),
                "frames": spike_frames,
            }
        )
    return checks


def _quaternion_angular_velocities_deg_s(values: tuple[tuple[float, ...], ...], frame_time: float) -> list[float]:
    normalized = [_normalize_quaternion_xyzw(tuple(float(component) for component in value)) for value in values]
    continuous = [normalized[0]]
    for value in normalized[1:]:
        continuous.append(_same_hemisphere_quaternion(value, continuous[-1]))
    velocities = []
    for index in range(1, len(continuous)):
        delta = _multiply_quaternion_xyzw(continuous[index], _inverse_quaternion_xyzw(continuous[index - 1]))
        velocities.append(math.degrees(_quaternion_angle_radians(_normalize_quaternion_xyzw(delta))) / frame_time)
    return velocities


def _visual_angular_threshold_for_role(role: str) -> float | None:
    settings = _motion_quality_settings_for_role(role)
    if settings is None:
        return None
    return float(settings[1]) * 30.0


def _angular_warning_code(role: str) -> str | None:
    if any(part in role for part in ("leg", "foot")):
        return "visual_quality/leg_angular_velocity_spike"
    if any(part in role for part in ("arm", "forearm", "hand", "shoulder")):
        return "visual_quality/arm_angular_velocity_spike"
    return None


def _percentile95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return ordered[index]


def _visual_foot_contact_check(
    foot_contacts: tuple[tuple[float, ...], ...] | None,
    *,
    world_foot_samples: dict[str, tuple[tuple[float, float, float], ...]] | None,
    slide_threshold: float,
) -> dict[str, object]:
    if not foot_contacts:
        return {
            "status": "not_evaluated",
            "confidence": "not_evaluated",
            "reason": "missing_contact_data",
            "warnings": ["visual_quality/foot_contact_not_evaluated"],
        }
    shape = _array_shape(foot_contacts)
    if world_foot_samples is None:
        return {
            "status": "not_evaluated",
            "confidence": "proxy_only",
            "reason": "world_space_foot_samples_unavailable",
            "contact_shape": list(shape),
            "contact_ratio": _clean_float(_contact_ratio(foot_contacts)),
            "warnings": ["visual_quality/foot_contact_proxy_only"],
        }
    if any(len(samples) != len(foot_contacts) for samples in world_foot_samples.values()):
        return {
            "status": "not_evaluated",
            "confidence": "not_evaluated",
            "reason": "foot_samples_contact_frame_count_mismatch",
            "contact_shape": list(shape),
            "warnings": ["visual_quality/foot_contact_not_evaluated"],
        }
    slide = stance_foot_slide_diagnostics(world_foot_samples, foot_contacts)
    max_slide = float(slide["summary"]["max_slide"])
    warnings = ["visual_quality/foot_contact_slide"] if max_slide > slide_threshold else []
    return {
        "status": "warning" if warnings else "ok",
        "confidence": "measured",
        "contact_shape": slide["contact_shape"],
        "contact_ratio": slide["contact_ratio"],
        "contact_windows": {"left": slide["left"]["windows"], "right": slide["right"]["windows"]},
        "max_slide": _clean_float(max_slide),
        "threshold": slide_threshold,
        "warnings": warnings,
    }


def _euler_wrap_advisories(source_euler_degrees: dict[str, tuple[float, ...]], angular_checks: list[dict[str, object]]) -> list[dict[str, object]]:
    by_role = {str(check.get("role")): check for check in angular_checks}
    advisories: list[dict[str, object]] = []
    for role, samples in source_euler_degrees.items():
        if not _has_euler_wrap(samples):
            continue
        check = by_role.get(role)
        if check is not None and int(check.get("spike_count") or 0) > 0:
            continue
        advisories.append({"role": role, "status": "advisory", "reason": "source_euler_wrap_with_acceptable_quaternion_continuity"})
    return advisories


def _has_euler_wrap(samples: tuple[float, ...]) -> bool:
    return any(abs(float(samples[index]) - float(samples[index - 1])) >= 180.0 for index in range(1, len(samples)))


def _trajectory_axis_metrics(samples: tuple[tuple[float, float, float], ...]) -> dict[str, object]:
    if not samples:
        return {"sample_count": 0, "lateral_range": 0.0, "forward_range": 0.0, "dominant_axis": "none"}
    xs = [float(sample[0]) for sample in samples]
    zs = [float(sample[2]) for sample in samples]
    lateral_range = max(xs) - min(xs)
    forward_range = max(zs) - min(zs)
    dominant = "forward" if forward_range >= lateral_range else "lateral"
    if max(forward_range, lateral_range) <= 1e-9:
        dominant = "none"
    return {
        "sample_count": len(samples),
        "lateral_range": _clean_float(lateral_range),
        "forward_range": _clean_float(forward_range),
        "dominant_axis": dominant,
    }


def _contact_locking_metadata_for_context(motion_context: MotionContext | None, *, frame_count: int) -> dict[str, object] | None:
    if motion_context is None or motion_context.foot_contacts is None:
        return None
    if len(motion_context.foot_contacts) != frame_count:
        return {
            "status": "not_applied",
            "reason": "foot_contacts_frame_count_mismatch",
            "contact_shape": list(motion_context.foot_contact_shape),
            "contact_ratio": _clean_float(motion_context.foot_contact_ratio),
            "deferred": ["world_space_leg_ik"],
        }
    diagnostics = stance_foot_slide_diagnostics(
        {
            "left": tuple((0.0, 0.0, 0.0) for _ in range(frame_count)),
            "right": tuple((0.0, 0.0, 0.0) for _ in range(frame_count)),
        },
        motion_context.foot_contacts,
    )
    return {
        "status": "metadata_only",
        "reason": "no_target_foot_translation_channels_available_for_world_space_lock",
        "algorithm": "stance_window_anchor_lock",
        "contact_shape": diagnostics["contact_shape"],
        "contact_ratio": diagnostics["contact_ratio"],
        "left_windows": diagnostics["left"]["windows"],
        "right_windows": diagnostics["right"]["windows"],
        "deferred": ["world_space_leg_ik"],
    }


def _contact_ratio(foot_contacts: tuple[tuple[float, ...], ...] | None) -> float:
    if not foot_contacts:
        return 0.0
    values = [float(value) for row in foot_contacts for value in row]
    return sum(values) / len(values) if values else 0.0


def _contact_windows(
    foot_contacts: tuple[tuple[float, ...], ...] | None,
    *,
    side: str,
    frame_count: int,
    contact_threshold: float,
) -> list[tuple[int, int]]:
    if not foot_contacts or frame_count <= 0:
        return []
    column = 0 if side == "left" else 1
    active = []
    for index in range(min(frame_count, len(foot_contacts))):
        row = foot_contacts[index]
        active.append(len(row) > column and float(row[column]) >= contact_threshold)
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


def _stance_window_slide(samples: tuple[tuple[float, float, float], ...], start: int, end: int) -> float:
    if not samples or start >= len(samples):
        return 0.0
    anchor = samples[start]
    max_distance = 0.0
    for index in range(start, min(end + 1, len(samples))):
        sample = samples[index]
        max_distance = max(max_distance, math.hypot(float(sample[0]) - float(anchor[0]), float(sample[2]) - float(anchor[2])))
    return max_distance


def _apply_motion_quality_filter(channel: RetargetChannel) -> RetargetChannel:
    if channel.path != "rotation":
        return channel
    role = str(channel.diagnostics.get("role", ""))
    settings = _motion_quality_settings_for_role(role)
    if settings is None or len(channel.values) < 2:
        return channel

    max_angle_degrees, max_jump_degrees = settings
    pre_values = tuple(tuple(float(component) for component in value) for value in channel.values)
    max_pre_amplitude_degrees = _max_quaternion_amplitude_degrees(pre_values)
    max_pre_frame_jump_degrees = _max_quaternion_frame_jump_degrees(pre_values)
    max_angle_radians = math.radians(max_angle_degrees)
    max_jump_radians = math.radians(max_jump_degrees - 1e-6)
    identity = (0.0, 0.0, 0.0, 1.0)
    filtered: list[tuple[float, float, float, float]] = [channel.values[0]]
    changed = False
    changed_sample_count = 0
    clamp_count = 0
    jump_reduction_count = 0
    for raw_value in channel.values[1:]:
        candidate = _normalize_quaternion_xyzw(tuple(float(component) for component in raw_value))
        candidate = _same_hemisphere_quaternion(candidate, filtered[-1])
        candidate, amplitude_changed = _limit_quaternion_delta(identity, candidate, max_angle_radians)
        candidate = _same_hemisphere_quaternion(candidate, filtered[-1])
        candidate, jump_changed = _limit_quaternion_delta(filtered[-1], candidate, max_jump_radians)
        candidate, final_amplitude_changed = _limit_quaternion_delta(identity, candidate, max_angle_radians)
        filtered.append(candidate)
        if amplitude_changed or jump_changed or final_amplitude_changed:
            changed_sample_count += 1
        if amplitude_changed or final_amplitude_changed:
            clamp_count += int(amplitude_changed) + int(final_amplitude_changed)
        if jump_changed:
            jump_reduction_count += 1
        changed = changed or amplitude_changed or jump_changed or final_amplitude_changed

    diagnostics = dict(channel.diagnostics)
    diagnostics.update(
        {
            "motion_quality_filter": "role_aware_conservative",
            "motion_quality_role": role,
            "motion_quality_filter_changed_samples": changed,
            "motion_quality_changed_sample_count": changed_sample_count,
            "motion_quality_clamp_count": clamp_count,
            "motion_quality_jump_reduction_count": jump_reduction_count,
            "motion_quality_max_angle_degrees": max_angle_degrees,
            "motion_quality_max_frame_jump_degrees": max_jump_degrees,
            "motion_quality_max_pre_amplitude_degrees": max_pre_amplitude_degrees,
            "motion_quality_max_post_amplitude_degrees": _max_quaternion_amplitude_degrees(tuple(filtered)),
            "motion_quality_max_pre_frame_jump_degrees": max_pre_frame_jump_degrees,
            "motion_quality_max_post_frame_jump_degrees": _max_quaternion_frame_jump_degrees(tuple(filtered)),
        }
    )
    return RetargetChannel(
        target_joint=channel.target_joint,
        path=channel.path,
        times=channel.times,
        values=tuple(filtered),
        interpolation=channel.interpolation,
        diagnostics=diagnostics,
    )


def _motion_quality_diagnostics_metadata(channels: list[RetargetChannel]) -> dict[str, object] | None:
    entries = []
    for channel in channels:
        diagnostics = channel.diagnostics
        if channel.path != "rotation" or diagnostics.get("motion_quality_filter") != "role_aware_conservative":
            continue
        entries.append(
            {
                "role": diagnostics.get("role"),
                "target_joint": channel.target_joint,
                "path": channel.path,
                "filter_profile": diagnostics.get("motion_quality_filter"),
                "max_angle_degrees": diagnostics.get("motion_quality_max_angle_degrees"),
                "max_frame_jump_degrees": diagnostics.get("motion_quality_max_frame_jump_degrees"),
                "filter_changed_samples": diagnostics.get("motion_quality_filter_changed_samples"),
                "changed_sample_count": diagnostics.get("motion_quality_changed_sample_count"),
                "filtered_sample_count": diagnostics.get("motion_quality_changed_sample_count"),
                "clamp_count": diagnostics.get("motion_quality_clamp_count"),
                "jump_reduction_count": diagnostics.get("motion_quality_jump_reduction_count"),
                "smoothing_reduction_count": diagnostics.get("motion_quality_jump_reduction_count"),
                "max_pre_amplitude_degrees": diagnostics.get("motion_quality_max_pre_amplitude_degrees"),
                "max_post_amplitude_degrees": diagnostics.get("motion_quality_max_post_amplitude_degrees"),
                "max_pre_frame_jump_degrees": diagnostics.get("motion_quality_max_pre_frame_jump_degrees"),
                "max_post_frame_jump_degrees": diagnostics.get("motion_quality_max_post_frame_jump_degrees"),
            }
        )
    if not entries:
        return None
    return {
        "motion_quality_filter": "role_aware_conservative",
        "summary": {
            "channel_count": len(entries),
            "changed_channel_count": sum(1 for entry in entries if entry["filter_changed_samples"]),
            "changed_sample_count": sum(int(entry["changed_sample_count"] or 0) for entry in entries),
            "clamp_count": sum(int(entry["clamp_count"] or 0) for entry in entries),
            "jump_reduction_count": sum(int(entry["jump_reduction_count"] or 0) for entry in entries),
        },
        "left_right_summary": _motion_quality_left_right_summary(entries),
        "channels": entries,
    }


def _motion_quality_left_right_summary(entries: list[dict[str, object]]) -> dict[str, object]:
    by_role = {str(entry.get("role")): entry for entry in entries}
    summary = {}
    for suffix in ("shoulder", "upper_arm", "forearm", "hand", "upper_leg", "lower_leg", "foot"):
        left = by_role.get(f"left_{suffix}")
        right = by_role.get(f"right_{suffix}")
        if left is None or right is None:
            continue
        summary[suffix] = {
            "symmetric_settings": left.get("filter_profile") == right.get("filter_profile")
            and left.get("max_angle_degrees") == right.get("max_angle_degrees")
            and left.get("max_frame_jump_degrees") == right.get("max_frame_jump_degrees"),
            "left_target_joint": left.get("target_joint"),
            "right_target_joint": right.get("target_joint"),
            "left_changed_sample_count": left.get("changed_sample_count"),
            "right_changed_sample_count": right.get("changed_sample_count"),
        }
    return summary


def _max_quaternion_amplitude_degrees(values: tuple[tuple[float, ...], ...]) -> float:
    if not values:
        return 0.0
    return max(math.degrees(_quaternion_angle_radians(_normalize_quaternion_xyzw(tuple(value)))) for value in values)


def _max_quaternion_frame_jump_degrees(values: tuple[tuple[float, ...], ...]) -> float:
    if len(values) < 2:
        return 0.0
    jumps = []
    for index in range(1, len(values)):
        current = _normalize_quaternion_xyzw(tuple(values[index]))
        previous = _normalize_quaternion_xyzw(tuple(values[index - 1]))
        delta = _multiply_quaternion_xyzw(current, _inverse_quaternion_xyzw(previous))
        jumps.append(math.degrees(_quaternion_angle_radians(delta)))
    return max(jumps)


def _motion_quality_settings_for_role(role: str) -> tuple[float, float] | None:
    if role == "hips":
        return (45.0, 12.0)
    if role == "spine":
        return (35.0, 10.0)
    if role == "chest":
        return (45.0, 12.0)
    if role == "neck":
        return (35.0, 12.0)
    if role == "head":
        return (45.0, 16.0)
    if role.endswith("_foot"):
        return (70.0, 14.0)
    if role.endswith("_lower_leg"):
        return (125.0, 18.0)
    if role.endswith("_upper_leg"):
        return (110.0, 20.0)
    if role.endswith("_hand"):
        return (65.0, 12.0)
    if role.endswith("_forearm") or role.endswith("_lower_arm"):
        return (75.0, 16.0)
    if role.endswith("_upper_arm"):
        return (120.0, 180.0)
    if role.endswith("_shoulder"):
        return (90.0, 180.0)
    return None


def _limit_quaternion_delta(
    reference: tuple[float, float, float, float],
    candidate: tuple[float, float, float, float],
    max_radians: float,
) -> tuple[tuple[float, float, float, float], bool]:
    delta = _multiply_quaternion_xyzw(candidate, _inverse_quaternion_xyzw(reference))
    delta = _same_hemisphere_quaternion(_normalize_quaternion_xyzw(delta), (0.0, 0.0, 0.0, 1.0))
    angle = _quaternion_angle_radians(delta)
    if angle <= max_radians or angle <= 1e-9:
        return candidate, False
    limited_delta = _slerp_quaternion_xyzw((0.0, 0.0, 0.0, 1.0), delta, max_radians / angle)
    return _normalize_quaternion_xyzw(_multiply_quaternion_xyzw(limited_delta, reference)), True


def _quaternion_angle_radians(quaternion: tuple[float, float, float, float]) -> float:
    return 2.0 * math.acos(max(-1.0, min(1.0, abs(quaternion[3]))))


def _same_hemisphere_quaternion(
    quaternion: tuple[float, float, float, float],
    reference: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    if sum(left * right for left, right in zip(quaternion, reference)) < 0.0:
        return tuple(-component for component in quaternion)  # type: ignore[return-value]
    return quaternion


def _slerp_quaternion_xyzw(
    start: tuple[float, float, float, float],
    end: tuple[float, float, float, float],
    amount: float,
) -> tuple[float, float, float, float]:
    amount = max(0.0, min(1.0, amount))
    end = _same_hemisphere_quaternion(end, start)
    dot = max(-1.0, min(1.0, sum(left * right for left, right in zip(start, end))))
    if dot > 0.9995:
        return _normalize_quaternion_xyzw(tuple(start[index] + amount * (end[index] - start[index]) for index in range(4)))
    theta = math.acos(dot)
    sin_theta = math.sin(theta)
    start_scale = math.sin((1.0 - amount) * theta) / sin_theta
    end_scale = math.sin(amount * theta) / sin_theta
    return _normalize_quaternion_xyzw(tuple(start_scale * start[index] + end_scale * end[index] for index in range(4)))


def _target_basis_uses_rest_matrix(mapping: JointMapping, target_joint: str) -> bool:
    diagnostics = mapping.basis_sources.get(target_joint, {})
    return str(diagnostics.get("target", "")).startswith("target_rest_matrix:")


def _source_basis_quaternion_from_mapping(mapping: JointMapping, target_joint: str) -> tuple[float, float, float, float]:
    diagnostics = mapping.basis_sources.get(target_joint, {})
    quaternion = diagnostics.get("source_quaternion")
    if isinstance(quaternion, tuple) and len(quaternion) == 4:
        return quaternion
    return (0.0, 0.0, 0.0, 1.0)


def _target_basis_quaternion_from_mapping(mapping: JointMapping, target_joint: str) -> tuple[float, float, float, float]:
    diagnostics = mapping.basis_sources.get(target_joint, {})
    quaternion = diagnostics.get("target_quaternion")
    if isinstance(quaternion, tuple) and len(quaternion) == 4:
        return quaternion
    return (0.0, 0.0, 0.0, 1.0)


def _vector_length(vector: tuple[float, float, float]) -> float:
    return math.sqrt(sum(component * component for component in vector))


def _normalize_vector(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    length = _vector_length(vector)
    if length <= 0.0:
        return (0.0, 1.0, 0.0)
    return (vector[0] / length, vector[1] / length, vector[2] / length)


def _cross(left: tuple[float, float, float], right: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _dot(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2]


def _quaternion_from_y_axis_to_vector(vector: tuple[float, float, float]) -> tuple[float, float, float, float]:
    source_axis = (0.0, 1.0, 0.0)
    target_axis = _normalize_vector(vector)
    dot = max(-1.0, min(1.0, _dot(source_axis, target_axis)))
    if dot > 1.0 - 1e-9:
        return (0.0, 0.0, 0.0, 1.0)
    if dot < -1.0 + 1e-9:
        return (1.0, 0.0, 0.0, 0.0)
    axis = _cross(source_axis, target_axis)
    quaternion = (axis[0], axis[1], axis[2], 1.0 + dot)
    return _normalize_quaternion_xyzw(quaternion)


def _rotation_indices(source: BvhMotion) -> dict[str, list[tuple[int, str]]]:
    rotation_indices: dict[str, list[tuple[int, str]]] = {}
    for index, (joint, channel) in enumerate(source.channel_order):
        if channel.lower().endswith("rotation"):
            rotation_indices.setdefault(joint, []).append((index, channel))
    return rotation_indices


def _calibration_metadata(target: RigInspection, mapping: JointMapping) -> dict[str, object]:
    serializable_basis = {}
    for joint, data in mapping.basis_sources.items():
        serializable_basis[joint] = {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in data.items()
        }
    target_basis_sources = tuple(str(data.get("target", "")) for data in mapping.basis_sources.values())
    uses_full_matrix_basis = any(source.startswith("target_rest_matrix:") for source in target_basis_sources)
    roll_unresolved = TARGET_REST_BASIS_ROLL_WARNING in mapping.warnings
    return {
        "version": CALIBRATION_VERSION,
        "sidecar_provenance": mapping.sidecar_provenance,
        "chains": [
            {
                "name": chain.chain_name,
                "roles": list(chain.roles),
                "source_joints": list(chain.source_joints),
                "target_joints": list(chain.target_joints),
                "required": chain.required,
                "coverage": chain.coverage,
            }
            for chain in mapping.chains
        ],
        "chain_coverage": dict(mapping.coverage),
        "basis_sources": serializable_basis,
        "composition_order": COMPOSITION_ORDER,
        "composition_orders": {
            "default": COMPOSITION_ORDER,
            "full_rest_matrix_basis": FULL_MATRIX_COMPOSITION_ORDER,
        },
        "target_basis_composition": {
            "default": DEFAULT_TARGET_BASIS_COMPOSITION,
            "full_rest_matrix_basis": FULL_MATRIX_TARGET_BASIS_COMPOSITION,
        },
        "warnings_by_severity": {
            "warning": [warning for warning in mapping.warnings if warning != "humanoid_contract_calibrated_mapping"],
            "info": ["humanoid_contract_calibrated_mapping"],
        },
        "basis_quality": _basis_quality_label(roll_unresolved, uses_full_matrix_basis),
        "visual_safety": _visual_safety_label(roll_unresolved, uses_full_matrix_basis),
        "fallback_widened": False,
        "contract_diagnostics": list(target.humanoid_diagnostics),
    }


def _basis_quality_label(roll_unresolved: bool, uses_full_matrix_basis: bool) -> str:
    if roll_unresolved:
        return "inferred_axis_roll_unresolved"
    if uses_full_matrix_basis:
        return "full_rest_matrix_basis"
    return "explicit_or_hint_based"


def _visual_safety_label(roll_unresolved: bool, uses_full_matrix_basis: bool) -> str:
    if roll_unresolved:
        return "not_validated"
    if uses_full_matrix_basis:
        return "not_validated_full_basis_requires_visual_review"
    return "basis_hint_only_not_visually_validated"


def _build_rotation_channels(source: BvhMotion, mapping: JointMapping, times: tuple[float, ...]) -> tuple[RetargetChannel, ...]:
    """Build bounded rotation channels for source joints with explicit mappings.

    This is intentionally not an arbitrary-joint fallback: only BVH joints already
    accepted by ``build_joint_mapping`` are considered. BVH rotation channels are
    interpreted as degrees in the order listed on that joint's CHANNELS line and
    composed as local-axis rotations. glTF expects quaternions as ``x, y, z, w``.
    The conversion is sufficient for structural MVP validation; it is not a full
    coordinate-system retargeting solution.
    """

    rotation_indices: dict[str, list[tuple[int, str]]] = {}
    for index, (joint, channel) in enumerate(source.channel_order):
        if channel.lower().endswith("rotation"):
            rotation_indices.setdefault(joint, []).append((index, channel))

    channels: list[RetargetChannel] = []
    for source_joint, target_joint in mapping.pairs.items():
        joint_rotation_indices = rotation_indices.get(source_joint)
        if not joint_rotation_indices:
            continue
        values = []
        for frame in source.frames:
            ordered_rotations = tuple((channel, frame[index]) for index, channel in joint_rotation_indices)
            values.append(_bvh_euler_degrees_to_quaternion_xyzw(ordered_rotations))
        channels.append(
            RetargetChannel(
                target_joint=target_joint,
                path="rotation",
                times=times,
                values=tuple(values),
            )
        )
    return tuple(channels)


def _bvh_euler_degrees_to_quaternion_xyzw(ordered_rotations: tuple[tuple[str, float], ...]) -> tuple[float, float, float, float]:
    quaternion = (0.0, 0.0, 0.0, 1.0)
    for channel, degrees in ordered_rotations:
        axis = channel[0].lower()
        quaternion = _multiply_quaternion_xyzw(quaternion, _axis_angle_quaternion_xyzw(axis, math.radians(float(degrees))))
    return _normalize_quaternion_xyzw(quaternion)


def _axis_angle_quaternion_xyzw(axis: str, radians: float) -> tuple[float, float, float, float]:
    half = radians / 2.0
    sine = math.sin(half)
    cosine = math.cos(half)
    if axis == "x":
        return (sine, 0.0, 0.0, cosine)
    if axis == "y":
        return (0.0, sine, 0.0, cosine)
    if axis == "z":
        return (0.0, 0.0, sine, cosine)
    return (0.0, 0.0, 0.0, 1.0)


def _multiply_quaternion_xyzw(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return (
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    )


def _inverse_quaternion_xyzw(quaternion: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x, y, z, w = quaternion
    length_squared = x * x + y * y + z * z + w * w
    if length_squared <= 0.0:
        return (0.0, 0.0, 0.0, 1.0)
    return (-x / length_squared, -y / length_squared, -z / length_squared, w / length_squared)


def _normalize_quaternion_xyzw(quaternion: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    length = math.sqrt(sum(component * component for component in quaternion))
    if length <= 0.0:
        return (0.0, 0.0, 0.0, 1.0)
    x, y, z, w = quaternion
    return (x / length, y / length, z / length, w / length)


def _unique_tuple(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _locomotion_policy_for_context(motion_context: MotionContext | None, *, frame_count: int, fallback_mode: str) -> dict[str, object]:
    if motion_context is None:
        return {
            "mode": fallback_mode,
            "source": "none",
            "detected_locomotion": False,
            "root_horizontal_range": 0.0,
            "root_displacement": [0.0, 0.0, 0.0],
            "foot_contact_shape": [],
            "foot_contact_ratio": 0.0,
            "warnings": ["motion_context_unavailable"],
            "deferred": ["contact_aware_foot_locking"],
        }
    return motion_context.locomotion_policy(frame_count=frame_count, fallback_mode=fallback_mode)


def _array_to_tuple(value: object) -> tuple[tuple[float, ...], ...]:
    try:
        return tuple(tuple(float(component) for component in row) for row in value)  # type: ignore[union-attr]
    except TypeError:
        return ()


def _valid_root_positions(value: tuple[tuple[float, ...], ...] | None) -> bool:
    return bool(value) and all(len(sample) == 3 for sample in value)


def _invalid_root_position_warnings(context: MotionContext) -> list[str]:
    warnings = []
    if "smooth_root_pos" in context.source_keys and not _valid_root_positions(context.smooth_root_pos):
        warnings.append("invalid_smooth_root_pos_shape")
    if "root_positions" in context.source_keys and not _valid_root_positions(context.root_positions):
        warnings.append("invalid_root_positions_shape")
    if "smooth_root_pos" not in context.source_keys and "root_positions" not in context.source_keys:
        warnings.append("root_positions_unavailable")
    return warnings


def _array_shape(value: tuple[tuple[float, ...], ...] | None) -> tuple[int, ...]:
    if value is None:
        return ()
    if not value:
        return (0,)
    return (len(value), len(value[0]))


def _clean_float(value: float) -> float:
    cleaned = round(float(value), 10)
    return 0.0 if abs(cleaned) < 1e-10 else cleaned


def _unique_list(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _root_motion_target(source: BvhMotion, mapping: JointMapping) -> str | None:
    root_target = mapping.pairs.get(source.root)
    if root_target is not None:
        return root_target

    if _normalize(source.root) != "root":
        return None

    root_joint = source.joints.get(source.root)
    if root_joint is None:
        return None

    for child_name in root_joint.children:
        if _normalize(child_name) in {"hips", "hip", "pelvis"} and child_name in mapping.pairs:
            return mapping.pairs[child_name]
    return None


def _root_motion_source_joint(source: BvhMotion, mapping: JointMapping) -> str | None:
    return source.root


def _suppressed_hips_root_motion_source(source: BvhMotion, mapping: JointMapping) -> str | None:
    if source.root in mapping.pairs:
        return None
    hips_child = _semantic_hips_child(source)
    if hips_child is None or hips_child not in mapping.pairs or not _position_indices_for_joint(source, hips_child):
        return None
    root_samples = _position_samples_for_joint(source, source.root)
    if root_samples and any(sample != root_samples[0] for sample in root_samples):
        return None
    return hips_child


def _semantic_hips_child(source: BvhMotion) -> str | None:
    if _normalize(source.root) != "root":
        return None
    root_joint = source.joints.get(source.root)
    if root_joint is None:
        return None
    for child_name in root_joint.children:
        if _normalize(child_name) in {"hips", "hip", "pelvis"}:
            return child_name
    return None


def _position_samples_for_joint(source: BvhMotion, joint_name: str | None) -> list[tuple[float, float, float]]:
    position_indices = _position_indices_for_joint(source, joint_name)
    translations = []
    for frame in source.frames:
        xyz = [0.0, 0.0, 0.0]
        for idx in position_indices:
            channel = source.channel_order[idx][1].lower()
            if channel.startswith("x"):
                xyz[0] = frame[idx]
            elif channel.startswith("y"):
                xyz[1] = frame[idx]
            elif channel.startswith("z"):
                xyz[2] = frame[idx]
        translations.append(tuple(xyz))
    return translations


def _position_indices_for_joint(source: BvhMotion, joint_name: str | None) -> list[int]:
    if joint_name is None:
        return []
    return [
        idx
        for idx, (joint, channel) in enumerate(source.channel_order)
        if joint == joint_name and channel.lower().endswith("position")
    ]


def _root_motion_source_label(source: BvhMotion, joint_name: str | None) -> str:
    if joint_name is not None and _normalize(joint_name) in {"hips", "hip", "pelvis"}:
        return "hips"
    if joint_name == source.root:
        return "root"
    return str(joint_name or "none")


def _first_match(lookup: dict[str, str], aliases: tuple[str, ...]) -> str | None:
    for alias in aliases:
        normalized = _normalize(alias)
        if normalized in lookup:
            return lookup[normalized]
    return None


def _build_anonymous_unirig_mapping(source_lookup: dict[str, str], target: RigInspection) -> JointMapping | None:
    """Map bounded anonymous UniRig humanoid topologies by shape.

    Some valid UniRig GLBs expose skin joints as anonymous ``bone_N`` names. A
    broad anonymous-rig fallback would be unsafe, so this mapper first infers a
    specific humanoid shape: one root/hips joint with a spine branch, two arm
    chains with finger-like terminal branches, and two leg chains. Unrelated
    anonymous rigs still fail closed with MAPPING_INCOMPATIBLE.
    """
    semantic_targets = _infer_anonymous_unirig_semantics(target)
    if semantic_targets is None:
        return None
    pairs: dict[str, str] = {}
    missing: list[str] = []
    for semantic, aliases in BVH_TO_TARGET_ALIASES.items():
        source_name = _first_match(source_lookup, aliases)
        target_name = semantic_targets[semantic]
        if source_name:
            pairs[source_name] = target_name
        else:
            missing.append(semantic)

    required_missing = {"hips", "spine", "head"}.intersection(missing)
    if required_missing:
        raise RetargetError(MAPPING_INCOMPATIBLE, f"Required humanoid joints could not be mapped: {sorted(required_missing)}")
    warnings = tuple(f"missing_{semantic}_mapping" for semantic in missing) + (
        "anonymous_unirig_topology_mapping",
        "anonymous_unirig_left_right_assumption",
    )
    confidence = "compatible" if len(pairs) >= 5 else "best_effort"
    return JointMapping(pairs=pairs, confidence=confidence, warnings=warnings)


def _looks_like_anonymous_unirig(target: RigInspection) -> bool:
    return _infer_anonymous_unirig_semantics(target) is not None


def _infer_anonymous_unirig_semantics(target: RigInspection) -> dict[str, str] | None:
    names = set(target.joint_names)
    if len(names) < 40 or len(names) > 80 or any(not _is_bone_number(name) for name in names):
        return None
    indices = sorted(_bone_number(name) for name in names)
    if indices != list(range(indices[-1] + 1)):
        return None

    children = {name: tuple(child_names) for name, child_names in target.joint_children}
    child_set = {child for child_names in children.values() for child in child_names}
    if child_set != names - {"bone_0"}:
        return None
    if set(children.get("bone_0", ())) != set(_direct_children(children, "bone_0")):
        return None
    root_children = tuple(children.get("bone_0", ()))
    if len(root_children) != 3:
        return None

    torso_candidate: tuple[str, str, str] | None = None
    leg_candidates: list[str] = []
    for child in root_children:
        chain = _single_child_chain(children, child, max_steps=4)
        if len(chain) >= 3 and len(children.get(chain[2], ())) >= 3:
            torso_candidate = (child, chain[1], chain[2])
        elif _is_plain_chain(children, child, min_edges=3):
            leg_candidates.append(child)
    if torso_candidate is None or len(leg_candidates) != 2:
        return None

    spine_start, _spine_mid, torso_hub = torso_candidate
    head_candidates: list[str] = []
    arm_candidates: list[str] = []
    for child in children.get(torso_hub, ()):
        if _is_arm_chain(children, child):
            arm_candidates.append(child)
        elif len(children.get(child, ())) <= 1:
            head_candidates.append(child)
    if len(head_candidates) != 1 or len(arm_candidates) != 2:
        return None

    left_arm, right_arm = sorted(arm_candidates, key=_bone_number)
    left_leg, right_leg = sorted(leg_candidates, key=_bone_number)
    return {
        "hips": "bone_0",
        "spine": spine_start,
        "head": head_candidates[0],
        "left_arm": left_arm,
        "right_arm": right_arm,
        "left_leg": left_leg,
        "right_leg": right_leg,
    }


def _direct_children(children: dict[str, tuple[str, ...]], name: str) -> tuple[str, ...]:
    return tuple(child for child in children.get(name, ()) if _is_bone_number(child))


def _single_child_chain(children: dict[str, tuple[str, ...]], start: str, *, max_steps: int) -> tuple[str, ...]:
    chain = [start]
    current = start
    for _ in range(max_steps):
        child_names = children.get(current, ())
        if len(child_names) != 1:
            break
        current = child_names[0]
        chain.append(current)
    return tuple(chain)


def _is_plain_chain(children: dict[str, tuple[str, ...]], start: str, *, min_edges: int) -> bool:
    chain = _single_child_chain(children, start, max_steps=min_edges + 1)
    return len(chain) >= min_edges + 1


def _is_arm_chain(children: dict[str, tuple[str, ...]], start: str) -> bool:
    chain = _single_child_chain(children, start, max_steps=4)
    if len(chain) < 4:
        return False
    terminal_children = children.get(chain[-1], ())
    return 4 <= len(terminal_children) <= 5 and all(_is_bone_number(child) for child in terminal_children)


def _bone_number(name: str) -> int:
    return int(name[len("bone_"):])


def _is_bone_number(name: str) -> bool:
    prefix = "bone_"
    return name.startswith(prefix) and name[len(prefix):].isdigit()


def _normalize(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())
