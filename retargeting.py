"""Conservative BVH-to-target-rig mapping and keyframe seams."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from kimodo_bvh import BvhMotion
from retarget_errors import CALIBRATION_UNAVAILABLE, MAPPING_INCOMPATIBLE, RetargetError
from rig_inspector import HUMANOID_CONTRACT_SCHEMA, REQUIRED_HUMANOID_CHAINS, RigInspection


BVH_TO_TARGET_ALIASES = {
    "hips": ("hips", "hip", "pelvis", "root"),
    "spine": ("spine", "spine1", "chest"),
    "head": ("head", "neck"),
    "left_arm": ("leftarm", "leftupperarm", "lupperarm", "leftshoulder"),
    "right_arm": ("rightarm", "rightupperarm", "rupperarm", "rightshoulder"),
    "left_leg": ("leftupleg", "leftleg", "lthigh", "leftthigh"),
    "right_leg": ("rightupleg", "rightleg", "rthigh", "rightthigh"),
}

SOURCE_ROLE_ALIASES = {
    "hips": ("hips", "hip", "pelvis", "root"),
    "spine": ("spine", "spine1"),
    "chest": ("chest", "spine2", "spine3", "upperchest"),
    "neck": ("neck",),
    "head": ("head",),
    "left_upper_arm": ("leftupperarm", "leftarm", "lupperarm", "leftshoulder"),
    "left_forearm": ("leftforearm", "leftlowerarm", "llowerarm", "leftelbow"),
    "left_hand": ("lefthand", "lhand", "leftwrist"),
    "right_upper_arm": ("rightupperarm", "rightarm", "rupperarm", "rightshoulder"),
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

CALIBRATION_VERSION = "kimodo.retarget_calibration.v1"
COMPOSITION_ORDER = "target_rest * (target_basis * inverse(source_basis)) * (source_sample * inverse(source_first)) * inverse(target_basis * inverse(source_basis))"


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
    root_target = _root_motion_target(source, mapping)
    channels: list[RetargetChannel] = []
    if root_target is not None:
        channels.append(
            RetargetChannel(
                target_joint=root_target,
                path="translation",
                times=times,
                values=tuple(translations),
            ),
        )
    if mapping.confidence == "calibrated":
        channels.extend(_build_calibrated_rotation_channels(source, mapping, times))
    else:
        channels.extend(_build_rotation_channels(source, mapping, times))
    warnings = _unique_tuple(mapping.warnings + target.warnings + (
        "axis_assumption_bvh_xyz_degrees",
        "rotation_order_assumption_bvh_channel_order_local_xyzw_quaternion",
        "root_motion_mode_preserve_translation",
    ))
    calibration = _calibration_metadata(target, mapping) if mapping.confidence == "calibrated" else {}
    return RetargetClip(
        name="Kimodo Motion",
        duration=source.duration,
        frame_time=source.frame_time,
        root_translations=tuple(translations),
        channels=tuple(channels),
        mapping=mapping,
        warnings=warnings,
        calibration=calibration,
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
            source_joint = _first_match(source_lookup, SOURCE_ROLE_ALIASES.get(role, (role,)))
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

    warnings = _optional_chain_warnings(source, source_lookup, target, role_to_target, chain_by_name)
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


def _target_joint_basis(target: RigInspection, role: str, joint_name: str) -> JointBasis | None:
    hint = _basis_hint_for_role(target.humanoid_basis_hints, role)
    if hint is not None:
        return JointBasis(joint=joint_name, quaternion=hint[0], source=hint[1])
    rest_vector = _rest_vector_for_role(target.humanoid_rest, role, joint_name)
    if rest_vector is not None:
        return JointBasis(
            joint=joint_name,
            quaternion=_quaternion_from_y_axis_to_vector(rest_vector),
            source=f"target_rest_transform:{role}",
        )
    if target.humanoid_rest:
        return JointBasis(joint=joint_name, quaternion=(0.0, 0.0, 0.0, 1.0), source="target_rest_transform")
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
            source_joint = _first_match(source_lookup, SOURCE_ROLE_ALIASES.get(role, (role,)))
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


def _rest_vector_for_role(raw_rest: object, role: str, joint_name: str) -> tuple[float, float, float] | None:
    if not isinstance(raw_rest, dict):
        return None
    for section_name in ("roles", "joints", "nodes", "transforms"):
        section = raw_rest.get(section_name)
        if not isinstance(section, dict):
            continue
        vector = _rest_vector_from_entry(section.get(role)) or _rest_vector_from_entry(section.get(joint_name))
        if vector is not None:
            return vector
    return _rest_vector_from_entry(raw_rest.get(role)) or _rest_vector_from_entry(raw_rest.get(joint_name))


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
        calibration = _multiply_quaternion_xyzw(target_basis, _inverse_quaternion_xyzw(source_basis))
        samples = [_bvh_euler_degrees_to_quaternion_xyzw(tuple((channel, frame[index]) for index, channel in joint_rotation_indices)) for frame in source.frames]
        first_inverse = _inverse_quaternion_xyzw(samples[0]) if samples else (0.0, 0.0, 0.0, 1.0)
        values = []
        for sample in samples:
            source_delta = _multiply_quaternion_xyzw(sample, first_inverse)
            target_delta = _multiply_quaternion_xyzw(_multiply_quaternion_xyzw(calibration, source_delta), _inverse_quaternion_xyzw(calibration))
            values.append(_normalize_quaternion_xyzw(target_delta))
        channels.append(
            RetargetChannel(
                target_joint=target_joint,
                path="rotation",
                times=times,
                values=tuple(values),
                diagnostics={
                    "source_joint": source_joint,
                    "target_joint": target_joint,
                    "role": role,
                    "composition_order": COMPOSITION_ORDER,
                    "calibration_quaternion": calibration,
                },
            )
        )
    return tuple(channels)


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
        "warnings_by_severity": {
            "warning": [warning for warning in mapping.warnings if warning != "humanoid_contract_calibrated_mapping"],
            "info": ["humanoid_contract_calibrated_mapping"],
        },
        "fallback_widened": False,
        "contract_diagnostics": list(target.humanoid_diagnostics),
    }


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
