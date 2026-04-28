"""Rigged GLB/GLTF inspection for the retarget/export seam."""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from retarget_errors import RIG_HAS_NO_SKIN_OR_SKELETON, UNPARSEABLE_RIG, RetargetError
from retarget_errors import MAPPING_INCOMPATIBLE


HUMANOID_CONTRACT_SCHEMA = "modly.humanoid.v1"
REQUIRED_HUMANOID_ROLES = {
    "hips",
    "spine",
    "chest",
    "neck",
    "head",
    "left_upper_arm",
    "left_forearm",
    "left_hand",
    "right_upper_arm",
    "right_forearm",
    "right_hand",
    "left_upper_leg",
    "left_lower_leg",
    "left_foot",
    "right_upper_leg",
    "right_lower_leg",
    "right_foot",
}
REQUIRED_HUMANOID_CHAINS = {
    "torso": ("hips", "spine", "chest", "neck", "head"),
    "left_arm": ("left_upper_arm", "left_forearm", "left_hand"),
    "right_arm": ("right_upper_arm", "right_forearm", "right_hand"),
    "left_leg": ("left_upper_leg", "left_lower_leg", "left_foot"),
    "right_leg": ("right_upper_leg", "right_lower_leg", "right_foot"),
}


@dataclass(frozen=True)
class HumanoidRoleBinding:
    role: str
    node_index: int
    node_name: str


@dataclass(frozen=True)
class HumanoidChain:
    name: str
    roles: tuple[str, ...]


@dataclass(frozen=True)
class RigInspection:
    path: Path
    source_kind: str
    classification: str
    joint_names: tuple[str, ...]
    skin_count: int
    has_inverse_bind_matrices: bool
    joint_children: tuple[tuple[str, tuple[str, ...]], ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)
    humanoid_contract_path: Path | None = None
    humanoid_contract_schema: str | None = None
    humanoid_roles: tuple[HumanoidRoleBinding, ...] = field(default_factory=tuple)
    humanoid_chains: tuple[HumanoidChain, ...] = field(default_factory=tuple)
    humanoid_quality: tuple[str, ...] = field(default_factory=tuple)
    humanoid_rest: dict[str, Any] = field(default_factory=dict)
    humanoid_basis_hints: dict[str, Any] = field(default_factory=dict)
    humanoid_diagnostics: tuple[str, ...] = field(default_factory=tuple)

    @property
    def compatible(self) -> bool:
        return self.skin_count > 0 and bool(self.joint_names)


def inspect_rig(path: Path, *, source_kind: str = "unknown_mesh_source") -> RigInspection:
    path = Path(path)
    if path.suffix.lower() not in {".glb", ".gltf"} or not path.is_file():
        raise RetargetError(UNPARSEABLE_RIG, f"Rig path must be an existing .glb/.gltf file: {path}")
    document = _read_gltf_document(path)
    nodes = document.get("nodes") if isinstance(document.get("nodes"), list) else []
    skins = document.get("skins") if isinstance(document.get("skins"), list) else []
    joint_indices: list[int] = []
    has_inverse_bind = False
    for skin in skins:
        if not isinstance(skin, dict):
            continue
        if "inverseBindMatrices" in skin:
            has_inverse_bind = True
        for joint in skin.get("joints", []):
            if isinstance(joint, int):
                joint_indices.append(joint)
    joint_names = tuple(_node_name(nodes, index) for index in joint_indices if 0 <= index < len(nodes))
    if not skins or not joint_names:
        raise RetargetError(RIG_HAS_NO_SKIN_OR_SKELETON, "Rig must contain at least one skin with skeleton joints.")
    joint_children = _joint_children(nodes, joint_indices)
    classification, warnings = _classify(joint_names)
    if not has_inverse_bind:
        warnings.append("missing_inverse_bind_matrices")
    contract = _inspect_humanoid_contract(path, nodes, set(joint_indices))
    warnings.extend(contract["warnings"])
    return RigInspection(
        path=path.resolve(),
        source_kind=source_kind,
        classification=classification,
        joint_names=joint_names,
        skin_count=len(skins),
        has_inverse_bind_matrices=has_inverse_bind,
        joint_children=joint_children,
        warnings=tuple(warnings),
        humanoid_contract_path=contract["path"],
        humanoid_contract_schema=contract["schema"],
        humanoid_roles=contract["roles"],
        humanoid_chains=contract["chains"],
        humanoid_quality=contract["quality"],
        humanoid_rest=contract["rest"],
        humanoid_basis_hints=contract["basis_hints"],
        humanoid_diagnostics=contract["diagnostics"],
    )


def _inspect_humanoid_contract(path: Path, nodes: list[Any], skin_joint_indices: set[int]) -> dict[str, Any]:
    sidecar_path = path.with_suffix(".rigmeta.json")
    empty = {
        "path": None,
        "schema": None,
        "roles": (),
        "chains": (),
        "quality": (),
        "rest": {},
        "basis_hints": {},
        "diagnostics": (),
        "warnings": [],
    }
    if not sidecar_path.is_file():
        return empty
    diagnostics: list[str] = []
    try:
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        diagnostics.append("humanoid_contract_invalid_json")
        _raise_contract_error(sidecar_path, diagnostics)
    if not isinstance(payload, dict):
        diagnostics.append("humanoid_contract_invalid_shape:root")
        _raise_contract_error(sidecar_path, diagnostics)
    if "humanoid_contract" not in payload:
        return empty
    contract = payload.get("humanoid_contract")
    if not isinstance(contract, dict):
        diagnostics.append("humanoid_contract_invalid_shape:humanoid_contract")
        _raise_contract_error(sidecar_path, diagnostics)
    schema = contract.get("schema")
    if schema != HUMANOID_CONTRACT_SCHEMA:
        diagnostics.append(f"humanoid_contract_schema_unsupported:{schema}")
        _raise_contract_error(sidecar_path, diagnostics)

    roles = _parse_humanoid_roles(contract.get("roles"), nodes, skin_joint_indices, diagnostics)
    chains = _parse_humanoid_chains(contract.get("chains"), roles, diagnostics)
    _validate_required_humanoid_contract(roles, chains, diagnostics)
    if diagnostics:
        _raise_contract_error(sidecar_path, diagnostics)

    rest = contract.get("rest") or contract.get("rest_pose") or contract.get("bind_data") or {}
    basis_hints = contract.get("basis_hints") or contract.get("basis") or {}
    warnings = []
    quality = contract.get("quality") or contract.get("warnings") or ()
    return {
        "path": sidecar_path.resolve(),
        "schema": schema,
        "roles": tuple(roles[role] for role in sorted(roles)),
        "chains": tuple(chains[name] for name in sorted(chains)),
        "quality": _string_tuple(quality),
        "rest": rest if isinstance(rest, dict) else {},
        "basis_hints": basis_hints if isinstance(basis_hints, dict) else {},
        "diagnostics": ("humanoid_contract_validated",),
        "warnings": sorted(set(warnings)),
    }


def _parse_humanoid_roles(raw: Any, nodes: list[Any], skin_joint_indices: set[int], diagnostics: list[str]) -> dict[str, HumanoidRoleBinding]:
    entries: list[tuple[str, Any]] = []
    if isinstance(raw, dict):
        entries = [(str(role), binding) for role, binding in raw.items()]
    elif isinstance(raw, list):
        for entry in raw:
            if isinstance(entry, dict) and isinstance(entry.get("role"), str):
                entries.append((entry["role"], entry))
    else:
        diagnostics.append("humanoid_contract_invalid_shape:roles")
        return {}
    roles: dict[str, HumanoidRoleBinding] = {}
    node_to_role: dict[int, str] = {}
    for raw_role, binding in entries:
        role = _canonical_role(raw_role)
        if not role or not isinstance(binding, dict):
            diagnostics.append(f"humanoid_contract_invalid_role:{raw_role}")
            continue
        if role in roles:
            diagnostics.append(f"humanoid_contract_duplicate_role:{role}")
            continue
        node_index = binding.get("node_index", binding.get("node"))
        node_name = binding.get("node_name", binding.get("name"))
        if not isinstance(node_index, int):
            diagnostics.append(f"humanoid_contract_role_missing_node_index:{role}")
            continue
        if node_index < 0 or node_index >= len(nodes):
            diagnostics.append(f"humanoid_contract_role_node_index_out_of_range:{role}:{node_index}")
            continue
        actual_name = _node_name(nodes, node_index)
        if isinstance(node_name, str) and node_name.strip() and node_name.strip() != actual_name:
            diagnostics.append(f"humanoid_contract_role_node_name_mismatch:{role}:{node_name}:{actual_name}")
            continue
        if node_index not in skin_joint_indices:
            diagnostics.append(f"humanoid_contract_role_not_skin_joint:{role}:{node_index}")
            continue
        if node_index in node_to_role:
            diagnostics.append(f"humanoid_contract_duplicate_node_binding:{role}:{node_to_role[node_index]}:{node_index}")
            continue
        node_to_role[node_index] = role
        roles[role] = HumanoidRoleBinding(role=role, node_index=node_index, node_name=actual_name)
    return roles


def _parse_humanoid_chains(raw: Any, roles: dict[str, HumanoidRoleBinding], diagnostics: list[str]) -> dict[str, HumanoidChain]:
    entries: list[tuple[str, Any]] = []
    if isinstance(raw, dict):
        entries = [(str(name), value) for name, value in raw.items()]
    elif isinstance(raw, list):
        for entry in raw:
            if isinstance(entry, dict) and isinstance(entry.get("name"), str):
                entries.append((entry["name"], entry.get("roles")))
    else:
        diagnostics.append("humanoid_contract_invalid_shape:chains")
        return {}
    chains: dict[str, HumanoidChain] = {}
    for raw_name, value in entries:
        name = _canonical_chain_name(raw_name)
        raw_roles = value.get("roles") if isinstance(value, dict) else value
        if not isinstance(raw_roles, list) or not all(isinstance(role, str) for role in raw_roles):
            diagnostics.append(f"humanoid_contract_invalid_chain:{raw_name}")
            continue
        chain_roles = tuple(_canonical_role(role) for role in raw_roles)
        missing = [role for role in chain_roles if role not in roles]
        if missing:
            diagnostics.append(f"humanoid_contract_chain_unknown_role:{name}:{','.join(missing)}")
            continue
        chains[name] = HumanoidChain(name=name, roles=chain_roles)
    return chains


def _validate_required_humanoid_contract(
    roles: dict[str, HumanoidRoleBinding],
    chains: dict[str, HumanoidChain],
    diagnostics: list[str],
) -> None:
    for role in sorted(REQUIRED_HUMANOID_ROLES - set(roles)):
        diagnostics.append(f"humanoid_contract_missing_role:{role}")
    for name, required_roles in REQUIRED_HUMANOID_CHAINS.items():
        chain = chains.get(name)
        if chain is None:
            diagnostics.append(f"humanoid_contract_missing_chain:{name}")
            continue
        positions = [chain.roles.index(role) for role in required_roles if role in chain.roles]
        if len(positions) != len(required_roles) or positions != sorted(positions):
            diagnostics.append(f"humanoid_contract_unusable_chain:{name}")


def _raise_contract_error(path: Path, diagnostics: list[str]) -> None:
    unique = tuple(dict.fromkeys(diagnostics))
    raise RetargetError(
        MAPPING_INCOMPATIBLE,
        f"Humanoid contract sidecar validation failed: {', '.join(unique)}",
        diagnostics={"humanoid_contract": {"path": str(path.resolve()), "diagnostics": list(unique)}},
    )


def _canonical_role(value: str) -> str:
    normalized = _normalize(value)
    aliases = {
        "leftarm": "left_upper_arm",
        "leftupperarm": "left_upper_arm",
        "rightarm": "right_upper_arm",
        "rightupperarm": "right_upper_arm",
        "leftleg": "left_upper_leg",
        "rightleg": "right_upper_leg",
        "leftupleg": "left_upper_leg",
        "rightupleg": "right_upper_leg",
        "leftupperleg": "left_upper_leg",
        "rightupperleg": "right_upper_leg",
        "leftlowerarm": "left_forearm",
        "rightlowerarm": "right_forearm",
        "leftlowerleg": "left_lower_leg",
        "rightlowerleg": "right_lower_leg",
    }
    if normalized in aliases:
        return aliases[normalized]
    parts = []
    token = normalized
    for prefix in ("left", "right"):
        if token.startswith(prefix) and token != prefix:
            parts.append(prefix)
            token = token[len(prefix):]
            break
    parts.append(token)
    return "_".join(part for part in parts if part)


def _canonical_chain_name(value: str) -> str:
    normalized = _normalize(value)
    aliases = {
        "torso": "torso",
        "spine": "torso",
        "leftarm": "left_arm",
        "rightarm": "right_arm",
        "leftleg": "left_leg",
        "rightleg": "right_leg",
    }
    return aliases.get(normalized, normalized)


def _string_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, list):
        return tuple(str(item) for item in value)
    if isinstance(value, dict):
        return tuple(f"{key}:{value[key]}" for key in sorted(value))
    if isinstance(value, str):
        return (value,)
    return ()


def _read_gltf_document(path: Path) -> dict[str, Any]:
    try:
        if path.suffix.lower() == ".gltf":
            payload = json.loads(path.read_text(encoding="utf-8"))
        else:
            payload = _read_glb_json(path)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError, struct.error) as exc:
        raise RetargetError(UNPARSEABLE_RIG, f"Could not parse rig file {path}.") from exc
    if not isinstance(payload, dict) or "asset" not in payload:
        raise RetargetError(UNPARSEABLE_RIG, f"Rig file {path} is not a valid glTF document.")
    return payload


def _read_glb_json(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    if len(data) < 20 or data[:4] != b"glTF":
        raise ValueError("invalid GLB header")
    version, _length = struct.unpack_from("<II", data, 4)
    if version != 2:
        raise ValueError("only GLB v2 is supported")
    chunk_length, chunk_type = struct.unpack_from("<II", data, 12)
    if chunk_type != 0x4E4F534A:
        raise ValueError("first GLB chunk is not JSON")
    json_bytes = data[20:20 + chunk_length].rstrip(b" \t\r\n\x00")
    return json.loads(json_bytes.decode("utf-8"))


def _node_name(nodes: list[Any], index: int) -> str:
    node = nodes[index]
    if isinstance(node, dict) and isinstance(node.get("name"), str) and node["name"].strip():
        return node["name"].strip()
    return f"node_{index}"


def _joint_children(nodes: list[Any], joint_indices: list[int]) -> tuple[tuple[str, tuple[str, ...]], ...]:
    joint_index_set = set(joint_indices)
    relationships: list[tuple[str, tuple[str, ...]]] = []
    for index in joint_indices:
        if index < 0 or index >= len(nodes):
            continue
        node = nodes[index]
        if not isinstance(node, dict):
            continue
        child_names = tuple(
            _node_name(nodes, child_index)
            for child_index in node.get("children", [])
            if isinstance(child_index, int) and child_index in joint_index_set and 0 <= child_index < len(nodes)
        )
        relationships.append((_node_name(nodes, index), child_names))
    return tuple(relationships)


def _classify(joint_names: tuple[str, ...]) -> tuple[str, list[str]]:
    normalized = {_normalize(name) for name in joint_names}
    required_groups = {
        "hips": {"hips", "hip", "pelvis", "root"},
        "spine": {"spine", "spine1", "chest"},
        "head": {"head", "neck"},
        "left_arm": {"leftarm", "leftupperarm", "lupperarm", "leftshoulder"},
        "right_arm": {"rightarm", "rightupperarm", "rupperarm", "rightshoulder"},
        "left_leg": {"leftupleg", "leftleg", "lthigh", "leftthigh"},
        "right_leg": {"rightupleg", "rightleg", "rthigh", "rightthigh"},
    }
    hits = sum(1 for aliases in required_groups.values() if normalized.intersection(aliases))
    if hits >= 5:
        return "humanoid", []
    return "unknown", ["classification_unknown_or_non_humanoid"]


def _normalize(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())
