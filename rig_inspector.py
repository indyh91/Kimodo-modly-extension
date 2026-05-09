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
SEMANTIC_CANDIDATES_SCHEMA = "unirig.semantic_candidates.v1"
SEMANTIC_CANDIDATE_CONFIDENCE_THRESHOLD = 0.60
REQUIRED_SEMANTIC_CANDIDATE_CORE_ROLES = {"hips", "spine", "head"}
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
class SemanticCandidateBinding:
    role: str
    node_id: str
    node_index: int
    node_name: str
    confidence: float
    reasons: tuple[str, ...] = field(default_factory=tuple)
    source: str | None = None


@dataclass(frozen=True)
class SemanticCandidateChain:
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
    contract_status: str = "contract_missing_or_untrusted"
    semantic_candidate_path: Path | None = None
    semantic_candidate_schema: str | None = None
    semantic_candidate_bindings: tuple[SemanticCandidateBinding, ...] = field(default_factory=tuple)
    semantic_candidate_chains: tuple[SemanticCandidateChain, ...] = field(default_factory=tuple)
    semantic_candidate_diagnostics: tuple[str, ...] = field(default_factory=tuple)

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
        contract_status=contract["status"],
        semantic_candidate_path=contract["semantic_candidate_path"],
        semantic_candidate_schema=contract["semantic_candidate_schema"],
        semantic_candidate_bindings=contract["semantic_candidate_bindings"],
        semantic_candidate_chains=contract["semantic_candidate_chains"],
        semantic_candidate_diagnostics=contract["semantic_candidate_diagnostics"],
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
        "status": "contract_missing_or_untrusted",
        "semantic_candidate_path": None,
        "semantic_candidate_schema": None,
        "semantic_candidate_bindings": (),
        "semantic_candidate_chains": (),
        "semantic_candidate_diagnostics": (),
    }
    if not sidecar_path.is_file():
        return empty
    diagnostics: list[str] = []
    try:
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        diagnostics.append("humanoid_contract_invalid_json")
        return _untrusted_contract(sidecar_path, diagnostics)
    if not isinstance(payload, dict):
        diagnostics.append("humanoid_contract_invalid_shape:root")
        return _untrusted_contract(sidecar_path, diagnostics)
    semantic = _inspect_semantic_candidates(sidecar_path, payload, nodes, skin_joint_indices)
    if "humanoid_contract" not in payload:
        diagnostics.extend(_missing_humanoid_contract_diagnostics(payload))
        if diagnostics:
            return _with_semantic_candidates(_untrusted_contract(sidecar_path, diagnostics), semantic)
        return _with_semantic_candidates(empty, semantic)
    contract = payload.get("humanoid_contract")
    if not isinstance(contract, dict):
        diagnostics.append("humanoid_contract_invalid_shape:humanoid_contract")
        return _with_semantic_candidates(_untrusted_contract(sidecar_path, diagnostics), semantic)
    schema = contract.get("schema")
    if schema != HUMANOID_CONTRACT_SCHEMA:
        diagnostics.append(f"humanoid_contract_schema_unsupported:{schema}")
        return _with_semantic_candidates(_untrusted_contract(sidecar_path, diagnostics), semantic)

    contract_nodes = contract.get("nodes")
    role_source = contract.get("roles") if "roles" in contract else _merged_canonical_role_source(contract, diagnostics)
    canonical_role_nodes = contract_nodes if "roles" not in contract else None
    roles = _parse_humanoid_roles(role_source, nodes, skin_joint_indices, diagnostics, contract_nodes=canonical_role_nodes)
    chains = _parse_humanoid_chains(contract.get("chains"), roles, diagnostics, contract_nodes=contract_nodes)
    _validate_required_humanoid_contract(roles, chains, diagnostics)
    if diagnostics:
        return _with_semantic_candidates(_untrusted_contract(sidecar_path, diagnostics, schema=schema), semantic)

    rest = contract.get("rest") or contract.get("rest_pose") or contract.get("bind_data") or {}
    if not isinstance(rest, dict) or not rest:
        rest = _canonical_rest_from_nodes(contract_nodes, roles)
    basis_hints = contract.get("basis_hints") or contract.get("basis") or {}
    warnings = []
    quality = contract.get("quality") or contract.get("warnings") or ()
    return _with_semantic_candidates({
        "path": sidecar_path.resolve(),
        "schema": schema,
        "roles": tuple(roles[role] for role in sorted(roles)),
        "chains": tuple(chains[name] for name in sorted(chains)),
        "quality": _string_tuple(quality),
        "rest": rest if isinstance(rest, dict) else {},
        "basis_hints": basis_hints if isinstance(basis_hints, dict) else {},
        "diagnostics": ("humanoid_contract_validated",),
        "warnings": sorted(set(warnings)),
        "status": "trusted",
    }, semantic)


def _untrusted_contract(sidecar_path: Path, diagnostics: list[str], *, schema: str | None = None) -> dict[str, Any]:
    unique = tuple(dict.fromkeys(diagnostics))
    return {
        "path": sidecar_path.resolve(),
        "schema": schema,
        "roles": (),
        "chains": (),
        "quality": (),
        "rest": {},
        "basis_hints": {},
        "diagnostics": unique,
        "warnings": ["humanoid_contract_missing_or_untrusted"],
        "status": "contract_missing_or_untrusted",
        "semantic_candidate_path": None,
        "semantic_candidate_schema": None,
        "semantic_candidate_bindings": (),
        "semantic_candidate_chains": (),
        "semantic_candidate_diagnostics": (),
    }


def _with_semantic_candidates(contract: dict[str, Any], semantic: dict[str, Any]) -> dict[str, Any]:
    merged = dict(contract)
    merged["semantic_candidate_path"] = semantic["path"]
    merged["semantic_candidate_schema"] = semantic["schema"]
    merged["semantic_candidate_bindings"] = semantic["bindings"]
    merged["semantic_candidate_chains"] = semantic["chains"]
    merged["semantic_candidate_diagnostics"] = semantic["diagnostics"]
    return merged


def _empty_semantic_candidates() -> dict[str, Any]:
    return {
        "path": None,
        "schema": None,
        "bindings": (),
        "chains": (),
        "diagnostics": (),
    }


def _inspect_semantic_candidates(
    sidecar_path: Path,
    payload: dict[str, Any],
    nodes: list[Any],
    skin_joint_indices: set[int],
) -> dict[str, Any]:
    raw = payload.get("semantic_candidates")
    if raw is None:
        return _empty_semantic_candidates()
    diagnostics: list[str] = []
    if not isinstance(raw, dict):
        return _semantic_candidate_result(sidecar_path, None, (), (), ["semantic_candidates_invalid_shape:semantic_candidates"])

    schema = raw.get("schema")
    if schema != SEMANTIC_CANDIDATES_SCHEMA:
        diagnostics.append(f"semantic_candidates_schema_unsupported:{schema}")
        return _semantic_candidate_result(sidecar_path, schema if isinstance(schema, str) else None, (), (), diagnostics)

    diagnostics.extend(_string_tuple(raw.get("diagnostics")))
    trust = raw.get("trust")
    if isinstance(trust, dict):
        if trust.get("trusted") is not False:
            diagnostics.append("semantic_candidates_trust_not_false")
        if trust.get("stabilization_eligible") is not False:
            diagnostics.append("semantic_candidates_stabilization_not_false")
    else:
        diagnostics.append("semantic_candidates_invalid_shape:trust")

    raw_roles = raw.get("roles")
    if not isinstance(raw_roles, dict):
        diagnostics.append("semantic_candidates_invalid_shape:roles")
        return _semantic_candidate_result(sidecar_path, schema, (), (), diagnostics)

    bindings: dict[str, SemanticCandidateBinding] = {}
    node_to_role: dict[int, str] = {}
    for raw_role, candidate in raw_roles.items():
        role = _canonical_role(str(raw_role))
        binding = _parse_semantic_candidate_binding(role, candidate, nodes, skin_joint_indices, diagnostics)
        if binding is None:
            continue
        if role in bindings:
            diagnostics.append(f"semantic_candidate_duplicate_role:{role}")
            continue
        if binding.node_index in node_to_role:
            diagnostics.append(f"semantic_candidate_duplicate_node_binding:{role}:{node_to_role[binding.node_index]}:{binding.node_index}")
            continue
        bindings[role] = binding
        node_to_role[binding.node_index] = role

    missing_core = sorted(REQUIRED_SEMANTIC_CANDIDATE_CORE_ROLES - set(bindings))
    if missing_core:
        diagnostics.append("semantic_candidates_core_incomplete")
        diagnostics.extend(f"semantic_candidate_missing_core:{role}" for role in missing_core)
        return _semantic_candidate_result(sidecar_path, schema, (), (), diagnostics)

    chains = _parse_semantic_candidate_chains(raw.get("chains"), set(bindings), diagnostics)
    diagnostics.append("semantic_candidates_validated")
    ordered_bindings = tuple(bindings[role] for role in sorted(bindings))
    return _semantic_candidate_result(sidecar_path, schema, ordered_bindings, tuple(chains), diagnostics)


def _semantic_candidate_result(
    sidecar_path: Path,
    schema: str | None,
    bindings: tuple[SemanticCandidateBinding, ...],
    chains: tuple[SemanticCandidateChain, ...],
    diagnostics: list[str] | tuple[str, ...],
) -> dict[str, Any]:
    return {
        "path": sidecar_path.resolve(),
        "schema": schema,
        "bindings": bindings,
        "chains": chains,
        "diagnostics": tuple(dict.fromkeys(str(item) for item in diagnostics)),
    }


def _parse_semantic_candidate_binding(
    role: str,
    candidate: Any,
    nodes: list[Any],
    skin_joint_indices: set[int],
    diagnostics: list[str],
) -> SemanticCandidateBinding | None:
    if role not in REQUIRED_HUMANOID_ROLES and role not in {"left_shoulder", "right_shoulder"}:
        diagnostics.append(f"semantic_candidate_unknown_role:{role}")
        return None
    if not isinstance(candidate, dict):
        diagnostics.append(f"semantic_candidate_invalid_shape:{role}")
        return None
    raw_candidate_role = candidate.get("role")
    if isinstance(raw_candidate_role, str) and _canonical_role(raw_candidate_role) != role:
        diagnostics.append(f"semantic_candidate_role_mismatch:{role}:{raw_candidate_role}")
        return None
    if candidate.get("rejected") is True:
        diagnostics.append(f"semantic_candidate_rejected:{role}")
        return None
    unsafe_flags = candidate.get("unsafe_flags")
    if isinstance(unsafe_flags, list) and any(str(flag).strip() for flag in unsafe_flags):
        if role in REQUIRED_SEMANTIC_CANDIDATE_CORE_ROLES:
            diagnostics.append(f"semantic_candidate_core_unsafe:{role}")
        else:
            diagnostics.append(f"semantic_candidate_unsafe:{role}")
        return None
    node_index = candidate.get("node_index", candidate.get("node"))
    if not isinstance(node_index, int):
        diagnostics.append(f"semantic_candidate_missing_node_index:{role}")
        return None
    if node_index < 0 or node_index >= len(nodes):
        diagnostics.append(f"semantic_candidate_node_index_out_of_range:{role}:{node_index}")
        return None
    if node_index not in skin_joint_indices:
        diagnostics.append(f"semantic_candidate_not_skin_joint:{role}:{node_index}")
        return None
    actual_name = _node_name(nodes, node_index)
    node_name = candidate.get("node_name", candidate.get("name"))
    if not isinstance(node_name, str) or not node_name.strip():
        diagnostics.append(f"semantic_candidate_missing_node_name:{role}")
        return None
    if node_name.strip() != actual_name:
        diagnostics.append(f"semantic_candidate_node_name_mismatch:{role}:{node_name}:{actual_name}")
        return None
    node_id = candidate.get("node_id")
    if not isinstance(node_id, str) or not node_id.strip():
        diagnostics.append(f"semantic_candidate_missing_node_id:{role}")
        return None
    if node_id.strip() != actual_name:
        diagnostics.append(f"semantic_candidate_node_id_mismatch:{role}:{node_id}:{actual_name}")
        return None
    try:
        confidence = float(candidate.get("confidence"))
    except (TypeError, ValueError):
        diagnostics.append(f"semantic_candidate_missing_confidence:{role}")
        return None
    if confidence < SEMANTIC_CANDIDATE_CONFIDENCE_THRESHOLD:
        diagnostics.append(f"semantic_candidate_low_confidence:{role}")
        return None
    source = candidate.get("source")
    return SemanticCandidateBinding(
        role=role,
        node_id=node_id.strip(),
        node_index=node_index,
        node_name=actual_name,
        confidence=confidence,
        reasons=_string_tuple(candidate.get("reasons")),
        source=source.strip() if isinstance(source, str) and source.strip() else None,
    )


def _parse_semantic_candidate_chains(raw: Any, roles: set[str], diagnostics: list[str]) -> list[SemanticCandidateChain]:
    if raw is None:
        return []
    if not isinstance(raw, dict):
        diagnostics.append("semantic_candidates_invalid_shape:chains")
        return []
    chains: list[SemanticCandidateChain] = []
    for raw_name, raw_roles in raw.items():
        name = _canonical_chain_name(str(raw_name))
        if not isinstance(raw_roles, list) or not all(isinstance(role, str) for role in raw_roles):
            diagnostics.append(f"semantic_candidate_invalid_chain:{raw_name}")
            continue
        chain_roles = tuple(_canonical_role(role) for role in raw_roles if _canonical_role(role) in roles)
        if chain_roles:
            chains.append(SemanticCandidateChain(name=name, roles=chain_roles))
    return chains


def _missing_humanoid_contract_diagnostics(payload: dict[str, Any]) -> list[str]:
    diagnostics: list[str] = []
    metadata_mode = str(payload.get("metadata_mode") or "").strip().lower()
    if metadata_mode == "auto":
        diagnostics.append("humanoid_contract_missing:metadata_mode_auto")
    if isinstance(payload.get("humanoid_provenance"), dict):
        diagnostics.append("humanoid_contract_missing:provenance_only")
    if isinstance(payload.get("legacy_humanoid_mapping"), dict) or isinstance(payload.get("humanoid_mapping"), dict):
        diagnostics.append("humanoid_contract_missing:legacy_mapping_only")
    if _contains_exact_bone_profile_signal(payload):
        diagnostics.append("humanoid_contract_missing:bone_profile_only")
    if not diagnostics and payload:
        diagnostics.append("humanoid_contract_missing")
    return diagnostics


def _contains_exact_bone_profile_signal(value: Any) -> bool:
    if isinstance(value, str):
        token = value.strip().lower()
        if not token.startswith("bone_"):
            return False
        suffix = token[5:]
        return bool(suffix) and suffix.isdigit()
    if isinstance(value, dict):
        return any(_contains_exact_bone_profile_signal(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_exact_bone_profile_signal(item) for item in value)
    return False


def _parse_humanoid_roles(
    raw: Any,
    nodes: list[Any],
    skin_joint_indices: set[int],
    diagnostics: list[str],
    *,
    contract_nodes: Any = None,
) -> dict[str, HumanoidRoleBinding]:
    entries: list[tuple[str, Any]] = []
    if isinstance(contract_nodes, dict):
        entries = _canonical_humanoid_role_entries(raw, contract_nodes, diagnostics)
    elif contract_nodes is not None:
        diagnostics.append("humanoid_contract_invalid_shape:nodes")
        return {}
    elif isinstance(raw, dict):
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


def _canonical_humanoid_role_entries(
    raw_required_roles: Any,
    contract_nodes: dict[Any, Any],
    diagnostics: list[str],
) -> list[tuple[str, Any]]:
    if not isinstance(raw_required_roles, dict):
        diagnostics.append("humanoid_contract_invalid_shape:required_roles")
        return []
    entries: list[tuple[str, Any]] = []
    for raw_role, raw_node_id in raw_required_roles.items():
        role = str(raw_role)
        if not isinstance(raw_node_id, str) or not raw_node_id.strip():
            diagnostics.append(f"humanoid_contract_role_missing_node_id:{role}")
            continue
        node_id = raw_node_id.strip()
        node = contract_nodes.get(node_id)
        if not isinstance(node, dict):
            diagnostics.append(f"humanoid_contract_role_unknown_node_id:{role}:{node_id}")
            continue
        node_index = node.get("index")
        node_name = node.get("name", node_id)
        if not isinstance(node_index, int):
            diagnostics.append(f"humanoid_contract_node_missing_index:{node_id}")
            continue
        binding = {"node_index": node_index}
        if isinstance(node_name, str) and node_name.strip():
            binding["node_name"] = node_name.strip()
        entries.append((role, binding))
    return entries


def _merged_canonical_role_source(contract: dict[str, Any], diagnostics: list[str]) -> dict[str, Any]:
    required_roles = contract.get("required_roles")
    if not isinstance(required_roles, dict):
        diagnostics.append("humanoid_contract_invalid_shape:required_roles")
        return {}
    merged = dict(required_roles)
    optional_roles = contract.get("optional_roles", {})
    if optional_roles is None:
        optional_roles = {}
    if not isinstance(optional_roles, dict):
        diagnostics.append("humanoid_contract_invalid_shape:optional_roles")
        return merged
    for role, node_id in optional_roles.items():
        merged[str(role)] = node_id
    return merged


def _canonical_rest_from_nodes(contract_nodes: Any, roles: dict[str, HumanoidRoleBinding]) -> dict[str, Any]:
    if not isinstance(contract_nodes, dict):
        return {}
    by_index = {binding.node_index: binding for binding in roles.values()}
    role_rest: dict[str, Any] = {}
    node_rest: dict[str, Any] = {}
    for raw_node_id, node in contract_nodes.items():
        if not isinstance(raw_node_id, str) or not isinstance(node, dict):
            continue
        transforms = node.get("transforms")
        if not isinstance(transforms, dict):
            continue
        rest_local = _parse_matrix4(transforms.get("rest_local"))
        rest_world = _parse_matrix4(transforms.get("rest_world"))
        if rest_local is None and rest_world is None:
            continue
        entry: dict[str, Any] = {"node_id": raw_node_id}
        local_translation = _matrix_translation(rest_local)
        world_translation = _matrix_translation(rest_world)
        if rest_local is not None:
            entry["rest_local"] = rest_local
        if rest_world is not None:
            entry["rest_world"] = rest_world
        if local_translation is not None:
            entry["local_translation"] = local_translation
        if world_translation is not None:
            entry["world_translation"] = world_translation
        node_name = node.get("name")
        if isinstance(node_name, str) and node_name.strip():
            node_rest[node_name.strip()] = dict(entry)
        binding = by_index.get(node.get("index"))
        if binding is not None:
            role_rest[binding.role] = dict(entry)
    if not role_rest and not node_rest:
        return {}
    return {
        "source": "humanoid_contract.nodes.transforms",
        "matrix_order": "row-major",
        "roles": role_rest,
        "nodes": node_rest,
    }


def _parse_matrix4(value: Any) -> list[list[float]] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    rows: list[list[float]] = []
    try:
        for row in value:
            if not isinstance(row, list) or len(row) != 4:
                return None
            rows.append([float(component) for component in row])
    except (TypeError, ValueError):
        return None
    return rows


def _matrix_translation(matrix: list[list[float]] | None) -> list[float] | None:
    if matrix is None:
        return None
    return [matrix[0][3], matrix[1][3], matrix[2][3]]


def _parse_humanoid_chains(
    raw: Any,
    roles: dict[str, HumanoidRoleBinding],
    diagnostics: list[str],
    *,
    contract_nodes: Any = None,
) -> dict[str, HumanoidChain]:
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
        chain_roles, missing = _resolve_humanoid_chain_roles(raw_roles, roles, contract_nodes)
        if missing:
            diagnostics.append(f"humanoid_contract_chain_unknown_role:{name}:{','.join(missing)}")
            continue
        chains[name] = HumanoidChain(name=name, roles=chain_roles)
    return chains


def _resolve_humanoid_chain_roles(
    raw_roles: list[str],
    roles: dict[str, HumanoidRoleBinding],
    contract_nodes: Any,
) -> tuple[tuple[str, ...], list[str]]:
    node_id_to_role: dict[str, str] = {}
    if isinstance(contract_nodes, dict):
        index_to_role = {binding.node_index: role for role, binding in roles.items()}
        for raw_node_id, node in contract_nodes.items():
            if not isinstance(raw_node_id, str) or not isinstance(node, dict):
                continue
            node_index = node.get("index")
            if isinstance(node_index, int) and node_index in index_to_role:
                node_id_to_role[raw_node_id] = index_to_role[node_index]

    resolved: list[str] = []
    missing: list[str] = []
    for raw_role in raw_roles:
        role = _canonical_role(raw_role)
        if role in roles:
            resolved.append(role)
            continue
        if raw_role in node_id_to_role:
            resolved.append(node_id_to_role[raw_role])
            continue
        if isinstance(contract_nodes, dict) and raw_role in contract_nodes:
            continue
        missing.append(role)
    return tuple(resolved), missing


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
        "leftshoulder": "left_shoulder",
        "leftclavicle": "left_shoulder",
        "rightshoulder": "right_shoulder",
        "rightclavicle": "right_shoulder",
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
        "leftarmproximal": "left_arm_proximal",
        "rightarmproximal": "right_arm_proximal",
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
