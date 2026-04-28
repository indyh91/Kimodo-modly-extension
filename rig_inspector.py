"""Rigged GLB/GLTF inspection for the retarget/export seam."""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from retarget_errors import RIG_HAS_NO_SKIN_OR_SKELETON, UNPARSEABLE_RIG, RetargetError


@dataclass(frozen=True)
class RigInspection:
    path: Path
    source_kind: str
    classification: str
    joint_names: tuple[str, ...]
    skin_count: int
    has_inverse_bind_matrices: bool
    warnings: tuple[str, ...] = field(default_factory=tuple)

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
    classification, warnings = _classify(joint_names)
    if not has_inverse_bind:
        warnings.append("missing_inverse_bind_matrices")
    return RigInspection(
        path=path.resolve(),
        source_kind=source_kind,
        classification=classification,
        joint_names=joint_names,
        skin_count=len(skins),
        has_inverse_bind_matrices=has_inverse_bind,
        warnings=tuple(warnings),
    )


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
