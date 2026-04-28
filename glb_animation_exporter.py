"""Animated GLTF/GLB writer adapter for conservative retarget clips."""

from __future__ import annotations

import base64
import copy
import importlib.util
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from retarget_errors import EXPORT_FAILED, RetargetError
from retargeting import RetargetClip
from rig_inspector import RigInspection


@dataclass(frozen=True)
class ExporterDependencyReport:
    pygltflib_available: bool
    gltflib_available: bool
    trimesh_available: bool

    @property
    def animated_skin_writer_confirmed(self) -> bool:
        return self.pygltflib_available or self.gltflib_available


def inspect_exporter_dependencies() -> ExporterDependencyReport:
    return ExporterDependencyReport(
        pygltflib_available=importlib.util.find_spec("pygltflib") is not None,
        gltflib_available=importlib.util.find_spec("gltflib") is not None,
        trimesh_available=importlib.util.find_spec("trimesh") is not None,
    )


class GlbAnimationExporter:
    """Append clip animation data while preserving the source glTF document.

    This deliberately supports the MVP seam only: translation channels generated
    by :func:`retargeting.build_retarget_clip`. It does not pretend to solve full
    arbitrary-rig retargeting; unsupported channel shapes fail closed.
    """

    clip_name = "Kimodo Motion"

    def write_animation(self, rig: RigInspection, clip: RetargetClip, output_path: Path) -> Path:
        output_path = Path(output_path)
        if output_path.suffix.lower() not in {".glb", ".gltf"}:
            raise RetargetError(EXPORT_FAILED, f"Animation output must be .glb or .gltf: {output_path}")
        if not clip.channels:
            raise RetargetError(EXPORT_FAILED, "Retarget clip contains no animation channels to write.")

        document = _read_source_document(rig.path)
        document = copy.deepcopy(document)
        _append_animation(document, rig, clip)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.suffix.lower() == ".glb":
            _write_json_glb(document, output_path)
        else:
            output_path.write_text(f"{json.dumps(document, indent=2)}\n", encoding="utf-8")
        return output_path


def _read_source_document(path: Path) -> dict[str, Any]:
    try:
        if path.suffix.lower() == ".gltf":
            payload = json.loads(path.read_text(encoding="utf-8"))
        elif path.suffix.lower() == ".glb":
            payload = _read_glb_json(path)
        else:
            raise ValueError("unsupported source extension")
    except Exception as exc:
        raise RetargetError(EXPORT_FAILED, f"Could not read source rig document for animation export: {path}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("asset"), dict):
        raise RetargetError(EXPORT_FAILED, f"Source rig document is not valid glTF: {path}")
    return payload


def _append_animation(document: dict[str, Any], rig: RigInspection, clip: RetargetClip) -> None:
    nodes = document.get("nodes")
    if not isinstance(nodes, list):
        raise RetargetError(EXPORT_FAILED, "Source rig document has no nodes array.")

    samplers: list[dict[str, Any]] = []
    channels: list[dict[str, Any]] = []
    for channel in clip.channels:
        if channel.path != "translation":
            raise RetargetError(EXPORT_FAILED, f"Unsupported retarget channel path: {channel.path}")
        if not channel.times or len(channel.times) != len(channel.values):
            raise RetargetError(EXPORT_FAILED, "Retarget channel has mismatched time/value samples.")
        target_node = _find_node_index(nodes, channel.target_joint)
        if target_node is None:
            raise RetargetError(EXPORT_FAILED, f"Mapped target joint is missing from glTF nodes: {channel.target_joint}")

        input_accessor = _append_accessor_buffer(
            document,
            _pack_float_scalars(channel.times),
            component_type=5126,
            accessor_type="SCALAR",
            count=len(channel.times),
            minimum=[min(channel.times)],
            maximum=[max(channel.times)],
        )
        output_accessor = _append_accessor_buffer(
            document,
            _pack_vec3_values(channel.values),
            component_type=5126,
            accessor_type="VEC3",
            count=len(channel.values),
        )
        samplers.append({"input": input_accessor, "output": output_accessor, "interpolation": channel.interpolation})
        channels.append({"sampler": len(samplers) - 1, "target": {"node": target_node, "path": channel.path}})

    animations = document.setdefault("animations", [])
    if not isinstance(animations, list):
        raise RetargetError(EXPORT_FAILED, "Source rig document has a non-list animations field.")
    animations.append({"name": clip.name or GlbAnimationExporter.clip_name, "samplers": samplers, "channels": channels})


def _append_accessor_buffer(
    document: dict[str, Any],
    payload: bytes,
    *,
    component_type: int,
    accessor_type: str,
    count: int,
    minimum: list[float] | None = None,
    maximum: list[float] | None = None,
) -> int:
    buffers = document.setdefault("buffers", [])
    buffer_views = document.setdefault("bufferViews", [])
    accessors = document.setdefault("accessors", [])
    if not isinstance(buffers, list) or not isinstance(buffer_views, list) or not isinstance(accessors, list):
        raise RetargetError(EXPORT_FAILED, "Source rig document has invalid buffer/accessor arrays.")

    buffer_index = len(buffers)
    buffers.append({
        "byteLength": len(payload),
        "uri": "data:application/octet-stream;base64," + base64.b64encode(payload).decode("ascii"),
    })
    buffer_view_index = len(buffer_views)
    buffer_views.append({"buffer": buffer_index, "byteOffset": 0, "byteLength": len(payload)})
    accessor: dict[str, Any] = {
        "bufferView": buffer_view_index,
        "componentType": component_type,
        "count": count,
        "type": accessor_type,
    }
    if minimum is not None:
        accessor["min"] = minimum
    if maximum is not None:
        accessor["max"] = maximum
    accessors.append(accessor)
    return len(accessors) - 1


def _pack_float_scalars(values: tuple[float, ...]) -> bytes:
    return struct.pack(f"<{len(values)}f", *values)


def _pack_vec3_values(values: tuple[tuple[float, ...], ...]) -> bytes:
    flat: list[float] = []
    for value in values:
        if len(value) != 3:
            raise RetargetError(EXPORT_FAILED, "Only VEC3 translation channels are supported.")
        flat.extend(float(component) for component in value)
    return struct.pack(f"<{len(flat)}f", *flat)


def _find_node_index(nodes: list[Any], target_joint: str) -> int | None:
    for index, node in enumerate(nodes):
        if isinstance(node, dict) and node.get("name") == target_joint:
            return index
    return None


def _write_json_glb(document: dict[str, Any], output_path: Path) -> None:
    json_chunk = json.dumps(document, separators=(",", ":")).encode("utf-8")
    json_padding = (4 - (len(json_chunk) % 4)) % 4
    json_chunk += b" " * json_padding
    total_length = 12 + 8 + len(json_chunk)
    with output_path.open("wb") as handle:
        handle.write(b"glTF")
        handle.write(struct.pack("<II", 2, total_length))
        handle.write(struct.pack("<II", len(json_chunk), 0x4E4F534A))
        handle.write(json_chunk)


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
