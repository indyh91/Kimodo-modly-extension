"""Diagnostics-only GLB world-space foot node sampling."""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


JSON_CHUNK = 0x4E4F534A
BIN_CHUNK = 0x004E4942
FLOAT_COMPONENT_TYPE = 5126
TYPE_ARITY = {"SCALAR": 1, "VEC3": 3, "VEC4": 4}
SUPPORTED_INTERPOLATIONS = {"LINEAR", "STEP"}


@dataclass(frozen=True)
class WorldFootSamplesResult:
    status: str
    samples: dict[str, tuple[tuple[float, float, float], ...]] = field(default_factory=dict)
    reason: str | None = None
    details: dict[str, object] = field(default_factory=dict)


class GlbWorldSpaceError(ValueError):
    def __init__(self, reason: str, details: dict[str, object] | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.details = details or {}


def extract_world_foot_samples(
    path: str | Path,
    humanoid_roles: Iterable[object],
    *,
    expected_frame_count: int | None = None,
) -> WorldFootSamplesResult:
    """Extract trusted left/right foot node-origin world translations from GLB.

    This function is intentionally fail-closed. Unsupported or malformed data is
    reported as ``not_evaluated`` instead of raising so export success remains
    structural and diagnostics-only.
    """

    try:
        document, bin_chunk = _load_glb(Path(path))
        bindings = _trusted_foot_bindings(humanoid_roles, document)
        animation = _select_animation(document)
        times = _animation_times(document, bin_chunk, animation)
        if expected_frame_count is not None and len(times) != expected_frame_count:
            raise GlbWorldSpaceError("sampler_frame_mismatch", {"expected_frame_count": expected_frame_count, "frame_count": len(times)})
        samples = _sample_world_translations(document, bin_chunk, animation, times, bindings)
        return WorldFootSamplesResult(
            status="success",
            samples=samples,
            details={"frame_count": len(times), "left_node": bindings["left"], "right_node": bindings["right"]},
        )
    except GlbWorldSpaceError as exc:
        return WorldFootSamplesResult(status="not_evaluated", reason=exc.reason, details=exc.details)
    except Exception as exc:  # noqa: BLE001 - diagnostics must not block export
        return WorldFootSamplesResult(status="not_evaluated", reason="malformed_glb", details={"error": f"{type(exc).__name__}: {exc}"})


def _load_glb(path: Path) -> tuple[dict[str, Any], bytes]:
    data = path.read_bytes()
    if len(data) < 20 or data[:4] != b"glTF":
        raise GlbWorldSpaceError("malformed_glb")
    version, declared_length = struct.unpack_from("<II", data, 4)
    if version != 2 or declared_length != len(data):
        raise GlbWorldSpaceError("unsupported_asset", {"version": version, "declared_length": declared_length, "actual_length": len(data)})
    offset = 12
    json_payload: bytes | None = None
    bin_payload: bytes | None = None
    while offset + 8 <= len(data):
        chunk_length, chunk_type = struct.unpack_from("<II", data, offset)
        offset += 8
        if offset + chunk_length > len(data):
            raise GlbWorldSpaceError("malformed_glb")
        payload = data[offset:offset + chunk_length]
        offset += chunk_length
        if chunk_type == JSON_CHUNK and json_payload is None:
            json_payload = payload.rstrip(b" \t\r\n\x00")
        elif chunk_type == BIN_CHUNK and bin_payload is None:
            bin_payload = payload
    if offset != len(data) or json_payload is None:
        raise GlbWorldSpaceError("malformed_glb")
    try:
        document = json.loads(json_payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GlbWorldSpaceError("malformed_glb") from exc
    if not isinstance(document, dict) or not isinstance(document.get("nodes"), list) or not isinstance(document.get("animations"), list):
        raise GlbWorldSpaceError("unsupported_asset")
    return document, bin_payload or b""


def _trusted_foot_bindings(humanoid_roles: Iterable[object], document: dict[str, Any]) -> dict[str, int]:
    nodes = document.get("nodes")
    if not isinstance(nodes, list):
        raise GlbWorldSpaceError("unsupported_asset")
    by_role = {getattr(binding, "role", None): binding for binding in humanoid_roles}
    foot_bindings: dict[str, int] = {}
    for side, role in (("left", "left_foot"), ("right", "right_foot")):
        binding = by_role.get(role)
        if binding is None:
            raise GlbWorldSpaceError("missing_trusted_foot_roles")
        node_index = getattr(binding, "node_index", None)
        node_name = getattr(binding, "node_name", None)
        if not isinstance(node_index, int) or node_index < 0 or node_index >= len(nodes):
            raise GlbWorldSpaceError("node_binding_mismatch", {"role": role, "node_index": node_index})
        actual_name = _node_name(nodes, node_index)
        if isinstance(node_name, str) and node_name and node_name != actual_name:
            raise GlbWorldSpaceError("node_binding_mismatch", {"role": role, "node_index": node_index, "expected": node_name, "actual": actual_name})
        foot_bindings[side] = node_index
    return foot_bindings


def _select_animation(document: dict[str, Any]) -> dict[str, Any]:
    animations = document.get("animations")
    if not isinstance(animations, list) or not animations:
        raise GlbWorldSpaceError("unsupported_asset", {"reason": "missing_animations"})
    for animation in reversed(animations):
        if isinstance(animation, dict) and animation.get("name") == "Kimodo Motion":
            return animation
    for animation in reversed(animations):
        if isinstance(animation, dict):
            return animation
    raise GlbWorldSpaceError("unsupported_asset", {"reason": "invalid_animation"})


def _animation_times(document: dict[str, Any], bin_chunk: bytes, animation: dict[str, Any]) -> tuple[float, ...]:
    samplers = animation.get("samplers")
    channels = animation.get("channels")
    if not isinstance(samplers, list) or not isinstance(channels, list) or not samplers or not channels:
        raise GlbWorldSpaceError("unsupported_asset", {"reason": "missing_samplers_or_channels"})
    first_sampler_index = channels[0].get("sampler") if isinstance(channels[0], dict) else None
    if not isinstance(first_sampler_index, int) or first_sampler_index < 0 or first_sampler_index >= len(samplers):
        raise GlbWorldSpaceError("unsupported_asset", {"reason": "invalid_sampler"})
    sampler = samplers[first_sampler_index]
    if not isinstance(sampler, dict) or not isinstance(sampler.get("input"), int):
        raise GlbWorldSpaceError("unsupported_asset", {"reason": "invalid_sampler"})
    raw_times = _decode_accessor(document, bin_chunk, sampler["input"], expected_types={"SCALAR"})
    times = tuple(float(sample[0]) for sample in raw_times)
    if not times or any(times[index] > times[index + 1] for index in range(len(times) - 1)):
        raise GlbWorldSpaceError("sampler_frame_mismatch", {"reason": "non_monotonic_or_empty_times"})
    return times


def _sample_world_translations(
    document: dict[str, Any],
    bin_chunk: bytes,
    animation: dict[str, Any],
    times: tuple[float, ...],
    bindings: dict[str, int],
) -> dict[str, tuple[tuple[float, float, float], ...]]:
    nodes = document["nodes"]
    parents = _parents(nodes)
    animated: dict[int, dict[str, tuple[tuple[float, ...], ...]]] = {}
    samplers = animation["samplers"]
    for channel in animation["channels"]:
        if not isinstance(channel, dict) or not isinstance(channel.get("target"), dict):
            raise GlbWorldSpaceError("unsupported_asset", {"reason": "invalid_channel"})
        sampler_index = channel.get("sampler")
        target = channel["target"]
        node_index = target.get("node")
        path = target.get("path")
        if not isinstance(sampler_index, int) or sampler_index < 0 or sampler_index >= len(samplers) or not isinstance(node_index, int) or path not in {"translation", "rotation", "scale"}:
            raise GlbWorldSpaceError("unsupported_asset", {"reason": "invalid_channel"})
        sampler = samplers[sampler_index]
        interpolation = str(sampler.get("interpolation", "LINEAR"))
        if interpolation not in SUPPORTED_INTERPOLATIONS:
            raise GlbWorldSpaceError(f"unsupported_interpolation:{interpolation}")
        sampler_times = tuple(float(sample[0]) for sample in _decode_accessor(document, bin_chunk, sampler["input"], expected_types={"SCALAR"}))
        if len(sampler_times) != len(times) or any(abs(left - right) > 1e-6 for left, right in zip(sampler_times, times)):
            raise GlbWorldSpaceError("sampler_frame_mismatch", {"node": node_index, "path": path})
        expected_types = {"rotation": {"VEC4"}, "translation": {"VEC3"}, "scale": {"VEC3"}}[path]
        outputs = _decode_accessor(document, bin_chunk, sampler["output"], expected_types=expected_types)
        if len(outputs) != len(times):
            raise GlbWorldSpaceError("sampler_frame_mismatch", {"node": node_index, "path": path})
        animated.setdefault(node_index, {})[path] = tuple(outputs)
    world_samples = []
    for frame_index in range(len(times)):
        local = [_local_matrix(node, animated.get(index, {}), frame_index) for index, node in enumerate(nodes)]
        world = [_identity_matrix() for _ in nodes]
        for node_index in range(len(nodes)):
            world[node_index] = _world_matrix(node_index, parents, local, world)
        world_samples.append(world)
    return {
        side: tuple(_matrix_translation(world[bindings[side]]) for world in world_samples)
        for side in ("left", "right")
    }


def _decode_accessor(document: dict[str, Any], bin_chunk: bytes, accessor_index: int, *, expected_types: set[str]) -> tuple[tuple[float, ...], ...]:
    try:
        accessors = document["accessors"]
        buffer_views = document["bufferViews"]
        accessor = accessors[accessor_index]
        accessor_type = accessor.get("type")
        if accessor_type not in expected_types or accessor_type not in TYPE_ARITY:
            raise GlbWorldSpaceError("unsupported_accessor", {"accessor": accessor_index, "type": accessor_type})
        if accessor.get("componentType") != FLOAT_COMPONENT_TYPE:
            raise GlbWorldSpaceError("unsupported_accessor", {"accessor": accessor_index, "componentType": accessor.get("componentType")})
        count = accessor.get("count")
        view_index = accessor.get("bufferView")
        if not isinstance(count, int) or count < 1 or not isinstance(view_index, int):
            raise GlbWorldSpaceError("unsupported_accessor", {"accessor": accessor_index})
        view = buffer_views[view_index]
        if not isinstance(view, dict) or view.get("buffer", 0) != 0:
            raise GlbWorldSpaceError("unsupported_accessor", {"accessor": accessor_index, "reason": "invalid_buffer_view"})
        arity = TYPE_ARITY[accessor_type]
        item_size = arity * 4
        stride = view.get("byteStride", item_size)
        if not isinstance(stride, int) or stride < item_size:
            raise GlbWorldSpaceError("unsupported_accessor", {"accessor": accessor_index, "reason": "invalid_stride"})
        view_offset = int(view.get("byteOffset") or 0)
        accessor_offset = int(accessor.get("byteOffset") or 0)
        view_length = view.get("byteLength")
        if not isinstance(view_length, int):
            raise GlbWorldSpaceError("unsupported_accessor", {"accessor": accessor_index, "reason": "invalid_length"})
        required = accessor_offset + (count - 1) * stride + item_size
        if required > view_length or view_offset + required > len(bin_chunk):
            raise GlbWorldSpaceError("unsupported_accessor", {"accessor": accessor_index, "reason": "out_of_bounds"})
        samples = []
        for sample_index in range(count):
            offset = view_offset + accessor_offset + sample_index * stride
            values = struct.unpack_from(f"<{arity}f", bin_chunk, offset)
            if not all(math.isfinite(value) for value in values):
                raise GlbWorldSpaceError("unsupported_accessor", {"accessor": accessor_index, "reason": "non_finite"})
            samples.append(tuple(float(value) for value in values))
        return tuple(samples)
    except GlbWorldSpaceError:
        raise
    except Exception as exc:
        raise GlbWorldSpaceError("unsupported_accessor", {"accessor": accessor_index, "error": f"{type(exc).__name__}: {exc}"}) from exc


def _parents(nodes: list[Any]) -> dict[int, int]:
    parents: dict[int, int] = {}
    for index, node in enumerate(nodes):
        if not isinstance(node, dict):
            continue
        for child in node.get("children", []):
            if isinstance(child, int) and 0 <= child < len(nodes):
                parents[child] = index
    return parents


def _local_matrix(node: Any, animated: dict[str, tuple[tuple[float, ...], ...]], frame_index: int) -> tuple[tuple[float, float, float, float], ...]:
    node = node if isinstance(node, dict) else {}
    translation = animated.get("translation", ())
    rotation = animated.get("rotation", ())
    scale = animated.get("scale", ())
    t = _vec3(translation[frame_index] if translation else node.get("translation"), (0.0, 0.0, 0.0))
    r = _quat(rotation[frame_index] if rotation else node.get("rotation"))
    s = _vec3(scale[frame_index] if scale else node.get("scale"), (1.0, 1.0, 1.0))
    return _matmul(_translation_matrix(t), _matmul(_rotation_matrix(r), _scale_matrix(s)))


def _world_matrix(index: int, parents: dict[int, int], local: list[tuple[tuple[float, float, float, float], ...]], world: list[tuple[tuple[float, float, float, float], ...]]) -> tuple[tuple[float, float, float, float], ...]:
    parent = parents.get(index)
    if parent is None:
        return local[index]
    parent_world = world[parent]
    if parent > index:
        parent_world = _world_matrix(parent, parents, local, world)
    return _matmul(parent_world, local[index])


def _vec3(value: Any, default: tuple[float, float, float]) -> tuple[float, float, float]:
    if isinstance(value, (list, tuple)) and len(value) == 3:
        try:
            return (float(value[0]), float(value[1]), float(value[2]))
        except (TypeError, ValueError):
            return default
    return default


def _quat(value: Any) -> tuple[float, float, float, float]:
    if isinstance(value, (list, tuple)) and len(value) == 4:
        try:
            x, y, z, w = (float(component) for component in value)
        except (TypeError, ValueError):
            return (0.0, 0.0, 0.0, 1.0)
        length = math.sqrt(x * x + y * y + z * z + w * w)
        if length > 0.0:
            return (x / length, y / length, z / length, w / length)
    return (0.0, 0.0, 0.0, 1.0)


def _identity_matrix() -> tuple[tuple[float, float, float, float], ...]:
    return ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0))


def _translation_matrix(v: tuple[float, float, float]) -> tuple[tuple[float, float, float, float], ...]:
    return ((1.0, 0.0, 0.0, v[0]), (0.0, 1.0, 0.0, v[1]), (0.0, 0.0, 1.0, v[2]), (0.0, 0.0, 0.0, 1.0))


def _scale_matrix(v: tuple[float, float, float]) -> tuple[tuple[float, float, float, float], ...]:
    return ((v[0], 0.0, 0.0, 0.0), (0.0, v[1], 0.0, 0.0), (0.0, 0.0, v[2], 0.0), (0.0, 0.0, 0.0, 1.0))


def _rotation_matrix(q: tuple[float, float, float, float]) -> tuple[tuple[float, float, float, float], ...]:
    x, y, z, w = q
    return (
        (1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w, 0.0),
        (2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w, 0.0),
        (2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )


def _matmul(left: tuple[tuple[float, float, float, float], ...], right: tuple[tuple[float, float, float, float], ...]) -> tuple[tuple[float, float, float, float], ...]:
    return tuple(
        tuple(sum(left[row][k] * right[k][column] for k in range(4)) for column in range(4))
        for row in range(4)
    )


def _matrix_translation(matrix: tuple[tuple[float, float, float, float], ...]) -> tuple[float, float, float]:
    return (_clean_float(matrix[0][3]), _clean_float(matrix[1][3]), _clean_float(matrix[2][3]))


def _node_name(nodes: list[Any], index: int) -> str:
    node = nodes[index]
    if isinstance(node, dict) and isinstance(node.get("name"), str) and node["name"].strip():
        return node["name"].strip()
    return f"node_{index}"


def _clean_float(value: float) -> float:
    rounded = round(float(value), 10)
    return 0.0 if abs(rounded) < 1e-10 else rounded
