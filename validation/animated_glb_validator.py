"""Structural validator for Kimodo animated GLB/GLTF motion bundles.

This module intentionally validates structure only. A passing report means the
bundle contains parseable glTF 2.0 animation data, non-zero decoded samples, and
matching Kimodo retarget metadata. It does not prove visual quality.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import struct
from pathlib import Path
from typing import Any


JSON_CHUNK = 0x4E4F534A
BIN_CHUNK = 0x004E4942
FLOAT_COMPONENT_TYPE = 5126
EXPECTED_ARTIFACTS = ("motion.npz", "motion.bvh", "metadata.json", "preview.glb", "animated.glb")
TARGET_OUTPUT_TYPES = {
    "translation": {"VEC3"},
    "scale": {"VEC3"},
    "rotation": {"VEC4"},
    "weights": {"SCALAR"},
}
TYPE_ARITY = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}
LIMITATION = "Structural validation only: parseable non-zero animation data does not prove visual quality."


def validate_bundle(bundle_path: str | Path, *, animated_name: str = "animated.glb", epsilon: float = 1e-6) -> dict[str, Any]:
    """Validate a Kimodo animation bundle and return a JSON-serializable report."""

    bundle = Path(bundle_path)
    report = _new_report(bundle, bundle / animated_name)
    _validate_artifacts(bundle, report, animated_name=animated_name)
    metadata = _validate_metadata(bundle, report, animated_name=animated_name)

    animated_path = bundle / animated_name
    if animated_path.exists():
        try:
            document, bin_chunk = load_gltf(animated_path)
        except Exception as exc:  # noqa: BLE001 - report must classify all parse failures
            _error(report, "invalid_glb_structure", f"Could not parse animated asset: {exc}", str(animated_path))
        else:
            validate_gltf_document(document, report=report, source_path=animated_path, bin_chunk=bin_chunk, epsilon=epsilon)

    report["metadata"] = metadata
    report["passed"] = not report["errors"]
    return report


def validate_gltf_document(
    document: dict[str, Any],
    *,
    epsilon: float = 1e-6,
    report: dict[str, Any] | None = None,
    source_path: str | Path | None = None,
    bin_chunk: bytes | None = None,
) -> dict[str, Any]:
    """Validate glTF animation structure from an already loaded document."""

    own_report = report is None
    if report is None:
        source = Path(source_path) if source_path is not None else Path("<in-memory>")
        report = _new_report(source.parent, source)

    if not isinstance(document, dict):
        _error(report, "invalid_glb_structure", "glTF document is not an object.", str(source_path or "<in-memory>"))
        report["passed"] = False
        return report

    if not _validate_document_shape(document, report, source_path):
        report["passed"] = False
        return report

    animations = document["animations"]
    if not animations:
        _error(report, "no_animation_clips", "No animation clips found.", str(source_path or "<in-memory>"))
        report["passed"] = False
        return report

    animation = _select_animation(animations, report)
    if not isinstance(animation, dict):
        _error(report, "bad_animation_reference", "Selected animation is not an object.", str(source_path or "<in-memory>"))
        report["passed"] = False
        return report

    resolver = BufferResolver(document, bin_chunk=bin_chunk)
    _validate_animation(document, animation, resolver, report, epsilon=epsilon, source_path=source_path)
    report["passed"] = not report["errors"] if own_report else report.get("passed", False)
    return report


def load_gltf(path: str | Path) -> tuple[dict[str, Any], bytes | None]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".gltf":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload, None
    if suffix == ".glb":
        return _load_glb(path)
    raise ValueError(f"unsupported animated asset extension: {path.suffix}")


class BufferResolver:
    """Resolve glTF buffers for MVP accessor decoding."""

    def __init__(self, document: dict[str, Any], *, bin_chunk: bytes | None = None):
        self.document = document
        self.bin_chunk = bin_chunk
        self._cache: dict[int, bytes] = {}

    def get(self, index: int) -> bytes:
        if index in self._cache:
            return self._cache[index]
        buffers = self.document.get("buffers", [])
        if index < 0 or index >= len(buffers) or not isinstance(buffers[index], dict):
            raise AccessorError(f"buffer index {index} is invalid")
        buffer = buffers[index]
        uri = buffer.get("uri")
        if uri is None:
            if index == 0 and self.bin_chunk is not None:
                data = self.bin_chunk
            else:
                raise AccessorError("buffer uses unsupported external/BIN storage")
        elif isinstance(uri, str) and uri.startswith("data:"):
            data = _decode_data_uri(uri)
        else:
            raise AccessorError("external buffer URIs are unsupported by this validator")
        byte_length = buffer.get("byteLength")
        if isinstance(byte_length, int) and len(data) < byte_length:
            raise AccessorError(f"buffer {index} shorter than declared byteLength")
        self._cache[index] = data
        return data


class AccessorError(ValueError):
    """Raised when an accessor cannot be structurally decoded."""


def decode_accessor(document: dict[str, Any], resolver: BufferResolver, accessor_index: int, *, expected_types: set[str]) -> list[tuple[float, ...]]:
    accessors = document["accessors"]
    buffer_views = document["bufferViews"]
    if accessor_index < 0 or accessor_index >= len(accessors) or not isinstance(accessors[accessor_index], dict):
        raise AccessorError(f"accessor index {accessor_index} is invalid")
    accessor = accessors[accessor_index]
    accessor_type = accessor.get("type")
    if accessor_type not in expected_types or accessor_type not in TYPE_ARITY:
        raise AccessorError(f"accessor {accessor_index} has unsupported type {accessor_type!r}")
    if accessor.get("componentType") != FLOAT_COMPONENT_TYPE:
        raise AccessorError(f"accessor {accessor_index} must use FLOAT componentType")
    count = accessor.get("count")
    if not isinstance(count, int) or count < 1:
        raise AccessorError(f"accessor {accessor_index} has invalid count")
    buffer_view_index = accessor.get("bufferView")
    if not isinstance(buffer_view_index, int) or buffer_view_index < 0 or buffer_view_index >= len(buffer_views):
        raise AccessorError(f"accessor {accessor_index} references invalid bufferView")
    buffer_view = buffer_views[buffer_view_index]
    if not isinstance(buffer_view, dict):
        raise AccessorError(f"bufferView {buffer_view_index} is invalid")
    buffer_index = buffer_view.get("buffer")
    if not isinstance(buffer_index, int):
        raise AccessorError(f"bufferView {buffer_view_index} has invalid buffer reference")
    arity = TYPE_ARITY[accessor_type]
    stride = buffer_view.get("byteStride")
    item_size = arity * 4
    if stride is None:
        stride = item_size
    if not isinstance(stride, int) or stride < item_size:
        raise AccessorError(f"bufferView {buffer_view_index} has invalid byteStride")
    base_offset = int(buffer_view.get("byteOffset") or 0) + int(accessor.get("byteOffset") or 0)
    view_length = buffer_view.get("byteLength")
    if not isinstance(view_length, int) or view_length < item_size:
        raise AccessorError(f"bufferView {buffer_view_index} has invalid byteLength")
    required_within_view = (count - 1) * stride + item_size
    if int(accessor.get("byteOffset") or 0) + required_within_view > view_length:
        raise AccessorError(f"accessor {accessor_index} exceeds bufferView bounds")
    data = resolver.get(buffer_index)
    if base_offset + required_within_view > len(data):
        raise AccessorError(f"accessor {accessor_index} exceeds buffer bounds")
    samples: list[tuple[float, ...]] = []
    for sample_index in range(count):
        offset = base_offset + sample_index * stride
        values = struct.unpack_from(f"<{arity}f", data, offset)
        if not all(math.isfinite(value) for value in values):
            raise AccessorError(f"accessor {accessor_index} contains non-finite samples")
        samples.append(tuple(float(value) for value in values))
    return samples


def _new_report(bundle_path: Path, animated_path: Path) -> dict[str, Any]:
    return {
        "passed": False,
        "errors": [],
        "warnings": [],
        "bundle_path": str(bundle_path),
        "animated_glb": str(animated_path),
        "animation": {
            "name": None,
            "channels": 0,
            "samplers": 0,
            "changed_samples": 0,
            "target_paths": [],
            "skin_joint_targets": [],
        },
        "artifacts": {"expected": list(EXPECTED_ARTIFACTS), "present": [], "missing": []},
        "metadata": {},
        "limitations": [LIMITATION],
    }


def _validate_artifacts(bundle: Path, report: dict[str, Any], *, animated_name: str) -> None:
    expected = tuple(animated_name if artifact == "animated.glb" else artifact for artifact in EXPECTED_ARTIFACTS)
    present = [artifact for artifact in expected if (bundle / artifact).is_file()]
    missing = [artifact for artifact in expected if artifact not in present]
    report["artifacts"] = {"expected": list(expected), "present": present, "missing": missing}
    for artifact in missing:
        _error(report, "missing_artifact", f"Required artifact is missing: {artifact}", str(bundle / artifact))


def _validate_metadata(bundle: Path, report: dict[str, Any], *, animated_name: str) -> dict[str, Any]:
    path = bundle / "metadata.json"
    if not path.exists():
        return {}
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        _error(report, "metadata_mismatch", f"metadata.json is not valid JSON: {exc}", str(path))
        return {}
    if not isinstance(metadata, dict):
        _error(report, "metadata_mismatch", "metadata.json must contain an object.", str(path))
        return {}

    if metadata.get("retarget_status") != "success":
        _error(report, "metadata_mismatch", "metadata retarget_status must be success.", str(path))
    source_path = metadata.get("source_rigged_mesh") or metadata.get("source_rig_path") or metadata.get("animated_glb")
    source_kind = metadata.get("source_kind") or metadata.get("rigged_mesh_source_kind")
    if not source_path:
        _error(report, "metadata_mismatch", "metadata must record source rig path.", str(path))
    if not source_kind:
        _error(report, "metadata_mismatch", "metadata must record source kind.", str(path))
    if not metadata.get("classification"):
        _error(report, "metadata_mismatch", "metadata must record rig classification.", str(path))
    warnings = metadata.get("warnings")
    if metadata.get("mapping_confidence") in (None, "") and not (isinstance(warnings, list) and warnings):
        _error(report, "metadata_mismatch", "metadata must include mapping_confidence or warnings.", str(path))

    artifacts = metadata.get("artifacts")
    if not isinstance(artifacts, list):
        _error(report, "metadata_mismatch", "metadata artifacts must be a list.", str(path))
    else:
        for expected in report["artifacts"]["expected"]:
            if expected not in artifacts:
                _error(report, "metadata_mismatch", f"metadata artifacts missing {expected}.", str(path))
        for artifact in artifacts:
            if isinstance(artifact, str) and not (bundle / artifact).exists():
                _error(report, "metadata_mismatch", f"metadata artifact does not exist: {artifact}.", str(bundle / artifact))

    return {
        "retarget_status": metadata.get("retarget_status"),
        "source_kind": source_kind,
        "classification": metadata.get("classification"),
        "mapping_confidence": metadata.get("mapping_confidence"),
        "warnings": warnings if isinstance(warnings, list) else [],
        "source_rigged_mesh": source_path,
        "artifacts": artifacts if isinstance(artifacts, list) else [],
    }


def _load_glb(path: Path) -> tuple[dict[str, Any], bytes | None]:
    data = path.read_bytes()
    if len(data) < 20 or data[:4] != b"glTF":
        raise ValueError("invalid GLB header")
    version, length = struct.unpack_from("<II", data, 4)
    if version != 2:
        raise ValueError("only GLB v2 is supported")
    if length != len(data):
        raise ValueError("GLB declared length does not match file length")
    offset = 12
    json_payload: bytes | None = None
    bin_payload: bytes | None = None
    while offset + 8 <= len(data):
        chunk_length, chunk_type = struct.unpack_from("<II", data, offset)
        offset += 8
        if offset + chunk_length > len(data):
            raise ValueError("GLB chunk exceeds file bounds")
        payload = data[offset:offset + chunk_length]
        offset += chunk_length
        if chunk_type == JSON_CHUNK and json_payload is None:
            json_payload = payload.rstrip(b" \t\r\n\x00")
        elif chunk_type == BIN_CHUNK and bin_payload is None:
            bin_payload = payload
    if offset != len(data):
        raise ValueError("GLB chunks are not aligned with file length")
    if json_payload is None:
        raise ValueError("GLB JSON chunk is missing")
    return json.loads(json_payload.decode("utf-8")), bin_payload


def _validate_document_shape(document: dict[str, Any], report: dict[str, Any], source_path: str | Path | None) -> bool:
    ok = True
    asset = document.get("asset")
    if not isinstance(asset, dict) or str(asset.get("version")) != "2.0":
        _error(report, "invalid_glb_structure", "asset.version must be 2.0.", str(source_path or "<in-memory>"))
        ok = False
    for field in ("nodes", "skins", "buffers", "bufferViews", "accessors", "animations"):
        if not isinstance(document.get(field), list):
            _error(report, "invalid_glb_structure", f"glTF field {field} must be a list.", str(source_path or "<in-memory>"))
            ok = False
    if ok and not document["skins"]:
        _error(report, "no_skin_skeleton", "glTF document contains no skins/skeleton.", str(source_path or "<in-memory>"))
        ok = False
    return ok


def _select_animation(animations: list[Any], report: dict[str, Any]) -> Any:
    for animation in animations:
        if isinstance(animation, dict) and animation.get("name") == "Kimodo Motion":
            return animation
    _warning(report, "animation_name_fallback", "Animation named 'Kimodo Motion' not found; validating first animation.")
    return animations[0]


def _validate_animation(
    document: dict[str, Any],
    animation: dict[str, Any],
    resolver: BufferResolver,
    report: dict[str, Any],
    *,
    epsilon: float,
    source_path: str | Path | None,
) -> None:
    samplers = animation.get("samplers")
    channels = animation.get("channels")
    if not isinstance(samplers, list) or not samplers:
        _error(report, "bad_animation_reference", "Animation has no samplers.", str(source_path or "<in-memory>"))
        return
    if not isinstance(channels, list) or not channels:
        _error(report, "bad_animation_reference", "Animation has no channels.", str(source_path or "<in-memory>"))
        return
    report["animation"]["name"] = animation.get("name")
    report["animation"]["channels"] = len(channels)
    report["animation"]["samplers"] = len(samplers)

    skin_joints = _skin_joints(document)
    changed_samples = 0
    valid_channels = 0
    target_paths: list[str] = []
    skin_target_nodes: list[int] = []

    for channel_index, channel in enumerate(channels):
        if not isinstance(channel, dict):
            _error(report, "bad_animation_reference", f"Channel {channel_index} is not an object.", str(source_path or "<in-memory>"))
            continue
        sampler_index = channel.get("sampler")
        target = channel.get("target")
        if not isinstance(sampler_index, int) or sampler_index < 0 or sampler_index >= len(samplers):
            _error(report, "bad_animation_reference", f"Channel {channel_index} references invalid sampler.", str(source_path or "<in-memory>"))
            continue
        if not isinstance(target, dict) or not isinstance(target.get("node"), int) or not isinstance(target.get("path"), str):
            _error(report, "bad_animation_reference", f"Channel {channel_index} has invalid target.", str(source_path or "<in-memory>"))
            continue
        node_index = target["node"]
        target_path = target["path"]
        if node_index < 0 or node_index >= len(document["nodes"]):
            _error(report, "bad_animation_reference", f"Channel {channel_index} targets invalid node {node_index}.", str(source_path or "<in-memory>"))
            continue
        if target_path not in TARGET_OUTPUT_TYPES:
            _error(report, "bad_animation_reference", f"Unsupported animation target path: {target_path}.", str(source_path or "<in-memory>"))
            continue
        sampler = samplers[sampler_index]
        if not isinstance(sampler, dict) or not isinstance(sampler.get("input"), int) or not isinstance(sampler.get("output"), int):
            _error(report, "bad_animation_reference", f"Sampler {sampler_index} is invalid.", str(source_path or "<in-memory>"))
            continue
        try:
            times = decode_accessor(document, resolver, sampler["input"], expected_types={"SCALAR"})
            outputs = decode_accessor(document, resolver, sampler["output"], expected_types=TARGET_OUTPUT_TYPES[target_path])
        except AccessorError as exc:
            _error(report, "malformed_accessor", str(exc), str(source_path or "<in-memory>"))
            continue

        if len(times) < 2:
            _error(report, "bad_animation_reference", f"Sampler {sampler_index} must contain at least 2 timestamps.", str(source_path or "<in-memory>"))
            continue
        if len(times) != len(outputs):
            _error(report, "bad_animation_reference", f"Sampler {sampler_index} input/output counts differ.", str(source_path or "<in-memory>"))
            continue
        if any(times[index][0] > times[index + 1][0] for index in range(len(times) - 1)):
            _error(report, "bad_animation_reference", f"Sampler {sampler_index} timestamps are not monotonic.", str(source_path or "<in-memory>"))
            continue

        base = outputs[0]
        channel_changed = sum(1 for sample in outputs[1:] if any(abs(value - base[i]) > epsilon for i, value in enumerate(sample)))
        changed_samples += channel_changed
        valid_channels += 1
        target_paths.append(target_path)
        if skin_joints is None:
            _warning(report, "skin_relation_unavailable", "No determinable skin joint list for target membership check.")
        elif node_index in skin_joints:
            skin_target_nodes.append(node_index)
        else:
            _warning(report, "target_not_skin_joint", f"Animation targets node {node_index}, which is not listed in skins[].joints.")

    report["animation"]["changed_samples"] = changed_samples
    report["animation"]["target_paths"] = sorted(set(target_paths))
    report["animation"]["skin_joint_targets"] = sorted(set(skin_target_nodes))
    if valid_channels == 0:
        _error(report, "bad_animation_reference", "No valid animation channels could be decoded.", str(source_path or "<in-memory>"))
    elif changed_samples == 0:
        _error(report, "static_noop_animation", "Decoded animation samples do not change beyond epsilon.", str(source_path or "<in-memory>"))


def _skin_joints(document: dict[str, Any]) -> set[int] | None:
    joints: set[int] = set()
    determinable = False
    for skin in document.get("skins", []):
        if isinstance(skin, dict) and isinstance(skin.get("joints"), list):
            determinable = True
            joints.update(joint for joint in skin["joints"] if isinstance(joint, int))
    return joints if determinable else None


def _decode_data_uri(uri: str) -> bytes:
    try:
        header, payload = uri.split(",", 1)
    except ValueError as exc:
        raise AccessorError("data URI is malformed") from exc
    if ";base64" not in header:
        raise AccessorError("only base64 data URIs are supported")
    try:
        return base64.b64decode(payload, validate=True)
    except Exception as exc:  # noqa: BLE001
        raise AccessorError("base64 data URI could not be decoded") from exc


def _error(report: dict[str, Any], code: str, message: str, path: str) -> None:
    report["errors"].append({"code": code, "message": message, "path": path})


def _warning(report: dict[str, Any], code: str, message: str) -> None:
    report["warnings"].append({"code": code, "message": message})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate Kimodo animated GLB bundle structure.")
    parser.add_argument("--bundle", required=True, help="Path to Kimodo output bundle directory.")
    parser.add_argument("--animated-name", default="animated.glb", help="Animated asset filename inside bundle.")
    parser.add_argument("--epsilon", type=float, default=1e-6, help="Motion delta epsilon.")
    parser.add_argument("--json-report", help="Optional path to write JSON report.")
    args = parser.parse_args(argv)

    report = validate_bundle(args.bundle, animated_name=args.animated_name, epsilon=args.epsilon)
    output = json.dumps(report, indent=2, sort_keys=True)
    if args.json_report:
        Path(args.json_report).write_text(f"{output}\n", encoding="utf-8")
    else:
        print(output)
    return 0 if report["passed"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
