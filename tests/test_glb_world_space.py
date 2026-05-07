import json
import math
import struct
import tempfile
import unittest
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from glb_world_space import extract_world_foot_samples  # noqa: E402
from rig_inspector import HumanoidRoleBinding  # noqa: E402


JSON_CHUNK = 0x4E4F534A
BIN_CHUNK = 0x004E4942


class GlbWorldSpaceTests(unittest.TestCase):
    def test_extracts_world_foot_samples_from_glb_with_strided_float_accessors(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "animated.glb"
            self._write_world_foot_glb(path, left_stride=True)

            result = extract_world_foot_samples(path, self._foot_roles())

        self.assertEqual(result.status, "success")
        self.assertIsNone(result.reason)
        self.assertEqual(result.details["frame_count"], 2)
        self.assertEqual(result.samples["left"][0], (1.0, 0.0, 0.0))
        self.assertTupleAlmostEqual(result.samples["left"][1], (0.0, 2.0, 0.0))
        self.assertEqual(result.samples["right"][0], (0.0, 1.0, 0.0))
        self.assertTupleAlmostEqual(result.samples["right"][1], (-1.0, 0.0, 0.0))

    def test_trs_defaults_parent_world_propagation_and_xyzw_quaternion_rotation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "animated.glb"
            self._write_world_foot_glb(path, root_scale=(2.0, 1.0, 1.0))

            result = extract_world_foot_samples(path, self._foot_roles())

        self.assertEqual(result.status, "success")
        self.assertEqual(result.samples["left"][0], (2.0, 0.0, 0.0))
        self.assertTupleAlmostEqual(result.samples["left"][1], (0.0, 4.0, 0.0))
        self.assertTupleAlmostEqual(result.samples["right"][1], (-1.0, 0.0, 0.0))

    def test_degrades_for_missing_roles_node_mismatch_malformed_accessor_and_sampler_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "animated.glb"
            self._write_world_foot_glb(path)
            malformed = Path(tmp) / "bad.glb"
            malformed.write_bytes(b"not a glb")
            unsupported_accessor = Path(tmp) / "unsupported_accessor.glb"
            self._write_world_foot_glb(unsupported_accessor, left_component_type=5123)
            cubic = Path(tmp) / "cubic.glb"
            self._write_world_foot_glb(cubic, interpolation="CUBICSPLINE")

            cases = (
                (path, (), "missing_trusted_foot_roles"),
                (path, (HumanoidRoleBinding(role="left_foot", node_index=1, node_name="wrong"), HumanoidRoleBinding(role="right_foot", node_index=2, node_name="RightFoot")), "node_binding_mismatch"),
                (malformed, self._foot_roles(), "malformed_glb"),
                (unsupported_accessor, self._foot_roles(), "unsupported_accessor"),
                (cubic, self._foot_roles(), "unsupported_interpolation:CUBICSPLINE"),
                (path, self._foot_roles(), "sampler_frame_mismatch"),
            )

            for asset, roles, reason in cases:
                with self.subTest(reason=reason):
                    expected_frame_count = 3 if reason == "sampler_frame_mismatch" else None
                    result = extract_world_foot_samples(asset, roles, expected_frame_count=expected_frame_count)
                    self.assertEqual(result.status, "not_evaluated")
                    self.assertEqual(result.reason, reason)
                    self.assertEqual(result.samples, {})

    def _foot_roles(self):
        return (
            HumanoidRoleBinding(role="left_foot", node_index=1, node_name="LeftFoot"),
            HumanoidRoleBinding(role="right_foot", node_index=2, node_name="RightFoot"),
        )

    def _write_world_foot_glb(
        self,
        path,
        *,
        left_stride=False,
        root_scale=(1.0, 1.0, 1.0),
        left_component_type=5126,
        interpolation="LINEAR",
    ):
        frames = [0.0, 1.0]
        root_rotations = [(0.0, 0.0, 0.0, 1.0), (0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5))]
        left_translations = [(1.0, 0.0, 0.0), (2.0, 0.0, 0.0)]
        right_translations = [(0.0, 1.0, 0.0), (0.0, 1.0, 0.0)]

        binary = bytearray()
        buffer_views = []
        accessors = []

        def add_accessor(samples, accessor_type, *, component_type=5126, stride=None, prefix=b""):
            arity = {"SCALAR": 1, "VEC3": 3, "VEC4": 4}[accessor_type]
            binary.extend(b"\x00" * ((4 - (len(binary) % 4)) % 4))
            view_offset = len(binary)
            binary.extend(prefix)
            accessor_offset = len(prefix)
            item_size = arity * 4
            actual_stride = stride or item_size
            for sample in samples:
                values = (sample,) if isinstance(sample, float) else sample
                binary.extend(struct.pack(f"<{arity}f", *values))
                binary.extend(b"\x99" * (actual_stride - item_size))
            byte_length = len(binary) - view_offset
            view = {"buffer": 0, "byteOffset": view_offset, "byteLength": byte_length}
            if stride is not None:
                view["byteStride"] = stride
            buffer_views.append(view)
            accessor = {"bufferView": len(buffer_views) - 1, "byteOffset": accessor_offset, "componentType": component_type, "count": len(samples), "type": accessor_type}
            accessors.append(accessor)
            return len(accessors) - 1

        time_accessor = add_accessor(frames, "SCALAR")
        root_rotation_accessor = add_accessor(root_rotations, "VEC4")
        left_accessor = add_accessor(left_translations, "VEC3", component_type=left_component_type, stride=16 if left_stride else None, prefix=b"xxxx" if left_stride else b"")
        right_accessor = add_accessor(right_translations, "VEC3")
        document = {
            "asset": {"version": "2.0"},
            "nodes": [
                {"name": "Root", "children": [1, 2], "scale": list(root_scale)},
                {"name": "LeftFoot"},
                {"name": "RightFoot", "translation": [0.0, 1.0, 0.0]},
            ],
            "skins": [{"joints": [0, 1, 2]}],
            "buffers": [{"byteLength": len(binary)}],
            "bufferViews": buffer_views,
            "accessors": accessors,
            "animations": [
                {
                    "name": "Kimodo Motion",
                    "samplers": [
                        {"input": time_accessor, "output": root_rotation_accessor, "interpolation": interpolation},
                        {"input": time_accessor, "output": left_accessor, "interpolation": interpolation},
                        {"input": time_accessor, "output": right_accessor, "interpolation": "STEP"},
                    ],
                    "channels": [
                        {"sampler": 0, "target": {"node": 0, "path": "rotation"}},
                        {"sampler": 1, "target": {"node": 1, "path": "translation"}},
                        {"sampler": 2, "target": {"node": 2, "path": "translation"}},
                    ],
                }
            ],
        }
        self._write_glb(path, document, bytes(binary))

    def _write_glb(self, path, document, bin_chunk):
        json_payload = json.dumps(document, separators=(",", ":")).encode("utf-8")
        json_payload += b" " * ((4 - (len(json_payload) % 4)) % 4)
        bin_chunk += b"\x00" * ((4 - (len(bin_chunk) % 4)) % 4)
        total = 12 + 8 + len(json_payload) + 8 + len(bin_chunk)
        path.write_bytes(
            b"glTF"
            + struct.pack("<II", 2, total)
            + struct.pack("<II", len(json_payload), JSON_CHUNK)
            + json_payload
            + struct.pack("<II", len(bin_chunk), BIN_CHUNK)
            + bin_chunk
        )

    def assertTupleAlmostEqual(self, actual, expected, places=6):
        self.assertEqual(len(actual), len(expected))
        for left, right in zip(actual, expected):
            self.assertAlmostEqual(left, right, places=places)


if __name__ == "__main__":
    unittest.main()
