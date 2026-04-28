import json
import struct
import tempfile
import unittest
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from glb_animation_exporter import GlbAnimationExporter, inspect_exporter_dependencies  # noqa: E402
from kimodo_bvh import locate_bvh, parse_bvh  # noqa: E402
from retarget_errors import EXPORT_FAILED, MAPPING_INCOMPATIBLE, MISSING_BVH, RetargetError  # noqa: E402
from retargeting import build_joint_mapping, build_retarget_clip  # noqa: E402
from rig_inspector import inspect_rig  # noqa: E402


FIXTURES = ROOT / "tests" / "fixtures"


class RetargetingFoundationTests(unittest.TestCase):
    def test_parse_synthetic_bvh(self):
        motion = parse_bvh(FIXTURES / "synthetic_walk.bvh")

        self.assertEqual(motion.root, "Hips")
        self.assertEqual(len(motion.frames), 2)
        self.assertAlmostEqual(motion.frame_time, 0.0333333)
        self.assertIn(("Hips", "Xposition"), motion.channel_order)

    def test_locate_bvh_reports_missing_bvh(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RetargetError) as ctx:
                locate_bvh(Path(tmp))

        self.assertEqual(ctx.exception.code, MISSING_BVH)

    def test_inspect_tiny_rigged_gltf(self):
        rig = inspect_rig(FIXTURES / "tiny_rigged.gltf", source_kind="load_mesh_existing")

        self.assertEqual(rig.classification, "humanoid")
        self.assertEqual(rig.source_kind, "load_mesh_existing")
        self.assertTrue(rig.has_inverse_bind_matrices)
        self.assertIn("Hips", rig.joint_names)

    def test_mapping_and_clip_seam(self):
        motion = parse_bvh(FIXTURES / "synthetic_walk.bvh")
        rig = inspect_rig(FIXTURES / "tiny_rigged.gltf")

        mapping = build_joint_mapping(motion, rig)
        clip = build_retarget_clip(motion, rig)

        self.assertEqual(mapping.confidence, "compatible")
        self.assertEqual(clip.name, "Kimodo Motion")
        self.assertEqual(clip.root_translations[-1], (1.0, 0.0, 0.0))
        self.assertEqual(clip.channels[0].target_joint, "Hips")
        self.assertEqual(clip.channels[0].path, "translation")
        self.assertEqual(clip.channels[0].values[-1], (1.0, 0.0, 0.0))

    def test_mapping_incompatible_without_required_joints(self):
        motion = parse_bvh(FIXTURES / "synthetic_walk.bvh")
        with tempfile.TemporaryDirectory() as tmp:
            rig_path = Path(tmp) / "bad.gltf"
            rig_path.write_text('{"asset":{"version":"2.0"},"nodes":[{"name":"Tail"}],"skins":[{"joints":[0]}]}', encoding="utf-8")
            rig = inspect_rig(rig_path)

        with self.assertRaises(RetargetError) as ctx:
            build_joint_mapping(motion, rig)
        self.assertEqual(ctx.exception.code, MAPPING_INCOMPATIBLE)

    def test_exporter_writes_animation_clip_to_gltf(self):
        exporter = GlbAnimationExporter()
        motion = parse_bvh(FIXTURES / "synthetic_walk.bvh")
        rig = inspect_rig(FIXTURES / "tiny_rigged.gltf")
        clip = build_retarget_clip(motion, rig)

        with tempfile.TemporaryDirectory() as tmp:
            output = exporter.write_animation(rig, clip, Path(tmp) / "animated.gltf")
            document = json.loads(output.read_text(encoding="utf-8"))

        animation = document["animations"][-1]
        self.assertEqual(animation["name"], "Kimodo Motion")
        self.assertEqual(len(animation["samplers"]), 1)
        self.assertEqual(len(animation["channels"]), 1)
        self.assertEqual(animation["channels"][0]["target"], {"node": 0, "path": "translation"})
        sampler = animation["samplers"][0]
        self.assertEqual(document["accessors"][sampler["input"]]["type"], "SCALAR")
        self.assertEqual(document["accessors"][sampler["output"]]["type"], "VEC3")

    def test_exporter_writes_animation_clip_to_glb_json_chunk(self):
        exporter = GlbAnimationExporter()
        motion = parse_bvh(FIXTURES / "synthetic_walk.bvh")
        rig = inspect_rig(FIXTURES / "tiny_rigged.gltf")
        clip = build_retarget_clip(motion, rig)

        with tempfile.TemporaryDirectory() as tmp:
            output = exporter.write_animation(rig, clip, Path(tmp) / "animated.glb")
            data = output.read_bytes()

        self.assertEqual(data[:4], b"glTF")
        version, total_length = struct.unpack_from("<II", data, 4)
        chunk_length, chunk_type = struct.unpack_from("<II", data, 12)
        self.assertEqual(version, 2)
        self.assertEqual(total_length, len(data))
        self.assertEqual(chunk_type, 0x4E4F534A)
        document = json.loads(data[20:20 + chunk_length].rstrip(b" \t\r\n\x00").decode("utf-8"))
        self.assertEqual(document["animations"][-1]["name"], "Kimodo Motion")
        self.assertEqual(document["animations"][-1]["channels"][0]["target"]["path"], "translation")

    def test_exporter_fails_closed_without_clip_channels(self):
        exporter = GlbAnimationExporter()
        motion = parse_bvh(FIXTURES / "synthetic_walk.bvh")
        rig = inspect_rig(FIXTURES / "tiny_rigged.gltf")
        clip = build_retarget_clip(motion, rig)
        empty_clip = type(clip)(
            name=clip.name,
            duration=clip.duration,
            frame_time=clip.frame_time,
            root_translations=clip.root_translations,
            channels=(),
            mapping=clip.mapping,
            warnings=clip.warnings,
        )

        with self.assertRaises(RetargetError) as ctx:
            exporter.write_animation(rig, empty_clip, Path("out.glb"))
        self.assertEqual(ctx.exception.code, EXPORT_FAILED)

    def test_dependency_report_remains_diagnostic_only(self):
        report = inspect_exporter_dependencies()
        self.assertIsInstance(report.trimesh_available, bool)


if __name__ == "__main__":
    unittest.main()
