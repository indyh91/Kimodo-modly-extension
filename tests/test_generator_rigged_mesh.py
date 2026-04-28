import os
import contextlib
import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import generator  # noqa: E402
from generator import ANIMATE_RIGGED_MESH_NODE_ID, KimodoGenerator, KimodoRuntimeError, normalize_request  # noqa: E402
from retarget_errors import RetargetError, legacy_error_aliases  # noqa: E402


FIXTURES = ROOT / "tests" / "fixtures"


class GeneratorRiggedMeshTests(unittest.TestCase):
    def base_params(self):
        return {
            "prompt": "Walk forward",
            "node_id": ANIMATE_RIGGED_MESH_NODE_ID,
        }

    def test_preview_branch_does_not_require_mesh(self):
        request = normalize_request({"prompt": "Walk forward"})

        self.assertEqual(request["node_id"], "text-to-motion-preview")
        self.assertIsNone(request["rigged_mesh_path"])

    def test_animate_branch_requires_rigged_mesh_path(self):
        with self.assertRaisesRegex(KimodoRuntimeError, "rigged_mesh_path is required"):
            normalize_request(self.base_params())

    def test_animate_branch_accepts_mesh_path_alias(self):
        with tempfile.TemporaryDirectory() as tmp:
            mesh = Path(tmp) / "avatar.glb"
            mesh.write_bytes(b"glb")

            request = normalize_request({**self.base_params(), "mesh_path": str(mesh)})

        self.assertEqual(request["rigged_mesh_path"], mesh.resolve())

    def test_animate_branch_rejects_invalid_suffix(self):
        with tempfile.TemporaryDirectory() as tmp:
            mesh = Path(tmp) / "avatar.obj"
            mesh.write_text("obj", encoding="utf-8")

            with self.assertRaisesRegex(KimodoRuntimeError, "supported mesh file"):
                normalize_request({**self.base_params(), "rigged_mesh_path": str(mesh)})

    def test_animate_branch_resolves_workspace_relative_mesh(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            mesh = workspace / "rigs" / "avatar.gltf"
            mesh.parent.mkdir()
            mesh.write_text("{}", encoding="utf-8")
            old_workspace = os.environ.get("WORKSPACE_DIR")
            os.environ["WORKSPACE_DIR"] = str(workspace)
            try:
                request = normalize_request({**self.base_params(), "rigged_mesh_path": "rigs/avatar.gltf"})
            finally:
                if old_workspace is None:
                    os.environ.pop("WORKSPACE_DIR", None)
                else:
                    os.environ["WORKSPACE_DIR"] = old_workspace

        self.assertEqual(request["rigged_mesh_path"], mesh.resolve())

    def test_animate_branch_forces_bvh_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            mesh = Path(tmp) / "avatar.glb"
            mesh.write_bytes(b"glb")

            request = normalize_request({**self.base_params(), "rigged_mesh_path": str(mesh), "export_bvh": False})
            command = generator.kimodo_command(Path(tmp) / "motion", request)

        self.assertTrue(request["export_bvh"])
        self.assertIn("--bvh", command)

    def test_preview_command_is_not_forced_to_bvh(self):
        request = normalize_request({"prompt": "Walk forward", "export_bvh": False})
        command = generator.kimodo_command(Path("motion"), request)

        self.assertNotIn("--bvh", command)

    def test_generate_animate_rigged_mesh_exports_animated_glb_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            outputs = Path(tmp) / "outputs"
            rig = FIXTURES / "tiny_rigged.gltf"
            instance = KimodoGenerator(Path(tmp) / "model", outputs)
            instance._loaded = True

            with self._patched_runtime(run_side_effect=self._write_motion_bundle):
                animated = instance.generate(b"", {**self.base_params(), "rigged_mesh_path": str(rig), "source_kind": "load_mesh_existing"})

            metadata = json.loads((animated.parent / "metadata.json").read_text(encoding="utf-8"))
            document = self._read_glb_document(animated)

        self.assertEqual(metadata["retarget_status"], "success")
        self.assertEqual(metadata["source_rigged_mesh"], str(rig))
        self.assertEqual(metadata["source_kind"], "load_mesh_existing")
        self.assertEqual(metadata["diagnostics"]["rigged_mesh_path"], str(rig))
        self.assertEqual(metadata["clip_name"], "Kimodo Motion")
        self.assertEqual(metadata["mapping_confidence"], "compatible")
        self.assertIn("axis_assumption_bvh_xyz_degrees", metadata["warnings"])
        self.assertIn("root_motion_mode_preserve_translation", metadata["warnings"])
        self.assertEqual(metadata["canonical_motion_artifact"], "motion.npz")
        self.assertEqual(Path(metadata["diagnostics"]["bvh_path"]).name, "motion.bvh")
        self.assertEqual(
            set(metadata["artifacts"]),
            {"animated.glb", "metadata.json", "motion.bvh", "motion.npz", "preview.glb"},
        )
        self.assertEqual(document["animations"][-1]["name"], "Kimodo Motion")
        self.assertEqual(document["animations"][-1]["channels"][0]["target"]["path"], "translation")

    def test_generate_animate_rigged_mesh_preserves_unknown_best_effort_warnings(self):
        with tempfile.TemporaryDirectory() as tmp:
            outputs = Path(tmp) / "outputs"
            rig = Path(tmp) / "unknown_but_mappable.gltf"
            rig.write_text(
                json.dumps({
                    "asset": {"version": "2.0"},
                    "nodes": [{"name": "Hips"}, {"name": "Spine"}, {"name": "Head"}],
                    "skins": [{"joints": [0, 1, 2], "inverseBindMatrices": 0}],
                }),
                encoding="utf-8",
            )
            instance = KimodoGenerator(Path(tmp) / "model", outputs)
            instance._loaded = True

            with self._patched_runtime(run_side_effect=self._write_motion_bundle):
                animated = instance.generate(b"", {**self.base_params(), "rigged_mesh_path": str(rig), "source_kind": "unknown_mesh_source"})

            metadata = json.loads((animated.parent / "metadata.json").read_text(encoding="utf-8"))

        self.assertEqual(metadata["retarget_status"], "success")
        self.assertEqual(metadata["classification"], "unknown")
        self.assertEqual(metadata["mapping_confidence"], "best_effort")
        self.assertIn("classification_unknown_or_non_humanoid", metadata["warnings"])
        self.assertIn("missing_left_arm_mapping", metadata["warnings"])
        self.assertIn("missing_right_leg_mapping", metadata["warnings"])

    def test_generate_animate_rigged_mesh_reports_missing_bvh(self):
        with tempfile.TemporaryDirectory() as tmp:
            outputs = Path(tmp) / "outputs"
            rig = FIXTURES / "tiny_rigged.gltf"
            instance = KimodoGenerator(Path(tmp) / "model", outputs)
            instance._loaded = True

            with self._patched_runtime(run_side_effect=self._write_npz_only):
                with self.assertRaises(KimodoRuntimeError) as ctx:
                    instance.generate(b"", {**self.base_params(), "rigged_mesh_path": str(rig)})

            bundle = next(outputs.iterdir())
            metadata = json.loads((bundle / "metadata.json").read_text(encoding="utf-8"))

        self.assertEqual(ctx.exception.code, "MISSING_BVH")
        self.assertEqual(metadata["retarget_status"], "failed")
        self.assertEqual(metadata["retarget_error_code"], "MISSING_BVH")
        self.assertEqual(metadata["retarget_error_aliases"], ["BVH_UNAVAILABLE"])
        self.assertFalse((bundle / "animated.glb").exists())

    def test_generate_animate_rigged_mesh_reports_unparseable_bvh_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            outputs = Path(tmp) / "outputs"
            rig = FIXTURES / "tiny_rigged.gltf"
            instance = KimodoGenerator(Path(tmp) / "model", outputs)
            instance._loaded = True

            with self._patched_runtime(run_side_effect=self._write_invalid_bvh_bundle):
                with self.assertRaises(KimodoRuntimeError) as ctx:
                    instance.generate(b"", {**self.base_params(), "rigged_mesh_path": str(rig)})

            bundle = next(outputs.iterdir())
            metadata = json.loads((bundle / "metadata.json").read_text(encoding="utf-8"))

        self.assertEqual(ctx.exception.code, "UNPARSEABLE_BVH")
        self.assertEqual(metadata["retarget_status"], "failed")
        self.assertEqual(metadata["retarget_error_code"], "UNPARSEABLE_BVH")
        self.assertEqual(metadata["retarget_error_aliases"], ["BVH_UNPARSEABLE"])
        self.assertEqual(Path(metadata["diagnostics"]["bvh_path"]).name, "motion.bvh")
        self.assertFalse((bundle / "animated.glb").exists())

    def test_generate_animate_rigged_mesh_reports_unparseable_or_unskinned_rig(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad_rig = Path(tmp) / "bad.gltf"
            bad_rig.write_text('{"asset":{"version":"2.0"},"nodes":[{"name":"Hips"}]}', encoding="utf-8")
            outputs = Path(tmp) / "outputs"
            instance = KimodoGenerator(Path(tmp) / "model", outputs)
            instance._loaded = True

            with self._patched_runtime(run_side_effect=self._write_motion_bundle):
                with self.assertRaises(KimodoRuntimeError) as ctx:
                    instance.generate(b"", {**self.base_params(), "rigged_mesh_path": str(bad_rig)})

            bundle = next(outputs.iterdir())
            metadata = json.loads((bundle / "metadata.json").read_text(encoding="utf-8"))

        self.assertEqual(ctx.exception.code, "RIG_HAS_NO_SKIN_OR_SKELETON")
        self.assertEqual(metadata["retarget_status"], "failed")
        self.assertEqual(metadata["retarget_error_code"], "RIG_HAS_NO_SKIN_OR_SKELETON")
        self.assertEqual(metadata["retarget_error_aliases"], ["RIG_INVALID"])
        self.assertFalse((bundle / "animated.glb").exists())

    def test_generate_animate_rigged_mesh_reports_unparseable_rig_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad_rig = Path(tmp) / "bad.gltf"
            bad_rig.write_text("not json", encoding="utf-8")
            outputs = Path(tmp) / "outputs"
            instance = KimodoGenerator(Path(tmp) / "model", outputs)
            instance._loaded = True

            with self._patched_runtime(run_side_effect=self._write_motion_bundle):
                with self.assertRaises(KimodoRuntimeError) as ctx:
                    instance.generate(b"", {**self.base_params(), "rigged_mesh_path": str(bad_rig)})

            bundle = next(outputs.iterdir())
            metadata = json.loads((bundle / "metadata.json").read_text(encoding="utf-8"))

        self.assertEqual(ctx.exception.code, "UNPARSEABLE_RIG")
        self.assertEqual(metadata["retarget_status"], "failed")
        self.assertEqual(metadata["retarget_error_code"], "UNPARSEABLE_RIG")
        self.assertEqual(metadata["retarget_error_aliases"], ["RIG_INVALID"])
        self.assertEqual(metadata["diagnostics"]["rigged_mesh_path"], str(bad_rig))
        self.assertFalse((bundle / "animated.glb").exists())

    def test_generate_animate_rigged_mesh_reports_mapping_incompatible_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            incompatible_rig = Path(tmp) / "tail.gltf"
            incompatible_rig.write_text(
                '{"asset":{"version":"2.0"},"nodes":[{"name":"Tail"}],"skins":[{"joints":[0]}]}',
                encoding="utf-8",
            )
            outputs = Path(tmp) / "outputs"
            instance = KimodoGenerator(Path(tmp) / "model", outputs)
            instance._loaded = True

            with self._patched_runtime(run_side_effect=self._write_motion_bundle):
                with self.assertRaises(KimodoRuntimeError) as ctx:
                    instance.generate(b"", {**self.base_params(), "rigged_mesh_path": str(incompatible_rig)})

            bundle = next(outputs.iterdir())
            metadata = json.loads((bundle / "metadata.json").read_text(encoding="utf-8"))

        self.assertEqual(ctx.exception.code, "MAPPING_INCOMPATIBLE")
        self.assertEqual(metadata["retarget_status"], "failed")
        self.assertEqual(metadata["retarget_error_code"], "MAPPING_INCOMPATIBLE")
        self.assertEqual(metadata["retarget_error_aliases"], [])
        self.assertEqual(metadata["diagnostics"]["rigged_mesh_path"], str(incompatible_rig))
        self.assertFalse((bundle / "animated.glb").exists())

    def test_generate_animate_rigged_mesh_reports_export_failed_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            outputs = Path(tmp) / "outputs"
            rig = FIXTURES / "tiny_rigged.gltf"
            instance = KimodoGenerator(Path(tmp) / "model", outputs)
            instance._loaded = True

            with self._patched_runtime(run_side_effect=self._write_motion_bundle):
                with mock.patch.object(
                    generator.GlbAnimationExporter,
                    "write_animation",
                    side_effect=RetargetError("EXPORT_FAILED", "forced exporter failure"),
                ):
                    with self.assertRaises(KimodoRuntimeError) as ctx:
                        instance.generate(b"", {**self.base_params(), "rigged_mesh_path": str(rig)})

            bundle = next(outputs.iterdir())
            metadata = json.loads((bundle / "metadata.json").read_text(encoding="utf-8"))

        self.assertEqual(ctx.exception.code, "EXPORT_FAILED")
        self.assertEqual(metadata["retarget_status"], "failed")
        self.assertEqual(metadata["retarget_error_code"], "EXPORT_FAILED")
        self.assertEqual(metadata["retarget_error_aliases"], [])
        self.assertEqual(metadata["diagnostics"]["mapping_confidence"], "compatible")
        self.assertFalse((bundle / "animated.glb").exists())

    def test_legacy_error_aliases_document_primary_code_mapping(self):
        self.assertEqual(legacy_error_aliases("MISSING_BVH"), ("BVH_UNAVAILABLE",))
        self.assertEqual(legacy_error_aliases("UNPARSEABLE_BVH"), ("BVH_UNPARSEABLE",))
        self.assertEqual(legacy_error_aliases("UNPARSEABLE_RIG"), ("RIG_INVALID",))
        self.assertEqual(legacy_error_aliases("RIG_HAS_NO_SKIN_OR_SKELETON"), ("RIG_INVALID",))
        self.assertEqual(legacy_error_aliases("EXPORT_FAILED"), ())

    def _patched_runtime(self, *, run_side_effect):
        stack = contextlib.ExitStack()
        stack.enter_context(mock.patch.object(generator, "discover_hf_token", return_value=True))
        stack.enter_context(mock.patch.object(generator, "resolve_checkpoint_root", return_value=Path("/tmp/checkpoints")))
        stack.enter_context(mock.patch.object(generator, "run_kimodo_cli", side_effect=run_side_effect))
        stack.enter_context(mock.patch.object(generator, "build_preview_glb", side_effect=lambda _npz, preview: preview.write_bytes(b"preview")))
        return stack

    def _write_motion_bundle(self, bundle_dir, request, checkpoint_root, progress_cb=None):
        del request, checkpoint_root, progress_cb
        (bundle_dir / "motion.npz").write_bytes(b"npz")
        (bundle_dir / "motion.bvh").write_text((FIXTURES / "synthetic_walk.bvh").read_text(encoding="utf-8"), encoding="utf-8")
        return {"output_stem_name": "motion", "num_samples": 1}

    def _write_npz_only(self, bundle_dir, request, checkpoint_root, progress_cb=None):
        del request, checkpoint_root, progress_cb
        (bundle_dir / "motion.npz").write_bytes(b"npz")
        return {"output_stem_name": "motion", "num_samples": 1}

    def _write_invalid_bvh_bundle(self, bundle_dir, request, checkpoint_root, progress_cb=None):
        del request, checkpoint_root, progress_cb
        (bundle_dir / "motion.npz").write_bytes(b"npz")
        (bundle_dir / "motion.bvh").write_text("HIERARCHY\nROOT Hips\n", encoding="utf-8")
        return {"output_stem_name": "motion", "num_samples": 1}

    def _read_glb_document(self, path):
        data = path.read_bytes()
        chunk_length, chunk_type = struct.unpack_from("<II", data, 12)
        self.assertEqual(chunk_type, 0x4E4F534A)
        return json.loads(data[20:20 + chunk_length].rstrip(b" \t\r\n\x00").decode("utf-8"))


if __name__ == "__main__":
    unittest.main()
