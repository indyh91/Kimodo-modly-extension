import os
import contextlib
import json
import shutil
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import generator  # noqa: E402
from generator import ANIMATE_RIGGED_MESH_NODE_ID, KimodoGenerator, KimodoRuntimeError, normalize_request  # noqa: E402
from retarget_errors import RetargetError, legacy_error_aliases  # noqa: E402
from validation.animated_glb_validator import validate_bundle  # noqa: E402


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
            validation_report = validate_bundle(animated.parent)

        self.assertEqual(metadata["retarget_status"], "success")
        self.assertEqual(metadata["visual_quality_status"], "not_evaluated")
        self.assertIn("visual_quality_warning", metadata["warnings"])
        self.assertIn("visual_quality/foot_contact_not_evaluated", metadata["warnings"])
        self.assertEqual(metadata["diagnostics"]["visual_quality"]["version"], "kimodo.visual_quality.v1")
        self.assertFalse(metadata["diagnostics"]["visual_quality"]["blocking"])
        self.assertIsNone(metadata["retarget_error_code"])
        self.assertIsNone(metadata["retarget_error_message"])
        self.assertEqual(metadata["retarget_error_aliases"], [])
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
        self.assertTrue(validation_report["passed"], validation_report)

    def test_generate_animate_rigged_mesh_reports_humanoid_contract_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            outputs = Path(tmp) / "outputs"
            rig = Path(tmp) / "contract_avatar.gltf"
            self._write_contract_rig_and_sidecar(rig)
            instance = KimodoGenerator(Path(tmp) / "model", outputs)
            instance._loaded = True

            with self._patched_runtime(run_side_effect=self._write_full_contract_motion_bundle):
                animated = instance.generate(b"", {**self.base_params(), "rigged_mesh_path": str(rig), "source_kind": "unirig_recommended"})

            metadata = json.loads((animated.parent / "metadata.json").read_text(encoding="utf-8"))

        self.assertEqual(metadata["retarget_status"], "success")
        self.assertEqual(metadata["humanoid_contract"]["schema"], "modly.humanoid.v1")
        self.assertEqual(Path(metadata["humanoid_contract"]["path"]).name, "contract_avatar.rigmeta.json")
        self.assertIn("humanoid_contract_calibrated_mapping", metadata["warnings"])
        self.assertEqual(metadata["mapping_confidence"], "calibrated")
        self.assertEqual(metadata["retarget_calibration"]["chain_coverage"]["torso"], "5/5")
        self.assertEqual(metadata["diagnostics"]["humanoid_contract"]["diagnostics"], ["humanoid_contract_validated"])

    def test_retarget_motion_bundle_enriches_visual_quality_after_export_without_blocking_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp)
            metadata_path = bundle / "metadata.json"
            metadata_path.write_text(json.dumps({"canonical_motion_artifact": "motion.npz"}), encoding="utf-8")
            (bundle / "motion.bvh").write_text((FIXTURES / "synthetic_walk.bvh").read_text(encoding="utf-8"), encoding="utf-8")
            np.savez(bundle / "motion.npz", foot_contacts=np.array([[1.0, 0.0], [1.0, 0.0]], dtype=float))
            rig = FIXTURES / "tiny_rigged.gltf"
            request = {"rigged_mesh_path": rig, "rigged_mesh_source_kind": "load_mesh_existing"}
            completed = {"output_stem_name": "motion", "num_samples": 1}
            calls = []

            def fake_extract(path, humanoid_roles, *, expected_frame_count=None):
                calls.append(("extract", Path(path).name, expected_frame_count, len(tuple(humanoid_roles))))
                return type(
                    "Result",
                    (),
                    {
                        "status": "success",
                        "samples": {
                            "left": ((0.0, 0.0, 0.0), (0.08, 0.0, 0.0)),
                            "right": ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                        },
                        "reason": None,
                        "details": {"frame_count": 2},
                    },
                )()

            with mock.patch("generator.extract_world_foot_samples", side_effect=fake_extract):
                animated = generator.retarget_motion_bundle(bundle, metadata_path, request, completed)

            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        self.assertEqual(animated.name, "animated.glb")
        self.assertEqual(metadata["retarget_status"], "success")
        self.assertEqual(metadata["visual_quality_status"], "warning")
        self.assertIn("visual_quality/foot_contact_slide", metadata["warnings"])
        foot_contact = metadata["diagnostics"]["visual_quality"]["checks"]["foot_contact"]
        self.assertEqual(foot_contact["confidence"], "measured")
        self.assertEqual(foot_contact["measurement_space"], "world")
        self.assertEqual(foot_contact["measurement_target"], "node_origin")
        self.assertEqual(calls[0][0], "extract")
        self.assertEqual(calls[0][1], "animated.glb")
        self.assertEqual(calls[0][2], 2)

    def test_generate_animate_rigged_mesh_reports_untrusted_contract_without_blocking_runtime_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            outputs = Path(tmp) / "outputs"
            rig = Path(tmp) / "contract_avatar.gltf"
            shutil.copyfile(FIXTURES / "tiny_rigged.gltf", rig)
            rig.with_suffix(".rigmeta.json").write_text(
                json.dumps({"humanoid_contract": {"schema": "wrong.schema"}}),
                encoding="utf-8",
            )
            instance = KimodoGenerator(Path(tmp) / "model", outputs)
            instance._loaded = True

            with self._patched_runtime(run_side_effect=self._write_motion_bundle):
                animated = instance.generate(b"", {**self.base_params(), "rigged_mesh_path": str(rig)})

            metadata = json.loads((animated.parent / "metadata.json").read_text(encoding="utf-8"))

        self.assertEqual(metadata["retarget_status"], "success")
        self.assertEqual(metadata["runtime_status"], "success")
        self.assertEqual(metadata["contract_status"], "contract_missing_or_untrusted")
        self.assertEqual(metadata["stabilization_status"], "not_evaluated")
        self.assertIn("humanoid_contract", metadata["diagnostics"])
        self.assertFalse(metadata["diagnostics"]["humanoid_contract"]["trusted"])
        self.assertIn("humanoid_contract_schema_unsupported:wrong.schema", metadata["diagnostics"]["humanoid_contract"]["diagnostics"])

    def test_generate_animate_rigged_mesh_degrades_untrusted_contract_matrix_without_blocking_runtime_success(self):
        cases = [
            ("missing_sidecar", None, None),
            ("malformed_json", "{not json", "humanoid_contract_invalid_json"),
            ("provenance_only", {"humanoid_provenance": {"source": "legacy-export"}}, "humanoid_contract_missing:provenance_only"),
            ("auto_metadata", {"metadata_mode": "auto", "humanoid_provenance": {"source": "auto"}}, "humanoid_contract_missing:metadata_mode_auto"),
            ("legacy_fallback", {"legacy_humanoid_mapping": {"hips": "Hips"}}, "humanoid_contract_missing:legacy_mapping_only"),
            ("bone_profile", {"humanoid_profile": {"roles": {"hips": "bone_0", "left_foot": "bone_12"}}}, "humanoid_contract_missing:bone_profile_only"),
        ]
        for case_name, sidecar_payload, expected_diagnostic in cases:
            with self.subTest(case_name=case_name), tempfile.TemporaryDirectory() as tmp:
                outputs = Path(tmp) / "outputs"
                rig = Path(tmp) / f"{case_name}.gltf"
                shutil.copyfile(FIXTURES / "tiny_rigged.gltf", rig)
                if isinstance(sidecar_payload, str):
                    rig.with_suffix(".rigmeta.json").write_text(sidecar_payload, encoding="utf-8")
                elif sidecar_payload is not None:
                    rig.with_suffix(".rigmeta.json").write_text(json.dumps(sidecar_payload), encoding="utf-8")
                instance = KimodoGenerator(Path(tmp) / "model", outputs)
                instance._loaded = True

                with self._patched_runtime(run_side_effect=self._write_motion_bundle):
                    animated = instance.generate(b"", {**self.base_params(), "rigged_mesh_path": str(rig)})

                metadata = json.loads((animated.parent / "metadata.json").read_text(encoding="utf-8"))

            self.assertEqual(metadata["retarget_status"], "success")
            self.assertEqual(metadata["runtime_status"], "success")
            self.assertEqual(metadata["contract_status"], "contract_missing_or_untrusted")
            self.assertEqual(metadata["stabilization_status"], "not_evaluated")
            if expected_diagnostic is None:
                self.assertNotIn("humanoid_contract", metadata["diagnostics"])
            else:
                contract = metadata["diagnostics"]["humanoid_contract"]
                self.assertFalse(contract["trusted"])
                self.assertIn(expected_diagnostic, contract["diagnostics"])

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

    def test_generate_animate_rigged_mesh_returns_preview_fallback_for_mapping_incompatible(self):
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
                fallback = instance.generate(b"", {**self.base_params(), "rigged_mesh_path": str(incompatible_rig)})

            bundle = next(outputs.iterdir())
            metadata = json.loads((bundle / "metadata.json").read_text(encoding="utf-8"))
            preview_exists = (bundle / "preview.glb").exists()
            animated_exists = (bundle / "animated.glb").exists()

        self.assertEqual(fallback.name, "preview.glb")
        self.assertEqual(metadata["retarget_status"], "failed")
        self.assertEqual(metadata["runtime_status"], "success_with_fallback")
        self.assertEqual(metadata["retarget_error_code"], "MAPPING_INCOMPATIBLE")
        self.assertEqual(metadata["retarget_error_aliases"], [])
        self.assertEqual(metadata["stabilization_status"], "not_evaluated")
        self.assertEqual(metadata["fallback_output"], "preview.glb")
        self.assertEqual(metadata["fallback_reason"], "retarget_mapping_incompatible")
        self.assertEqual(metadata["diagnostics"]["rigged_mesh_path"], str(incompatible_rig))
        self.assertTrue(preview_exists)
        self.assertFalse(animated_exists)

    def test_retarget_motion_bundle_still_raises_mapping_incompatible_without_safe_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp)
            metadata_path = bundle / "metadata.json"
            metadata_path.write_text(json.dumps({"canonical_motion_artifact": "motion.npz"}), encoding="utf-8")
            (bundle / "motion.npz").write_bytes(b"npz")
            (bundle / "motion.bvh").write_text((FIXTURES / "synthetic_walk.bvh").read_text(encoding="utf-8"), encoding="utf-8")
            incompatible_rig = bundle / "tail.gltf"
            incompatible_rig.write_text(
                '{"asset":{"version":"2.0"},"nodes":[{"name":"Tail"}],"skins":[{"joints":[0]}]}',
                encoding="utf-8",
            )
            request = {"rigged_mesh_path": incompatible_rig, "rigged_mesh_source_kind": "unknown_mesh_source", "prompt": "Walk forward"}
            completed = {"output_stem_name": "motion", "num_samples": 1}

            with self.assertRaises(KimodoRuntimeError) as ctx:
                generator.retarget_motion_bundle(bundle, metadata_path, request, completed)

            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            animated_exists = (bundle / "animated.glb").exists()

        self.assertEqual(ctx.exception.code, "MAPPING_INCOMPATIBLE")
        self.assertEqual(metadata["retarget_status"], "failed")
        self.assertEqual(metadata["retarget_error_code"], "MAPPING_INCOMPATIBLE")
        self.assertNotIn("runtime_status", metadata)
        self.assertNotIn("fallback_output", metadata)
        self.assertFalse(animated_exists)

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

    def _write_full_contract_motion_bundle(self, bundle_dir, request, checkpoint_root, progress_cb=None):
        del request, checkpoint_root, progress_cb
        (bundle_dir / "motion.npz").write_bytes(b"npz")
        hierarchy = """HIERARCHY
ROOT Hips
{
  OFFSET 0.0 0.0 0.0
  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
  JOINT Spine
  {
    OFFSET 0.0 1.0 0.0
    CHANNELS 3 Zrotation Xrotation Yrotation
    JOINT Chest
    {
      OFFSET 0.0 1.0 0.0
      CHANNELS 3 Zrotation Xrotation Yrotation
      JOINT Neck
      {
        OFFSET 0.0 1.0 0.0
        CHANNELS 3 Zrotation Xrotation Yrotation
        JOINT Head
        {
          OFFSET 0.0 0.5 0.0
          CHANNELS 3 Zrotation Xrotation Yrotation
        }
      }
      JOINT LeftUpperArm
      {
        OFFSET -1.0 0.5 0.0
        CHANNELS 3 Zrotation Xrotation Yrotation
        JOINT LeftForeArm
        {
          OFFSET 0.0 1.0 0.0
          CHANNELS 3 Zrotation Xrotation Yrotation
          JOINT LeftHand
          {
            OFFSET 0.0 0.5 0.0
            CHANNELS 3 Zrotation Xrotation Yrotation
          }
        }
      }
      JOINT RightUpperArm
      {
        OFFSET 1.0 0.5 0.0
        CHANNELS 3 Zrotation Xrotation Yrotation
        JOINT RightForeArm
        {
          OFFSET 0.0 1.0 0.0
          CHANNELS 3 Zrotation Xrotation Yrotation
          JOINT RightHand
          {
            OFFSET 0.0 0.5 0.0
            CHANNELS 3 Zrotation Xrotation Yrotation
          }
        }
      }
    }
  }
  JOINT LeftUpperLeg
  {
    OFFSET -0.5 -1.0 0.0
    CHANNELS 3 Zrotation Xrotation Yrotation
    JOINT LeftLowerLeg
    {
      OFFSET 0.0 -1.0 0.0
      CHANNELS 3 Zrotation Xrotation Yrotation
      JOINT LeftFoot
      {
        OFFSET 0.0 -0.5 0.5
        CHANNELS 3 Zrotation Xrotation Yrotation
      }
    }
  }
  JOINT RightUpperLeg
  {
    OFFSET 0.5 -1.0 0.0
    CHANNELS 3 Zrotation Xrotation Yrotation
    JOINT RightLowerLeg
    {
      OFFSET 0.0 -1.0 0.0
      CHANNELS 3 Zrotation Xrotation Yrotation
      JOINT RightFoot
      {
        OFFSET 0.0 -0.5 0.5
        CHANNELS 3 Zrotation Xrotation Yrotation
      }
    }
  }
}
MOTION
Frames: 2
Frame Time: 0.0333333
"""
        first_frame = [0.0] * 54
        second_frame = [0.0] * 54
        second_frame[0] = 1.0
        second_frame[19] = 15.0
        (bundle_dir / "motion.bvh").write_text(
            hierarchy + " ".join(str(value) for value in first_frame) + "\n" + " ".join(str(value) for value in second_frame) + "\n",
            encoding="utf-8",
        )
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

    def _write_contract_rig_and_sidecar(self, path, *, schema="modly.humanoid.v1"):
        roles = [
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
        ]
        nodes = [{"name": f"contract_{role}"} for role in roles]
        path.write_text(
            json.dumps({
                "asset": {"version": "2.0"},
                "nodes": nodes,
                "skins": [{"joints": list(range(len(roles))), "inverseBindMatrices": 0}],
            }),
            encoding="utf-8",
        )
        chains = {
            "torso": ["hips", "spine", "chest", "neck", "head"],
            "left_arm": ["left_upper_arm", "left_forearm", "left_hand"],
            "right_arm": ["right_upper_arm", "right_forearm", "right_hand"],
            "left_leg": ["left_upper_leg", "left_lower_leg", "left_foot"],
            "right_leg": ["right_upper_leg", "right_lower_leg", "right_foot"],
        }
        sidecar = {
            "humanoid_contract": {
                "schema": schema,
                "roles": {role: {"node_index": index, "node_name": f"contract_{role}"} for index, role in enumerate(roles)},
                "chains": chains,
                "basis_hints": {"up_axis": "Y"},
            }
        }
        path.with_suffix(".rigmeta.json").write_text(json.dumps(sidecar), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
