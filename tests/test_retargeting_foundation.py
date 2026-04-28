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
from retarget_errors import CALIBRATION_UNAVAILABLE, EXPORT_FAILED, MAPPING_INCOMPATIBLE, MISSING_BVH, RetargetError  # noqa: E402
from retargeting import JointMapping, RetargetChannel, RetargetClip, build_joint_mapping, build_retarget_clip  # noqa: E402
from rig_inspector import inspect_rig  # noqa: E402
from validation.animated_glb_validator import BufferResolver, decode_accessor  # noqa: E402


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
        rotation_channels = [channel for channel in clip.channels if channel.path == "rotation"]
        self.assertTrue(any(channel.target_joint == "LeftArm" for channel in rotation_channels))

    def test_mapping_incompatible_without_required_joints(self):
        motion = parse_bvh(FIXTURES / "synthetic_walk.bvh")
        with tempfile.TemporaryDirectory() as tmp:
            rig_path = Path(tmp) / "bad.gltf"
            rig_path.write_text('{"asset":{"version":"2.0"},"nodes":[{"name":"Tail"}],"skins":[{"joints":[0]}]}', encoding="utf-8")
            rig = inspect_rig(rig_path)

        with self.assertRaises(RetargetError) as ctx:
            build_joint_mapping(motion, rig)
        self.assertEqual(ctx.exception.code, MAPPING_INCOMPATIBLE)

    def test_mapping_supports_46_joint_anonymous_unirig_topology(self):
        motion = parse_bvh(FIXTURES / "synthetic_walk.bvh")
        with tempfile.TemporaryDirectory() as tmp:
            rig_path = Path(tmp) / "anonymous_unirig.gltf"
            self._write_anonymous_unirig_46(rig_path)
            rig = inspect_rig(rig_path)

        mapping = build_joint_mapping(motion, rig)
        clip = build_retarget_clip(motion, rig)

        self.assertEqual(mapping.confidence, "compatible")
        self.assertEqual(mapping.pairs["Hips"], "bone_0")
        self.assertEqual(mapping.pairs["Spine"], "bone_1")
        self.assertEqual(mapping.pairs["Head"], "bone_4")
        self.assertEqual(clip.channels[0].target_joint, "bone_0")
        self.assertIn("anonymous_unirig_topology_mapping", clip.warnings)
        self.assertIn("classification_unknown_or_non_humanoid", clip.warnings)

    def test_mapping_supports_52_joint_anonymous_unirig_topology(self):
        motion = parse_bvh(FIXTURES / "synthetic_walk.bvh")
        with tempfile.TemporaryDirectory() as tmp:
            rig_path = Path(tmp) / "anonymous_unirig_52.gltf"
            self._write_anonymous_unirig_52(rig_path)
            rig = inspect_rig(rig_path)

        mapping = build_joint_mapping(motion, rig)
        clip = build_retarget_clip(motion, rig)

        self.assertEqual(mapping.confidence, "compatible")
        self.assertEqual(mapping.pairs["Hips"], "bone_0")
        self.assertEqual(mapping.pairs["Spine"], "bone_1")
        self.assertEqual(mapping.pairs["Head"], "bone_4")
        self.assertEqual(mapping.pairs["LeftArm"], "bone_6")
        self.assertEqual(mapping.pairs["RightArm"], "bone_25")
        self.assertEqual(mapping.pairs["LeftUpLeg"], "bone_44")
        self.assertEqual(mapping.pairs["RightUpLeg"], "bone_48")
        self.assertEqual(clip.channels[0].target_joint, "bone_0")
        self.assertIn("anonymous_unirig_topology_mapping", clip.warnings)

    def test_root_bvh_motion_drives_mapped_anonymous_unirig_hips(self):
        with tempfile.TemporaryDirectory() as tmp:
            bvh_path = Path(tmp) / "root_with_hips.bvh"
            rig_path = Path(tmp) / "anonymous_unirig.gltf"
            self._write_root_bvh(bvh_path, include_hips=True)
            self._write_anonymous_unirig_46(rig_path)
            motion = parse_bvh(bvh_path)
            rig = inspect_rig(rig_path)

        mapping = build_joint_mapping(motion, rig)
        clip = build_retarget_clip(motion, rig)

        self.assertEqual(motion.root, "Root")
        self.assertNotIn("Root", mapping.pairs)
        self.assertEqual(mapping.pairs["Hips"], "bone_0")
        self.assertEqual(clip.channels[0].target_joint, "bone_0")
        self.assertEqual(clip.channels[0].path, "translation")
        self.assertEqual(clip.channels[0].values[-1], (2.0, 3.0, 4.0))

    def test_root_bvh_without_hips_child_still_uses_direct_root_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            bvh_path = Path(tmp) / "root_without_hips.bvh"
            self._write_root_bvh(bvh_path, include_hips=False)
            motion = parse_bvh(bvh_path)
        rig = inspect_rig(FIXTURES / "tiny_rigged.gltf")

        mapping = build_joint_mapping(motion, rig)
        clip = build_retarget_clip(motion, rig)

        self.assertEqual(mapping.pairs["Root"], "Hips")
        self.assertEqual(clip.channels[0].target_joint, "Hips")
        self.assertEqual(clip.channels[0].values[-1], (2.0, 3.0, 4.0))

    def test_non_root_mapped_joint_rotation_creates_rotation_channel(self):
        motion = parse_bvh(FIXTURES / "synthetic_walk.bvh")
        rig = inspect_rig(FIXTURES / "tiny_rigged.gltf")

        clip = build_retarget_clip(motion, rig)

        left_arm = next(channel for channel in clip.channels if channel.target_joint == "LeftArm" and channel.path == "rotation")
        self.assertEqual(left_arm.times, (0.0, motion.frame_time))
        self.assertEqual(left_arm.values[0], (0.0, 0.0, 0.0, 1.0))
        self.assertNotEqual(left_arm.values[0], left_arm.values[1])

    def test_anonymous_bone_names_without_unirig_topology_remain_incompatible(self):
        motion = parse_bvh(FIXTURES / "synthetic_walk.bvh")
        with tempfile.TemporaryDirectory() as tmp:
            rig_path = Path(tmp) / "anonymous_non_unirig.gltf"
            nodes = [{"name": f"bone_{index}"} for index in range(46)]
            nodes[0]["children"] = [1]
            rig_path.write_text(
                json.dumps({
                    "asset": {"version": "2.0"},
                    "nodes": nodes,
                    "skins": [{"joints": list(range(46)), "inverseBindMatrices": 0}],
                }),
                encoding="utf-8",
            )
            rig = inspect_rig(rig_path)

        with self.assertRaises(RetargetError) as ctx:
            build_joint_mapping(motion, rig)
        self.assertEqual(ctx.exception.code, MAPPING_INCOMPATIBLE)

    def test_humanoid_contract_sidecar_maps_before_fallback_and_reports_basis(self):
        with tempfile.TemporaryDirectory() as tmp:
            rig_path = Path(tmp) / "contract_avatar.gltf"
            bvh_path = Path(tmp) / "contract_motion.bvh"
            self._write_contract_rig(rig_path)
            self._write_contract_sidecar(rig_path, basis=True)
            self._write_full_contract_bvh(bvh_path)
            rig = inspect_rig(rig_path)
            motion = parse_bvh(bvh_path)

        mapping = build_joint_mapping(motion, rig)
        clip = build_retarget_clip(motion, rig)

        self.assertEqual(rig.humanoid_contract_schema, "modly.humanoid.v1")
        self.assertEqual(mapping.pairs["Hips"], "contract_hips")
        self.assertEqual(mapping.pairs["LeftUpperArm"], "contract_left_upper_arm")
        self.assertEqual(mapping.confidence, "calibrated")
        self.assertIn("humanoid_contract_calibrated_mapping", mapping.warnings)
        self.assertEqual(clip.calibration["version"], "kimodo.retarget_calibration.v1")
        self.assertEqual(clip.calibration["chain_coverage"]["left_arm"], "3/3")
        self.assertEqual(clip.channels[0].target_joint, "contract_hips")

    def test_calibrated_quaternion_conjugates_source_delta_into_target_basis(self):
        with tempfile.TemporaryDirectory() as tmp:
            rig_path = Path(tmp) / "contract_avatar.gltf"
            bvh_path = Path(tmp) / "contract_motion.bvh"
            self._write_contract_rig(rig_path)
            self._write_contract_sidecar(rig_path, basis=True)
            payload = json.loads(rig_path.with_suffix(".rigmeta.json").read_text(encoding="utf-8"))
            payload["humanoid_contract"]["basis_hints"] = {
                "roles": {"left_upper_arm": {"basis_quaternion": [0.0, 0.0, 0.7071067811865475, 0.7071067811865476]}},
                "up_axis": "Y",
            }
            rig_path.with_suffix(".rigmeta.json").write_text(json.dumps(payload), encoding="utf-8")
            self._write_full_contract_bvh(bvh_path, left_upper_arm_x_degrees=90.0)
            rig = inspect_rig(rig_path)
            motion = parse_bvh(bvh_path)

        clip = build_retarget_clip(motion, rig)

        channel = next(channel for channel in clip.channels if channel.target_joint == "contract_left_upper_arm" and channel.path == "rotation")
        self.assertTupleAlmostEqual(channel.values[0], (0.0, 0.0, 0.0, 1.0))
        self.assertTupleAlmostEqual(channel.values[1], (0.0, 0.7071067811865475, 0.0, 0.7071067811865476))
        self.assertEqual(channel.diagnostics["composition_order"], clip.calibration["composition_order"])

    def test_calibrated_target_basis_can_derive_from_non_identity_rest_transform(self):
        with tempfile.TemporaryDirectory() as tmp:
            rig_path = Path(tmp) / "contract_avatar.gltf"
            bvh_path = Path(tmp) / "contract_motion.bvh"
            self._write_contract_rig(rig_path)
            self._write_contract_sidecar(rig_path, basis=True)
            payload = json.loads(rig_path.with_suffix(".rigmeta.json").read_text(encoding="utf-8"))
            payload["humanoid_contract"].pop("basis_hints")
            payload["humanoid_contract"]["rest"] = {
                "roles": {
                    "left_upper_arm": {"child_offset": [1.0, 0.0, 0.0]},
                }
            }
            rig_path.with_suffix(".rigmeta.json").write_text(json.dumps(payload), encoding="utf-8")
            self._write_full_contract_bvh(bvh_path, left_upper_arm_x_degrees=90.0)
            rig = inspect_rig(rig_path)
            motion = parse_bvh(bvh_path)

        clip = build_retarget_clip(motion, rig)

        basis = clip.calibration["basis_sources"]["contract_left_upper_arm"]
        channel = next(channel for channel in clip.channels if channel.target_joint == "contract_left_upper_arm" and channel.path == "rotation")
        self.assertEqual(basis["target"], "target_rest_transform:left_upper_arm")
        self.assertNotEqual(tuple(basis["target_quaternion"]), (0.0, 0.0, 0.0, 1.0))
        self.assertTupleAlmostEqual(channel.values[0], (0.0, 0.0, 0.0, 1.0))
        self.assertTupleAlmostEqual(channel.values[1], (0.0, -0.7071067811865475, 0.0, 0.7071067811865476))

    def test_calibrated_optional_fingers_and_toes_are_warnings_not_required_gaps(self):
        with tempfile.TemporaryDirectory() as tmp:
            rig_path = Path(tmp) / "contract_avatar.gltf"
            bvh_path = Path(tmp) / "contract_motion.bvh"
            self._write_contract_rig(rig_path)
            self._write_contract_sidecar(rig_path, basis=True)
            self._write_full_contract_bvh(bvh_path)
            rig = inspect_rig(rig_path)
            motion = parse_bvh(bvh_path)

        clip = build_retarget_clip(motion, rig)

        warnings = clip.calibration["warnings_by_severity"]["warning"]
        self.assertEqual(clip.mapping.confidence, "calibrated")
        self.assertIn("optional_chain_unavailable:left_fingers", warnings)
        self.assertIn("optional_chain_unavailable:right_fingers", warnings)
        self.assertIn("optional_chain_unavailable:left_toes", warnings)
        self.assertIn("optional_chain_unavailable:right_toes", warnings)
        self.assertEqual(clip.calibration["chain_coverage"]["left_arm"], "3/3")

    def test_humanoid_contract_absent_preserves_anonymous_fallback(self):
        motion = parse_bvh(FIXTURES / "synthetic_walk.bvh")
        with tempfile.TemporaryDirectory() as tmp:
            rig_path = Path(tmp) / "anonymous_unirig.gltf"
            self._write_anonymous_unirig_46(rig_path)
            rig = inspect_rig(rig_path)

        mapping = build_joint_mapping(motion, rig)

        self.assertIsNone(rig.humanoid_contract_path)
        self.assertEqual(mapping.pairs["Hips"], "bone_0")
        self.assertIn("anonymous_unirig_topology_mapping", mapping.warnings)

    def test_legacy_rigmeta_without_humanoid_contract_preserves_anonymous_fallback(self):
        motion = parse_bvh(FIXTURES / "synthetic_walk.bvh")
        with tempfile.TemporaryDirectory() as tmp:
            rig_path = Path(tmp) / "anonymous_unirig.gltf"
            self._write_anonymous_unirig_46(rig_path)
            rig_path.with_suffix(".rigmeta.json").write_text(
                json.dumps({
                    "extension_id": "kimodo-soma-rp",
                    "metadata_version": "1.0",
                    "node_id": "unirig",
                    "output_mesh": str(rig_path),
                    "pipeline": "unirig",
                    "runtime": "legacy",
                }),
                encoding="utf-8",
            )
            rig = inspect_rig(rig_path)

        mapping = build_joint_mapping(motion, rig)

        self.assertIsNone(rig.humanoid_contract_path)
        self.assertEqual(mapping.pairs["Hips"], "bone_0")
        self.assertIn("anonymous_unirig_topology_mapping", mapping.warnings)

    def test_malformed_present_humanoid_contract_still_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            rig_path = Path(tmp) / "contract_avatar.gltf"
            self._write_contract_rig(rig_path)
            rig_path.with_suffix(".rigmeta.json").write_text(
                json.dumps({"humanoid_contract": "legacy string is not a contract"}),
                encoding="utf-8",
            )

            with self.assertRaises(RetargetError) as ctx:
                inspect_rig(rig_path)

        self.assertEqual(ctx.exception.code, MAPPING_INCOMPATIBLE)
        self.assertIn("humanoid_contract_invalid_shape:humanoid_contract", ctx.exception.message)

    def test_humanoid_contract_wrong_schema_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            rig_path = Path(tmp) / "contract_avatar.gltf"
            self._write_contract_rig(rig_path)
            self._write_contract_sidecar(rig_path, schema="wrong.schema")

            with self.assertRaises(RetargetError) as ctx:
                inspect_rig(rig_path)

        self.assertEqual(ctx.exception.code, MAPPING_INCOMPATIBLE)
        self.assertIn("humanoid_contract_schema_unsupported:wrong.schema", ctx.exception.message)

    def test_humanoid_contract_invalid_json_fails_closed_with_stable_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp:
            rig_path = Path(tmp) / "contract_avatar.gltf"
            self._write_contract_rig(rig_path)
            rig_path.with_suffix(".rigmeta.json").write_text("{not json", encoding="utf-8")

            with self.assertRaises(RetargetError) as ctx:
                inspect_rig(rig_path)

        self.assertEqual(ctx.exception.code, MAPPING_INCOMPATIBLE)
        diagnostics = ctx.exception.diagnostics["humanoid_contract"]["diagnostics"]
        self.assertEqual(diagnostics, ["humanoid_contract_invalid_json"])
        self.assertIn("humanoid_contract_invalid_json", ctx.exception.message)

    def test_humanoid_contract_invalid_root_roles_and_chains_shapes_fail_closed(self):
        cases = (
            ([], "humanoid_contract_invalid_shape:root"),
            ({"humanoid_contract": {"schema": "modly.humanoid.v1", "roles": "bad", "chains": {}}}, "humanoid_contract_invalid_shape:roles"),
            ({"humanoid_contract": {"schema": "modly.humanoid.v1", "roles": {}, "chains": "bad"}}, "humanoid_contract_invalid_shape:chains"),
        )
        for payload, expected in cases:
            with self.subTest(expected=expected):
                with tempfile.TemporaryDirectory() as tmp:
                    rig_path = Path(tmp) / "contract_avatar.gltf"
                    self._write_contract_rig(rig_path)
                    rig_path.with_suffix(".rigmeta.json").write_text(json.dumps(payload), encoding="utf-8")

                    with self.assertRaises(RetargetError) as ctx:
                        inspect_rig(rig_path)

                self.assertEqual(ctx.exception.code, MAPPING_INCOMPATIBLE)
                diagnostics = ctx.exception.diagnostics["humanoid_contract"]["diagnostics"]
                self.assertEqual(diagnostics[0], expected)
                self.assertEqual(len(diagnostics), len(dict.fromkeys(diagnostics)))

    def test_humanoid_contract_bad_role_reference_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            rig_path = Path(tmp) / "contract_avatar.gltf"
            self._write_contract_rig(rig_path)
            self._write_contract_sidecar(rig_path, overrides={"head": {"node_index": 4, "node_name": "not_head"}})

            with self.assertRaises(RetargetError) as ctx:
                inspect_rig(rig_path)

        self.assertEqual(ctx.exception.code, MAPPING_INCOMPATIBLE)
        self.assertIn("humanoid_contract_role_node_name_mismatch:head", ctx.exception.message)

    def test_humanoid_contract_out_of_range_node_reference_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            rig_path = Path(tmp) / "contract_avatar.gltf"
            self._write_contract_rig(rig_path)
            self._write_contract_sidecar(rig_path, overrides={"head": {"node_index": 999, "node_name": "missing"}})

            with self.assertRaises(RetargetError) as ctx:
                inspect_rig(rig_path)

        self.assertEqual(ctx.exception.code, MAPPING_INCOMPATIBLE)
        diagnostics = ctx.exception.diagnostics["humanoid_contract"]["diagnostics"]
        self.assertIn("humanoid_contract_role_node_index_out_of_range:head:999", diagnostics)

    def test_humanoid_contract_non_skin_joint_reference_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            rig_path = Path(tmp) / "contract_avatar.gltf"
            self._write_contract_rig(rig_path)
            document = json.loads(rig_path.read_text(encoding="utf-8"))
            document["nodes"].append({"name": "contract_extra"})
            rig_path.write_text(json.dumps(document), encoding="utf-8")
            self._write_contract_sidecar(rig_path, overrides={"head": {"node_index": len(document["nodes"]) - 1, "node_name": "contract_extra"}})

            with self.assertRaises(RetargetError) as ctx:
                inspect_rig(rig_path)

        self.assertEqual(ctx.exception.code, MAPPING_INCOMPATIBLE)
        diagnostics = ctx.exception.diagnostics["humanoid_contract"]["diagnostics"]
        self.assertIn("humanoid_contract_role_not_skin_joint:head:17", diagnostics)

    def test_humanoid_contract_duplicate_roles_and_node_bindings_fail_closed(self):
        roles = self._contract_roles()
        valid_bindings = [
            {"role": role, "node_index": index, "node_name": f"contract_{role}"}
            for index, role in enumerate(roles)
        ]
        duplicate_role_bindings = valid_bindings + [{"role": "head", "node_index": 4, "node_name": "contract_head"}]
        duplicate_node_bindings = [
            {**binding, "node_index": 0, "node_name": "contract_hips"} if binding["role"] == "head" else binding
            for binding in valid_bindings
        ]
        cases = (
            (duplicate_role_bindings, "humanoid_contract_duplicate_role:head"),
            (duplicate_node_bindings, "humanoid_contract_duplicate_node_binding:head:hips:0"),
        )
        for bindings, expected in cases:
            with self.subTest(expected=expected):
                with tempfile.TemporaryDirectory() as tmp:
                    rig_path = Path(tmp) / "contract_avatar.gltf"
                    self._write_contract_rig(rig_path)
                    self._write_contract_sidecar_payload(rig_path, roles=bindings)

                    with self.assertRaises(RetargetError) as ctx:
                        inspect_rig(rig_path)

                self.assertEqual(ctx.exception.code, MAPPING_INCOMPATIBLE)
                diagnostics = ctx.exception.diagnostics["humanoid_contract"]["diagnostics"]
                self.assertIn(expected, diagnostics)
                self.assertEqual(len(diagnostics), len(dict.fromkeys(diagnostics)))

    def test_humanoid_contract_required_source_gaps_fail_closed_for_calibration(self):
        with tempfile.TemporaryDirectory() as tmp:
            rig_path = Path(tmp) / "contract_avatar.gltf"
            bvh_path = Path(tmp) / "root_with_core_only.bvh"
            self._write_contract_rig(rig_path)
            self._write_contract_sidecar(rig_path, basis=True)
            self._write_root_bvh(bvh_path, include_hips=True)
            rig = inspect_rig(rig_path)
            motion = parse_bvh(bvh_path)

        with self.assertRaises(RetargetError) as ctx:
            build_retarget_clip(motion, rig)

        self.assertEqual(ctx.exception.code, CALIBRATION_UNAVAILABLE)
        self.assertIn("missing_source_role:left_arm:left_upper_arm", ctx.exception.message)
        self.assertFalse(ctx.exception.diagnostics["retarget_calibration"]["fallback_widened"])

    def test_humanoid_contract_missing_basis_fails_closed_for_calibration(self):
        with tempfile.TemporaryDirectory() as tmp:
            rig_path = Path(tmp) / "contract_avatar.gltf"
            bvh_path = Path(tmp) / "contract_motion.bvh"
            self._write_contract_rig(rig_path)
            self._write_contract_sidecar(rig_path, basis=False)
            self._write_full_contract_bvh(bvh_path)
            rig = inspect_rig(rig_path)
            motion = parse_bvh(bvh_path)

        with self.assertRaises(RetargetError) as ctx:
            build_retarget_clip(motion, rig)

        self.assertEqual(ctx.exception.code, CALIBRATION_UNAVAILABLE)
        self.assertIn("requires target rest transforms or basis hints", ctx.exception.message)

    def test_humanoid_contract_missing_required_role_and_chain_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            rig_path = Path(tmp) / "contract_avatar.gltf"
            self._write_contract_rig(rig_path)
            self._write_contract_sidecar(rig_path, omit_roles={"neck"}, omit_chains={"torso"})

            with self.assertRaises(RetargetError) as ctx:
                inspect_rig(rig_path)

        self.assertEqual(ctx.exception.code, MAPPING_INCOMPATIBLE)
        self.assertIn("humanoid_contract_missing_role:neck", ctx.exception.message)
        self.assertIn("humanoid_contract_missing_chain:torso", ctx.exception.message)

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
        self.assertGreaterEqual(len(animation["samplers"]), 2)
        self.assertGreaterEqual(len(animation["channels"]), 2)
        self.assertEqual(animation["channels"][0]["target"], {"node": 0, "path": "translation"})
        sampler = animation["samplers"][0]
        self.assertEqual(document["accessors"][sampler["input"]]["type"], "SCALAR")
        self.assertEqual(document["accessors"][sampler["output"]]["type"], "VEC3")
        rotation_channel = next(channel for channel in animation["channels"] if channel["target"]["path"] == "rotation")
        rotation_sampler = animation["samplers"][rotation_channel["sampler"]]
        self.assertEqual(document["accessors"][rotation_sampler["output"]]["type"], "VEC4")

    def test_exporter_preserves_target_rest_transforms_at_first_frame(self):
        exporter = GlbAnimationExporter()
        rest_translation = (3.0, 4.0, 5.0)
        rest_rotation = (0.0, 0.0, 0.3826834323650898, 0.9238795325112867)
        source_first_rotation = (0.0, 0.0, 0.7071067811865475, 0.7071067811865476)
        source_second_rotation = (0.0, 0.0, 1.0, 0.0)
        expected_second_rotation = (0.0, 0.0, 0.9238795325112867, 0.38268343236508984)
        clip = RetargetClip(
            name="Kimodo Motion",
            duration=1.0,
            frame_time=1.0,
            root_translations=((10.0, 0.0, 0.0), (11.0, 2.0, 3.0)),
            channels=(
                RetargetChannel(target_joint="Hips", path="translation", times=(0.0, 1.0), values=((10.0, 0.0, 0.0), (11.0, 2.0, 3.0))),
                RetargetChannel(target_joint="LeftArm", path="rotation", times=(0.0, 1.0), values=(source_first_rotation, source_second_rotation)),
            ),
            mapping=JointMapping(confidence="compatible", pairs={"Hips": "Hips", "LeftArm": "LeftArm"}, warnings=()),
            warnings=(),
        )

        with tempfile.TemporaryDirectory() as tmp:
            rig_path = Path(tmp) / "rest_rigged.gltf"
            document = json.loads((FIXTURES / "tiny_rigged.gltf").read_text(encoding="utf-8"))
            document["nodes"][0]["translation"] = list(rest_translation)
            document["nodes"][5]["rotation"] = list(rest_rotation)
            rig_path.write_text(json.dumps(document), encoding="utf-8")
            rig = inspect_rig(rig_path)
            output = exporter.write_animation(rig, clip, Path(tmp) / "animated.gltf")
            animated = json.loads(output.read_text(encoding="utf-8"))

        resolver = BufferResolver(animated)
        animation = animated["animations"][-1]
        translation_samples = self._decoded_channel_outputs(animated, animation, "translation", resolver)
        rotation_samples = self._decoded_channel_outputs(animated, animation, "rotation", resolver)

        self.assertEqual(translation_samples[0], rest_translation)
        self.assertEqual(translation_samples[1], (4.0, 6.0, 8.0))
        self.assertTupleAlmostEqual(rotation_samples[0], rest_rotation)
        self.assertTupleAlmostEqual(rotation_samples[1], expected_second_rotation)

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
        self.assertIn("rotation", {channel["target"]["path"] for channel in document["animations"][-1]["channels"]})

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

    def _write_anonymous_unirig_46(self, path: Path) -> None:
        children = {
            0: [1, 38, 42],
            1: [2],
            2: [3],
            3: [4, 6, 22],
            4: [5],
            6: [7],
            7: [8],
            8: [9],
            9: [10, 13, 16, 19],
            10: [11],
            11: [12],
            13: [14],
            14: [15],
            16: [17],
            17: [18],
            19: [20],
            20: [21],
            22: [23],
            23: [24],
            24: [25],
            25: [26, 29, 32, 35],
            26: [27],
            27: [28],
            29: [30],
            30: [31],
            32: [33],
            33: [34],
            35: [36],
            36: [37],
            38: [39],
            39: [40],
            40: [41],
            42: [43],
            43: [44],
            44: [45],
        }
        self._write_anonymous_bone_rig(path, joint_count=46, children=children)

    def _write_contract_rig(self, path: Path) -> None:
        roles = self._contract_roles()
        nodes = []
        for index, role in enumerate(roles):
            node = {"name": f"contract_{role}"}
            children = [index + 1] if index + 1 < len(roles) else []
            if children:
                node["children"] = children
            nodes.append(node)
        path.write_text(
            json.dumps({
                "asset": {"version": "2.0"},
                "nodes": nodes,
                "skins": [{"joints": list(range(len(roles))), "inverseBindMatrices": 0}],
            }),
            encoding="utf-8",
        )

    def _write_full_contract_bvh(self, path: Path, *, left_upper_arm_x_degrees: float = 0.0) -> None:
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
        second_frame[19] = left_upper_arm_x_degrees
        path.write_text(
            hierarchy
            + " ".join(str(value) for value in first_frame)
            + "\n"
            + " ".join(str(value) for value in second_frame)
            + "\n",
            encoding="utf-8",
        )

    def _write_contract_sidecar(
        self,
        path: Path,
        *,
        schema: str = "modly.humanoid.v1",
        basis: bool = False,
        overrides: dict | None = None,
        omit_roles: set[str] | None = None,
        omit_chains: set[str] | None = None,
    ) -> None:
        roles = self._contract_roles()
        omit_roles = omit_roles or set()
        bindings = {
            role: {"node_index": index, "node_name": f"contract_{role}"}
            for index, role in enumerate(roles)
            if role not in omit_roles
        }
        if overrides:
            bindings.update(overrides)
        chains = {
            "torso": ["hips", "spine", "chest", "neck", "head"],
            "left_arm": ["left_upper_arm", "left_forearm", "left_hand"],
            "right_arm": ["right_upper_arm", "right_forearm", "right_hand"],
            "left_leg": ["left_upper_leg", "left_lower_leg", "left_foot"],
            "right_leg": ["right_upper_leg", "right_lower_leg", "right_foot"],
        }
        for chain in omit_chains or set():
            chains.pop(chain, None)
        contract = {"schema": schema, "roles": bindings, "chains": chains, "quality": ["contract_level_provenance"]}
        if basis:
            contract["basis_hints"] = {"up_axis": "Y"}
            contract["rest"] = {"source": "synthetic"}
        path.with_suffix(".rigmeta.json").write_text(json.dumps({"humanoid_contract": contract}), encoding="utf-8")

    def _write_contract_sidecar_payload(self, path: Path, *, roles) -> None:
        chains = {
            "torso": ["hips", "spine", "chest", "neck", "head"],
            "left_arm": ["left_upper_arm", "left_forearm", "left_hand"],
            "right_arm": ["right_upper_arm", "right_forearm", "right_hand"],
            "left_leg": ["left_upper_leg", "left_lower_leg", "left_foot"],
            "right_leg": ["right_upper_leg", "right_lower_leg", "right_foot"],
        }
        path.with_suffix(".rigmeta.json").write_text(
            json.dumps({"humanoid_contract": {"schema": "modly.humanoid.v1", "roles": roles, "chains": chains}}),
            encoding="utf-8",
        )

    def _contract_roles(self) -> list[str]:
        return [
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

    def _write_anonymous_unirig_52(self, path: Path) -> None:
        children = {
            0: [1, 44, 48],
            1: [2],
            2: [3],
            3: [4, 6, 25],
            4: [5],
            6: [7],
            7: [8],
            8: [9],
            9: [10, 13, 16, 19, 22],
            10: [11],
            11: [12],
            13: [14],
            14: [15],
            16: [17],
            17: [18],
            19: [20],
            20: [21],
            22: [23],
            23: [24],
            25: [26],
            26: [27],
            27: [28],
            28: [29, 32, 35, 38, 41],
            29: [30],
            30: [31],
            32: [33],
            33: [34],
            35: [36],
            36: [37],
            38: [39],
            39: [40],
            41: [42],
            42: [43],
            44: [45],
            45: [46],
            46: [47],
            48: [49],
            49: [50],
            50: [51],
        }
        self._write_anonymous_bone_rig(path, joint_count=52, children=children)

    def _write_anonymous_bone_rig(self, path: Path, *, joint_count: int, children: dict[int, list[int]]) -> None:
        nodes = []
        for index in range(joint_count):
            node = {"name": f"bone_{index}"}
            if index in children:
                node["children"] = children[index]
            nodes.append(node)
        path.write_text(
            json.dumps({
                "asset": {"version": "2.0"},
                "nodes": nodes,
                "skins": [{"joints": list(range(joint_count)), "inverseBindMatrices": 0}],
            }),
            encoding="utf-8",
        )

    def _write_root_bvh(self, path: Path, *, include_hips: bool) -> None:
        pelvis_joint = "Hips" if include_hips else "Spine"
        hierarchy = f"""HIERARCHY
ROOT Root
{{
  OFFSET 0.0 0.0 0.0
  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
  JOINT {pelvis_joint}
  {{
    OFFSET 0.0 1.0 0.0
    CHANNELS 3 Zrotation Xrotation Yrotation
"""
        if include_hips:
            hierarchy += """    JOINT Spine
    {
      OFFSET 0.0 1.0 0.0
      CHANNELS 3 Zrotation Xrotation Yrotation
      JOINT Head
      {
        OFFSET 0.0 1.0 0.0
        CHANNELS 3 Zrotation Xrotation Yrotation
        End Site
        {
          OFFSET 0.0 0.5 0.0
        }
      }
    }
"""
        else:
            hierarchy += """    JOINT Head
    {
      OFFSET 0.0 1.0 0.0
      CHANNELS 3 Zrotation Xrotation Yrotation
      End Site
      {
        OFFSET 0.0 0.5 0.0
      }
    }
"""
        hierarchy += """  }
}
MOTION
Frames: 2
Frame Time: 0.0333333
"""
        frame_width = 15 if include_hips else 12
        first_frame = " ".join("0" for _ in range(frame_width))
        second_values = ["2", "3", "4"] + ["0" for _ in range(frame_width - 3)]
        path.write_text(f"{hierarchy}{first_frame}\n{' '.join(second_values)}\n", encoding="utf-8")

    def _decoded_channel_outputs(self, document, animation, path, resolver):
        channel = next(channel for channel in animation["channels"] if channel["target"]["path"] == path)
        sampler = animation["samplers"][channel["sampler"]]
        expected = {"VEC4"} if path == "rotation" else {"VEC3"}
        return decode_accessor(document, resolver, sampler["output"], expected_types=expected)

    def assertTupleAlmostEqual(self, actual, expected, places=6):
        self.assertEqual(len(actual), len(expected))
        for actual_value, expected_value in zip(actual, expected):
            self.assertAlmostEqual(actual_value, expected_value, places=places)


if __name__ == "__main__":
    unittest.main()
