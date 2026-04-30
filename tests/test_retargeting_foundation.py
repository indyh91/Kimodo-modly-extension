import json
import math
import struct
import tempfile
import unittest
from pathlib import Path

import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from glb_animation_exporter import GlbAnimationExporter, inspect_exporter_dependencies  # noqa: E402
from kimodo_bvh import locate_bvh, parse_bvh  # noqa: E402
from retarget_errors import CALIBRATION_UNAVAILABLE, EXPORT_FAILED, MAPPING_INCOMPATIBLE, MISSING_BVH, RetargetError  # noqa: E402
from retarget_metadata import retarget_success_metadata  # noqa: E402
from retargeting import (  # noqa: E402
    JointMapping,
    MotionContext,
    RetargetChannel,
    RetargetClip,
    apply_contact_aware_foot_locking,
    foot_trajectory_diagnostics,
    stance_foot_slide_diagnostics,
    _motion_quality_settings_for_role,
    _target_basis_child_role,
    build_joint_mapping,
    build_retarget_clip,
    load_motion_context,
)
from rig_inspector import inspect_rig  # noqa: E402
from validation.animated_glb_validator import BufferResolver, decode_accessor  # noqa: E402


FIXTURES = ROOT / "tests" / "fixtures"
FIVE_MOTION_WORKFLOWS = {
    "walking": Path("/home/drhepa/Documentos/Modly/workspace/Workflows/kimodo-20260429-230826-7caf3768"),
    "dancing": Path("/home/drhepa/Documentos/Modly/workspace/Workflows/kimodo-20260429-231135-0118d38d"),
}


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

    def test_static_synthetic_root_suppresses_child_hips_translation_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            bvh_path = Path(tmp) / "static_root_hips_motion.bvh"
            rig_path = Path(tmp) / "anonymous_unirig.gltf"
            self._write_static_root_with_hips_position_bvh(
                bvh_path,
                first_hips=(10.0, 0.5, -3.0),
                second_hips=(12.25, 1.5, 0.5),
                second_hips_y_rotation=4.0,
            )
            self._write_anonymous_unirig_46(rig_path)
            motion = parse_bvh(bvh_path)
            rig = inspect_rig(rig_path)

        clip = build_retarget_clip(motion, rig)

        translation = next(channel for channel in clip.channels if channel.target_joint == "bone_0" and channel.path == "translation")
        self.assertEqual(motion.root, "Root")
        self.assertNotIn("Root", clip.mapping.pairs)
        self.assertEqual(clip.mapping.pairs["Hips"], "bone_0")
        self.assertEqual(translation.values, ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)))
        self.assertEqual(translation.diagnostics["root_motion_source"], "hips_suppressed_in_place")
        self.assertEqual(translation.diagnostics["suppressed_source_joint"], "Hips")
        self.assertIn("root_motion_mode_in_place_default", clip.warnings)
        self.assertIn("root_motion_source:hips_suppressed_in_place", clip.warnings)
        rotation = next(channel for channel in clip.channels if channel.target_joint == "bone_0" and channel.path == "rotation")
        self.assertTupleAlmostEqual(rotation.values[1], (0.0, 0.034899496702501, 0.0, 0.999390827019096))

    def test_exporter_keeps_rest_space_when_child_hips_root_motion_is_suppressed(self):
        rest_translation = (3.0, 4.0, 5.0)
        with tempfile.TemporaryDirectory() as tmp:
            bvh_path = Path(tmp) / "static_root_hips_motion.bvh"
            rig_path = Path(tmp) / "anonymous_unirig.gltf"
            self._write_static_root_with_hips_position_bvh(
                bvh_path,
                first_hips=(10.0, 0.5, -3.0),
                second_hips=(12.25, 1.5, 0.5),
            )
            self._write_anonymous_unirig_46(rig_path)
            document = json.loads(rig_path.read_text(encoding="utf-8"))
            document["nodes"][0]["translation"] = list(rest_translation)
            rig_path.write_text(json.dumps(document), encoding="utf-8")
            motion = parse_bvh(bvh_path)
            rig = inspect_rig(rig_path)
            clip = build_retarget_clip(motion, rig)
            output = GlbAnimationExporter().write_animation(rig, clip, Path(tmp) / "animated.gltf")
            animated = json.loads(output.read_text(encoding="utf-8"))

        animation = animated["animations"][-1]
        translation_samples = self._decoded_channel_outputs(animated, animation, "translation", BufferResolver(animated))
        self.assertEqual(translation_samples[0], rest_translation)
        self.assertEqual(translation_samples[1], rest_translation)

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
        self.assertEqual(clip.channels[0].diagnostics["root_motion_source"], "root")

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

    def test_calibrated_target_basis_for_hips_uses_spine_child_vector_not_hips_placement(self):
        with tempfile.TemporaryDirectory() as tmp:
            rig_path = Path(tmp) / "contract_avatar.gltf"
            bvh_path = Path(tmp) / "contract_motion.bvh"
            self._write_contract_rig(rig_path)
            self._write_contract_sidecar(rig_path)
            self._replace_contract_rest_roles(
                rig_path,
                {
                    "hips": [9.0, 0.0, 0.0],
                    "spine": [0.0, 2.0, 0.0],
                },
            )
            self._write_full_contract_bvh(bvh_path)
            rig = inspect_rig(rig_path)
            motion = parse_bvh(bvh_path)

        clip = build_retarget_clip(motion, rig)

        basis = clip.calibration["basis_sources"]["contract_hips"]
        self.assertEqual(basis["target"], "target_rest_child_transform:hips->spine")
        self.assertTupleAlmostEqual(tuple(basis["target_quaternion"]), (0.0, 0.0, 0.0, 1.0))

    def test_calibrated_target_basis_for_shoulder_uses_upper_arm_child_vector_not_chest_to_shoulder(self):
        with tempfile.TemporaryDirectory() as tmp:
            rig_path = Path(tmp) / "canonical_unirig_shoulders.gltf"
            bvh_path = Path(tmp) / "contract_shoulder_motion.bvh"
            self._write_canonical_unirig_shoulder_contract_rig_and_sidecar(rig_path)
            self._replace_contract_rest_roles(
                rig_path,
                {
                    "left_shoulder": [0.0, 9.0, 0.0],
                    "left_upper_arm": [1.0, 0.0, 0.0],
                },
            )
            self._write_shoulder_contract_bvh(bvh_path, left_shoulder_x_degrees=45.0)
            rig = inspect_rig(rig_path)
            motion = parse_bvh(bvh_path)

        clip = build_retarget_clip(motion, rig)

        basis = clip.calibration["basis_sources"]["bone_6"]
        self.assertEqual(basis["target"], "target_rest_child_transform:left_shoulder->left_upper_arm")
        self.assertTupleAlmostEqual(tuple(basis["target_quaternion"]), (0.0, 0.0, -0.7071067811865475, 0.7071067811865476))

    def test_calibrated_target_basis_for_upper_leg_uses_lower_leg_child_vector(self):
        with tempfile.TemporaryDirectory() as tmp:
            rig_path = Path(tmp) / "contract_avatar.gltf"
            bvh_path = Path(tmp) / "contract_motion.bvh"
            self._write_contract_rig(rig_path)
            self._write_contract_sidecar(rig_path)
            self._replace_contract_rest_roles(
                rig_path,
                {
                    "left_upper_leg": [7.0, 0.0, 0.0],
                    "left_lower_leg": [0.0, -2.0, 0.0],
                },
            )
            self._write_full_contract_bvh(bvh_path)
            rig = inspect_rig(rig_path)
            motion = parse_bvh(bvh_path)

        clip = build_retarget_clip(motion, rig)

        basis = clip.calibration["basis_sources"]["contract_left_upper_leg"]
        self.assertEqual(basis["target"], "target_rest_child_transform:left_upper_leg->left_lower_leg")
        self.assertTupleAlmostEqual(tuple(basis["target_quaternion"]), (1.0, 0.0, 0.0, 0.0))

    def test_calibrated_target_basis_for_upper_arm_uses_forearm_child_vector(self):
        with tempfile.TemporaryDirectory() as tmp:
            rig_path = Path(tmp) / "contract_avatar.gltf"
            bvh_path = Path(tmp) / "contract_motion.bvh"
            self._write_contract_rig(rig_path)
            self._write_contract_sidecar(rig_path)
            self._replace_contract_rest_roles(
                rig_path,
                {
                    "left_upper_arm": [7.0, 0.0, 0.0],
                    "left_forearm": [0.0, 3.0, 0.0],
                },
            )
            self._write_full_contract_bvh(bvh_path, left_upper_arm_x_degrees=90.0)
            rig = inspect_rig(rig_path)
            motion = parse_bvh(bvh_path)

        clip = build_retarget_clip(motion, rig)

        basis = clip.calibration["basis_sources"]["contract_left_upper_arm"]
        self.assertEqual(basis["target"], "target_rest_child_transform:left_upper_arm->left_forearm")
        self.assertTupleAlmostEqual(tuple(basis["target_quaternion"]), (0.0, 0.0, 0.0, 1.0))

    def test_calibrated_metadata_warns_when_basis_roll_remains_inferred_not_visual_safe(self):
        with tempfile.TemporaryDirectory() as tmp:
            rig_path = Path(tmp) / "contract_avatar.gltf"
            bvh_path = Path(tmp) / "contract_motion.bvh"
            self._write_contract_rig(rig_path)
            self._write_contract_sidecar(rig_path)
            self._replace_contract_rest_roles(rig_path, {"hips": [0.0, 1.0, 0.0], "spine": [0.0, 1.0, 0.0]})
            self._write_full_contract_bvh(bvh_path)
            rig = inspect_rig(rig_path)
            motion = parse_bvh(bvh_path)

        clip = build_retarget_clip(motion, rig)

        self.assertEqual(clip.mapping.confidence, "calibrated")
        self.assertIn("target_basis_roll_inferred_not_visual_safe", clip.warnings)
        self.assertIn("target_basis_roll_inferred_not_visual_safe", clip.calibration["warnings_by_severity"]["warning"])
        self.assertEqual(clip.calibration["visual_safety"], "not_validated")

    def test_calibrated_target_basis_prefers_rest_local_matrix_and_preserves_roll(self):
        with tempfile.TemporaryDirectory() as tmp:
            rig_path = Path(tmp) / "contract_avatar.gltf"
            bvh_path = Path(tmp) / "contract_motion.bvh"
            self._write_contract_rig(rig_path)
            self._write_contract_sidecar(rig_path)
            role_entries = {role: {"rest_local": self._translation_matrix([0.0, 0.0, 0.0])} for role in self._contract_roles()}
            role_entries.update(
                {
                    "left_upper_arm": {
                        "rest_local": self._rotation_y_matrix(90.0, translation=[7.0, 0.0, 0.0]),
                        "local_translation": [7.0, 0.0, 0.0],
                    },
                    "left_forearm": {"rest_local": self._translation_matrix([0.0, 0.0, 0.0]), "local_translation": [0.0, 3.0, 0.0]},
                }
            )
            self._replace_contract_rest_role_entries(
                rig_path,
                role_entries,
            )
            self._write_full_contract_bvh(bvh_path, left_upper_arm_x_degrees=90.0)
            rig = inspect_rig(rig_path)
            motion = parse_bvh(bvh_path)

        clip = build_retarget_clip(motion, rig)

        basis = clip.calibration["basis_sources"]["contract_left_upper_arm"]
        self.assertEqual(basis["target"], "target_rest_matrix:rest_local:left_upper_arm")
        self.assertTupleAlmostEqual(tuple(basis["target_quaternion"]), (0.0, 0.7071067811865475, 0.0, 0.7071067811865476))
        self.assertNotIn("target_basis_roll_inferred_not_visual_safe", clip.warnings)
        self.assertEqual(clip.calibration["basis_quality"], "full_rest_matrix_basis")
        self.assertEqual(clip.calibration["visual_safety"], "not_validated_full_basis_requires_visual_review")

    def test_full_matrix_target_basis_delta_uses_target_local_frame_without_double_rest(self):
        target_rest_rotation = (0.0, 0.7071067811865475, 0.0, 0.7071067811865476)
        expected_target_delta = (0.0, 0.0, 0.7071067811865475, 0.7071067811865476)
        expected_exported_second_sample = (0.5, 0.5, 0.5, 0.5)
        with tempfile.TemporaryDirectory() as tmp:
            rig_path = Path(tmp) / "contract_avatar.gltf"
            bvh_path = Path(tmp) / "contract_motion.bvh"
            self._write_contract_rig(rig_path)
            self._write_contract_sidecar(rig_path)
            self._set_contract_node_rotation(rig_path, "left_upper_arm", target_rest_rotation)
            role_entries = {role: {"rest_local": self._translation_matrix([0.0, 0.0, 0.0])} for role in self._contract_roles()}
            role_entries["left_upper_arm"] = {
                "rest_local": self._rotation_y_matrix(90.0),
                "local_translation": [7.0, 0.0, 0.0],
            }
            self._replace_contract_rest_role_entries(rig_path, role_entries)
            self._write_full_contract_bvh(bvh_path, left_upper_arm_x_degrees=90.0)
            rig = inspect_rig(rig_path)
            motion = parse_bvh(bvh_path)

            clip = build_retarget_clip(motion, rig)
            output = GlbAnimationExporter().write_animation(rig, clip, Path(tmp) / "animated.gltf")
            animated = json.loads(output.read_text(encoding="utf-8"))

        channel = next(channel for channel in clip.channels if channel.target_joint == "contract_left_upper_arm" and channel.path == "rotation")
        self.assertEqual(channel.diagnostics["target_basis_composition"], "inverse(target_basis) * source_basis")
        self.assertTupleAlmostEqual(channel.values[1], expected_target_delta)

        target_node = next(index for index, node in enumerate(animated["nodes"]) if node.get("name") == "contract_left_upper_arm")
        animation = animated["animations"][-1]
        animation_channel = next(
            channel
            for channel in animation["channels"]
            if channel["target"] == {"node": target_node, "path": "rotation"}
        )
        output_accessor = animation["samplers"][animation_channel["sampler"]]["output"]
        rotation_samples = decode_accessor(animated, BufferResolver(animated), output_accessor, expected_types={"VEC4"})
        self.assertTupleAlmostEqual(rotation_samples[0], target_rest_rotation)
        self.assertTupleAlmostEqual(rotation_samples[1], expected_exported_second_sample)

    def test_full_matrix_target_basis_mirrored_rest_rotation_preserves_delta_sign_direction(self):
        with tempfile.TemporaryDirectory() as tmp:
            rig_path = Path(tmp) / "contract_avatar.gltf"
            bvh_path = Path(tmp) / "contract_motion.bvh"
            self._write_contract_rig(rig_path)
            self._write_contract_sidecar(rig_path)
            role_entries = {role: {"rest_local": self._translation_matrix([0.0, 0.0, 0.0])} for role in self._contract_roles()}
            role_entries["left_upper_arm"] = {"rest_local": self._rotation_y_matrix(-90.0)}
            self._replace_contract_rest_role_entries(rig_path, role_entries)
            self._write_full_contract_bvh(bvh_path, left_upper_arm_x_degrees=90.0)
            rig = inspect_rig(rig_path)
            motion = parse_bvh(bvh_path)

        clip = build_retarget_clip(motion, rig)

        channel = next(channel for channel in clip.channels if channel.target_joint == "contract_left_upper_arm" and channel.path == "rotation")
        self.assertEqual(channel.diagnostics["target_basis_composition"], "inverse(target_basis) * source_basis")
        self.assertTupleAlmostEqual(channel.values[1], (0.0, 0.0, -0.7071067811865475, 0.7071067811865476))

    def test_calibrated_target_basis_prefers_rest_world_matrix_when_local_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            rig_path = Path(tmp) / "contract_avatar.gltf"
            bvh_path = Path(tmp) / "contract_motion.bvh"
            self._write_contract_rig(rig_path)
            self._write_contract_sidecar(rig_path)
            role_entries = {role: {"rest_world": self._translation_matrix([0.0, 0.0, 0.0])} for role in self._contract_roles()}
            role_entries["left_upper_arm"] = {"rest_world": self._rotation_y_matrix(-90.0)}
            self._replace_contract_rest_role_entries(
                rig_path,
                role_entries,
            )
            self._write_full_contract_bvh(bvh_path, left_upper_arm_x_degrees=90.0)
            rig = inspect_rig(rig_path)
            motion = parse_bvh(bvh_path)

        clip = build_retarget_clip(motion, rig)

        basis = clip.calibration["basis_sources"]["contract_left_upper_arm"]
        self.assertEqual(basis["target"], "target_rest_matrix:rest_world:left_upper_arm")
        self.assertTupleAlmostEqual(tuple(basis["target_quaternion"]), (0.0, -0.7071067811865475, 0.0, 0.7071067811865476))
        self.assertNotIn("target_basis_roll_inferred_not_visual_safe", clip.calibration["warnings_by_severity"]["warning"])

    def test_calibrated_target_basis_falls_back_to_child_vector_when_matrix_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            rig_path = Path(tmp) / "contract_avatar.gltf"
            bvh_path = Path(tmp) / "contract_motion.bvh"
            self._write_contract_rig(rig_path)
            self._write_contract_sidecar(rig_path)
            self._replace_contract_rest_role_entries(
                rig_path,
                {
                    "left_upper_arm": {"rest_local": [[1.0, 0.0], [0.0, 1.0]], "local_translation": [7.0, 0.0, 0.0]},
                    "left_forearm": {"local_translation": [0.0, 3.0, 0.0]},
                },
            )
            self._write_full_contract_bvh(bvh_path, left_upper_arm_x_degrees=90.0)
            rig = inspect_rig(rig_path)
            motion = parse_bvh(bvh_path)

        clip = build_retarget_clip(motion, rig)

        basis = clip.calibration["basis_sources"]["contract_left_upper_arm"]
        self.assertEqual(basis["target"], "target_rest_child_transform:left_upper_arm->left_forearm")
        self.assertTupleAlmostEqual(tuple(basis["target_quaternion"]), (0.0, 0.0, 0.0, 1.0))
        self.assertIn("target_basis_roll_inferred_not_visual_safe", clip.warnings)
        self.assertEqual(clip.calibration["basis_quality"], "inferred_axis_roll_unresolved")

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

    def test_calibrated_motion_quality_filter_damps_large_right_hand_forearm_spikes(self):
        with tempfile.TemporaryDirectory() as tmp:
            rig_path = Path(tmp) / "contract_avatar.gltf"
            bvh_path = Path(tmp) / "spiky_contract_motion.bvh"
            self._write_contract_rig(rig_path)
            self._write_contract_sidecar(rig_path, basis=True)
            self._write_motion_quality_contract_bvh(
                bvh_path,
                right_forearm_x_degrees=(0.0, 18.0, 150.0, 24.0),
                right_hand_x_degrees=(0.0, 12.0, 145.0, 18.0),
                left_upper_arm_x_degrees=(0.0, 8.0, 16.0, 20.0),
            )
            rig = inspect_rig(rig_path)
            motion = parse_bvh(bvh_path)

        clip = build_retarget_clip(motion, rig)

        forearm = next(channel for channel in clip.channels if channel.target_joint == "contract_right_forearm" and channel.path == "rotation")
        hand = next(channel for channel in clip.channels if channel.target_joint == "contract_right_hand" and channel.path == "rotation")
        left_upper_arm = next(channel for channel in clip.channels if channel.target_joint == "contract_left_upper_arm" and channel.path == "rotation")
        self.assertLessEqual(max(self._quaternion_angle_degrees(value) for value in forearm.values), 75.0)
        self.assertLessEqual(max(self._quaternion_angle_degrees(value) for value in hand.values), 65.0)
        self.assertEqual(forearm.values[0], (0.0, 0.0, 0.0, 1.0))
        self.assertEqual(hand.values[0], (0.0, 0.0, 0.0, 1.0))
        self.assertEqual(forearm.diagnostics["motion_quality_filter"], "role_aware_conservative")
        self.assertEqual(hand.diagnostics["motion_quality_filter"], "role_aware_conservative")
        self.assertTupleAlmostEqual(left_upper_arm.values[-1], self._axis_angle_quaternion_x(20.0))

    def test_motion_quality_settings_keep_hand_at_65_and_forearm_lower_arm_at_75(self):
        self.assertEqual(_motion_quality_settings_for_role("left_forearm"), (75.0, 16.0))
        self.assertEqual(_motion_quality_settings_for_role("right_lower_arm"), (75.0, 16.0))
        self.assertEqual(_motion_quality_settings_for_role("left_hand"), (65.0, 12.0))

    def test_motion_quality_settings_cover_exact_full_body_roles(self):
        expected = {
            "hips": (45.0, 12.0),
            "spine": (35.0, 10.0),
            "chest": (45.0, 12.0),
            "neck": (35.0, 12.0),
            "head": (45.0, 16.0),
            "left_shoulder": (90.0, 180.0),
            "right_shoulder": (90.0, 180.0),
            "left_upper_arm": (120.0, 180.0),
            "right_upper_arm": (120.0, 180.0),
            "left_forearm": (75.0, 16.0),
            "right_lower_arm": (75.0, 16.0),
            "left_hand": (65.0, 12.0),
            "right_hand": (65.0, 12.0),
            "left_upper_leg": (110.0, 20.0),
            "right_upper_leg": (110.0, 20.0),
            "left_lower_leg": (125.0, 18.0),
            "right_lower_leg": (125.0, 18.0),
            "left_foot": (70.0, 14.0),
            "right_foot": (70.0, 14.0),
        }

        for role, settings in expected.items():
            with self.subTest(role=role):
                self.assertEqual(_motion_quality_settings_for_role(role), settings)

    def test_motion_context_prefers_smooth_root_and_reports_foot_contact_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp:
            npz_path = Path(tmp) / "motion.npz"
            np.savez(
                npz_path,
                root_positions=np.array([[0.0, 0.0, 0.0], [50.0, 0.0, 0.0]], dtype=float),
                smooth_root_pos=np.array([[2.0, 1.0, 3.0], [2.6, 1.4, 4.0]], dtype=float),
                foot_contacts=np.array([[1, 0, 1, 0, 0, 1], [0, 1, 0, 1, 1, 0]], dtype=float),
            )

            context = load_motion_context(npz_path)

        self.assertIsInstance(context, MotionContext)
        self.assertEqual(context.root_source, "smooth_root_pos")
        self.assertEqual(context.root_frame_count, 2)
        self.assertEqual(context.foot_contact_shape, (2, 6))
        self.assertAlmostEqual(context.foot_contact_ratio, 0.5)
        self.assertEqual(context.root_deltas(frame_count=2), ((0.0, 0.0, 0.0), (0.6, 0.0, 1.0)))

    def test_locomotion_policy_uses_npz_deltas_not_raw_child_hips_translation(self):
        with tempfile.TemporaryDirectory() as tmp:
            bvh_path = Path(tmp) / "walking_child_hips.bvh"
            rig_path = Path(tmp) / "anonymous_unirig.gltf"
            npz_path = Path(tmp) / "motion.npz"
            self._write_static_root_with_hips_position_bvh(
                bvh_path,
                first_hips=(0.0, 0.0, 0.0),
                second_hips=(0.0, 0.0, 650.0),
            )
            self._write_anonymous_unirig_46(rig_path)
            np.savez(
                npz_path,
                smooth_root_pos=np.array([[10.0, 2.0, 20.0], [11.2, 2.7, 20.8]], dtype=float),
                root_positions=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 650.0]], dtype=float),
                foot_contacts=np.ones((2, 6), dtype=float),
            )
            motion = parse_bvh(bvh_path)
            rig = inspect_rig(rig_path)
            context = load_motion_context(npz_path)

        clip = build_retarget_clip(motion, rig, motion_context=context)

        translation = next(channel for channel in clip.channels if channel.target_joint == "bone_0" and channel.path == "translation")
        self.assertEqual(translation.values, ((0.0, 0.0, 0.0), (1.2, 0.0, 0.8)))
        self.assertEqual(translation.diagnostics["root_motion_source"], "smooth_root_pos")
        self.assertEqual(translation.diagnostics["root_motion_mode"], "preserve_scaled_npz_root_motion")
        self.assertNotEqual(translation.values[-1], (0.0, 0.0, 650.0))
        policy = clip.calibration["locomotion_policy"]
        self.assertTrue(policy["detected_locomotion"])
        self.assertEqual(policy["mode"], "preserve_scaled_npz_root_motion")
        self.assertEqual(policy["source"], "smooth_root_pos")
        self.assertEqual(policy["foot_contact_shape"], [2, 6])
        self.assertIn("contact_aware_foot_locking", policy["deferred"])

    def test_invalid_motion_context_keeps_hips_suppressed_in_place(self):
        with tempfile.TemporaryDirectory() as tmp:
            bvh_path = Path(tmp) / "static_root_hips_motion.bvh"
            rig_path = Path(tmp) / "anonymous_unirig.gltf"
            self._write_static_root_with_hips_position_bvh(
                bvh_path,
                first_hips=(10.0, 0.5, -3.0),
                second_hips=(12.25, 1.5, 0.5),
            )
            self._write_anonymous_unirig_46(rig_path)
            motion = parse_bvh(bvh_path)
            rig = inspect_rig(rig_path)
            context = MotionContext(path=None, root_positions=((0.0, 0.0),), smooth_root_pos=None, foot_contacts=None, source_keys=("root_positions",), warnings=())

        clip = build_retarget_clip(motion, rig, motion_context=context)

        translation = next(channel for channel in clip.channels if channel.target_joint == "bone_0" and channel.path == "translation")
        self.assertEqual(translation.values, ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)))
        policy = clip.calibration["locomotion_policy"]
        self.assertFalse(policy["detected_locomotion"])
        self.assertEqual(policy["mode"], "hips_suppressed_in_place")
        self.assertEqual(policy["source"], "none")
        self.assertIn("invalid_root_positions_shape", policy["warnings"])

    def test_source_and_retarget_foot_trajectory_diagnostics_report_forward_lateral_dominance(self):
        source_left = ((0.0, 0.0, 0.0), (0.04, 0.0, 0.50), (0.08, 0.0, 1.00))
        source_right = ((0.0, 0.0, 0.0), (-0.05, 0.0, 0.45), (-0.10, 0.0, 0.95))
        retarget_left = ((0.0, 0.0, 0.0), (0.70, 0.0, 0.20), (1.40, 0.0, 0.58))
        retarget_right = ((0.0, 0.0, 0.0), (-0.56, 0.0, 0.16), (-1.12, 0.0, 0.34))

        diagnostics = foot_trajectory_diagnostics(
            source_feet={"left": source_left, "right": source_right},
            retarget_feet={"left": retarget_left, "right": retarget_right},
        )

        self.assertEqual(diagnostics["source"]["left"]["dominant_axis"], "forward")
        self.assertEqual(diagnostics["source"]["right"]["dominant_axis"], "forward")
        self.assertEqual(diagnostics["retarget"]["left"]["dominant_axis"], "lateral")
        self.assertEqual(diagnostics["retarget"]["right"]["dominant_axis"], "lateral")
        self.assertAlmostEqual(diagnostics["source"]["left"]["forward_range"], 1.0)
        self.assertAlmostEqual(diagnostics["retarget"]["left"]["lateral_range"], 1.4)
        self.assertEqual(diagnostics["alignment"], "mismatch")

    def test_stance_foot_slide_diagnostics_use_foot_contacts_windows(self):
        foot_samples = {
            "left": ((0.0, 0.0, 0.0), (0.05, 0.0, 0.02), (0.80, 0.0, 0.00), (0.82, 0.0, 0.01)),
            "right": ((0.0, 0.0, 0.0), (0.40, 0.0, 0.0), (0.42, 0.0, 0.0), (0.43, 0.0, 0.0)),
        }
        contacts = (
            (1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
        )

        diagnostics = stance_foot_slide_diagnostics(foot_samples, contacts)

        self.assertEqual(diagnostics["contact_shape"], [4, 6])
        self.assertAlmostEqual(diagnostics["contact_ratio"], 2.0 / 12.0)
        self.assertEqual(diagnostics["left"]["windows"], [[0, 1]])
        self.assertEqual(diagnostics["right"]["windows"], [[2, 3]])
        self.assertAlmostEqual(diagnostics["left"]["max_slide"], math.hypot(0.05, 0.02))
        self.assertAlmostEqual(diagnostics["right"]["max_slide"], 0.01)

    def test_contact_aware_foot_locking_reduces_stance_slide_without_moving_swing_samples(self):
        foot_samples = {
            "left": ((0.0, 0.0, 0.0), (0.20, 0.0, 0.00), (0.40, 0.0, 0.00), (0.60, 0.0, 0.00)),
            "right": ((0.0, 0.0, 0.0), (0.00, 0.0, 0.30), (0.00, 0.0, 0.60), (0.00, 0.0, 0.90)),
        }
        contacts = (
            (1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
        )

        locked, diagnostics = apply_contact_aware_foot_locking(foot_samples, contacts)

        self.assertEqual(locked["left"][0], (0.0, 0.0, 0.0))
        self.assertEqual(locked["left"][1], (0.0, 0.0, 0.0))
        self.assertEqual(locked["left"][2], foot_samples["left"][2])
        self.assertEqual(locked["right"][2], foot_samples["right"][2])
        self.assertEqual(locked["right"][3], foot_samples["right"][2])
        self.assertEqual(diagnostics["status"], "applied")
        self.assertLess(diagnostics["post"]["summary"]["max_slide"], diagnostics["pre"]["summary"]["max_slide"])

    def test_leg_basis_child_roles_use_knee_to_foot_axis_without_affecting_arms(self):
        self.assertEqual(_target_basis_child_role("left_lower_leg"), "left_foot")
        self.assertEqual(_target_basis_child_role("right_lower_leg"), "right_foot")
        self.assertEqual(_target_basis_child_role("left_upper_leg"), "left_lower_leg")
        self.assertEqual(_target_basis_child_role("left_upper_arm"), "left_forearm")

    def test_calibrated_motion_quality_filter_reduces_hand_forearm_frame_jumps_without_root_translation_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            rig_path = Path(tmp) / "contract_avatar.gltf"
            bvh_path = Path(tmp) / "jumpy_contract_motion.bvh"
            self._write_contract_rig(rig_path)
            self._write_contract_sidecar(rig_path, basis=True)
            self._write_motion_quality_contract_bvh(
                bvh_path,
                right_forearm_x_degrees=(0.0, 10.0, 120.0, 20.0),
                right_hand_x_degrees=(0.0, 8.0, 115.0, 16.0),
                root_x_positions=(0.0, 1.0, 2.0, 3.0),
            )
            rig = inspect_rig(rig_path)
            motion = parse_bvh(bvh_path)

        clip = build_retarget_clip(motion, rig)

        forearm = next(channel for channel in clip.channels if channel.target_joint == "contract_right_forearm" and channel.path == "rotation")
        hand = next(channel for channel in clip.channels if channel.target_joint == "contract_right_hand" and channel.path == "rotation")
        translation = next(channel for channel in clip.channels if channel.target_joint == "contract_hips" and channel.path == "translation")
        self.assertLessEqual(max(self._frame_jump_degrees(forearm.values)), 16.0)
        self.assertLessEqual(max(self._frame_jump_degrees(hand.values)), 12.0)
        self.assertEqual(translation.values, ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (3.0, 0.0, 0.0)))
        self.assertNotIn("motion_quality_filter", translation.diagnostics)

    def test_motion_quality_diagnostics_report_per_channel_filter_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            rig_path = Path(tmp) / "contract_avatar.gltf"
            bvh_path = Path(tmp) / "diagnostic_contract_motion.bvh"
            self._write_contract_rig(rig_path)
            self._write_contract_sidecar(rig_path, basis=True)
            self._write_motion_quality_contract_bvh(
                bvh_path,
                right_forearm_x_degrees=(0.0, 18.0, 150.0, 24.0),
                right_hand_x_degrees=(0.0, 12.0, 145.0, 18.0),
                left_upper_arm_x_degrees=(0.0, 8.0, 16.0, 20.0),
            )
            rig = inspect_rig(rig_path)
            motion = parse_bvh(bvh_path)

        clip = build_retarget_clip(motion, rig)

        diagnostics = clip.calibration["motion_quality_diagnostics"]
        self.assertEqual(clip.calibration["motion_quality_filter"], "role_aware_conservative")
        self.assertEqual(diagnostics["motion_quality_filter"], "role_aware_conservative")
        channels = {entry["target_joint"]: entry for entry in diagnostics["channels"]}
        forearm = channels["contract_right_forearm"]
        self.assertEqual(forearm["role"], "right_forearm")
        self.assertEqual(forearm["filter_profile"], "role_aware_conservative")
        self.assertEqual(forearm["max_angle_degrees"], 75.0)
        self.assertEqual(forearm["max_frame_jump_degrees"], 16.0)
        self.assertTrue(forearm["filter_changed_samples"])
        self.assertGreater(forearm["changed_sample_count"], 0)
        self.assertGreater(forearm["clamp_count"], 0)
        self.assertGreater(forearm["jump_reduction_count"], 0)
        self.assertGreater(forearm["max_pre_amplitude_degrees"], forearm["max_post_amplitude_degrees"])
        self.assertGreater(forearm["max_pre_frame_jump_degrees"], forearm["max_post_frame_jump_degrees"])

    def test_motion_quality_diagnostics_show_symmetric_left_right_arm_filter_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            rig_path = Path(tmp) / "canonical_unirig_shoulders.gltf"
            bvh_path = Path(tmp) / "symmetric_arm_motion.bvh"
            self._write_canonical_unirig_shoulder_contract_rig_and_sidecar(rig_path)
            self._write_canonical_node_rest_transforms(rig_path, left_upper_arm_translation=[1.0, 0.0, 0.0])
            self._write_shoulder_contract_bvh(
                bvh_path,
                left_shoulder_x_degrees=100.0,
                right_shoulder_x_degrees=100.0,
                left_upper_arm_x_degrees=150.0,
                right_upper_arm_x_degrees=150.0,
                left_forearm_x_degrees=150.0,
                right_forearm_x_degrees=150.0,
                left_hand_x_degrees=145.0,
                right_hand_x_degrees=145.0,
            )
            rig = inspect_rig(rig_path)
            motion = parse_bvh(bvh_path)

        clip = build_retarget_clip(motion, rig)

        channels = {entry["role"]: entry for entry in clip.calibration["motion_quality_diagnostics"]["channels"]}
        for suffix in ("shoulder", "upper_arm", "forearm", "hand"):
            with self.subTest(suffix=suffix):
                left = channels[f"left_{suffix}"]
                right = channels[f"right_{suffix}"]
                self.assertEqual(left["filter_profile"], right["filter_profile"])
                self.assertEqual(left["max_angle_degrees"], right["max_angle_degrees"])
                self.assertEqual(left["max_frame_jump_degrees"], right["max_frame_jump_degrees"])
                self.assertEqual(left["clamp_count"], right["clamp_count"])

    def test_motion_quality_diagnostics_include_torso_legs_feet_without_filtering_translation(self):
        with tempfile.TemporaryDirectory() as tmp:
            rig_path = Path(tmp) / "full_body_diagnostic_avatar.gltf"
            bvh_path = Path(tmp) / "full_body_diagnostic_motion.bvh"
            self._write_contract_rig(rig_path)
            self._write_contract_sidecar(rig_path, basis=True)
            self._write_motion_quality_contract_bvh(
                bvh_path,
                hips_x_degrees=(0.0, 80.0, 10.0),
                spine_x_degrees=(0.0, 60.0, 8.0),
                chest_x_degrees=(0.0, 80.0, 10.0),
                neck_x_degrees=(0.0, 60.0, 8.0),
                head_x_degrees=(0.0, 80.0, 10.0),
                left_upper_leg_x_degrees=(0.0, 140.0, 15.0),
                right_upper_leg_x_degrees=(0.0, 140.0, 15.0),
                left_lower_leg_x_degrees=(0.0, 150.0, 15.0),
                right_lower_leg_x_degrees=(0.0, 150.0, 15.0),
                left_foot_x_degrees=(0.0, 95.0, 10.0),
                right_foot_x_degrees=(0.0, 95.0, 10.0),
                root_x_positions=(0.0, 1.0, 2.0),
            )
            rig = inspect_rig(rig_path)
            motion = parse_bvh(bvh_path)

        clip = build_retarget_clip(motion, rig)

        diagnostics = clip.calibration["motion_quality_diagnostics"]
        roles = {entry["role"]: entry for entry in diagnostics["channels"]}
        for role in ("hips", "spine", "chest", "neck", "head", "left_upper_leg", "right_upper_leg", "left_lower_leg", "right_lower_leg", "left_foot", "right_foot"):
            with self.subTest(role=role):
                self.assertIn(role, roles)
                self.assertEqual(roles[role]["filter_profile"], "role_aware_conservative")
                self.assertGreater(roles[role]["changed_sample_count"], 0)
        self.assertIn("upper_leg", diagnostics["left_right_summary"])
        self.assertIn("lower_leg", diagnostics["left_right_summary"])
        self.assertIn("foot", diagnostics["left_right_summary"])
        translation = next(channel for channel in clip.channels if channel.path == "translation")
        self.assertNotIn("motion_quality_filter", translation.diagnostics)
        self.assertTrue(all(entry["path"] == "rotation" for entry in diagnostics["channels"]))

    def test_motion_quality_diagnostics_do_not_mark_translation_channels_filtered(self):
        with tempfile.TemporaryDirectory() as tmp:
            rig_path = Path(tmp) / "contract_avatar.gltf"
            bvh_path = Path(tmp) / "translation_diagnostic_motion.bvh"
            self._write_contract_rig(rig_path)
            self._write_contract_sidecar(rig_path, basis=True)
            self._write_motion_quality_contract_bvh(
                bvh_path,
                right_forearm_x_degrees=(0.0, 10.0, 120.0, 20.0),
                root_x_positions=(0.0, 1.0, 2.0, 3.0),
            )
            rig = inspect_rig(rig_path)
            motion = parse_bvh(bvh_path)

        clip = build_retarget_clip(motion, rig)

        translation = next(channel for channel in clip.channels if channel.path == "translation")
        diagnostics = clip.calibration["motion_quality_diagnostics"]
        self.assertNotIn("motion_quality_filter", translation.diagnostics)
        self.assertTrue(all(entry["path"] == "rotation" for entry in diagnostics["channels"]))

    def test_motion_quality_diagnostics_survive_success_metadata_integration(self):
        with tempfile.TemporaryDirectory() as tmp:
            rig_path = Path(tmp) / "contract_avatar.gltf"
            bvh_path = Path(tmp) / "metadata_diagnostic_motion.bvh"
            output_path = Path(tmp) / "animated.glb"
            self._write_contract_rig(rig_path)
            self._write_contract_sidecar(rig_path, basis=True)
            self._write_motion_quality_contract_bvh(
                bvh_path,
                right_forearm_x_degrees=(0.0, 18.0, 150.0, 24.0),
                right_hand_x_degrees=(0.0, 12.0, 145.0, 18.0),
            )
            rig = inspect_rig(rig_path)
            motion = parse_bvh(bvh_path)

        clip = build_retarget_clip(motion, rig)
        metadata = retarget_success_metadata(output_path, rig=rig, clip=clip, warnings=clip.warnings)

        diagnostics = metadata["diagnostics"]["retarget_calibration"]["motion_quality_diagnostics"]
        top_level = metadata["retarget_calibration"]["motion_quality_diagnostics"]
        self.assertEqual(diagnostics["motion_quality_filter"], "role_aware_conservative")
        self.assertEqual(top_level["motion_quality_filter"], "role_aware_conservative")
        self.assertGreater(len(diagnostics["channels"]), 0)

    def test_locomotion_policy_survives_success_metadata_integration(self):
        with tempfile.TemporaryDirectory() as tmp:
            bvh_path = Path(tmp) / "walking_child_hips.bvh"
            rig_path = Path(tmp) / "anonymous_unirig.gltf"
            output_path = Path(tmp) / "animated.glb"
            self._write_static_root_with_hips_position_bvh(
                bvh_path,
                first_hips=(0.0, 0.0, 0.0),
                second_hips=(0.0, 0.0, 650.0),
            )
            self._write_anonymous_unirig_46(rig_path)
            motion = parse_bvh(bvh_path)
            rig = inspect_rig(rig_path)
            context = MotionContext(
                path=Path(tmp) / "motion.npz",
                root_positions=None,
                smooth_root_pos=((1.0, 0.5, 2.0), (1.8, 0.7, 2.1)),
                foot_contacts=((1.0, 0.0, 1.0, 0.0, 1.0, 0.0), (0.0, 1.0, 0.0, 1.0, 0.0, 1.0)),
                source_keys=("smooth_root_pos", "foot_contacts"),
                warnings=(),
            )

        clip = build_retarget_clip(motion, rig, motion_context=context)
        metadata = retarget_success_metadata(output_path, rig=rig, clip=clip, warnings=clip.warnings)

        policy = metadata["diagnostics"]["retarget_calibration"]["locomotion_policy"]
        self.assertEqual(policy["mode"], "preserve_scaled_npz_root_motion")
        self.assertEqual(policy["source"], "smooth_root_pos")
        self.assertEqual(policy["foot_contact_shape"], [2, 6])
        contact_locking = metadata["diagnostics"]["retarget_calibration"]["contact_aware_foot_locking"]
        self.assertEqual(contact_locking["status"], "metadata_only")
        self.assertEqual(contact_locking["contact_shape"], [2, 6])
        self.assertEqual(contact_locking["left_windows"], [[0, 0]])
        self.assertEqual(contact_locking["right_windows"], [[1, 1]])
        self.assertIn("world_space_leg_ik", contact_locking["deferred"])
        self.assertIsNone(metadata["retarget_calibration"])

    def test_five_motion_npz_contexts_expose_required_keys_and_locomotion_evidence(self):
        contexts = {}
        for label, workflow_dir in FIVE_MOTION_WORKFLOWS.items():
            npz_path = workflow_dir / "motion.npz"
            self.assertTrue(npz_path.is_file(), f"missing fixture motion.npz for {label}")
            with np.load(npz_path) as payload:
                self.assertIn("foot_contacts", payload.files)
                self.assertIn("root_positions", payload.files)
                self.assertIn("smooth_root_pos", payload.files)
            contexts[label] = load_motion_context(npz_path)

        walking_policy = contexts["walking"].locomotion_policy(frame_count=contexts["walking"].root_frame_count)
        dancing_policy = contexts["dancing"].locomotion_policy(frame_count=contexts["dancing"].root_frame_count)
        self.assertTrue(walking_policy["detected_locomotion"])
        self.assertEqual(walking_policy["source"], "smooth_root_pos")
        self.assertEqual(dancing_policy["foot_contact_shape"], list(contexts["dancing"].foot_contact_shape))
        self.assertGreaterEqual(dancing_policy["foot_contact_ratio"], 0.0)

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

    def test_humanoid_contract_accepts_canonical_unirig_required_roles_nodes_and_node_chains(self):
        with tempfile.TemporaryDirectory() as tmp:
            rig_path = Path(tmp) / "canonical_unirig.gltf"
            self._write_canonical_unirig_contract_rig_and_sidecar(rig_path)

            rig = inspect_rig(rig_path)

        bindings = {binding.role: binding for binding in rig.humanoid_roles}
        chains = {chain.name: chain.roles for chain in rig.humanoid_chains}
        self.assertEqual(rig.humanoid_contract_schema, "modly.humanoid.v1")
        self.assertIsNotNone(rig.humanoid_contract_path)
        self.assertEqual(bindings["hips"].node_name, "bone_0")
        self.assertEqual(bindings["chest"].node_index, 3)
        self.assertEqual(chains["torso"], ("hips", "spine", "chest", "neck", "head"))
        self.assertEqual(chains["left_arm"], ("left_upper_arm", "left_forearm", "left_hand"))
        self.assertEqual(rig.humanoid_diagnostics, ("humanoid_contract_validated",))

    def test_humanoid_contract_parses_optional_shoulder_roles_and_proximal_chains(self):
        with tempfile.TemporaryDirectory() as tmp:
            rig_path = Path(tmp) / "canonical_unirig_shoulders.gltf"
            self._write_canonical_unirig_shoulder_contract_rig_and_sidecar(rig_path)

            rig = inspect_rig(rig_path)

        bindings = {binding.role: binding for binding in rig.humanoid_roles}
        chains = {chain.name: chain.roles for chain in rig.humanoid_chains}
        self.assertEqual(bindings["left_shoulder"].node_name, "bone_6")
        self.assertEqual(bindings["right_shoulder"].node_name, "bone_25")
        self.assertEqual(bindings["left_upper_arm"].node_name, "bone_7")
        self.assertEqual(chains["left_arm"], ("left_upper_arm", "left_forearm", "left_hand"))
        self.assertEqual(chains["left_arm_proximal"], ("left_shoulder", "left_upper_arm", "left_forearm", "left_hand"))
        self.assertEqual(chains["right_arm_proximal"], ("right_shoulder", "right_upper_arm", "right_forearm", "right_hand"))

    def test_humanoid_contract_optional_shoulder_unknown_node_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            rig_path = Path(tmp) / "canonical_unirig_shoulders.gltf"
            self._write_canonical_unirig_shoulder_contract_rig_and_sidecar(rig_path)
            payload = json.loads(rig_path.with_suffix(".rigmeta.json").read_text(encoding="utf-8"))
            payload["humanoid_contract"]["optional_roles"]["left_shoulder"] = "bone_missing"
            rig_path.with_suffix(".rigmeta.json").write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaises(RetargetError) as ctx:
                inspect_rig(rig_path)

        self.assertEqual(ctx.exception.code, MAPPING_INCOMPATIBLE)
        self.assertIn("humanoid_contract_role_unknown_node_id:left_shoulder:bone_missing", ctx.exception.message)

    def test_calibrated_shoulder_source_maps_to_optional_proximal_target_channel(self):
        with tempfile.TemporaryDirectory() as tmp:
            rig_path = Path(tmp) / "canonical_unirig_shoulders.gltf"
            bvh_path = Path(tmp) / "contract_shoulder_motion.bvh"
            self._write_canonical_unirig_shoulder_contract_rig_and_sidecar(rig_path)
            self._write_canonical_node_rest_transforms(rig_path, left_upper_arm_translation=[1.0, 0.0, 0.0])
            self._write_shoulder_contract_bvh(bvh_path, left_shoulder_x_degrees=45.0, left_upper_arm_x_degrees=10.0)
            rig = inspect_rig(rig_path)
            motion = parse_bvh(bvh_path)

        clip = build_retarget_clip(motion, rig)

        self.assertEqual(clip.mapping.pairs["LeftShoulder"], "bone_6")
        self.assertEqual(clip.mapping.pairs["LeftUpperArm"], "bone_7")
        self.assertEqual(clip.calibration["chain_coverage"]["left_arm_proximal"], "4/4")
        shoulder_channel = next(channel for channel in clip.channels if channel.target_joint == "bone_6" and channel.path == "rotation")
        upper_arm_channel = next(channel for channel in clip.channels if channel.target_joint == "bone_7" and channel.path == "rotation")
        self.assertEqual(shoulder_channel.diagnostics["role"], "left_shoulder")
        self.assertNotEqual(shoulder_channel.values[0], shoulder_channel.values[1])
        self.assertNotEqual(upper_arm_channel.values[0], upper_arm_channel.values[1])

    def test_optional_shoulder_missing_source_warns_without_blocking_calibrated_body(self):
        with tempfile.TemporaryDirectory() as tmp:
            rig_path = Path(tmp) / "canonical_unirig_shoulders.gltf"
            bvh_path = Path(tmp) / "contract_motion.bvh"
            self._write_canonical_unirig_shoulder_contract_rig_and_sidecar(rig_path)
            self._write_canonical_node_rest_transforms(rig_path, left_upper_arm_translation=[1.0, 0.0, 0.0])
            self._write_full_contract_bvh(bvh_path)
            rig = inspect_rig(rig_path)
            motion = parse_bvh(bvh_path)

        clip = build_retarget_clip(motion, rig)

        self.assertEqual(clip.mapping.confidence, "calibrated")
        self.assertNotIn("LeftShoulder", clip.mapping.pairs)
        warnings = clip.calibration["warnings_by_severity"]["warning"]
        self.assertIn("optional_missing_source_role:left_arm_proximal:left_shoulder", warnings)

    def test_canonical_unirig_node_rest_transforms_feed_calibrated_target_basis(self):
        with tempfile.TemporaryDirectory() as tmp:
            rig_path = Path(tmp) / "canonical_unirig.gltf"
            bvh_path = Path(tmp) / "contract_motion.bvh"
            self._write_canonical_unirig_contract_rig_and_sidecar(rig_path)
            self._write_canonical_node_rest_transforms(rig_path, left_upper_arm_translation=[1.0, 0.0, 0.0])
            self._write_full_contract_bvh(bvh_path, left_upper_arm_x_degrees=90.0)
            rig = inspect_rig(rig_path)
            motion = parse_bvh(bvh_path)

        clip = build_retarget_clip(motion, rig)

        basis = clip.calibration["basis_sources"]["bone_6"]
        self.assertEqual(rig.humanoid_rest["source"], "humanoid_contract.nodes.transforms")
        self.assertEqual(basis["target"], "target_rest_matrix:rest_local:left_upper_arm")
        self.assertTupleAlmostEqual(tuple(basis["target_quaternion"]), (0.0, 0.0, 0.0, 1.0))
        channel = next(channel for channel in clip.channels if channel.target_joint == "bone_6" and channel.path == "rotation")
        self.assertTupleAlmostEqual(channel.values[0], (0.0, 0.0, 0.0, 1.0))

    def test_canonical_unirig_without_rest_transforms_or_basis_still_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            rig_path = Path(tmp) / "canonical_unirig.gltf"
            bvh_path = Path(tmp) / "contract_motion.bvh"
            self._write_canonical_unirig_contract_rig_and_sidecar(rig_path)
            self._write_full_contract_bvh(bvh_path)
            rig = inspect_rig(rig_path)
            motion = parse_bvh(bvh_path)

        with self.assertRaises(RetargetError) as ctx:
            build_retarget_clip(motion, rig)

        self.assertEqual(ctx.exception.code, CALIBRATION_UNAVAILABLE)
        self.assertIn("requires target rest transforms or basis hints", ctx.exception.message)

    def test_missing_source_neck_bridges_known_numbered_neck_chain_with_warning(self):
        with tempfile.TemporaryDirectory() as tmp:
            rig_path = Path(tmp) / "contract_avatar.gltf"
            bvh_path = Path(tmp) / "numbered_neck_motion.bvh"
            self._write_contract_rig(rig_path)
            self._write_contract_sidecar(rig_path, basis=True)
            self._write_numbered_neck_contract_bvh(bvh_path)
            rig = inspect_rig(rig_path)
            motion = parse_bvh(bvh_path)

        clip = build_retarget_clip(motion, rig)

        self.assertEqual(clip.mapping.pairs["Neck1"], "contract_neck")
        self.assertIn("source_neck_role_bridged:Neck1", clip.mapping.warnings)
        self.assertIn("source_neck_role_bridged:Neck1", clip.calibration["warnings_by_severity"]["warning"])

    def test_missing_source_neck_still_fails_without_head_or_numbered_neck_chain(self):
        with tempfile.TemporaryDirectory() as tmp:
            rig_path = Path(tmp) / "contract_avatar.gltf"
            bvh_path = Path(tmp) / "no_safe_neck_motion.bvh"
            self._write_contract_rig(rig_path)
            self._write_contract_sidecar(rig_path, basis=True)
            self._write_no_safe_neck_contract_bvh(bvh_path)
            rig = inspect_rig(rig_path)
            motion = parse_bvh(bvh_path)

        with self.assertRaises(RetargetError) as ctx:
            build_retarget_clip(motion, rig)

        self.assertEqual(ctx.exception.code, CALIBRATION_UNAVAILABLE)
        self.assertIn("missing_source_role:torso:neck", ctx.exception.message)
        self.assertNotIn("source_neck_role_bridged", ctx.exception.message)

    def test_humanoid_contract_canonical_bad_node_reference_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            rig_path = Path(tmp) / "canonical_unirig.gltf"
            self._write_canonical_unirig_contract_rig_and_sidecar(rig_path)
            payload = json.loads(rig_path.with_suffix(".rigmeta.json").read_text(encoding="utf-8"))
            payload["humanoid_contract"]["required_roles"]["head"] = "bone_missing"
            rig_path.with_suffix(".rigmeta.json").write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaises(RetargetError) as ctx:
                inspect_rig(rig_path)

        self.assertEqual(ctx.exception.code, MAPPING_INCOMPATIBLE)
        self.assertIn("humanoid_contract_role_unknown_node_id:head:bone_missing", ctx.exception.message)

    def test_humanoid_contract_legacy_roles_shape_remains_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            rig_path = Path(tmp) / "contract_avatar.gltf"
            self._write_contract_rig(rig_path)
            self._write_contract_sidecar(rig_path)

            rig = inspect_rig(rig_path)

        self.assertEqual(rig.humanoid_contract_schema, "modly.humanoid.v1")
        self.assertEqual({binding.role for binding in rig.humanoid_roles}, set(self._contract_roles()))

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

    def _write_motion_quality_contract_bvh(
        self,
        path: Path,
        *,
        right_forearm_x_degrees: tuple[float, ...] = (0.0,),
        right_hand_x_degrees: tuple[float, ...] = (0.0,),
        left_upper_arm_x_degrees: tuple[float, ...] = (0.0,),
        hips_x_degrees: tuple[float, ...] = (0.0,),
        spine_x_degrees: tuple[float, ...] = (0.0,),
        chest_x_degrees: tuple[float, ...] = (0.0,),
        neck_x_degrees: tuple[float, ...] = (0.0,),
        head_x_degrees: tuple[float, ...] = (0.0,),
        left_upper_leg_x_degrees: tuple[float, ...] = (0.0,),
        right_upper_leg_x_degrees: tuple[float, ...] = (0.0,),
        left_lower_leg_x_degrees: tuple[float, ...] = (0.0,),
        right_lower_leg_x_degrees: tuple[float, ...] = (0.0,),
        left_foot_x_degrees: tuple[float, ...] = (0.0,),
        right_foot_x_degrees: tuple[float, ...] = (0.0,),
        root_x_positions: tuple[float, ...] = (0.0,),
    ) -> None:
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
"""
        frame_count = max(
            len(right_forearm_x_degrees),
            len(right_hand_x_degrees),
            len(left_upper_arm_x_degrees),
            len(hips_x_degrees),
            len(spine_x_degrees),
            len(chest_x_degrees),
            len(neck_x_degrees),
            len(head_x_degrees),
            len(left_upper_leg_x_degrees),
            len(right_upper_leg_x_degrees),
            len(left_lower_leg_x_degrees),
            len(right_lower_leg_x_degrees),
            len(left_foot_x_degrees),
            len(right_foot_x_degrees),
            len(root_x_positions),
        )
        frames = []
        for index in range(frame_count):
            frame = [0.0] * 54
            frame[0] = root_x_positions[min(index, len(root_x_positions) - 1)]
            frame[4] = hips_x_degrees[min(index, len(hips_x_degrees) - 1)]
            frame[7] = spine_x_degrees[min(index, len(spine_x_degrees) - 1)]
            frame[10] = chest_x_degrees[min(index, len(chest_x_degrees) - 1)]
            frame[13] = neck_x_degrees[min(index, len(neck_x_degrees) - 1)]
            frame[16] = head_x_degrees[min(index, len(head_x_degrees) - 1)]
            frame[19] = left_upper_arm_x_degrees[min(index, len(left_upper_arm_x_degrees) - 1)]
            frame[31] = right_forearm_x_degrees[min(index, len(right_forearm_x_degrees) - 1)]
            frame[34] = right_hand_x_degrees[min(index, len(right_hand_x_degrees) - 1)]
            frame[37] = left_upper_leg_x_degrees[min(index, len(left_upper_leg_x_degrees) - 1)]
            frame[40] = left_lower_leg_x_degrees[min(index, len(left_lower_leg_x_degrees) - 1)]
            frame[43] = left_foot_x_degrees[min(index, len(left_foot_x_degrees) - 1)]
            frame[46] = right_upper_leg_x_degrees[min(index, len(right_upper_leg_x_degrees) - 1)]
            frame[49] = right_lower_leg_x_degrees[min(index, len(right_lower_leg_x_degrees) - 1)]
            frame[52] = right_foot_x_degrees[min(index, len(right_foot_x_degrees) - 1)]
            frames.append(" ".join(str(value) for value in frame))
        path.write_text(f"{hierarchy}Frames: {frame_count}\nFrame Time: 0.0333333\n" + "\n".join(frames) + "\n", encoding="utf-8")

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

    def _write_canonical_unirig_contract_rig_and_sidecar(self, path: Path) -> None:
        role_nodes = {
            "hips": "bone_0",
            "spine": "bone_1",
            "chest": "bone_3",
            "neck": "bone_4",
            "head": "bone_5",
            "left_upper_arm": "bone_6",
            "left_forearm": "bone_7",
            "left_hand": "bone_9",
            "right_upper_arm": "bone_10",
            "right_forearm": "bone_11",
            "right_hand": "bone_13",
            "left_upper_leg": "bone_14",
            "left_lower_leg": "bone_15",
            "left_foot": "bone_17",
            "right_upper_leg": "bone_18",
            "right_lower_leg": "bone_19",
            "right_foot": "bone_21",
        }
        nodes = []
        contract_nodes = {}
        for index in range(22):
            name = f"bone_{index}"
            nodes.append({"name": name})
            contract_nodes[name] = {
                "index": index,
                "name": name,
                "parent": None,
                "children": [],
                "transforms": {"rest_local": {}, "rest_world": {}},
            }
        path.write_text(
            json.dumps({
                "asset": {"version": "2.0"},
                "nodes": nodes,
                "skins": [{"joints": list(range(22)), "inverseBindMatrices": 0}],
            }),
            encoding="utf-8",
        )
        contract = {
            "schema": "modly.humanoid.v1",
            "required_roles": role_nodes,
            "nodes": contract_nodes,
            "chains": {
                "torso": ["bone_0", "bone_1", "bone_2", "bone_3", "bone_4", "bone_5"],
                "left_arm": ["bone_6", "bone_7", "bone_8", "bone_9"],
                "right_arm": ["bone_10", "bone_11", "bone_12", "bone_13"],
                "left_leg": ["bone_14", "bone_15", "bone_16", "bone_17"],
                "right_leg": ["bone_18", "bone_19", "bone_20", "bone_21"],
            },
            "quality": ["canonical_unirig_fixture"],
        }
        path.with_suffix(".rigmeta.json").write_text(json.dumps({"humanoid_contract": contract}), encoding="utf-8")

    def _write_canonical_unirig_shoulder_contract_rig_and_sidecar(self, path: Path) -> None:
        required_role_nodes = {
            "hips": "bone_0",
            "spine": "bone_1",
            "chest": "bone_3",
            "neck": "bone_4",
            "head": "bone_5",
            "left_upper_arm": "bone_7",
            "left_lower_arm": "bone_8",
            "left_hand": "bone_9",
            "right_upper_arm": "bone_26",
            "right_lower_arm": "bone_27",
            "right_hand": "bone_28",
            "left_upper_leg": "bone_14",
            "left_lower_leg": "bone_15",
            "left_foot": "bone_17",
            "right_upper_leg": "bone_18",
            "right_lower_leg": "bone_19",
            "right_foot": "bone_21",
        }
        optional_role_nodes = {"left_shoulder": "bone_6", "right_shoulder": "bone_25"}
        parents = {
            "bone_0": None,
            "bone_1": "bone_0",
            "bone_3": "bone_1",
            "bone_4": "bone_3",
            "bone_5": "bone_4",
            "bone_6": "bone_3",
            "bone_7": "bone_6",
            "bone_8": "bone_7",
            "bone_9": "bone_8",
            "bone_25": "bone_3",
            "bone_26": "bone_25",
            "bone_27": "bone_26",
            "bone_28": "bone_27",
            "bone_14": "bone_0",
            "bone_15": "bone_14",
            "bone_17": "bone_15",
            "bone_18": "bone_0",
            "bone_19": "bone_18",
            "bone_21": "bone_19",
        }
        nodes = []
        contract_nodes = {}
        for index in range(29):
            name = f"bone_{index}"
            nodes.append({"name": name})
            contract_nodes[name] = {
                "index": index,
                "name": name,
                "parent": parents.get(name),
                "children": [],
                "transforms": {"rest_local": {}, "rest_world": {}},
            }
        for node_id, parent in parents.items():
            if parent is not None:
                contract_nodes[parent]["children"].append(node_id)
        path.write_text(
            json.dumps({
                "asset": {"version": "2.0"},
                "nodes": nodes,
                "skins": [{"joints": list(range(29)), "inverseBindMatrices": 0}],
            }),
            encoding="utf-8",
        )
        contract = {
            "schema": "modly.humanoid.v1",
            "required_roles": required_role_nodes,
            "optional_roles": optional_role_nodes,
            "nodes": contract_nodes,
            "chains": {
                "spine": ["bone_0", "bone_1", "bone_3", "bone_4", "bone_5"],
                "left_arm": ["bone_7", "bone_8", "bone_9"],
                "right_arm": ["bone_26", "bone_27", "bone_28"],
                "left_arm_proximal": ["bone_6", "bone_7", "bone_8", "bone_9"],
                "right_arm_proximal": ["bone_25", "bone_26", "bone_27", "bone_28"],
                "left_leg": ["bone_14", "bone_15", "bone_17"],
                "right_leg": ["bone_18", "bone_19", "bone_21"],
            },
            "quality": ["canonical_unirig_shoulder_fixture"],
        }
        path.with_suffix(".rigmeta.json").write_text(json.dumps({"humanoid_contract": contract}), encoding="utf-8")

    def _write_shoulder_contract_bvh(
        self,
        path: Path,
        *,
        left_shoulder_x_degrees: float = 0.0,
        right_shoulder_x_degrees: float = 0.0,
        left_upper_arm_x_degrees: float = 0.0,
        right_upper_arm_x_degrees: float = 0.0,
        left_forearm_x_degrees: float = 0.0,
        right_forearm_x_degrees: float = 0.0,
        left_hand_x_degrees: float = 0.0,
        right_hand_x_degrees: float = 0.0,
    ) -> None:
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
      JOINT LeftShoulder
      {
        OFFSET -0.5 0.25 0.0
        CHANNELS 3 Zrotation Xrotation Yrotation
        JOINT LeftUpperArm
        {
          OFFSET -0.5 0.25 0.0
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
      }
      JOINT RightShoulder
      {
        OFFSET 0.5 0.25 0.0
        CHANNELS 3 Zrotation Xrotation Yrotation
        JOINT RightUpperArm
        {
          OFFSET 0.5 0.25 0.0
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
        first_frame = [0.0] * 60
        second_frame = [0.0] * 60
        second_frame[0] = 1.0
        second_frame[19] = left_shoulder_x_degrees
        second_frame[22] = left_upper_arm_x_degrees
        second_frame[25] = left_forearm_x_degrees
        second_frame[28] = left_hand_x_degrees
        second_frame[31] = right_shoulder_x_degrees
        second_frame[34] = right_upper_arm_x_degrees
        second_frame[37] = right_forearm_x_degrees
        second_frame[40] = right_hand_x_degrees
        path.write_text(
            hierarchy
            + " ".join(str(value) for value in first_frame)
            + "\n"
            + " ".join(str(value) for value in second_frame)
            + "\n",
            encoding="utf-8",
        )

    def _write_canonical_node_rest_transforms(self, path: Path, *, left_upper_arm_translation: list[float]) -> None:
        payload = json.loads(path.with_suffix(".rigmeta.json").read_text(encoding="utf-8"))
        role_nodes = payload["humanoid_contract"]["required_roles"]
        for role, node_id in role_nodes.items():
            translation = left_upper_arm_translation if role in {"left_upper_arm", "left_forearm"} else [0.0, 1.0, 0.0]
            payload["humanoid_contract"]["nodes"][node_id]["transforms"] = {
                "matrix_order": "row-major",
                "rest_local": self._translation_matrix(translation),
                "rest_world": self._translation_matrix(translation),
            }
        path.with_suffix(".rigmeta.json").write_text(json.dumps(payload), encoding="utf-8")

    def _replace_contract_rest_roles(self, path: Path, role_translations: dict[str, list[float]]) -> None:
        payload = json.loads(path.with_suffix(".rigmeta.json").read_text(encoding="utf-8"))
        payload["humanoid_contract"].pop("basis_hints", None)
        payload["humanoid_contract"]["rest"] = {
            "source": "synthetic_role_translations",
            "roles": {
                role: {"local_translation": translation}
                for role, translation in role_translations.items()
            },
        }
        path.with_suffix(".rigmeta.json").write_text(json.dumps(payload), encoding="utf-8")

    def _replace_contract_rest_role_entries(self, path: Path, role_entries: dict[str, dict]) -> None:
        payload = json.loads(path.with_suffix(".rigmeta.json").read_text(encoding="utf-8"))
        payload["humanoid_contract"].pop("basis_hints", None)
        payload["humanoid_contract"]["rest"] = {
            "source": "synthetic_role_entries",
            "matrix_order": "row-major",
            "roles": role_entries,
        }
        path.with_suffix(".rigmeta.json").write_text(json.dumps(payload), encoding="utf-8")

    def _set_contract_node_rotation(self, path: Path, role: str, rotation: tuple[float, float, float, float]) -> None:
        payload = json.loads(path.read_text(encoding="utf-8"))
        node_name = f"contract_{role}"
        node = next(node for node in payload["nodes"] if node.get("name") == node_name)
        node["rotation"] = list(rotation)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def _translation_matrix(self, translation: list[float]) -> list[list[float]]:
        return [
            [1.0, 0.0, 0.0, translation[0]],
            [0.0, 1.0, 0.0, translation[1]],
            [0.0, 0.0, 1.0, translation[2]],
            [0.0, 0.0, 0.0, 1.0],
        ]

    def _rotation_y_matrix(self, degrees: float, *, translation: list[float] | None = None) -> list[list[float]]:
        radians = degrees * 3.141592653589793 / 180.0
        cosine = round(math.cos(radians), 12)
        sine = round(math.sin(radians), 12)
        translation = translation or [0.0, 0.0, 0.0]
        return [
            [cosine, 0.0, sine, translation[0]],
            [0.0, 1.0, 0.0, translation[1]],
            [-sine, 0.0, cosine, translation[2]],
            [0.0, 0.0, 0.0, 1.0],
        ]

    def _write_numbered_neck_contract_bvh(self, path: Path) -> None:
        self._write_full_contract_bvh(path)
        text = path.read_text(encoding="utf-8")
        path.write_text(text.replace("JOINT Neck", "JOINT Neck1", 1), encoding="utf-8")

    def _write_no_safe_neck_contract_bvh(self, path: Path) -> None:
        self._write_full_contract_bvh(path)
        text = path.read_text(encoding="utf-8")
        text = text.replace("JOINT Neck", "JOINT Collar", 1).replace("JOINT Head", "JOINT Skull", 1)
        path.write_text(text, encoding="utf-8")

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

    def _write_static_root_with_hips_position_bvh(
        self,
        path: Path,
        *,
        first_hips: tuple[float, float, float],
        second_hips: tuple[float, float, float],
        second_hips_y_rotation: float = 0.0,
    ) -> None:
        hierarchy = """HIERARCHY
ROOT Root
{
  OFFSET 0.0 0.0 0.0
  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
  JOINT Hips
  {
    OFFSET 0.0 1.0 0.0
    CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
    JOINT Spine
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
  }
}
MOTION
Frames: 2
Frame Time: 0.0333333
"""
        first_frame = [0.0] * 18
        second_frame = [0.0] * 18
        first_frame[6:9] = list(first_hips)
        second_frame[6:9] = list(second_hips)
        second_frame[11] = second_hips_y_rotation
        path.write_text(
            hierarchy
            + " ".join(str(value) for value in first_frame)
            + "\n"
            + " ".join(str(value) for value in second_frame)
            + "\n",
            encoding="utf-8",
        )

    def _decoded_channel_outputs(self, document, animation, path, resolver):
        channel = next(channel for channel in animation["channels"] if channel["target"]["path"] == path)
        sampler = animation["samplers"][channel["sampler"]]
        expected = {"VEC4"} if path == "rotation" else {"VEC3"}
        return decode_accessor(document, resolver, sampler["output"], expected_types=expected)

    def _quaternion_angle_degrees(self, quaternion):
        _x, _y, _z, w = quaternion
        clamped_w = max(-1.0, min(1.0, abs(w)))
        return math.degrees(2.0 * math.acos(clamped_w))

    def _frame_jump_degrees(self, values):
        return [
            self._quaternion_angle_degrees(self._multiply_quaternion_xyzw(values[index], self._inverse_quaternion_xyzw(values[index - 1])))
            for index in range(1, len(values))
        ]

    def _axis_angle_quaternion_x(self, degrees):
        half = math.radians(degrees) / 2.0
        return (math.sin(half), 0.0, 0.0, math.cos(half))

    def _multiply_quaternion_xyzw(self, left, right):
        lx, ly, lz, lw = left
        rx, ry, rz, rw = right
        return (
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        )

    def _inverse_quaternion_xyzw(self, quaternion):
        x, y, z, w = quaternion
        length_squared = x * x + y * y + z * z + w * w
        return (-x / length_squared, -y / length_squared, -z / length_squared, w / length_squared)

    def assertTupleAlmostEqual(self, actual, expected, places=6):
        self.assertEqual(len(actual), len(expected))
        for actual_value, expected_value in zip(actual, expected):
            self.assertAlmostEqual(actual_value, expected_value, places=places)


if __name__ == "__main__":
    unittest.main()
