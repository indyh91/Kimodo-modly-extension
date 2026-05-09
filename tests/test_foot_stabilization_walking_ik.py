from dataclasses import replace
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from foot_stabilization import stabilize_feet  # noqa: E402
from retargeting import JointMapping, MotionContext, RetargetChannel, RetargetClip  # noqa: E402
from rig_inspector import HumanoidChain, HumanoidRoleBinding, RigInspection  # noqa: E402


def _channel(target, path, values, role):
    return RetargetChannel(
        target_joint=target,
        path=path,
        times=tuple(index / 30.0 for index in range(len(values))),
        values=tuple(values),
        diagnostics={"role": role},
    )


def _clip(*channels):
    root_values = tuple(channels[0].values) if channels else ()
    foot_samples = {
        "status": "success",
        "samples": {
            "left": tuple((value[0] - 0.2, value[1], value[2] + 0.3) for value in root_values),
            "right": tuple((value[0] + 0.2, value[1], value[2] + 0.3) for value in root_values),
        },
    }
    return RetargetClip(
        name="Kimodo Walking",
        duration=0.2,
        frame_time=1.0 / 30.0,
        root_translations=root_values,
        channels=tuple(channels),
        mapping=JointMapping(confidence="calibrated", pairs={}, warnings=()),
        warnings=(),
        calibration={"locomotion_policy": {"detected_locomotion": True}, "pre_export_foot_samples": foot_samples},
        visual_quality={"version": "kimodo.visual_quality.v1", "blocking": False, "warnings": [], "limitations": [], "checks": {}},
    )


def _walking_channels(frame_count=6):
    root = _channel("Hips", "translation", tuple((index * 0.02, 1.0, 0.0) for index in range(frame_count)), "hips")
    left_upper = _channel("LeftUpperLeg", "rotation", ((0.0, 0.0, 0.0, 1.0),) * frame_count, "left_upper_leg")
    left_lower = _channel("LeftLowerLeg", "rotation", ((0.0, 0.0, 0.0, 1.0),) * frame_count, "left_lower_leg")
    right_upper = _channel("RightUpperLeg", "rotation", ((0.0, 0.0, 0.0, 1.0),) * frame_count, "right_upper_leg")
    right_lower = _channel("RightLowerLeg", "rotation", ((0.0, 0.0, 0.0, 1.0),) * frame_count, "right_lower_leg")
    arm = _channel("LeftUpperArm", "rotation", ((0.0, 0.0, 0.0, 1.0),) * frame_count, "left_upper_arm")
    return root, left_upper, left_lower, right_upper, right_lower, arm


def _rig(*, rest=None, basis=None, chains=None):
    roles = (
        HumanoidRoleBinding("hips", 0, "Hips"),
        HumanoidRoleBinding("left_upper_leg", 1, "LeftUpperLeg"),
        HumanoidRoleBinding("left_lower_leg", 2, "LeftLowerLeg"),
        HumanoidRoleBinding("left_foot", 3, "LeftFoot"),
        HumanoidRoleBinding("right_upper_leg", 4, "RightUpperLeg"),
        HumanoidRoleBinding("right_lower_leg", 5, "RightLowerLeg"),
        HumanoidRoleBinding("right_foot", 6, "RightFoot"),
    )
    return RigInspection(
        path=Path("rig.gltf"),
        source_kind="unirig_recommended",
        classification="humanoid",
        joint_names=tuple(role.node_name for role in roles),
        skin_count=1,
        has_inverse_bind_matrices=True,
        humanoid_roles=roles,
        humanoid_chains=chains
        if chains is not None
        else (
            HumanoidChain("left_leg", ("left_upper_leg", "left_lower_leg", "left_foot")),
            HumanoidChain("right_leg", ("right_upper_leg", "right_lower_leg", "right_foot")),
        ),
        humanoid_rest=rest
        if rest is not None
        else {
            "roles": {
                "left_upper_leg": {"local_translation": [0.0, -0.5, 0.0]},
                "left_lower_leg": {"local_translation": [0.0, -0.5, 0.0]},
                "left_foot": {"local_translation": [0.0, -0.1, 0.08]},
                "right_upper_leg": {"local_translation": [0.0, -0.5, 0.0]},
                "right_lower_leg": {"local_translation": [0.0, -0.5, 0.0]},
                "right_foot": {"local_translation": [0.0, -0.1, 0.08]},
            }
        },
        humanoid_basis_hints=basis if basis is not None else {"up_axis": "Y", "knee_plane_normal": [0.0, 0.0, 1.0]},
        contract_status="trusted",
    )


def test_walking_two_bone_ik_applies_only_lower_body_rotations_with_valid_rest_basis_and_aligned_frames():
    root, left_upper, left_lower, right_upper, right_lower, arm = _walking_channels()
    clip = _clip(root, left_upper, left_lower, right_upper, right_lower, arm)
    context = MotionContext(path=None, foot_contacts=((1.0, 0.0),) * 6)

    stabilized, result = stabilize_feet(clip, _rig(), context, motion_class="walking")

    by_role = {channel.diagnostics["role"]: channel for channel in stabilized.channels}
    assert result.status == "applied"
    assert result.mode == "two_bone_ik"
    assert result.corrected_frames == 6
    assert result.max_correction == 0.08
    assert by_role["hips"].values == root.values
    assert by_role["left_upper_leg"].values != left_upper.values
    assert by_role["left_lower_leg"].values != left_lower.values
    assert by_role["right_upper_leg"].values == right_upper.values
    assert by_role["right_lower_leg"].values == right_lower.values
    assert by_role["left_upper_arm"].values == arm.values
    assert stabilized.visual_quality["foot_stabilization"]["status"] == "applied"


def test_walking_two_bone_ik_blends_boundary_rotations_and_preserves_swing_frames():
    root, left_upper, left_lower, right_upper, right_lower, arm = _walking_channels(frame_count=8)
    clip = _clip(root, left_upper, left_lower, right_upper, right_lower, arm)
    context = MotionContext(path=None, foot_contacts=((0.0, 0.0),) + ((1.0, 0.0),) * 6 + ((0.0, 0.0),))

    stabilized, result = stabilize_feet(clip, _rig(), context, motion_class="walking")

    by_role = {channel.diagnostics["role"]: channel for channel in stabilized.channels}
    left_upper_values = by_role["left_upper_leg"].values
    left_lower_values = by_role["left_lower_leg"].values
    assert result.status == "applied"
    assert result.corrected_frames == 6
    assert left_upper_values[0] == left_upper.values[0]
    assert left_lower_values[0] == left_lower.values[0]
    assert left_upper_values[1] == left_upper.values[1]
    assert left_lower_values[1] == left_lower.values[1]
    assert left_upper_values[2] != left_upper.values[2]
    assert left_lower_values[2] != left_lower.values[2]
    assert abs(left_lower_values[2][2]) < abs(left_lower_values[3][2])
    assert left_upper_values[6] != left_upper.values[6]
    assert left_lower_values[6] != left_lower.values[6]
    assert abs(left_lower_values[6][2]) < abs(left_lower_values[5][2])
    assert left_upper_values[7] == left_upper.values[7]
    assert left_lower_values[7] == left_lower.values[7]
    assert by_role["right_upper_leg"].values == right_upper.values
    assert by_role["right_lower_leg"].values == right_lower.values
    assert by_role["left_upper_arm"].values == arm.values


def test_walking_ik_fails_closed_when_rest_or_basis_contains_non_finite_values():
    root, left_upper, left_lower, right_upper, right_lower, arm = _walking_channels()
    clip = _clip(root, left_upper, left_lower, right_upper, right_lower, arm)
    context = MotionContext(path=None, foot_contacts=((1.0, 0.0),) * 6)
    invalid_rest = {"roles": {"left_upper_leg": {"local_translation": [0.0, float("nan"), 0.0]}, "left_lower_leg": {"local_translation": [0.0, -0.5, 0.0]}}}
    invalid_basis = {"up_axis": "Y", "knee_plane_normal": [0.0, float("inf"), 1.0]}

    bad_rest = stabilize_feet(clip, _rig(rest=invalid_rest), context, motion_class="walking")
    bad_basis = stabilize_feet(clip, _rig(basis=invalid_basis), context, motion_class="walking")

    assert bad_rest[1].status == "skipped"
    assert bad_rest[1].skipped_reason == "invalid_ik_rest_basis"
    assert bad_rest[0].channels == clip.channels
    assert bad_basis[1].status == "skipped"
    assert bad_basis[1].skipped_reason == "invalid_ik_rest_basis"
    assert bad_basis[0].channels == clip.channels


def test_walking_ik_rejects_impossible_reach_without_mutation():
    root, left_upper, left_lower, right_upper, right_lower, arm = _walking_channels()
    clip = _clip(root, left_upper, left_lower, right_upper, right_lower, arm)
    context = MotionContext(path=None, foot_contacts=((1.0, 0.0),) * 6)
    short_limb_rest = {
        "roles": {
            "left_upper_leg": {"local_translation": [0.0, -0.03, 0.0]},
            "left_lower_leg": {"local_translation": [0.0, -0.03, 0.0]},
            "right_upper_leg": {"local_translation": [0.0, -0.5, 0.0]},
            "right_lower_leg": {"local_translation": [0.0, -0.5, 0.0]},
        }
    }

    stabilized, result = stabilize_feet(clip, _rig(rest=short_limb_rest), context, motion_class="walking")

    assert result.status == "skipped"
    assert result.skipped_reason == "ik_target_exceeds_limb_reach"
    assert stabilized.channels == clip.channels


def test_walking_ik_rejects_knee_inversion_risk_without_mutation():
    root, left_upper, left_lower, right_upper, right_lower, arm = _walking_channels()
    clip = _clip(root, left_upper, left_lower, right_upper, right_lower, arm)
    context = MotionContext(path=None, foot_contacts=((1.0, 0.0),) * 6)

    stabilized, result = stabilize_feet(clip, _rig(basis={"up_axis": "Y", "knee_plane_normal": [0.0, 0.0, -1.0]}), context, motion_class="walking")

    assert result.status == "skipped"
    assert result.skipped_reason == "knee_inversion_risk"
    assert stabilized.channels == clip.channels


def test_walking_ik_rejects_invalid_chain_length_and_unaligned_rotation_frames():
    root, left_upper, left_lower, right_upper, right_lower, arm = _walking_channels()
    clip = _clip(root, replace(left_upper, values=left_upper.values[:-1]), left_lower, right_upper, right_lower, arm)
    context = MotionContext(path=None, foot_contacts=((1.0, 0.0),) * 6)
    invalid_chain = (HumanoidChain("left_leg", ("left_upper_leg", "left_foot")), HumanoidChain("right_leg", ("right_upper_leg", "right_lower_leg", "right_foot")))

    bad_chain = stabilize_feet(_clip(root, left_upper, left_lower, right_upper, right_lower, arm), _rig(chains=invalid_chain), context, motion_class="walking")
    bad_frames = stabilize_feet(clip, _rig(), context, motion_class="walking")

    assert bad_chain[1].status == "skipped"
    assert bad_chain[1].skipped_reason == "missing_valid_leg_chains"
    assert bad_chain[0].channels == _clip(root, left_upper, left_lower, right_upper, right_lower, arm).channels
    assert bad_frames[1].status == "skipped"
    assert bad_frames[1].skipped_reason == "ik_rotation_frame_count_mismatch"
    assert bad_frames[0].channels == clip.channels
