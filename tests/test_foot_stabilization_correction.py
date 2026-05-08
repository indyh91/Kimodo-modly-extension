from dataclasses import replace
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from foot_stabilization import stabilize_feet  # noqa: E402
from retargeting import JointMapping, MotionContext, RetargetChannel, RetargetClip  # noqa: E402
from rig_inspector import HumanoidChain, HumanoidRoleBinding, RigInspection  # noqa: E402


def _clip_with_channels(*channels):
    return RetargetClip(
        name="Kimodo Motion",
        duration=0.1,
        frame_time=1.0 / 30.0,
        root_translations=tuple(channels[0].values) if channels else (),
        channels=tuple(channels),
        mapping=JointMapping(confidence="calibrated", pairs={}, warnings=()),
        warnings=(),
        visual_quality={"version": "kimodo.visual_quality.v1", "blocking": False, "status": "not_evaluated", "warnings": [], "limitations": [], "checks": {}},
    )


def _channel(target, path, values, role):
    return RetargetChannel(
        target_joint=target,
        path=path,
        times=tuple(index / 30.0 for index in range(len(values))),
        values=tuple(values),
        diagnostics={"role": role},
    )


def _rig(*, calibrated=True):
    roles = (
        HumanoidRoleBinding("hips", 0, "Hips"),
        HumanoidRoleBinding("left_upper_leg", 1, "LeftUpperLeg"),
        HumanoidRoleBinding("left_lower_leg", 2, "LeftLowerLeg"),
        HumanoidRoleBinding("left_foot", 3, "LeftFoot"),
        HumanoidRoleBinding("right_upper_leg", 4, "RightUpperLeg"),
        HumanoidRoleBinding("right_lower_leg", 5, "RightLowerLeg"),
        HumanoidRoleBinding("right_foot", 6, "RightFoot"),
    )
    chains = (
        HumanoidChain("left_leg", ("left_upper_leg", "left_lower_leg", "left_foot")),
        HumanoidChain("right_leg", ("right_upper_leg", "right_lower_leg", "right_foot")),
    )
    return RigInspection(
        path=Path("rig.gltf"),
        source_kind="unirig_recommended",
        classification="humanoid",
        joint_names=tuple(role.node_name for role in roles),
        skin_count=1,
        has_inverse_bind_matrices=True,
        humanoid_roles=roles if calibrated else (),
        humanoid_chains=chains if calibrated else (),
        humanoid_rest={"roles": {role.role: {"local_translation": [0.0, -1.0, 0.0]} for role in roles}},
        humanoid_basis_hints={"up_axis": "Y"},
    )


def test_static_double_support_applies_horizontal_root_compensation_only_inside_window():
    root = _channel("Hips", "translation", ((0.0, 1.0, 0.0), (0.03, 1.0, 0.0), (0.06, 1.0, 0.0), (0.09, 1.0, 0.0), (0.50, 1.0, 0.0)), "hips")
    arm = _channel("LeftUpperArm", "rotation", ((0.0, 0.0, 0.0, 1.0),) * 5, "left_upper_arm")
    clip = _clip_with_channels(root, arm)
    context = MotionContext(path=None, foot_contacts=((1.0, 1.0), (1.0, 1.0), (1.0, 1.0), (1.0, 1.0), (0.0, 0.0)))

    stabilized, result = stabilize_feet(clip, _rig(), context, motion_class="static")

    stabilized_root = next(channel for channel in stabilized.channels if channel.target_joint == "Hips" and channel.path == "translation")
    stabilized_arm = next(channel for channel in stabilized.channels if channel.target_joint == "LeftUpperArm")
    assert result.status == "applied"
    assert result.mode == "root_compensation"
    assert result.corrected_frames == 4
    assert stabilized_root.values[:4] == ((0.0, 1.0, 0.0), (0.0, 1.0, 0.0), (0.0, 1.0, 0.0), (0.045, 1.0, 0.0))
    assert stabilized_root.values[4] == (0.5, 1.0, 0.0)
    assert stabilized_arm.values == arm.values
    assert stabilized.visual_quality["foot_stabilization"]["status"] == "applied"


def test_static_root_compensation_blends_window_boundaries_instead_of_hard_anchoring():
    values = tuple((index * 0.005, 1.0, 0.0) for index in range(20)) + ((0.50, 1.0, 0.0),)
    root = _channel("Hips", "translation", values, "hips")
    clip = _clip_with_channels(root)
    context = MotionContext(path=None, foot_contacts=((1.0, 1.0),) * 20 + ((0.0, 0.0),))

    stabilized, result = stabilize_feet(clip, _rig(), context, motion_class="static")

    stabilized_root = next(channel for channel in stabilized.channels if channel.target_joint == "Hips" and channel.path == "translation")
    assert result.status == "applied"
    assert result.corrected_frames == 20
    assert stabilized_root.values[0] == root.values[0]
    assert stabilized_root.values[1][0] == 0.0025
    assert stabilized_root.values[2][0] == 0.0025
    assert stabilized_root.values[10] == (0.0, 1.0, 0.0)
    assert stabilized_root.values[17][0] == 0.02125
    assert stabilized_root.values[18][0] == 0.045
    assert stabilized_root.values[19][0] == 0.07125
    assert stabilized_root.values[20] == (0.50, 1.0, 0.0)


def test_static_root_compensation_uses_short_window_boundary_blend():
    root = _channel("Hips", "translation", ((0.0, 1.0, 0.0), (0.03, 1.0, 0.0), (0.06, 1.0, 0.0), (0.09, 1.0, 0.0), (0.50, 1.0, 0.0)), "hips")
    clip = _clip_with_channels(root)
    context = MotionContext(path=None, foot_contacts=((1.0, 1.0), (1.0, 1.0), (1.0, 1.0), (1.0, 1.0), (0.0, 0.0)))

    stabilized, result = stabilize_feet(clip, _rig(), context, motion_class="static")

    stabilized_root = next(channel for channel in stabilized.channels if channel.target_joint == "Hips" and channel.path == "translation")
    assert result.status == "applied"
    assert stabilized_root.values[:4] == ((0.0, 1.0, 0.0), (0.0, 1.0, 0.0), (0.0, 1.0, 0.0), (0.045, 1.0, 0.0))
    assert stabilized_root.values[4] == root.values[4]


def test_static_compensation_skips_over_cap_without_mutating_clip():
    root = _channel("Hips", "translation", ((0.0, 0.0, 0.0), (0.07, 0.0, 0.0), (0.14, 0.0, 0.0), (0.21, 0.0, 0.0)), "hips")
    clip = _clip_with_channels(root)
    context = MotionContext(path=None, foot_contacts=((1.0, 1.0),) * 4)

    stabilized, result = stabilize_feet(clip, _rig(), context, motion_class="static")

    assert result.status == "skipped"
    assert result.skipped_reason == "no_eligible_contact_windows"
    assert stabilized.channels == clip.channels


def test_dynamic_motion_skips_without_hard_locking_by_default():
    root = _channel("Hips", "translation", tuple((index * 0.001, 0.0, 0.0) for index in range(12)), "hips")
    clip = _clip_with_channels(root)
    context = MotionContext(path=None, foot_contacts=((1.0, 1.0),) * 12)

    stabilized, result = stabilize_feet(clip, _rig(), context, motion_class="dynamic")

    stabilized_root = next(channel for channel in stabilized.channels if channel.target_joint == "Hips")
    assert result.status == "skipped"
    assert result.skipped_reason == "dynamic_motion_warning_only_by_default"
    assert stabilized_root.values == root.values


def test_walking_requires_calibrated_leg_chains_and_fails_closed_without_them():
    root = _channel("Hips", "translation", tuple((index * 0.02, 0.0, 0.0) for index in range(6)), "hips")
    clip = replace(_clip_with_channels(root), mapping=JointMapping(confidence="compatible", pairs={}, warnings=()))
    context = MotionContext(path=None, foot_contacts=tuple((1.0, 0.0) for _ in range(6)))

    stabilized, result = stabilize_feet(clip, _rig(calibrated=False), context, motion_class="walking")

    assert result.status == "skipped"
    assert result.mode == "none"
    assert result.skipped_reason == "missing_valid_leg_chains"
    assert stabilized.channels == clip.channels


def test_walking_over_cap_correction_fails_closed_instead_of_snapping():
    root = _channel("Hips", "translation", tuple((index * 0.04, 0.0, 0.0) for index in range(6)), "hips")
    clip = _clip_with_channels(root)
    context = MotionContext(path=None, foot_contacts=tuple((1.0, 0.0) for _ in range(6)))

    stabilized, result = stabilize_feet(clip, _rig(), context, motion_class="walking")

    assert result.status == "skipped"
    assert result.skipped_reason == "no_eligible_contact_windows"
    assert stabilized.channels == clip.channels
