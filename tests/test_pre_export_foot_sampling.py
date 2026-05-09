from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from foot_stabilization import stabilize_feet  # noqa: E402
from retargeting import JointMapping, RetargetChannel, RetargetClip, sample_pre_export_foot_nodes  # noqa: E402
from rig_inspector import HumanoidChain, HumanoidRoleBinding, RigInspection  # noqa: E402


def _channel(target, path, values, role):
    return RetargetChannel(
        target_joint=target,
        path=path,
        times=tuple(index / 30.0 for index in range(len(values))),
        values=tuple(values),
        diagnostics={"role": role},
    )


def _clip(root_values):
    root = _channel("Hips", "translation", root_values, "hips")
    return RetargetClip(
        name="Kimodo Motion",
        duration=0.1,
        frame_time=1.0 / 30.0,
        root_translations=tuple(root_values),
        channels=(root,),
        mapping=JointMapping(confidence="calibrated", pairs={}, warnings=()),
        warnings=(),
    )


def _trusted_rig(*, rest=None, joint_children=()):
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
        joint_children=joint_children,
        humanoid_roles=roles,
        humanoid_chains=(
            HumanoidChain("left_leg", ("left_upper_leg", "left_lower_leg", "left_foot")),
            HumanoidChain("right_leg", ("right_upper_leg", "right_lower_leg", "right_foot")),
        ),
        humanoid_rest=rest or {
            "roles": {
                "left_foot": {"world_translation": [-0.2, 0.0, 0.3]},
                "right_foot": {"world_translation": [0.2, 0.0, 0.3]},
            }
        },
        humanoid_basis_hints={"up_axis": "Y"},
        contract_status="trusted",
    )


def _rotation_y(degrees):
    import math

    radians = math.radians(degrees)
    return (0.0, math.sin(radians / 2.0), 0.0, math.cos(radians / 2.0))


def _hierarchy_rig():
    return _trusted_rig(
        joint_children=(
            ("Hips", ("LeftUpperLeg", "RightUpperLeg")),
            ("LeftUpperLeg", ("LeftLowerLeg",)),
            ("LeftLowerLeg", ("LeftFoot",)),
            ("RightUpperLeg", ("RightLowerLeg",)),
            ("RightLowerLeg", ("RightFoot",)),
        ),
        rest={
            "roles": {
                "hips": {"local_translation": [0.0, 0.0, 0.0]},
                "left_upper_leg": {"local_translation": [-1.0, 0.0, 0.0]},
                "left_lower_leg": {"local_translation": [0.0, -1.0, 0.0]},
                "left_foot": {"local_translation": [0.0, -1.0, 0.0]},
                "right_upper_leg": {"local_translation": [1.0, 0.0, 0.0]},
                "right_lower_leg": {"local_translation": [0.0, -1.0, 0.0]},
                "right_foot": {"local_translation": [0.0, -1.0, 0.0]},
            }
        },
    )


def test_pre_export_sampler_uses_real_left_right_foot_nodes_over_clip_frame_domain():
    clip = _clip(((0.0, 1.0, 0.0), (1.0, 1.0, 0.0), (2.0, 1.0, 0.0)))

    result = sample_pre_export_foot_nodes(clip, _trusted_rig())

    assert result["status"] == "success"
    assert result["source"] == "retarget_clip_trs"
    assert result["frame_count"] == 3
    assert result["samples"]["left"] == ((-0.2, 1.0, 0.3), (0.8, 1.0, 0.3), (1.8, 1.0, 0.3))
    assert result["samples"]["right"] == ((0.2, 1.0, 0.3), (1.2, 1.0, 0.3), (2.2, 1.0, 0.3))
    assert result["samples"]["left"] != result["samples"]["right"]


def test_pre_export_sampler_rejects_root_translation_duplication_as_trusted_samples():
    clip = _clip(((0.0, 1.0, 0.0), (1.0, 1.0, 0.0)))

    result = sample_pre_export_foot_nodes(clip, _trusted_rig(rest={"roles": {}}))

    assert result["status"] == "not_evaluated"
    assert result["reason"] == "missing_trusted_foot_rest_transforms"
    assert result["samples"] == {}
    assert result["warnings"] == ["pre_export_foot_sampling/root_proxy_not_trusted"]


def test_pre_export_sampler_rotates_local_child_offsets_with_root_rotation_over_frame_domain():
    root_values = ((0.0, 1.0, 0.0), (0.0, 1.0, 2.0))
    root = _channel("Hips", "translation", root_values, "hips")
    root_rotation = _channel("Hips", "rotation", ((0.0, 0.0, 0.0, 1.0), _rotation_y(90.0)), "hips")
    clip = RetargetClip(
        name="Kimodo Motion",
        duration=1.0 / 30.0,
        frame_time=1.0 / 30.0,
        root_translations=root_values,
        channels=(root, root_rotation),
        mapping=JointMapping(confidence="calibrated", pairs={}, warnings=()),
        warnings=(),
    )

    result = sample_pre_export_foot_nodes(clip, _hierarchy_rig())

    assert result["status"] == "success"
    assert result["frame_count"] == 2
    assert result["frame_time"] == 0.0333333333
    assert result["samples"]["left"] == ((-1.0, -1.0, 0.0), (0.0, -1.0, 3.0))
    assert result["samples"]["right"] == ((1.0, -1.0, 0.0), (0.0, -1.0, 1.0))
    assert result["samples"]["left"] != result["samples"]["right"]
    assert result["samples"]["left"] != root_values
    assert result["samples"]["right"] != root_values


def test_pre_export_sampler_composes_chained_local_rotations_before_sampling_foot_origins():
    root_values = ((0.0, 1.0, 0.0), (0.0, 1.0, 0.0))
    root = _channel("Hips", "translation", root_values, "hips")
    left_upper_rotation = _channel("LeftUpperLeg", "rotation", ((0.0, 0.0, 0.0, 1.0), _rotation_y(90.0)), "left_upper_leg")
    right_upper_rotation = _channel("RightUpperLeg", "rotation", ((0.0, 0.0, 0.0, 1.0), _rotation_y(-90.0)), "right_upper_leg")
    clip = RetargetClip(
        name="Kimodo Motion",
        duration=1.0 / 30.0,
        frame_time=1.0 / 30.0,
        root_translations=root_values,
        channels=(root, left_upper_rotation, right_upper_rotation),
        mapping=JointMapping(confidence="calibrated", pairs={}, warnings=()),
        warnings=(),
    )

    rig = _trusted_rig(
        joint_children=(
            ("Hips", ("LeftUpperLeg", "RightUpperLeg")),
            ("LeftUpperLeg", ("LeftLowerLeg",)),
            ("LeftLowerLeg", ("LeftFoot",)),
            ("RightUpperLeg", ("RightLowerLeg",)),
            ("RightLowerLeg", ("RightFoot",)),
        ),
        rest={
            "roles": {
                "hips": {"local_translation": [0.0, 0.0, 0.0]},
                "left_upper_leg": {"local_translation": [-1.0, 0.0, 0.0]},
                "left_lower_leg": {"local_translation": [0.0, 0.0, 1.0]},
                "left_foot": {"local_translation": [0.0, 0.0, 1.0]},
                "right_upper_leg": {"local_translation": [1.0, 0.0, 0.0]},
                "right_lower_leg": {"local_translation": [0.0, 0.0, 1.0]},
                "right_foot": {"local_translation": [0.0, 0.0, 1.0]},
            }
        },
    )

    result = sample_pre_export_foot_nodes(clip, rig)

    assert result["status"] == "success"
    assert result["samples"]["left"] == ((-1.0, 1.0, 2.0), (1.0, 1.0, 0.0))
    assert result["samples"]["right"] == ((1.0, 1.0, 2.0), (-1.0, 1.0, 0.0))


def test_pre_export_sampler_marks_local_rest_without_hierarchy_as_not_evaluated():
    clip = _clip(((0.0, 1.0, 0.0), (1.0, 1.0, 0.0)))

    result = sample_pre_export_foot_nodes(
        clip,
        _trusted_rig(
            rest={
                "roles": {
                    "left_foot": {"local_translation": [-0.2, 0.0, 0.3]},
                    "right_foot": {"local_translation": [0.2, 0.0, 0.3]},
                }
            }
        ),
    )

    assert result["status"] == "not_evaluated"
    assert result["reason"] == "insufficient_hierarchy_or_rotation_evidence"
    assert result["frame_count"] == 2
    assert result["samples"] == {}


def test_stabilization_consumes_pre_export_foot_samples_instead_of_root_proxy_samples():
    root_values = ((0.0, 1.0, 0.0), (0.05, 1.0, 0.0), (0.10, 1.0, 0.0), (0.15, 1.0, 0.0))
    clip = _clip(root_values)
    pre_export = {
        "status": "success",
        "samples": {
            "left": ((-0.2, 0.0, 0.3), (-0.2, 0.0, 0.3), (-0.2, 0.0, 0.3), (-0.2, 0.0, 0.3)),
            "right": ((0.2, 0.0, 0.3), (0.2, 0.0, 0.3), (0.2, 0.0, 0.3), (0.2, 0.0, 0.3)),
        },
    }
    clip = RetargetClip(
        name=clip.name,
        duration=clip.duration,
        frame_time=clip.frame_time,
        root_translations=clip.root_translations,
        channels=clip.channels,
        mapping=clip.mapping,
        warnings=clip.warnings,
        calibration={"pre_export_foot_samples": pre_export},
    )
    context = type("Context", (), {"foot_contacts": ((1.0, 1.0),) * 4})()

    _stabilized, result = stabilize_feet(clip, _trusted_rig(), context, motion_class="static")

    assert result.status == "applied"
    assert result.pre_max_slide == 0.0
    assert result.corrected_frames == 4
