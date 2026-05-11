from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from locomotion_axis import diagnose_locomotion_axes  # noqa: E402
from motion_evidence import classify_motion_evidence  # noqa: E402


def _alternating_walk_kwargs(*, root_axis: str) -> dict:
    if root_axis == "x":
        root_samples = ((0.0, 0.0, 0.0), (0.25, 0.0, 0.0), (0.5, 0.0, 0.0), (0.8, 0.0, 0.0))
        left_foot = ((-0.2, 0.0, 0.0), (-0.2, 0.0, 0.0), (0.2, 0.0, 0.0), (0.2, 0.0, 0.0))
        right_foot = ((0.2, 0.0, 0.0), (0.6, 0.0, 0.0), (0.6, 0.0, 0.0), (0.9, 0.0, 0.0))
    elif root_axis == "z":
        root_samples = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.25), (0.0, 0.0, 0.5), (0.0, 0.0, 0.8))
        left_foot = ((-0.2, 0.0, 0.0), (-0.2, 0.0, 0.0), (-0.2, 0.0, 0.4), (-0.2, 0.0, 0.4))
        right_foot = ((0.2, 0.0, 0.0), (0.2, 0.0, 0.4), (0.2, 0.0, 0.4), (0.2, 0.0, 0.8))
    else:
        raise ValueError(root_axis)

    return {
        "root_samples": root_samples,
        "foot_samples": {"left": left_foot, "right": right_foot},
        "foot_contacts": ((1.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.0, 1.0)),
        "dynamic_intensity": 0.35,
        "trusted_leg_chains": True,
    }


def _diagnostics(*, root_axis: str, facing_axis: str | None = None, swing_axis: str | None = None) -> dict:
    roots = {
        "x": [(0.0, 0.0, 0.0), (0.8, 0.0, 0.0)],
        "z": [(0.0, 0.0, 0.0), (0.0, 0.0, 0.8)],
    }
    facing_vectors = None
    if facing_axis == "x":
        facing_vectors = [(1.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
    elif facing_axis == "z":
        facing_vectors = [(0.0, 0.0, 1.0), (0.0, 0.0, 1.0)]

    foot_positions = None
    if swing_axis == "x":
        foot_positions = {"left": [(-0.4, 0.0, 0.0), (0.4, 0.0, 0.0)], "right": [(-0.3, 0.0, 0.1), (0.5, 0.0, 0.1)]}
    elif swing_axis == "z":
        foot_positions = {"left": [(-0.2, 0.0, 0.0), (-0.2, 0.0, 0.8)], "right": [(0.2, 0.0, 0.0), (0.2, 0.0, 0.8)]}

    return diagnose_locomotion_axes(
        root_positions=roots[root_axis],
        contract_basis={"forward": "+Z", "up": "+Y", "handedness": "right", "status": "trusted"},
        facing_vectors=facing_vectors,
        foot_positions=foot_positions,
    )


def _lateral_limb_swing_diagnostics() -> dict:
    return {
        "schema": "kimodo.limb_swing_plane.diagnostics.v1",
        "status": "evaluated",
        "classification": "lateral_dominated_limb_swing",
        "risk": "lateral_dominated_limb_swing",
        "root_hips_axis": {"aligned": True, "signed_axis": "+Z", "lateral_drift_ratio": 0.0},
        "feet": {
            "left": {
                "sample_count": 4,
                "lateral_forward_ratio": 4.0,
                "midline_crossing": {"detected": True, "wrong_side_fraction": 0.5},
            },
            "right": {
                "sample_count": 4,
                "lateral_forward_ratio": 4.0,
                "midline_crossing": {"detected": True, "wrong_side_fraction": 0.5},
            },
        },
        "correction_eligibility": {"eligible": False, "status": "skipped", "reasons": ["basis/inferred_not_authorized"]},
        "correction": {"status": "skipped", "applied": False, "applied_channels": []},
        "warnings": [{"code": "limb_swing_plane/lateral_dominated_limb_swing"}],
        "reasons": [{"code": "limb_swing_plane/lateral_dominated"}, {"code": "midline/crossing_detected"}],
    }


def test_classifier_derives_gesture_like_from_stable_root_feet_and_low_intensity():
    evidence = classify_motion_evidence(
        root_samples=((0.0, 0.0, 0.0), (0.01, 0.0, 0.0), (0.015, 0.0, 0.0), (0.02, 0.0, 0.0)),
        foot_samples={
            "left": ((-0.2, 0.0, 0.0), (-0.2, 0.0, 0.0), (-0.19, 0.0, 0.0), (-0.2, 0.0, 0.0)),
            "right": ((0.2, 0.0, 0.0), (0.2, 0.0, 0.0), (0.21, 0.0, 0.0), (0.2, 0.0, 0.0)),
        },
        foot_contacts=((1.0, 1.0), (1.0, 1.0), (1.0, 1.0), (1.0, 1.0)),
        dynamic_intensity=0.05,
        trusted_leg_chains=True,
        prompt_hint="walk forward",
    )

    assert evidence["class"] == "gesture_like"
    assert evidence["confidence"] >= 0.7
    assert evidence["prompt_hint"] == "walk forward"
    assert "prompt_conflicts_with_kinematics" in evidence["conflicts"]
    assert evidence["evidence"]["root_displacement"] == 0.02
    assert evidence["evidence"]["stance_consistency"] == "stable"


def test_classifier_derives_walking_from_root_displacement_and_alternating_contacts():
    evidence = classify_motion_evidence(
        root_samples=((0.0, 0.0, 0.0), (0.25, 0.0, 0.0), (0.5, 0.0, 0.0), (0.8, 0.0, 0.0)),
        foot_samples={
            "left": ((-0.2, 0.0, 0.0), (-0.2, 0.0, 0.0), (0.1, 0.0, 0.0), (0.1, 0.0, 0.0)),
            "right": ((0.2, 0.0, 0.0), (0.5, 0.0, 0.0), (0.5, 0.0, 0.0), (0.8, 0.0, 0.0)),
        },
        foot_contacts=((1.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.0, 1.0)),
        dynamic_intensity=0.35,
        trusted_leg_chains=True,
        prompt_hint="Person raising one arm",
    )

    assert evidence["class"] == "walking"
    assert evidence["confidence"] >= 0.8
    assert "root_displacement_locomotive" in evidence["reasons"]
    assert "alternating_foot_contacts" in evidence["reasons"]
    assert "prompt_conflicts_with_kinematics" in evidence["conflicts"]
    assert evidence["evidence"]["contact_windows"]["pattern"] == "alternating"
    assert evidence["evidence"]["foot_velocities"]["max"] > 0.2


def test_axis_diagnostics_prevent_lateral_high_magnitude_motion_from_becoming_forward_walking():
    diagnostics = _diagnostics(root_axis="x", facing_axis="x", swing_axis="x")

    evidence = classify_motion_evidence(
        axis_diagnostics=diagnostics,
        prompt_hint="Walking forward",
        **_alternating_walk_kwargs(root_axis="x"),
    )

    assert diagnostics["classification"] == "lateral_locomotion"
    assert evidence["class"] == "lateral_locomotion"
    assert evidence["class"] not in {"walking", "forward_locomotion"}
    assert "locomotion/axis_mismatch" in evidence["warnings"]
    assert evidence["evidence"]["locomotion_axis"]["root_advance"]["signed_axis"] == "+X"


def test_axis_diagnostics_promote_aligned_forward_motion_to_forward_locomotion():
    diagnostics = _diagnostics(root_axis="z", facing_axis="z", swing_axis="z")

    evidence = classify_motion_evidence(
        axis_diagnostics=diagnostics,
        prompt_hint="Person raising one arm",
        **_alternating_walk_kwargs(root_axis="z"),
    )

    assert diagnostics["classification"] == "forward_locomotion"
    assert evidence["class"] == "forward_locomotion"
    assert evidence["confidence"] >= 0.8
    assert "locomotion_axis/forward_aligned" in evidence["reasons"]
    assert evidence["evidence"]["locomotion_axis"]["contract_forward"]["signed_axis"] == "+Z"


def test_forward_absolute_root_and_feet_motion_preserves_lateral_limb_swing_risk():
    diagnostics = _diagnostics(root_axis="z", facing_axis="z", swing_axis="z")

    evidence = classify_motion_evidence(
        axis_diagnostics=diagnostics,
        limb_swing_plane_diagnostics=_lateral_limb_swing_diagnostics(),
        prompt_hint="Walking forward",
        **_alternating_walk_kwargs(root_axis="z"),
    )

    assert diagnostics["classification"] == "forward_locomotion"
    assert evidence["class"] == "forward_locomotion"
    assert evidence["evidence"]["locomotion_axis"]["root_advance"]["signed_axis"] == "+Z"
    assert evidence["evidence"]["limb_swing_plane"]["risk"] == "lateral_dominated_limb_swing"
    assert evidence["evidence"]["limb_swing_plane"]["feet"]["left"]["lateral_forward_ratio"] >= 2.0
    assert evidence["evidence"]["limb_swing_plane"]["feet"]["right"]["midline_crossing"]["detected"] is True
    assert "limb_swing_plane/lateral_dominated_limb_swing" in evidence["warnings"]
    assert "limb_swing_plane/lateral_dominated" in evidence["reasons"]
    assert "limb_swing_plane/correction_not_eligible" in evidence["warnings"]


def test_not_evaluated_limb_swing_plane_diagnostics_do_not_change_forward_classification():
    not_evaluated = {
        "schema": "kimodo.limb_swing_plane.diagnostics.v1",
        "status": "not_evaluated",
        "risk": "unknown",
        "reasons": [{"code": "samples/insufficient"}],
        "warnings": [{"code": "limb_swing_plane/diagnostics_not_evaluated"}],
    }

    evidence = classify_motion_evidence(
        axis_diagnostics=_diagnostics(root_axis="z", facing_axis="z", swing_axis="z"),
        limb_swing_plane_diagnostics=not_evaluated,
        prompt_hint="Walking forward",
        **_alternating_walk_kwargs(root_axis="z"),
    )

    assert evidence["class"] == "forward_locomotion"
    assert "limb_swing_plane" not in evidence["evidence"]
    assert "samples/insufficient" not in evidence["reasons"]
    assert "limb_swing_plane/diagnostics_not_evaluated" not in evidence["warnings"]


def test_axis_diagnostics_classify_stream_disagreement_as_ambiguous_not_forward():
    diagnostics = _diagnostics(root_axis="z", facing_axis="x", swing_axis="x")

    evidence = classify_motion_evidence(
        axis_diagnostics=diagnostics,
        prompt_hint="Walking forward",
        **_alternating_walk_kwargs(root_axis="z"),
    )

    assert diagnostics["classification"] == "ambiguous_locomotion"
    assert evidence["class"] == "ambiguous_locomotion"
    assert evidence["class"] not in {"walking", "forward_locomotion"}
    assert "evidence/stream_disagreement" in evidence["warnings"]
    assert evidence["evidence"]["locomotion_axis"]["hips_facing"]["signed_axis"] == "+X"


def test_missing_axis_diagnostics_preserve_existing_non_blocking_classification():
    evidence = classify_motion_evidence(
        prompt_hint="Walking forward",
        **_alternating_walk_kwargs(root_axis="x"),
    )

    assert evidence["class"] == "walking"
    assert evidence["warnings"] == []
    assert "locomotion_axis" not in evidence["evidence"]


def test_prompt_hint_cannot_override_lateral_or_ambiguous_axis_diagnostics():
    lateral = classify_motion_evidence(
        axis_diagnostics=_diagnostics(root_axis="x", facing_axis="x", swing_axis="x"),
        prompt_hint="Walking",
        **_alternating_walk_kwargs(root_axis="x"),
    )
    ambiguous = classify_motion_evidence(
        axis_diagnostics=_diagnostics(root_axis="z", facing_axis="x", swing_axis="x"),
        prompt_hint="Walking",
        **_alternating_walk_kwargs(root_axis="z"),
    )

    assert lateral["class"] == "lateral_locomotion"
    assert ambiguous["class"] == "ambiguous_locomotion"
    assert lateral["class"] != "walking"
    assert ambiguous["class"] != "walking"
    assert "prompt_conflicts_with_kinematics" in lateral["conflicts"]
    assert "prompt_conflicts_with_kinematics" in ambiguous["conflicts"]


def test_classifier_derives_dynamic_from_high_intensity_and_unstable_contacts():
    evidence = classify_motion_evidence(
        root_samples=((0.0, 0.0, 0.0), (0.05, 0.0, 0.0), (0.07, 0.0, 0.0), (0.1, 0.0, 0.0)),
        foot_samples={
            "left": ((-0.2, 0.0, 0.0), (0.3, 0.0, 0.2), (-0.4, 0.0, -0.1), (0.2, 0.0, 0.3)),
            "right": ((0.2, 0.0, 0.0), (-0.3, 0.0, -0.2), (0.4, 0.0, 0.1), (-0.2, 0.0, -0.3)),
        },
        foot_contacts=((1.0, 1.0), (0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),
        dynamic_intensity=0.92,
        trusted_leg_chains=True,
        prompt_hint="stand idle",
    )

    assert evidence["class"] == "dynamic"
    assert "high_dynamic_intensity" in evidence["reasons"]
    assert "unstable_contact_windows" in evidence["reasons"]
    assert evidence["evidence"]["stance_consistency"] == "unstable"


def test_classifier_returns_ineligible_without_trusted_leg_chains_and_unknown_for_insufficient_motion_evidence():
    no_trust = classify_motion_evidence(
        root_samples=((0.0, 0.0, 0.0), (0.5, 0.0, 0.0)),
        foot_samples={"left": ((0.0, 0.0, 0.0), (0.1, 0.0, 0.0)), "right": ((0.0, 0.0, 0.0), (0.1, 0.0, 0.0))},
        foot_contacts=((1.0, 0.0), (0.0, 1.0)),
        dynamic_intensity=0.4,
        trusted_leg_chains=False,
        prompt_hint="walk forward",
    )
    missing_frames = classify_motion_evidence(
        root_samples=((0.0, 0.0, 0.0),),
        foot_samples={"left": ((0.0, 0.0, 0.0),), "right": ((0.0, 0.0, 0.0),)},
        foot_contacts=((1.0, 1.0),),
        dynamic_intensity=None,
        trusted_leg_chains=True,
        prompt_hint="dance wildly",
    )

    assert no_trust["class"] == "ineligible"
    assert no_trust["warnings"] == ["motion_evidence/missing_trusted_leg_chains"]
    assert missing_frames["class"] == "unknown"
    assert missing_frames["confidence"] == 0.0
    assert "insufficient_frame_domain" in missing_frames["reasons"]


def test_classifier_does_not_branch_on_exact_full_prompt_text_when_evidence_is_identical():
    kwargs = dict(
        root_samples=((0.0, 0.0, 0.0), (0.005, 0.0, 0.0), (0.01, 0.0, 0.0)),
        foot_samples={
            "left": ((-0.1, 0.0, 0.0), (-0.1, 0.0, 0.0), (-0.1, 0.0, 0.0)),
            "right": ((0.1, 0.0, 0.0), (0.1, 0.0, 0.0), (0.1, 0.0, 0.0)),
        },
        foot_contacts=((1.0, 1.0), (1.0, 1.0), (1.0, 1.0)),
        dynamic_intensity=0.01,
        trusted_leg_chains=True,
    )

    typo_prompt = classify_motion_evidence(prompt_hint="Person raisin one arm", **kwargs)
    normal_prompt = classify_motion_evidence(prompt_hint="A completely different sentence", **kwargs)

    assert typo_prompt["class"] == "gesture_like"
    assert normal_prompt["class"] == "gesture_like"
    assert typo_prompt["reasons"] == normal_prompt["reasons"]
