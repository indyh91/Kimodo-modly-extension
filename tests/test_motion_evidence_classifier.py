from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from motion_evidence import classify_motion_evidence  # noqa: E402


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
