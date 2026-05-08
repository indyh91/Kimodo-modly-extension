import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from foot_stabilization import (  # noqa: E402
    FootStabilizationConfig,
    evaluate_stabilization_gates,
)


def test_static_window_uses_threshold_min_window_velocity_and_correction_caps():
    samples = {
        "left": tuple((index * 0.01, 0.0, 0.0) for index in range(4)),
        "right": tuple((index * 0.01, 0.0, 0.0) for index in range(4)),
    }
    contacts = tuple((0.5, 0.5) for _ in range(4))

    result = evaluate_stabilization_gates(
        foot_samples=samples,
        foot_contacts=contacts,
        motion_class="static",
        trusted_foot_roles=True,
    )

    assert result.status == "eligible"
    assert result.gates["contact_threshold"] == 0.5
    assert result.gates["min_window_frames"] == 4
    assert result.gates["max_velocity_per_frame"] == 0.03
    assert result.gates["max_correction"] == 0.12
    assert result.pre_max_slide == 0.03
    assert [window.as_metadata() for window in result.windows] == [
        {"side": "left", "start": 0, "end": 3, "frame_count": 4, "max_slide": 0.03, "max_velocity": 0.01},
        {"side": "right", "start": 0, "end": 3, "frame_count": 4, "max_slide": 0.03, "max_velocity": 0.01},
    ]


def test_motion_classes_have_distinct_min_windows_velocity_and_correction_caps():
    config = FootStabilizationConfig()

    assert config.defaults_for("static") == {"min_window_frames": 4, "max_velocity_per_frame": 0.03, "max_correction": 0.12}
    assert config.defaults_for("gesture") == {"min_window_frames": 4, "max_velocity_per_frame": 0.03, "max_correction": 0.12}
    assert config.defaults_for("walking") == {"min_window_frames": 6, "max_velocity_per_frame": 0.06, "max_correction": 0.18}
    assert config.defaults_for("locomotion") == {"min_window_frames": 6, "max_velocity_per_frame": 0.06, "max_correction": 0.18}
    assert config.defaults_for("dynamic") == {"min_window_frames": 12, "max_velocity_per_frame": 0.02, "max_correction": 0.05}


def test_dynamic_motion_skips_warning_only_by_default_even_with_long_contact():
    samples = {
        "left": tuple((0.001 * index, 0.0, 0.0) for index in range(12)),
        "right": tuple((0.0, 0.0, 0.001 * index) for index in range(12)),
    }
    contacts = tuple((1.0, 1.0) for _ in range(12))

    result = evaluate_stabilization_gates(
        foot_samples=samples,
        foot_contacts=contacts,
        motion_class="dynamic",
        trusted_foot_roles=True,
    )

    assert result.status == "skipped"
    assert result.skipped_reason == "dynamic_motion_warning_only_by_default"
    assert "foot_stabilization/dynamic_default_skip" in result.warnings


def test_over_velocity_and_over_correction_windows_fail_closed():
    contacts = tuple((1.0, 0.0) for _ in range(4))
    fast_samples = {"left": ((0.0, 0.0, 0.0), (0.04, 0.0, 0.0), (0.08, 0.0, 0.0), (0.12, 0.0, 0.0)), "right": ((0.0, 0.0, 0.0),) * 4}
    far_contacts = tuple((1.0, 0.0) for _ in range(6))
    far_samples = {"left": ((0.0, 0.0, 0.0), (0.026, 0.0, 0.0), (0.052, 0.0, 0.0), (0.078, 0.0, 0.0), (0.104, 0.0, 0.0), (0.13, 0.0, 0.0)), "right": ((0.0, 0.0, 0.0),) * 6}

    fast = evaluate_stabilization_gates(foot_samples=fast_samples, foot_contacts=contacts, motion_class="static", trusted_foot_roles=True)
    far = evaluate_stabilization_gates(foot_samples=far_samples, foot_contacts=far_contacts, motion_class="static", trusted_foot_roles=True)

    assert fast.status == "skipped"
    assert fast.skipped_reason == "no_eligible_contact_windows"
    assert fast.windows[0].rejected_reason == "velocity_exceeds_cap"
    assert math.isclose(fast.windows[0].max_velocity, 0.04)
    assert far.status == "skipped"
    assert far.windows[0].rejected_reason == "correction_exceeds_cap"


def test_missing_or_invalid_evidence_is_not_evaluated():
    valid_samples = {"left": ((0.0, 0.0, 0.0),) * 4, "right": ((0.0, 0.0, 0.0),) * 4}
    contacts = tuple((1.0, 1.0) for _ in range(4))

    missing_roles = evaluate_stabilization_gates(foot_samples=valid_samples, foot_contacts=contacts, motion_class="static", trusted_foot_roles=False)
    mismatch = evaluate_stabilization_gates(foot_samples=valid_samples, foot_contacts=contacts[:3], motion_class="static", trusted_foot_roles=True)
    non_finite = evaluate_stabilization_gates(
        foot_samples={"left": ((0.0, 0.0, 0.0), (float("nan"), 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), "right": ((0.0, 0.0, 0.0),) * 4},
        foot_contacts=contacts,
        motion_class="static",
        trusted_foot_roles=True,
    )

    assert missing_roles.status == "not_evaluated"
    assert missing_roles.skipped_reason == "missing_trusted_foot_roles"
    assert mismatch.status == "not_evaluated"
    assert mismatch.skipped_reason == "foot_samples_contact_frame_count_mismatch"
    assert non_finite.status == "not_evaluated"
    assert non_finite.skipped_reason == "non_finite_foot_samples"
