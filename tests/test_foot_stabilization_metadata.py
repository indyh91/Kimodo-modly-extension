from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from foot_stabilization import (  # noqa: E402
    FootStabilizationResult,
    foot_stabilization_metadata,
    update_stabilization_residuals,
)


def test_metadata_shape_contains_auditable_status_gates_windows_and_limitations():
    result = FootStabilizationResult(
        status="applied",
        mode="root_compensation",
        motion_class="static",
        gates={"contact_threshold": 0.5, "residual_success_threshold": 0.05},
        windows=({"side": "left", "start": 0, "end": 3, "frame_count": 4, "max_slide": 0.04, "max_velocity": 0.01},),
        corrected_frames=4,
        strength=0.75,
        max_correction=0.04,
        pre_max_slide=0.08,
        post_max_slide=None,
        residual_slide=None,
        skipped_reason=None,
        warnings=("foot_stabilization/residual_pending",),
        limitations=("v1_stabilization_does_not_guarantee_perfect_animation",),
    )

    metadata = foot_stabilization_metadata(result)

    assert metadata == {
        "status": "applied",
        "mode": "root_compensation",
        "motion_class": "static",
        "gates": {"contact_threshold": 0.5, "residual_success_threshold": 0.05},
        "windows": [{"side": "left", "start": 0, "end": 3, "frame_count": 4, "max_slide": 0.04, "max_velocity": 0.01}],
        "corrected_frames": 4,
        "strength": 0.75,
        "max_correction": 0.04,
        "pre_max_slide": 0.08,
        "post_max_slide": None,
        "residual_slide": None,
        "skipped_reason": None,
        "warnings": ["foot_stabilization/residual_pending"],
        "limitations": ["v1_stabilization_does_not_guarantee_perfect_animation"],
    }


def test_metadata_rejects_unknown_status_to_protect_contract():
    result = FootStabilizationResult(status="bad", mode="none", motion_class="unknown")

    try:
        foot_stabilization_metadata(result)
    except ValueError as exc:
        assert str(exc) == "Unsupported foot stabilization status: bad"
    else:
        raise AssertionError("unknown stabilization status should fail contract validation")


def test_residual_update_classifies_partial_when_post_export_slide_remains_high():
    result = FootStabilizationResult(
        status="applied",
        mode="root_compensation",
        motion_class="static",
        pre_max_slide=0.12,
        corrected_frames=4,
    )

    updated = update_stabilization_residuals(result, post_max_slide=0.07)

    assert updated.status == "partial"
    assert updated.post_max_slide == 0.07
    assert updated.residual_slide == 0.07
    assert "foot_stabilization/residual_slide_above_threshold" in updated.warnings


def test_residual_update_preserves_applied_status_when_residual_is_below_success_threshold():
    result = FootStabilizationResult(
        status="applied",
        mode="root_compensation",
        motion_class="static",
        pre_max_slide=0.12,
        corrected_frames=4,
    )

    updated = update_stabilization_residuals(result, post_max_slide=0.05)

    assert updated.status == "applied"
    assert updated.post_max_slide == 0.05
    assert updated.residual_slide == 0.05
    assert "foot_stabilization/residual_slide_above_threshold" not in updated.warnings
