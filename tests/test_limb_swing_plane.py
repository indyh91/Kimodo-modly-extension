import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import limb_swing_plane  # noqa: E402
from limb_swing_plane import diagnose_limb_swing_plane  # noqa: E402


def _reason_codes(payload):
    return {reason.get("code", reason) for reason in payload.get("reasons", [])}


def _healthy_walk_samples():
    root_positions = [(0.0, 0.0, 0.0), (0.01, 0.0, 1.0), (0.01, 0.0, 2.0), (0.02, 0.0, 3.0)]
    hips_positions = root_positions
    left_foot_positions = [(-0.25, 0.0, 0.0), (-0.23, 0.0, 1.35), (-0.27, 0.0, 2.7), (-0.24, 0.0, 4.05)]
    right_foot_positions = [(0.25, 0.0, 0.05), (0.27, 0.0, 1.4), (0.24, 0.0, 2.75), (0.26, 0.0, 4.1)]
    return {
        "root_positions": root_positions,
        "hips_positions": hips_positions,
        "left_foot_positions": left_foot_positions,
        "right_foot_positions": right_foot_positions,
    }


def _trusted_basis():
    return {
        "source": "calibrated_mapping",
        "status": "asserted",
        "forward": "+Z",
        "up": "+Y",
        "handedness": "right",
    }


def _passing_mirror_evidence():
    return {
        "status": "passed",
        "left_chain_roles": ["left_hip", "left_knee", "left_ankle", "left_foot"],
        "right_chain_roles": ["right_hip", "right_knee", "right_ankle", "right_foot"],
        "rest_side_signs": {"left": -1, "right": 1},
        "symmetry_score": 0.93,
    }


def _passing_chain_evidence():
    return {
        "status": "passed",
        "left_chain_roles": ["left_hip", "left_knee", "left_ankle", "left_foot"],
        "right_chain_roles": ["right_hip", "right_knee", "right_ankle", "right_foot"],
        "ordered": True,
        "feet_present": True,
        "toes_present": True,
    }


def test_diagnostics_report_lateral_dominated_relative_feet_despite_forward_root_and_hips():
    root_positions = [(0.0, 0.0, 0.0), (0.02, 0.0, 1.0), (0.03, 0.0, 2.0), (0.04, 0.0, 3.0), (0.05, 0.0, 4.0)]
    hips_positions = [(0.0, 0.0, 0.0), (0.01, 0.0, 1.0), (0.02, 0.0, 2.0), (0.02, 0.0, 3.0), (0.03, 0.0, 4.0)]
    left_relative = [(-0.35, 0.0, 0.0), (-0.1, 0.0, 0.04), (0.2, 0.0, 0.08), (0.45, 0.0, 0.14), (0.75, 0.0, 0.18)]
    right_relative = [(0.35, 0.0, 0.02), (0.1, 0.0, 0.05), (-0.2, 0.0, 0.09), (-0.45, 0.0, 0.15), (-0.7, 0.0, 0.2)]
    left_foot_positions = [(hips[0] + foot[0], hips[1] + foot[1], hips[2] + foot[2]) for hips, foot in zip(hips_positions, left_relative)]
    right_foot_positions = [(hips[0] + foot[0], hips[1] + foot[1], hips[2] + foot[2]) for hips, foot in zip(hips_positions, right_relative)]

    diagnostics = diagnose_limb_swing_plane(
        root_positions=root_positions,
        hips_positions=hips_positions,
        left_foot_positions=left_foot_positions,
        right_foot_positions=right_foot_positions,
    )

    assert diagnostics["schema"] == "kimodo.limb_swing_plane.diagnostics.v1"
    assert diagnostics["status"] == "evaluated"
    assert diagnostics["classification"] == "lateral_dominated_limb_swing"
    assert diagnostics["root_hips_axis"]["signed_axis"] == "+Z"
    assert diagnostics["root_hips_axis"]["aligned"] is True
    assert diagnostics["root_hips_axis"]["lateral_drift_ratio"] <= 0.1
    assert diagnostics["absolute_foot_motion_insufficient"] is True
    assert diagnostics["feet"]["left"]["lateral_forward_ratio"] >= 2.0
    assert diagnostics["feet"]["right"]["lateral_forward_ratio"] >= 2.0
    assert diagnostics["feet"]["left"]["midline_crossing"]["detected"] is True
    assert diagnostics["feet"]["right"]["midline_crossing"]["detected"] is True
    assert diagnostics["feet"]["left"]["wrong_side_fraction"] >= 0.05
    assert diagnostics["feet"]["right"]["wrong_side_fraction"] >= 0.05
    assert {"limb/lateral_dominated", "limb/midline_crossing_detected"}.issubset(_reason_codes(diagnostics))


def test_absolute_positive_z_foot_motion_does_not_clear_relative_limb_plane_warning():
    root_positions = [(0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 0.0, 2.0), (0.0, 0.0, 3.0)]
    hips_positions = root_positions
    left_foot_positions = [(-0.25, 0.0, 0.0), (0.1, 0.0, 1.03), (0.35, 0.0, 2.06), (0.6, 0.0, 3.09)]
    right_foot_positions = [(0.25, 0.0, 0.0), (-0.1, 0.0, 1.03), (-0.35, 0.0, 2.06), (-0.6, 0.0, 3.09)]

    diagnostics = diagnose_limb_swing_plane(
        root_positions=root_positions,
        hips_positions=hips_positions,
        left_foot_positions=left_foot_positions,
        right_foot_positions=right_foot_positions,
    )

    assert diagnostics["root_hips_axis"]["signed_axis"] == "+Z"
    assert diagnostics["feet"]["left"]["absolute_advance"]["signed_axis"] == "+Z"
    assert diagnostics["feet"]["right"]["absolute_advance"]["signed_axis"] == "+Z"
    assert diagnostics["feet"]["left"]["lateral_forward_ratio"] >= 2.0
    assert diagnostics["feet"]["right"]["lateral_forward_ratio"] >= 2.0
    assert diagnostics["absolute_foot_motion_insufficient"] is True
    assert diagnostics["classification"] == "lateral_dominated_limb_swing"


def test_healthy_forward_relative_swing_reports_no_midline_crossing_or_lateral_dominance():
    root_positions = [(0.0, 0.0, 0.0), (0.01, 0.0, 1.0), (0.01, 0.0, 2.0), (0.02, 0.0, 3.0)]
    hips_positions = root_positions
    left_foot_positions = [(-0.25, 0.0, 0.0), (-0.23, 0.0, 1.35), (-0.27, 0.0, 2.7), (-0.24, 0.0, 4.05)]
    right_foot_positions = [(0.25, 0.0, 0.05), (0.27, 0.0, 1.4), (0.24, 0.0, 2.75), (0.26, 0.0, 4.1)]

    diagnostics = diagnose_limb_swing_plane(
        root_positions=root_positions,
        hips_positions=hips_positions,
        left_foot_positions=left_foot_positions,
        right_foot_positions=right_foot_positions,
    )

    assert diagnostics["status"] == "evaluated"
    assert diagnostics["classification"] == "healthy_limb_swing"
    assert diagnostics["feet"]["left"]["lateral_forward_ratio"] < 2.0
    assert diagnostics["feet"]["right"]["lateral_forward_ratio"] < 2.0
    assert diagnostics["feet"]["left"]["midline_crossing"]["detected"] is False
    assert diagnostics["feet"]["right"]["midline_crossing"]["detected"] is False
    assert diagnostics["absolute_foot_motion_insufficient"] is False


def test_missing_or_insufficient_samples_fail_closed_as_not_evaluated():
    diagnostics = diagnose_limb_swing_plane(
        root_positions=[(0.0, 0.0, 0.0)],
        hips_positions=[],
        left_foot_positions=None,
        right_foot_positions=[(0.2, 0.0, 0.0)],
    )

    assert diagnostics["schema"] == "kimodo.limb_swing_plane.diagnostics.v1"
    assert diagnostics["status"] == "not_evaluated"
    assert diagnostics["classification"] == "not_evaluated"
    assert diagnostics["sample_count"] == 0
    assert "samples/insufficient" in _reason_codes(diagnostics)
    assert diagnostics["absolute_foot_motion_insufficient"] is True


def test_missing_basis_is_not_trusted_for_correction_and_fails_eligibility_closed():
    diagnostics = diagnose_limb_swing_plane(
        **_healthy_walk_samples(),
        mirror_evidence=_passing_mirror_evidence(),
        chain_evidence=_passing_chain_evidence(),
    )

    assert diagnostics["basis"] == {
        "source": None,
        "status": "missing",
        "forward": None,
        "up": None,
        "handedness": None,
        "trusted_for_correction": False,
        "reason": "basis/missing",
    }
    assert diagnostics["correction_eligibility"]["eligible"] is False
    assert "basis/missing" in diagnostics["correction_eligibility"]["reasons"]


def test_inferred_basis_remains_diagnostic_only_and_fails_eligibility_closed():
    diagnostics = diagnose_limb_swing_plane(
        **_healthy_walk_samples(),
        basis={
            "source": "child_vectors",
            "status": "inferred",
            "forward": "+Z",
            "up": "+Y",
            "handedness": "right",
        },
        mirror_evidence=_passing_mirror_evidence(),
        chain_evidence=_passing_chain_evidence(),
    )

    assert diagnostics["basis"]["trusted_for_correction"] is False
    assert diagnostics["basis"]["reason"] == "basis/inferred_not_authorized"
    assert diagnostics["correction_eligibility"]["eligible"] is False
    assert "basis/inferred_not_authorized" in diagnostics["correction_eligibility"]["reasons"]


def test_missing_or_failed_mirror_and_chain_evidence_fail_closed_with_reason_codes():
    missing_evidence = diagnose_limb_swing_plane(
        **_healthy_walk_samples(),
        basis=_trusted_basis(),
    )
    failed_evidence = diagnose_limb_swing_plane(
        **_healthy_walk_samples(),
        basis=_trusted_basis(),
        mirror_evidence={
            "status": "failed",
            "left_chain_roles": ["left_hip", "left_knee", "left_ankle", "left_foot"],
            "right_chain_roles": ["right_hip", "right_knee", "right_ankle", "right_foot"],
            "rest_side_signs": {"left": 1, "right": -1},
            "reasons": ["mirror/rest_sides_crossed"],
        },
        chain_evidence={
            "status": "failed",
            "left_chain_roles": ["left_hip", "left_knee"],
            "right_chain_roles": ["right_hip", "right_knee", "right_ankle", "right_foot"],
            "reasons": ["roles/missing_feet_or_toes"],
        },
    )

    assert missing_evidence["mirror_validation"]["status"] == "missing"
    assert missing_evidence["chain_validation"]["status"] == "missing"
    assert {"mirror/missing", "chain/missing"}.issubset(set(missing_evidence["correction_eligibility"]["reasons"]))
    assert failed_evidence["mirror_validation"]["status"] == "failed"
    assert failed_evidence["chain_validation"]["status"] == "failed"
    assert {"mirror/failed", "chain/failed", "roles/missing_feet_or_toes"}.issubset(
        set(failed_evidence["correction_eligibility"]["reasons"])
    )


def test_trusted_basis_and_passing_role_topology_evidence_marks_diagnostics_correction_eligible_without_applying():
    diagnostics = diagnose_limb_swing_plane(
        **_healthy_walk_samples(),
        basis=_trusted_basis(),
        mirror_evidence=_passing_mirror_evidence(),
        chain_evidence=_passing_chain_evidence(),
    )

    assert diagnostics["basis"]["trusted_for_correction"] is True
    assert diagnostics["mirror_validation"]["status"] == "passed"
    assert diagnostics["chain_validation"]["status"] == "passed"
    assert diagnostics["mirror_validation"]["evidence"]["left_chain_roles"][-1] == "left_foot"
    assert diagnostics["chain_validation"]["evidence"]["right_chain_roles"][-1] == "right_foot"
    assert diagnostics["correction_eligibility"] == {
        "eligible": True,
        "policy": "trusted_basis_and_passing_chain_mirror_required",
        "reasons": [],
    }
    assert diagnostics["correction"]["status"] == "skipped"
    assert diagnostics["correction"]["correction_status"] == "skipped"
    assert diagnostics["correction"]["applied"] is False
    assert diagnostics["correction"]["applied_channels"] == []
    assert diagnostics["correction"]["recommended_channels"] == []
    assert "motion/non_lateral_or_no_limb_risk" in diagnostics["correction"]["reasons"]


def test_correction_plan_skips_fail_closed_gate_and_motion_cases_without_applying():
    inferred_basis = diagnose_limb_swing_plane(
        **_healthy_walk_samples(),
        basis={"source": "child_vectors", "status": "inferred", "forward": "+Z", "up": "+Y"},
        mirror_evidence=_passing_mirror_evidence(),
        chain_evidence=_passing_chain_evidence(),
    )
    missing_mirror = diagnose_limb_swing_plane(
        **_healthy_walk_samples(),
        basis=_trusted_basis(),
        chain_evidence=_passing_chain_evidence(),
    )
    insufficient_samples = diagnose_limb_swing_plane(
        root_positions=[(0.0, 0.0, 0.0)],
        hips_positions=[],
        left_foot_positions=None,
        right_foot_positions=[(0.2, 0.0, 0.0)],
        basis=_trusted_basis(),
        mirror_evidence=_passing_mirror_evidence(),
        chain_evidence=_passing_chain_evidence(),
    )
    non_lateral = diagnose_limb_swing_plane(
        **_healthy_walk_samples(),
        basis=_trusted_basis(),
        mirror_evidence=_passing_mirror_evidence(),
        chain_evidence=_passing_chain_evidence(),
    )

    cases = (
        (inferred_basis, "basis/inferred_not_authorized"),
        (missing_mirror, "mirror/missing"),
        (insufficient_samples, "samples/insufficient"),
        (non_lateral, "motion/non_lateral_or_no_limb_risk"),
    )

    for diagnostics, expected_reason in cases:
        plan = limb_swing_plane.plan_limb_swing_plane_correction(diagnostics)

        assert plan["schema"] == "kimodo.limb_swing_plane.correction_plan.v1"
        assert plan["policy"] == "trusted_basis_and_mirror_required"
        assert plan["status"] == "skipped"
        assert plan["eligible"] is False
        assert plan["applied"] is False
        assert plan["applied_channels"] == []
        assert expected_reason in plan["reasons"]


def test_trusted_lateral_dominated_plan_is_eligible_not_applied_and_lower_body_rotation_only():
    root_positions = [(0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 0.0, 2.0), (0.0, 0.0, 3.0)]
    hips_positions = root_positions
    left_foot_positions = [(-0.35, 0.0, 0.0), (0.2, 0.0, 1.05), (-0.7, 0.0, 2.1), (0.8, 0.0, 3.15)]
    right_foot_positions = [(0.35, 0.0, 0.0), (-0.2, 0.0, 1.05), (0.7, 0.0, 2.1), (-0.8, 0.0, 3.15)]
    diagnostics = diagnose_limb_swing_plane(
        root_positions=root_positions,
        hips_positions=hips_positions,
        left_foot_positions=left_foot_positions,
        right_foot_positions=right_foot_positions,
        basis=_trusted_basis(),
        mirror_evidence=_passing_mirror_evidence(),
        chain_evidence=_passing_chain_evidence(),
    )

    plan = limb_swing_plane.plan_limb_swing_plane_correction(
        diagnostics,
        channels=[
            {"role": "hips", "path": "translation", "name": "hips_translation"},
            {"role": "left_upper_leg", "path": "rotation", "name": "left_thigh_rotation"},
            {"role": "spine", "path": "rotation", "name": "spine_rotation"},
            {"role": "right_foot", "path": "rotation", "name": "right_foot_rotation"},
            {"role": "left_hand", "path": "rotation", "name": "left_hand_rotation"},
        ],
    )

    assert diagnostics["risk"] == "lateral_dominated_limb_swing"
    assert plan["status"] == "eligible_not_applied"
    assert plan["eligible"] is True
    assert plan["applied"] is False
    assert plan["target_scope"] == "lower_body_rotation_channels"
    assert plan["recommended_channels"] == ["left_thigh_rotation", "right_foot_rotation"]
    assert plan["applied_channels"] == []
    assert plan["reasons"] == []
    assert plan["gate_summary"]["basis"]["status"] == "asserted"
    assert plan["gate_summary"]["mirror"]["status"] == "passed"
    assert plan["gate_summary"]["chain"]["status"] == "passed"
