import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import locomotion_axis  # noqa: E402
from locomotion_axis import (  # noqa: E402
    angle_degrees,
    axis_vector,
    dot_product,
    evaluate_axis_correction,
    horizontal_dominance,
    parse_contract_basis,
)


def _reason_codes(payload):
    return {reason.get("code", reason) for reason in payload.get("reasons", [])}


def _diagnose(**kwargs):
    return locomotion_axis.diagnose_locomotion_axes(**kwargs)


def _trusted_z_contract():
    return {"forward": "Z", "up": "Y", "handedness": "right", "status": "trusted"}


def test_parse_contract_basis_normalizes_signed_forward_up_and_handedness_metadata():
    positive_z = parse_contract_basis(
        {"forward": "Z", "up": "Y", "handedness": "right", "status": "trusted"},
        source="contract_basis",
    )
    negative_z = parse_contract_basis(
        {"forward": "-Z", "up": "Y", "handedness": "right", "status": "asserted"},
        source="contract_basis",
    )

    assert positive_z["status"] == "trusted"
    assert positive_z["valid"] is True
    assert positive_z["forward"] == {
        "axis": "z",
        "signed_axis": "+Z",
        "sign": 1,
        "vector": [0.0, 0.0, 1.0],
        "source": "contract_basis.forward",
        "sign_policy": "positive_default",
    }
    assert positive_z["up"] == {
        "axis": "y",
        "signed_axis": "+Y",
        "sign": 1,
        "vector": [0.0, 1.0, 0.0],
        "source": "contract_basis.up",
        "sign_policy": "positive_default",
    }
    assert positive_z["handedness"] == "right"
    assert positive_z["correction_eligible"] is True
    assert positive_z["diagnostics"] == []

    assert negative_z["status"] == "asserted"
    assert negative_z["valid"] is True
    assert negative_z["forward"]["signed_axis"] == "-Z"
    assert negative_z["forward"]["vector"] == [0.0, 0.0, -1.0]
    assert negative_z["up"]["signed_axis"] == "+Y"
    assert negative_z["correction_eligible"] is True


def test_parse_contract_basis_reports_invalid_axes_and_blocks_correction():
    basis = parse_contract_basis(
        {"forward": "north", "up": "Q", "handedness": "right", "status": "trusted"},
        source="contract_basis",
    )

    assert basis["valid"] is False
    assert basis["correction_eligible"] is False
    assert basis["forward"] is None
    assert basis["up"] is None
    assert basis["handedness"] == "right"
    assert _reason_codes(basis) == {
        "basis/invalid_forward_axis",
        "basis/invalid_up_axis",
        "basis/correction_ineligible_invalid_axes",
    }
    assert basis["diagnostics"] == [
        {"field": "forward", "code": "basis/invalid_forward_axis", "value": "north"},
        {"field": "up", "code": "basis/invalid_up_axis", "value": "Q"},
    ]


def test_axis_vector_dot_product_and_angle_for_aligned_axes():
    forward = axis_vector("+Z")
    same_forward = axis_vector("Z")
    reverse_forward = axis_vector("-Z")

    assert forward == (0.0, 0.0, 1.0)
    assert same_forward == (0.0, 0.0, 1.0)
    assert reverse_forward == (0.0, 0.0, -1.0)
    assert dot_product(forward, same_forward) == 1.0
    assert angle_degrees(forward, same_forward) == 0.0
    assert dot_product(forward, reverse_forward) == -1.0
    assert angle_degrees(forward, reverse_forward) == 180.0


def test_axis_vector_dot_product_and_angle_for_perpendicular_axes():
    right = axis_vector("+X")
    forward = axis_vector("+Z")

    assert dot_product(right, forward) == 0.0
    assert math.isclose(angle_degrees(right, forward), 90.0, abs_tol=1e-6)


def test_horizontal_dominance_reports_2d_and_3d_signed_axis_strength():
    z_dominant = horizontal_dominance((0.15, 0.0, 2.4), dead_zone=0.05, ambiguity_ratio=1.25)
    x_dominant = horizontal_dominance((-3.0, 0.4), dead_zone=0.05, ambiguity_ratio=1.25)
    ambiguous = horizontal_dominance((1.0, 0.0, 0.9), dead_zone=0.05, ambiguity_ratio=1.25)

    assert z_dominant == {
        "dominance": "z",
        "signed_axis": "+Z",
        "vector": [0.062378, 0.0, 0.998052],
        "primary_magnitude": 2.4,
        "secondary_magnitude": 0.15,
        "confidence": 0.9375,
    }
    assert x_dominant == {
        "dominance": "x",
        "signed_axis": "-X",
        "vector": [-0.991228, 0.0, 0.132164],
        "primary_magnitude": 3.0,
        "secondary_magnitude": 0.4,
        "confidence": 0.866667,
    }
    assert ambiguous["dominance"] == "none"
    assert ambiguous["signed_axis"] is None
    assert "dominance/ambiguous_horizontal_axis" in _reason_codes(ambiguous)


def test_inferred_basis_is_diagnostic_usable_but_not_correction_eligible_by_default():
    basis = parse_contract_basis(
        {"forward": "Z", "up": "Y", "handedness": "right", "status": "inferred"},
        source="humanoid_basis_hints",
    )

    decision = evaluate_axis_correction(
        basis=basis,
        evidence={
            "classification": "lateral_locomotion",
            "root_advance": {"signed_axis": "+X", "confidence": 0.94, "angle_to_forward": 90.0, "dot_to_forward": 0.0},
            "swing_plane": {"dominance": "x", "confidence": 0.91},
            "hips_facing": {"signed_axis": "+X", "confidence": 0.88},
        },
    )

    assert basis["valid"] is True
    assert basis["correction_eligible"] is False
    assert basis["forward"]["vector"] == [0.0, 0.0, 1.0]
    assert decision["status"] == "skipped"
    assert decision["eligible"] is False
    assert decision["skip_reason"] == "basis/inferred_not_authorized"
    assert "basis/inferred_diagnostic_only" in _reason_codes(decision)
    assert decision["evidence"]["basis_status"] == "inferred"


def test_trusted_basis_with_strong_recognized_sideways_evidence_is_correction_eligible():
    basis = parse_contract_basis(
        {"forward": "Z", "up": "Y", "handedness": "right", "status": "trusted"},
        source="contract_basis",
    )

    decision = evaluate_axis_correction(
        basis=basis,
        evidence={
            "classification": "lateral_locomotion",
            "root_advance": {"signed_axis": "+X", "confidence": 0.96, "angle_to_forward": 90.0, "dot_to_forward": 0.0},
            "swing_plane": {"dominance": "x", "confidence": 0.93},
            "hips_facing": {"signed_axis": "+X", "confidence": 0.9},
        },
        thresholds={"min_stream_confidence": 0.85, "sideways_angle_min": 75.0, "sideways_angle_max": 105.0},
    )

    assert decision["status"] == "eligible"
    assert decision["eligible"] is True
    assert decision["applied_channels"] == []
    assert decision["recommended_channels"] == ["root_translation"]
    assert decision["before"] == {"root_axis": "+X", "contract_forward": "+Z", "angle_degrees": 90.0, "dot": 0.0}
    assert decision["after"] == {"root_axis": "+Z", "contract_forward": "+Z"}
    assert decision["thresholds"] == {"min_stream_confidence": 0.85, "sideways_angle_min": 75.0, "sideways_angle_max": 105.0}
    assert "correction/recognized_sideways_locomotion" in _reason_codes(decision)


def test_low_confidence_or_ambiguous_evidence_skips_correction_with_machine_readable_reason():
    basis = parse_contract_basis(
        {"forward": "Z", "up": "Y", "handedness": "right", "status": "asserted"},
        source="contract_basis",
    )

    decision = evaluate_axis_correction(
        basis=basis,
        evidence={
            "classification": "ambiguous_locomotion",
            "root_advance": {"signed_axis": "+X", "confidence": 0.54, "angle_to_forward": 82.0, "dot_to_forward": 0.13},
            "swing_plane": {"dominance": "none", "confidence": 0.41},
            "hips_facing": {"signed_axis": "+Z", "confidence": 0.72},
        },
        thresholds={"min_stream_confidence": 0.85, "sideways_angle_min": 75.0, "sideways_angle_max": 105.0},
    )

    assert decision["status"] == "skipped"
    assert decision["eligible"] is False
    assert decision["skip_reason"] == "evidence/low_confidence_or_ambiguous"
    assert {"evidence/low_confidence", "evidence/stream_disagreement", "evidence/ambiguous_locomotion"}.issubset(
        _reason_codes(decision)
    )
    assert decision["evidence"]["minimum_confidence"] == 0.41
    assert decision["thresholds"]["min_stream_confidence"] == 0.85


def test_correction_decision_is_not_a_blind_axis_swap_and_records_threshold_evidence():
    basis = parse_contract_basis(
        {"forward": "Z", "up": "Y", "handedness": "right", "status": "trusted"},
        source="contract_basis",
    )

    decision = evaluate_axis_correction(
        basis=basis,
        evidence={
            "classification": "lateral_locomotion",
            "root_advance": {"signed_axis": "-X", "confidence": 0.95, "angle_to_forward": 90.0, "dot_to_forward": 0.0},
            "swing_plane": {"dominance": "x", "confidence": 0.92},
            "hips_facing": {"signed_axis": "-X", "confidence": 0.9},
        },
        thresholds={"min_stream_confidence": 0.85, "sideways_angle_min": 75.0, "sideways_angle_max": 105.0},
    )

    assert decision["eligible"] is True
    assert decision["target_axis"] == "+Z"
    assert decision["source_axis"] == "-X"
    assert decision["confidence"] == 0.9
    assert decision["reasons"]
    assert decision["thresholds"]["sideways_angle_min"] == 75.0
    assert decision["evidence"]["root_advance"]["angle_to_forward"] == 90.0
    assert decision["evidence"]["root_advance"]["dot_to_forward"] == 0.0
    assert decision != {"eligible": True, "target_axis": "+Z"}


def test_diagnose_locomotion_axes_reports_forward_root_advance_aligned_to_contract_forward():
    diagnostics = _diagnose(
        root_positions=[(0.0, 0.0, 0.0), (0.05, 0.0, 1.2), (0.1, 0.0, 2.4)],
        contract_basis=_trusted_z_contract(),
        facing_vectors=[(0.0, 0.0, 1.0), (0.05, 0.0, 0.99)],
        foot_positions={
            "left": [(0.15, 0.0, 0.0), (0.18, 0.0, 0.7), (0.16, 0.0, 1.4)],
            "right": [(-0.15, 0.0, 0.1), (-0.17, 0.0, 0.8), (-0.16, 0.0, 1.5)],
        },
    )

    assert diagnostics["classification"] == "forward_locomotion"
    assert diagnostics["contract_forward"]["axis"] == "z"
    assert diagnostics["contract_forward"]["vector"] == [0.0, 0.0, 1.0]
    assert diagnostics["root_advance"]["signed_axis"] == "+Z"
    assert diagnostics["root_advance"]["dominance"] == "z"
    assert diagnostics["root_advance"]["dot_to_forward"] > 0.95
    assert diagnostics["root_advance"]["angle_to_forward"] < 15.0
    assert diagnostics["root_advance"]["status"] == "aligned"
    assert diagnostics["correction_eligible"] is False


def test_diagnose_locomotion_axes_reports_perpendicular_root_advance_as_lateral_mismatch():
    diagnostics = _diagnose(
        root_positions=[(0.0, 0.0, 0.0), (1.1, 0.0, 0.04), (2.2, 0.0, 0.08)],
        contract_basis=_trusted_z_contract(),
        facing_vectors=[(1.0, 0.0, 0.0), (0.98, 0.0, 0.05)],
        foot_positions={
            "left": [(0.0, 0.0, 0.12), (0.7, 0.0, 0.13), (1.4, 0.0, 0.12)],
            "right": [(0.1, 0.0, -0.12), (0.8, 0.0, -0.13), (1.5, 0.0, -0.12)],
        },
    )

    assert diagnostics["classification"] == "lateral_locomotion"
    assert diagnostics["root_advance"]["signed_axis"] == "+X"
    assert diagnostics["root_advance"]["dominance"] == "x"
    assert diagnostics["root_advance"]["dot_to_forward"] < 0.15
    assert 75.0 <= diagnostics["root_advance"]["angle_to_forward"] <= 105.0
    assert diagnostics["root_advance"]["status"] == "misaligned"
    assert "locomotion/lateral_root_advance" in _reason_codes(diagnostics)


def test_diagnose_locomotion_axes_treats_high_x_dominant_displacement_with_z_contract_as_lateral_not_forward():
    diagnostics = _diagnose(
        root_positions=[(0.0, 0.0, 0.0), (1.5, 0.0, 0.02), (3.25, 0.0, 0.03)],
        contract_basis=_trusted_z_contract(),
    )

    assert diagnostics["root_advance"]["displacement"] > 3.0
    assert diagnostics["root_advance"]["dominance"] == "x"
    assert diagnostics["root_advance"]["status"] in {"lateral", "misaligned"}
    assert diagnostics["classification"] in {"lateral_locomotion", "ambiguous_locomotion"}
    assert diagnostics["classification"] != "forward_locomotion"
    assert "locomotion/axis_mismatch" in _reason_codes(diagnostics)


def test_diagnose_locomotion_axes_keeps_root_facing_and_swing_streams_separate_when_they_disagree():
    diagnostics = _diagnose(
        root_positions=[(0.0, 0.0, 0.0), (0.03, 0.0, 1.1), (0.04, 0.0, 2.2)],
        contract_basis=_trusted_z_contract(),
        facing_vectors=[(1.0, 0.0, 0.0), (0.99, 0.0, 0.02)],
        foot_positions={
            "left": [(0.0, 0.0, 0.2), (0.8, 0.0, 0.22), (1.6, 0.0, 0.21)],
            "right": [(0.1, 0.0, -0.2), (0.9, 0.0, -0.22), (1.7, 0.0, -0.21)],
        },
    )

    assert diagnostics["root_advance"]["signed_axis"] == "+Z"
    assert diagnostics["hips_facing"]["signed_axis"] == "+X"
    assert diagnostics["swing_plane"]["dominance"] == "x"
    assert diagnostics["streams"]["root_advance"] == diagnostics["root_advance"]
    assert diagnostics["streams"]["hips_facing"] == diagnostics["hips_facing"]
    assert diagnostics["streams"]["swing_plane"] == diagnostics["swing_plane"]
    assert diagnostics["classification"] == "ambiguous_locomotion"
    assert diagnostics["correction_eligible"] is False
    assert "evidence/stream_disagreement" in _reason_codes(diagnostics)


def test_diagnose_locomotion_axes_reports_insufficient_evidence_as_ambiguous_and_not_correction_eligible():
    diagnostics = _diagnose(
        root_positions=[(0.0, 0.0, 0.0), (0.01, 0.0, 0.015)],
        contract_basis=_trusted_z_contract(),
        facing_vectors=[],
        foot_positions={},
        thresholds={"min_root_displacement": 0.25},
    )

    assert diagnostics["classification"] in {"ambiguous_locomotion", "unknown"}
    assert diagnostics["root_advance"]["status"] == "insufficient_evidence"
    assert diagnostics["root_advance"]["confidence"] == 0.0
    assert diagnostics["hips_facing"]["status"] == "missing_evidence"
    assert diagnostics["swing_plane"]["status"] == "missing_evidence"
    assert diagnostics["correction_eligible"] is False
    assert {"evidence/insufficient_root_displacement", "evidence/missing_facing", "evidence/missing_swing_plane"}.issubset(
        _reason_codes(diagnostics)
    )


def _strong_lateral_axis_diagnostics(*, basis_status="trusted"):
    return _diagnose(
        root_positions=[(0.0, 0.0, 0.0), (1.2, 0.0, 0.02), (2.5, 0.0, 0.04)],
        contract_basis={"forward": "Z", "up": "Y", "handedness": "right", "status": basis_status},
        facing_vectors=[(1.0, 0.0, 0.0), (0.99, 0.0, 0.03)],
        foot_positions={
            "left": [(0.0, 0.0, 0.12), (0.8, 0.0, 0.13), (1.6, 0.0, 0.12)],
            "right": [(0.1, 0.0, -0.12), (0.9, 0.0, -0.13), (1.7, 0.0, -0.12)],
        },
    )


def test_plan_root_axis_correction_marks_strong_lateral_locomotion_eligible_with_threshold_evidence():
    diagnostics = _strong_lateral_axis_diagnostics(basis_status="trusted")

    decision = locomotion_axis.plan_root_axis_correction(
        axis_diagnostics=diagnostics,
        locomotion_classification="lateral_locomotion",
        thresholds={"min_stream_confidence": 0.85, "sideways_angle_min": 75.0, "sideways_angle_max": 105.0},
    )

    assert decision["correction_status"] == "eligible"
    assert decision["eligible"] is True
    assert decision["applied"] is False
    assert decision["confidence"] >= 0.85
    assert decision["thresholds"]["min_stream_confidence"] == 0.85
    assert decision["before"] == {"root_axis": "+X", "contract_forward": "+Z", "angle_degrees": 90.0, "dot": 0.0}
    assert decision["after"] == {"root_axis": "+Z", "contract_forward": "+Z"}
    assert decision["recommended_channels"] == ["root_translation"]
    assert "correction/recognized_sideways_locomotion" in _reason_codes(decision)


def test_apply_root_axis_correction_aligns_root_translation_only_and_preserves_magnitudes_and_timing():
    decision = {
        "correction_status": "eligible",
        "eligible": True,
        "source_axis": "+X",
        "target_axis": "+Z",
        "recommended_channels": ["root_translation"],
        "reasons": [{"code": "correction/recognized_sideways_locomotion"}],
    }
    root_positions = [
        {"time": 0.0, "translation": [0.0, 0.0, 0.0], "hips_yaw": 12.0},
        {"time": 0.5, "translation": [1.0, 0.1, 0.0], "hips_yaw": 13.0},
        {"time": 1.0, "translation": [2.25, 0.2, 0.0], "hips_yaw": 14.0},
    ]

    result = locomotion_axis.apply_root_axis_correction(root_positions=root_positions, decision=decision)

    assert result["correction_status"] == "applied"
    assert result["applied_channels"] == ["root_translation"]
    assert result["skipped_reason"] is None
    assert result["before_axis"] == "+X"
    assert result["after_axis"] == "+Z"
    assert [frame["time"] for frame in result["corrected_root_positions"]] == [0.0, 0.5, 1.0]
    assert [frame["hips_yaw"] for frame in result["corrected_root_positions"]] == [12.0, 13.0, 14.0]
    assert [frame["translation"] for frame in result["corrected_root_positions"]] == [
        [0.0, 0.0, 0.0],
        [0.0, 0.1, 1.0],
        [0.0, 0.2, 2.25],
    ]
    assert result["correction_status"] != "hips_yaw_applied"


def test_plan_root_axis_correction_skips_inferred_or_low_confidence_basis_with_explicit_reasons():
    inferred_diagnostics = _strong_lateral_axis_diagnostics(basis_status="inferred")
    low_confidence_diagnostics = _diagnose(
        root_positions=[(0.0, 0.0, 0.0), (0.4, 0.0, 0.32), (0.8, 0.0, 0.65)],
        contract_basis=_trusted_z_contract(),
        facing_vectors=[(0.7, 0.0, 0.7)],
        foot_positions={"left": [(0.0, 0.0, 0.0), (0.25, 0.0, 0.21)], "right": [(0.0, 0.0, 0.0), (0.2, 0.0, 0.24)]},
    )

    inferred_decision = locomotion_axis.plan_root_axis_correction(
        axis_diagnostics=inferred_diagnostics,
        locomotion_classification="lateral_locomotion",
    )
    low_confidence_decision = locomotion_axis.plan_root_axis_correction(
        axis_diagnostics=low_confidence_diagnostics,
        locomotion_classification="ambiguous_locomotion",
    )

    assert inferred_decision["correction_status"] == "skipped"
    assert inferred_decision["eligible"] is False
    assert inferred_decision["skipped_reason"] == "basis/inferred_not_authorized"
    assert "basis/inferred_diagnostic_only" in _reason_codes(inferred_decision)
    assert low_confidence_decision["correction_status"] == "skipped"
    assert low_confidence_decision["eligible"] is False
    assert low_confidence_decision["skipped_reason"] == "evidence/low_confidence_or_ambiguous"
    assert {"evidence/low_confidence", "evidence/ambiguous_locomotion"}.issubset(_reason_codes(low_confidence_decision))


def test_plan_root_axis_correction_skips_non_walking_classifications_even_when_root_moves():
    diagnostics = _strong_lateral_axis_diagnostics(basis_status="trusted")

    decisions = {
        classification: locomotion_axis.plan_root_axis_correction(
            axis_diagnostics=diagnostics,
            locomotion_classification=classification,
        )
        for classification in ("gesture_like", "dynamic", "unknown")
    }

    assert set(decisions) == {"gesture_like", "dynamic", "unknown"}
    for classification, decision in decisions.items():
        assert decision["correction_status"] == "skipped"
        assert decision["eligible"] is False
        assert decision["skipped_reason"] == "classification/not_lateral_locomotion"
        assert decision["classification"] == classification
        assert "classification/non_walking_not_corrected" in _reason_codes(decision)


def test_root_axis_correction_result_shape_is_machine_readable_for_applied_and_skipped_outcomes():
    skipped_decision = locomotion_axis.plan_root_axis_correction(
        axis_diagnostics=_strong_lateral_axis_diagnostics(basis_status="inferred"),
        locomotion_classification="lateral_locomotion",
    )
    applied_result = locomotion_axis.apply_root_axis_correction(
        root_positions=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
        decision={
            "correction_status": "eligible",
            "eligible": True,
            "source_axis": "+X",
            "target_axis": "+Z",
            "confidence": 0.9,
            "recommended_channels": ["root_translation"],
            "reasons": [{"code": "correction/recognized_sideways_locomotion"}],
        },
    )

    assert {"correction_status", "eligible", "applied", "skipped_reason", "reason", "confidence", "before", "after"}.issubset(
        skipped_decision
    )
    assert skipped_decision["correction_status"] == "skipped"
    assert skipped_decision["applied"] is False
    assert {"correction_status", "applied", "applied_channels", "reason", "confidence", "before_axis", "after_axis"}.issubset(
        applied_result
    )
    assert applied_result["correction_status"] == "applied"
    assert applied_result["applied"] is True


def test_plan_root_axis_correction_rejects_blind_axis_name_swaps_without_diagnostic_evidence():
    decision = locomotion_axis.plan_root_axis_correction(
        source_axis="+X",
        target_axis="+Z",
        locomotion_classification="lateral_locomotion",
        axis_diagnostics=None,
    )

    assert decision["correction_status"] == "skipped"
    assert decision["eligible"] is False
    assert decision["skipped_reason"] == "evidence/explicit_diagnostics_required"
    assert decision["source_axis"] == "+X"
    assert decision["target_axis"] == "+Z"
    assert "correction/no_blind_axis_swap" in _reason_codes(decision)
