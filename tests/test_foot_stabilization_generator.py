from dataclasses import replace
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import generator  # noqa: E402
from foot_stabilization import FootStabilizationResult  # noqa: E402
from retargeting import JointMapping, RetargetClip  # noqa: E402


def _clip_with_stabilization(status="applied"):
    return RetargetClip(
        name="Kimodo Motion",
        duration=0.1,
        frame_time=1.0 / 30.0,
        root_translations=(),
        channels=(),
        mapping=JointMapping(confidence="calibrated", pairs={}, warnings=()),
        warnings=(),
        visual_quality={
            "version": "kimodo.visual_quality.v1",
            "blocking": False,
            "status": "not_evaluated",
            "warnings": [],
            "limitations": [],
            "checks": {},
            "foot_stabilization": {
                "status": status,
                "mode": "root_compensation",
                "motion_class": "static",
                "gates": {"residual_success_threshold": 0.05},
                "windows": [],
                "corrected_frames": 4,
                "strength": 1.0,
                "max_correction": 0.04,
                "pre_max_slide": 0.08,
                "post_max_slide": None,
                "residual_slide": None,
                "skipped_reason": None,
                "warnings": [],
                "limitations": ["v1_stabilization_does_not_guarantee_perfect_animation"],
            },
        },
    )


def test_generator_enrichment_updates_stabilization_residual_without_blocking_success(monkeypatch):
    clip = _clip_with_stabilization()
    motion_context = type("MotionContext", (), {"foot_contacts": ((1.0, 0.0), (1.0, 0.0))})()
    rig = type("Rig", (), {"humanoid_roles": ()})()

    def fake_extract(path, humanoid_roles, *, expected_frame_count=None):
        assert Path(path).name == "animated.glb"
        assert expected_frame_count == 2
        return type(
            "Result",
            (),
            {
                "status": "success",
                "samples": {"left": ((0.0, 0.0, 0.0), (0.07, 0.0, 0.0)), "right": ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))},
                "reason": None,
                "details": {"frame_count": 2},
            },
        )()

    monkeypatch.setattr(generator, "extract_world_foot_samples", fake_extract)

    enriched = generator._enrich_visual_quality_after_export(Path("animated.glb"), rig, clip, motion_context, {})

    stabilization = enriched.visual_quality["foot_stabilization"]
    assert stabilization["status"] == "partial"
    assert stabilization["pre_max_slide"] == 0.08
    assert stabilization["post_max_slide"] == 0.07
    assert stabilization["residual_slide"] == 0.07
    assert "foot_stabilization/residual_slide_above_threshold" in stabilization["warnings"]


def test_generator_enrichment_preserves_success_metadata_when_stabilization_was_skipped(monkeypatch):
    clip = _clip_with_stabilization(status="skipped")
    clip = replace(clip, visual_quality={**clip.visual_quality, "foot_stabilization": {**clip.visual_quality["foot_stabilization"], "skipped_reason": "missing_valid_leg_chains"}})
    motion_context = type("MotionContext", (), {"foot_contacts": ((1.0, 0.0), (1.0, 0.0))})()
    rig = type("Rig", (), {"humanoid_roles": ()})()

    def fake_extract(path, humanoid_roles, *, expected_frame_count=None):
        return type("Result", (), {"status": "not_evaluated", "samples": {}, "reason": "missing_trusted_foot_roles", "details": {}})()

    monkeypatch.setattr(generator, "extract_world_foot_samples", fake_extract)

    enriched = generator._enrich_visual_quality_after_export(Path("animated.glb"), rig, clip, motion_context, {})

    stabilization = enriched.visual_quality["foot_stabilization"]
    assert stabilization["status"] == "skipped"
    assert stabilization["skipped_reason"] == "missing_valid_leg_chains"
    assert stabilization["post_max_slide"] is None
