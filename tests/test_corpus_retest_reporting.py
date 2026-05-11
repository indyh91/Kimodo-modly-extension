import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from corpus_retest import (  # noqa: E402
    analyze_glb_locomotion_axis,
    classify_corpus_result,
    prepare_safe_corpus_analysis,
    prepare_safe_workflow_axis_analysis,
    prepare_safe_retest_asset,
    summarize_corpus_results,
)


def test_classifies_known_mapping_error_as_legacy_mapping_failure_not_regression():
    entry = classify_corpus_result(
        {
            "runtime_status": "failed",
            "retarget_error_code": "MAPPING_INCOMPATIBLE",
            "contract_status": "trusted",
            "stabilization_status": "not_evaluated",
        }
    )

    assert entry["bucket"] == "legacy_mapping_failure"
    assert entry["legacy_error_code"] == "MAPPING_INCOMPATIBLE"
    assert entry["eligible"] is False
    assert entry["regression"] is False
    assert entry["stabilization_evaluated"] is False
    assert entry["runtime_status"] == "failed"


def test_classifies_contract_missing_or_untrusted_as_ineligible_not_success_with_successful_runtime():
    entry = classify_corpus_result(
        {
            "runtime_status": "success",
            "contract_status": "contract_missing_or_untrusted",
            "stabilization_status": "not_evaluated",
        }
    )

    assert entry["bucket"] == "contract_missing_or_untrusted"
    assert entry["eligible"] is False
    assert entry["stabilization_evaluated"] is False
    assert entry["trusted_stabilization_success"] is False
    assert entry["runtime_status"] == "success"
    assert entry["regression"] is False


def test_classifies_trusted_evaluated_stabilization_and_regression_from_residual_slide():
    stable = classify_corpus_result(
        {
            "runtime_status": "success",
            "contract_status": "trusted",
            "stabilization_status": "applied",
            "pre_max_slide": 0.18,
            "post_max_slide": 0.03,
        },
        regression_threshold=0.05,
    )
    regressed = classify_corpus_result(
        {
            "runtime_status": "success",
            "contract_status": "trusted",
            "stabilization_status": "partial",
            "pre_max_slide": 0.10,
            "post_max_slide": 0.12,
        },
        regression_threshold=0.05,
    )

    assert stable["bucket"] == "stabilization_result"
    assert stable["eligible"] is True
    assert stable["stabilization_evaluated"] is True
    assert stable["trusted_stabilization_success"] is True
    assert stable["regression"] is False
    assert regressed["bucket"] == "stabilization_result"
    assert regressed["regression"] is True


def test_summarize_corpus_results_counts_exact_buckets_and_machine_readable_fields():
    summary = summarize_corpus_results(
        [
            {"runtime_status": "failed", "retarget_error_code": "MAPPING_INCOMPATIBLE"},
            {"runtime_status": "success", "contract_status": "contract_missing_or_untrusted", "stabilization_status": "not_evaluated"},
            {"runtime_status": "success", "contract_status": "trusted", "stabilization_status": "applied", "pre_max_slide": 0.2, "post_max_slide": 0.02},
        ]
    )

    assert summary["bucket_counts"] == {
        "legacy_mapping_failure": 1,
        "contract_missing_or_untrusted": 1,
        "stabilization_result": 1,
    }
    assert summary["stabilization_regressions"] == 0
    assert summary["schema"] == "kimodo.corpus_retest.v1"
    assert summary["results"][1]["bucket"] == "contract_missing_or_untrusted"
    assert summary["results"][1]["trusted_stabilization_success"] is False


def test_prepare_safe_retest_asset_copies_under_explicit_temp_root_and_reports_hash_bucket_and_copy_path(tmp_path):
    source_root = tmp_path / "source"
    source_root.mkdir()
    original = source_root / "case.glb"
    original.write_bytes(b"original asset bytes")
    temp_root = tmp_path / "opencode-safe"

    report = prepare_safe_retest_asset(
        original,
        temp_root=temp_root,
        result={"runtime_status": "success", "contract_status": "contract_missing_or_untrusted", "stabilization_status": "not_evaluated"},
    )

    expected_hash = hashlib.sha256(b"original asset bytes").hexdigest()
    copy_path = Path(report["artifact_copy"])

    assert report["source_artifact_hash"] == expected_hash
    assert report["source_artifact"] == str(original)
    assert copy_path.is_file()
    assert temp_root in copy_path.parents
    assert copy_path != original
    assert report["mutation_target"] == str(copy_path)
    assert report["mutation_target"] != report["source_artifact"]
    assert report["bucket"] == "contract_missing_or_untrusted"
    assert report["runtime_status"] == "success"


def test_prepare_safe_retest_asset_defaults_to_tmp_opencode_and_never_targets_original(tmp_path):
    original = tmp_path / "nested" / "asset.bvh"
    original.parent.mkdir()
    original.write_text("HIERARCHY\n", encoding="utf-8")

    report = prepare_safe_retest_asset(
        original,
        result={"runtime_status": "success", "contract_status": "trusted", "stabilization_status": "applied"},
    )

    copy_path = Path(report["artifact_copy"])
    assert Path("/tmp/opencode") in copy_path.parents
    assert report["output_root"].startswith("/tmp/opencode")
    assert report["mutation_target"] == str(copy_path)
    assert report["mutation_target"] != str(original)
    assert original.read_text(encoding="utf-8") == "HIERARCHY\n"


def test_prepare_safe_corpus_analysis_copies_glbs_under_safe_root_records_hashes_and_read_only_status(tmp_path):
    source_root = tmp_path / "exports"
    source_root.mkdir()
    first = source_root / "walk-a.glb"
    second = source_root / "walk-b.glb"
    first.write_bytes(b"first glb bytes")
    second.write_bytes(b"second glb bytes")
    before = {first: first.read_bytes(), second: second.read_bytes()}
    safe_root = tmp_path / "tmp" / "opencode" / "kimodo-corpus-retest"

    def read_only_analyzer(copy_path):
        return {
            "analysis_status": "synthetic_ok",
            "locomotion_axis": {
                "classification": "lateral_locomotion",
                "root_advance": {"signed_axis": "+X"},
                "contract_forward": {"signed_axis": "+Z"},
                "correction": {"correction_status": "skipped", "reason": "analysis_only"},
            },
            "copy_size": Path(copy_path).stat().st_size,
        }

    summary = prepare_safe_corpus_analysis([first, second], temp_root=safe_root, analyzer=read_only_analyzer)

    assert summary["schema"] == "kimodo.corpus_retest.safe_analysis.v1"
    assert summary["total"] == 2
    assert summary["analysis_mode"] == "read_only_copy"
    assert summary["workflow_invocation"] == "none"
    for entry, source in zip(summary["assets"], (first, second)):
        copy_path = Path(entry["copy_path"])
        expected_hash = hashlib.sha256(before[source]).hexdigest()
        assert entry["source_path"] == str(source.resolve())
        assert entry["source_hash"] == expected_hash
        assert entry["copy_hash"] == expected_hash
        assert entry["mutation_guard"]["original_unchanged"] is True
        assert entry["mutation_guard"]["source_is_mutation_target"] is False
        assert entry["analysis_status"] == "synthetic_ok"
        assert entry["locomotion_axis"]["classification"] == "lateral_locomotion"
        assert safe_root in copy_path.parents
        assert copy_path.read_bytes() == before[source]
        assert source.read_bytes() == before[source]


def test_prepare_safe_corpus_analysis_rejects_target_root_outside_opencode_like_safe_area(tmp_path):
    original = tmp_path / "walk.glb"
    original.write_bytes(b"glb")
    unsafe_root = tmp_path / "ordinary-working-dir"

    try:
        prepare_safe_corpus_analysis([original], temp_root=unsafe_root)
    except ValueError as exc:
        assert "safe corpus root" in str(exc)
    else:
        raise AssertionError("unsafe corpus root was accepted")


def test_analyze_glb_locomotion_axis_extracts_translation_axis_and_basis_without_mutation(tmp_path):
    glb = tmp_path / "walk-x.glb"
    glb.write_bytes(
        _minimal_animated_glb(
            translations=[(0.0, 0.0, 0.0), (2.0, 0.0, 0.1)],
            extras={"contract_basis": {"forward": "+Z", "up": "+Y", "handedness": "right", "status": "trusted"}},
        )
    )
    before = hashlib.sha256(glb.read_bytes()).hexdigest()

    result = analyze_glb_locomotion_axis(glb)

    assert hashlib.sha256(glb.read_bytes()).hexdigest() == before
    assert result["analysis_status"] == "analyzed"
    assert result["basis_status"] == "available"
    assert result["contract_basis"] == {"forward": "+Z", "up": "+Y", "handedness": "right", "status": "trusted"}
    assert result["root_advance"]["signed_axis"] == "+X"
    assert result["root_advance"]["horizontal_magnitude"] == 2.002498
    assert result["root_advance"]["confidence"] > 0.9
    assert result["classification"] == "lateral_locomotion"
    assert {reason["code"] for reason in result["reasons"]} == {"locomotion/lateral_root_advance", "basis/metadata_found"}


def test_analyze_glb_locomotion_axis_reports_unavailable_basis_conservatively(tmp_path):
    glb = tmp_path / "walk-z-no-basis.glb"
    glb.write_bytes(_minimal_animated_glb(translations=[(0.0, 0.0, 0.0), (0.0, 0.0, 3.0)]))

    result = analyze_glb_locomotion_axis(glb)

    assert result["analysis_status"] == "analyzed"
    assert result["basis_status"] == "unavailable"
    assert result["contract_basis"] is None
    assert result["root_advance"]["signed_axis"] == "+Z"
    assert result["classification"] == "ambiguous_locomotion"
    assert "basis/unavailable" in {reason["code"] for reason in result["reasons"]}
    assert "basis metadata unavailable; classification remains conservative" in result["limitations"]


def test_prepare_safe_workflow_axis_analysis_copies_bundle_and_rigmeta_then_reports_inferred_lateral_bvh(tmp_path):
    workflows = tmp_path / "Workflows"
    bundle = workflows / "kimodo-synthetic-lateral"
    bundle.mkdir(parents=True)
    rigmeta = workflows / "source_unirig.rigmeta.json"
    rigmeta.write_text(
        json.dumps(
            {
                "humanoid_contract": {
                    "basis": {"forward": "Z", "up": "Y", "handedness": "right", "status": "inferred"},
                    "roles": {"hips": "bone_0", "left_foot": "bone_1", "right_foot": "bone_2"},
                }
            }
        ),
        encoding="utf-8",
    )
    (bundle / "metadata.json").write_text(
        json.dumps(
            {
                "prompt": "Walk and then jump",
                "source_prompt": "Walk and then jump",
                "runtime_status": "success",
                "retarget_status": "success",
                "contract_status": "trusted",
                "animation_mapping_status": "trusted_contract",
                "source_rigged_mesh": str(workflows / "source_unirig.glb"),
                "diagnostics": {"humanoid_contract": {"path": str(rigmeta), "status": "trusted", "roles": [{"role": "hips", "node_name": "bone_0"}]}},
            }
        ),
        encoding="utf-8",
    )
    (bundle / "motion.bvh").write_text(_minimal_bvh([(0.0, 0.0, 0.0), (3.0, 0.0, 0.1)]), encoding="utf-8")
    (bundle / "animated.glb").write_bytes(b"glb bytes")
    original_bytes = {path: path.read_bytes() for path in (bundle / "metadata.json", bundle / "motion.bvh", bundle / "animated.glb", rigmeta)}

    summary = prepare_safe_workflow_axis_analysis([bundle], temp_root=tmp_path / "tmp" / "opencode" / "workflow-axis")

    entry = summary["bundles"][0]
    copied_bundle = Path(entry["copied_bundle_path"])
    copied_rigmeta = Path(entry["copied_files"][str(rigmeta.resolve())]["copy_path"])
    assert summary["schema"] == "kimodo.workflow_axis_analysis.v1"
    assert summary["workflow_invocation"] == "none"
    assert copied_bundle != bundle
    assert (copied_bundle / "metadata.json").is_file()
    assert copied_rigmeta.is_file()
    assert entry["prompt"] == "Walk and then jump"
    assert entry["statuses"] == {"runtime_status": "success", "retarget_status": "success", "contract_status": "trusted", "animation_mapping_status": "trusted_contract"}
    assert entry["basis_status"] == "inferred"
    assert entry["basis_source"] == "rigmeta"
    assert entry["root_advance"]["signed_axis"] == "+X"
    assert entry["root_advance"]["basis_angle_degrees"] == 90.0
    assert entry["classification"] == "lateral_locomotion"
    assert entry["correction_eligibility"]["eligible"] is False
    assert entry["correction_eligibility"]["reason"] == "basis/inferred_diagnostic_only"
    assert entry["mutation_guard"]["all_originals_unchanged"] is True
    assert entry["mutation_guard"]["analysis_used_copies_only"] is True
    for path, before in original_bytes.items():
        assert path.read_bytes() == before


def test_prepare_safe_workflow_axis_analysis_uses_metadata_basis_and_classifies_forward(tmp_path):
    bundle = tmp_path / "Workflows" / "kimodo-synthetic-forward"
    bundle.mkdir(parents=True)
    (bundle / "metadata.json").write_text(
        json.dumps(
            {
                "prompt": "Walk forward",
                "runtime_status": "success",
                "retarget_status": "success",
                "contract_status": "trusted",
                "animation_mapping_status": "trusted_contract",
                "basis": {"forward": "+Z", "up": "+Y", "handedness": "right", "status": "trusted"},
            }
        ),
        encoding="utf-8",
    )
    (bundle / "motion.bvh").write_text(_minimal_bvh([(0.0, 0.0, 0.0), (0.1, 0.0, 2.0)]), encoding="utf-8")

    summary = prepare_safe_workflow_axis_analysis([bundle], temp_root=tmp_path / "tmp" / "opencode" / "workflow-axis")

    entry = summary["bundles"][0]
    assert entry["basis_status"] == "trusted"
    assert entry["basis_source"] == "metadata"
    assert entry["root_advance"]["signed_axis"] == "+Z"
    assert entry["classification"] == "forward_locomotion"
    assert entry["correction_eligibility"] == {"eligible": False, "reason": "classification/no_correction_needed", "policy": "trusted_or_asserted_basis_required"}
    assert summary["counts"]["classification"] == {"forward_locomotion": 1}


def test_prepare_safe_workflow_axis_analysis_falls_back_to_npz_when_bvh_root_is_static(tmp_path):
    bundle = tmp_path / "Workflows" / "kimodo-synthetic-npz-root"
    bundle.mkdir(parents=True)
    (bundle / "metadata.json").write_text(
        json.dumps(
            {
                "prompt": "Walk from npz",
                "runtime_status": "success",
                "retarget_status": "success",
                "contract_status": "trusted",
                "animation_mapping_status": "trusted_contract",
                "basis": {"forward": "+Z", "up": "+Y", "handedness": "right", "status": "trusted"},
            }
        ),
        encoding="utf-8",
    )
    (bundle / "motion.bvh").write_text(_minimal_bvh([(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)]), encoding="utf-8")
    np.savez(bundle / "motion.npz", smooth_root_pos=np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.1]], dtype=float))

    summary = prepare_safe_workflow_axis_analysis([bundle], temp_root=tmp_path / "tmp" / "opencode" / "workflow-axis")

    entry = summary["bundles"][0]
    assert entry["root_advance"]["source"] == "motion.npz"
    assert entry["root_advance"]["signed_axis"] == "+X"
    assert entry["classification"] == "lateral_locomotion"


def _minimal_animated_glb(*, translations, extras=None):
    times = [0.0, 1.0]
    bin_blob = b"".join(struct.pack("<f", value) for value in times)
    while len(bin_blob) % 4:
        bin_blob += b"\x00"
    translation_offset = len(bin_blob)
    for translation in translations:
        bin_blob += struct.pack("<fff", *translation)

    gltf = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": len(bin_blob)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(times) * 4},
            {"buffer": 0, "byteOffset": translation_offset, "byteLength": len(translations) * 12},
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": len(times), "type": "SCALAR"},
            {"bufferView": 1, "componentType": 5126, "count": len(translations), "type": "VEC3"},
        ],
        "nodes": [{"name": "Root"}],
        "animations": [
            {
                "samplers": [{"input": 0, "output": 1, "interpolation": "LINEAR"}],
                "channels": [{"sampler": 0, "target": {"node": 0, "path": "translation"}}],
            }
        ],
    }
    if extras is not None:
        gltf["asset"]["extras"] = extras

    json_blob = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    while len(json_blob) % 4:
        json_blob += b" "
    while len(bin_blob) % 4:
        bin_blob += b"\x00"
    total_length = 12 + 8 + len(json_blob) + 8 + len(bin_blob)
    return b"".join(
        [
            struct.pack("<III", 0x46546C67, 2, total_length),
            struct.pack("<I4s", len(json_blob), b"JSON"),
            json_blob,
            struct.pack("<I4s", len(bin_blob), b"BIN\x00"),
            bin_blob,
        ]
    )


def _minimal_bvh(root_positions):
    motion_rows = "\n".join(f"{x} {y} {z} 0 0 0" for x, y, z in root_positions)
    return f"""HIERARCHY
ROOT Root
{{
  OFFSET 0.0 0.0 0.0
  CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation
}}
MOTION
Frames: {len(root_positions)}
Frame Time: 0.0333333
{motion_rows}
"""
