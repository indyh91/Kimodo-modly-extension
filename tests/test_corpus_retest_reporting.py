from pathlib import Path
import hashlib
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from corpus_retest import (  # noqa: E402
    classify_corpus_result,
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
