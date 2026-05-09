"""Corpus retest classification and safe-copy reporting helpers.

These helpers are intentionally small and filesystem-local: they prepare retest
inputs as copies and summarize existing synthetic/run metadata. They do not run
Modly workflows and never choose original artifacts as mutation/output targets.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Any


SCHEMA = "kimodo.corpus_retest.v1"
BUCKETS = ("legacy_mapping_failure", "contract_missing_or_untrusted", "stabilization_result")
LEGACY_MAPPING_ERROR_CODES = {"MAPPING_INCOMPATIBLE", "LEGACY_MAPPING_INCOMPATIBLE", "MAPPING_FAILED"}
EVALUATED_STABILIZATION_STATUSES = {"applied", "partial", "skipped"}
SUCCESS_STABILIZATION_STATUSES = {"applied", "partial"}


def classify_corpus_result(result: dict[str, Any], *, regression_threshold: float = 0.05) -> dict[str, Any]:
    """Classify a retarget/stabilization result into the acceptance bucket taxonomy."""
    runtime_status = str(result.get("runtime_status") or _runtime_status_from_retarget(result))
    legacy_error_code = _legacy_error_code(result)
    contract_status = str(result.get("contract_status") or "contract_missing_or_untrusted")
    stabilization_status = str(result.get("stabilization_status") or "not_evaluated")
    pre_max_slide = _optional_float(result.get("pre_max_slide"))
    post_max_slide = _optional_float(result.get("post_max_slide"))

    if legacy_error_code in LEGACY_MAPPING_ERROR_CODES:
        bucket = "legacy_mapping_failure"
        eligible = False
        stabilization_evaluated = False
        trusted_success = False
        regression = False
    elif contract_status != "trusted":
        bucket = "contract_missing_or_untrusted"
        eligible = False
        stabilization_evaluated = False
        trusted_success = False
        regression = False
    else:
        bucket = "stabilization_result"
        eligible = True
        stabilization_evaluated = stabilization_status in EVALUATED_STABILIZATION_STATUSES
        trusted_success = stabilization_status in SUCCESS_STABILIZATION_STATUSES
        regression = _is_stabilization_regression(pre_max_slide, post_max_slide, regression_threshold)

    return {
        "schema": SCHEMA,
        "bucket": bucket,
        "eligible": eligible,
        "legacy_error_code": legacy_error_code,
        "contract_status": contract_status,
        "stabilization_status": stabilization_status,
        "stabilization_evaluated": stabilization_evaluated,
        "trusted_stabilization_success": trusted_success,
        "pre_max_slide": pre_max_slide,
        "post_max_slide": post_max_slide,
        "regression": regression,
        "runtime_status": runtime_status,
    }


def summarize_corpus_results(results: list[dict[str, Any]] | tuple[dict[str, Any], ...], *, regression_threshold: float = 0.05) -> dict[str, Any]:
    """Return a machine-readable summary for already-collected corpus result metadata."""
    classified = [classify_corpus_result(result, regression_threshold=regression_threshold) for result in results]
    bucket_counts = {bucket: 0 for bucket in BUCKETS}
    for entry in classified:
        bucket_counts[entry["bucket"]] += 1
    return {
        "schema": SCHEMA,
        "bucket_counts": bucket_counts,
        "stabilization_regressions": sum(1 for entry in classified if entry["bucket"] == "stabilization_result" and entry["regression"]),
        "total": len(classified),
        "results": classified,
    }


def prepare_safe_retest_asset(source_path: str | Path, *, result: dict[str, Any] | None = None, temp_root: str | Path | None = None) -> dict[str, Any]:
    """Copy a corpus artifact to a safe retest root and return copy/report metadata."""
    source = Path(source_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    root = Path(temp_root or "/tmp/opencode/kimodo-corpus-retest").resolve()
    source_hash = _sha256_file(source)
    copy_dir = root / source_hash[:12]
    copy_dir.mkdir(parents=True, exist_ok=True)
    copy_path = copy_dir / source.name
    if copy_path.resolve() == source:
        raise ValueError("Safe retest copy path must not equal the source artifact")
    shutil.copy2(source, copy_path)

    classified = classify_corpus_result(result or {})
    classified.update(
        {
            "source_artifact": str(source),
            "source_artifact_hash": source_hash,
            "artifact_copy": str(copy_path),
            "output_root": str(copy_dir),
            "mutation_target": str(copy_path),
        }
    )
    return classified


def _runtime_status_from_retarget(result: dict[str, Any]) -> str:
    if result.get("retarget_status") == "failed" or result.get("retarget_error_code"):
        return "failed"
    return "success"


def _legacy_error_code(result: dict[str, Any]) -> str | None:
    code = result.get("retarget_error_code") or result.get("legacy_error_code")
    return str(code) if code else None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_stabilization_regression(pre_max_slide: float | None, post_max_slide: float | None, threshold: float) -> bool:
    if post_max_slide is None:
        return False
    if pre_max_slide is not None and post_max_slide > pre_max_slide:
        return True
    return post_max_slide > threshold


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
