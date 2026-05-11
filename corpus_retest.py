"""Corpus retest classification and safe-copy reporting helpers.

These helpers are intentionally small and filesystem-local: they prepare retest
inputs as copies and summarize existing synthetic/run metadata. They do not run
Modly workflows and never choose original artifacts as mutation/output targets.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import struct
from pathlib import Path
from typing import Any, Callable

from locomotion_axis import angle_degrees, axis_vector, dot_product, horizontal_dominance


SCHEMA = "kimodo.corpus_retest.v1"
SAFE_ANALYSIS_SCHEMA = "kimodo.corpus_retest.safe_analysis.v1"
WORKFLOW_AXIS_SCHEMA = "kimodo.workflow_axis_analysis.v1"
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


def prepare_safe_corpus_analysis(
    export_paths: list[str | Path] | tuple[str | Path, ...],
    *,
    temp_root: str | Path | None = None,
    analyzer: Callable[[Path], dict[str, Any] | None] | None = None,
) -> dict[str, Any]:
    """Copy GLB exports to a safe root and summarize read-only analysis metadata.

    The helper is deliberately filesystem-only: it never invokes Modly workflows,
    treats copied files as the only mutation targets, and verifies original hashes
    after analysis.
    """

    root = _safe_corpus_root(temp_root)
    assets = []
    for source_path in export_paths:
        source = Path(source_path).resolve()
        if source.suffix.lower() != ".glb":
            raise ValueError(f"Corpus analysis only accepts .glb exports: {source}")
        if not source.is_file():
            raise FileNotFoundError(source)

        source_hash_before = _sha256_file(source)
        copy_dir = root / source_hash_before[:12]
        copy_dir.mkdir(parents=True, exist_ok=True)
        copy_path = copy_dir / source.name
        if copy_path.resolve() == source:
            raise ValueError("Safe corpus copy path must not equal the source artifact")
        shutil.copy2(source, copy_path)
        copy_hash_before = _sha256_file(copy_path)

        analysis = dict(analyzer(copy_path) or {}) if analyzer is not None else {"analysis_status": "copied_not_analyzed"}
        source_hash_after = _sha256_file(source)
        copy_hash_after = _sha256_file(copy_path)
        analysis_status = str(analysis.pop("analysis_status", "analyzed" if analyzer is not None else "copied_not_analyzed"))

        entry = {
            "source_path": str(source),
            "copy_path": str(copy_path),
            "source_hash": source_hash_before,
            "copy_hash": copy_hash_after,
            "analysis_status": analysis_status,
            "mutation_guard": {
                "original_unchanged": source_hash_before == source_hash_after,
                "copy_hash_before": copy_hash_before,
                "copy_hash_after": copy_hash_after,
                "source_is_mutation_target": source == copy_path.resolve(),
                "mutation_target": str(copy_path),
            },
        }
        entry.update(analysis)
        assets.append(entry)

    return {
        "schema": SAFE_ANALYSIS_SCHEMA,
        "analysis_mode": "read_only_copy",
        "workflow_invocation": "none",
        "safe_root": str(root),
        "total": len(assets),
        "assets": assets,
    }


def prepare_safe_workflow_axis_analysis(
    workflow_bundle_paths: list[str | Path] | tuple[str | Path, ...],
    *,
    temp_root: str | Path | None = None,
) -> dict[str, Any]:
    """Copy Kimodo workflow bundles to a safe root and analyze copied evidence.

    Originals are only hashed/read for copy planning and post-analysis mutation
    guards. Metadata, BVH/NPZ motion, GLB previews, and referenced rigmeta
    sidecars are analyzed from the copied files under ``/tmp/opencode``.
    """

    root = _safe_corpus_root(temp_root or "/tmp/opencode/kimodo-workflow-axis-retest")
    bundles: list[dict[str, Any]] = []
    for bundle_path in workflow_bundle_paths:
        source_bundle = Path(bundle_path).resolve()
        if not source_bundle.is_dir():
            raise FileNotFoundError(source_bundle)
        metadata_path = source_bundle / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(metadata_path)

        metadata = _read_json_file(metadata_path)
        original_files = _workflow_files_to_copy(source_bundle, metadata)
        hashes_before = {path: _sha256_file(path) for path in original_files}
        copied_bundle = root / source_bundle.name
        copied_bundle.mkdir(parents=True, exist_ok=True)
        copied_files: dict[Path, Path] = {}

        for original in original_files:
            copy_path = _workflow_copy_path(original, source_bundle, copied_bundle, root)
            copy_path.parent.mkdir(parents=True, exist_ok=True)
            if copy_path.resolve() == original:
                raise ValueError("Safe workflow copy path must not equal an original artifact")
            shutil.copy2(original, copy_path)
            copied_files[original] = copy_path

        copied_metadata = copied_files[metadata_path]
        analysis = analyze_copied_workflow_axis(copied_metadata, source_bundle=source_bundle, copied_bundle=copied_bundle)
        hashes_after = {path: _sha256_file(path) for path in original_files}
        copy_hashes_after = {original: _sha256_file(copy_path) for original, copy_path in copied_files.items()}

        analysis.update(
            {
                "bundle_path": str(source_bundle),
                "copied_bundle_path": str(copied_bundle),
                "copied_files": {
                    str(original): {
                        "copy_path": str(copied_files[original]),
                        "original_hash_before": hashes_before[original],
                        "original_hash_after": hashes_after[original],
                        "copy_hash_after": copy_hashes_after[original],
                    }
                    for original in original_files
                },
                "mutation_guard": {
                    "all_originals_unchanged": all(hashes_before[path] == hashes_after[path] for path in original_files),
                    "analysis_used_copies_only": True,
                    "workflow_invocation": "none",
                    "original_files_hashed": len(original_files),
                },
            }
        )
        bundles.append(analysis)

    return {
        "schema": WORKFLOW_AXIS_SCHEMA,
        "analysis_mode": "read_only_copied_workflow_bundle",
        "workflow_invocation": "none",
        "safe_root": str(root),
        "total": len(bundles),
        "counts": _workflow_axis_counts(bundles),
        "bundles": bundles,
    }


def analyze_copied_workflow_axis(metadata_path: str | Path, *, source_bundle: Path | None = None, copied_bundle: Path | None = None) -> dict[str, Any]:
    """Analyze a copied workflow bundle's metadata and motion artifacts."""

    copied_metadata_path = Path(metadata_path).resolve()
    bundle_dir = Path(copied_bundle or copied_metadata_path.parent).resolve()
    metadata = _read_json_file(copied_metadata_path)
    diagnostics = metadata.get("diagnostics") if isinstance(metadata.get("diagnostics"), dict) else {}
    humanoid_contract = diagnostics.get("humanoid_contract") if isinstance(diagnostics.get("humanoid_contract"), dict) else {}
    copied_rigmeta_path = _copied_rigmeta_path(metadata, bundle_dir, source_bundle)
    rigmeta = _read_json_file(copied_rigmeta_path) if copied_rigmeta_path and copied_rigmeta_path.is_file() else None

    basis, basis_source = _workflow_basis(metadata, rigmeta)
    basis_status = str((basis or {}).get("status") or "unavailable").lower() if basis else "unavailable"
    roles = _workflow_roles(metadata, rigmeta)
    root_advance = _workflow_root_advance(bundle_dir, basis)
    classification, classification_reasons = _classify_workflow_axis(root_advance, basis)
    limitations = []
    reasons = []
    if basis is None:
        reasons.append({"code": "basis/unavailable"})
        limitations.append("basis metadata unavailable; classification remains conservative")
    if root_advance is None:
        reasons.append({"code": "evidence/no_root_motion"})
        limitations.append("no extractable BVH/NPZ root horizontal displacement found")
    reasons.extend(classification_reasons)
    correction_eligibility = _workflow_correction_eligibility(classification, basis_status)

    return {
        "prompt": metadata.get("prompt"),
        "source_prompt": metadata.get("source_prompt"),
        "statuses": {
            "runtime_status": metadata.get("runtime_status"),
            "retarget_status": metadata.get("retarget_status"),
            "contract_status": metadata.get("contract_status") or humanoid_contract.get("status"),
            "animation_mapping_status": metadata.get("animation_mapping_status"),
        },
        "warnings": list(metadata.get("warnings") or []),
        "source_rigged_mesh": metadata.get("source_rigged_mesh"),
        "humanoid_contract_path": humanoid_contract.get("path"),
        "roles": roles,
        "basis": basis,
        "basis_status": basis_status,
        "basis_source": basis_source,
        "root_advance": root_advance,
        "correction_eligibility": correction_eligibility,
        "classification": classification,
        "reasons": _unique_reason_codes(reasons),
        "limitations": limitations,
    }


def analyze_glb_locomotion_axis(glb_path: str | Path) -> dict[str, Any]:
    """Inspect a copied GLB read-only and report generic locomotion-axis evidence.

    The analyzer reads only local GLB JSON/BIN chunks. It does not invoke Modly,
    does not write beside the GLB, and deliberately keeps basis evidence
    unavailable when metadata cannot be recovered from glTF extras.
    """

    path = Path(glb_path).resolve()
    reasons: list[dict[str, Any]] = []
    limitations: list[str] = []
    try:
        gltf, bin_blob = _read_glb_chunks(path)
        contract_basis = _find_contract_basis(gltf)
        if contract_basis is None:
            basis_status = "unavailable"
            reasons.append({"code": "basis/unavailable"})
            limitations.append("basis metadata unavailable; classification remains conservative")
        else:
            basis_status = "available"
            reasons.append({"code": "basis/metadata_found"})

        candidates = _translation_channel_candidates(gltf, bin_blob)
        root_candidate = _select_root_candidate(candidates)
        if root_candidate is None:
            reasons.append({"code": "evidence/no_translation_animation_channels"})
            limitations.append("no extractable GLB translation animation channel found")
            return {
                "analysis_status": "not_evaluated",
                "basis_status": basis_status,
                "contract_basis": contract_basis,
                "root_advance": None,
                "classification": "not_evaluated",
                "reasons": reasons,
                "limitations": limitations,
            }

        root_advance = _root_advance_from_candidate(root_candidate, contract_basis)
        reasons.extend(root_advance.pop("reasons", []))
        classification = _classify_copied_glb_axis(root_advance, contract_basis)
        if classification == "ambiguous_locomotion" and contract_basis is not None:
            reasons.append({"code": "locomotion/ambiguous_root_advance"})

        return {
            "analysis_status": "analyzed",
            "basis_status": basis_status,
            "contract_basis": contract_basis,
            "root_advance": root_advance,
            "translation_candidates": candidates,
            "classification": classification,
            "reasons": _unique_reason_codes(reasons),
            "limitations": limitations,
        }
    except Exception as exc:  # pragma: no cover - defensive corpus reporting path
        return {
            "analysis_status": "not_evaluated",
            "basis_status": "unavailable",
            "contract_basis": None,
            "root_advance": None,
            "classification": "not_evaluated",
            "reasons": [{"code": "analysis/error", "message": str(exc)}],
            "limitations": ["GLB parser could not extract axis evidence"],
        }


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


def _safe_corpus_root(temp_root: str | Path | None) -> Path:
    root = Path(temp_root or "/tmp/opencode/kimodo-corpus-retest").resolve()
    opencode_root = Path("/tmp/opencode").resolve()
    if _is_relative_to(root, opencode_root):
        return root

    tmp_root = Path(os.environ.get("TMPDIR") or "/tmp").resolve()
    if _is_relative_to(root, tmp_root) and "opencode" in {part.lower() for part in root.parts}:
        return root
    raise ValueError(f"unsafe safe corpus root outside /tmp/opencode-like area: {root}")


def _read_json_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def _workflow_files_to_copy(source_bundle: Path, metadata: dict[str, Any]) -> list[Path]:
    candidates: list[Path] = []
    for name in ("metadata.json", "motion.bvh", "motion.npz", "animated.glb", "preview.glb"):
        path = source_bundle / name
        if path.is_file():
            candidates.append(path.resolve())
    for artifact in metadata.get("artifacts") or []:
        path = source_bundle / str(artifact)
        if path.is_file():
            candidates.append(path.resolve())
    diagnostics = metadata.get("diagnostics") if isinstance(metadata.get("diagnostics"), dict) else {}
    humanoid_contract = diagnostics.get("humanoid_contract") if isinstance(diagnostics.get("humanoid_contract"), dict) else {}
    rigmeta_path = humanoid_contract.get("path")
    if rigmeta_path:
        path = Path(str(rigmeta_path)).expanduser().resolve()
        if path.is_file():
            candidates.append(path)
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return unique


def _workflow_copy_path(original: Path, source_bundle: Path, copied_bundle: Path, safe_root: Path) -> Path:
    if _is_relative_to(original, source_bundle):
        return copied_bundle / original.relative_to(source_bundle)
    return safe_root / "referenced" / original.name


def _copied_rigmeta_path(metadata: dict[str, Any], bundle_dir: Path, source_bundle: Path | None) -> Path | None:
    diagnostics = metadata.get("diagnostics") if isinstance(metadata.get("diagnostics"), dict) else {}
    humanoid_contract = diagnostics.get("humanoid_contract") if isinstance(diagnostics.get("humanoid_contract"), dict) else {}
    original_path_value = humanoid_contract.get("path")
    if not original_path_value:
        return None
    original = Path(str(original_path_value)).expanduser().resolve()
    if source_bundle and _is_relative_to(original, source_bundle):
        return bundle_dir / original.relative_to(source_bundle)
    return bundle_dir.parent / "referenced" / original.name


def _workflow_basis(metadata: dict[str, Any], rigmeta: dict[str, Any] | None) -> tuple[dict[str, Any] | None, str]:
    metadata_basis = _extract_basis_from_mapping(metadata)
    if metadata_basis:
        return _normalize_workflow_basis(metadata_basis), "metadata"
    rigmeta_basis = _extract_basis_from_mapping(rigmeta or {})
    if rigmeta_basis:
        return _normalize_workflow_basis(rigmeta_basis), "rigmeta"
    return None, "unavailable"


def _extract_basis_from_mapping(payload: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    for key in ("basis", "contract_basis", "humanoid_basis"):
        value = payload.get(key)
        if isinstance(value, dict):
            return value
    humanoid_contract = payload.get("humanoid_contract")
    if isinstance(humanoid_contract, dict):
        nested = _extract_basis_from_mapping(humanoid_contract)
        if nested:
            return nested
    diagnostics = payload.get("diagnostics")
    if isinstance(diagnostics, dict):
        nested_contract = diagnostics.get("humanoid_contract")
        if isinstance(nested_contract, dict):
            nested = _extract_basis_from_mapping(nested_contract)
            if nested:
                return nested
    return None


def _normalize_workflow_basis(basis: dict[str, Any]) -> dict[str, Any]:
    return {
        "forward": _signed_axis_string(basis.get("forward")),
        "up": _signed_axis_string(basis.get("up")),
        "handedness": basis.get("handedness"),
        "status": str(basis.get("status") or basis.get("provenance") or "unknown").lower(),
    }


def _signed_axis_string(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    token = value.strip().upper()
    if token in {"X", "Y", "Z"}:
        return f"+{token}"
    if token in {"+X", "-X", "+Y", "-Y", "+Z", "-Z"}:
        return token
    return None


def _workflow_roles(metadata: dict[str, Any], rigmeta: dict[str, Any] | None) -> dict[str, Any]:
    diagnostics = metadata.get("diagnostics") if isinstance(metadata.get("diagnostics"), dict) else {}
    humanoid_contract = diagnostics.get("humanoid_contract") if isinstance(diagnostics.get("humanoid_contract"), dict) else {}
    roles = humanoid_contract.get("roles")
    if isinstance(roles, list):
        return {str(item.get("role")): item for item in roles if isinstance(item, dict) and item.get("role")}
    rig_contract = (rigmeta or {}).get("humanoid_contract") if isinstance((rigmeta or {}).get("humanoid_contract"), dict) else {}
    rig_roles = rig_contract.get("roles")
    return rig_roles if isinstance(rig_roles, dict) else {}


def _workflow_root_advance(bundle_dir: Path, basis: dict[str, Any] | None) -> dict[str, Any] | None:
    bvh_payload = None
    bvh_path = bundle_dir / "motion.bvh"
    if bvh_path.is_file():
        bvh_positions = _read_bvh_root_positions(bvh_path)
        if bvh_positions:
            bvh_payload = _workflow_root_payload("motion.bvh", bvh_positions, basis)
            if bvh_payload.get("signed_axis"):
                return bvh_payload
    npz_path = bundle_dir / "motion.npz"
    if npz_path.is_file():
        npz_positions = _read_npz_root_positions(npz_path)
        if npz_positions:
            return _workflow_root_payload("motion.npz", npz_positions, basis)
    return bvh_payload


def _read_bvh_root_positions(path: Path) -> list[list[float]]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    channel_names: list[str] = []
    motion_start = None
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not channel_names and stripped.startswith("CHANNELS"):
            parts = stripped.split()
            count = int(parts[1])
            channel_names = parts[2 : 2 + count]
        if stripped == "MOTION":
            motion_start = index + 1
            break
    if motion_start is None or not channel_names:
        return []
    data_start = motion_start
    while data_start < len(lines) and not _looks_like_motion_values(lines[data_start]):
        data_start += 1
    indices = [_channel_index(channel_names, name) for name in ("Xposition", "Yposition", "Zposition")]
    if any(index is None for index in indices):
        return []
    positions = []
    for line in lines[data_start:]:
        if not _looks_like_motion_values(line):
            continue
        values = [float(part) for part in line.strip().split()]
        if len(values) <= max(index for index in indices if index is not None):
            continue
        positions.append([values[indices[0]], values[indices[1]], values[indices[2]]])
    return positions


def _channel_index(channels: list[str], name: str) -> int | None:
    try:
        return channels.index(name)
    except ValueError:
        return None


def _looks_like_motion_values(line: str) -> bool:
    stripped = line.strip()
    if not stripped or stripped.startswith(("Frames:", "Frame Time:")):
        return False
    try:
        [float(part) for part in stripped.split()]
        return True
    except ValueError:
        return False


def _read_npz_root_positions(path: Path) -> list[list[float]]:
    try:
        import numpy as np  # type: ignore
    except Exception:  # pragma: no cover - optional runtime dependency
        return []
    try:
        with np.load(path, allow_pickle=False) as data:
            for key in ("root_positions", "root_pos", "smooth_root_pos", "translations", "trans"):
                if key not in data:
                    continue
                array = data[key]
                if getattr(array, "ndim", 0) >= 2 and array.shape[-1] >= 3 and array.shape[0] >= 2:
                    return [[float(row[0]), float(row[1]), float(row[2])] for row in array.reshape((-1, array.shape[-1]))]
    except Exception:  # pragma: no cover - defensive corpus path
        return []
    return []


def _workflow_root_payload(source: str, positions: list[list[float]], basis: dict[str, Any] | None) -> dict[str, Any]:
    start = positions[0]
    end = positions[-1]
    displacement = [round(end[0] - start[0], 6), 0.0, round(end[2] - start[2], 6)]
    dominance = horizontal_dominance(displacement)
    magnitude = round(math.sqrt(displacement[0] ** 2 + displacement[2] ** 2), 6)
    payload = {
        "source": source,
        "sample_count": len(positions),
        "displacement_vector": displacement,
        "horizontal_magnitude": magnitude,
        "dominance": dominance.get("dominance"),
        "signed_axis": dominance.get("signed_axis"),
        "confidence": dominance.get("confidence", 0.0),
    }
    forward = (basis or {}).get("forward")
    if forward and payload["signed_axis"]:
        payload["basis_dot"] = dot_product(axis_vector(payload["signed_axis"]), axis_vector(forward))
        payload["basis_angle_degrees"] = angle_degrees(axis_vector(payload["signed_axis"]), axis_vector(forward))
    return payload


def _classify_workflow_axis(root_advance: dict[str, Any] | None, basis: dict[str, Any] | None) -> tuple[str, list[dict[str, Any]]]:
    if root_advance is None or not root_advance.get("signed_axis"):
        return "not_evaluated", [{"code": "classification/not_evaluated"}]
    if basis is None or not basis.get("forward"):
        return "ambiguous_locomotion", [{"code": "basis/unavailable"}]
    angle = root_advance.get("basis_angle_degrees")
    if angle is None:
        return "ambiguous_locomotion", [{"code": "basis/unavailable"}]
    if angle <= 25.0:
        return "forward_locomotion", [{"code": "locomotion/forward_root_advance"}]
    if 75.0 <= angle <= 105.0:
        return "lateral_locomotion", [{"code": "locomotion/lateral_root_advance"}, {"code": "locomotion/axis_mismatch"}]
    return "ambiguous_locomotion", [{"code": "locomotion/ambiguous_root_advance"}]


def _workflow_correction_eligibility(classification: str, basis_status: str) -> dict[str, Any]:
    policy = "trusted_or_asserted_basis_required"
    if basis_status == "inferred":
        return {"eligible": False, "reason": "basis/inferred_diagnostic_only", "policy": policy}
    if basis_status not in {"trusted", "asserted"}:
        return {"eligible": False, "reason": "basis/not_correction_eligible", "policy": policy}
    if classification == "lateral_locomotion":
        return {"eligible": True, "reason": "policy/trusted_basis_lateral_locomotion", "policy": policy}
    if classification == "forward_locomotion":
        return {"eligible": False, "reason": "classification/no_correction_needed", "policy": policy}
    return {"eligible": False, "reason": "classification/ambiguous_or_not_evaluated", "policy": policy}


def _workflow_axis_counts(bundles: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    counts = {"classification": {}, "basis_status": {}, "correction_eligible": {}}
    for bundle in bundles:
        for key, value in (
            ("classification", bundle.get("classification")),
            ("basis_status", bundle.get("basis_status")),
            ("correction_eligible", str(bool((bundle.get("correction_eligibility") or {}).get("eligible"))).lower()),
        ):
            bucket = str(value or "unknown")
            counts[key][bucket] = counts[key].get(bucket, 0) + 1
    return counts


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _read_glb_chunks(path: Path) -> tuple[dict[str, Any], bytes]:
    with path.open("rb") as handle:
        header = handle.read(12)
        if len(header) != 12:
            raise ValueError("invalid GLB header")
        magic, version, length = struct.unpack("<III", header)
        if magic != 0x46546C67 or version != 2:
            raise ValueError("unsupported GLB container")
        payload = handle.read(length - 12)

    offset = 0
    gltf: dict[str, Any] | None = None
    bin_blob = b""
    while offset + 8 <= len(payload):
        chunk_length, chunk_type = struct.unpack("<I4s", payload[offset : offset + 8])
        offset += 8
        chunk_data = payload[offset : offset + chunk_length]
        offset += chunk_length
        if chunk_type == b"JSON":
            gltf = json.loads(chunk_data.decode("utf-8"))
        elif chunk_type == b"BIN\x00":
            bin_blob = chunk_data
    if gltf is None:
        raise ValueError("GLB JSON chunk missing")
    return gltf, bin_blob


def _find_contract_basis(gltf: dict[str, Any]) -> dict[str, Any] | None:
    for extras in _iter_extras(gltf):
        basis = extras.get("contract_basis") or extras.get("basis") or extras.get("humanoid_basis")
        if isinstance(basis, dict):
            normalized = {
                "forward": basis.get("forward"),
                "up": basis.get("up"),
                "handedness": basis.get("handedness"),
                "status": basis.get("status") or basis.get("provenance") or "unknown",
            }
            if normalized["forward"] or normalized["up"] or normalized["handedness"]:
                return normalized
        if any(key in extras for key in ("forward", "up", "handedness")):
            return {
                "forward": extras.get("forward"),
                "up": extras.get("up"),
                "handedness": extras.get("handedness"),
                "status": extras.get("status") or extras.get("basis_status") or "unknown",
            }
    return None


def _iter_extras(value: Any):
    if isinstance(value, dict):
        extras = value.get("extras")
        if isinstance(extras, dict):
            yield extras
        for child in value.values():
            yield from _iter_extras(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_extras(child)


def _translation_channel_candidates(gltf: dict[str, Any], bin_blob: bytes) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    nodes = gltf.get("nodes") or []
    for animation_index, animation in enumerate(gltf.get("animations") or []):
        samplers = animation.get("samplers") or []
        for channel_index, channel in enumerate(animation.get("channels") or []):
            target = channel.get("target") or {}
            if target.get("path") != "translation":
                continue
            sampler_index = channel.get("sampler")
            if not isinstance(sampler_index, int) or sampler_index >= len(samplers):
                continue
            sampler = samplers[sampler_index]
            output_index = sampler.get("output")
            if not isinstance(output_index, int):
                continue
            translations = _read_accessor_vec3(gltf, bin_blob, output_index)
            if len(translations) < 2:
                continue
            node_index = target.get("node")
            node_name = None
            if isinstance(node_index, int) and node_index < len(nodes):
                node_name = str((nodes[node_index] or {}).get("name") or f"node_{node_index}")
            candidate = _candidate_payload(animation_index, channel_index, node_index, node_name, translations)
            if candidate["horizontal_magnitude"] > 0.0:
                candidates.append(candidate)
    return candidates


def _read_accessor_vec3(gltf: dict[str, Any], bin_blob: bytes, accessor_index: int) -> list[list[float]]:
    accessors = gltf.get("accessors") or []
    buffer_views = gltf.get("bufferViews") or []
    accessor = accessors[accessor_index]
    if accessor.get("componentType") != 5126 or accessor.get("type") != "VEC3":
        return []
    view = buffer_views[accessor["bufferView"]]
    byte_offset = int(view.get("byteOffset") or 0) + int(accessor.get("byteOffset") or 0)
    stride = int(view.get("byteStride") or 12)
    count = int(accessor.get("count") or 0)
    values = []
    for index in range(count):
        start = byte_offset + index * stride
        values.append([round(component, 6) for component in struct.unpack_from("<fff", bin_blob, start)])
    return values


def _candidate_payload(animation_index: int, channel_index: int, node_index: Any, node_name: str | None, translations: list[list[float]]) -> dict[str, Any]:
    start = translations[0]
    end = translations[-1]
    displacement = [round(end[0] - start[0], 6), 0.0, round(end[2] - start[2], 6)]
    magnitude = round(math.sqrt(displacement[0] ** 2 + displacement[2] ** 2), 6)
    dominance = horizontal_dominance(displacement)
    return {
        "animation_index": animation_index,
        "channel_index": channel_index,
        "node_index": node_index,
        "node_name": node_name,
        "sample_count": len(translations),
        "displacement_vector": displacement,
        "horizontal_magnitude": magnitude,
        "dominance": dominance.get("dominance"),
        "signed_axis": dominance.get("signed_axis"),
        "confidence": dominance.get("confidence", 0.0),
        "root_like": _is_root_like_node(node_name),
    }


def _select_root_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not candidates:
        return None
    return max(candidates, key=lambda item: (1 if item.get("root_like") else 0, item.get("horizontal_magnitude") or 0.0))


def _root_advance_from_candidate(candidate: dict[str, Any], contract_basis: dict[str, Any] | None) -> dict[str, Any]:
    root = {
        "source": "glb.animation.translation_channel",
        "node_name": candidate.get("node_name"),
        "node_index": candidate.get("node_index"),
        "animation_index": candidate.get("animation_index"),
        "channel_index": candidate.get("channel_index"),
        "sample_count": candidate.get("sample_count"),
        "displacement_vector": candidate.get("displacement_vector"),
        "horizontal_magnitude": candidate.get("horizontal_magnitude"),
        "dominance": candidate.get("dominance"),
        "signed_axis": candidate.get("signed_axis"),
        "confidence": candidate.get("confidence"),
    }
    reasons = []
    forward = (contract_basis or {}).get("forward")
    if forward and root["signed_axis"]:
        root_vector = axis_vector(root["signed_axis"])
        forward_vector = axis_vector(str(forward))
        root["dot_to_forward"] = dot_product(root_vector, forward_vector)
        root["angle_to_forward"] = angle_degrees(root_vector, forward_vector)
        if 75.0 <= root["angle_to_forward"] <= 105.0:
            reasons.append({"code": "locomotion/lateral_root_advance"})
    root["reasons"] = reasons
    return root


def _classify_copied_glb_axis(root_advance: dict[str, Any], contract_basis: dict[str, Any] | None) -> str:
    if not root_advance or not root_advance.get("signed_axis"):
        return "not_evaluated"
    if contract_basis is None or not contract_basis.get("forward"):
        return "ambiguous_locomotion"
    angle = root_advance.get("angle_to_forward")
    if angle is None:
        return "ambiguous_locomotion"
    if angle <= 25.0:
        return "forward_locomotion"
    if 75.0 <= angle <= 105.0:
        return "lateral_locomotion"
    return "ambiguous_locomotion"


def _is_root_like_node(node_name: str | None) -> bool:
    if not node_name:
        return False
    lowered = node_name.lower()
    return any(token in lowered for token in ("root", "hips", "pelvis", "armature"))


def _unique_reason_codes(reasons: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique = []
    seen = set()
    for reason in reasons:
        code = reason.get("code")
        if code in seen:
            continue
        seen.add(code)
        unique.append(reason)
    return unique
