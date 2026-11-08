#!/usr/bin/env python3
"""Kimodo runtime-core generator shell for Modly."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
import uuid
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Modly executes extension entrypoints via runpy/importlib, which may not put
# this directory on sys.path. Make sibling modules importable without relying
# on the caller's working directory or modifying any global Python settings.
_EXTENSION_ROOT = Path(__file__).resolve().parent
if str(_EXTENSION_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXTENSION_ROOT))

from host_compat import EXPERIMENTAL, UNSUPPORTED, resolve_host_compatibility
from glb_animation_exporter import GlbAnimationExporter
from glb_world_space import extract_world_foot_samples
from foot_stabilization import foot_stabilization_metadata, result_from_metadata, stabilize_feet, update_stabilization_residuals
from kimodo_bvh import locate_bvh, parse_bvh
from motion_evidence import classify_motion_evidence
from retarget_errors import MAPPING_INCOMPATIBLE, RetargetError
from retarget_metadata import _humanoid_contract_metadata, retarget_failure_metadata, retarget_success_metadata
from retargeting import build_retarget_clip, load_motion_context, visual_quality_diagnostics
from rig_inspector import inspect_rig

try:
    from services.generators.base import BaseGenerator
except ModuleNotFoundError:
    class BaseGenerator:  # type: ignore[override]
        def __init__(self, model_dir: Path, outputs_dir: Path) -> None:
            self.model_dir = model_dir
            self.outputs_dir = outputs_dir
            self._params_schema: list[dict[str, Any]] = []


EXT_DIR = Path(__file__).resolve().parent
SETUP_SUMMARY_FILE = ".kimodo-runtime-core-setup.json"
KIMODO_MODEL_NAME = "Kimodo-SOMA-RP-v1"
KIMODO_DOWNLOAD_CHECK = "config.yaml"
MOTION_BUNDLE_VERSION = "kimodo-runtime-core/v1"
PREVIEW_NODE_ID = "text-to-motion-preview"
ANIMATE_RIGGED_MESH_NODE_ID = "animate-rigged-mesh"
ALLOWED_RIGGED_MESH_SUFFIXES = {".glb", ".gltf"}


def read_setup_summary() -> dict[str, Any] | None:
    path = EXT_DIR / SETUP_SUMMARY_FILE
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def discover_hf_token() -> bool:
    env_keys = (
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "HUGGINGFACE_HUB_TOKEN",
    )
    for key in env_keys:
        value = os.environ.get(key)
        if isinstance(value, str) and value.strip():
            return True
    return (Path.home() / ".cache" / "huggingface" / "token").is_file()


def runtime_available() -> bool:
    return importlib.util.find_spec("kimodo") is not None


def resolve_runtime_host(summary: dict[str, Any] | None = None):
    payload = summary or {}
    host_facts = payload.get("host_facts") if isinstance(payload.get("host_facts"), dict) else {}
    gpu_sm = safe_int(host_facts.get("gpu_sm", payload.get("gpu_sm", 0)))
    cuda_version = safe_int(host_facts.get("cuda_version", payload.get("cuda_version", 0)))
    python_tag = host_facts.get("python_tag") if isinstance(host_facts.get("python_tag"), str) else None
    gpu_name = host_facts.get("gpu_name") if isinstance(host_facts.get("gpu_name"), str) else None
    return resolve_host_compatibility(
        python_tag=python_tag,
        gpu_sm=gpu_sm,
        cuda_version=cuda_version,
        gpu_name=gpu_name,
    )


def safe_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def runtime_reason(summary: dict[str, Any] | None = None) -> str | None:
    facts, compatibility = resolve_runtime_host(summary)
    if compatibility.state == UNSUPPORTED:
        return compatibility.reason
    if not runtime_available():
        if compatibility.state == EXPERIMENTAL:
            return (
                "Experimental host lane selected for Kimodo, but the runtime is not installed in the extension venv yet. "
                "Run setup.py first."
            )
        return f"Kimodo runtime is not installed in the extension venv for host {facts.platform_key}. Run setup.py first."
    return None


class KimodoRuntimeError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code


class KimodoGenerator(BaseGenerator):
    def __init__(self, model_dir: Path, outputs_dir: Path) -> None:
        super().__init__(model_dir, outputs_dir)
        self._loaded = False

    @classmethod
    def params_schema(cls) -> list[dict[str, Any]]:
        return []

    def load(self) -> None:
        summary = read_setup_summary()
        reason = runtime_reason(summary)
        if reason is not None:
            raise KimodoRuntimeError("RUNTIME_UNAVAILABLE", reason)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    def is_loaded(self) -> bool:
        return self._loaded

    def readiness_status(self) -> dict[str, Any]:
        summary = read_setup_summary()
        facts, compatibility = resolve_runtime_host(summary)
        checked_at = datetime.now(timezone.utc).isoformat()
        summary_status = summary.get("status", "ready") if isinstance(summary, dict) else "ready"
        token_present = discover_hf_token()
        runtime_ready = runtime_available()
        experimental = compatibility.state == EXPERIMENTAL

        if compatibility.state == UNSUPPORTED:
            return {
                "ok": False,
                "machine_code": "preflight/unsupported_platform",
                "label_hint": "Unsupported",
                "reason": compatibility.reason,
                "checked_at": checked_at,
                "details": {
                    "platform_key": facts.platform_key,
                    "platform_supported": "false",
                    "extension_setup_state": "unsupported_platform",
                    "readiness_source": "generator-shell",
                    "diagnostic_status": "blocking",
                    "last_checked_at": checked_at,
                },
            }

        if not runtime_ready:
            return {
                "ok": False,
                "machine_code": "preflight/runtime_missing",
                "label_hint": "Setup Runtime",
                "reason": runtime_reason(summary),
                "checked_at": checked_at,
                "details": {
                    "platform_key": facts.platform_key,
                    "platform_supported": "true",
                    "extension_setup_state": "experimental_runtime_missing" if experimental else "runtime_missing",
                    "readiness_source": "generator-shell",
                    "diagnostic_status": "blocking",
                    "last_checked_at": checked_at,
                },
            }

        if not token_present:
            return {
                "ok": False,
                "machine_code": "preflight/hf_token_missing",
                "label_hint": "Login",
                "reason": "A Hugging Face token is required for Kimodo's gated text encoder runtime.",
                "checked_at": checked_at,
                "details": {
                    "platform_key": facts.platform_key,
                    "platform_supported": "true",
                    "auth_state": "missing_hf_token",
                    "extension_setup_state": "experimental_runtime_installed" if experimental else "runtime_installed",
                    "readiness_source": "generator-shell",
                    "diagnostic_status": "blocking",
                    "last_checked_at": checked_at,
                },
            }

        if resolve_checkpoint_root(self.model_dir) is None:
            return {
                "ok": False,
                "machine_code": "preflight/model_weights_missing",
                "checked_at": checked_at,
                "details": {
                    "diagnostics": {
                        "platform_key": facts.platform_key,
                        "platform_supported": "true",
                        "extension_setup_state": "experimental_runtime_ready" if experimental else str(summary_status),
                        "readiness_source": "generator-shell",
                        "diagnostic_status": "waiting_for_download",
                        "last_checked_at": checked_at,
                    }
                },
            }

        return {
            "ok": True,
            "machine_code": "ready/experimental" if experimental else "ready",
            "checked_at": checked_at,
            "details": {
                "diagnostics": {
                    "platform_key": facts.platform_key,
                    "platform_supported": "true",
                    "auth_state": "hf_token_present",
                    "extension_setup_state": "experimental_runtime_ready" if experimental else str(summary_status),
                    "readiness_source": "generator-shell",
                    "diagnostic_status": "experimental" if experimental else "ready",
                    "last_checked_at": checked_at,
                }
            },
        }

    def generate(
        self,
        image_bytes: bytes,
        params: dict[str, Any],
        progress_cb=None,
        cancel_event=None,
    ) -> Path:
        del image_bytes, cancel_event
        if not self._loaded:
            self.load()
        if not discover_hf_token():
            raise KimodoRuntimeError(
                "HF_TOKEN_MISSING",
                "A Hugging Face token is required before running Kimodo generation.",
            )

        request = normalize_request(params)
        bundle_dir = create_bundle_dir(self.outputs_dir)
        if progress_cb:
            progress_cb(5, "Preparing Kimodo runtime…")

        checkpoint_root = resolve_checkpoint_root(self.model_dir)
        if checkpoint_root is None:
            raise KimodoRuntimeError(
                "MODEL_WEIGHTS_MISSING",
                "Kimodo model weights are not downloaded. Use Modly's Download button for this node first.",
            )

        completed = run_kimodo_cli(bundle_dir, request, checkpoint_root, progress_cb)
        first_npz = find_primary_npz(bundle_dir, completed["output_stem_name"], completed["num_samples"])
        preview_path = bundle_dir / "preview.glb"

        if progress_cb:
            progress_cb(80, "Building compatibility preview…")
        build_preview_glb(first_npz, preview_path)

        metadata_path = bundle_dir / "metadata.json"
        write_bundle_metadata(metadata_path, bundle_dir, preview_path, first_npz, request)

        if request["node_id"] == ANIMATE_RIGGED_MESH_NODE_ID:
            animated_path = retarget_motion_bundle(bundle_dir, metadata_path, request, completed)
            if progress_cb:
                progress_cb(100, "Kimodo retarget/export complete")
            return animated_path

        if progress_cb:
            progress_cb(100, "Kimodo runtime complete")
        return preview_path


def normalize_request(params: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(params, dict):
        raise KimodoRuntimeError("INVALID_INPUT", "params must be a JSON object.")

    prompt = params.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise KimodoRuntimeError("INVALID_INPUT", "prompt is required for Kimodo text generation.")

    duration = params.get("duration", 5.0)
    if not isinstance(duration, (int, float)) or duration <= 0 or duration > 10:
        raise KimodoRuntimeError("INVALID_INPUT", "duration must be a number between 0 and 10 seconds.")

    diffusion_steps = params.get("diffusion_steps", 100)
    if not isinstance(diffusion_steps, int) or diffusion_steps <= 0:
        raise KimodoRuntimeError("INVALID_INPUT", "diffusion_steps must be a positive integer.")

    num_samples = params.get("num_samples", 1)
    if not isinstance(num_samples, int) or num_samples < 1 or num_samples > 4:
        raise KimodoRuntimeError("INVALID_INPUT", "num_samples must be an integer between 1 and 4.")

    seed = params.get("seed", -1)
    if not isinstance(seed, int):
        raise KimodoRuntimeError("INVALID_INPUT", "seed must be an integer.")

    export_bvh = params.get("export_bvh", False)
    if not isinstance(export_bvh, bool):
        raise KimodoRuntimeError("INVALID_INPUT", "export_bvh must be a boolean.")

    disable_postprocess = params.get("disable_postprocess", True)
    if not isinstance(disable_postprocess, bool):
        raise KimodoRuntimeError("INVALID_INPUT", "disable_postprocess must be a boolean.")
    if not disable_postprocess:
        raise KimodoRuntimeError(
            "INVALID_INPUT",
            "kimodo-runtime-core does not support postprocess yet because setup intentionally skips MotionCorrection in this phase.",
        )

    text_encoder_device = params.get("text_encoder_device", "cpu")
    if text_encoder_device not in {"cpu", "auto", "cuda"}:
        raise KimodoRuntimeError("INVALID_INPUT", "text_encoder_device must be one of: cpu, auto, cuda.")

    node_id = normalize_node_id(params)
    constraints_path = resolve_constraints_path(params.get("constraints_path", ""))
    rigged_mesh_path = None
    if node_id == ANIMATE_RIGGED_MESH_NODE_ID:
        rigged_mesh_path = resolve_rigged_mesh_path(params.get("rigged_mesh_path", params.get("mesh_path", "")))
        export_bvh = True

    return {
        "node_id": node_id,
        "prompt": prompt.strip(),
        "duration": float(duration),
        "diffusion_steps": diffusion_steps,
        "num_samples": num_samples,
        "seed": seed,
        "constraints_path": constraints_path,
        "export_bvh": export_bvh,
        "disable_postprocess": disable_postprocess,
        "text_encoder_device": text_encoder_device,
        "rigged_mesh_path": rigged_mesh_path,
        "rigged_mesh_source_kind": normalize_rigged_mesh_source_kind(params),
    }


def normalize_rigged_mesh_source_kind(params: dict[str, Any]) -> str:
    raw = params.get("rigged_mesh_source_kind", params.get("source_kind", "unknown_mesh_source"))
    if not isinstance(raw, str):
        return "unknown_mesh_source"
    value = raw.strip()
    if value in {"unirig_recommended", "load_mesh_existing", "unknown_mesh_source"}:
        return value
    return "unknown_mesh_source"


def normalize_node_id(params: dict[str, Any]) -> str:
    raw_node_id = params.get("node_id") or params.get("model_node_id") or params.get("capability_id") or os.environ.get("MODEL_NODE_ID")
    if isinstance(raw_node_id, str) and raw_node_id.strip():
        value = raw_node_id.strip()
        if value.endswith(f"/{ANIMATE_RIGGED_MESH_NODE_ID}") or value == ANIMATE_RIGGED_MESH_NODE_ID:
            return ANIMATE_RIGGED_MESH_NODE_ID
        if value.endswith(f"/{PREVIEW_NODE_ID}") or value == PREVIEW_NODE_ID:
            return PREVIEW_NODE_ID

    raw_model_id = params.get("model_id")
    if isinstance(raw_model_id, str) and raw_model_id.strip():
        if raw_model_id.strip().endswith(f"/{ANIMATE_RIGGED_MESH_NODE_ID}"):
            return ANIMATE_RIGGED_MESH_NODE_ID

    return PREVIEW_NODE_ID


def resolve_constraints_path(raw: Any) -> Path | None:
    if raw in (None, ""):
        return None
    if not isinstance(raw, str):
        raise KimodoRuntimeError("INVALID_INPUT", "constraints_path must be a string path when provided.")

    candidate = Path(raw.strip()).expanduser()
    candidates = []
    if candidate.is_absolute():
        candidates.append(candidate)
    else:
        workspace_dir = os.environ.get("WORKSPACE_DIR")
        if isinstance(workspace_dir, str) and workspace_dir.strip():
            candidates.append(Path(workspace_dir.strip()) / candidate)
        candidates.append(Path.cwd() / candidate)
        candidates.append(EXT_DIR / candidate)

    for resolved in candidates:
        if resolved.is_file():
            return resolved.resolve()

    searched = ", ".join(str(path) for path in candidates)
    raise KimodoRuntimeError(
        "INVALID_INPUT",
        f"constraints_path could not be resolved to an existing file. Searched: {searched}",
    )


def resolve_rigged_mesh_path(raw: Any) -> Path:
    if raw in (None, ""):
        raise KimodoRuntimeError(
            "INVALID_INPUT",
            "rigged_mesh_path is required for Animate Rigged Mesh. Connect a rigged_mesh input or provide mesh_path as an alias.",
        )
    if not isinstance(raw, str):
        raise KimodoRuntimeError("INVALID_INPUT", "rigged_mesh_path must be a string path when provided.")

    candidate = Path(raw.strip()).expanduser()
    if candidate.suffix.lower() not in ALLOWED_RIGGED_MESH_SUFFIXES:
        allowed = ", ".join(sorted(ALLOWED_RIGGED_MESH_SUFFIXES))
        raise KimodoRuntimeError("INVALID_INPUT", f"rigged_mesh_path must point to a supported mesh file: {allowed}.")

    candidates = []
    if candidate.is_absolute():
        candidates.append(candidate)
    else:
        workspace_dir = os.environ.get("WORKSPACE_DIR")
        if isinstance(workspace_dir, str) and workspace_dir.strip():
            candidates.append(Path(workspace_dir.strip()) / candidate)
        candidates.append(Path.cwd() / candidate)
        candidates.append(EXT_DIR / candidate)

    for resolved in candidates:
        if resolved.is_file():
            return resolved.resolve()

    searched = ", ".join(str(path) for path in candidates)
    raise KimodoRuntimeError(
        "INVALID_INPUT",
        f"rigged_mesh_path could not be resolved to an existing .glb/.gltf file. Searched: {searched}",
    )


def create_bundle_dir(outputs_dir: Path) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    bundle_dir = outputs_dir / f"kimodo-{stamp}-{uuid.uuid4().hex[:8]}"
    bundle_dir.mkdir(parents=True, exist_ok=False)
    return bundle_dir


def kimodo_command(output_stem: Path, request: dict[str, Any]) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "kimodo.scripts.generate",
        request["prompt"],
        "--model",
        KIMODO_MODEL_NAME,
        "--duration",
        str(request["duration"]),
        "--diffusion_steps",
        str(request["diffusion_steps"]),
        "--num_samples",
        str(request["num_samples"]),
        "--output",
        str(output_stem),
    ]
    if request["constraints_path"] is not None:
        command.extend(["--constraints", str(request["constraints_path"])])
    if request["export_bvh"]:
        command.append("--bvh")
    if request["disable_postprocess"]:
        command.append("--no-postprocess")
    if request["seed"] >= 0:
        command.extend(["--seed", str(request["seed"])])
    return command


def resolve_checkpoint_root(model_dir: Path | None) -> Path | None:
    if model_dir is None:
        return None

    model_dir = Path(model_dir)
    if (model_dir / KIMODO_DOWNLOAD_CHECK).is_file():
        return model_dir.parent

    nested_model_dir = model_dir / KIMODO_MODEL_NAME
    if (nested_model_dir / KIMODO_DOWNLOAD_CHECK).is_file():
        return model_dir

    return None


def run_kimodo_cli(
    bundle_dir: Path,
    request: dict[str, Any],
    checkpoint_root: Path,
    progress_cb=None,
) -> dict[str, Any]:
    output_stem = bundle_dir / "motion"
    env = os.environ.copy()
    env["CHECKPOINT_DIR"] = str(checkpoint_root)
    if request["text_encoder_device"] in {"cpu", "cuda"}:
        env["TEXT_ENCODER_DEVICE"] = request["text_encoder_device"]

    command = kimodo_command(output_stem, request)
    if progress_cb:
        progress_cb(20, "Running Kimodo CLI…")
    result = subprocess.run(
        command,
        cwd=str(bundle_dir),
        env=env,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        stderr_lines = tail_lines(result.stderr, 20)
        stdout_lines = tail_lines(result.stdout, 10)
        detail_parts = []
        if stderr_lines:
            detail_parts.append("stderr:\n" + stderr_lines)
        if stdout_lines:
            detail_parts.append("stdout:\n" + stdout_lines)
        detail = "\n\n".join(detail_parts) if detail_parts else "Kimodo CLI exited without diagnostic output."
        raise KimodoRuntimeError(
            "GENERATION_FAILED",
            "Kimodo CLI failed during runtime-core generation.\n\n" + detail,
        )

    return {
        "output_stem_name": output_stem.name,
        "num_samples": request["num_samples"],
    }


def tail_lines(value: str, limit: int) -> str:
    lines = [line for line in value.strip().splitlines() if line.strip()]
    return "\n".join(lines[-limit:])


def find_primary_npz(bundle_dir: Path, output_stem_name: str, num_samples: int) -> Path:
    if num_samples == 1:
        candidate = bundle_dir / f"{output_stem_name}.npz"
        if candidate.is_file():
            return candidate
    else:
        sample_dir = bundle_dir / output_stem_name
        if sample_dir.is_dir():
            npz_files = sorted(sample_dir.glob("*.npz"))
            if npz_files:
                return npz_files[0]

    npz_files = sorted(bundle_dir.rglob("*.npz"))
    if npz_files:
        return npz_files[0]

    raise KimodoRuntimeError(
        "MOTION_BUNDLE_MISSING",
        "Kimodo completed but no NPZ artifact was found in the generated bundle.",
    )


def motion_context_npz_path(bundle_dir: Path, metadata_path: Path, completed: dict[str, Any]) -> Path | None:
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        payload = {}
    canonical = payload.get("canonical_motion_artifact") if isinstance(payload, dict) else None
    if isinstance(canonical, str) and canonical.strip():
        candidate = bundle_dir / canonical
        if candidate.is_file():
            return candidate
    try:
        return find_primary_npz(bundle_dir, str(completed.get("output_stem_name") or "motion"), safe_int(completed.get("num_samples", 1)) or 1)
    except KimodoRuntimeError:
        return None


def retarget_motion_bundle(
    bundle_dir: Path,
    metadata_path: Path,
    request: dict[str, Any],
    completed: dict[str, Any],
) -> Path:
    """Retarget Kimodo BVH motion onto the supplied rig and export a real animated GLB."""

    output_path = bundle_dir / "animated.glb"
    diagnostics = {
        "bundle_dir": str(bundle_dir),
        "metadata_path": str(metadata_path),
        "rigged_mesh_path": str(request["rigged_mesh_path"]),
        "source_kind": request.get("rigged_mesh_source_kind", "unknown_mesh_source"),
        "output_stem_name": completed.get("output_stem_name"),
        "num_samples": completed.get("num_samples"),
    }
    try:
        bvh_path = locate_bvh(bundle_dir, str(completed.get("output_stem_name") or "motion"))
        diagnostics["bvh_path"] = str(bvh_path)
        motion = parse_bvh(bvh_path)
        rig = inspect_rig(
            request["rigged_mesh_path"],
            source_kind=str(request.get("rigged_mesh_source_kind") or "unknown_mesh_source"),
        )
        humanoid_contract = _humanoid_contract_metadata(rig)
        if humanoid_contract:
            diagnostics["humanoid_contract"] = humanoid_contract
        motion_context_path = motion_context_npz_path(bundle_dir, metadata_path, completed)
        motion_context = load_motion_context(motion_context_path)
        if motion_context_path is not None:
            diagnostics["motion_context_npz"] = str(motion_context_path)
        clip = build_retarget_clip(motion, rig, motion_context=motion_context)
        motion_classification = _classify_motion_for_stabilization(request=request, clip=clip, rig=rig, motion_context=motion_context)
        diagnostics["motion_evidence_classification"] = motion_classification
        clip, stabilization_result = stabilize_feet(clip, rig, motion_context, motion_class=motion_classification["class"])
        diagnostics["foot_stabilization"] = foot_stabilization_metadata(stabilization_result)
        diagnostics["mapping_confidence"] = clip.mapping.confidence
        animated_path = GlbAnimationExporter().write_animation(rig, clip, output_path)
        clip = _enrich_visual_quality_after_export(animated_path, rig, clip, motion_context, diagnostics)
        update_bundle_metadata(
            metadata_path,
            retarget_success_metadata(animated_path, rig=rig, clip=clip, warnings=clip.warnings, diagnostics=diagnostics),
            bundle_dir=bundle_dir,
        )
        return animated_path
    except RetargetError as exc:
        failure_metadata = retarget_failure_metadata(exc, diagnostics=diagnostics)
        fallback_path = bundle_dir / "preview.glb"
        if exc.code == MAPPING_INCOMPATIBLE and fallback_path.is_file():
            update_bundle_metadata(
                metadata_path,
                {
                    **failure_metadata,
                    "runtime_status": "success_with_fallback",
                    "stabilization_status": "not_evaluated",
                    "fallback_output": str(fallback_path.relative_to(bundle_dir)),
                    "fallback_reason": "retarget_mapping_incompatible",
                },
                bundle_dir=bundle_dir,
            )
            return fallback_path
        update_bundle_metadata(
            metadata_path,
            failure_metadata,
            bundle_dir=bundle_dir,
        )
        raise KimodoRuntimeError(
            exc.code,
            f"Animate Rigged Mesh could not produce a real animated GLB: {exc.message}. Metadata was written to {metadata_path}.",
        ) from exc


def _enrich_visual_quality_after_export(animated_path: Path, rig: Any, clip: Any, motion_context: Any, diagnostics: dict[str, Any]) -> Any:
    """Enrich visual diagnostics from exported GLB without changing export success."""

    try:
        expected_frame_count = len(motion_context.foot_contacts) if motion_context is not None and motion_context.foot_contacts is not None else None
        extraction = extract_world_foot_samples(animated_path, getattr(rig, "humanoid_roles", ()), expected_frame_count=expected_frame_count)
        diagnostics["world_space_foot_samples"] = {
            "status": extraction.status,
            "reason": extraction.reason,
            "details": extraction.details,
        }
        samples = extraction.samples if extraction.status == "success" else None
        details = {"status": extraction.status, "reason": extraction.reason, "details": extraction.details}
        visual_quality = visual_quality_diagnostics(
            clip,
            motion_context=motion_context,
            world_foot_samples=samples,
            world_foot_sample_details=details,
        )
        existing_stabilization = getattr(clip, "visual_quality", {}) or {}
        stabilization = existing_stabilization.get("foot_stabilization") if isinstance(existing_stabilization, dict) else None
        if isinstance(stabilization, dict):
            result = result_from_metadata(stabilization)
            foot_contact = visual_quality.get("checks", {}).get("foot_contact") if isinstance(visual_quality.get("checks"), dict) else None
            post_max_slide = foot_contact.get("max_slide") if isinstance(foot_contact, dict) and foot_contact.get("confidence") == "measured" else None
            visual_quality["foot_stabilization"] = foot_stabilization_metadata(update_stabilization_residuals(result, post_max_slide=post_max_slide))
        return replace(clip, visual_quality=visual_quality)
    except Exception as exc:  # noqa: BLE001 - post-export diagnostics must never block success
        diagnostics["world_space_foot_samples"] = {"status": "not_evaluated", "reason": "metrics_exception", "details": {"error": f"{type(exc).__name__}: {exc}"}}
        return clip


def _classify_motion_for_stabilization(*, request: dict[str, Any], clip: Any, rig: Any, motion_context: Any) -> dict[str, Any]:
    samples = None
    calibration = getattr(clip, "calibration", None)
    pre_export = calibration.get("pre_export_foot_samples") if isinstance(calibration, dict) else None
    if isinstance(pre_export, dict) and pre_export.get("status") == "success" and isinstance(pre_export.get("samples"), dict):
        samples = pre_export.get("samples")
    root_samples = getattr(motion_context, "root_samples", None) if motion_context is not None else None
    if root_samples is None:
        root_samples = getattr(clip, "root_translations", ())
    contacts = getattr(motion_context, "foot_contacts", None) if motion_context is not None else None
    return classify_motion_evidence(
        root_samples=root_samples,
        foot_samples=samples,
        foot_contacts=contacts,
        dynamic_intensity=None,
        trusted_leg_chains=_has_classifier_leg_chain_evidence(rig),
        prompt_hint=str(request.get("prompt") or "") or None,
    )


def _has_classifier_leg_chain_evidence(rig: Any) -> bool:
    if getattr(rig, "contract_status", "contract_missing_or_untrusted") != "trusted":
        return False
    chains = getattr(rig, "humanoid_chains", ()) or ()
    return any("leg" in str(getattr(chain, "name", "")).lower() and tuple(getattr(chain, "roles", ()) or ()) for chain in chains)


def update_bundle_metadata(metadata_path: Path, updates: dict[str, Any], *, bundle_dir: Path | None = None) -> None:
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    payload.update(updates)
    if bundle_dir is not None:
        payload["artifacts"] = [
            str(artifact.relative_to(bundle_dir))
            for artifact in sorted(bundle_dir.rglob("*"))
            if artifact.is_file()
        ]
    metadata_path.write_text(f"{json.dumps(payload, indent=2)}\n", encoding="utf-8")


def build_preview_glb(npz_path: Path, preview_path: Path) -> None:
    try:
        import numpy as np
        import trimesh
    except ImportError as exc:
        raise KimodoRuntimeError(
            "RUNTIME_UNAVAILABLE",
            "numpy and trimesh are required to build the Modly compatibility preview.",
        ) from exc

    trajectory = load_root_trajectory(npz_path)
    points = decimate_points(trajectory, max_points=96)
    scene = trimesh.Scene()

    for index, point in enumerate(points):
        radius = 0.04 if index in {0, len(points) - 1} else 0.02
        sphere = trimesh.creation.uv_sphere(radius=radius, count=[12, 12])
        sphere.apply_translation(point)
        if index == 0:
            color = [80, 220, 120, 255]
        elif index == len(points) - 1:
            color = [255, 120, 120, 255]
        else:
            color = [120, 160, 255, 255]
        sphere.visual.vertex_colors = color
        scene.add_geometry(sphere, node_name=f"point-{index}")

    for index in range(len(points) - 1):
        start = points[index]
        end = points[index + 1]
        if distance(start, end) <= 1e-6:
            continue
        segment = trimesh.creation.cylinder(radius=0.01, segment=[start, end], sections=12)
        segment.visual.vertex_colors = [180, 190, 255, 255]
        scene.add_geometry(segment, node_name=f"segment-{index}")

    scene.export(preview_path)


def load_root_trajectory(npz_path: Path):
    import numpy as np

    with np.load(npz_path, allow_pickle=False) as data:
        if "root_positions" in data:
            trajectory = data["root_positions"]
        elif "smooth_root_pos" in data:
            trajectory = data["smooth_root_pos"]
        elif "posed_joints" in data:
            posed = data["posed_joints"]
            if posed.ndim != 3 or posed.shape[1] < 1 or posed.shape[2] < 3:
                raise KimodoRuntimeError(
                    "PREVIEW_BUILD_FAILED",
                    "posed_joints was present but did not contain a usable root trajectory.",
                )
            trajectory = posed[:, 0, :3]
        else:
            raise KimodoRuntimeError(
                "PREVIEW_BUILD_FAILED",
                "No usable root trajectory was found in the Kimodo NPZ artifact.",
            )

    if trajectory.ndim != 2 or trajectory.shape[0] < 1 or trajectory.shape[1] < 3:
        raise KimodoRuntimeError(
            "PREVIEW_BUILD_FAILED",
            "The resolved root trajectory does not have shape [T, 3].",
        )
    return trajectory[:, :3]


def decimate_points(points, max_points: int):
    import numpy as np

    count = len(points)
    if count <= max_points:
        return points
    indices = np.linspace(0, count - 1, max_points, dtype=int)
    return points[indices]


def distance(start, end) -> float:
    import numpy as np

    return float(np.linalg.norm(end - start))


def write_bundle_metadata(
    metadata_path: Path,
    bundle_dir: Path,
    preview_path: Path,
    primary_npz: Path,
    request: dict[str, Any],
) -> None:
    artifacts = []
    for artifact in sorted(bundle_dir.rglob("*")):
        if artifact.is_file():
            artifacts.append(str(artifact.relative_to(bundle_dir)))

    payload = {
        "contract_version": MOTION_BUNDLE_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": KIMODO_MODEL_NAME,
        "node_id": request["node_id"],
        "prompt": request["prompt"],
        "source_prompt": request["prompt"],
        "source_rigged_mesh": str(request["rigged_mesh_path"]) if request["rigged_mesh_path"] else None,
        "semantics": "animated_mesh_candidate" if request["node_id"] == ANIMATE_RIGGED_MESH_NODE_ID else "motion_preview",
        "retarget_status": "requested" if request["node_id"] == ANIMATE_RIGGED_MESH_NODE_ID else "not_requested",
        "rigged_mesh_source_kind": request.get("rigged_mesh_source_kind"),
        "duration": request["duration"],
        "diffusion_steps": request["diffusion_steps"],
        "num_samples": request["num_samples"],
        "seed": request["seed"],
        "constraints_path": str(request["constraints_path"]) if request["constraints_path"] else None,
        "export_bvh": request["export_bvh"],
        "disable_postprocess": request["disable_postprocess"],
        "text_encoder_device": request["text_encoder_device"],
        "canonical_motion_artifact": str(primary_npz.relative_to(bundle_dir)),
        "preview_artifact": str(preview_path.relative_to(bundle_dir)),
        "artifacts": artifacts,
        "notes": [
            "preview.glb is a compatibility mesh preview for the current Modly surface.",
            "The canonical Kimodo output remains the motion bundle artifacts written alongside this preview."
        ],
    }
    metadata_path.write_text(f"{json.dumps(payload, indent=2)}\n", encoding="utf-8")
