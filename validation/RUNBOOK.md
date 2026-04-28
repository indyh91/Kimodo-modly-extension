# Real animated GLB structural validation runbook

This runbook validates the **structure** of a real Kimodo `animate-rigged-mesh` output bundle. A pass means `animated.glb` / `animated.gltf` contains parseable glTF 2.0 animation data, non-zero decoded samples, required bundle artifacts, and matching retarget metadata.

It does **not** prove Blender playback, visual quality, rest-pose correctness, axis/scale correctness, or production retargeting quality.

## 1. Find a real UniRig candidate

Prefer a real UniRig humanoid `.glb` from your Modly workspace. Search examples:

```bash
export MODLY_WORKSPACE="/path/to/Modly/workspace"
python3 - <<'PY'
import os
from pathlib import Path
root = Path(os.environ["MODLY_WORKSPACE"])
for path in sorted(root.glob('**/*_unirig.glb')):
    print(path)
PY
```

Candidate files usually live under workspace folders such as:

- `Workflows/*_unirig.glb`
- `Exports/*_unirig.glb`

The fixture must be `.glb` or `.gltf`, parseable, and skinned. If UniRig is unavailable, use a known rigged humanoid loaded through Modly Load Mesh and record the fallback limitation.

## 2. Run the Modly UI workflow

After installing or repairing the extension through Modly, run this workflow in the Modly UI:

```text
Text prompt + Load Mesh / UniRig candidate
  -> Animate Rigged Mesh
  -> output bundle containing animated.glb
```

Suggested prompt:

```text
walk forward with natural arm swing
```

Pass the selected UniRig / rigged mesh path into the Animate Rigged Mesh node. The intended runtime path is `rigged_mesh + prompt -> animate-rigged-mesh -> animated.glb`.

## 3. Expected bundle artifacts

Locate the newest Kimodo output bundle, or use the exact bundle path returned by runtime diagnostics. The bundle should contain:

- `motion.npz`
- `motion.bvh`
- `metadata.json`
- `preview.glb`
- `animated.glb`

`metadata.json` should report `retarget_status: success`, source rig path/source kind, rig classification, mapping confidence or warnings, and an artifact list matching the files above.

## 4. Invoke the validator

From this extension repository:

```bash
python3 -m validation.animated_glb_validator \
  --bundle /path/to/kimodo-output-bundle \
  --json-report /path/to/kimodo-output-bundle/animated_glb_validation_report.json
```

If the animated asset uses a non-default name:

```bash
python3 -m validation.animated_glb_validator \
  --bundle /path/to/kimodo-output-bundle \
  --animated-name animated.gltf \
  --json-report /path/to/kimodo-output-bundle/animated_glb_validation_report.json
```

The command exits `0` when structural validation passes and `1` when it fails. Without `--json-report`, the JSON report is printed to stdout.

## 5. Interpret results

- `passed: true`: required artifacts, metadata, glTF 2.0 structure, animation references, float accessors, monotonic timestamps, and non-zero sampled motion passed.
- `warnings`: review manually, but do not treat as automatic failure. Examples: fallback animation-name selection or an animated node not listed in `skins[].joints`.
- `errors`: fail the run and triage by stable code. Common codes include:
  - `missing_artifact`
  - `metadata_mismatch`
  - `invalid_glb_structure`
  - `no_animation_clips`
  - `bad_animation_reference`
  - `malformed_accessor`
  - `static_noop_animation`
  - `no_skin_skeleton`

## 6. Explicit limitation

This validator is structural only. It is a fast gate before Blender/manual review or future Modly playback work; it must not be used to claim visual correctness, animation quality, or full retargeting correctness.
