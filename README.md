# Kimodo SOMA RP for Modly

Modly extension for NVIDIA Kimodo SOMA RP human motion generation.

This repository provides the extension shell that lets Modly prepare a Kimodo runtime, download the Kimodo SOMA RP weights through Modly's normal model-download flow, and expose a text-to-motion node inside the current mesh-centric Modly UI.

## Current State

This is an initial private base, not a finished production integration.

What is wired today:

- **Install / Repair** prepares the extension runtime by running `setup.py` and creating the extension-local Python environment.
- **Download** uses Modly's existing Hugging Face download flow for `nvidia/Kimodo-SOMA-RP-v1` and stores weights under Modly's configured models directory.
- The runtime exposes a `Text to Motion Preview` node with `text -> mesh` compatibility metadata.
- The runtime exposes an `Animate Rigged Mesh` model node with `text + rigged mesh -> mesh` compatibility metadata. It shares the same Kimodo weight ownership as the preview node.
- Kimodo's real output remains motion-first. For `Text to Motion Preview`, the `.glb` returned to Modly is still a preview/compatibility artifact and the canonical output is a motion bundle on disk.
- For `Animate Rigged Mesh`, the supported MVP path now attempts a real BVH-first retarget/export and returns `animated.glb` only after animation data is structurally authored into the GLB.

## Requirements

- Modly with extension support.
- A Hugging Face token configured in Modly. Kimodo depends on gated Hugging Face resources.
- Kimodo's text encoder uses LLM2Vec on top of Meta Llama 3: `McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp`, `McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised`, and `meta-llama/Meta-Llama-3-8B-Instruct`. Before first use, sign in to Hugging Face with the token used by Modly and accept or request the required Meta Llama 3 access conditions before download; otherwise encoder/model download can fail even when Kimodo itself is installed.
- NVIDIA GPU runtime.
- Linux x86_64 is the intended supported baseline.
- Linux arm64 + NVIDIA Blackwell/CUDA is currently an **experimental** host lane, validated locally on NVIDIA GB10 with pinned PyTorch ARM64 CUDA wheels.
- Windows x86_64 is available as an experimental compatibility lane. The setup path is unit-tested for Windows-style virtualenv paths, but native Windows CUDA/PyTorch/DLL execution has not yet been validated by this repository.

## Install and Download Flow

1. Install this extension from GitHub through Modly.
2. Let Modly run the extension setup. If needed, use the extension repair/setup action to rerun `setup.py`.
3. Click **Download** on the Kimodo SOMA RP node. This downloads `nvidia/Kimodo-SOMA-RP-v1` into Modly's models directory.
4. After weights are downloaded, use the text-to-motion node from Modly.

The important split is intentional:

- `setup.py` installs the Python runtime and Kimodo package dependencies.
- Modly's **Download** button owns model weights.

Gated encoder access is separate from the Kimodo SOMA RP weight download: Hugging Face/Meta access conditions for the LLM2Vec and Meta Llama 3 repositories must be accepted or requested before the first encoder/model download.

Do not run setup from the source repository when validating Modly behavior. Install, repair, staging, and extension directory placement are owned by Modly's extension-management flow, not by this repository's source tree.

## Runtime Contract

- Extension id: `kimodo-soma-rp`
- Node id: `text-to-motion-preview`
- Node id: `animate-rigged-mesh`
- Input: `text`
- Multi-input animation path: required `prompt:text` plus `rigged_mesh:mesh`
- Compatibility output: `mesh`
- Valid rig input source: a rigged `.glb`/`.gltf` from UniRig or an already-rigged mesh supplied through Modly Load Mesh. `mesh_path` remains a compatibility alias for `rigged_mesh_path`.
- Weight repository: `nvidia/Kimodo-SOMA-RP-v1`
- Weight check file: `config.yaml`
- Canonical generated artifacts: Kimodo motion files such as `.npz`, metadata, and optional exports.
- Runtime discriminator: Modly passes the model node id in generation params. `animate-rigged-mesh` requires `rigged_mesh_path`; `mesh_path` is accepted as a compatibility alias.

## Known Limitations

- The integration is experimental.
- The current Modly product surface is mesh-centric, so motion output is represented through a preview `.glb` compatibility layer.
- Full Kimodo feature parity is not implemented yet.
- Motion postprocess is disabled in this runtime-core baseline because MotionCorrection is intentionally skipped during setup.
- Linux arm64 support is best-effort and must not be treated as parity with the main Linux x86_64 baseline.
- Windows setup can still fail because of CUDA, PyTorch, DLL loading, or gated model access; those are known experimental-lane risk categories, not evidence that native Windows runtime execution has been validated.
- The current `Animate Rigged Mesh` MVP only authors a root-translation animation channel. Full joint rotations, rest-pose correction, axis conversion beyond recorded assumptions, scale handling, and visual-quality validation are deferred.
- Structurally real animation data does not mean production-quality retargeting. The GLB can contain a valid `Kimodo Motion` clip while still looking wrong for rigs outside the supported humanoid path.
- Future research: evaluate alternative text encoders or external encoder service paths to reduce gated-model friction, VRAM pressure, or Windows-specific runtime risk. No alternative encoder implementation is part of this repository state.

## Rigging and Retargeting Direction

Kimodo SOMA RP is a human-motion model. The recommended downstream path is to combine it with a rigged humanoid GLB, especially one prepared by UniRig, and produce an animated GLB / Motion Mesh-compatible artifact from the `Animate Rigged Mesh` node. An already-rigged GLB/GLTF loaded through Modly Load Mesh is also a valid input source when it contains skin and skeleton data.

Non-humanoid or unusual rigs should be treated as experimental best-effort inputs. They are not forbidden, but users should expect possible deformation, incorrect retargeting, `MAPPING_INCOMPATIBLE` failure, or unusable animation when the rig does not match a humanoid motion profile.

The current animation node validates and records the supplied rigged mesh path, requests BVH alongside the canonical NPZ motion artifact, inspects the rig, builds a conservative humanoid mapping, and writes metadata with retarget/export status, source rig, source kind, classification, mapping confidence, warnings, and retained diagnostic artifact paths. It returns `animated.glb` only on real export success; failure metadata uses primary machine-readable codes such as `MISSING_BVH`, `UNPARSEABLE_BVH`, `UNPARSEABLE_RIG`, `RIG_HAS_NO_SKIN_OR_SKELETON`, `MAPPING_INCOMPATIBLE`, and `EXPORT_FAILED`. For compatibility with earlier SDD/spec wording, metadata also exposes `retarget_error_aliases` for legacy names: `BVH_UNAVAILABLE` for `MISSING_BVH`, `BVH_UNPARSEABLE` for `UNPARSEABLE_BVH`, and `RIG_INVALID` for unparseable or structurally unskinned rigs.

The MVP animation channel is intentionally narrow: root translation is preserved, while full per-joint rotations, arbitrary rest-pose alignment, production axis/scale conversion, binary-buffer optimization, and visual validation are deferred follow-up work.

## Development Notes

- `manifest.json` declares the Modly node and model-weight metadata.
- `setup.py` prepares the extension-local runtime.
- `generator.py` runs Kimodo and resolves weights downloaded by Modly via `CHECKPOINT_DIR`.
- `host_compat.py` centralizes host compatibility decisions.
- `docs/` is intentionally gitignored for now; working notes remain local until the public documentation surface is stable.

## License

This extension shell is licensed under the MIT License. See [LICENSE](LICENSE).

Kimodo, PyTorch, Hugging Face-hosted weights, and any third-party dependencies are governed by their own upstream licenses and terms.
