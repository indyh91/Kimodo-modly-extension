# Kimodo SOMA RP for Modly

Experimental Modly extension for NVIDIA Kimodo SOMA RP text-to-motion generation.

This repository provides the extension shell that lets Modly prepare a Kimodo runtime, download the Kimodo SOMA RP weights through Modly's normal model-download flow, and expose a text-to-motion node inside the current mesh-centric Modly UI.

## Current State

This is an initial private base, not a finished production integration.

What is wired today:

- **Install / Repair** prepares the extension runtime by running `setup.py` and creating the extension-local Python environment.
- **Download** uses Modly's existing Hugging Face download flow for `nvidia/Kimodo-SOMA-RP-v1` and stores weights under Modly's configured models directory.
- The runtime exposes a `Text to Motion Preview` node with `text -> mesh` compatibility metadata.
- Kimodo's real output remains motion-first. The `.glb` returned to Modly is a preview/compatibility artifact, while the canonical output is a motion bundle on disk.

## Requirements

- Modly with extension support.
- A Hugging Face token configured in Modly. Kimodo depends on gated Hugging Face resources.
- NVIDIA GPU runtime.
- Linux x86_64 is the intended supported baseline.
- Linux arm64 + NVIDIA Blackwell/CUDA is currently an **experimental** host lane, validated locally on NVIDIA GB10 with pinned PyTorch ARM64 CUDA wheels.

## Install and Download Flow

1. Install this extension from GitHub through Modly.
2. Let Modly run the extension setup. If needed, use the extension repair/setup action to rerun `setup.py`.
3. Click **Download** on the Kimodo SOMA RP node. This downloads `nvidia/Kimodo-SOMA-RP-v1` into Modly's models directory.
4. After weights are downloaded, use the text-to-motion node from Modly.

The important split is intentional:

- `setup.py` installs the Python runtime and Kimodo package dependencies.
- Modly's **Download** button owns model weights.

Do not run setup from the source repository when validating Modly behavior. Stage/apply belongs under Modly's extensions directory, for example:

```text
/home/drhepa/Documentos/Modly/extensions/kimodo-soma-rp
```

## Runtime Contract

- Extension id: `kimodo-soma-rp`
- Node id: `text-to-motion-preview`
- Input: `text`
- Compatibility output: `mesh`
- Weight repository: `nvidia/Kimodo-SOMA-RP-v1`
- Weight check file: `config.yaml`
- Canonical generated artifacts: Kimodo motion files such as `.npz`, metadata, and optional exports.

## Known Limitations

- The integration is experimental.
- The current Modly product surface is mesh-centric, so motion output is represented through a preview `.glb` compatibility layer.
- Full Kimodo feature parity is not implemented yet.
- Motion postprocess is disabled in this runtime-core baseline because MotionCorrection is intentionally skipped during setup.
- Linux arm64 support is best-effort and must not be treated as parity with the main Linux x86_64 baseline.

## Development Notes

- `manifest.json` declares the Modly node and model-weight metadata.
- `setup.py` prepares the extension-local runtime.
- `generator.py` runs Kimodo and resolves weights downloaded by Modly via `CHECKPOINT_DIR`.
- `host_compat.py` centralizes host compatibility decisions.
- `docs/` is intentionally gitignored for now; working notes remain local until the public documentation surface is stable.

## License

This extension shell is licensed under the MIT License. See [LICENSE](LICENSE).

Kimodo, PyTorch, Hugging Face-hosted weights, and any third-party dependencies are governed by their own upstream licenses and terms.
