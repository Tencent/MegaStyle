# MegaStyle ComfyUI Nodes

Fine-grained nodes that wrap `FluxImagePipeline` for style transfer inside
ComfyUI, mirroring the `mega_units` path (`enable_shift_rope=True`).

## Install

From the MegaStyle repo root (so the `ln -s` target is absolute):

```bash
ln -s "$(pwd)/comfyui" /path/to/ComfyUI/custom_nodes/MegaStyle
```

Overrides (all optional):

```bash
export MEGASTYLE_REPO_ROOT=/abs/path/to/MegaStyle           # override repo auto-detect
export MEGASTYLE_COMFY_ROOT=/abs/path/to/ComfyUI            # only if auto-detect fails
export MEGASTYLE_AUTO_INSTALL_WORKFLOW=0                    # skip auto-copy of the workflow
export MEGASTYLE_AUTO_INSTALL_REFS=0                        # skip symlinking ref_styles into ComfyUI/input/
```

Run ComfyUI inside the `megastyle` env (requires `diffsynth==1.1.8`).

## Nodes

The default workflow uses the display names on the left; the underlying
class names (shown on the right) are what gets serialised into workflow
JSON.

| Display name (canvas title) | Class name (JSON `type`) | In | Out |
|---|---|---|---|
| **Models Loader** | `MegaStyleModelLoader` | `flux_model_id`, `dtype` | `MEGASTYLE_PIPE` |
| **MegaStyle LoRA Loader** | `MegaStyleLoRALoader` | `pipe`, `lora_path`, `lora_alpha` | `MEGASTYLE_PIPE` |
| **Reference Style** | `LoadImage` *(ComfyUI built-in)* | filename from `ComfyUI/input/` | `IMAGE` |
| **Text Encode** | `MegaStyleTextEncode` | `pipe`, `prompt`, `t5_sequence_length`, `positive` | `MEGASTYLE_COND` |
| **VAE Encode** | `MegaStyleVAEEncode` | `pipe`, `image`, `height`, `width` | `MEGASTYLE_LATENT` |
| **Flow Matching Scheduler** | `MegaStyleSampler` | `pipe`, `conditioning`, `ref_latent`, `height`, `width`, `steps`, `embedded_guidance`, `seed`, `sigma_shift`, *(negative_conditioning, cfg_scale)* | `MEGASTYLE_LATENT` |
| **VAE Decode** | `MegaStyleVAEDecode` | `pipe`, `latent` | `IMAGE` |
| **Save Image** | `SaveImage` *(ComfyUI built-in)* | `images` | — |

Legacy one-shot nodes (`MegaStyleLoader`, `MegaStyleAllInOneSampler`) are still
shipped for backward compatibility.

## Pre-built workflow

The shipped graph is `./workflow_megastyle.json`. On first launch, this
package automatically copies it into
`ComfyUI/user/default/workflows/MegaStyle.json` and symlinks every image
from `../ref_styles/` into `ComfyUI/input/`, so the default `Reference
Style` (`LoadImage`) node resolves `00.jpg` out of the box.

From the ComfyUI UI, open the **Workflows** side panel and pick
**MegaStyle**.

## Wiring diagram

```
┌──────────────────────┐    ┌──────────────────────┐
│ Models Loader        │──► │ MegaStyle LoRA       │──┬───────────────┐
│ (FLUX.1-dev)         │    │ Loader               │  │               │
└──────────────────────┘    └──────────────────────┘  │               │
                                                      ▼               ▼
                              ┌──────────────────────┐   ┌──────────────────────┐
                              │ Text Encode          │   │ VAE Encode           │
                              │ (CLIP + T5)          │   │  image ◄── Reference │
                              │  prompt = "A bench"  │   │           Style      │
                              └──────────────────────┘   └──────────────────────┘
                                           │                       │
                                           ▼                       ▼
                                        ┌─────────────────────────────────┐
                                        │ Flow Matching Scheduler         │
                                        │ (enable_shift_rope=True)        │
                                        └─────────────────────────────────┘
                                                       │
                                                       ▼
                                        ┌─────────────────────────────────┐
                                        │ VAE Decode ──► Save Image       │
                                        └─────────────────────────────────┘
```

## Notes

- `MEGASTYLE_PIPE` carries the full `FluxImagePipeline`; passing it through
  every node keeps all sub-models on one device and avoids reloading.
- `enable_shift_rope=True` is hard-wired in the sampler to match
  `inference.py`; that branch expects a concatenated `[ref_latents, latents]`
  batch, constructed inside the node.
- CFG is optional. Leave `negative_conditioning` unconnected and
  `cfg_scale=1.0` for the fastest, `inference.py`-equivalent behaviour.
- `Save Image` writes to `ComfyUI/output/MegaStyle/megastyle_*.png`. Change
  the prefix on the node (default `MegaStyle/megastyle`) to redirect.
