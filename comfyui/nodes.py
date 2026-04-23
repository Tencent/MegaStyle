"""
ComfyUI custom nodes that wrap MegaStyle-FLUX. Node layout mirrors a standard
ComfyUI Flux workflow so they blend in with existing Flux graphs.

Bundled workflow (`workflow_megastyle.json`) wires them as:

    Models Loader  -> MegaStyle LoRA Loader
       Reference Style (LoadImage)
       Text Encode (CLIP + T5)   [positive prompt]
       VAE Encode                [reference style image]
       Flow Matching Scheduler   [enable_shift_rope=True]
       VAE Decode                -> Save Image

Only the LoRA loader and the Flow Matching Scheduler carry MegaStyle-specific
logic; the model loader, text encode, and VAE encode/decode nodes follow
Flux defaults but are wired through `FluxImagePipeline`.

Node class list (class name : display name shown in the Add Node menu):
    - MegaStyleModelLoader      : "Models Loader (MegaStyle / Flux)"
    - MegaStyleLoRALoader       : "MegaStyle LoRA Loader"
    - MegaStyleTextEncode       : "CLIP Text Encode (MegaStyle)"
    - MegaStyleVAEEncode        : "VAE Encode (MegaStyle)"
    - MegaStyleSampler          : "Flow Matching Scheduler (MegaStyle)"
    - MegaStyleVAEDecode        : "VAE Decode (MegaStyle)"

Legacy one-shot nodes (kept for backwards compatibility):
    - MegaStyleLoader           : Model + LoRA in one go.
    - MegaStyleAllInOneSampler  : Full pipeline in one node.
"""
import os
import sys
import numpy as np
import torch
from PIL import Image

# Make the MegaStyle repo root importable, regardless of whether this package
# is accessed directly, via a symlink to the repo, or via a symlink to the
# `comfyui/` sub-folder inside ComfyUI/custom_nodes/.
_THIS_FILE = os.path.realpath(__file__)                       # resolves symlinks
_THIS_DIR = os.path.dirname(_THIS_FILE)                       # .../MegaStyle/comfyui
_REPO_ROOT = os.path.dirname(_THIS_DIR)                       # .../MegaStyle

# Allow override via env var for edge cases.
_REPO_ROOT = os.environ.get("MEGASTYLE_REPO_ROOT", _REPO_ROOT)

if not os.path.isfile(os.path.join(_REPO_ROOT, "flux_image_mega.py")):
    raise ImportError(
        f"[MegaStyle] Could not locate flux_image_mega.py under {_REPO_ROOT}. "
        "Set the MEGASTYLE_REPO_ROOT env var to the MegaStyle repo root, "
        "or symlink the whole repo into ComfyUI/custom_nodes/."
    )

if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from flux_image_mega import (  # noqa: E402
    FluxImagePipeline,
    ModelConfig,
    model_fn_flux_image,
)
from tqdm import tqdm  # noqa: E402

try:
    import torch_npu  # noqa: F401
except Exception:
    pass


# ---------- helpers ----------

def _pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch, "npu", None) and torch.npu.is_available():
        return torch.device("npu")
    return torch.device("cpu")


def _tensor_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    """ComfyUI IMAGE: (B, H, W, C) float [0,1] -> PIL."""
    if image_tensor.ndim == 4:
        image_tensor = image_tensor[0]
    arr = image_tensor.detach().cpu().clamp(0, 1).numpy()
    arr = (arr * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr)


def _pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """PIL -> ComfyUI IMAGE (B, H, W, C) float [0,1]."""
    arr = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def _resolve_path(path: str) -> str:
    """Accept absolute paths or paths relative to the MegaStyle repo root."""
    if not path:
        raise FileNotFoundError("[MegaStyle] Empty path.")
    if os.path.isabs(path):
        if os.path.exists(path):
            return path
        raise FileNotFoundError(f"[MegaStyle] File not found: {path}")
    cand = os.path.join(_REPO_ROOT, path)
    if os.path.exists(cand):
        return cand
    raise FileNotFoundError(
        f"[MegaStyle] Could not resolve '{path}'. Tried: {cand}. "
        "Use an absolute path or place the file under the MegaStyle repo root."
    )


# ============================================================
# Fine-grained nodes
# ============================================================

class MegaStyleModelLoader:
    """Load FLUX.1-dev base weights (text encoders + DiT + VAE) into a pipe.

    Behaviour matches `inference.py`: `diffsynth` / `modelscope` resolves the
    weights and downloads them into the default cache on first use.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "flux_model_id": ("STRING", {"default": "black-forest-labs/FLUX.1-dev"}),
                "dtype": (["bfloat16", "float16"], {"default": "bfloat16"}),
            }
        }

    RETURN_TYPES = ("MEGASTYLE_PIPE",)
    RETURN_NAMES = ("pipe",)
    FUNCTION = "load"
    CATEGORY = "loaders/MegaStyle"

    def load(self, flux_model_id, dtype):
        device = _pick_device()
        torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16

        pipe = FluxImagePipeline.from_pretrained(
            torch_dtype=torch_dtype,
            device=device,
            model_configs=[
                ModelConfig(model_id=flux_model_id, origin_file_pattern="flux1-dev.safetensors"),
                ModelConfig(model_id=flux_model_id, origin_file_pattern="text_encoder/model.safetensors"),
                ModelConfig(model_id=flux_model_id, origin_file_pattern="text_encoder_2/"),
                ModelConfig(model_id=flux_model_id, origin_file_pattern="ae.safetensors"),
            ],
        )
        return (pipe,)


class MegaStyleLoRALoader:
    """Patch a (MegaStyle) LoRA onto the DiT. Run once per pipe."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("MEGASTYLE_PIPE",),
                "lora_path": ("STRING", {"default": "models/megastyle_flux.safetensors"}),
                "lora_alpha": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("MEGASTYLE_PIPE",)
    RETURN_NAMES = ("pipe",)
    FUNCTION = "load"
    CATEGORY = "loaders/MegaStyle"

    def load(self, pipe, lora_path, lora_alpha):
        lora_path = _resolve_path(lora_path)
        pipe.load_lora(pipe.dit, lora_path, alpha=float(lora_alpha))
        return (pipe,)


class MegaStyleTextEncode:
    """Encode a prompt with CLIP (text_encoder_1) + T5 (text_encoder_2)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("MEGASTYLE_PIPE",),
                "prompt": ("STRING", {"multiline": True, "default": "A bench"}),
                "t5_sequence_length": ("INT", {"default": 512, "min": 64, "max": 1024}),
                "positive": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MEGASTYLE_COND",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode"
    CATEGORY = "conditioning/MegaStyle"

    def encode(self, pipe, prompt, t5_sequence_length, positive):
        pipe.load_models_to_device(["text_encoder_1", "text_encoder_2"])
        with torch.no_grad():
            prompt_emb, pooled_prompt_emb, text_ids = pipe.prompter.encode_prompt(
                prompt,
                device=pipe.device,
                positive=bool(positive),
                t5_sequence_length=int(t5_sequence_length),
            )
        cond = {
            "prompt_emb": prompt_emb,
            "pooled_prompt_emb": pooled_prompt_emb,
            "text_ids": text_ids,
            "prompt": prompt,
        }
        return (cond,)


class MegaStyleVAEEncode:
    """Encode a reference (style) image to latents via FLUX VAE."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("MEGASTYLE_PIPE",),
                "image": ("IMAGE",),
                "height": ("INT", {"default": 512, "min": 256, "max": 2048, "step": 16}),
                "width": ("INT", {"default": 512, "min": 256, "max": 2048, "step": 16}),
            }
        }

    RETURN_TYPES = ("MEGASTYLE_LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "encode"
    CATEGORY = "latent/MegaStyle"

    def encode(self, pipe, image, height, width):
        pil = _tensor_to_pil(image).resize((int(width), int(height)))
        pipe.load_models_to_device(["vae_encoder"])
        with torch.no_grad():
            x = pipe.preprocess_image(pil).to(device=pipe.device, dtype=pipe.torch_dtype)
            latents = pipe.vae_encoder(x)
        return ({"latents": latents, "height": int(height), "width": int(width)},)


class MegaStyleSampler:
    """Denoise loop on MegaStyle-FLUX with `enable_shift_rope=True`.

    Consumes a text conditioning and a reference-image latent, runs the
    `mega_units` path of the pipeline, and returns the generated latent.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("MEGASTYLE_PIPE",),
                "conditioning": ("MEGASTYLE_COND",),
                "ref_latent": ("MEGASTYLE_LATENT",),
                "height": ("INT", {"default": 512, "min": 256, "max": 2048, "step": 16}),
                "width": ("INT", {"default": 512, "min": 256, "max": 2048, "step": 16}),
                "num_inference_steps": ("INT", {"default": 30, "min": 1, "max": 100}),
                "embedded_guidance": ("FLOAT", {"default": 3.5, "min": 1.0, "max": 10.0, "step": 0.1}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2**31 - 1}),
                "sigma_shift": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 10.0, "step": 0.01}),
            },
            "optional": {
                "negative_conditioning": ("MEGASTYLE_COND",),
                "cfg_scale": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 20.0, "step": 0.1}),
            },
        }

    RETURN_TYPES = ("MEGASTYLE_LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "sample"
    CATEGORY = "sampling/MegaStyle"

    @torch.no_grad()
    def sample(self, pipe, conditioning, ref_latent, height, width,
               num_inference_steps, embedded_guidance, seed, sigma_shift,
               negative_conditioning=None, cfg_scale=1.0):
        height, width = pipe.check_resize_height_width(int(height), int(width))

        # Use the user-provided shift when valid; otherwise fall back to the
        # FLUX.1 default (3.0). Guards against stale workflows that stored
        # sigma_shift < 1.0 (old widget default was 0.0).
        shift = float(sigma_shift) if sigma_shift and sigma_shift >= 1.0 else 3.0
        pipe.scheduler.set_timesteps(int(num_inference_steps), denoising_strength=1.0, shift=shift)

        # --- MegaInputImageEmbedder ---
        ref_latents = ref_latent["latents"]
        noise = pipe.generate_noise((1, 16, height // 8, width // 8),
                                    seed=int(seed), rand_device="cpu")
        noise = noise.to(device=pipe.device, dtype=pipe.torch_dtype)
        latents = torch.cat([ref_latents, noise], dim=0)

        # --- MegaImageIDs ---
        image_ids = pipe.dit.prepare_image_ids(latents[1:])

        # --- MegaEmbeddedGuidance ---
        guidance = torch.tensor([float(embedded_guidance)] * (latents.shape[0] - 1)).to(
            device=pipe.device, dtype=pipe.torch_dtype
        )

        inputs_posi = {
            "prompt_emb": conditioning["prompt_emb"],
            "pooled_prompt_emb": conditioning["pooled_prompt_emb"],
            "text_ids": conditioning["text_ids"],
        }
        if negative_conditioning is not None and cfg_scale != 1.0:
            inputs_nega = {
                "prompt_emb": negative_conditioning["prompt_emb"],
                "pooled_prompt_emb": negative_conditioning["pooled_prompt_emb"],
                "text_ids": negative_conditioning["text_ids"],
            }
        else:
            inputs_nega = None

        # --- Denoise ---
        pipe.load_models_to_device(list(pipe.in_iteration_models))
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}

        ref_part = latents[0:1].clone()
        for progress_id, timestep in enumerate(tqdm(pipe.scheduler.timesteps,
                                                    desc="MegaStyle Sampling")):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
            latents[0:1] = ref_part

            noise_pred_posi = model_fn_flux_image(
                **models,
                latents=latents,
                timestep=timestep,
                guidance=guidance,
                image_ids=image_ids,
                progress_id=progress_id,
                num_inference_steps=int(num_inference_steps),
                enable_shift_rope=True,
                **inputs_posi,
            )
            if inputs_nega is not None:
                noise_pred_nega = model_fn_flux_image(
                    **models,
                    latents=latents,
                    timestep=timestep,
                    guidance=guidance,
                    image_ids=image_ids,
                    progress_id=progress_id,
                    num_inference_steps=int(num_inference_steps),
                    enable_shift_rope=True,
                    **inputs_nega,
                )
                noise_pred = noise_pred_nega + float(cfg_scale) * (noise_pred_posi - noise_pred_nega)
            else:
                noise_pred = noise_pred_posi

            latents = pipe.scheduler.step(noise_pred, pipe.scheduler.timesteps[progress_id], latents)

        return ({"latents": latents[-1:], "height": int(height), "width": int(width)},)


class MegaStyleVAEDecode:
    """Decode MegaStyle latents to an IMAGE."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("MEGASTYLE_PIPE",),
                "latent": ("MEGASTYLE_LATENT",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "decode"
    CATEGORY = "latent/MegaStyle"

    @torch.no_grad()
    def decode(self, pipe, latent):
        pipe.load_models_to_device(["vae_decoder"])
        img = pipe.vae_decoder(latent["latents"], device=pipe.device)
        pil = pipe.vae_output_to_image(img[-1:])
        pipe.load_models_to_device([])
        return (_pil_to_tensor(pil),)


# ============================================================
# Legacy all-in-one nodes (kept for backwards compatibility)
# ============================================================

class MegaStyleLoader:
    """(Legacy) Load FLUX.1-dev + MegaStyle LoRA into a FluxImagePipeline."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "flux_model_id": ("STRING", {"default": "black-forest-labs/FLUX.1-dev"}),
                "lora_path": ("STRING", {"default": "models/megastyle_flux.safetensors"}),
                "lora_alpha": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05}),
                "dtype": (["bfloat16", "float16"], {"default": "bfloat16"}),
            }
        }

    RETURN_TYPES = ("MEGASTYLE_PIPE",)
    RETURN_NAMES = ("pipe",)
    FUNCTION = "load"
    CATEGORY = "MegaStyle/Legacy"

    def load(self, flux_model_id, lora_path, lora_alpha, dtype):
        device = _pick_device()
        torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16
        pipe = FluxImagePipeline.from_pretrained(
            torch_dtype=torch_dtype,
            device=device,
            model_configs=[
                ModelConfig(model_id=flux_model_id, origin_file_pattern="flux1-dev.safetensors"),
                ModelConfig(model_id=flux_model_id, origin_file_pattern="text_encoder/model.safetensors"),
                ModelConfig(model_id=flux_model_id, origin_file_pattern="text_encoder_2/"),
                ModelConfig(model_id=flux_model_id, origin_file_pattern="ae.safetensors"),
            ],
        )
        pipe.load_lora(pipe.dit, _resolve_path(lora_path), alpha=float(lora_alpha))
        return (pipe,)


class MegaStyleAllInOneSampler:
    """(Legacy) Full style transfer in one node (old MegaStyleSampler)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("MEGASTYLE_PIPE",),
                "style_image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": "A bench"}),
                "height": ("INT", {"default": 512, "min": 256, "max": 2048, "step": 16}),
                "width": ("INT", {"default": 512, "min": 256, "max": 2048, "step": 16}),
                "num_inference_steps": ("INT", {"default": 30, "min": 1, "max": 100}),
                "embedded_guidance": ("FLOAT", {"default": 3.5, "min": 1.0, "max": 10.0, "step": 0.1}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2**31 - 1}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "sample"
    CATEGORY = "MegaStyle/Legacy"

    def sample(self, pipe, style_image, prompt, height, width,
               num_inference_steps, embedded_guidance, seed, negative_prompt=""):
        style_pil = _tensor_to_pil(style_image).resize((int(width), int(height)))

        with torch.no_grad():
            out = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt or "",
                height=int(height),
                width=int(width),
                ipadapter_images=style_pil,
                seed=int(seed),
                num_inference_steps=int(num_inference_steps),
                embedded_guidance=float(embedded_guidance),
                enable_shift_rope=True,
            )
        return (_pil_to_tensor(out),)


NODE_CLASS_MAPPINGS = {
    # Fine-grained
    "MegaStyleModelLoader": MegaStyleModelLoader,
    "MegaStyleLoRALoader": MegaStyleLoRALoader,
    "MegaStyleTextEncode": MegaStyleTextEncode,
    "MegaStyleVAEEncode": MegaStyleVAEEncode,
    "MegaStyleSampler": MegaStyleSampler,
    "MegaStyleVAEDecode": MegaStyleVAEDecode,
    # Legacy
    "MegaStyleLoader": MegaStyleLoader,
    "MegaStyleAllInOneSampler": MegaStyleAllInOneSampler,
}

# Display names are aligned with the ComfyUI Flux/KSampler naming convention so
# the graph "looks" like a standard Flux workflow. Only the LoRA loader and the
# custom sampler stay MegaStyle-branded, since they embed MegaStyle specifics
# (LoRA weights + enable_shift_rope flow-matching loop).
NODE_DISPLAY_NAME_MAPPINGS = {
    "MegaStyleModelLoader": "Models Loader (MegaStyle / Flux)",
    "MegaStyleLoRALoader": "MegaStyle LoRA Loader",
    "MegaStyleTextEncode": "CLIP Text Encode (MegaStyle)",
    "MegaStyleVAEEncode": "VAE Encode (MegaStyle)",
    "MegaStyleSampler": "Flow Matching Scheduler (MegaStyle)",
    "MegaStyleVAEDecode": "VAE Decode (MegaStyle)",
    "MegaStyleLoader": "MegaStyle Loader (Legacy, All-in-one)",
    "MegaStyleAllInOneSampler": "MegaStyle Sampler (Legacy, All-in-one)",
}
