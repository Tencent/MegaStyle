"""
Gradio demo for MegaStyle-FLUX style transfer.

Usage:
    python gradio_demo.py --ckpt_path models/megastyle_flux.safetensors
    # Then open http://localhost:8080
"""
import os
import argparse
import random
import torch
import gradio as gr
from PIL import Image

from flux_image_mega import FluxImagePipeline, ModelConfig

try:
    import torch_npu  # noqa: F401
except Exception:
    pass


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="models/megastyle_flux.safetensors",
        help="Path to MegaStyle-FLUX LoRA checkpoint.",
    )
    parser.add_argument(
        "--ref_path",
        type=str,
        default="ref_styles",
        help="Directory with example reference style images.",
    )
    parser.add_argument("--server_name", type=str, default="0.0.0.0")
    parser.add_argument("--server_port", type=int, default=8080)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch, "npu", None) and torch.npu.is_available():
        return torch.device("npu")
    return torch.device("cpu")


def load_pipeline(ckpt_path: str, device: torch.device) -> FluxImagePipeline:
    pipe = FluxImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="flux1-dev.safetensors"),
            ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder/model.safetensors"),
            ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder_2/"),
            ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="ae.safetensors"),
        ],
    )
    pipe.load_lora(pipe.dit, ckpt_path, alpha=1)
    return pipe


def build_examples(ref_path: str):
    if not os.path.isdir(ref_path):
        return []
    files = sorted(
        os.path.join(ref_path, f) for f in os.listdir(ref_path) if f.lower().endswith(".jpg")
    )
    default_prompts = ["A bench", "A car", "A house with a tree beside"]
    examples = []
    for i, f in enumerate(files[:12]):
        examples.append([f, default_prompts[i % len(default_prompts)]])
    return examples


def main():
    args = parse_args()
    device = pick_device()
    print(f"[MegaStyle] Loading pipeline on {device} ...")
    pipe = load_pipeline(args.ckpt_path, device)
    print("[MegaStyle] Pipeline ready.")

    @torch.no_grad()
    def generate(style_image, prompt, height, width, num_inference_steps,
                 embedded_guidance, seed, randomize_seed):
        if style_image is None:
            raise gr.Error("Please provide a reference style image.")
        if not prompt or not prompt.strip():
            raise gr.Error("Please provide a content prompt.")

        if randomize_seed or seed is None or int(seed) < 0:
            seed = random.randint(0, 2**31 - 1)
        seed = int(seed)

        style_image = style_image.convert("RGB").resize((int(width), int(height)))
        image = pipe(
            prompt=prompt,
            height=int(height),
            width=int(width),
            ipadapter_images=style_image,
            seed=seed,
            num_inference_steps=int(num_inference_steps),
            embedded_guidance=float(embedded_guidance),
            enable_shift_rope=True,
        )
        return image, seed

    header_md = """
    # MegaStyle-FLUX: Style Transfer Demo

    **MegaStyle** is a scalable data curation pipeline that explores the
    consistent text-to-image style mapping ability of modern T2I models to
    build an intra-style consistent, inter-style diverse and high-quality
    style dataset — **MegaStyle-1.4M** (1.4M images across 170K curated style
    prompts and 400K content prompts).

    Trained on MegaStyle-1.4M, **MegaStyle-FLUX** performs generalizable
    reference-based style transfer: given any *reference style image* and a
    *content prompt*, it synthesizes the described content while faithfully
    preserving the reference's style.

    > Upload a reference style image on the left and enter a content prompt,
    > then click **Generate**.

    **References**
    - Paper: <a href="https://arxiv.org/abs/2604.08364" target="_blank">arXiv:2604.08364</a>
    - Project page: <a href="https://jeoyal.github.io/MegaStyle/" target="_blank">jeoyal.github.io/MegaStyle</a>
    - Code: <a href="https://github.com/Tencent/MegaStyle" target="_blank">github.com/Tencent/MegaStyle</a>
    - Model weights (MegaStyle-FLUX / Encoder):
      <a href="https://huggingface.co/Gaojunyao/MegaStyle" target="_blank">HuggingFace</a> ·
      <a href="https://modelscope.cn/models/junyaogao/MegaStyle" target="_blank">ModelScope</a>
    - Dataset (MegaStyle-1.4M):
      <a href="https://huggingface.co/datasets/tencent/MegaStyle-1.4M" target="_blank">HuggingFace</a> ·
      <a href="https://modelscope.cn/datasets/Tencent-Hunyuan/MegaStyle-1.4M" target="_blank">ModelScope</a>
    - Base model: <a href="https://huggingface.co/black-forest-labs/FLUX.1-dev" target="_blank">FLUX.1-dev</a> ·
      Style encoder vision backbone: <a href="https://huggingface.co/google/siglip-so400m-patch14-384" target="_blank">SigLIP-so400m</a>
    - Built on top of <a href="https://github.com/modelscope/DiffSynth-Studio" target="_blank">DiffSynth-Studio</a>
    """

    footer_md = r"""
    ---
    ### Citation
    If this work is helpful for your research, please consider citing:
    ```bibtex
    @article{gao2026megastyle,
      title   = {MegaStyle: Constructing Diverse and Scalable Style Dataset via
                 Consistent Text-to-Image Style Mapping},
      author  = {Gao, Junyao and Liu, Sibo and Li, Jiaxing and Sun, Yanan and
                 Tu, Yuanpeng and Shen, Fei and Zhang, Weidong and Zhao, Cairong and Zhang, Jun},
      journal = {arXiv preprint arXiv:2604.08364},
      year    = {2026}
    }
    ```

    ### Acknowledgements
    Built on top of [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio).
    All assets and code are released under the repository [LICENSE](https://github.com/Tencent/MegaStyle/blob/main/LICENSE.txt).
    """

    with gr.Blocks(title="MegaStyle-FLUX Demo") as demo:
        gr.Markdown(header_md)
        with gr.Row():
            with gr.Column():
                style_image = gr.Image(label="Reference Style Image", type="pil")
                prompt = gr.Textbox(label="Content Prompt", value="A bench",
                                    placeholder="e.g. A house with a tree beside")
                with gr.Row():
                    height = gr.Slider(256, 1536, value=512, step=16, label="Height")
                    width = gr.Slider(256, 1536, value=512, step=16, label="Width")
                with gr.Row():
                    num_inference_steps = gr.Slider(10, 50, value=30, step=1, label="Steps")
                    embedded_guidance = gr.Slider(1.0, 10.0, value=3.5, step=0.1,
                                                  label="Embedded Guidance")
                with gr.Row():
                    seed = gr.Number(value=42, label="Seed", precision=0)
                    randomize_seed = gr.Checkbox(value=False, label="Random seed")
                run_btn = gr.Button("Generate", variant="primary")
            with gr.Column():
                out_image = gr.Image(label="Generated Image", type="pil")
                used_seed = gr.Number(label="Used Seed", precision=0, interactive=False)

        examples = build_examples(args.ref_path)
        if examples:
            gr.Examples(
                examples=examples,
                inputs=[style_image, prompt],
                label="Reference style examples",
                examples_per_page=12,
            )

        run_btn.click(
            fn=generate,
            inputs=[style_image, prompt, height, width, num_inference_steps,
                    embedded_guidance, seed, randomize_seed],
            outputs=[out_image, used_seed],
        )

        gr.Markdown(footer_md)

    demo.queue().launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
