import torch
from flux_image_mega import FluxImagePipeline, ModelConfig
from diffsynth import load_state_dict
try:
    import torch_npu
except:
    pass
import os
import argparse
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument(
    "--ckpt_path",
    type=str,
    required=True,
)
parser.add_argument(
    "--ref_path",
    type=str,
    default="ref_styles",
)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else ("npu" if getattr(torch, "npu", None) and torch.npu.is_available() else "cpu"))

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
pipe.load_lora(pipe.dit, args.ckpt_path, alpha=1)

root = args.ref_path
style_files = sorted(
    os.path.join(root, file) for file in os.listdir(root)
    if file.endswith("jpg")
)

save_dir = os.path.join("results", os.path.basename(args.ckpt_path)[:-12])
os.makedirs(save_dir, exist_ok=True)

contents = [
    "A bench",
    "A car",
    "A house with a tree beside",
]


for s_i, style_file in enumerate(style_files):
    style_image = Image.open(style_file).resize((512, 512))
    style_image.save(os.path.join(save_dir, f"s{s_i+1}_ref.jpg"))
    for c_i, content in enumerate(contents):
        image = pipe(prompt=content, height=512, width=512,
        ipadapter_images=style_image, seed=42, enable_shift_rope=True)
        image.save(os.path.join(save_dir, f"s{s_i+1}_c{c_i+1}.jpg"))
