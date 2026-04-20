import os
from transformers import AutoProcessor, SiglipVisionModel
import torch
from PIL import Image
try:
    import torch_npu
except:
    pass
import argparse


parser = argparse.ArgumentParser()
parser.add_argument(
    "--ckpt_path",
    type=str,
    required=True,
)
parser.add_argument(
    "--real_image_path",
    type=str,
    required=True,
)
parser.add_argument(
    "--fake_image_path",
    type=str,
    required=True,
)
args = parser.parse_args()


@torch.no_grad()
def extract_features(
    image_path,
    model,
    processor,
    device="cuda",
):
    feats = []
    imgs = [Image.open(image_path).convert("RGB")]
    inputs = processor(images=imgs, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    outputs = model(pixel_values=pixel_values)
    emb = outputs.pooler_output
    emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
    feats.append(emb.cpu())

    feats = torch.cat(feats, dim=0)  # [N, D] on CPU
    return feats


fake_image_path = args.fake_image_path
real_image_path = args.real_image_path

device = torch.device("cuda" if torch.cuda.is_available() else ("npu" if getattr(torch, "npu", None) and torch.npu.is_available() else "cpu"))
model = SiglipVisionModel.from_pretrained("google/siglip-so400m-patch14-384").to(device).eval()
processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")

checkpoint = torch.load(args.ckpt_path, map_location="cpu")
state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
msg = model.load_state_dict(state, strict=False)
print(f"=> load_state_dict msg: {msg}")

fake_img = extract_features(fake_image_path, model, processor, device)
real_img = extract_features(real_image_path, model, processor, device)

sim = (fake_img * real_img).sum() / real_img.shape[0]

print(f"=> style score: {sim.item():.2f}")