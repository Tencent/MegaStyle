import random, os, pickle, datasets, math
from collections import defaultdict
from torch.utils.data import Dataset
from PIL import Image


class MegaDataset(Dataset):
    def __init__(self, base_path, height=512, width=512):
        parquet_glob = os.path.join(base_path, "*.parquet")
        self.ds = datasets.load_dataset(
            "parquet",
            data_files=parquet_glob,
            split="train",
        ).cast_column("image", datasets.Image())

        self.load_from_cache = False
        self.height, self.width = height, width

        style_indices_path = os.path.join(base_path, "style_indices.pkl")
        if os.path.exists(style_indices_path):
            with open(style_indices_path, "rb") as f:
                self.style_to_indices = pickle.load(f)
        else:
            self.style_to_indices = defaultdict(list)
            for idx, style in enumerate(self.ds["style"]):
                self.style_to_indices[style].append(idx)
            with open(style_indices_path, "wb") as f:
                pickle.dump(dict(self.style_to_indices), f)

        self.valid_styles = [
            s for s, idxs in self.style_to_indices.items()
            if len(idxs) >= 2
        ]
    
    def augment_style_pil(
        self,
        img: Image.Image,
        height: int = 512,
        width: int = 512,
        p_flip: float = 0.5,
        p_crop: float = 0.5,
        scale=(0.8, 1.0),  
    ) -> Image.Image:
        if img.mode != "RGB":
            img = img.convert("RGB")

        # 1) RandomHorizontalFlip
        if random.random() < p_flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        # 2) RandomResizedCrop
        if random.random() < p_crop:
            w, h = img.size
            area = w * h

            target_area = random.uniform(scale[0], scale[1]) * area
            side = int(round(math.sqrt(target_area)))
            side = max(1, min(side, w, h))

            left = random.randint(0, w - side) if w > side else 0
            top  = random.randint(0, h - side) if h > side else 0

            img = img.crop((left, top, left + side, top + side))
            img = img.resize((width, height), resample=Image.BICUBIC)

        return img
    
    def __len__(self):
        return len(self.valid_styles)

    def __getitem__(self, idx):
        style = self.valid_styles[idx]
        i, j = random.sample(self.style_to_indices[style], 2)

        ex1 = self.ds[i]
        ex2 = self.ds[j]

        img1 = ex1["image"].resize((self.width, self.height), resample=Image.BICUBIC)
        img2 = ex2["image"].resize((self.width, self.height), resample=Image.BICUBIC)

        return {
            "image": img1,
            "ipadapter_images": self.augment_style_pil(img2, self.height, self.width),
            "prompt": ex1["content"],
            }
