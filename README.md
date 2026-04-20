# MegaStyle: Constructing Diverse and Scalable Style Dataset via Consistent Text-to-Image Style Mapping

<a href='https://arxiv.org/abs/2604.08364'><img src='https://img.shields.io/badge/arXiv-2604.08364-b31b1b.svg'></a> 
<a href='https://jeoyal.github.io/MegaStyle/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
<a href='https://huggingface.co/Gaojunyao/MegaStylet'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue'></a>
<a href='https://huggingface.co/datasets/tencent/MegaStyle-1.4M'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue'></a>

**MegaStyle** is a novel and scalable data curation pipeline that first explores consistent T2I style mapping ability from current large generative models to construct intra-style consistent, inter-style diverse and high-quality style dataset.

**Your star is our fuel!  We're revving up the engines with it!**

<img src="assets/teaser.png">

## News
- [2026/4/21] 🔥 We release the training/inference codes, [models](https://huggingface.co/Gaojunyao/MegaStyle) and [dataset](https://huggingface.co/datasets/tencent/MegaStyle-1.4M) of MegaStyle!!!

## TODO List
- [ ] A more diverse and larger-scale style dataset.

## MegaStyle1.4M
[MegaStyle-1.4M](https://huggingface.co/datasets/tencent/MegaStyle-1.4M) is a large-scale style dataset built through a scalable pipeline that leverages consistent text-to-image style mapping of Qwen-Image. It combines 170K curated style prompts with 400K content prompts to generate 1.4M high-quality images that share strong intra-style consistency while covering diverse fine-grained styles.
<img src="assets/megastyle1.4M.jpeg">


## Get Started
Trained on MegaStyle1.4M, we introduce MegaStyle-FLUX and MegaStyle-Encoder for generalizable style transfer and reliable style similarity measurement.
### Clone the Repository

```
git clone git@github.com:Tencent/MegaStyle.git
cd ./MegaStyle
```

### Environment Setup
```
conda create -n megastyle python==3.10
conda activate megastyle
pip install diffsynth==1.1.8
```

### Downloading Checkpoints

1. Download the pretrained models of [SigLIP](https://huggingface.co/google/siglip-so400m-patch14-384) and [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev).

2. Download the checkpoints of [MegaStyle-FLUX](https://huggingface.co/Gaojunyao/MegaStyle/blob/main/megastyle_flux.safetensors) and [MegaStyle-Encoder](https://huggingface.co/Gaojunyao/MegaStyle/blob/main/megastyle_encoder.pth) into `./models/`. 

### Running Inference
For image style transfer, we provide 50 reference style images from <a href='https://drive.google.com/file/d/1Q_jbI25NfqZvuwWv53slmovqyW_L4k2r/view?usp=drive_link'>StyleBench</a> in `./ref_styles`:
```
python inference.py --ckpt_path models/megastyle_flux.safetensors --ref_path ./ref_styles
```
For computing style score:
```
python style_score.py --ckpt_path models/megastyle_encoder.pth --real_image_path <path/to/image.png> --fake_image_path <path/to/image.png>
```

### Training
To train a style transfer model with paired supervision, please download our style dataset, [MegaStyle1.4M](https://huggingface.co/datasets/tencent/MegaStyle-1.4M), and start training with:
```
bash FLUX.1-dev.sh # FLUX.1-dev-npu.sh for npu
```

## License and Citation
All assets and code are under the [license](./LICENSE) unless specified otherwise.

If this work is helpful for your research, please consider citing the following BibTeX entry.
```
@article{gao2026megastyle,
  title={MegaStyle: Constructing Diverse and Scalable Style Dataset via Consistent Text-to-Image Style Mapping},
  author={Gao, Junyao and Liu, Sibo and Li, Jiaxing and Sun, Yanan and Tu, Yuanpeng and Shen, Fei and Zhang, Weidong and Zhao, Cairong and Zhang, Jun},
  journal={arXiv preprint arXiv:2604.08364},
  year={2026}
}
```

## Acknowledgements
The code is built upon [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio).