#!/usr/bin/env bash
set -euo pipefail

LOG_DIR="checkpoints/megastyle_flux_1.4M/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/train_$(date +%Y%m%d_%H%M%S).log"

export PYTHONUNBUFFERED=1
export HCCL_WHITELIST_DISABLE=1
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export HCCL_CONNECT_TIMEOUT=1000
{
   python -m accelerate.commands.launch --use_deepspeed --config_file configs/accelerate_config_npu.yaml --num_processes 16 \
      train.py \
   --dataset_base_path "MegaStyle1.4M/" \
   --max_pixels 512512 \
   --dataset_repeat 1 \
   --model_id_with_origin_paths "black-forest-labs/FLUX.1-dev:flux1-dev.safetensors,black-forest-labs/FLUX.1-dev:text_encoder/model.safetensors,black-forest-labs/FLUX.1-dev:text_encoder_2/,black-forest-labs/FLUX.1-dev:ae.safetensors" \
   --learning_rate 1e-4 \
   --num_epochs 20 \
   --remove_prefix_in_ckpt "pipe.dit." \
   --output_path "$LOG_DIR" \
   --lora_base_model "dit" \
   --extra_inputs "ipadapter_images" \
   --lora_target_modules "a_to_qkv,b_to_qkv,ff_a.0,ff_a.2,ff_b.0,ff_b.2,a_to_out,b_to_out,proj_out,norm.linear,norm1_a.linear,norm1_b.linear,to_qkv_mlp" \
   --lora_rank 128 \
   --align_to_opensource_format \
   --use_gradient_checkpointing \
   --height 512 \
   --width 512 \
   --save_steps 2000
} 2>&1 | tee "$LOG_FILE"
