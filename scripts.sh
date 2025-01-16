#!/bin/bash

#SBATCH --job-name=ori_model
#SBATCH --output=ori_model.out.%j
#SBATCH --error=ori_model.out.%j
#SBATCH --cpus-per-task=16
#SBATCH --mem=80GB
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1

source /scratch/yl13585/miniconda3/bin/activate llava_ov
# source /fs/nexus-scratch/yliang17/miniconda3/bin/activate llava_ov

# wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
# wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
# wget https://huggingface.co/spaces/xinyu1205/Tag2Text/resolve/main/ram_swin_large_14m.pth

export CUDA_VISIBLE_DEVICES=0
python automatic_label_ram_demo.py \
  --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --ram_checkpoint ram_swin_large_14m.pth \
  --grounded_checkpoint groundingdino_swint_ogc.pth \
  --sam_checkpoint sam_vit_h_4b8939.pth \
  --input_folder "/scratch/yl13585/Diffusion/ColorBench/data/img" \
  --output_dir "/scratch/yl13585/Diffusion/ColorBench/segments" \
  --box_threshold 0.25 \
  --text_threshold 0.2 \
  --iou_threshold 0.5 \
  --device "cuda"
