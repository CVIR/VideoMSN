
# Image Classifiers are Efficient Self-Supervised Video Representation Learners

Self-supervised **pretraining** of **DINOv3** (small and base) using **MSN-style** (Masked Siamese Networks) objective on video datasets (primarily Kinetics-400, SSv2, UCF101 and HMDB51 with super-image inputs).

This codebase brings MSN principles to video-domain DINOv3 models with temporal augmentations.

## A. Setup & Environment

1. **Create Conda Environment**
   ```bash
   conda env create --name sifar_msn --file env.yaml
   ```
   Activate:
   ```bash
   conda activate sifar_msn
   ```


## B. Key Arguments Overview

| Argument              | Default / Typical       | Importance / Notes                                                                 |
|-----------------------|-------------------------|------------------------------------------------------------------------------------|
| `--model`             | `dino_small`            | Choose model architecture: `dino_small` or `dino_base`                            |
| `--class_numbers`     | 400                     | Number of classes (Kinetics-400 is 400 and SSv2 is 174). **Not very important here** — only used for logging / dummy classifier head initialization. Pretraining is self-supervised (no real classification loss). |
| `--dino_model_path`   | (empty str)              | Path to LVD-pretrained DinoV3 checkpoint (required when starting from pretrained weights). Leave empty only if you have another loading logic. |
| `--msn_pretraining`   | (flag — must set)       | Enables MSN-style masked siamese pretraining                                       |
| `--lr`                | `1e-6`                  | Very low learning rate — typical for stable MSN-style pretraining                 |
| `--patch_drop`        | `0.70`                  | Fraction of patches to mask (high masking ratio is key for MSN)                   |
| `--me_max` + `--memax_weight` | flag + `5.0`     | Enables ME-MAX loss (prototype-based entropy maximization)                        |
| `--fviews`            | `0.675 0.65 0.625 0.6`  | Focal/multi-crop view scales — controls local view sizes                          |
| `--temporal_aug`      | True                  | Enables video-specific temporal augmentations                                      |

## C. Training Examples

### 1. VideoMSN Pretraining – DINOv3 Small
```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=28427 main.py \
  --data_dir <dataset path> \
  --use_pyav --dataset kinetics400 \
  --opt adamw --lr 1e-6 --epochs 10 --sched cosine \
  --duration 16 --batch-size 4 --super_img_rows 4 \
  --disable_scaleup  --drop-path 0.1 \
  --warmup-epochs 2 --no-amp \
  --model dino_small \
  --output_dir <output dir path> \
  --msn_pretraining --patch_drop 0.70 --me_max --memax_weight 5.0 \
  --pretrained --fviews 0.675 0.65 0.625 0.6 \
  --clip-grad 1.0 --temporal_aug --focal_size 128 \
  --weight-decay 0.01 --blur --min-lr 1e-7 \
  --class_numbers 400 --num_workers 16 \
  --dino_model_path <LVD pretained chekpoint path>
```

### 2. VideoMSN Pretraining – DINOv3 Base
```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=28427 main.py \
  --data_dir <dataset path> \
  --use_pyav --dataset kinetics400 \
  --opt adamw --lr 1e-6 --epochs 50 --sched cosine \
  --duration 16 --batch-size 20 --super_img_rows 4 \
  --disable_scaleup --drop-path 0.1 \
  --warmup-epochs 5 --no-amp \
  --model dino_base \
  --output_dir <output dir path> \
  --msn_pretraining --patch_drop 0.70 --me_max --memax_weight 5.0 \
  --pretrained --fviews 0.675 0.65 0.625 0.6 \
  --clip-grad 1.0 --temporal_aug --focal_size 128 \
  --weight-decay 0.01 --blur --min-lr 1e-7 \
  --class_numbers 400 --num_workers 16 \
  --dino_model_path <LVD pretained chekpoint path>
```
<!--
## D. Pretrained Checkpoints & Resources
- **Initial DinoV3 checkpoints** (LVD pretraining): [[Google Drive link](https://drive.google.com/drive/folders/1e1PLmTIKrMzgnhplKvWIleIFo5iknFI6)]

## Contact
**Author:** Sudipta Sarkar  
**Date:** 16 February 2026  
**Email:** sudiptasarkar3600@gmail.com
-->
