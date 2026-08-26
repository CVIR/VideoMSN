
# Image Classifiers are Efficient Self-Supervised Video Representation Learners 

Fine-tuning **DINOv3** (small and base) pretrained checkpoints on video classification datasets (Kinetics-400, SSv2, UCF101, HMDB51, etc.) using SIFAR-style inputs or MSN-style pretraining.

## A. Dataset Preparation

1. Create annotation files: `train.txt` and `val.txt`
2. Format of each line:
   ```
   /Kinetics400/train_256/playing_drums/GJJUUAxgIYo_000007_000017.mp4 1 300 230
   ```
   → `video_path` `label` `start_frame` `end_frame`

3. Pass the **folder containing** these txt files (folder path) to `--data_dir`

4. Update `video_dataset_config.py` with your dataset name and paths


## B. Training Details

### General Arguments

| Argument          | Kinetics-400       | SSv2              | Notes                              |
|-------------------|--------------------|-------------------|------------------------------------|
| `--class_numbers` | 400                | 174               | Number of classes                  |
| `--model`         | `dino_small`       | `dino_small`      | or `dino_base`                     |
| `--duration`      | 16                 | 16                | Number of frames sampled           |

### 1. Standard SIFAR-style DinoV3 Finetuning

Use `--dino_model_path` + **do not** use `--msn_pretraining`

## C. VideoMSN Fine-tuning of DINOv3

Use `--msn_model_path` + `--msn_pretraining True`

**1. Example – VideoMSN Fine-tuning of DINOv3 Small**

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=28529 main.py \
  --data_dir <dataset path> \
  --use_pyav --dataset kinetics400 \
  --opt adamw --lr 5e-4 --epochs 30 --sched cosine \
  --duration 16 --batch-size 4 --super_img_rows 4 \
  --num_workers 16 --disable_scaleup \
  --mixup 0.8 --cutmix 1.0 --drop-path 0.05 \
  --pretrained --warmup-epochs 5 --no-amp \
  --model dino_small \
  --output_dir <output dir path>\
  --weight-decay 0.01 --clip-grad 1.0 \
  --class_numbers 400 \
  --msn_pretraining True \
  --msn_model_path <MSN pretrained checkpoint path>
```

**2. Example – VideoMSN Fine-tuning of DINOv3 base**

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=28529 main.py \
  --data_dir <dataset path> \
  --use_pyav --dataset kinetics400 \
  --opt adamw --lr 5e-4 --epochs 30 --sched cosine \
  --duration 16 --batch-size 4 --super_img_rows 4 \
  --num_workers 16 --disable_scaleup \
  --mixup 0.8 --cutmix 1.0 --drop-path 0.05 \
  --pretrained --warmup-epochs 5 --no-amp \
  --model dino_base \
  --output_dir <output dir path>\
  --weight-decay 0.01 --clip-grad 1.0 \
  --class_numbers 400 \
  --msn_pretraining True \
  --msn_model_path <MSN pretrained checkpoint path>
```
<!--
## C. Pretrained Checkpoints

- **DinoV3 Base** (LVD pretraining): [[link](https://drive.google.com/drive/folders/1e1PLmTIKrMzgnhplKvWIleIFo5iknFI6)]
- **MSN-pretrained DinoV3 checkpoints**: [[link](https://drive.google.com/drive/folders/14y_jOugOtlDFJOaRJaehSfcXvxRG6lvb)] *(continuously updated)*

## Contact

**Author:** Sudipta Sarkar  
**Date:** 16 Feb 2026  
**Email:** sudiptasarkar3600@gmail.com
-->


