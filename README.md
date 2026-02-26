# Robustness Evaluation and Fine-tuning of Medical Image Segmentation and Vision-Language Models

This repository contains the code for our paper submitted to MICCAI 2025.

## Overview

We present a comprehensive robustness benchmark for medical image AI models, covering:

- **Adversarial robustness**: FGSM and PGD attacks at 5 perturbation levels on medical segmentation models
- **Natural corruption robustness**: 15+ image corruption types (noise, blur, compression, etc.) across 5 severity levels
- **Fine-tuning strategies**: Decoder-only, LoRA, Adapter, and Encoder-partial fine-tuning to improve robustness
- **VLM evaluation**: Robustness of medical VLMs (LLaVA-Med, MedGemma) on VQA, captioning, and visual grounding tasks

**Supported segmentation models:** MedSAM, SAM-Med2D

**Supported VLMs:** LLaVA-Med, MedGemma, MedGemma 1.5, GPT-4V (API), Gemini (API)

**Supported datasets:** ISIC 2016, Brain Tumor MRI, Breast Ultrasound, Endoscopy, Retinal, Pathology, OmniMedVQA, ROCO, MeCoVQA

---

## Repository Structure

```
.
├── pipeline.py                          # Main evaluation pipeline (adversarial & perturbation)
├── model_zoo.py                         # Segmentation model wrappers (MedSAM, SAM-Med2D)
├── finetune.py                          # Fine-tuning training script
├── finetune_utils.py                    # Fine-tuning utilities (LoRA, Adapter, Optimizer, Loss)
├── segmentation_generate_perb_all_V10_adpative_efficient.py  # Perturbation dataset generator (segmentation)
├── generate_perturbation.py             # Perturbation dataset generator (VLM)
├── eval_vlm_perturbation.py             # VLM robustness evaluation script
├── merge_results_to_table.py            # Merge results into paper-ready tables
├── extract_csv_results.py               # CSV result extraction and organization tool
├── model_config.json                    # Model configuration
└── dataset_config.json                  # Dataset configuration
```

---

## Installation

```bash
# Clone the repository
git clone <repo_url>
cd <repo_name>

# Create conda environment
conda create -n med-robust python=3.10 -y
conda activate med-robust

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets peft tqdm pandas opencv-python matplotlib scikit-image scipy
pip install git+https://github.com/facebookresearch/segment-anything.git
```

---

## Quick Start

### 1. Generate Perturbation Datasets

**For segmentation datasets:**
```bash
# Generate perturbations for a single dataset
python segmentation_generate_perb_all_V10_adpative_efficient.py \
    --dataset_name isic_2016 \
    --adaptive

# Generate perturbations for all datasets
python segmentation_generate_perb_all_V10_adpative_efficient.py --all_datasets --adaptive

# Custom level intensity multipliers
python segmentation_generate_perb_all_V10_adpative_efficient.py \
    --dataset_name isic_2016 \
    --level_multipliers '{"1": 2.0, "3": 3.0, "5": 4.0}'
```

**For VLM datasets:**
```bash
python generate_perturbation.py --dataset omnimedvqa --sample_num 50
python generate_perturbation.py --all --sample_num 50
```

### 2. Evaluate Segmentation Model Robustness

**Adversarial attack evaluation:**
```bash
python pipeline.py \
    --model_name medsam \
    --dataset_name isic_2016 \
    --eval_mode adversarial \
    --attack_types fgsm pgd \
    --levels 1 2 3 4 5
```

**Corruption robustness evaluation:**
```bash
python pipeline.py \
    --model_name medsam \
    --dataset_name isic_2016_perturbed \
    --eval_mode perturbation \
    --perturbation_path /path/to/perturbed_datasets
```

**Both modes simultaneously:**
```bash
python pipeline.py \
    --model_name sammed2d \
    --dataset_name isic_2016 isic_2016_perturbed \
    --eval_mode both
```

### 3. Fine-tune for Robustness

```bash
# Decoder-only fine-tuning (fastest, recommended for small datasets)
python finetune.py \
    --model_name medsam \
    --strategy decoder_only \
    --data_path /path/to/data \
    --num_epochs 10

# LoRA fine-tuning (parameter-efficient)
python finetune.py \
    --model_name medsam \
    --strategy lora \
    --lora_r 8 \
    --lora_alpha 16 \
    --data_path /path/to/data \
    --num_epochs 20

# Adapter fine-tuning (SAM-Med2D)
python finetune.py \
    --model_name sammed2d \
    --strategy adapter_only \
    --data_path /path/to/data \
    --num_epochs 20

# Partial encoder fine-tuning
python finetune.py \
    --model_name medsam \
    --strategy encoder_partial \
    --unfreeze_encoder_layers 4 \
    --data_path /path/to/data
```

### 4. Evaluate Fine-tuned Model

```bash
python pipeline.py \
    --model_name medsam \
    --dataset_name isic_2016 \
    --eval_mode both \
    --finetune_checkpoint ./checkpoints/medsam_lora_best.pth \
    --data_split_json ./checkpoints/data_split.json
```

### 5. Evaluate VLM Robustness

```bash
# VQA task with LLaVA-Med
python eval_vlm_perturbation.py \
    --dataset omnimedvqa \
    --task vqa \
    --model_type llava-med \
    --sample_num 100

# Grounding task with MedGemma (with fine-tuning)
python eval_vlm_perturbation.py \
    --dataset mecovqa \
    --task grounding \
    --model_type medgemma \
    --finetune
```

### 6. Aggregate and Export Results

```bash
# Merge all results into paper-ready tables
python merge_results_to_table.py --results_dir ./results

# Extract and organize CSV files
python extract_csv_results.py \
    --results_dir ./results \
    --output_dir ./extracted_csv \
    --summary_only \
    --merge_summary
```

---

## Configuration

Model and dataset paths are configured via JSON files:

- `model_config.json`: Model weights paths, image sizes, and normalization parameters
- `dataset_config.json`: Dataset image/mask directories, bounding box annotations, and output settings

---

## Perturbation Types

| Category | Types |
|----------|-------|
| Noise | Gaussian noise, Shot noise, Impulse noise, Speckle noise |
| Blur | Gaussian blur, Motion blur, Defocus blur |
| Weather | Fog, Frost, Snow |
| Digital | JPEG compression, Pixelation, Contrast, Brightness, Saturation |

Each type has 5 severity levels (1=mild, 5=severe), generated with adaptive SSIM-calibrated intensity.

---

## Evaluation Metrics

- **IoU (Intersection over Union)**: Primary segmentation metric
- **Dice coefficient**: Secondary segmentation metric
- **IoU Drop / Dice Drop**: Performance degradation under perturbation
- **ΔTP**: Clean performance minus average corrupted performance

---

## Results

Results are automatically saved to `./results/` with:
- Per-image detailed CSV files
- `*_SUMMARY.csv`: Aggregated statistics per corruption type (Table 1 format)
- `*_STATS_BY_LEVEL.csv`: Breakdown by severity level

---

## License

This code is released for academic use only.
