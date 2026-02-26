#!/usr/bin/env python3
"""
V3
扰动图像生成脚本 - 独立于VLM环境
为VQA、Caption、Grounding任务的数据集生成扰动版本

生成的目录结构与原始images同级:
  原始: .../images/
  扰动: .../perturbed_images/{perturbation_type}/{severity}/

使用方法:
    python generate_perturbation.py --dataset omnimedvqa --sample_num 50
    python generate_perturbation.py --dataset roco --sample_num 50
    python generate_perturbation.py --dataset mecovqa --sample_num 50
    python generate_perturbation.py --all --sample_num 50
"""

import os
import io
import json
import random
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import argparse
import warnings
from typing import Dict, List, Tuple, Optional
from scipy.ndimage import gaussian_filter
from skimage.metrics import structural_similarity as ssim


# ============================================
# 配置
# ============================================

CONFIG = {
    "seed": 42,
    "default_severities": [1, 3, 5],
    
    "datasets": {
        "omnimedvqa": {
            "json_folder": "/mnt/fast/nobackup/scratch4weeks/ly0008/cxx/MedSAM/work_dir/medsam-vit-base/scripts/pipeline/data/OmniMedVQA/OmniMedVQA/QA_information/Open-access",
            "image_base_dir": "/mnt/fast/nobackup/scratch4weeks/ly0008/cxx/MedSAM/work_dir/medsam-vit-base/scripts/pipeline/data/OmniMedVQA/OmniMedVQA",
            "output_base": "/mnt/fast/nobackup/scratch4weeks/ly0008/cxx/MedSAM/work_dir/medsam-vit-base/scripts/pipeline/data/OmniMedVQA/OmniMedVQA",
        },
        "roco": {
            "parquet_folder": "/mnt/fast/nobackup/scratch4weeks/ly0008/cxx/MedSAM/work_dir/medsam-vit-base/scripts/pipeline/data/ROCOv2-radiology/data",
            "extracted_dir": "/mnt/fast/nobackup/scratch4weeks/ly0008/cxx/MedSAM/work_dir/medsam-vit-base/scripts/pipeline/data/ROCOv2-radiology/extracted",
            "output_base": "/mnt/fast/nobackup/scratch4weeks/ly0008/cxx/MedSAM/work_dir/medsam-vit-base/scripts/pipeline/data/ROCOv2-radiology",
            "split": "test",
        },
        "mecovqa": {
            "json_file": "/mnt/fast/nobackup/scratch4weeks/ly0008/cxx/MedSAM/work_dir/medsam-vit-base/scripts/pipeline/data/Medical_Visual_Grounding/MeCoVQA-G-Plus/mecovqa_bbox_format_test.json",
            "base_dir": "/mnt/fast/nobackup/scratch4weeks/ly0008/cxx/MedSAM/work_dir/medsam-vit-base/scripts/pipeline/data/Medical_Visual_Grounding/MeCoVQA-G-Plus",
            "output_base": "/mnt/fast/nobackup/scratch4weeks/ly0008/cxx/MedSAM/work_dir/medsam-vit-base/scripts/pipeline/data/Medical_Visual_Grounding/MeCoVQA-G-Plus",
        },
    },
}

MODALITY_MAPPING = {
    'CT': 'CT', 'ct': 'CT', 'MRI': 'MRI', 'mri': 'MRI',
    'Ultrasound': 'US', 'ultrasound': 'US', 'US': 'US',
    'Dermoscopy': 'DERM', 'dermoscopy': 'DERM',
    'Pathology': 'PATHOLOGY', 'pathology': 'PATHOLOGY',
    'Endoscopy': 'ENDOSCOPY', 'endoscopy': 'ENDOSCOPY',
    'Fundus': 'OCT', 'fundus': 'OCT', 'OCT': 'OCT',
    'X-Ray': 'XRAY', 'X-ray': 'XRAY', 'xray': 'XRAY',
}


# ============================================
# 扰动生成器
# ============================================

class MedicalPerturbationGenerator:
    def __init__(self, seed=42):
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
        self._init_params()
    
    def _init_params(self):
        self.severity_params = {
            'gaussian_noise': {1: 0.01, 2: 0.02, 3: 0.03, 4: 0.05, 5: 0.08},
            'salt_pepper_noise': {1: 0.001, 2: 0.003, 3: 0.005, 4: 0.01, 5: 0.02},
            'speckle_noise': {1: 0.05, 2: 0.1, 3: 0.15, 4: 0.2, 5: 0.25},
            'gaussian_blur': {1: 0.5, 2: 1.0, 3: 1.5, 4: 2.0, 5: 3.0},
            'motion_blur': {1: 3, 2: 5, 3: 7, 4: 9, 5: 11},
            'brightness_contrast': {1: (0.05, 0.05), 2: (0.1, 0.1), 3: (0.15, 0.15), 4: (0.2, 0.2), 5: (0.3, 0.3)},
            'compression_artifacts': {1: 90, 2: 70, 3: 50, 4: 30, 5: 10},
            'xray_scatter': {1: 0.05, 2: 0.10, 3: 0.15, 4: 0.22, 5: 0.30},
            'xray_underexposure': {1: 0.85, 2: 0.70, 3: 0.55, 4: 0.40, 5: 0.25},
            'xray_overexposure': {1: 1.15, 2: 1.30, 3: 1.50, 4: 1.75, 5: 2.0},
            'xray_grid_artifact': {1: 0.03, 2: 0.06, 3: 0.10, 4: 0.15, 5: 0.22},
            'window_level': {1: 0.10, 2: 0.20, 3: 0.30, 4: 0.40, 5: 0.50},
            'mri_bias': {1: 0.30, 2: 0.50, 3: 0.70, 4: 0.85, 5: 0.95},
            'mri_ghost': {1: 0.10, 2: 0.20, 3: 0.35, 4: 0.50, 5: 0.70},
            'ct_metal': {1: 0.10, 2: 0.18, 3: 0.28, 4: 0.40, 5: 0.55},
            'ct_beam': {1: 0.08, 2: 0.15, 3: 0.22, 4: 0.32, 5: 0.45},
            'path_stain': {1: 0.08, 2: 0.12, 3: 0.18, 4: 0.25, 5: 0.35},
        }
        
        self.base_perturbations = [
            'gaussian_noise', 'salt_pepper_noise', 'speckle_noise',
            'gaussian_blur', 'motion_blur', 'brightness_contrast', 'compression_artifacts'
        ]
        
        self.modality_perturbations = {
            'XRAY': ['xray_scatter', 'xray_underexposure', 'xray_overexposure', 'xray_grid_artifact', 'window_level'],
            'CT': ['ct_metal', 'ct_beam', 'window_level'],
            'MRI': ['mri_bias', 'mri_ghost'],
            'PATHOLOGY': ['path_stain'],
        }
    
    def _is_color(self, img):
        return img.ndim == 3 and img.shape[2] >= 3
    
    def load_image(self, path):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Cannot load: {path}")
        if img.ndim == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img.astype(np.float32)
    
    def load_image_from_bytes(self, img_bytes):
        img_pil = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        return np.array(img_pil).astype(np.float32)
    
    def save_image(self, img, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        if img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        if self._is_color(img):
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, img)
    
    def compute_ssim(self, orig, pert):
        o = (orig - orig.min()) / (orig.max() - orig.min() + 1e-8)
        p = (pert - pert.min()) / (pert.max() - pert.min() + 1e-8)
        if self._is_color(orig):
            return np.mean([ssim(o[:,:,c], p[:,:,c], data_range=1.0) for c in range(3)])
        return ssim(o, p, data_range=1.0)
    
    def get_perturbations_for_modality(self, modality):
        specific = self.modality_perturbations.get(modality, [])
        return self.base_perturbations + specific
    
    # 基础扰动
    def gaussian_noise(self, img, severity):
        sigma = self.severity_params['gaussian_noise'][severity]
        noise = np.random.normal(0, sigma * (img.max() - img.min()), img.shape)
        return np.clip(img + noise, img.min(), img.max())
    
    def salt_pepper_noise(self, img, severity):
        prob = self.severity_params['salt_pepper_noise'][severity]
        out = img.copy()
        salt = np.random.random(img.shape[:2]) < prob / 2
        pepper = np.random.random(img.shape[:2]) < prob / 2
        if self._is_color(img):
            for c in range(img.shape[2]):
                out[:,:,c][salt] = img[:,:,c].max()
                out[:,:,c][pepper] = img[:,:,c].min()
        else:
            out[salt], out[pepper] = img.max(), img.min()
        return out
    
    def speckle_noise(self, img, severity):
        sigma = self.severity_params['speckle_noise'][severity]
        return np.clip(img * np.random.normal(1.0, sigma, img.shape), img.min(), img.max())
    
    def gaussian_blur(self, img, severity):
        sigma = self.severity_params['gaussian_blur'][severity]
        if self._is_color(img):
            return np.stack([gaussian_filter(img[:,:,c], sigma) for c in range(img.shape[2])], axis=2)
        return gaussian_filter(img, sigma)
    
    def motion_blur(self, img, severity):
        ksize = self.severity_params['motion_blur'][severity]
        kernel = np.zeros((ksize, ksize))
        kernel[ksize//2, :] = 1.0 / ksize
        M = cv2.getRotationMatrix2D((ksize//2, ksize//2), np.random.uniform(-180, 180), 1.0)
        kernel = cv2.warpAffine(kernel, M, (ksize, ksize))
        if self._is_color(img):
            return np.stack([cv2.filter2D(img[:,:,c].astype(np.float32), -1, kernel) for c in range(img.shape[2])], axis=2)
        return cv2.filter2D(img.astype(np.float32), -1, kernel)
    
    def brightness_contrast(self, img, severity):
        bright_var, contrast_var = self.severity_params['brightness_contrast'][severity]
        alpha = 1 + np.random.uniform(-contrast_var, contrast_var)
        beta = np.random.uniform(-bright_var, bright_var) * (img.max() - img.min())
        return np.clip(alpha * img + beta, img.min(), img.max())
    
    def compression_artifacts(self, img, severity):
        quality = self.severity_params['compression_artifacts'][severity]
        img_uint8 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        if self._is_color(img):
            img_uint8 = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
        _, buf = cv2.imencode('.jpg', img_uint8, [cv2.IMWRITE_JPEG_QUALITY, quality])
        dec = cv2.imdecode(buf, cv2.IMREAD_COLOR if self._is_color(img) else cv2.IMREAD_GRAYSCALE)
        if self._is_color(img):
            dec = cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)
        return dec.astype(np.float32)
    
    # X-Ray专用扰动
    def xray_scatter(self, img, severity):
        intensity = self.severity_params['xray_scatter'][severity]
        scatter = gaussian_filter(img, sigma=30)
        return np.clip(img + intensity * scatter, img.min(), img.max())
    
    def xray_underexposure(self, img, severity):
        factor = self.severity_params['xray_underexposure'][severity]
        return img * factor
    
    def xray_overexposure(self, img, severity):
        factor = self.severity_params['xray_overexposure'][severity]
        return np.clip(img * factor, img.min(), img.max())
    
    def xray_grid_artifact(self, img, severity):
        intensity = self.severity_params['xray_grid_artifact'][severity]
        h, w = img.shape[:2]
        pattern = np.ones_like(img)
        for i in range(0, h, 4):
            if self._is_color(img):
                pattern[i, :, :] = 1 - intensity
            else:
                pattern[i, :] = 1 - intensity
        return img * pattern
    
    def window_level(self, img, severity):
        shift = self.severity_params['window_level'][severity]
        img_range = img.max() - img.min()
        center = (img.max() + img.min()) / 2
        new_center = center + np.random.uniform(-shift, shift) * img_range
        width = img_range * (1 - shift * 0.5)
        low = new_center - width / 2
        high = new_center + width / 2
        return np.clip((img - low) / (high - low) * img_range + img.min(), img.min(), img.max())
    
    # MRI专用扰动
    def mri_bias(self, img, severity):
        intensity = self.severity_params['mri_bias'][severity]
        h, w = img.shape[:2]
        y, x = np.ogrid[:h, :w]
        cx, cy = w * np.random.uniform(0.3, 0.7), h * np.random.uniform(0.3, 0.7)
        bias = 1 + intensity * np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * (min(h, w) * 0.3)**2))
        if self._is_color(img):
            bias = bias[:, :, np.newaxis]
        return np.clip(img * bias, img.min(), img.max())
    
    def mri_ghost(self, img, severity):
        intensity = self.severity_params['mri_ghost'][severity]
        shift = int(img.shape[0] * 0.1)
        ghost = np.roll(img, shift, axis=0)
        return np.clip(img + intensity * ghost, img.min(), img.max())
    
    # CT专用扰动
    def ct_metal(self, img, severity):
        intensity = self.severity_params['ct_metal'][severity]
        h, w = img.shape[:2]
        num_streaks = np.random.randint(3, 8)
        streak_img = np.zeros((h, w), dtype=np.float32)
        for _ in range(num_streaks):
            angle = np.random.uniform(0, np.pi)
            center = (np.random.randint(w // 4, 3 * w // 4), np.random.randint(h // 4, 3 * h // 4))
            length = np.random.randint(w // 2, w)
            cv2.line(streak_img, 
                    (int(center[0] - length * np.cos(angle) / 2), int(center[1] - length * np.sin(angle) / 2)),
                    (int(center[0] + length * np.cos(angle) / 2), int(center[1] + length * np.sin(angle) / 2)),
                    1.0, 2)
        streak_img = gaussian_filter(streak_img, sigma=3)
        if self._is_color(img):
            streak_img = streak_img[:, :, np.newaxis]
        return np.clip(img + intensity * streak_img * (img.max() - img.min()), img.min(), img.max())
    
    def ct_beam(self, img, severity):
        intensity = self.severity_params['ct_beam'][severity]
        h, w = img.shape[:2]
        gradient = np.linspace(0, 1, h).reshape(-1, 1)
        if self._is_color(img):
            gradient = gradient[:, :, np.newaxis]
        return np.clip(img * (1 - intensity * gradient), img.min(), img.max())
    
    # 病理专用扰动
    def path_stain(self, img, severity):
        if not self._is_color(img):
            return img
        intensity = self.severity_params['path_stain'][severity]
        variation = np.random.uniform(1 - intensity, 1 + intensity, 3)
        result = img.copy()
        for c in range(3):
            result[:, :, c] = np.clip(img[:, :, c] * variation[c], 0, 255)
        return result
    
    def apply(self, img, pert_type, severity):
        method = getattr(self, pert_type, None)
        if method is None:
            raise ValueError(f"Unknown perturbation: {pert_type}")
        return method(img, severity)


# ============================================
# 数据加载函数 (仅加载，不采样)
# ============================================

def load_omnimedvqa_all(config):
    """加载OmniMedVQA全部数据（不采样）"""
    json_folder = Path(config['json_folder'])
    image_base = Path(config['image_base_dir'])
    
    all_samples = []
    for jf in sorted(json_folder.glob("*.json")):
        with open(jf) as f:
            data = json.load(f)
        for item in (data if isinstance(data, list) else [data]):
            img_path = item.get('image_path', '')
            if '${dataset_root_path}' in img_path:
                img_path = img_path.replace('${dataset_root_path}/', '')
            full_path = image_base / img_path
            if full_path.exists():
                item['_image_path'] = str(full_path)
                item['_relative_path'] = img_path
                all_samples.append(item)
    
    return all_samples


def load_roco_all(config):
    """加载ROCO全部数据（不采样）"""
    extracted_dir = Path(config['extracted_dir'])
    split = config.get('split', 'test')
    
    # 方式1: 使用预提取的数据
    meta_file = extracted_dir / split / "metadata.json"
    if meta_file.exists():
        print(f"  [ROCO] 使用提取后的格式: {meta_file}")
        with open(meta_file) as f:
            metadata = json.load(f)
        
        samples = []
        # 支持两种metadata格式
        sample_list = metadata.get('samples', metadata) if isinstance(metadata, dict) else metadata
        for item in sample_list:
            img_id = item.get('image_id', item.get('id', ''))
            img_path = extracted_dir / split / "images" / f"{img_id}.png"
            if img_path.exists():
                samples.append({
                    'image_id': img_id,
                    'caption': item.get('caption', ''),
                    '_image_path': str(img_path),
                    '_relative_path': f"images/{img_id}.png",
                })
        return samples
    
    # 方式2: 使用原始parquet
    parquet_folder = Path(config['parquet_folder'])
    print(f"  [ROCO] 使用parquet格式: {parquet_folder}")
    
    all_samples = []
    try:
        for pf in sorted(parquet_folder.glob(f"{split}-*.parquet")):
            try:
                df = pd.read_parquet(pf)
                for idx, row in df.iterrows():
                    if 'image' in row and isinstance(row['image'], dict) and 'bytes' in row['image']:
                        all_samples.append({
                            'image_id': row.get('image_id', f"{pf.stem}_{idx}"),
                            'caption': row.get('caption', ''),
                            '_image_bytes': row['image']['bytes'],
                        })
            except Exception as e:
                print(f"  Error: {pf}: {e}")
    except Exception as e:
        print(f"  Parquet读取失败: {e}")
    
    return all_samples


def load_mecovqa_all(config):
    """加载MeCoVQA全部数据（不采样）"""
    json_file = Path(config['json_file'])
    base_dir = Path(config['base_dir'])
    
    with open(json_file) as f:
        data = json.load(f)
    
    valid = []
    for s in (data if isinstance(data, list) else [data]):
        img_path = base_dir / s.get('image', '')
        if img_path.exists():
            s['_image_path'] = str(img_path)
            s['_relative_path'] = s.get('image', '')
            valid.append(s)
    
    return valid


def sample_data(samples, sample_num, seed, dataset_name):
    """统一的采样函数"""
    original_count = len(samples)
    
    if sample_num > 0 and len(samples) > sample_num:
        random.seed(seed)
        samples = random.sample(samples, sample_num)
        print(f"  采样: {original_count} -> {len(samples)}")
    elif sample_num > 0 and len(samples) <= sample_num:
        print(f"  注意: 请求 {sample_num} 个样本，实际只有 {original_count} 个，将使用全部数据")
    else:
        print(f"  使用全部数据: {original_count}")
    
    return samples


def get_modality(sample, dataset_name):
    if dataset_name == 'omnimedvqa':
        return MODALITY_MAPPING.get(sample.get('modality_type', ''), 'XRAY')
    return 'XRAY'


# ============================================
# 数据统计显示
# ============================================

def print_dataset_statistics(datasets_info):
    """打印所有数据集的统计信息"""
    print("\n" + "=" * 70)
    print("📊 数据集统计")
    print("=" * 70)
    
    total = 0
    for name, count in datasets_info.items():
        print(f"  {name:15s}: {count:>8,} 个样本")
        total += count
    
    print("-" * 40)
    print(f"  {'总计':15s}: {total:>8,} 个样本")
    print("=" * 70 + "\n")


# ============================================
# 主生成流程
# ============================================

def generate_perturbations(dataset_name, sample_num, severities, perturbations=None, seed=42):
    print("\n" + "=" * 70)
    print(f"🔧 扰动图像生成 - {dataset_name}")
    print("=" * 70)
    
    config = CONFIG['datasets'][dataset_name]
    generator = MedicalPerturbationGenerator(seed)
    
    # ========== 第1步: 加载全部数据并统计 ==========
    print(f"\n[1/4] 📂 加载数据并统计原始数量...")
    if dataset_name == 'omnimedvqa':
        all_samples = load_omnimedvqa_all(config)
        output_base = Path(config['output_base'])
    elif dataset_name == 'roco':
        all_samples = load_roco_all(config)
        output_base = Path(config['output_base'])
    elif dataset_name == 'mecovqa':
        all_samples = load_mecovqa_all(config)
        output_base = Path(config['output_base'])
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    print(f"  ✓ 原始数据量: {len(all_samples):,} 个样本")
    
    # ========== 第2步: 采样 ==========
    print(f"\n[2/4] 🎯 采样设置...")
    print(f"  请求样本数: {sample_num if sample_num > 0 else '全部'}")
    samples = sample_data(all_samples, sample_num, seed, dataset_name)
    print(f"  实际使用量: {len(samples):,} 个样本")
    
    # ========== 第3步: 配置扰动 ==========
    if perturbations is None:
        if dataset_name == 'roco':
            perturbations = generator.base_perturbations + generator.modality_perturbations['XRAY']
        else:
            perturbations = generator.base_perturbations
    
    print(f"\n[3/4] ⚙️  扰动配置...")
    print(f"  扰动类型: {len(perturbations)} 种")
    print(f"  严重程度: {severities}")
    total_images = len(samples) * len(perturbations) * len(severities)
    print(f"  预计生成: {total_images:,} 张图像")
    
    perturbed_base = output_base / "perturbed_images"
    perturbed_base.mkdir(parents=True, exist_ok=True)
    
    generation_info = {
        'dataset': dataset_name,
        'timestamp': datetime.now().isoformat(),
        'original_count': len(all_samples),
        'sample_num': len(samples),
        'perturbations': perturbations,
        'severities': severities,
        'samples': [],
    }
    
    # ========== 第4步: 生成扰动图像 ==========
    print(f"\n[4/4] 🖼️  生成扰动图像...")
    
    for severity in severities:
        for pert_type in perturbations:
            pert_dir = perturbed_base / pert_type / str(severity)
            pert_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"  {pert_type} severity={severity}")
            
            for i, sample in enumerate(tqdm(samples, desc=f"{pert_type}_s{severity}", leave=False)):
                try:
                    modality = get_modality(sample, dataset_name)
                    applicable = generator.get_perturbations_for_modality(modality)
                    if pert_type not in applicable:
                        continue
                    
                    if dataset_name == 'roco':
                        if '_image_path' in sample:
                            img = generator.load_image(sample['_image_path'])
                            filename = f"{sample['image_id']}.png"
                            relative_path = sample.get('_relative_path', f"images/{filename}")
                        else:
                            img = generator.load_image_from_bytes(sample['_image_bytes'])
                            filename = f"{sample['image_id']}.png"
                            relative_path = f"images/{filename}"
                    else:
                        img = generator.load_image(sample['_image_path'])
                        filename = Path(sample['_image_path']).name
                        if not filename.lower().endswith('.png'):
                            filename = Path(filename).stem + '.png'
                        relative_path = sample.get('_relative_path', filename)
                    
                    perturbed = generator.apply(img, pert_type, severity)
                    ssim_val = generator.compute_ssim(img, perturbed)
                    
                    save_path = pert_dir / filename
                    generator.save_image(perturbed, str(save_path))
                    
                    if severity == severities[0] and pert_type == perturbations[0]:
                        generation_info['samples'].append({
                            'id': sample.get('image_id', sample.get('question_id', i)),
                            'original': relative_path,
                            'modality': modality,
                        })
                except Exception as e:
                    if i < 3:
                        print(f"    Error: {e}")
    
    # 保存元数据
    meta_file = perturbed_base / f"generation_info_{dataset_name}.json"
    with open(meta_file, 'w') as f:
        json.dump(generation_info, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*70}")
    print(f"✅ 完成! 输出: {perturbed_base}")
    print(f"{'='*70}")
    return str(perturbed_base)


def main():
    parser = argparse.ArgumentParser(description="生成扰动图像")
    parser.add_argument("--dataset", type=str, choices=['omnimedvqa', 'roco', 'mecovqa'])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--sample_num", type=int, default=50)
    parser.add_argument("--severities", type=str, default="1,3,5")
    parser.add_argument("--perturbations", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    severities = [int(s) for s in args.severities.split(',')]
    perturbations = args.perturbations.split(',') if args.perturbations else None
    datasets = ['omnimedvqa', 'roco', 'mecovqa'] if args.all else ([args.dataset] if args.dataset else [])
    
    if not datasets:
        print("请指定 --dataset 或 --all")
        return
    
    # ========== 首先统计所有数据集的原始数据量 ==========
    print("\n" + "=" * 70)
    print("🔍 正在统计各数据集原始数据量...")
    print("=" * 70)
    
    datasets_info = {}
    for ds in datasets:
        config = CONFIG['datasets'][ds]
        print(f"\n  检查 {ds}...")
        
        if ds == 'omnimedvqa':
            samples = load_omnimedvqa_all(config)
        elif ds == 'roco':
            samples = load_roco_all(config)
        elif ds == 'mecovqa':
            samples = load_mecovqa_all(config)
        
        datasets_info[ds] = len(samples)
    
    # 打印统计汇总
    print_dataset_statistics(datasets_info)
    
    print(f"📋 采样配置: 每个数据集使用 {args.sample_num if args.sample_num > 0 else '全部'} 个样本")
    print(f"📋 随机种子: {args.seed}")
    print()
    
    # ========== 然后依次处理各数据集 ==========
    for ds in datasets:
        generate_perturbations(ds, args.sample_num, severities, perturbations, args.seed)


if __name__ == "__main__":
    main()
