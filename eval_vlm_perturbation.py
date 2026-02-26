#!/usr/bin/env python3
"""
V6 - 整合版本 (支持所有开源模型微调)
VLM扰动评测脚本 - 支持多种模型
- 开源模型: LLaVA-Med, MedGemma, MedGemma 1.5 (均支持Grounding微调)
- 闭源API: Gemini, GPT-4V

所有三个开源模型都支持对visual grounding任务进行LoRA微调
QA和Caption任务保持不变（无需微调）

使用方法:
    # 使用LLaVA-Med评测原始图像 (默认)
    python eval_vlm_perturbation.py --dataset omnimedvqa --task vqa --sample_num 50
    
    # 使用MedGemma评测
    python eval_vlm_perturbation.py --dataset omnimedvqa --task vqa --sample_num 50 \
        --model_type medgemma
    
    # Grounding任务微调 (支持所有开源模型)
    python eval_vlm_perturbation.py --dataset mecovqa --task grounding --finetune \
        --model_type llava-med
    python eval_vlm_perturbation.py --dataset mecovqa --task grounding --finetune \
        --model_type medgemma
    python eval_vlm_perturbation.py --dataset mecovqa --task grounding --finetune \
        --model_type medgemma15
    
    # 使用微调后的模型评估
    python eval_vlm_perturbation.py --dataset mecovqa --task grounding \
        --model_type llava-med --model_path ./outputs/finetune/llava-med/mecovqa_grounding/final
    python eval_vlm_perturbation.py --dataset mecovqa --task grounding \
        --model_type medgemma --medgemma_path ./outputs/finetune/medgemma/mecovqa_grounding/final
    python eval_vlm_perturbation.py --dataset mecovqa --task grounding \
        --model_type medgemma15 --medgemma15_path ./outputs/finetune/medgemma15/mecovqa_grounding/final
"""

import os
os.environ["DISABLE_FLASH_ATTN"] = "1"

import io
import re
import json
import random
import torch
import pandas as pd
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from abc import ABC, abstractmethod
import argparse
import numpy as np

# ============================================
# 新增：Caption标准指标导入
# ============================================
try:
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.cider.cider import Cider
    CAPTION_METRICS_AVAILABLE = True
except ImportError:
    CAPTION_METRICS_AVAILABLE = False
# ============================================
# 新增结束
# ============================================

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return super().default(obj)


# ============================================
# 评估工具函数
# ============================================

def parse_bbox_from_text(text):
    """从文本中提取bbox坐标 [x1, y1, x2, y2] 或 [x, y, w, h]"""
    if text is None:
        return None
    
    patterns = [
        r'\[?\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]?',
        r'\(\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\)',
        r'(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            coords = [float(x) for x in match.groups()]
            return coords
    
    return None


def extract_bbox_from_mask(mask_path):
    """从mask图像中提取bbox（归一化坐标）"""
    try:
        mask = np.array(Image.open(mask_path).convert('L'))
        h, w = mask.shape
        
        rows = np.any(mask > 0, axis=1)
        cols = np.any(mask > 0, axis=0)
        
        if not rows.any() or not cols.any():
            return None
        
        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]
        
        return [x1/w, y1/h, x2/w, y2/h]
    except Exception as e:
        print(f"  Warning: Failed to extract bbox from mask {mask_path}: {e}")
        return None


def parse_mask_path(text):
    """从ground truth文本中提取mask路径"""
    if text is None:
        return None
    
    match = re.search(r'<mask>(.*?)</mask>', text)
    if match:
        return match.group(1)
    
    return None


def calculate_iou(box1, box2):
    """计算两个bbox的IoU"""
    if box1 is None or box2 is None:
        return 0.0
    
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    if union <= 0:
        return 0.0
    
    return intersection / union


class DeepSeekEvaluator:
    """使用DeepSeek API进行语义评估"""
    
    def __init__(self, config):
        self.api_key = config.get('api_key', '')
        self.base_url = config.get('base_url', 'https://api.deepseek.com')
        self.model = config.get('model', 'deepseek-chat')
        self.temperature = config.get('temperature', 0.1)
        self.max_tokens = config.get('max_tokens', 100)
        
        if not self.api_key or self.api_key == 'YOUR_DEEPSEEK_API_KEY':
            print("⚠️ DeepSeek API Key未设置，语义评估将跳过")
            self.enabled = False
        else:
            self.enabled = True
    
    def evaluate_vqa(self, question, prediction, ground_truth):
        """评估VQA答案正确性"""
        if not self.enabled:
            return None
        
        prompt = f"""判断以下医学VQA问答的预测答案是否正确。

问题: {question}
标准答案: {ground_truth}
预测答案: {prediction}

请只回答一个数字:
- 1 表示完全正确（语义一致）
- 0.5 表示部分正确
- 0 表示错误

回答:"""
        
        return self._call_api(prompt)
    
    def evaluate_caption(self, prediction, ground_truth):
        """评估Caption质量"""
        if not self.enabled:
            return None
        
        prompt = f"""评估以下医学图像描述的质量。

标准描述: {ground_truth}
预测描述: {prediction}

请从0到1打分，考虑:
1. 关键医学信息是否覆盖
2. 描述是否准确
3. 是否有错误信息

请只回答一个0到1之间的数字（如0.8）:"""
        
        return self._call_api(prompt)
    
    def _call_api(self, prompt):
        """调用DeepSeek API"""
        import requests
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
            
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content'].strip()
                match = re.search(r'(\d+\.?\d*)', content)
                if match:
                    return float(match.group(1))
            else:
                print(f"DeepSeek API error: {response.status_code}")
                
        except Exception as e:
            print(f"DeepSeek API error: {e}")
        
        return None


# ============================================
# 新增：Caption标准指标计算函数
# ============================================

def compute_caption_metrics(predictions, references):
    """
    计算Caption标准指标: BLEU-1/4, ROUGE-L, CIDEr
    predictions: 预测文本列表
    references: 参考文本列表 (每个元素可以是字符串或字符串列表)
    """
    if not CAPTION_METRICS_AVAILABLE or len(predictions) == 0:
        return {}
    
    # 转换为pycocoevalcap格式
    # gts: {id: [ref1, ref2, ...]}
    # res: {id: [pred]}
    gts = {}
    res = {}
    
    for i, (pred, ref) in enumerate(zip(predictions, references)):
        # 确保prediction是字符串
        pred_str = str(pred) if pred else ""
        # reference可以是单个字符串或列表
        if isinstance(ref, list):
            ref_list = [str(r) for r in ref]
        else:
            ref_list = [str(ref)]
        
        gts[i] = ref_list
        res[i] = [pred_str]
    
    metrics = {}
    
    try:
        # BLEU (1-4 grams)
        bleu_scorer = Bleu(4)
        bleu_scores, _ = bleu_scorer.compute_score(gts, res)
        metrics['BLEU-1'] = round(bleu_scores[0], 4)
        metrics['BLEU-4'] = round(bleu_scores[3], 4)
        
        # ROUGE-L
        rouge_scorer = Rouge()
        rouge_score, _ = rouge_scorer.compute_score(gts, res)
        metrics['ROUGE-L'] = round(rouge_score, 4)
        
        # CIDEr (医学图像描述常用)
        cider_scorer = Cider()
        cider_score, _ = cider_scorer.compute_score(gts, res)
        metrics['CIDEr'] = round(cider_score, 4)
        
    except Exception as e:
        print(f"计算Caption指标出错: {e}")
        return {}
    
    return metrics

# ============================================
# 新增结束
# ============================================


# ============================================
# 配置
# ============================================

CONFIG = {
    "output_dir": "./outputs/vlm_perturbation_results",
    "sample_num": 100,
    "seed": 42,
    
    # LLaVA-Med模型配置 (来自V3，保持不变)
    "model": {
        "type": "llava-med",
        "path": "../models/llava-med-v1.5-mistral-7b",
        "conv_template": "mistral_instruct",
        "temperature": 0.2,
        "max_new_tokens": 512,
    },
    
    # MedGemma专用配置 (来自V4)
    "medgemma": {
        "path": "/mnt/fast/nobackup/scratch4weeks/ly0008/cxx/MedSAM/work_dir/medsam-vit-base/scripts/pipeline/Multi-scenario/Pipeline_for_All/models/medgemma-4b-it",
        "torch_dtype": "bfloat16",
        "device_map": "auto",
        "temperature": 0.2,
        "max_new_tokens": 512,
        "do_sample": False,
        "system_prompt": "You are an expert medical imaging specialist.",
    },
    
    # MedGemma 1.5专用配置 (来自V4)
    "medgemma15": {
        "path": "/mnt/fast/nobackup/scratch4weeks/ly0008/cxx/MedSAM/work_dir/medsam-vit-base/scripts/pipeline/Multi-scenario/Pipeline_for_All/models/medgemma-1.5-4b-it",
        "torch_dtype": "bfloat16",
        "device_map": "auto",
        "temperature": 0.2,
        "max_new_tokens": 512,
        "do_sample": False,
        "system_prompt": "You are an expert medical imaging specialist.",
    },
    
    # Gemini API配置 (闭源，来自V4)
    "gemini": {
        "api_key": None,
        "model": "gemini-1.5-pro",
        "temperature": 0.2,
        "max_new_tokens": 512,
        "system_prompt": "You are an expert medical imaging specialist.",
        "max_retries": 3,
        "retry_delay": 5,
    },
    
    # GPT-4V API配置 (闭源，来自V4)
    "gpt4v": {
        "api_key": None,
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o",
        "temperature": 0.2,
        "max_new_tokens": 512,
        "system_prompt": "You are an expert medical imaging specialist.",
        "image_detail": "high",
        "max_retries": 3,
        "retry_delay": 5,
    },
    
    "datasets": {
        "omnimedvqa": {
            "json_folder": "/mnt/fast/nobackup/scratch4weeks/ly0008/cxx/MedSAM/work_dir/medsam-vit-base/scripts/pipeline/data/OmniMedVQA/OmniMedVQA/QA_information/Open-access",
            "image_base_dir": "/mnt/fast/nobackup/scratch4weeks/ly0008/cxx/MedSAM/work_dir/medsam-vit-base/scripts/pipeline/data/OmniMedVQA/OmniMedVQA",
            "perturbed_dir": "/mnt/fast/nobackup/scratch4weeks/ly0008/cxx/MedSAM/work_dir/medsam-vit-base/scripts/pipeline/data/OmniMedVQA/OmniMedVQA/perturbed_images",
            "default_task": "vqa",
        },
        "roco": {
            "parquet_folder": "/mnt/fast/nobackup/scratch4weeks/ly0008/cxx/MedSAM/work_dir/medsam-vit-base/scripts/pipeline/data/ROCOv2-radiology/data",
            "extracted_dir": "/mnt/fast/nobackup/scratch4weeks/ly0008/cxx/MedSAM/work_dir/medsam-vit-base/scripts/pipeline/data/ROCOv2-radiology/extracted",
            "perturbed_dir": "/mnt/fast/nobackup/scratch4weeks/ly0008/cxx/MedSAM/work_dir/medsam-vit-base/scripts/pipeline/data/ROCOv2-radiology/perturbed_images",
            "use_extracted": True,
            "split": "test",
            "default_task": "caption",
        },
        "mecovqa": {
            "json_file": "/mnt/fast/nobackup/scratch4weeks/ly0008/cxx/MedSAM/work_dir/medsam-vit-base/scripts/pipeline/data/Medical_Visual_Grounding/MeCoVQA-G-Plus/MeCoVQA_Grounding_test_merged_modified_v3.json",
            "bbox_json_file": "/mnt/fast/nobackup/scratch4weeks/ly0008/cxx/MedSAM/work_dir/medsam-vit-base/scripts/pipeline/data/Medical_Visual_Grounding/MeCoVQA-G-Plus/mecovqa_bbox_format_test.json",
            "train_json_file": "/mnt/fast/nobackup/scratch4weeks/ly0008/cxx/MedSAM/work_dir/medsam-vit-base/scripts/pipeline/data/Medical_Visual_Grounding/MeCoVQA-G-Plus/mecovqa_bbox_format_train.json",
            "base_dir": "/mnt/fast/nobackup/scratch4weeks/ly0008/cxx/MedSAM/work_dir/medsam-vit-base/scripts/pipeline/data/Medical_Visual_Grounding/MeCoVQA-G-Plus",
            "perturbed_dir": "/mnt/fast/nobackup/scratch4weeks/ly0008/cxx/MedSAM/work_dir/medsam-vit-base/scripts/pipeline/data/Medical_Visual_Grounding/MeCoVQA-G-Plus/perturbed_images",
            "default_task": "grounding",
        },
    },
    
    "prompts": {
        "vqa": {
            "with_options": "{question}\n\nOptions:\n{options}\n\nPlease select the correct answer.",
            "without_options": "{question}",
        },
        "caption": {
            "default": "Describe this medical image in detail.",
        },
        "grounding": {
            "use_original": True,
        },
    },
    
    "finetune": {
        "output_dir": "./outputs/finetune",
        "lora_r": 128,
        "lora_alpha": 256,
        "lora_dropout": 0.05,
        "epochs": 3,
        "batch_size": 64,
        "gradient_accumulation_steps": 4,
        "learning_rate": 2e-4,
        "warmup_ratio": 0.03,
        "weight_decay": 0.0,
        "max_grad_norm": 1.0,
        "bf16": True,
        "logging_steps": 10,
        "save_steps": 500,
        "save_total_limit": 3,
    },
    
    "deepseek": {
        "api_key": "YOUR_DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",
        "temperature": 0.1,
        "max_tokens": 100,
    },
    
    "evaluation": {
        "use_deepseek": True,
        "iou_thresholds": [0.5, 0.75],
    },
}


# ============================================
# 数据集适配器 (来自V3，保持不变)
# ============================================

class DatasetAdapter(ABC):
    def __init__(self, config):
        self.config = config
        self.samples = []
        self._original_count = 0
    
    @abstractmethod
    def load_all_data(self):
        pass
    
    def load_data(self, sample_num=-1, seed=42):
        all_samples = self.load_all_data()
        self._original_count = len(all_samples)
        
        if sample_num > 0 and len(all_samples) > sample_num:
            random.seed(seed)
            all_samples = random.sample(all_samples, sample_num)
            print(f"  采样: {self._original_count:,} -> {len(all_samples):,}")
        elif sample_num > 0 and len(all_samples) <= sample_num:
            print(f"  注意: 请求 {sample_num} 个样本，实际只有 {self._original_count} 个，将使用全部数据")
        else:
            print(f"  使用全部数据: {self._original_count:,}")
        
        self.samples = all_samples
        return all_samples
    
    def get_original_count(self):
        return self._original_count
    
    @abstractmethod
    def get_image(self, sample, perturbed=False, pert_type=None, severity=None):
        pass
    
    @abstractmethod
    def get_prompt(self, sample, task_config):
        pass
    
    @abstractmethod
    def get_ground_truth(self, sample):
        pass


class OmniMedVQAAdapter(DatasetAdapter):
    def __init__(self, config):
        super().__init__(config)
        self.json_folder = Path(config['json_folder'])
        self.image_base_dir = Path(config['image_base_dir'])
        self.perturbed_dir = Path(config.get('perturbed_dir', ''))
    
    def load_all_data(self):
        all_samples = []
        for jf in sorted(self.json_folder.glob("*.json")):
            with open(jf, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for item in (data if isinstance(data, list) else [data]):
                img_path = item.get('image_path', '')
                if '${dataset_root_path}' in img_path:
                    img_path = img_path.replace('${dataset_root_path}/', '')
                full_path = self.image_base_dir / img_path
                if full_path.exists():
                    item['_image_path'] = str(full_path)
                    item['_relative_path'] = img_path
                    all_samples.append(item)
        return all_samples
    
    def get_image(self, sample, perturbed=False, pert_type=None, severity=None):
        relative_path = sample.get('_relative_path', '')
        
        if perturbed and pert_type and severity:
            filename = Path(relative_path).name
            if not filename.lower().endswith('.png'):
                filename = Path(filename).stem + '.png'
            return self.perturbed_dir / pert_type / str(severity) / filename
        else:
            return Path(sample['_image_path'])
    
    def get_prompt(self, sample, task_config):
        question = sample.get('question', '')
        options = sample.get('options', [])
        
        if options:
            options_str = '\n'.join([f"  {opt}" for opt in options])
            return task_config.get('with_options', "{question}").format(
                question=question, options=options_str
            )
        else:
            return task_config.get('without_options', "{question}").format(question=question)
    
    def get_ground_truth(self, sample):
        return sample.get('answer', sample.get('gt_answer', ''))
    
    def get_metadata(self, sample):
        return {
            'question_id': sample.get('question_id', ''),
            'modality': sample.get('modality_type', ''),
            'image_path': sample.get('_relative_path', ''),
        }


class ROCOAdapter(DatasetAdapter):
    def __init__(self, config):
        super().__init__(config)
        self.parquet_folder = Path(config['parquet_folder'])
        self.extracted_dir = Path(config.get('extracted_dir', ''))
        self.perturbed_dir = Path(config.get('perturbed_dir', ''))
        self.split = config.get('split', 'test')
        self.use_extracted = config.get('use_extracted', True)
    
    def load_all_data(self):
        # 优先使用提取后的数据
        if self.use_extracted and self.extracted_dir.exists():
            meta_file = self.extracted_dir / self.split / "metadata.json"
            if meta_file.exists():
                print(f"  [ROCO] 使用提取后的格式: {meta_file}")
                with open(meta_file) as f:
                    metadata = json.load(f)
                
                samples = []
                sample_list = metadata.get('samples', metadata) if isinstance(metadata, dict) else metadata
                for item in sample_list:
                    img_id = item.get('image_id', item.get('id', ''))
                    img_path = self.extracted_dir / self.split / "images" / f"{img_id}.png"
                    if img_path.exists():
                        samples.append({
                            'image_id': img_id,
                            'caption': item.get('caption', ''),
                            '_image_path': str(img_path),
                            '_relative_path': f"images/{img_id}.png",
                        })
                return samples
        
        # 使用原始parquet
        print(f"  [ROCO] 使用parquet格式: {self.parquet_folder}")
        all_samples = []
        for pf in sorted(self.parquet_folder.glob(f"{self.split}-*.parquet")):
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
        return all_samples
    
    def get_image(self, sample, perturbed=False, pert_type=None, severity=None):
        if perturbed and pert_type and severity:
            filename = f"{sample['image_id']}.png"
            return self.perturbed_dir / pert_type / str(severity) / filename
        elif '_image_path' in sample:
            return Path(sample['_image_path'])
        else:
            # 从bytes加载
            return Image.open(io.BytesIO(sample['_image_bytes'])).convert('RGB')
    
    def get_prompt(self, sample, task_config):
        return task_config.get('default', "Describe this medical image in detail.")
    
    def get_ground_truth(self, sample):
        return sample.get('caption', '')
    
    def get_metadata(self, sample):
        return {
            'image_id': sample.get('image_id', ''),
        }


class MeCoVQAAdapter(DatasetAdapter):
    def __init__(self, config):
        super().__init__(config)
        self.json_file = Path(config['json_file'])
        self.bbox_json_file = Path(config.get('bbox_json_file', ''))
        self.train_json_file = Path(config.get('train_json_file', ''))
        self.base_dir = Path(config['base_dir'])
        self.perturbed_dir = Path(config.get('perturbed_dir', ''))
    
    def load_all_data(self):
        # 优先使用bbox格式
        if self.bbox_json_file.exists():
            with open(self.bbox_json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        elif self.json_file.exists():
            with open(self.json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            return []
        
        if not isinstance(data, list):
            data = [data]
        
        valid = []
        for s in data:
            img_path = self.base_dir / s.get('image', '')
            if img_path.exists():
                s['_relative_path'] = s.get('image', '')
                valid.append(s)
        
        return valid
    
    def get_finetune_data(self, split="train"):
        """获取微调数据，支持自动划分"""
        if split == "train" and self.train_json_file.exists():
            print(f"📄 加载训练数据: {self.train_json_file}")
            with open(self.train_json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        elif self.bbox_json_file.exists():
            print(f"📄 从bbox文件自动划分: {self.bbox_json_file}")
            with open(self.bbox_json_file, 'r', encoding='utf-8') as f:
                all_data = json.load(f)
            
            if not isinstance(all_data, list):
                all_data = [all_data]
            
            random.seed(42)
            shuffled = all_data.copy()
            random.shuffle(shuffled)
            split_idx = int(len(shuffled) * 0.8)
            
            if split == "train":
                data = shuffled[:split_idx]
                print(f"   训练集: {len(data)} 样本 (80%)")
            else:
                data = shuffled[split_idx:]
                print(f"   验证集: {len(data)} 样本 (20%)")
        elif self.json_file.exists():
            print(f"📄 从主文件自动划分: {self.json_file}")
            with open(self.json_file, 'r', encoding='utf-8') as f:
                all_data = json.load(f)
            
            if not isinstance(all_data, list):
                all_data = [all_data]
            
            random.seed(42)
            shuffled = all_data.copy()
            random.shuffle(shuffled)
            split_idx = int(len(shuffled) * 0.8)
            
            if split == "train":
                data = shuffled[:split_idx]
                print(f"   训练集: {len(data)} 样本 (80%)")
            else:
                data = shuffled[split_idx:]
                print(f"   验证集: {len(data)} 样本 (20%)")
        else:
            print(f"❌ 未找到数据文件!")
            return None
        
        valid_samples = []
        for sample in data:
            img_path = self.base_dir / sample.get('image', '')
            if img_path.exists():
                valid_samples.append(sample)
        
        print(f"   有效样本: {len(valid_samples)} / {len(data)}")
        return valid_samples
    
    def get_image(self, sample, perturbed=False, pert_type=None, severity=None):
        relative_path = sample.get('_relative_path', sample.get('image', ''))
        
        if perturbed and pert_type and severity:
            filename = Path(relative_path).name
            if not filename.lower().endswith('.png'):
                filename = Path(filename).stem + '.png'
            return self.perturbed_dir / pert_type / str(severity) / filename
        else:
            return self.base_dir / relative_path
    
    def get_prompt(self, sample, task_config):
        for conv in sample.get('conversations', []):
            if conv.get('from') == 'human':
                value = conv.get('value', '')
                value = value.replace('<image>\n', '').replace('<image>', '')
                return value.strip()
        return "Please locate the target region in this image."
    
    def get_ground_truth(self, sample):
        for conv in sample.get('conversations', []):
            if conv.get('from') == 'gpt':
                return conv.get('value', '')
        return None
    
    def get_metadata(self, sample):
        return {
            'id': sample.get('id', ''),
            'image': sample.get('image', ''),
        }


ADAPTERS = {
    'omnimedvqa': OmniMedVQAAdapter,
    'roco': ROCOAdapter,
    'mecovqa': MeCoVQAAdapter,
}


# ============================================
# 模型推理器基类
# ============================================

class BaseModelInference(ABC):
    """模型推理器基类"""
    
    def __init__(self, config):
        self.config = config
        self.model = None
    
    @abstractmethod
    def load_model(self):
        pass
    
    @abstractmethod
    def inference(self, image, prompt):
        pass


# ============================================
# LLaVA-Med 推理器 (来自V3，保持不变)
# ============================================

class LLaVAMedInference(BaseModelInference):
    def __init__(self, config):
        super().__init__(config)
        self.tokenizer = None
        self.image_processor = None
    
    def _is_lora_model(self, model_path):
        """检查是否是LoRA模型"""
        model_path = Path(model_path)
        return (model_path / "adapter_config.json").exists() or \
               (model_path / "training_config.json").exists()
    
    def load_model(self):
        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import get_model_name_from_path
        
        model_path = self.config['path']
        print(f"\n加载LLaVA-Med模型: {model_path}")
        
        if self._is_lora_model(model_path):
            print("   检测到LoRA模型，加载中...")
            self._load_lora_model(model_path)
        else:
            print("   加载完整模型...")
            model_name = get_model_name_from_path(model_path)
            self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
                model_path=model_path,
                model_base=None,
                model_name=model_name,
                device_map="auto"
            )
        
        if not hasattr(self.model.config, 'sliding_window') or self.model.config.sliding_window is None:
            self.model.config.sliding_window = 4096
        
        print("✓ LLaVA-Med模型加载完成!")
    
    def _load_lora_model(self, lora_path):
        """加载LoRA微调后的模型"""
        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import get_model_name_from_path
        from peft import PeftModel
        
        lora_path = Path(lora_path)
        
        training_config_file = lora_path / "training_config.json"
        if training_config_file.exists():
            with open(training_config_file, 'r') as f:
                training_config = json.load(f)
            base_model_path = training_config.get('model_config', {}).get('path', '../models/llava-med-v1.5-mistral-7b')
        else:
            base_model_path = '../models/llava-med-v1.5-mistral-7b'
        
        print(f"   基础模型: {base_model_path}")
        print(f"   LoRA权重: {lora_path}")
        
        model_name = get_model_name_from_path(base_model_path)
        self.tokenizer, base_model, self.image_processor, _ = load_pretrained_model(
            model_path=base_model_path,
            model_base=None,
            model_name=model_name,
            device_map="auto"
        )
        
        print("   加载LoRA权重...")
        self.model = PeftModel.from_pretrained(base_model, lora_path)
        
        print("   合并LoRA权重...")
        self.model = self.model.merge_and_unload()
        self.model.eval()
        
        print("   ✓ LoRA模型加载完成!")
    
    def inference(self, image, prompt):
        from llava.mm_utils import process_images, tokenizer_image_token
        from llava.conversation import conv_templates
        from llava.constants import IMAGE_TOKEN_INDEX
        
        try:
            if isinstance(image, (str, Path)):
                image = Image.open(image).convert('RGB')
            elif not isinstance(image, Image.Image):
                return None, "Invalid image type"
            
            conv = conv_templates[self.config['conv_template']].copy()
            conv.append_message(conv.roles[0], f"<image>\n{prompt}")
            conv.append_message(conv.roles[1], None)
            prompt_text = conv.get_prompt()
            
            image_tensor = process_images([image], self.image_processor, self.model.config)
            if isinstance(image_tensor, list):
                image_tensor = [t.to(self.model.device, dtype=torch.float16) for t in image_tensor]
            else:
                image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)
            
            input_ids = tokenizer_image_token(prompt_text, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
            input_ids = input_ids.unsqueeze(0).to(self.model.device)

            input_len = input_ids.shape[-1] # Abner
            
            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=image_tensor,
                    image_sizes=[image.size],
                    do_sample=self.config['temperature'] > 0,
                    temperature=self.config['temperature'],
                    max_new_tokens=self.config['max_new_tokens'],
                    use_cache=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )


            output_ids = output_ids[:, input_len:] # Abner
            response = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
			
            return response, None
            
        except Exception as e:
            return None, str(e)


# ============================================
# MedGemma 推理器 (支持LoRA微调模型)
# ============================================

class MedGemmaInference(BaseModelInference):
    """MedGemma模型推理器 (支持加载LoRA微调模型)"""
    
    def __init__(self, config):
        super().__init__(config)
        self.processor = None
    
    def _is_lora_model(self, model_path):
        """检查是否是LoRA模型"""
        model_path = Path(model_path)
        return (model_path / "adapter_config.json").exists() or \
               (model_path / "training_config.json").exists()
    
    def load_model(self):
        """加载MedGemma模型"""
        from transformers import AutoProcessor, AutoModelForImageTextToText
        
        model_path = self.config['path']
        # 将相对路径转换为绝对路径，避免huggingface_hub误认为是repo_id
        if model_path.startswith('./') or model_path.startswith('../'):
            model_path = str(Path(model_path).resolve())
        print(f"\n加载MedGemma模型: {model_path}")
        
        dtype_map = {
            'bfloat16': torch.bfloat16,
            'float16': torch.float16,
            'float32': torch.float32,
        }
        torch_dtype = dtype_map.get(self.config.get('torch_dtype', 'bfloat16'), torch.bfloat16)
        
        print(f"   数据类型: {torch_dtype}")
        print(f"   设备映射: {self.config.get('device_map', 'auto')}")
        
        if self._is_lora_model(model_path):
            print("   检测到LoRA模型，加载中...")
            self._load_lora_model(model_path, torch_dtype)
        else:
            print("   加载完整模型...")
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                device_map=self.config.get('device_map', 'auto'),
                trust_remote_code=True,
            )
            
            self.processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=True,
            )
        
        self.model.eval()
        print("✓ MedGemma模型加载完成!")
    
    def _load_lora_model(self, lora_path, torch_dtype):
        """加载LoRA微调后的MedGemma模型"""
        from transformers import AutoProcessor, AutoModelForImageTextToText
        from peft import PeftModel
        
        lora_path = Path(lora_path)
        
        # 读取训练配置获取基础模型路径
        training_config_file = lora_path / "training_config.json"
        if training_config_file.exists():
            with open(training_config_file, 'r') as f:
                training_config = json.load(f)
            base_model_path = training_config.get('model_config', {}).get('path', self.config.get('path'))
        else:
            # 尝试从adapter_config.json获取
            adapter_config_file = lora_path / "adapter_config.json"
            if adapter_config_file.exists():
                with open(adapter_config_file, 'r') as f:
                    adapter_config = json.load(f)
                base_model_path = adapter_config.get('base_model_name_or_path', self.config.get('path'))
            else:
                base_model_path = self.config.get('path')
        
        print(f"   基础模型: {base_model_path}")
        print(f"   LoRA权重: {lora_path}")
        
        # 将相对路径转换为绝对路径，避免huggingface_hub误认为是repo_id
        if isinstance(base_model_path, str) and (base_model_path.startswith('./') or base_model_path.startswith('../')):
            base_model_path = str(Path(base_model_path).resolve())
        
        # 加载基础模型
        base_model = AutoModelForImageTextToText.from_pretrained(
            base_model_path,
            torch_dtype=torch_dtype,
            device_map=self.config.get('device_map', 'auto'),
            trust_remote_code=True,
        )
        
        self.processor = AutoProcessor.from_pretrained(
            base_model_path,
            trust_remote_code=True,
        )
        
        print("   加载LoRA权重...")
        self.model = PeftModel.from_pretrained(base_model, lora_path)
        
        print("   合并LoRA权重...")
        self.model = self.model.merge_and_unload()
        self.model.eval()
        
        print("   ✓ LoRA模型加载完成!")
    
    def inference(self, image, prompt):
        """执行MedGemma推理"""
        try:
            if isinstance(image, (str, Path)):
                image = Image.open(image).convert('RGB')
            elif not isinstance(image, Image.Image):
                return None, "Invalid image type"
            
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": self.config.get('system_prompt', 'You are an expert medical imaging specialist.')}]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image", "image": image}
                    ]
                }
            ]
            
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            ).to(self.model.device)
            
            dtype_map = {
                'bfloat16': torch.bfloat16,
                'float16': torch.float16,
                'float32': torch.float32,
            }
            torch_dtype = dtype_map.get(self.config.get('torch_dtype', 'bfloat16'), torch.bfloat16)
            
            if 'pixel_values' in inputs:
                inputs['pixel_values'] = inputs['pixel_values'].to(torch_dtype)
            
            input_len = inputs["input_ids"].shape[-1]
            
            with torch.inference_mode():
                generation = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.get('max_new_tokens', 512),
                    do_sample=self.config.get('do_sample', False),
                    temperature=self.config.get('temperature', 0.2) if self.config.get('do_sample', False) else None,
                )
                generation = generation[0][input_len:]
            
            response = self.processor.decode(generation, skip_special_tokens=True)
            return response.strip(), None
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return None, str(e)


# ============================================
# Gemini 推理器 (闭源API，来自V4)
# ============================================

class GeminiInference(BaseModelInference):
    """Google Gemini模型推理器 (通过API调用)"""
    
    def __init__(self, config):
        super().__init__(config)
    
    def load_model(self):
        """初始化Gemini API客户端"""
        import google.generativeai as genai
        
        api_key = self.config.get('api_key')
        if not api_key:
            raise ValueError("Gemini API Key未设置!")
        
        print(f"\n初始化Gemini API...")
        print(f"   模型: {self.config.get('model', 'gemini-1.5-pro')}")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            model_name=self.config.get('model', 'gemini-1.5-pro'),
            system_instruction=self.config.get('system_prompt', 'You are an expert medical imaging specialist.')
        )
        
        print("✓ Gemini API初始化完成!")
    
    def inference(self, image, prompt):
        """执行Gemini推理"""
        import time
        
        try:
            if isinstance(image, (str, Path)):
                image = Image.open(image).convert('RGB')
            elif not isinstance(image, Image.Image):
                return None, "Invalid image type"
            
            max_retries = self.config.get('max_retries', 3)
            retry_delay = self.config.get('retry_delay', 5)
            
            for attempt in range(max_retries):
                try:
                    response = self.model.generate_content(
                        [prompt, image],
                        generation_config={
                            "temperature": self.config.get('temperature', 0.2),
                            "max_output_tokens": self.config.get('max_new_tokens', 512),
                        }
                    )
                    
                    return response.text.strip(), None
                    
                except Exception as e:
                    if attempt < max_retries - 1:
                        print(f"  Gemini API错误，重试中 ({attempt + 1}/{max_retries}): {e}")
                        time.sleep(retry_delay)
                    else:
                        raise e
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return None, str(e)


# ============================================
# GPT-4V 推理器 (闭源API，来自V4)
# ============================================

class GPT4VInference(BaseModelInference):
    """OpenAI GPT-4V模型推理器 (通过API调用)"""
    
    def __init__(self, config):
        super().__init__(config)
        self.client = None
    
    def load_model(self):
        """初始化OpenAI API客户端"""
        from openai import OpenAI
        
        api_key = self.config.get('api_key')
        if not api_key:
            raise ValueError("OpenAI API Key未设置!")
        
        print(f"\n初始化OpenAI API...")
        print(f"   模型: {self.config.get('model', 'gpt-4o')}")
        print(f"   Base URL: {self.config.get('base_url', 'https://api.openai.com/v1')}")
        
        self.client = OpenAI(
            api_key=api_key,
            base_url=self.config.get('base_url', 'https://api.openai.com/v1')
        )
        
        print("✓ OpenAI API初始化完成!")
    
    def _encode_image_base64(self, image):
        """将图像编码为base64"""
        import base64
        from io import BytesIO
        
        if isinstance(image, (str, Path)):
            with open(image, "rb") as f:
                return base64.b64encode(f.read()).decode('utf-8')
        elif isinstance(image, Image.Image):
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
        return None
    
    def inference(self, image, prompt):
        """执行GPT-4V推理"""
        import time
        
        try:
            if isinstance(image, (str, Path)):
                image_pil = Image.open(image).convert('RGB')
            elif isinstance(image, Image.Image):
                image_pil = image
            else:
                return None, "Invalid image type"
            
            image_base64 = self._encode_image_base64(image_pil)
            if not image_base64:
                return None, "Failed to encode image"
            
            messages = [
                {
                    "role": "system",
                    "content": self.config.get('system_prompt', 'You are an expert medical imaging specialist.')
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}",
                                "detail": self.config.get('image_detail', 'high')
                            }
                        }
                    ]
                }
            ]
            
            max_retries = self.config.get('max_retries', 3)
            retry_delay = self.config.get('retry_delay', 5)
            
            for attempt in range(max_retries):
                try:
                    response = self.client.chat.completions.create(
                        model=self.config.get('model', 'gpt-4o'),
                        messages=messages,
                        temperature=self.config.get('temperature', 0.2),
                        max_tokens=self.config.get('max_new_tokens', 512),
                    )
                    
                    return response.choices[0].message.content.strip(), None
                    
                except Exception as e:
                    if attempt < max_retries - 1:
                        print(f"  OpenAI API错误，重试中 ({attempt + 1}/{max_retries}): {e}")
                        time.sleep(retry_delay)
                    else:
                        raise e
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return None, str(e)


# ============================================
# 模型工厂函数
# ============================================

def create_model_inference(config, model_type=None):
    """创建模型推理器"""
    if model_type is None:
        model_type = config.get('model', {}).get('type', 'llava-med')
    
    if model_type == 'llava-med':
        return LLaVAMedInference(config['model'])
    elif model_type == 'medgemma':
        return MedGemmaInference(config['medgemma'])
    elif model_type == 'medgemma15':
        return MedGemmaInference(config['medgemma15'])
    elif model_type == 'gemini':
        return GeminiInference(config['gemini'])
    elif model_type == 'gpt4v':
        return GPT4VInference(config['gpt4v'])
    else:
        raise ValueError(f"未知的模型类型: {model_type}")


def check_api_available(config, model_type):
    """检查闭源API是否可用"""
    if model_type == 'gemini':
        api_key = config.get('gemini', {}).get('api_key')
        return api_key is not None and api_key != ''
    elif model_type == 'gpt4v':
        api_key = config.get('gpt4v', {}).get('api_key')
        return api_key is not None and api_key != ''
    return True


# ============================================
# LLaVA-Med 微调器
# ============================================

class LLaVAMedFineTuner:
    """LLaVA-Med LoRA微调器"""
    
    def __init__(self, model_config, finetune_config):
        self.model_config = model_config
        self.finetune_config = finetune_config
        self.model = None
        self.tokenizer = None
        self.image_processor = None

    def prepare_model(self):
        """准备模型和LoRA"""
        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import get_model_name_from_path
        from peft import LoraConfig, get_peft_model
        
        print(f"\n加载基础模型: {self.model_config['path']}")
        
        model_name = get_model_name_from_path(self.model_config['path'])
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            model_path=self.model_config['path'],
            model_base=None,
            model_name=model_name,
            device_map="auto",
        )
        
        if not hasattr(self.model.config, 'sliding_window') or self.model.config.sliding_window is None:
            self.model.config.sliding_window = 4096
        
        if self.finetune_config['bf16']:
            self.model = self.model.to(torch.bfloat16)
        else:
            self.model = self.model.to(torch.float16)
        
        print("\n配置LoRA...")
        lora_config = LoraConfig(
            r=self.finetune_config['lora_r'],
            lora_alpha=self.finetune_config['lora_alpha'],
            lora_dropout=self.finetune_config['lora_dropout'],
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        self.model.enable_input_require_grads()
        self.model = get_peft_model(self.model, lora_config)
        
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"✓ LoRA配置完成!")
        print(f"   可训练参数: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
        print(f"   总参数: {total_params:,}")
        
        return self.model, self.tokenizer, self.image_processor
    
    def create_dataset(self, samples, image_base_dir):
        """创建PyTorch Dataset"""
        from torch.utils.data import Dataset
        from llava.conversation import conv_templates
        
        class LLaVAFinetuneDataset(Dataset):
            def __init__(self, samples, tokenizer, image_processor, model_config, image_base_dir):
                self.samples = samples
                self.tokenizer = tokenizer
                self.image_processor = image_processor
                self.model_config = model_config
                self.image_base_dir = Path(image_base_dir)
            
            def __len__(self):
                return len(self.samples)
            
            def __getitem__(self, idx):
                sample = self.samples[idx]
                
                image_path = self.image_base_dir / sample.get('image', '')
                try:
                    image = Image.open(image_path).convert('RGB')
                except Exception as e:
                    print(f"Warning: Failed to load image {image_path}: {e}")
                    image = Image.new('RGB', (224, 224), (128, 128, 128))
                
                conversations = sample.get('conversations', [])
                human_msg = ""
                gpt_msg = ""
                for conv in conversations:
                    if conv['from'] == 'human':
                        human_msg = conv['value']
                    elif conv['from'] == 'gpt':
                        gpt_msg = conv['value']
                
                conv_template = conv_templates[self.model_config['conv_template']].copy()
                conv_template.append_message(conv_template.roles[0], human_msg)
                conv_template.append_message(conv_template.roles[1], gpt_msg)
                full_prompt = conv_template.get_prompt()
                
                return {
                    'image': image,
                    'conversations': full_prompt,
                    'human_msg': human_msg,
                    'gpt_msg': gpt_msg,
                }
        
        return LLaVAFinetuneDataset(
            samples, self.tokenizer, self.image_processor, 
            self.model_config, image_base_dir
        )
    
    def train(self, train_samples, image_base_dir, output_dir):
        """执行微调训练"""
        from transformers import get_linear_schedule_with_warmup
        from llava.mm_utils import process_images, tokenizer_image_token
        from llava.constants import IMAGE_TOKEN_INDEX
        
        print("\n" + "=" * 70)
        print("开始微调训练")
        print("=" * 70)
        
        self.prepare_model()
        
        train_dataset = self.create_dataset(train_samples, image_base_dir)
        print(f"训练样本数: {len(train_dataset)}")
        
        epochs = self.finetune_config['epochs']
        batch_size = self.finetune_config['batch_size']
        grad_accum = self.finetune_config['gradient_accumulation_steps']
        lr = self.finetune_config['learning_rate']
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=self.finetune_config['weight_decay']
        )
        
        total_steps = (len(train_dataset) // batch_size // grad_accum) * epochs
        warmup_steps = int(total_steps * self.finetune_config['warmup_ratio'])
        scheduler = get_linear_schedule_with_warmup(
            optimizer, warmup_steps, total_steps
        )
        
        self.model.train()
        global_step = 0
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            epoch_loss = 0
            
            indices = list(range(len(train_dataset)))
            random.shuffle(indices)
            
            pbar = tqdm(range(0, len(indices), batch_size), desc=f"Training")
            
            for batch_start in pbar:
                batch_indices = indices[batch_start:batch_start + batch_size]
                batch_loss = 0
                
                for idx in batch_indices:
                    sample = train_dataset[idx]
                    
                    try:
                        image = sample['image']
                        image_tensor = process_images([image], self.image_processor, self.model.config)
                        if isinstance(image_tensor, list):
                            image_tensor = image_tensor[0]
                        image_tensor = image_tensor.to(self.model.device, dtype=torch.bfloat16 if self.finetune_config['bf16'] else torch.float16)
                   		
                        full_text = sample['conversations']
                        input_ids = tokenizer_image_token(
                            full_text, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
                        ).to(self.model.device)
                        
                        labels = input_ids.clone()
                        
                        # 只对assistant回复部分计算loss
                        gpt_msg = sample['gpt_msg']
                        gpt_start = full_text.rfind(gpt_msg)
                        if gpt_start > 0:
                            prefix_text = full_text[:gpt_start]
                            prefix_ids = tokenizer_image_token(
                                prefix_text, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
                            )
                            labels[:len(prefix_ids)] = -100
                        
                        
                        outputs = self.model(
                            input_ids=input_ids.unsqueeze(0),
                            images=image_tensor.unsqueeze(0),
                            labels=labels.unsqueeze(0),
                        )
                        
                        loss = outputs.loss / grad_accum
                        loss.backward()
                        batch_loss += loss.item()
                        
                    except Exception as e:
                        print(f"Warning: Error processing sample {idx}: {e}")
                        continue
                
                if (batch_start // batch_size + 1) % grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.finetune_config['max_grad_norm']
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                
                epoch_loss += batch_loss
                pbar.set_postfix({'loss': f'{batch_loss:.4f}'})
                
                if global_step > 0 and global_step % self.finetune_config['save_steps'] == 0:
                    checkpoint_dir = output_dir / f"checkpoint-{global_step}"
                    self.save_model(checkpoint_dir)
            
            avg_loss = epoch_loss / (len(indices) / batch_size)
            print(f"Epoch {epoch + 1} 平均损失: {avg_loss:.4f}")
        
        final_dir = output_dir / "final"
        self.save_model(final_dir)
        print(f"\n✓ 训练完成! 模型保存至: {final_dir}")
        
        return final_dir
    
    def save_model(self, output_dir):
        """保存LoRA权重"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        config = {
            'model_config': self.model_config,
            'finetune_config': self.finetune_config,
        }
        with open(output_dir / 'training_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"   保存checkpoint: {output_dir}")


# ============================================
# MedGemma 微调器 (新增)
# ============================================

class MedGemmaFineTuner:
    """MedGemma LoRA微调器 (支持MedGemma和MedGemma 1.5)"""
    
    def __init__(self, model_config, finetune_config, model_type='medgemma'):
        self.model_config = model_config
        self.finetune_config = finetune_config
        self.model_type = model_type  # 'medgemma' or 'medgemma15'
        self.model = None
        self.processor = None
        self.torch_dtype = None
    
    def prepare_model(self):
        """准备模型和LoRA"""
        from transformers import AutoProcessor, AutoModelForImageTextToText
        from peft import LoraConfig, get_peft_model
        
        model_path = self.model_config['path']
        print(f"\n加载基础模型: {model_path}")
        
        dtype_map = {
            'bfloat16': torch.bfloat16,
            'float16': torch.float16,
            'float32': torch.float32,
        }
        self.torch_dtype = dtype_map.get(self.model_config.get('torch_dtype', 'bfloat16'), torch.bfloat16)
        
        print(f"   数据类型: {self.torch_dtype}")
        
        # 加载基础模型
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=self.torch_dtype,
            device_map=self.model_config.get('device_map', 'auto'),
            trust_remote_code=True,
        )
        
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        
        print("\n配置LoRA...")
        
        # MedGemma的target_modules
        # 根据模型架构选择合适的目标模块
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
        
        lora_config = LoraConfig(
            r=self.finetune_config['lora_r'],
            lora_alpha=self.finetune_config['lora_alpha'],
            lora_dropout=self.finetune_config['lora_dropout'],
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        # 启用梯度计算
        self.model.enable_input_require_grads()
        self.model = get_peft_model(self.model, lora_config)
        
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"✓ LoRA配置完成!")
        print(f"   可训练参数: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
        print(f"   总参数: {total_params:,}")
        
        return self.model, self.processor
    
    def create_dataset(self, samples, image_base_dir):
        """创建PyTorch Dataset"""
        from torch.utils.data import Dataset
        
        class MedGemmaFinetuneDataset(Dataset):
            def __init__(self, samples, processor, model_config, image_base_dir, torch_dtype):
                self.samples = samples
                self.processor = processor
                self.model_config = model_config
                self.image_base_dir = Path(image_base_dir)
                self.torch_dtype = torch_dtype
            
            def __len__(self):
                return len(self.samples)
            
            def __getitem__(self, idx):
                sample = self.samples[idx]
                
                image_path = self.image_base_dir / sample.get('image', '')
                try:
                    image = Image.open(image_path).convert('RGB')
                except Exception as e:
                    print(f"Warning: Failed to load image {image_path}: {e}")
                    image = Image.new('RGB', (224, 224), (128, 128, 128))
                
                # 提取对话内容
                conversations = sample.get('conversations', [])
                human_msg = ""
                gpt_msg = ""
                for conv in conversations:
                    if conv['from'] == 'human':
                        human_msg = conv['value']
                        # 移除<image>标记
                        human_msg = human_msg.replace('<image>\n', '').replace('<image>', '')
                    elif conv['from'] == 'gpt':
                        gpt_msg = conv['value']
                
                return {
                    'image': image,
                    'human_msg': human_msg.strip(),
                    'gpt_msg': gpt_msg.strip(),
                }
        
        return MedGemmaFinetuneDataset(
            samples, self.processor, self.model_config, 
            image_base_dir, self.torch_dtype
        )
    
    def train(self, train_samples, image_base_dir, output_dir):
        """执行微调训练"""
        from transformers import get_linear_schedule_with_warmup
        
        print("\n" + "=" * 70)
        print(f"开始微调训练 ({self.model_type})")
        print("=" * 70)
        
        self.prepare_model()
        
        train_dataset = self.create_dataset(train_samples, image_base_dir)
        print(f"训练样本数: {len(train_dataset)}")
        
        epochs = self.finetune_config['epochs']
        batch_size = self.finetune_config['batch_size']
        grad_accum = self.finetune_config['gradient_accumulation_steps']
        lr = self.finetune_config['learning_rate']
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=self.finetune_config['weight_decay']
        )
        
        total_steps = (len(train_dataset) // batch_size // grad_accum) * epochs
        warmup_steps = int(total_steps * self.finetune_config['warmup_ratio'])
        scheduler = get_linear_schedule_with_warmup(
            optimizer, warmup_steps, total_steps
        )
        
        self.model.train()
        global_step = 0
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            epoch_loss = 0
            
            indices = list(range(len(train_dataset)))
            random.shuffle(indices)
            
            pbar = tqdm(range(0, len(indices), batch_size), desc=f"Training")
            
            for batch_start in pbar:
                batch_indices = indices[batch_start:batch_start + batch_size]
                batch_loss = 0
                
                for idx in batch_indices:
                    sample = train_dataset[idx]
                    
                    try:
                        image = sample['image']
                        human_msg = sample['human_msg']
                        gpt_msg = sample['gpt_msg']
                        
                        # 构建消息格式
                        messages = [
                            {
                                "role": "system",
                                "content": [{"type": "text", "text": self.model_config.get('system_prompt', 'You are an expert medical imaging specialist.')}]
                            },
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": human_msg},
                                    {"type": "image", "image": image}
                                ]
                            },
                            {
                                "role": "assistant",
                                "content": [{"type": "text", "text": gpt_msg}]
                            }
                        ]
                        
                        # 处理输入
                        inputs = self.processor.apply_chat_template(
                            messages,
                            add_generation_prompt=False,
                            tokenize=True,
                            return_dict=True,
                            return_tensors="pt"
                        ).to(self.model.device)
                        
                        if 'pixel_values' in inputs:
                            inputs['pixel_values'] = inputs['pixel_values'].to(self.torch_dtype)
                        
                        # 设置labels
                        # labels = inputs['input_ids'].clone()
                        # 构建不含assistant回复的前缀，计算其token长度
                        prefix_messages = [messages[0], messages[1]]  # 只保留system和user
                        prefix_inputs = self.processor.apply_chat_template(
                            prefix_messages,
                            add_generation_prompt=True,   # 会追加 <start_of_turn>model\n
                            tokenize=True,
                            return_dict=True,
                            return_tensors="pt"
                        )
                        prefix_len = prefix_inputs['input_ids'].shape[1]
                        
                        # 只对assistant回复部分计算loss
                        labels = inputs['input_ids'].clone()
                        labels[:, :prefix_len] = -100
                        
                        # 前向传播
                        outputs = self.model(
                            **inputs,
                            labels=labels,
                        )
                        
                        loss = outputs.loss / grad_accum
                        loss.backward()
                        batch_loss += loss.item()
                        
                    except Exception as e:
                        print(f"Warning: Error processing sample {idx}: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
                
                if (batch_start // batch_size + 1) % grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.finetune_config['max_grad_norm']
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                
                epoch_loss += batch_loss
                pbar.set_postfix({'loss': f'{batch_loss:.4f}'})
                
                if global_step > 0 and global_step % self.finetune_config['save_steps'] == 0:
                    checkpoint_dir = output_dir / f"checkpoint-{global_step}"
                    self.save_model(checkpoint_dir)
            
            avg_loss = epoch_loss / max(1, len(indices) / batch_size)
            print(f"Epoch {epoch + 1} 平均损失: {avg_loss:.4f}")
        
        final_dir = output_dir / "final"
        self.save_model(final_dir)
        print(f"\n✓ 训练完成! 模型保存至: {final_dir}")
        
        return final_dir
    
    def save_model(self, output_dir):
        """保存LoRA权重"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存LoRA权重
        self.model.save_pretrained(output_dir)
        
        # 保存processor (tokenizer等)
        self.processor.save_pretrained(output_dir)
        
        # 保存训练配置
        config = {
            'model_config': self.model_config,
            'finetune_config': self.finetune_config,
            'model_type': self.model_type,
        }
        with open(output_dir / 'training_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"   保存checkpoint: {output_dir}")


# ============================================
# 统一的微调入口函数
# ============================================

def run_finetune(dataset_name, task_name, config, model_type='llava-med'):
    """运行微调 (支持所有开源模型)"""
    print("=" * 70)
    print(f"医学VLM微调 ({model_type})")
    print("=" * 70)
    print(f"数据集:  {dataset_name}")
    print(f"任务:    {task_name}")
    print(f"模型:    {model_type}")
    print(f"LoRA r:  {config['finetune']['lora_r']}")
    print(f"Epochs:  {config['finetune']['epochs']}")
    print("=" * 70)
    
    # 检查模型类型是否支持微调
    if model_type not in ['llava-med', 'medgemma', 'medgemma15']:
        print(f"❌ 模型 {model_type} 不支持微调!")
        print("   支持微调的模型: llava-med, medgemma, medgemma15")
        return
    
    dataset_config = config['datasets'].get(dataset_name)
    if not dataset_config:
        print(f"❌ 未知数据集: {dataset_name}")
        return
    
    adapter = ADAPTERS.get(dataset_name)
    if not adapter:
        print(f"❌ 不支持的数据集: {dataset_name}")
        return
    
    adapter = adapter(dataset_config)
    
    if not hasattr(adapter, 'get_finetune_data'):
        print(f"❌ 数据集 {dataset_name} 不支持微调!")
        return
    
    train_samples = adapter.get_finetune_data(split="train")
    if not train_samples:
        print("❌ 没有找到微调数据!")
        return
    
    # 根据模型类型选择微调器
    if model_type == 'llava-med':
        finetuner = LLaVAMedFineTuner(config['model'], config['finetune'])
    elif model_type == 'medgemma':
        finetuner = MedGemmaFineTuner(config['medgemma'], config['finetune'], model_type='medgemma')
    elif model_type == 'medgemma15':
        finetuner = MedGemmaFineTuner(config['medgemma15'], config['finetune'], model_type='medgemma15')
    
    # 输出目录按模型类型区分
    output_dir = Path(config['finetune']['output_dir']) / model_type / f"{dataset_name}_{task_name}"
    
    final_model_path = finetuner.train(
        train_samples=train_samples,
        image_base_dir=dataset_config['base_dir'],
        output_dir=output_dir
    )
    
    print("\n" + "=" * 70)
    print("微调完成!")
    print("=" * 70)
    print(f"模型保存路径: {final_model_path}")
    print(f"\n使用微调模型评估:")
    
    if model_type == 'llava-med':
        print(f"  python eval_vlm_perturbation.py --dataset {dataset_name} --task {task_name} --model_type {model_type} --model_path {final_model_path}")
    elif model_type == 'medgemma':
        print(f"  python eval_vlm_perturbation.py --dataset {dataset_name} --task {task_name} --model_type {model_type} --medgemma_path {final_model_path}")
    elif model_type == 'medgemma15':
        print(f"  python eval_vlm_perturbation.py --dataset {dataset_name} --task {task_name} --model_type {model_type} --medgemma15_path {final_model_path}")


# ============================================
# 评测流程
# ============================================

def run_evaluation(
    dataset_name: str,
    task_name: str,
    config: dict,
    model_type: str = None,
    perturbed: bool = False,
    pert_type: str = None,
    severity: int = None,
    all_perturbations: bool = False,
):
    """运行评测"""
    if model_type is None:
        model_type = config.get('model', {}).get('type', 'llava-med')
    
    print("=" * 70)
    print("🔬 VLM扰动评测")
    print("=" * 70)
    print(f"模型: {model_type}")
    print(f"数据集: {dataset_name}")
    print(f"任务: {task_name}")
    print(f"扰动模式: {'是' if perturbed else '否'}")
    if perturbed:
        print(f"扰动类型: {pert_type if pert_type else 'all'}")
        print(f"严重程度: {severity if severity else 'all'}")
    print("=" * 70)
    
    dataset_config = config['datasets'].get(dataset_name)
    if not dataset_config:
        print(f"❌ 未知数据集: {dataset_name}")
        return
    
    adapter = ADAPTERS[dataset_name](dataset_config)
    
    print("\n[1/3] 📂 加载数据...")
    samples = adapter.load_data(sample_num=config['sample_num'], seed=config['seed'])
    
    print(f"\n  ✓ 原始数据量: {adapter.get_original_count():,}")
    print(f"  ✓ 实际使用量: {len(samples):,}")
    
    if not samples:
        print("❌ 没有数据!")
        return
    
    print("\n[2/3] 🤖 加载模型...")
    model = create_model_inference(config, model_type)
    model.load_model()
    
    task_config = config['prompts'].get(task_name, {})
    
    if perturbed:
        if all_perturbations:
            perturbed_dir = Path(dataset_config['perturbed_dir'])
            info_file = perturbed_dir / f"generation_info_{dataset_name}.json"
            if info_file.exists():
                with open(info_file) as f:
                    gen_info = json.load(f)
                eval_configs = [
                    (pt, sev) 
                    for pt in gen_info['perturbations'] 
                    for sev in gen_info['severities']
                ]
            else:
                print(f"❌ 未找到扰动信息: {info_file}")
                return
        else:
            eval_configs = [(pert_type, severity)]
    else:
        eval_configs = [(None, None)]
    
    # 输出目录按模型类型区分
    output_dir = Path(config['output_dir']) / model_type / f"{dataset_name}_{task_name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    deepseek_config = config.get('deepseek', {})
    eval_config = config.get('evaluation', {})
    use_deepseek = eval_config.get('use_deepseek', False)
    deepseek_evaluator = None
    
    if use_deepseek and task_name in ['vqa', 'caption']:
        print("\n初始化DeepSeek评估器...")
        deepseek_evaluator = DeepSeekEvaluator(deepseek_config)
        if deepseek_evaluator.enabled:
            print("  ✓ DeepSeek API已启用")
        else:
            print("  ⚠️ DeepSeek API未配置，跳过语义评估")
    
    iou_thresholds = eval_config.get('iou_thresholds', [0.5, 0.75])
    
    print("\n[3/3] 🚀 运行评测...")
    
    all_results = []
    
    for pt, sev in eval_configs:
        desc = f"{pt}_s{sev}" if pt else "original"
        print(f"\n--- {desc} ---")
        
        for i, sample in enumerate(tqdm(samples, desc=desc)):
            try:
                image = adapter.get_image(sample, perturbed=(pt is not None), pert_type=pt, severity=sev)
                
                if image is None:
                    continue
                
                if pt and isinstance(image, Path) and not image.exists():
                    continue
                
                prompt = adapter.get_prompt(sample, task_config)
                
                prediction, error = model.inference(image, prompt)
                
                ground_truth = adapter.get_ground_truth(sample)

                # 保存图像并记录路径
                image_save_dir = output_dir / "images" / f"{pt or 'original'}_s{sev or 0}"
                image_save_dir.mkdir(parents=True, exist_ok=True)
        
                if isinstance(image, (str, Path)):
                    image_pil = Image.open(image).convert('RGB')
                else:
                    image_pil = image
        
                sample_id = sample.get('image_id') or sample.get('question_id') or sample.get('id') or str(i)
                image_filename = f"{sample_id}.png"
                image_save_path = image_save_dir / image_filename
                image_pil.save(image_save_path)
                
                result = {
                    'perturbation': pt if pt else 'none',
                    'severity': sev if sev else 0,
                    'ground_truth': ground_truth,
                    'prediction': prediction,
                    'error': error,
                    'image_save_path': str(image_save_path.relative_to(output_dir)),
                    **adapter.get_metadata(sample),
                }
                
                if task_name == 'grounding' and prediction:
                    pred_bbox = parse_bbox_from_text(prediction)
                    
                    mask_rel_path = parse_mask_path(ground_truth)
                    if mask_rel_path:
                        mask_full_path = Path(dataset_config.get('base_dir', '')) / mask_rel_path
                        gt_bbox = extract_bbox_from_mask(mask_full_path)
                        result['gt_mask_path'] = str(mask_rel_path)
                    else:
                        gt_bbox = parse_bbox_from_text(ground_truth)
                    
                    result['pred_bbox'] = pred_bbox
                    result['gt_bbox'] = gt_bbox
                    
                    if pred_bbox and gt_bbox:
                        iou = calculate_iou(pred_bbox, gt_bbox)
                        result['iou'] = iou
                        for thresh in iou_thresholds:
                            result[f'correct@{thresh}'] = bool(iou >= thresh)
                    else:
                        result['iou'] = 0.0
                        for thresh in iou_thresholds:
                            result[f'correct@{thresh}'] = False
                
                elif task_name == 'vqa' and deepseek_evaluator and deepseek_evaluator.enabled:
                    question = prompt
                    score = deepseek_evaluator.evaluate_vqa(question, prediction, ground_truth)
                    result['semantic_score'] = score
                    if score is not None:
                        result['correct'] = score >= 0.5
                
                elif task_name == 'caption' and deepseek_evaluator and deepseek_evaluator.enabled:
                    score = deepseek_evaluator.evaluate_caption(prediction, ground_truth)
                    result['semantic_score'] = score
                
                all_results.append(result)
                
            except Exception as e:
                print(f"  Error: {e}")
    
    # ============================================
    # Caption任务计算标准指标 (BLEU/ROUGE/CIDEr)
    # ============================================
    if task_name == 'caption' and CAPTION_METRICS_AVAILABLE:
        print("\n[3.5/4] 📊 计算Caption标准指标 (BLEU/ROUGE/CIDEr)...")

        # Abner 
        # predictions = [r['prediction'] for r in all_results if r.get('prediction')]
        # references = [r['ground_truth'] for r in all_results if r.get('ground_truth')]
        valid_pairs = [(r['prediction'], r['ground_truth']) for r in all_results if r.get('prediction') and r.get('ground_truth')]
        predictions = [p[0] for p in valid_pairs]
        references = [p[1] for p in valid_pairs]
        
        if predictions and references:
            caption_metrics = compute_caption_metrics(predictions, references)
            
            from collections import defaultdict
            pert_groups = defaultdict(lambda: {'preds': [], 'refs': []})
            
            for r in all_results:
                key = f"{r['perturbation']}_s{r['severity']}"
                if r.get('prediction') and r.get('ground_truth'):
                    pert_groups[key]['preds'].append(r['prediction'])
                    pert_groups[key]['refs'].append(r['ground_truth'])
            
            group_metrics = {}
            for key, group in pert_groups.items():
                if group['preds']:
                    group_metrics[key] = compute_caption_metrics(group['preds'], group['refs'])
            
            if caption_metrics:
                print(f"  BLEU-1: {caption_metrics.get('BLEU-1', 0):.4f}")
                print(f"  BLEU-4: {caption_metrics.get('BLEU-4', 0):.4f}")
                print(f"  ROUGE-L: {caption_metrics.get('ROUGE-L', 0):.4f}")
                print(f"  CIDEr: {caption_metrics.get('CIDEr', 0):.4f}")
        else:
            caption_metrics = {}
            group_metrics = {}
    else:
        caption_metrics = {}
        group_metrics = {}

    stats = compute_statistics(all_results, task_name, iou_thresholds)
    
    if caption_metrics:
        stats['caption_standard_metrics'] = {
            'overall': caption_metrics,
            'by_group': group_metrics
        }
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = output_dir / f"results_{timestamp}.json"
    
    with open(result_file, 'w') as f:
        json.dump({
            'config': {
                'model_type': model_type,
                'dataset': dataset_name,
                'task': task_name,
                'perturbed': perturbed,
                'original_count': adapter.get_original_count(),
                'sample_num': len(samples),
            },
            'statistics': stats,
            'results': all_results,
        }, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    
    # 生成论文风格的Excel表格
    if perturbed:
        excel_file = output_dir / f"table_{timestamp}.xlsx"
        csv_file = output_dir / f"table_{timestamp}.csv"
        generate_paper_table(stats, task_name, model_type, dataset_name, excel_file, csv_file)
        print(f"\n📊 论文表格已保存: {excel_file}")
        print(f"📊 CSV表格已保存: {csv_file}")
    
    print(f"\n{'='*70}")
    print("✅ 评测完成!")
    print(f"{'='*70}")
    print(f"模型: {model_type}")
    print(f"原始数据量: {adapter.get_original_count():,}")
    print(f"实际使用量: {len(samples):,}")
    print(f"结果数: {len(all_results)}")
    print_statistics(stats, task_name)
    print(f"\n保存: {result_file}")
    print(f"{'='*70}")
    
    return all_results


def compute_statistics(results, task_name, iou_thresholds):
    """计算统计信息"""
    stats = {}
    
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in results:
        key = f"{r['perturbation']}_s{r['severity']}"
        grouped[key].append(r)
    
    for key, group in grouped.items():
        group_stats = {'count': len(group)}
        
        if task_name == 'grounding':
            ious = [r.get('iou', 0) for r in group if r.get('iou') is not None]
            if ious:
                group_stats['mean_iou'] = np.mean(ious)
                group_stats['median_iou'] = np.median(ious)
                for thresh in iou_thresholds:
                    correct = [r.get(f'correct@{thresh}', False) for r in group]
                    group_stats[f'accuracy@{thresh}'] = sum(correct) / len(correct) if correct else 0
        
        elif task_name == 'vqa':
            scores = [r.get('semantic_score') for r in group if r.get('semantic_score') is not None]
            if scores:
                group_stats['mean_score'] = np.mean(scores)
                group_stats['accuracy'] = sum(1 for s in scores if s >= 0.5) / len(scores)
        
        elif task_name == 'caption':
            scores = [r.get('semantic_score') for r in group if r.get('semantic_score') is not None]
            if scores:
                group_stats['mean_score'] = np.mean(scores)
                group_stats['median_score'] = np.median(scores)
        
        stats[key] = group_stats
    
    # 计算所有扰动的整体平均
    perturbed_results = [r for r in results if r['perturbation'] != 'none']
    
    if perturbed_results:
        overall_stats = {'count': len(perturbed_results)}
        
        if task_name == 'grounding':
            ious = [r.get('iou', 0) for r in perturbed_results if r.get('iou') is not None]
            if ious:
                overall_stats['mean_iou'] = np.mean(ious)
                overall_stats['median_iou'] = np.median(ious)
                for thresh in iou_thresholds:
                    correct = [r.get(f'correct@{thresh}', False) for r in perturbed_results]
                    overall_stats[f'accuracy@{thresh}'] = sum(correct) / len(correct) if correct else 0
        
        elif task_name == 'vqa':
            scores = [r.get('semantic_score') for r in perturbed_results if r.get('semantic_score') is not None]
            if scores:
                overall_stats['mean_score'] = np.mean(scores)
                overall_stats['accuracy'] = sum(1 for s in scores if s >= 0.5) / len(scores)
        
        elif task_name == 'caption':
            scores = [r.get('semantic_score') for r in perturbed_results if r.get('semantic_score') is not None]
            if scores:
                overall_stats['mean_score'] = np.mean(scores)
                overall_stats['median_score'] = np.median(scores)
        
        stats['all_perturbations_overall'] = overall_stats
    
    return stats


def print_statistics(stats, task_name):
    """打印统计信息"""
    print(f"\n📊 统计:")
    
    for key, group_stats in sorted(stats.items()):
        if key == 'all_perturbations_overall':
            print(f"\n  ===== 所有扰动总体 =====")
        else:
            print(f"\n  {key}:")
        
        print(f"    样本数: {group_stats.get('count', 0)}")
        
        if task_name == 'grounding':
            print(f"    Mean IoU: {group_stats.get('mean_iou', 0):.4f}")
            print(f"    Acc@0.5: {group_stats.get('accuracy@0.5', 0):.4f}")
            print(f"    Acc@0.75: {group_stats.get('accuracy@0.75', 0):.4f}")
        elif task_name == 'vqa':
            print(f"    Mean Score: {group_stats.get('mean_score', 0):.4f}")
            print(f"    Accuracy: {group_stats.get('accuracy', 0):.4f}")
        elif task_name == 'caption':
            print(f"    Mean Score: {group_stats.get('mean_score', 0):.4f}")


def generate_paper_table(stats, task_name, model_type, dataset_name, excel_file, csv_file):
    """生成论文风格的表格"""
    
    # 确定主要指标
    if task_name == 'grounding':
        metric_key = 'mean_iou'
        metric_name = 'Mean IoU'
        is_percentage = False
    elif task_name == 'vqa':
        metric_key = 'accuracy'
        metric_name = 'Accuracy'
        is_percentage = True
    elif task_name == 'caption':
        metric_key = 'mean_score'
        metric_name = 'Mean Score'
        is_percentage = False
    else:
        metric_key = 'mean_score'
        metric_name = 'Score'
        is_percentage = False
    
    def format_value(val):
        if val is None:
            return '-'
        if is_percentage:
            return f"{val * 100:.1f}"
        return f"{val:.4f}"
    
    # 提取Clean结果
    clean_key = 'none_s0'
    clean_stats = stats.get(clean_key, {})
    clean_metric = clean_stats.get(metric_key, None)
    
    # 提取扰动结果
    perturbation_data = {}
    for key, group_stats in stats.items():
        if key in ['none_s0', 'all_perturbations_overall']:
            continue
        
        parts = key.rsplit('_s', 1)
        if len(parts) == 2:
            pert_type = parts[0]
            try:
                severity = int(parts[1])
            except ValueError:
                continue
            
            if pert_type not in perturbation_data:
                perturbation_data[pert_type] = {}
            
            value = group_stats.get(metric_key, None)
            perturbation_data[pert_type][severity] = value
    
    # 计算每种扰动的平均值
    pert_avg = {}
    for pert_type, severities in perturbation_data.items():
        values = [v for v in severities.values() if v is not None]
        if values:
            pert_avg[pert_type] = np.mean(values)
    
    # 按扰动类型排序
    sorted_perts = sorted(perturbation_data.keys())
    
    # 表格1: 宽表格
    wide_row = {'Model': model_type, 'Clean': format_value(clean_metric)}
    for pert_type in sorted_perts:
        wide_row[pert_type.replace('_', ' ').title()] = format_value(pert_avg.get(pert_type))
    
    if 'all_perturbations_overall' in stats:
        overall = stats['all_perturbations_overall']
        overall_val = overall.get(metric_key, None)
        wide_row['Avg'] = format_value(overall_val)
        if clean_metric is not None and overall_val is not None:
            wide_row['ΔTP'] = format_value(clean_metric - overall_val)
    
    df_wide = pd.DataFrame([wide_row])
    
    # 表格2: 详细表格
    detailed_rows = []
    detailed_rows.append({
        'Model': model_type,
        'Perturbation': 'Clean',
        'Severity': '-',
        metric_name: format_value(clean_metric),
        'Count': clean_stats.get('count', 0)
    })
    
    for pert_type in sorted_perts:
        for severity in sorted(perturbation_data[pert_type].keys()):
            key = f"{pert_type}_s{severity}"
            group_stats = stats.get(key, {})
            detailed_rows.append({
                'Model': model_type,
                'Perturbation': pert_type.replace('_', ' ').title(),
                'Severity': severity,
                metric_name: format_value(group_stats.get(metric_key)),
                'Count': group_stats.get('count', 0)
            })
    
    df_detailed = pd.DataFrame(detailed_rows)
    
    # 表格3: 按严重程度分组
    severity_rows = []
    all_severities = set()
    for severities in perturbation_data.values():
        all_severities.update(severities.keys())
    
    for severity in sorted(all_severities):
        values = []
        for pert_type in sorted_perts:
            if severity in perturbation_data[pert_type]:
                val = perturbation_data[pert_type][severity]
                if val is not None:
                    values.append(val)
        
        if values:
            severity_rows.append({
                'Severity': severity,
                'Mean': format_value(np.mean(values)),
                'Std': round(np.std(values) * (100 if is_percentage else 1), 2),
                'Count': len(values)
            })
    
    df_severity = pd.DataFrame(severity_rows)
    
    # 表格4: 按扰动类型分组
    pert_rows = []
    for pert_type in sorted_perts:
        if pert_type in perturbation_data:
            severities = perturbation_data[pert_type]
            values = list(severities.values())
            pert_row = {
                'Perturbation': pert_type.replace('_', ' ').title(),
                'S1': format_value(severities.get(1)),
                'S2': format_value(severities.get(2)),
                'S3': format_value(severities.get(3)),
                'S4': format_value(severities.get(4)),
                'S5': format_value(severities.get(5)),
                'Avg': format_value(np.mean(values)),
                'Std': round(np.std(values) * (100 if is_percentage else 1), 2) if len(values) > 1 else '-'
            }
            pert_rows.append(pert_row)
    
    df_by_pert = pd.DataFrame(pert_rows)
    
    # LaTeX格式表格
    latex_header = ['Model', 'Clean'] + [p.replace('_', ' ').title() for p in sorted_perts] + ['Avg', 'ΔTP']
    
    latex_data = [model_type, format_value(clean_metric)]
    for pert_type in sorted_perts:
        latex_data.append(format_value(pert_avg.get(pert_type)))
    if 'all_perturbations_overall' in stats:
        overall = stats['all_perturbations_overall']
        overall_val = overall.get(metric_key, overall.get('mean_score', overall.get('mean_iou', 0)))
        latex_data.append(format_value(overall_val))
        if clean_metric is not None:
            latex_data.append(format_value(clean_metric - overall_val))
    
    latex_line = ' & '.join(str(x) for x in latex_data) + ' \\\\'
    
    # 保存Excel
    try:
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            df_wide.to_excel(writer, sheet_name='Summary_Wide', index=False)
            df_detailed.to_excel(writer, sheet_name='Detailed', index=False)
            df_severity.to_excel(writer, sheet_name='By_Severity', index=False)
            df_by_pert.to_excel(writer, sheet_name='By_Perturbation', index=False)
            
            df_latex = pd.DataFrame({
                'LaTeX_Header': [' & '.join(latex_header) + ' \\\\'],
                'LaTeX_Data': [latex_line]
            })
            df_latex.to_excel(writer, sheet_name='LaTeX', index=False)
            
        print(f"  ✓ Excel保存成功: {excel_file}")
    except Exception as e:
        print(f"  ⚠️ Excel保存失败: {e}")
        print(f"    尝试安装: pip install openpyxl --break-system-packages")
    
    # 保存CSV
    try:
        df_wide.to_csv(csv_file, index=False)
        print(f"  ✓ CSV保存成功: {csv_file}")
    except Exception as e:
        print(f"  ⚠️ CSV保存失败: {e}")
    
    # 打印预览
    print(f"\n📋 论文表格预览 (REOBench风格):")
    print(f"   任务: {task_name}, 指标: {metric_name}")
    print(df_wide.to_string(index=False))
    
    print(f"\n📋 LaTeX格式 (可直接复制):")
    print(f"   Header: {' & '.join(latex_header)} \\\\")
    print(f"   Data:   {latex_line}")
    
    return df_wide, df_detailed, df_severity, df_by_pert


# ============================================
# 主程序
# ============================================

def main():
    parser = argparse.ArgumentParser(description="VLM扰动评测/微调 - 支持多种模型")
    
    parser.add_argument("--dataset", type=str, required=True, choices=['omnimedvqa', 'roco', 'mecovqa'])
    parser.add_argument("--task", type=str, default=None, choices=['vqa', 'caption', 'grounding'])
    parser.add_argument("--sample_num", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    
    # 模型选择
    parser.add_argument("--model_type", type=str, default=None, 
                        choices=['llava-med', 'medgemma', 'medgemma15', 'gemini', 'gpt4v'],
                        help="选择模型类型")
    parser.add_argument("--model_path", type=str, default=None, help="LLaVA-Med模型路径")
    
    # MedGemma专用参数
    parser.add_argument("--medgemma_path", type=str, default=None, help="MedGemma模型路径")
    parser.add_argument("--medgemma15_path", type=str, default=None, help="MedGemma 1.5模型路径")
    parser.add_argument("--torch_dtype", type=str, default=None, choices=['bfloat16', 'float16', 'float32'],
                        help="MedGemma使用的数据类型")
    
    # Gemini API参数
    parser.add_argument("--gemini_api_key", type=str, default=None, help="Gemini API Key")
    parser.add_argument("--gemini_model", type=str, default=None, help="Gemini模型名称")
    
    # GPT-4V API参数
    parser.add_argument("--openai_api_key", type=str, default=None, help="OpenAI API Key")
    parser.add_argument("--openai_base_url", type=str, default=None, help="OpenAI API Base URL")
    parser.add_argument("--gpt4v_model", type=str, default=None, help="GPT-4V模型名称")
    
    parser.add_argument("--perturbed", action="store_true", help="评测扰动图像")
    parser.add_argument("--pert_type", type=str, default=None, help="扰动类型")
    parser.add_argument("--severity", type=int, default=None, help="扰动程度")
    parser.add_argument("--all_perturbations", action="store_true", help="评测所有扰动")
    
    # 微调参数 (支持所有开源模型)
    parser.add_argument("--finetune", action="store_true", help="启用微调模式 (支持llava-med, medgemma, medgemma15)")
    parser.add_argument("--lora_r", type=int, default=None, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=None, help="LoRA alpha")
    parser.add_argument("--epochs", type=int, default=None, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=None, help="批大小")
    parser.add_argument("--learning_rate", type=float, default=None, help="学习率")
    parser.add_argument("--finetune_output", type=str, default=None, help="微调输出目录")
    
    parser.add_argument("--deepseek_api_key", type=str, default=None, help="DeepSeek API Key")
    parser.add_argument("--deepseek_base_url", type=str, default=None, help="DeepSeek API Base URL")
    parser.add_argument("--use_deepseek", action="store_true", help="启用DeepSeek语义评估")
    parser.add_argument("--no_deepseek", action="store_true", help="禁用DeepSeek语义评估")
    
    args = parser.parse_args()
    
    config = json.loads(json.dumps(CONFIG))
    
    if args.sample_num:
        config['sample_num'] = args.sample_num
    if args.seed:
        config['seed'] = args.seed
    
    # 模型类型设置
    if args.model_type:
        config['model']['type'] = args.model_type
    
    # LLaVA-Med路径设置
    if args.model_path:
        config['model']['path'] = args.model_path
    
    # MedGemma设置
    if args.medgemma_path:
        config['medgemma']['path'] = args.medgemma_path
    if args.medgemma15_path:
        config['medgemma15']['path'] = args.medgemma15_path
    if args.torch_dtype:
        config['medgemma']['torch_dtype'] = args.torch_dtype
        config['medgemma15']['torch_dtype'] = args.torch_dtype
    
    # Gemini设置
    if args.gemini_api_key:
        config['gemini']['api_key'] = args.gemini_api_key
    if args.gemini_model:
        config['gemini']['model'] = args.gemini_model
    
    # GPT-4V设置
    if args.openai_api_key:
        config['gpt4v']['api_key'] = args.openai_api_key
    if args.openai_base_url:
        config['gpt4v']['base_url'] = args.openai_base_url
    if args.gpt4v_model:
        config['gpt4v']['model'] = args.gpt4v_model
    
    # 微调参数
    if args.lora_r is not None:
        config['finetune']['lora_r'] = args.lora_r
    if args.lora_alpha is not None:
        config['finetune']['lora_alpha'] = args.lora_alpha
    if args.epochs is not None:
        config['finetune']['epochs'] = args.epochs
    if args.batch_size is not None:
        config['finetune']['batch_size'] = args.batch_size
    if args.learning_rate is not None:
        config['finetune']['learning_rate'] = args.learning_rate
    if args.finetune_output is not None:
        config['finetune']['output_dir'] = args.finetune_output
    
    # DeepSeek设置
    if args.deepseek_api_key:
        config['deepseek']['api_key'] = args.deepseek_api_key
    if args.deepseek_base_url:
        config['deepseek']['base_url'] = args.deepseek_base_url
    if args.use_deepseek:
        config['evaluation']['use_deepseek'] = True
    if args.no_deepseek:
        config['evaluation']['use_deepseek'] = False
    
    task_name = args.task
    if task_name is None:
        task_name = config['datasets'][args.dataset].get('default_task', 'vqa')
    
    # 确定模型类型
    model_type = args.model_type or config['model']['type']
    
    if args.finetune:
        # 微调模式 - 支持所有开源模型
        if model_type not in ['llava-med', 'medgemma', 'medgemma15']:
            print(f"⚠️ 微调功能仅支持开源模型: llava-med, medgemma, medgemma15")
            print(f"   当前选择: {model_type}")
            return
        run_finetune(args.dataset, task_name, config, model_type)
    else:
        # 评测模式
        if model_type in ['gemini', 'gpt4v']:
            if not check_api_available(config, model_type):
                print(f"❌ {model_type} API Key未设置，跳过评测")
                print(f"   请通过 --{'gemini_api_key' if model_type == 'gemini' else 'openai_api_key'} 参数设置API Key")
                return
        
        run_evaluation(
            dataset_name=args.dataset,
            task_name=task_name,
            config=config,
            model_type=model_type,
            perturbed=args.perturbed,
            pert_type=args.pert_type,
            severity=args.severity,
            all_perturbations=args.all_perturbations,
        )


if __name__ == "__main__":
    main()
