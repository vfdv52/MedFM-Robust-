#!/usr/bin/env python3
"""
CSV结果提取工具
================
从 results 文件夹中提取所有CSV文件，并保留完整的上下文信息：
- 数据集名称 (dataset)
- 模型名称 (model)
- 微调策略 (strategy)
- 评估类型 (adversarial/perturbation)

输出结构:
extracted_csv/
├── index.csv                   # 索引文件，记录所有CSV的元信息
├── medsam/
│   ├── isic_2016/
│   │   ├── decoder_only/
│   │   │   ├── results_adversarial_xxx.csv
│   │   │   └── results_adversarial_xxx_SUMMARY.csv
│   │   └── lora/
│   │       └── ...
│   └── brain-tumor/
│       └── ...
└── sammed2d/
    └── ...

使用方法:
    python extract_csv_results.py --results_dir ./results --output_dir ./extracted_csv
    python extract_csv_results.py --results_dir ./results --output_dir ./extracted_csv --flat  # 扁平化输出
"""

# # 基本用法 - 保持层级结构
# python extract_csv_results.py --results_dir ./results --output_dir ./extracted_csv

# # 只提取汇总文件 (最实用 - 文件小)
# python extract_csv_results.py --results_dir ./results --output_dir ./extracted_csv --summary_only

# # 扁平化输出 (所有文件放一个目录，文件名包含所有信息)
# python extract_csv_results.py --results_dir ./results --output_dir ./extracted_csv --flat --summary_only

# # 同时生成合并对比表
# python extract_csv_results.py --results_dir ./results --output_dir ./extracted_csv --summary_only --merge_summary
# ```

# **输出结构:**
# ```
# extracted_csv/
# ├── index.csv              # 索引文件，记录所有CSV的元信息
# ├── index.json             # JSON格式索引
# ├── merged_summary_adversarial.csv   # (--merge_summary时生成)
# ├── merged_summary_perturbation.csv  # (--merge_summary时生成)
# └── medsam/
#     └── isic_2016/
#         └── decoder_only/
#             └── results_adversarial_xxx_SUMMARY.csv

import os
import re
import shutil
import argparse
import json
from pathlib import Path
from datetime import datetime
import csv


def parse_experiment_path(csv_path: str) -> dict:
    """
    从CSV路径解析实验元信息
    
    路径模式示例:
    ./results/20250120_143000_decoder_only/pipeline_medsam_isic_2016/results/results_adversarial_xxx.csv
    ./results/20250120_143000_pretrained/pipeline_sammed2d_brain-tumor/results/results_perturbation_xxx.csv
    """
    path_parts = Path(csv_path).parts
    info = {
        'csv_path': csv_path,
        'csv_name': os.path.basename(csv_path),
        'strategy': 'unknown',
        'model': 'unknown',
        'dataset': 'unknown',
        'eval_type': 'unknown',
        'timestamp': 'unknown'
    }
    
    # 1. 从主目录名提取 timestamp 和 strategy
    # 格式: {timestamp}_{strategy} 或 {timestamp}_pretrained
    for part in path_parts:
        # 匹配 20250120_143000_decoder_only 这样的格式
        match = re.match(r'^(\d{8}_\d{6})_(.+)$', part)
        if match:
            info['timestamp'] = match.group(1)
            info['strategy'] = match.group(2)
            break
    
    # 2. 从 pipeline_* 目录名提取 model 和 dataset
    # 格式: pipeline_{model}_{dataset}
    for part in path_parts:
        match = re.match(r'^pipeline_([^_]+)_(.+)$', part)
        if match:
            info['model'] = match.group(1)
            info['dataset'] = match.group(2)
            break
    
    # 3. 从CSV文件名提取 eval_type
    # 格式: results_adversarial_xxx.csv 或 results_perturbation_xxx.csv
    csv_name = info['csv_name']
    if 'adversarial' in csv_name.lower():
        info['eval_type'] = 'adversarial'
    elif 'perturbation' in csv_name.lower():
        info['eval_type'] = 'perturbation'
    
    # 4. 判断是否为汇总文件
    info['is_summary'] = 'SUMMARY' in csv_name or 'STATS' in csv_name
    
    return info


def find_all_csv_files(results_dir: str) -> list:
    """递归查找所有CSV文件"""
    csv_files = []
    for root, dirs, files in os.walk(results_dir):
        for file in files:
            if file.endswith('.csv'):
                full_path = os.path.join(root, file)
                csv_files.append(full_path)
    return csv_files


def generate_output_filename(info: dict, flat: bool = False) -> str:
    """
    生成输出文件名
    
    层级模式: {model}/{dataset}/{strategy}/{原始文件名}
    扁平模式: {model}__{dataset}__{strategy}__{eval_type}__{原始文件名}
    """
    if flat:
        # 扁平化命名，便于快速查看
        parts = [
            info['model'],
            info['dataset'],
            info['strategy'],
            info['eval_type']
        ]
        base_name = info['csv_name']
        return f"{'__'.join(parts)}__{base_name}"
    else:
        # 层级结构
        return os.path.join(
            info['model'],
            info['dataset'],
            info['strategy'],
            info['csv_name']
        )


def extract_csv_files(results_dir: str, output_dir: str, flat: bool = False, 
                       summary_only: bool = False, detail_only: bool = False):
    """
    主提取函数
    
    Args:
        results_dir: 结果目录 (./results)
        output_dir: 输出目录
        flat: 是否使用扁平化命名
        summary_only: 只提取汇总文件 (*_SUMMARY.csv, *_STATS*.csv)
        detail_only: 只提取详细结果文件 (非汇总文件)
    """
    print(f"📁 扫描目录: {results_dir}")
    csv_files = find_all_csv_files(results_dir)
    print(f"✅ 找到 {len(csv_files)} 个CSV文件")
    
    if not csv_files:
        print("⚠️ 未找到任何CSV文件")
        return
    
    # 解析所有CSV文件的元信息
    all_info = []
    for csv_path in csv_files:
        info = parse_experiment_path(csv_path)
        all_info.append(info)
    
    # 过滤
    if summary_only:
        all_info = [info for info in all_info if info['is_summary']]
        print(f"📊 筛选汇总文件: {len(all_info)} 个")
    elif detail_only:
        all_info = [info for info in all_info if not info['is_summary']]
        print(f"📋 筛选详细文件: {len(all_info)} 个")
    
    if not all_info:
        print("⚠️ 筛选后无文件可提取")
        return
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 复制文件
    copied_count = 0
    index_data = []
    
    for info in all_info:
        src_path = info['csv_path']
        rel_output = generate_output_filename(info, flat)
        dst_path = os.path.join(output_dir, rel_output)
        
        # 创建目标目录
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        
        # 复制文件
        try:
            shutil.copy2(src_path, dst_path)
            copied_count += 1
            
            # 记录索引
            index_data.append({
                'model': info['model'],
                'dataset': info['dataset'],
                'strategy': info['strategy'],
                'eval_type': info['eval_type'],
                'is_summary': info['is_summary'],
                'timestamp': info['timestamp'],
                'output_path': rel_output,
                'original_path': src_path
            })
            
            print(f"✓ [{info['model']}][{info['dataset']}][{info['strategy']}] {info['csv_name']}")
            
        except Exception as e:
            print(f"✗ 复制失败: {src_path} -> {e}")
    
    # 保存索引文件
    index_path = os.path.join(output_dir, 'index.csv')
    with open(index_path, 'w', newline='', encoding='utf-8') as f:
        if index_data:
            writer = csv.DictWriter(f, fieldnames=index_data[0].keys())
            writer.writeheader()
            writer.writerows(index_data)
    
    # 同时保存JSON格式的索引
    index_json_path = os.path.join(output_dir, 'index.json')
    with open(index_json_path, 'w', encoding='utf-8') as f:
        json.dump(index_data, f, indent=2, ensure_ascii=False)
    
    # 生成汇总统计
    print(f"\n{'='*60}")
    print(f"📊 提取完成!")
    print(f"{'='*60}")
    print(f"   总计复制: {copied_count} 个文件")
    print(f"   输出目录: {output_dir}")
    print(f"   索引文件: {index_path}")
    
    # 按维度统计
    models = set(info['model'] for info in index_data)
    datasets = set(info['dataset'] for info in index_data)
    strategies = set(info['strategy'] for info in index_data)
    
    print(f"\n📈 维度统计:")
    print(f"   模型: {len(models)} 个 - {', '.join(sorted(models))}")
    print(f"   数据集: {len(datasets)} 个 - {', '.join(sorted(datasets))}")
    print(f"   策略: {len(strategies)} 个 - {', '.join(sorted(strategies))}")
    print(f"{'='*60}\n")
    
    return index_data


def create_merged_summary(output_dir: str, index_data: list):
    """
    创建合并的汇总表，方便快速对比不同策略的性能
    
    输出:
    - merged_summary_adversarial.csv: 所有对抗攻击的汇总数据
    - merged_summary_perturbation.csv: 所有扰动评估的汇总数据
    """
    import pandas as pd
    
    print(f"\n📊 正在生成合并汇总表...")
    
    # 只处理汇总文件
    summary_files = [info for info in index_data if info['is_summary'] and 'SUMMARY' in info['output_path']]
    
    if not summary_files:
        print("⚠️ 未找到汇总文件，跳过合并")
        return
    
    # 分别处理 adversarial 和 perturbation
    for eval_type in ['adversarial', 'perturbation']:
        type_files = [info for info in summary_files if info['eval_type'] == eval_type]
        
        if not type_files:
            continue
        
        merged_rows = []
        for info in type_files:
            csv_path = os.path.join(output_dir, info['output_path'])
            try:
                df = pd.read_csv(csv_path)
                # 添加元信息列
                df['Model'] = info['model']
                df['Dataset'] = info['dataset']
                df['Strategy'] = info['strategy']
                df['Timestamp'] = info['timestamp']
                merged_rows.append(df)
            except Exception as e:
                print(f"⚠️ 读取失败: {csv_path} - {e}")
        
        if merged_rows:
            merged_df = pd.concat(merged_rows, ignore_index=True)
            # 重排列顺序，元信息放前面
            cols = ['Model', 'Dataset', 'Strategy', 'Timestamp'] + \
                   [c for c in merged_df.columns if c not in ['Model', 'Dataset', 'Strategy', 'Timestamp']]
            merged_df = merged_df[cols]
            
            output_path = os.path.join(output_dir, f'merged_summary_{eval_type}.csv')
            merged_df.to_csv(output_path, index=False, float_format='%.4f')
            print(f"✅ 合并汇总表: {output_path} ({len(merged_df)} 行)")


def main():
    parser = argparse.ArgumentParser(
        description="从results目录提取CSV文件，保留数据集/模型/策略信息",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
---------
# 基本用法 - 层级目录结构
python extract_csv_results.py --results_dir ./results --output_dir ./extracted_csv

# 扁平化输出 - 所有文件放在一个目录下，文件名包含所有信息
python extract_csv_results.py --results_dir ./results --output_dir ./extracted_csv --flat

# 只提取汇总文件 (SUMMARY.csv)
python extract_csv_results.py --results_dir ./results --output_dir ./extracted_csv --summary_only

# 只提取详细结果文件 (不含SUMMARY)
python extract_csv_results.py --results_dir ./results --output_dir ./extracted_csv --detail_only

# 生成合并汇总表
python extract_csv_results.py --results_dir ./results --output_dir ./extracted_csv --merge_summary
        """
    )
    
    parser.add_argument('--results_dir', type=str, default='./results',
                        help='结果目录路径 (默认: ./results)')
    parser.add_argument('--output_dir', type=str, default='./extracted_csv',
                        help='输出目录路径 (默认: ./extracted_csv)')
    parser.add_argument('--flat', action='store_true',
                        help='使用扁平化命名 (所有文件放同一目录)')
    parser.add_argument('--summary_only', action='store_true',
                        help='只提取汇总文件 (*_SUMMARY.csv, *_STATS*.csv)')
    parser.add_argument('--detail_only', action='store_true',
                        help='只提取详细结果文件 (非汇总文件)')
    parser.add_argument('--merge_summary', action='store_true',
                        help='生成合并的汇总表 (需要pandas)')
    
    args = parser.parse_args()
    
    # 验证参数
    if args.summary_only and args.detail_only:
        print("❌ 错误: --summary_only 和 --detail_only 不能同时使用")
        return
    
    if not os.path.exists(args.results_dir):
        print(f"❌ 错误: 结果目录不存在: {args.results_dir}")
        return
    
    # 执行提取
    index_data = extract_csv_files(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        flat=args.flat,
        summary_only=args.summary_only,
        detail_only=args.detail_only
    )
    
    # 可选: 生成合并汇总表
    if args.merge_summary and index_data:
        try:
            create_merged_summary(args.output_dir, index_data)
        except ImportError:
            print("⚠️ 合并汇总需要 pandas 库，请安装: pip install pandas")


if __name__ == "__main__":
    main()
