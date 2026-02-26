#!/usr/bin/env python3
"""
合并多个模型的评测结果，生成完整的论文风格表格

参考REOBench论文表格格式:
- Table 1: Scene classification
- Table 2: Semantic segmentation
- Table 3: Object detection
- Table 4: Image captioning
- Table 5: VQA
- Table 6: Visual grounding

使用方法:
    python merge_results_to_table.py --input_dir ./outputs --task vqa --output paper_table.xlsx

    或者指定多个JSON文件:
    python merge_results_to_table.py --files results1.json results2.json --output paper_table.xlsx
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime


def load_results(json_file):
    """加载单个结果文件"""
    with open(json_file, 'r') as f:
        return json.load(f)


def extract_metrics_from_stats(stats, task_name):
    """从统计信息中提取指标"""
    if task_name == 'grounding':
        metric_key = 'accuracy@0.5'
        is_percentage = True
    elif task_name == 'vqa':
        metric_key = 'accuracy'
        is_percentage = True
    elif task_name == 'caption':
        metric_key = 'mean_score'
        is_percentage = False
    else:
        metric_key = 'accuracy'
        is_percentage = True

    return metric_key, is_percentage


def process_single_result(result_data):
    """处理单个结果文件，提取关键指标"""
    config = result_data.get('config', {})
    stats = result_data.get('statistics', {})

    model_type = config.get('model_type', 'unknown')
    dataset = config.get('dataset', 'unknown')
    task = config.get('task', 'unknown')

    metric_key, is_percentage = extract_metrics_from_stats(stats, task)

    # 提取各扰动类型的数据
    perturbation_data = {}
    clean_metric = None
    overall_metric = None

    for key, s in stats.items():
        if key == 'all_perturbations_overall':
            overall_metric = s.get(metric_key, s.get('mean_score', s.get('mean_iou', 0)))
            continue

        if key == 'none_s0':
            clean_metric = s.get(metric_key, s.get('mean_score', s.get('mean_iou', 0)))
        else:
            # 跳过非扰动统计键
            if key in ['caption_standard_metrics'] or not key.endswith(tuple(f'_s{i}' for i in range(10))):
            	continue
            # 解析扰动类型和严重程度
            parts = key.split('_s')
            if len(parts) == 2:
                perturb_type = parts[0]
                severity = int(parts[1])

                if perturb_type not in perturbation_data:
                    perturbation_data[perturb_type] = {}

                perturbation_data[perturb_type][severity] = s.get(metric_key, s.get('mean_score', s.get('mean_iou', 0)))

    # 计算每种扰动类型的平均值
    perturb_avg = {}
    for perturb_type, severity_data in perturbation_data.items():
        if severity_data:
            perturb_avg[perturb_type] = np.mean(list(severity_data.values()))

    return {
        'model_type': model_type,
        'dataset': dataset,
        'task': task,
        'clean': clean_metric,
        'overall_avg': overall_metric,
        'perturbation_avg': perturb_avg,
        'perturbation_detail': perturbation_data,
        'is_percentage': is_percentage
    }


def merge_same_setting(results_list):
    """\
    将同一个 (model_type, dataset, task) 的多份结果合并成一条完整记录。

    说明：
    - 你的 pipeline 可能会对同一组配置分别跑 clean 和 perturbed 两次评测，
      导致生成两份 results_*.json（分别只有 Clean 或只有扰动列）。
    - 该函数会把它们合并，避免最终 CSV/Excel 中大量 '-' 的占位。

    合并策略（尽量保持原有核心功能不变）：
    - clean / overall_avg：取非 None 的值（优先已存在的）
    - perturbation_detail：按扰动类型、强度做字典深合并
    - perturbation_avg：基于合并后的 detail 重新计算均值
    - is_percentage：必须一致，否则报错（避免不同任务指标混淆）
    """
    if not results_list:
        return results_list

    merged = {}

    for r in results_list:
        key = (r.get('model_type'), r.get('dataset'), r.get('task'))

        if key not in merged:
            merged[key] = {
                'model_type': r.get('model_type'),
                'dataset': r.get('dataset'),
                'task': r.get('task'),
                'clean': None,
                'overall_avg': None,
                'perturbation_avg': {},
                'perturbation_detail': {},
                'is_percentage': r.get('is_percentage', True),
            }

        m = merged[key]

        # 指标类型必须一致
        if m.get('is_percentage') != r.get('is_percentage', True):
            raise ValueError(
                f"is_percentage mismatch for {key}: {m.get('is_percentage')} vs {r.get('is_percentage')}"
            )

        # clean / overall 取非 None
        if m.get('clean') is None and r.get('clean') is not None:
            m['clean'] = r.get('clean')

        if m.get('overall_avg') is None and r.get('overall_avg') is not None:
            m['overall_avg'] = r.get('overall_avg')

        # 深合并扰动明细
        detail = r.get('perturbation_detail') or {}
        for pt, sev_map in detail.items():
            if pt not in m['perturbation_detail']:
                m['perturbation_detail'][pt] = {}
            if isinstance(sev_map, dict):
                m['perturbation_detail'][pt].update(sev_map)

    # 基于 detail 重算 perturbation_avg
    for m in merged.values():
        pert_avg = {}
        for pt, sev_map in (m.get('perturbation_detail') or {}).items():
            if isinstance(sev_map, dict) and len(sev_map) > 0:
                pert_avg[pt] = float(np.mean(list(sev_map.values())))
        m['perturbation_avg'] = pert_avg

    return list(merged.values())


def format_value(val, is_pct=True, decimals=2):
    """格式化数值"""
    if val is None:
        return '-'
    if is_pct:
        return round(val * 100, decimals)
    return round(val, decimals + 2)


def generate_merged_table(results_list, output_excel, output_csv=None):
    """
    合并多个模型结果，生成论文格式表格

    Args:
        results_list: 处理后的结果列表
        output_excel: 输出Excel文件路径
        output_csv: 输出CSV文件路径（可选）
    """
    if not results_list:
        print("❌ 没有结果可以合并")
        return

    # 获取所有扰动类型
    all_perturb_types = set()
    for r in results_list:
        all_perturb_types.update(r['perturbation_avg'].keys())

    all_perturb_types = sorted(list(all_perturb_types))

    # 构建表格数据
    table_data = []

    for r in results_list:
        row = {
            'Model': r['model_type'],
            'Dataset': r['dataset'],
            'Task': r['task'],
            'Clean': format_value(r['clean'], r['is_percentage']),
        }

        # 添加各扰动类型的平均值列
        for perturb_type in all_perturb_types:
            val = r['perturbation_avg'].get(perturb_type)
            row[perturb_type] = format_value(val, r['is_percentage'])

        # 添加overall average
        row['Avg'] = format_value(r['overall_avg'], r['is_percentage'])

        # 计算delta (clean - avg)
        if r['clean'] is not None and r['overall_avg'] is not None:
            delta = r['clean'] - r['overall_avg']
            row['ΔTP'] = format_value(delta, r['is_percentage'])
        else:
            row['ΔTP'] = '-'

        table_data.append(row)

    # 转为DataFrame
    df = pd.DataFrame(table_data)

    # 排序
    df = df.sort_values(['Task', 'Dataset', 'Model'])

    # 保存Excel
    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Results', index=False)

    print(f"✅ 已保存Excel表格: {output_excel}")

    # 保存CSV（可选）
    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"✅ 已保存CSV表格: {output_csv}")


def find_result_files(input_dir, task_filter=None, dataset_filter=None):
    """在目录中查找结果文件"""
    input_dir = Path(input_dir)
    if not input_dir.exists():
        return []

    result_files = list(input_dir.rglob("results_*.json"))

    # 过滤任务和数据集
    filtered_files = []
    for f in result_files:
        try:
            data = load_results(f)
            config = data.get('config', {})

            if task_filter and config.get('task') != task_filter:
                continue

            if dataset_filter and config.get('dataset') != dataset_filter:
                continue

            filtered_files.append(str(f))
        except:
            continue

    return filtered_files


def main():
    parser = argparse.ArgumentParser(description="合并多个模型评测结果生成表格")
    parser.add_argument("--input_dir", type=str, default=None,
                        help="包含结果JSON文件的目录")
    parser.add_argument("--files", nargs='+', type=str, default=None,
                        help="直接指定结果JSON文件列表")
    parser.add_argument("--task", type=str, default=None,
                        choices=['vqa', 'caption', 'grounding'],
                        help="过滤特定任务")
    parser.add_argument("--dataset", type=str, default=None,
                        help="过滤特定数据集")
    parser.add_argument("--output", type=str, default=None,
                        help="输出Excel文件名")
    parser.add_argument("--output_csv", type=str, default=None,
                        help="输出CSV文件名")

    args = parser.parse_args()

    # 收集结果文件
    result_files = []

    if args.files:
        result_files = args.files
    elif args.input_dir:
        result_files = find_result_files(args.input_dir, args.task, args.dataset)
    else:
        print("❌ 请指定 --input_dir 或 --files")
        return

    if not result_files:
        print("❌ 未找到任何结果文件")
        return

    print(f"📂 找到 {len(result_files)} 个结果文件:")
    for f in result_files:
        print(f"   - {f}")

    # 加载并处理结果
    results_list = []
    for filepath in result_files:
        try:
            data = load_results(filepath)
            processed = process_single_result(data)
            results_list.append(processed)
            print(f"   ✓ 已加载: {processed['model_type']} / {processed['dataset']} / {processed['task']}")
        except Exception as e:
            print(f"   ⚠️ 加载失败 {filepath}: {e}")

    if not results_list:
        print("❌ 没有有效的结果")
        return

    # 合并同一组 (model/dataset/task) 的多份结果（例如 clean-run 与 perturbed-run）
    results_list = merge_same_setting(results_list)

    # 生成输出文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.output:
        output_excel = args.output
    else:
        task_str = args.task or 'all'
        output_excel = f"paper_table_{task_str}_{timestamp}.xlsx"

    output_csv = args.output_csv or output_excel.replace('.xlsx', '.csv')

    # 生成表格
    generate_merged_table(results_list, output_excel, output_csv)


if __name__ == "__main__":
    main()

# #!/usr/bin/env python3
# """
# 合并多个模型的评测结果，生成完整的论文风格表格

# 参考REOBench论文表格格式:
# - Table 1: Scene classification
# - Table 2: Semantic segmentation  
# - Table 3: Object detection
# - Table 4: Image captioning
# - Table 5: VQA
# - Table 6: Visual grounding

# 使用方法:
#     python merge_results_to_table.py --input_dir ./outputs --task vqa --output paper_table.xlsx
    
#     或者指定多个JSON文件:
#     python merge_results_to_table.py --files results1.json results2.json --output paper_table.xlsx
# """

# import os
# import json
# import argparse
# import numpy as np
# import pandas as pd
# from pathlib import Path
# from datetime import datetime


# def load_results(json_file):
#     """加载单个结果文件"""
#     with open(json_file, 'r') as f:
#         return json.load(f)


# def extract_metrics_from_stats(stats, task_name):
#     """从统计信息中提取指标"""
#     if task_name == 'grounding':
#         metric_key = 'accuracy@0.5'
#         is_percentage = True
#     elif task_name == 'vqa':
#         metric_key = 'accuracy'
#         is_percentage = True
#     elif task_name == 'caption':
#         metric_key = 'mean_score'
#         is_percentage = False
#     else:
#         metric_key = 'accuracy'
#         is_percentage = True
    
#     return metric_key, is_percentage


# def process_single_result(result_data):
#     """处理单个结果文件，提取关键指标"""
#     config = result_data.get('config', {})
#     stats = result_data.get('statistics', {})
    
#     model_type = config.get('model_type', 'unknown')
#     dataset = config.get('dataset', 'unknown')
#     task = config.get('task', 'unknown')
    
#     metric_key, is_percentage = extract_metrics_from_stats(stats, task)
    
#     # 提取各扰动类型的数据
#     perturbation_data = {}
#     clean_metric = None
#     overall_metric = None
    
#     for key, s in stats.items():
#         if key == 'all_perturbations_overall':
#             overall_metric = s.get(metric_key, s.get('mean_score', s.get('mean_iou', 0)))
#             continue
        
#         if key == 'none_s0':
#             clean_metric = s.get(metric_key, s.get('mean_score', s.get('mean_iou', 0)))
#         else:
#             parts = key.rsplit('_s', 1)
#             if len(parts) == 2:
#                 pert_type = parts[0]
#                 severity = int(parts[1])
                
#                 if pert_type not in perturbation_data:
#                     perturbation_data[pert_type] = {}
                
#                 metric_value = s.get(metric_key, s.get('mean_score', s.get('mean_iou', 0)))
#                 perturbation_data[pert_type][severity] = metric_value
    
#     # 计算每种扰动的平均值
#     pert_avg = {}
#     for pert_type, severities in perturbation_data.items():
#         pert_avg[pert_type] = np.mean(list(severities.values()))
    
#     return {
#         'model_type': model_type,
#         'dataset': dataset,
#         'task': task,
#         'clean': clean_metric,
#         'overall_avg': overall_metric,
#         'perturbation_avg': pert_avg,
#         'perturbation_detail': perturbation_data,
#         'is_percentage': is_percentage
#     }


# def format_value(val, is_pct=True, decimals=2):
#     """格式化数值"""
#     if val is None:
#         return '-'
#     if is_pct:
#         return round(val * 100, decimals)
#     return round(val, decimals + 2)


# def generate_merged_table(results_list, output_excel, output_csv=None):
#     """
#     合并多个模型结果，生成论文格式表格
    
#     Args:
#         results_list: 处理后的结果列表
#         output_excel: 输出Excel文件路径
#         output_csv: 输出CSV文件路径（可选）
#     """
#     if not results_list:
#         print("❌ 没有结果可以合并")
#         return
    
#     # 收集所有扰动类型
#     all_perturbations = set()
#     for r in results_list:
#         all_perturbations.update(r['perturbation_avg'].keys())
    
#     # 定义标准列顺序 (参考REOBench)
#     standard_order = [
#         'brightness', 'cloud', 'compression', 'gap', 'gaussian_blur', 
#         'gaussian_noise', 'haze', 'motion_blur', 'salt_pepper', 
#         'rotate', 'scale', 'translate'
#     ]
    
#     sorted_perts = []
#     for p in standard_order:
#         if p in all_perturbations:
#             sorted_perts.append(p)
#     for p in sorted(all_perturbations):
#         if p not in sorted_perts:
#             sorted_perts.append(p)
    
#     # ============================================
#     # 生成主表格 (REOBench风格)
#     # ============================================
#     rows = []
#     for r in results_list:
#         is_pct = r['is_percentage']
        
#         row = {
#             'Model': r['model_type'],
#             'Backbone': 'ViT-L',  # 可根据实际修改
#             'Clean': format_value(r['clean'], is_pct)
#         }
        
#         for pert_type in sorted_perts:
#             col_name = pert_type.replace('_', ' ').title()
#             if pert_type in r['perturbation_avg']:
#                 row[col_name] = format_value(r['perturbation_avg'][pert_type], is_pct)
#             else:
#                 row[col_name] = '-'
        
#         row['Avg'] = format_value(r['overall_avg'], is_pct)
        
#         if r['clean'] is not None and r['overall_avg'] is not None:
#             delta_tp = r['clean'] - r['overall_avg']
#             row['ΔTP'] = format_value(delta_tp, is_pct)
#         else:
#             row['ΔTP'] = '-'
        
#         rows.append(row)
    
#     df_main = pd.DataFrame(rows)
    
#     # ============================================
#     # 生成详细表格 (每个模型每个扰动每个severity)
#     # ============================================
#     detail_rows = []
#     for r in results_list:
#         is_pct = r['is_percentage']
        
#         # Clean
#         detail_rows.append({
#             'Model': r['model_type'],
#             'Perturbation': 'Clean',
#             'Severity': '-',
#             'Value': format_value(r['clean'], is_pct)
#         })
        
#         # 各扰动
#         for pert_type in sorted_perts:
#             if pert_type in r['perturbation_detail']:
#                 for sev, val in sorted(r['perturbation_detail'][pert_type].items()):
#                     detail_rows.append({
#                         'Model': r['model_type'],
#                         'Perturbation': pert_type.replace('_', ' ').title(),
#                         'Severity': sev,
#                         'Value': format_value(val, is_pct)
#                     })
        
#         # Overall
#         detail_rows.append({
#             'Model': r['model_type'],
#             'Perturbation': 'All Perturbations Avg',
#             'Severity': '-',
#             'Value': format_value(r['overall_avg'], is_pct)
#         })
    
#     df_detail = pd.DataFrame(detail_rows)
    
#     # ============================================
#     # 生成LaTeX表格代码
#     # ============================================
#     latex_header = ['Model', 'Backbone', 'Clean'] + \
#                    [p.replace('_', ' ').title() for p in sorted_perts] + \
#                    ['Avg', 'ΔTP']
    
#     latex_lines = []
#     latex_lines.append('% LaTeX Table Code (copy to your paper)')
#     latex_lines.append('\\begin{table}[h]')
#     latex_lines.append('\\centering')
#     latex_lines.append('\\caption{Performance under different image perturbations}')
#     latex_lines.append('\\resizebox{\\textwidth}{!}{')
#     latex_lines.append('\\begin{tabular}{' + 'l' * len(latex_header) + '}')
#     latex_lines.append('\\toprule')
#     latex_lines.append(' & '.join(latex_header) + ' \\\\')
#     latex_lines.append('\\midrule')
    
#     for _, row in df_main.iterrows():
#         values = [str(row.get(col, '-')) for col in latex_header]
#         latex_lines.append(' & '.join(values) + ' \\\\')
    
#     latex_lines.append('\\bottomrule')
#     latex_lines.append('\\end{tabular}}')
#     latex_lines.append('\\end{table}')
    
#     latex_code = '\n'.join(latex_lines)
    
#     # ============================================
#     # 保存文件
#     # ============================================
#     try:
#         with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
#             df_main.to_excel(writer, sheet_name='Main_Table', index=False)
#             df_detail.to_excel(writer, sheet_name='Detailed', index=False)
            
#             # LaTeX代码
#             df_latex = pd.DataFrame({'LaTeX_Code': [latex_code]})
#             df_latex.to_excel(writer, sheet_name='LaTeX', index=False)
        
#         print(f"✅ Excel表格已保存: {output_excel}")
#     except Exception as e:
#         print(f"❌ Excel保存失败: {e}")
    
#     if output_csv:
#         try:
#             df_main.to_csv(output_csv, index=False)
#             print(f"✅ CSV表格已保存: {output_csv}")
#         except Exception as e:
#             print(f"❌ CSV保存失败: {e}")
    
#     # 打印预览
#     print(f"\n{'='*80}")
#     print("📋 论文表格预览 (REOBench风格)")
#     print(f"{'='*80}")
#     print(df_main.to_string(index=False))
#     print(f"\n{'='*80}")
#     print("📋 LaTeX代码预览")
#     print(f"{'='*80}")
#     print(latex_code)
    
#     return df_main, df_detail


# def find_result_files(input_dir, task=None, dataset=None):
#     """在目录中查找结果文件"""
#     result_files = []
    
#     for root, dirs, files in os.walk(input_dir):
#         for f in files:
#             if f.startswith('results_') and f.endswith('.json'):
#                 filepath = os.path.join(root, f)
                
#                 # 过滤
#                 if task or dataset:
#                     try:
#                         with open(filepath, 'r') as fp:
#                             data = json.load(fp)
#                             config = data.get('config', {})
                            
#                             if task and config.get('task') != task:
#                                 continue
#                             if dataset and config.get('dataset') != dataset:
#                                 continue
#                     except:
#                         continue
                
#                 result_files.append(filepath)
    
#     return result_files


# def main():
#     parser = argparse.ArgumentParser(description="合并多个模型结果生成论文表格")
    
#     parser.add_argument("--input_dir", type=str, default=None,
#                         help="包含结果文件的目录")
#     parser.add_argument("--files", nargs='+', type=str, default=None,
#                         help="直接指定结果JSON文件列表")
#     parser.add_argument("--task", type=str, default=None,
#                         choices=['vqa', 'caption', 'grounding'],
#                         help="过滤特定任务")
#     parser.add_argument("--dataset", type=str, default=None,
#                         help="过滤特定数据集")
#     parser.add_argument("--output", type=str, default=None,
#                         help="输出Excel文件名")
#     parser.add_argument("--output_csv", type=str, default=None,
#                         help="输出CSV文件名")
    
#     args = parser.parse_args()
    
#     # 收集结果文件
#     result_files = []
    
#     if args.files:
#         result_files = args.files
#     elif args.input_dir:
#         result_files = find_result_files(args.input_dir, args.task, args.dataset)
#     else:
#         print("❌ 请指定 --input_dir 或 --files")
#         return
    
#     if not result_files:
#         print("❌ 未找到任何结果文件")
#         return
    
#     print(f"📂 找到 {len(result_files)} 个结果文件:")
#     for f in result_files:
#         print(f"   - {f}")
    
#     # 加载并处理结果
#     results_list = []
#     for filepath in result_files:
#         try:
#             data = load_results(filepath)
#             processed = process_single_result(data)
#             results_list.append(processed)
#             print(f"   ✓ 已加载: {processed['model_type']} / {processed['dataset']} / {processed['task']}")
#         except Exception as e:
#             print(f"   ⚠️ 加载失败 {filepath}: {e}")
    
#     if not results_list:
#         print("❌ 没有有效的结果")
#         return
    
#     # 生成输出文件名
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
#     if args.output:
#         output_excel = args.output
#     else:
#         task_str = args.task or 'all'
#         output_excel = f"paper_table_{task_str}_{timestamp}.xlsx"
    
#     output_csv = args.output_csv or output_excel.replace('.xlsx', '.csv')
    
#     # 生成表格
#     generate_merged_table(results_list, output_excel, output_csv)


# if __name__ == "__main__":
#     main()
