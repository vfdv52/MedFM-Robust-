#!/usr/bin/env python3
"""
CSV Results Extraction Tool

================ Extracts all CSV files from the results folder, preserving complete context information:

- Dataset Name

- Model Name

- Fine-tuning Strategy

- Evaluation Type (adversarial/perturbation)


Usage:

python extract_csv_results.py --results_dir ./results --output_dir ./extracted_csv

python extract_csv_results.py --results_dir ./results --output_dir ./extracted_csv --flat # Flatten the output

"""


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
    Parsing Experimental Metadata from CSV Paths
    
    Example Path Pattern:
    
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
    
    # 1. Extract timestamp and strategy from the main directory name
    for part in path_parts:
        match = re.match(r'^(\d{8}_\d{6})_(.+)$', part)
        if match:
            info['timestamp'] = match.group(1)
            info['strategy'] = match.group(2)
            break
    
    # 2. Extract the model and dataset from the pipeline_* directory names.
    for part in path_parts:
        match = re.match(r'^pipeline_([^_]+)_(.+)$', part)
        if match:
            info['model'] = match.group(1)
            info['dataset'] = match.group(2)
            break
    
    # 3. Extract eval_type from CSV filenames
    csv_name = info['csv_name']
    if 'adversarial' in csv_name.lower():
        info['eval_type'] = 'adversarial'
    elif 'perturbation' in csv_name.lower():
        info['eval_type'] = 'perturbation'
    
    # 4. Determine if it is a summary file
    info['is_summary'] = 'SUMMARY' in csv_name or 'STATS' in csv_name
    
    return info


def find_all_csv_files(results_dir: str) -> list:
    """Recursively search all CSV files"""
    csv_files = []
    for root, dirs, files in os.walk(results_dir):
        for file in files:
            if file.endswith('.csv'):
                full_path = os.path.join(root, file)
                csv_files.append(full_path)
    return csv_files


def generate_output_filename(info: dict, flat: bool = False) -> str:
    """
    Output filename generation:
    
    Hierarchical mode: {model}/{dataset}/{strategy}/{original filename}
    
    Flat mode: {model}__{dataset}__{strategy}__{eval_type}__{original filename}
    """
    if flat:
    
        parts = [
            info['model'],
            info['dataset'],
            info['strategy'],
            info['eval_type']
        ]
        base_name = info['csv_name']
        return f"{'__'.join(parts)}__{base_name}"
    else:

        return os.path.join(
            info['model'],
            info['dataset'],
            info['strategy'],
            info['csv_name']
        )


def extract_csv_files(results_dir: str, output_dir: str, flat: bool = False, 
                       summary_only: bool = False, detail_only: bool = False):
    """
    Main Extraction Function
    
    Args:
    
        results_dir: Results directory (./results)
        
        output_dir: Output directory
        
        flat: Whether to use flat naming
        
        summary_only: Extract only summary files (*_SUMMARY.csv, *_STATS*.csv)
        
        detail_only: Extract only detailed result files (not summary files)
    """
    print(f"Scan Catalog: {results_dir}")
    csv_files = find_all_csv_files(results_dir)
    print(f"Find {len(csv_files)} CSV files")
    
    if not csv_files:
        print("No CSV files found")
        return
    
    all_info = []
    for csv_path in csv_files:
        info = parse_experiment_path(csv_path)
        all_info.append(info)
    
    if summary_only:
        all_info = [info for info in all_info if info['is_summary']]
        print(f"Filter summary files: {len(all_info)}")
    elif detail_only:
        all_info = [info for info in all_info if not info['is_summary']]
        print(f"Filter detailed files: {len(all_info)}")
    
    if not all_info:
        print("No files can be extracted after filtering.")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    copied_count = 0
    index_data = []
    
    for info in all_info:
        src_path = info['csv_path']
        rel_output = generate_output_filename(info, flat)
        dst_path = os.path.join(output_dir, rel_output)
        
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        
        try:
            shutil.copy2(src_path, dst_path)
            copied_count += 1
            
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
            
            print(f" [{info['model']}][{info['dataset']}][{info['strategy']}] {info['csv_name']}")
            
        except Exception as e:
            print(f" Copy failed: {src_path} -> {e}")
    
    index_path = os.path.join(output_dir, 'index.csv')
    with open(index_path, 'w', newline='', encoding='utf-8') as f:
        if index_data:
            writer = csv.DictWriter(f, fieldnames=index_data[0].keys())
            writer.writeheader()
            writer.writerows(index_data)
    
    index_json_path = os.path.join(output_dir, 'index.json')
    with open(index_json_path, 'w', encoding='utf-8') as f:
        json.dump(index_data, f, indent=2, ensure_ascii=False)
    
    print(f"Total copied: {copied_count} files")
    print(f"Output directory: {output_dir}")
    print(f"Index file: {index_path}")
    
    models = set(info['model'] for info in index_data)
    datasets = set(info['dataset'] for info in index_data)
    strategies = set(info['strategy'] for info in index_data)
    
    print(f"\nDimension Statistics:")
    print(f"Models: {len(models)} - {', '.join(sorted(models))}")
    print(f"Datasets: {len(datasets)} - {', '.join(sorted(datasets))}")
    print(f"Strategies: {len(strategies)} - {', '.join(sorted(strategies))}")
    
    return index_data


def create_merged_summary(output_dir: str, index_data: list):
    """
    Create a merged summary table for easy and quick comparison of the performance of different strategies.
    
    Output:
    
        - merged_summary_adversarial.csv: Summary data for all adversarial attacks
        
        - merged_summary_perturbation.csv: Summary data for all perturbation assessments
    """
    import pandas as pd
    

    summary_files = [info for info in index_data if info['is_summary'] and 'SUMMARY' in info['output_path']]
    
    if not summary_files:
        print("Summary file not found, skip merging.")
        return
    
    # Treat adversarial and perturbation separately
    for eval_type in ['adversarial', 'perturbation']:
        type_files = [info for info in summary_files if info['eval_type'] == eval_type]
        
        if not type_files:
            continue
        
        merged_rows = []
        for info in type_files:
            csv_path = os.path.join(output_dir, info['output_path'])
            try:
                df = pd.read_csv(csv_path)

                df['Model'] = info['model']
                df['Dataset'] = info['dataset']
                df['Strategy'] = info['strategy']
                df['Timestamp'] = info['timestamp']
                merged_rows.append(df)
            except Exception as e:
                print(f"Read failed: {csv_path} - {e}")
        
        if merged_rows:
            merged_df = pd.concat(merged_rows, ignore_index=True)

            cols = ['Model', 'Dataset', 'Strategy', 'Timestamp'] + \
                   [c for c in merged_df.columns if c not in ['Model', 'Dataset', 'Strategy', 'Timestamp']]
            merged_df = merged_df[cols]
            
            output_path = os.path.join(output_dir, f'merged_summary_{eval_type}.csv')
            merged_df.to_csv(output_path, index=False, float_format='%.4f')
            print(f"Merged summary table: {output_path} ({len(merged_df)} rows)")


def main():
    parser = argparse.ArgumentParser(
        description=Extract CSV files from the results directory, retaining dataset/model/policy information.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Usage Examples:
        
        ---------
        # Basic Usage - Hierarchical Directory Structure
        
        python extract_csv_results.py --results_dir ./results --output_dir ./extracted_csv
        
        # Flatten Output - All files are placed in one directory, and the filenames contain all information
        
        python extract_csv_results.py --results_dir ./results --output_dir ./extracted_csv --flat
        
        # Extract Only the Summary File (SUMMARY.csv)
        
        python extract_csv_results.py --results_dir ./results --output_dir ./extracted_csv --summary_only
        
        # Extract Only the Detailed Results File (Excluding SUMMARY)
        
        python extract_csv_results.py --results_dir ./results --output_dir ./extracted_csv --detail_only
        
        # Generate a Merged Summary Table
        
        python extract_csv_results.py --results_dir ./results --output_dir ./extracted_csv --merge_summary
        """
    )
    
    parser.add_argument('--results_dir', type=str, default='./results',
    
    help='results directory path (default: ./results)')
    parser.add_argument('--output_dir', type=str, default='./extracted_csv',
    
    help='output directory path (default: ./extracted_csv)')
    parser.add_argument('--flat', action='store_true',
    
    help='Use flat naming (place all files in the same directory)')
    parser.add_argument('--summary_only', action='store_true',
    
    help='Extract only summary files (*_SUMMARY.csv, *_STATS*.csv)')
    parser.add_argument('--detail_only', action='store_true',
    
    help='Extract only detailed result files'` (Non-summary file)') parser.add_argument('--merge_summary', action='store_true', help='Generate merged summary table (requires pandas)')
    
    args = parser.parse_args()
    
    if args.summary_only and args.detail_only:
        print("Error: --summary_only and --detail_only cannot be used together.")
        return
    
    if not os.path.exists(args.results_dir):
        print(f"Error: Result directory does not exist: {args.results_dir}")
        return
    
    index_data = extract_csv_files(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        flat=args.flat,
        summary_only=args.summary_only,
        detail_only=args.detail_only
    )
    
    if args.merge_summary and index_data:
        try:
            create_merged_summary(args.output_dir, index_data)
        except ImportError:
            print("Merging and summarizing requires the pandas library. Please install it: `pip install pandas`")


if __name__ == "__main__":
    main()
