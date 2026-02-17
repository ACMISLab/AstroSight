#!/usr/bin/env python3
"""
GPT-5 Thinking (占位) Galaxy Classification Evaluator
- 说明：当前OpenAI未发布“GPT-5”，审稿人常指的“GPT-5 Thinking”多为 GPT-o1 / o 系列推理模型。
- 注意：o1 系列目前不支持视觉输入（仅文本）。本脚本默认使用支持视觉的多模态模型（默认 gpt-4o），
  你也可以通过环境变量 MODEL_NAME 覆盖为你代理平台映射的模型名（如 "gpt-5-thinking"），前提是该模型支持图像输入。
- 协议：OpenAI Chat Completions 兼容协议，通过环境变量 OPENAI_API_BASE 与 OPENAI_API_KEY 配置。

输出：
- gpt5_thinking_galaxy_classification_results.json  （逐样本与汇总）
- gpt5_thinking_galaxy_classification_results_summary.xlsx （宏/加权平均与每类指标、混淆矩阵）

使用：
  set OPENAI_API_BASE=https://api.openai-proxy.org/v1
  set OPENAI_API_KEY=sk-********
  set MODEL_NAME=gpt-4o   # 或者你的代理映射模型名
  python gpt5_thinking_galaxy_classifier.py
"""
import os
import io
import sys
import time
import json
import base64
import random
import logging
import traceback
import argparse
from typing import List, Dict, Any

import requests
# from PIL import Image  # 不需要，因为我们改用直接base64编码
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import pandas as pd

# ------------------------- 配置 -------------------------
CONFIG = {
    # 使用本地测试文件路径
    'test_file': 'd:/学习/我的论文/Galaxy_Eperiement/Galaxy-Zoo-Classification/Contrast_experiment/commercial_LLM/test.jsonl',
    'output_file': 'gpt5_thinking_galaxy_classification_results.json',
    'checkpoint_file': 'gpt5_thinking_checkpoint.json',  # 断点续传文件
    'sleep_s': 0.2,    # 增加限速避免API问题
    'max_retries': 2,  # 降低重试次数避免卡死
    'timeout_s': 180,  # 降低超时时间快速跳过问题图片
    # OpenAI 兼容协议
    'api_base': os.getenv('OPENAI_API_BASE', 'https://api.openai-proxy.org/v1'),
    'api_key': os.getenv('OPENAI_API_KEY', ''),
    'model': os.getenv('MODEL_NAME', 'gpt-5'),  # 使用GPT-5作为默认模型
}

# 星系分类标签映射（与其他脚本完全一致）
LABEL_MAP = {
    'A': 'round elliptical',
    'B': 'in-between elliptical', 
    'C': 'cigar-shaped elliptical',
    'D': 'edge-on',
    'E': 'Barred spirals',
    'F': 'Unbarred spirals',
    'G': 'Irregular',
    'H': 'merger'
}

# 反向映射
REVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

INSTRUCTION = "[Output Constraints] Return the answer as one of the following options: 'A:round elliptical', 'B:in-between elliptical', 'C:cigar-shaped elliptical', 'D:edge-on', 'E:Barred spirals', 'F:Unbarred spirals', 'G:Irregular', 'H:merger'  Now, based on the morphological image, return the choice."

# 清理重复定义，使用LABEL_MAP作为主要映射

# ------------------------- 工具函数 -------------------------

def setup_logger():
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')


def encode_image_to_base64(path: str) -> str:
    """将图片编码为base64（与其他脚本保持一致）"""
    with open(path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def load_test_data(test_file: str) -> List[Dict]:
    """加载测试数据（与其他脚本完全一致）"""
    test_data = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            test_data.append(data)
    return test_data

def extract_true_label(data_item: Dict) -> str:
    """从数据项中提取真实标签（与其他脚本完全一致）"""
    assistant_content = data_item['messages'][1]['content']
    # 查找 "option X is selected" 模式
    for letter in 'ABCDEFGH':
        if f"option {letter} is selected" in assistant_content:
            return letter
    return None


def openai_chat_completion_with_image(api_base: str, api_key: str, model: str, base64_image: str, instruction: str) -> str:
    url = f"{api_base.rstrip('/')}/chat/completions"
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f"Bearer {api_key}",
    }
    payload = {
        'model': model,
        'messages': [
            {
                'role': 'user',
                'content': [
                    { 'type': 'text', 'text': instruction },
                    { 
                        'type': 'image_url', 
                        'image_url': { 
                            'url': f"data:image/jpeg;base64,{base64_image}" 
                        } 
                    },
                ]
            }
        ],
        'max_completion_tokens': 600,  # 平衡成功率和成本
        # 'temperature': 0.1,  # gpt-5不支持自定义temperature，只能使用默认值1
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=CONFIG['timeout_s'])
    
    # 添加调试信息
    if resp.status_code != 200:
        logging.error(f"API错误: {resp.status_code}")
        logging.error(f"错误响应: {resp.text}")
        logging.error(f"请求模型: {model}")
        
    resp.raise_for_status()
    data = resp.json()
    
    # 添加调试信息：显示完整响应结构
    logging.info(f"API响应结构: {data}")
    
    # 兼容不同返回结构
    if 'choices' in data and data['choices']:
        content = data['choices'][0]['message']['content']
        logging.info(f"提取的内容: {content}")
        return content
    # Response协议自动转换的情况（少见）
    if 'output' in data:
        logging.info(f"Response协议输出: {data['output']}")
        return data['output']
    
    logging.warning(f"无法从响应中提取内容: {data}")
    return ''


def extract_label(response: str) -> str:
    """从响应中提取标签（与其他脚本保持一致的逻辑）"""
    response = response.strip().upper()
    
    # 查找 "option X is selected" 模式 (原版格式)
    for letter in 'ABCDEFGH':
        if f"OPTION {letter} IS SELECTED" in response:
            return letter
    
    # 查找单个字母A-H
    for char in response:
        if char in 'ABCDEFGH':
            return char
    
    # 查找完整标签名称
    reverse_map = {v.upper(): k for k, v in LABEL_MAP.items()}
    for label_name, letter in reverse_map.items():
        if label_name in response:
            return letter
            
    return None


# ------------------------- 主流程 -------------------------

def map_remote_to_local_path(remote_path: str) -> str:
    """将远程图像路径映射为本地路径"""
    # 从远程路径中提取文件名
    filename = os.path.basename(remote_path)
    # 映射到本地test_images目录
    local_path = f"d:/学习/我的论文/Galaxy_Eperiement/Galaxy-Zoo-Classification/Contrast_experiment/commercial_LLM/test_images/{filename}"
    return local_path

def load_checkpoint():
    """加载断点续传数据"""
    try:
        if os.path.exists(CONFIG['checkpoint_file']):
            with open(CONFIG['checkpoint_file'], 'r') as f:
                checkpoint = json.load(f)
            logging.info(f"加载断点续传文件，已处理 {checkpoint.get('processed_count', 0)} 张图片")
            return checkpoint
    except Exception as e:
        logging.warning(f"加载断点续传文件失败: {e}")
    return {'processed_count': 0, 'y_true': [], 'y_pred': [], 'detailed_results': [], 'success_count': 0}

def save_checkpoint(checkpoint_data):
    """保存断点续传数据"""
    try:
        with open(CONFIG['checkpoint_file'], 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
    except Exception as e:
        logging.warning(f"保存断点续传文件失败: {e}")

def run_evaluation(test_file: str, output_file: str, model: str, api_base: str, api_key: str, sample_size: int = None):
    """运行完整评估（与其他脚本结构一致）"""
    logging.info("Loading test data...")
    test_data = load_test_data(test_file)
    
    if sample_size:
        test_data = test_data[:sample_size]
        logging.info(f"Using sample of {sample_size} images for evaluation")
    
    logging.info(f"Total test samples: {len(test_data)}")
    logging.info(f"Model: {model} | API_BASE: {api_base}")
    
    # 加载断点续传数据
    checkpoint = load_checkpoint()
    y_true = checkpoint['y_true']
    y_pred = checkpoint['y_pred'] 
    detailed_results = checkpoint['detailed_results']
    success_count = checkpoint['success_count']
    processed_count = checkpoint['processed_count']
    
    total_samples = len(test_data)

    # 从断点位置继续处理
    start_index = processed_count
    if start_index > 0:
        logging.info(f"从第 {start_index + 1} 张图片继续处理")
    
    # 进行分类
    for i, data_item in enumerate(test_data[start_index:], start_index + 1):
        try:
            remote_image_path = data_item['images'][0]
            # 映射到本地路径
            image_path = map_remote_to_local_path(remote_image_path)
            true_label = extract_true_label(data_item)
            
            if not true_label:
                logging.warning(f"Cannot extract true label from item {i}")
                continue
            
            # 重试机制
            predicted_label = None
            last_err = None
            for attempt in range(1, CONFIG['max_retries'] + 1):
                try:
                    base64_image = encode_image_to_base64(image_path)
                    response = openai_chat_completion_with_image(api_base, api_key, model, base64_image, INSTRUCTION)
                    predicted_label = extract_label(response)
                    if predicted_label:
                        break
                    else:
                        logging.warning(f"Could not extract valid label from response: {response[:200]}...")
                        logging.info(f"完整响应内容: {response}")
                except Exception as e:
                    last_err = e
                    logging.warning(f"[{i}/{len(test_data)}] 请求失败（第{attempt}次）：{e}")
                    if attempt < CONFIG['max_retries']:
                        time.sleep(1.5 * attempt)
            
            if not predicted_label:
                logging.error(f"[{i}] 放弃：{image_path} | 错误：{last_err}")
                continue  # 跳过失败样本，与其他模型保持一致
            
            y_true.append(true_label)
            y_pred.append(predicted_label)
            success_count += 1
            
            detailed_results.append({
                'index': i,
                'image_path': image_path,
                'true_label': true_label,
                'true_category': LABEL_MAP[true_label],
                'predicted_label': predicted_label,
                'predicted_category': LABEL_MAP[predicted_label],
                'correct': int(true_label == predicted_label)
            })
            
            if CONFIG['sleep_s'] > 0:
                time.sleep(CONFIG['sleep_s'])
            
            # 每100张保存一次断点
            if i % 100 == 0:
                checkpoint_data = {
                    'processed_count': i,
                    'y_true': y_true,
                    'y_pred': y_pred,
                    'detailed_results': detailed_results,
                    'success_count': success_count
                }
                save_checkpoint(checkpoint_data)
                logging.info(f"已处理 {i}/{len(test_data)} (已保存断点)")
                
        except Exception as e:
            logging.error(f"Error processing item {i}: {str(e)}")
            # 即使出错也保存断点
            checkpoint_data = {
                'processed_count': i,
                'y_true': y_true,
                'y_pred': y_pred,
                'detailed_results': detailed_results,
                'success_count': success_count
            }
            save_checkpoint(checkpoint_data)
            continue
    
    return y_true, y_pred, detailed_results, total_samples, success_count

def main():
    parser = argparse.ArgumentParser(description='GPT-5 Galaxy Classification Test')
    parser.add_argument('--limit', type=int, default=None, help='限制测试样本数量（用于小规模测试）')
    parser.add_argument('--model', type=str, default=None, help='覆盖环境变量中的模型名')
    args = parser.parse_args()
    
    setup_logger()
    if not CONFIG['api_key']:
        logging.error('OPENAI_API_KEY 未设置，请在环境变量中配置。')
        sys.exit(1)

    # 如果命令行指定了模型，覆盖配置
    model = args.model if args.model else CONFIG['model']
    
    logging.info(f"Starting Galaxy Classification Experiment with {model}")
    logging.info(f"API Key: {CONFIG['api_key'][:10]}...")
    logging.info(f"Test file: {CONFIG['test_file']}")
    logging.info(f"Sample size: {args.limit}")

    # 运行评估
    try:
        y_true, y_pred, detailed_results, total_samples, success_count = run_evaluation(
            test_file=CONFIG['test_file'],
            output_file=CONFIG['output_file'],
            model=model,
            api_base=CONFIG['api_base'],
            api_key=CONFIG['api_key'],
            sample_size=args.limit
        )

        labels_order = list('ABCDEFGH')

        if y_true:
            report = classification_report(y_true, y_pred, labels=labels_order, output_dict=True, zero_division=0)
            cm = confusion_matrix(y_true, y_pred, labels=labels_order)
            accuracy = accuracy_score(y_true, y_pred)
            precision, recall, f1, support = precision_recall_fscore_support(
                y_true, y_pred, labels=labels_order, average=None, zero_division=0
            )
        else:
            report = {}
            cm = [[0 for _ in labels_order] for _ in labels_order]
            accuracy = 0.0
            precision = recall = f1 = [0.0] * len(labels_order)
            support = [0] * len(labels_order)
        
        success_rate = (success_count / total_samples) * 100 if total_samples else 0

        # 组织结果
        results = {
            'model': model,
            'api_base': CONFIG['api_base'],
            'test_samples': total_samples,
            'successful_predictions': success_count,
            'failed_predictions': total_samples - success_count,
            'success_rate': success_rate,
            'accuracy': accuracy * 100,
            'detailed_results': detailed_results,
            'metrics': {
                'overall': {
                    'accuracy': accuracy * 100,
                    'precision': float(precision.mean()) * 100 if len(precision) > 0 else 0.0,
                    'recall': float(recall.mean()) * 100 if len(recall) > 0 else 0.0,
                    'f1_score': float(f1.mean()) * 100 if len(f1) > 0 else 0.0,
                    'total_samples': total_samples,
                    'successful_samples': success_count
                },
                'per_class': {}
            },
            'confusion_matrix': cm.tolist()
        }
        
        # 添加每个类别的详细指标
        for i, label in enumerate(labels_order):
            results['metrics']['per_class'][label] = {
                'category': LABEL_MAP.get(label, f'Unknown-{label}'),
                'precision': float(precision[i]) * 100,
                'recall': float(recall[i]) * 100,
                'f1_score': float(f1[i]) * 100,
                'support': int(support[i])
            }

        # 保存结果
        with open(CONFIG['output_file'], 'w') as f:
            json.dump(results, f, indent=2)
        
        # 生成Excel汇总
        try:
            with pd.ExcelWriter('gpt5_thinking_galaxy_classification_results_summary.xlsx') as writer:
                # 总体指标
                overview_data = {
                    'accuracy(%)': results['accuracy'],
                    'success_rate(%)': results['success_rate']
                }
                pd.DataFrame([overview_data]).to_excel(writer, index=False, sheet_name='overview')
                
                # 每类指标
                per_class_data = []
                for label, metrics in results['metrics']['per_class'].items():
                    per_class_data.append({
                        'Label': label,
                        'Category': metrics['category'],
                        'Precision(%)': metrics['precision'],
                        'Recall(%)': metrics['recall'],
                        'F1-Score(%)': metrics['f1_score'],
                        'Support': metrics['support']
                    })
                pd.DataFrame(per_class_data).to_excel(writer, index=False, sheet_name='per_class')
                
                # 混淆矩阵
                cm_df = pd.DataFrame(results['confusion_matrix'], 
                                   index=[f"{l}({LABEL_MAP.get(l, 'Unknown')})" for l in labels_order],
                                   columns=[f"{l}({LABEL_MAP.get(l, 'Unknown')})" for l in labels_order])
                cm_df.to_excel(writer, sheet_name='confusion_matrix')
        except Exception as e:
            logging.warning(f'写入Excel失败：{e}')
        
        # 删除断点文件
        try:
            if os.path.exists(CONFIG['checkpoint_file']):
                os.remove(CONFIG['checkpoint_file'])
                logging.info("已删除断点续传文件")
        except Exception as e:
            logging.warning(f"删除断点续传文件失败: {e}")
            
        logging.info(f"实验完成！结果已保存到 {CONFIG['output_file']}")
        
        # 打印结果摘要
        print("\n" + "="*60)
        print("实验结果摘要:")
        print("="*60)
        print(f"模型: {model}")
        print(f"测试样本数: {results['test_samples']}")
        print(f"成功样本数: {results['successful_predictions']} (成功率: {results['success_rate']:.2f}%)")
        print(f"失败样本数: {results['failed_predictions']}")
        print(f"准确率: {results['accuracy']:.2f}% (基于成功样本)")
        
        if 'metrics' in results and 'per_class' in results['metrics']:
            print(f"\n每类性能:")
            print(f"{'Label':<5} {'Category':<20} {'Prec(%)':<8} {'Rec(%)':<8} {'F1(%)':<8} {'Support':<8}")
            print("-" * 65)
            
            for label, data in results['metrics']['per_class'].items():
                print(f"{label:<5} {data['category']:<20} {data['precision']:<8.2f} "
                      f"{data['recall']:<8.2f} {data['f1_score']:<8.2f} {data['support']:<8}")
        
        print("="*60)
        
    except Exception as e:
        logging.error(f"实验失败: {str(e)}")
        raise


if __name__ == '__main__':
    main()
