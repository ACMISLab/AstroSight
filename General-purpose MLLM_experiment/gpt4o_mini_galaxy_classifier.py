#!/usr/bin/env python3
"""
Galaxy Morphology Classification using GPT-4o-mini (OpenAI)
星系形态分类实验 - 使用OpenAI GPT-4o-mini视觉模型
"""

import os
import json
import time
import base64
from typing import Dict, List, Tuple
from collections import defaultdict
import logging
from tqdm import tqdm
import pandas as pd
from openai import OpenAI
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GPT4oMiniGalaxyClassifier:
    def __init__(self, api_key: str):
        """初始化分类器"""
        self.client = OpenAI(
            base_url="https://api.gptsapi.net/v1",
            api_key=api_key
        )
        
        # 星系分类标签映射
        self.label_map = {
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
        self.reverse_label_map = {v: k for k, v in self.label_map.items()}
        
        self.results = []
        
    def encode_image(self, image_path: str) -> str:
        """将图片编码为base64"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def classify_galaxy(self, image_path: str, max_retries: int = 3) -> Tuple[str, str]:
        """
        分类单张星系图片
        
        Args:
            image_path: 图片路径
            max_retries: 最大重试次数
            
        Returns:
            (predicted_label, raw_response): 预测标签和原始响应
        """
        
        # 构建提示词 - 使用原版提示词，不做任何修改
        prompt = "[Output Constraints] Return the answer as one of the following options: 'A:round elliptical', 'B:in-between elliptical', 'C:cigar-shaped elliptical', 'D:edge-on', 'E:Barred spirals', 'F:Unbarred spirals', 'G:Irregular', 'H:merger'  Now, based on the morphological image, return the choice."

        for attempt in range(max_retries):
            try:
                # 读取图片并编码
                base64_image = self.encode_image(image_path)
                
                completion = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}"
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=20,  # 优化：减少输出token浪费
                    temperature=0.1
                )
                
                response = completion.choices[0].message.content
                logger.info(f"Raw response: {response}")
                
                # 提取预测标签
                predicted_label = self.extract_label(response)
                
                if predicted_label:
                    return predicted_label, response
                else:
                    logger.warning(f"Could not extract valid label from response: {response}")
                    
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # 指数退避
                else:
                    logger.error(f"All attempts failed for image: {image_path}")
                    return None, str(e)
        
        return None, "Max retries exceeded"
    
    def extract_label(self, response: str) -> str:
        """从响应中提取标签"""
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
        for label_name, letter in self.reverse_label_map.items():
            if label_name.lower() in response.lower():
                return letter
                
        return None
    
    def load_test_data(self, test_file: str) -> List[Dict]:
        """加载测试数据"""
        test_data = []
        with open(test_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                test_data.append(data)
        return test_data
    
    def extract_true_label(self, data_item: Dict) -> str:
        """从数据项中提取真实标签"""
        assistant_content = data_item['messages'][1]['content']
        # 查找 "option X is selected" 模式
        for letter in 'ABCDEFGH':
            if f"option {letter} is selected" in assistant_content:
                return letter
        return None
    
    def run_evaluation(self, test_file: str, output_file: str = None, sample_size: int = None):
        """运行完整评估"""
        logger.info("Loading test data...")
        test_data = self.load_test_data(test_file)
        
        if sample_size:
            test_data = test_data[:sample_size]
            logger.info(f"Using sample of {sample_size} images for evaluation")
        
        logger.info(f"Total test samples: {len(test_data)}")
        
        y_true = []
        y_pred = []
        detailed_results = []
        
        # 进行分类
        for i, data_item in enumerate(tqdm(test_data, desc="Classifying galaxies")):
            try:
                image_path = data_item['images'][0]
                true_label = self.extract_true_label(data_item)
                
                if not true_label:
                    logger.warning(f"Could not extract true label for item {i}")
                    continue
                
                if not os.path.exists(image_path):
                    logger.warning(f"Image not found: {image_path}")
                    continue
                
                predicted_label, raw_response = self.classify_galaxy(image_path)
                
                if predicted_label:
                    y_true.append(true_label)
                    y_pred.append(predicted_label)
                    
                    detailed_results.append({
                        'image_path': image_path,
                        'true_label': true_label,
                        'true_category': self.label_map[true_label],
                        'predicted_label': predicted_label,
                        'predicted_category': self.label_map[predicted_label],
                        'correct': true_label == predicted_label,
                        'raw_response': raw_response
                    })
                    
                    logger.info(f"Sample {i+1}: True={true_label}({self.label_map[true_label]}), "
                              f"Pred={predicted_label}({self.label_map[predicted_label]}), "
                              f"Correct={true_label == predicted_label}")
                else:
                    logger.error(f"Failed to classify image {image_path}")
                    
            except Exception as e:
                logger.error(f"Error processing item {i}: {str(e)}")
                continue
        
        # 计算评估指标
        metrics = self.calculate_metrics(y_true, y_pred)
        
        # 保存详细结果
        final_results = {
            'model': 'GPT-4o-mini (OpenAI)',
            'metrics': metrics,
            'detailed_results': detailed_results,
            'total_samples': len(test_data),
            'successful_predictions': len(y_true),
            'success_rate': len(y_true) / len(test_data) * 100
        }
        
        if output_file:
            self.save_results(final_results, output_file)
        
        return final_results
    
    def calculate_metrics(self, y_true: List[str], y_pred: List[str]) -> Dict:
        """计算评估指标"""
        if not y_true or not y_pred:
            logger.warning("Empty predictions or true labels")
            return {}
        
        # 整体指标
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        
        # 各类别指标
        precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0, labels=list('ABCDEFGH')
        )
        
        # 混淆矩阵
        cm = confusion_matrix(y_true, y_pred, labels=list('ABCDEFGH'))
        
        metrics = {
            'overall': {
                'accuracy': round(accuracy * 100, 2),
                'precision': round(precision * 100, 2),
                'recall': round(recall * 100, 2),
                'f1_score': round(f1 * 100, 2),
                'total_samples': len(y_true)
            },
            'per_class': {},
            'confusion_matrix': cm.tolist()
        }
        
        # 每个类别的详细指标
        for i, label in enumerate('ABCDEFGH'):
            if i < len(precision_per_class):
                metrics['per_class'][label] = {
                    'category': self.label_map[label],
                    'precision': round(precision_per_class[i] * 100, 2),
                    'recall': round(recall_per_class[i] * 100, 2),
                    'f1_score': round(f1_per_class[i] * 100, 2),
                    'support': int(support_per_class[i]) if i < len(support_per_class) else 0
                }
        
        return metrics
    
    def save_results(self, results: Dict, filename: str):
        """保存最终结果"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"Results saved to {filename}")
            
            # 同时保存为CSV格式
            csv_filename = filename.replace('.json', '_summary.csv')
            self.save_metrics_as_csv(results['metrics'], csv_filename)
            
        except Exception as e:
            logger.error(f"Failed to save results: {str(e)}")
    
    def save_metrics_as_csv(self, metrics: Dict, filename: str):
        """将指标保存为CSV格式"""
        try:
            # 整体指标
            overall_data = {
                'Metric': ['Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F1-Score (%)'],
                'Value': [
                    metrics['overall']['accuracy'],
                    metrics['overall']['precision'],
                    metrics['overall']['recall'],
                    metrics['overall']['f1_score']
                ]
            }
            overall_df = pd.DataFrame(overall_data)
            
            # 各类别指标
            per_class_data = []
            for label, data in metrics['per_class'].items():
                per_class_data.append({
                    'Label': label,
                    'Category': data['category'],
                    'Precision (%)': data['precision'],
                    'Recall (%)': data['recall'],
                    'F1-Score (%)': data['f1_score'],
                    'Support': data['support']
                })
            per_class_df = pd.DataFrame(per_class_data)
            
            # 保存到同一个Excel文件的不同sheet
            excel_filename = filename.replace('.csv', '.xlsx')
            with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
                overall_df.to_excel(writer, sheet_name='Overall_Metrics', index=False)
                per_class_df.to_excel(writer, sheet_name='Per_Class_Metrics', index=False)
            
            logger.info(f"Metrics saved to {excel_filename}")
            
        except Exception as e:
            logger.error(f"Failed to save metrics as CSV: {str(e)}")
    
    def print_results_summary(self, results: Dict):
        """打印结果摘要"""
        print("\n" + "="*60)
        print("GALAXY CLASSIFICATION RESULTS - GPT-4o-mini (OpenAI)")
        print("="*60)
        
        if 'metrics' in results and 'overall' in results['metrics']:
            overall = results['metrics']['overall']
            print(f"\nOverall Performance:")
            print(f"  Accuracy:  {overall['accuracy']:.2f}%")
            print(f"  Precision: {overall['precision']:.2f}%")
            print(f"  Recall:    {overall['recall']:.2f}%")
            print(f"  F1-Score:  {overall['f1_score']:.2f}%")
            print(f"  Total Samples: {overall['total_samples']}")
        
        if 'success_rate' in results:
            print(f"  Success Rate: {results['success_rate']:.2f}%")
        
        if 'metrics' in results and 'per_class' in results['metrics']:
            print(f"\nPer-Class Performance:")
            print(f"{'Label':<5} {'Category':<20} {'Prec(%)':<8} {'Rec(%)':<8} {'F1(%)':<8} {'Support':<8}")
            print("-" * 65)
            
            for label, data in results['metrics']['per_class'].items():
                print(f"{label:<5} {data['category']:<20} {data['precision']:<8.2f} "
                      f"{data['recall']:<8.2f} {data['f1_score']:<8.2f} {data['support']:<8}")
        
        print("="*60)


def main():
    """主函数"""
    # 配置参数
    API_KEY = "sk-7aHfff7ffa383a7d85600b9d2b657d34040160f42efbV4pZ"
    TEST_FILE = "/remote-home/cs_acmis_hby/code/FT-LLM/galaxy_classification/dataset_07/test.jsonl"
    OUTPUT_FILE = "gpt4o_mini_galaxy_classification_results.json"
    
    # 设置样本数量（用于测试，设为None使用全部数据）
    SAMPLE_SIZE = None  # 先用100个样本测试，后续可以改为None使用全部数据
    
    print("Starting Galaxy Classification Experiment with GPT-4o-mini (OpenAI)")
    print(f"API Key: {API_KEY[:10]}...")
    print(f"Test file: {TEST_FILE}")
    print(f"Sample size: {SAMPLE_SIZE}")
    
    # 创建分类器
    classifier = GPT4oMiniGalaxyClassifier(API_KEY)
    
    # 运行评估
    try:
        results = classifier.run_evaluation(
            test_file=TEST_FILE,
            output_file=OUTPUT_FILE,
            sample_size=SAMPLE_SIZE
        )
        
        # 打印结果摘要
        classifier.print_results_summary(results)
        
        print(f"\nExperiment completed! Results saved to {OUTPUT_FILE}")
        
    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
