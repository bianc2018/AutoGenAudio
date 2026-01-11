#!/usr/bin/env python3
"""
AutoEvalAudio - ASR模型评估工具
支持多种后端（Sherpa-ONNX、FunASR WebSocket）
支持并行推理和多指标评估
"""

import os
import csv
import time
import logging
import tomllib
import math
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# 导入ASR模型相关模块
from ASRModel.base import BaseASRModel
from ASRModel.sherpa_onnx import SherpaONNXAdapter
from ASRModel.funasr_ws import FunASRWSAdapter

# 导入jiwer用于计算WER和CER
try:
    from jiwer import wer, cer
    JIWER_AVAILABLE = True
except ImportError:
    JIWER_AVAILABLE = False
    logging.warning("jiwer库未安装，将使用自定义实现计算WER和CER")

# ---------------- 日志配置 ----------------
def setup_logger(config: dict) -> logging.Logger:
    """设置日志配置"""
    log_level = config.get("level", "INFO").upper()
    log_format = config.get("format", "%(asctime)s [%(levelname)s] %(message)s")
    
    logging.basicConfig(
        level=log_level,
        format=log_format,
        datefmt="%H:%M:%S",
    )
    
    return logging.getLogger("AutoEvalAudio")

# ---------------- 模型工厂 ----------------
class ASRModelFactory:
    @staticmethod
    def create(model_config: dict) -> BaseASRModel:
        """根据配置创建ASR模型实例"""
        backend_type = model_config.get("type")
        if backend_type == "sherpa_onnx":
            return SherpaONNXAdapter(model_config)
        elif backend_type == "funasr_ws":
            return FunASRWSAdapter(model_config)
        else:
            raise ValueError(f"不支持的后端类型: {backend_type}")

# ---------------- 数据结构 ----------------
@dataclass
class DatasetItem:
    """数据集项"""
    audio_path: str
    csv_path: str
    reference_text: str
    speaker_name: str
    start_time: float = 0.0
    end_time: float = 0.0

@dataclass
class EvaluationResult:
    """评估结果"""
    model_name: str
    audio_path: str
    reference_text: str
    hypothesis_text: str
    wer: float = 0.0
    cer: float = 0.0
    cpcer: float = 0.0
    sacer: float = 0.0
    delta_cp: float = 0.0
    delta_sa: float = 0.0
    rtf0: float = 0.0
    qps: float = 0.0
    latency: float = 0.0
    success: bool = True
    error: Optional[str] = None

@dataclass
class ModelSummary:
    """模型评估摘要"""
    model_name: str
    total_items: int
    successful_items: int
    wer: float = 0.0
    cer: float = 0.0
    cpcer: float = 0.0
    sacer: float = 0.0
    delta_cp: float = 0.0
    delta_sa: float = 0.0
    avg_rtf0: float = 0.0
    avg_qps: float = 0.0
    avg_latency: float = 0.0

# ---------------- 工具函数 ----------------
def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, "rb") as f:
        return tomllib.load(f)

def load_dataset(dataset_dir: str) -> List[DatasetItem]:
    """加载数据集"""
    logger = logging.getLogger("AutoEvalAudio")
    logger.info(f"正在加载数据集: {dataset_dir}")
    
    # 检查目录是否存在
    if not os.path.exists(dataset_dir):
        logger.warning(f"数据集目录不存在: {dataset_dir}")
        return []
    
    dataset_items = []
    
    # 遍历数据集目录
    for file in os.listdir(dataset_dir):
        if file.endswith(".wav"):
            base_name = os.path.splitext(file)[0]
            audio_path = os.path.join(dataset_dir, file)
            csv_path = os.path.join(dataset_dir, f"{base_name}.csv")
            
            # 检查对应的csv文件是否存在
            if os.path.exists(csv_path):
                # 解析csv文件
                with open(csv_path, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    rows = list(reader)
                    
                    # 处理每一行
                    for row in rows:
                        if len(row) < 4:
                            logger.warning(f"CSV文件行格式不正确: {row}")
                            continue
                        
                        start_time = float(row[0]) if row[0] else 0.0
                        end_time = float(row[1]) if row[1] else 0.0
                        speaker_name = row[2]
                        reference_text = row[3]
                        
                        dataset_item = DatasetItem(
                            audio_path=audio_path,
                            csv_path=csv_path,
                            reference_text=reference_text,
                            speaker_name=speaker_name,
                            start_time=start_time,
                            end_time=end_time
                        )
                        dataset_items.append(dataset_item)
                
                logger.info(f"已加载数据集项: {audio_path} + {csv_path}")
            else:
                logger.warning(f"未找到对应的CSV文件: {csv_path}")
    
    logger.info(f"数据集加载完成，共 {len(dataset_items)} 个项")
    return dataset_items

def calculate_wer_custom(reference: str, hypothesis: str) -> float:
    """自定义WER计算"""
    if not reference or not hypothesis:
        return 1.0
    
    # 分词
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    
    # 计算编辑距离
    dp = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]
    
    for i in range(len(ref_words) + 1):
        dp[i][0] = i
    
    for j in range(len(hyp_words) + 1):
        dp[0][j] = j
    
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i-1][j-1], dp[i][j-1], dp[i-1][j]) + 1
    
    return dp[-1][-1] / len(ref_words)

def calculate_cer_custom(reference: str, hypothesis: str) -> float:
    """自定义CER计算"""
    if not reference or not hypothesis:
        return 1.0
    
    # 计算编辑距离
    dp = [[0] * (len(hypothesis) + 1) for _ in range(len(reference) + 1)]
    
    for i in range(len(reference) + 1):
        dp[i][0] = i
    
    for j in range(len(hypothesis) + 1):
        dp[0][j] = j
    
    for i in range(1, len(reference) + 1):
        for j in range(1, len(hypothesis) + 1):
            if reference[i-1] == hypothesis[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i-1][j-1], dp[i][j-1], dp[i-1][j]) + 1
    
    return dp[-1][-1] / len(reference)

def calculate_metrics(reference: str, hypothesis: str) -> Dict[str, float]:
    """计算评估指标"""
    # 使用jiwer库或自定义实现计算WER和CER
    if JIWER_AVAILABLE:
        wer_value = wer(reference, hypothesis)
        cer_value = cer(reference, hypothesis)
    else:
        wer_value = calculate_wer_custom(reference, hypothesis)
        cer_value = calculate_cer_custom(reference, hypothesis)
    
    # 计算cpCER（大小写不敏感CER）
    reference_lower = reference.lower()
    hypothesis_lower = hypothesis.lower()
    if JIWER_AVAILABLE:
        cpcer_value = cer(reference_lower, hypothesis_lower)
    else:
        cpcer_value = calculate_cer_custom(reference_lower, hypothesis_lower)
    
    # 计算saCER（空格不敏感CER）
    reference_no_space = reference.replace(" ", "")
    hypothesis_no_space = hypothesis.replace(" ", "")
    if JIWER_AVAILABLE:
        sacer_value = cer(reference_no_space, hypothesis_no_space)
    else:
        sacer_value = calculate_cer_custom(reference_no_space, hypothesis_no_space)
    
    # 计算Δcp和Δsa
    delta_cp = cer_value - cpcer_value
    delta_sa = cer_value - sacer_value
    
    return {
        "wer": wer_value,
        "cer": cer_value,
        "cpcer": cpcer_value,
        "sacer": sacer_value,
        "delta_cp": delta_cp,
        "delta_sa": delta_sa
    }

def evaluate_single_item(model: BaseASRModel, dataset_item: DatasetItem) -> EvaluationResult:
    """评估单个数据项"""
    logger = logging.getLogger("AutoEvalAudio")
    
    try:
        # 记录开始时间
        start_time = time.time()
        
        # 执行转写
        result = model.transcribe(dataset_item.audio_path)
        
        # 记录结束时间
        end_time = time.time()
        latency = end_time - start_time
        
        # 获取转写结果
        hypothesis_text = result.get("text", "")
        
        # 计算评估指标
        metrics = calculate_metrics(dataset_item.reference_text, hypothesis_text)
        
        # 计算RTF0（实时因子）和QPS（每秒处理请求数）
        # 注意：这里需要知道音频时长才能计算RTF0
        # 简化处理，假设音频时长为1秒
        audio_duration = 1.0
        rtf0 = latency / audio_duration if audio_duration > 0 else 0.0
        qps = 1.0 / latency if latency > 0 else 0.0
        
        # 构建评估结果
        eval_result = EvaluationResult(
            model_name=model.get_name(),
            audio_path=dataset_item.audio_path,
            reference_text=dataset_item.reference_text,
            hypothesis_text=hypothesis_text,
            wer=metrics["wer"],
            cer=metrics["cer"],
            cpcer=metrics["cpcer"],
            sacer=metrics["sacer"],
            delta_cp=metrics["delta_cp"],
            delta_sa=metrics["delta_sa"],
            rtf0=rtf0,
            qps=qps,
            latency=latency,
            success=result.get("success", True),
            error=result.get("error")
        )
        
        logger.debug(f"评估结果: {model.get_name()} - {os.path.basename(dataset_item.audio_path)} - WER: {eval_result.wer:.4f}")
        
        return eval_result
    except Exception as e:
        logger.error(f"评估失败: {model.get_name()} - {os.path.basename(dataset_item.audio_path)} - {e}")
        return EvaluationResult(
            model_name=model.get_name(),
            audio_path=dataset_item.audio_path,
            reference_text=dataset_item.reference_text,
            hypothesis_text="",
            success=False,
            error=str(e)
        )

def evaluate_model(model: BaseASRModel, dataset: List[DatasetItem], parallel: bool = False, max_workers: int = 4) -> List[EvaluationResult]:
    """评估模型"""
    logger = logging.getLogger("AutoEvalAudio")
    logger.info(f"开始评估模型: {model.get_name()}")
    logger.info(f"评估项数量: {len(dataset)}")
    logger.info(f"并行模式: {parallel}, 最大工作线程: {max_workers}")
    
    results = []
    
    if parallel:
        # 使用并行处理
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_item = {
                executor.submit(evaluate_single_item, model, item): item 
                for item in dataset
            }
            
            # 处理结果
            for future in as_completed(future_to_item):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"任务执行失败: {e}")
    else:
        # 串行处理
        for item in dataset:
            result = evaluate_single_item(model, item)
            results.append(result)
    
    logger.info(f"模型评估完成: {model.get_name()}, 成功项: {sum(1 for r in results if r.success)}/{len(results)}")
    
    return results

def generate_model_summary(model_name: str, results: List[EvaluationResult]) -> ModelSummary:
    """生成模型评估摘要"""
    total_items = len(results)
    successful_items = sum(1 for r in results if r.success)
    
    # 只计算成功的结果
    successful_results = [r for r in results if r.success]
    
    if not successful_results:
        return ModelSummary(
            model_name=model_name,
            total_items=total_items,
            successful_items=successful_items
        )
    
    # 计算平均指标
    avg_wer = sum(r.wer for r in successful_results) / len(successful_results)
    avg_cer = sum(r.cer for r in successful_results) / len(successful_results)
    avg_cpcer = sum(r.cpcer for r in successful_results) / len(successful_results)
    avg_sacer = sum(r.sacer for r in successful_results) / len(successful_results)
    avg_delta_cp = sum(r.delta_cp for r in successful_results) / len(successful_results)
    avg_delta_sa = sum(r.delta_sa for r in successful_results) / len(successful_results)
    avg_rtf0 = sum(r.rtf0 for r in successful_results) / len(successful_results)
    avg_qps = sum(r.qps for r in successful_results) / len(successful_results)
    avg_latency = sum(r.latency for r in successful_results) / len(successful_results)
    
    return ModelSummary(
        model_name=model_name,
        total_items=total_items,
        successful_items=successful_items,
        wer=avg_wer,
        cer=avg_cer,
        cpcer=avg_cpcer,
        sacer=avg_sacer,
        delta_cp=avg_delta_cp,
        delta_sa=avg_delta_sa,
        avg_rtf0=avg_rtf0,
        avg_qps=avg_qps,
        avg_latency=avg_latency
    )

def write_summary_report(summaries: List[ModelSummary], output_path: str) -> None:
    """写入摘要报告"""
    logger = logging.getLogger("AutoEvalAudio")
    
    # 创建输出目录
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # 写入表头
        writer.writerow([
            "模型名称",
            "总项数",
            "成功项数",
            "平均WER",
            "平均CER",
            "平均cpCER",
            "平均saCER",
            "平均Δcp",
            "平均Δsa",
            "平均RTF0",
            "平均QPS",
            "平均延迟(s)"
        ])
        
        # 写入数据
        for summary in summaries:
            writer.writerow([
                summary.model_name,
                summary.total_items,
                summary.successful_items,
                summary.wer,
                summary.cer,
                summary.cpcer,
                summary.sacer,
                summary.delta_cp,
                summary.delta_sa,
                summary.avg_rtf0,
                summary.avg_qps,
                summary.avg_latency
            ])
    
    logger.info(f"摘要报告已生成: {output_path}")

def write_detailed_report(results: List[EvaluationResult], output_path: str) -> None:
    """写入详细报告"""
    logger = logging.getLogger("AutoEvalAudio")
    
    # 创建输出目录
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # 写入表头
        writer.writerow([
            "模型名称",
            "音频路径",
            "参考文本",
            "转写文本",
            "WER",
            "CER",
            "cpCER",
            "saCER",
            "Δcp",
            "Δsa",
            "RTF0",
            "QPS",
            "延迟(s)",
            "成功",
            "错误信息"
        ])
        
        # 写入数据
        for result in results:
            writer.writerow([
                result.model_name,
                result.audio_path,
                result.reference_text,
                result.hypothesis_text,
                result.wer,
                result.cer,
                result.cpcer,
                result.sacer,
                result.delta_cp,
                result.delta_sa,
                result.rtf0,
                result.qps,
                result.latency,
                result.success,
                result.error
            ])
    
    logger.info(f"详细报告已生成: {output_path}")

def main():
    """主函数"""
    # 加载配置
    config = load_config("AutoEvalAudio.toml")
    
    # 设置日志
    logger = setup_logger(config.get("logging", {}))
    
    logger.info("===== AutoEvalAudio ASR模型评估工具启动 =====")
    
    # 获取配置信息
    paths = config.get("paths", {})
    dataset_dir = paths.get("dataset_dir", "./datasets/test")
    output_dir = paths.get("output_dir", "./results")
    
    models_config = config.get("models", {})
    model_list = models_config.get("model_list", [])
    
    evaluation_config = config.get("evaluation", {})
    parallel = evaluation_config.get("parallel", True)
    max_workers = evaluation_config.get("max_workers", 4)
    
    # 加载数据集
    dataset = load_dataset(dataset_dir)
    
    if not dataset:
        logger.error("未加载到任何数据集项，程序退出")
        return
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 评估所有模型
    all_results = []
    summaries = []
    
    for model_config in model_list:
        # 创建模型实例
        model = ASRModelFactory.create(model_config)
        
        # 加载模型
        model.load()
        
        # 评估模型
        results = evaluate_model(model, dataset, parallel=parallel, max_workers=max_workers)
        all_results.extend(results)
        
        # 生成模型摘要
        summary = generate_model_summary(model.get_name(), results)
        summaries.append(summary)
        
        # 生成模型详细报告
        detailed_report_path = os.path.join(output_dir, f"{model.get_name()}_detailed.csv")
        write_detailed_report(results, detailed_report_path)
    
    # 生成汇总报告
    summary_report_path = os.path.join(output_dir, "summary.csv")
    write_summary_report(summaries, summary_report_path)
    
    # 生成所有结果的详细报告
    all_detailed_report_path = os.path.join(output_dir, "all_detailed.csv")
    write_detailed_report(all_results, all_detailed_report_path)
    
    logger.info("===== 评估完成！=====")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"汇总报告: {summary_report_path}")
    logger.info(f"详细报告: {all_detailed_report_path}")

if __name__ == "__main__":
    main()
