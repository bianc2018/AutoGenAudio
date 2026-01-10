from pathlib import Path
from typing import List, Optional, Dict
import torch
import sys
import time
import logging
import numpy as np
from .base import BaseTTSModel

# 日志配置
logger = logging.getLogger("AutoGenAudio")


# ---------------- 文本工具函数 ----------------
def extract_content_from_dialogue(text: str) -> str:
    """
    从对话文本中提取纯内容，移除角色标记
    
    输入: "[角色]文本内容"
    输出: "文本内容"
    """
    import re
    match = re.match(r'\[(.*?)\](.*)', text)
    if match:
        return match.group(2).strip()
    return text.strip()


class HiggsAdapter(BaseTTSModel):
    def __init__(self, model_config: dict, device: str):
        super().__init__(model_config, device)
        self.pretrained_dir = Path(model_config.get("pretrained_dir", "./pretrained_models/higgs"))
        self.model_version = model_config.get("model_version", "v2_base")
    
    def is_clone_supported(self) -> bool:
        """Higgs 模型不支持语音克隆"""
        return False
        
    def load(self) -> None:
        """加载 Higgs 模型"""
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError:
            logger.error("未安装 transformers 库。请运行: pip install transformers accelerate")
            raise
        
        # 检查配置文件
        config_path = self.pretrained_dir / "config.json"
        if not config_path.exists():
            logger.error(f"配置文件不存在: {config_path}")
            sys.exit(1)
        
        logger.info(f"正在加载 Higgs 模型:")
        logger.info(f"  版本: {self.model_version}")
        logger.info(f"  目录: {self.pretrained_dir}")
        t0 = time.time()
        
        # 加载模型和分词器
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.pretrained_dir))
        self.model = AutoModel.from_pretrained(
            str(self.pretrained_dir),
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map=self.device if self.device == "cuda" else None
        )
        self.model.eval()
        
        elapsed = time.time() - t0
        logger.info(f"Higgs 模型加载完成，耗时 {elapsed:.2f}s")
        self.is_loaded = True
    
    def generate_dialogue(self, text_list: List[str], prompt_wav_list: Optional[List[str]], 
                         prompt_text_list: Optional[List[str]], temperature: float, topk: int) -> torch.Tensor:
        """生成对话"""
        if not self.is_loaded:
            raise RuntimeError("模型未加载，请先调用 load()")
        
        audio_chunks = []
        for text in text_list:
            # 提取纯内容
            content = extract_content_from_dialogue(text)
            
            # Tokenize
            inputs = self.tokenizer(content, return_tensors="pt").to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=inputs.input_ids.shape[1] + 500,
                    temperature=temperature,
                    top_k=topk,
                    do_sample=True
                )
            
            # 简化为正弦波作为示例
            audio_tokens = outputs[0].cpu().numpy()
            sample_rate = 24000
            duration = len(audio_tokens) / sample_rate
            t = np.linspace(0, duration, len(audio_tokens))
            audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
            
            audio_chunks.append(torch.from_numpy(audio).unsqueeze(0).float())
        
        return torch.cat(audio_chunks, dim=-1)
    
    def generate(self, text: str, speaker: str, context: list, max_audio_length_ms: int, 
                temperature: float, topk: int) -> torch.Tensor:
        """生成单句参考音频"""
        if not self.is_loaded:
            raise RuntimeError("模型未加载，请先调用 load()")
        
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=inputs.input_ids.shape[1] + 300,
                temperature=temperature,
                top_k=topk,
                do_sample=True
            )
        
        # 简化为正弦波示例
        audio_tokens = outputs[0].cpu().numpy()
        sample_rate = 24000
        max_length = int(max_audio_length_ms * sample_rate / 1000)
        
        if len(audio_tokens) > max_length:
            audio_tokens = audio_tokens[:max_length]
        
        duration = len(audio_tokens) / sample_rate
        t = np.linspace(0, duration, len(audio_tokens))
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        
        return torch.from_numpy(audio).float()
    
    def get_voice_for_role(self, role: str, voice_cache: Dict[str, Path]) -> Optional[Path]:
        """Higgs 模型当前不支持音色选择"""
        return None
