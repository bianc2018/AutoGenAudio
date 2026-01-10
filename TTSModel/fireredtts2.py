from pathlib import Path
from typing import List, Optional
import torch
import sys
import time
import logging
from .base import BaseTTSModel

# 日志配置
logger = logging.getLogger("AutoGenAudio")


def find_model_file(directory: Path, patterns: List[str], model_name: str) -> Optional[Path]:
    """自动扫描匹配的模型文件"""
    for pattern in patterns:
        matches = list(directory.glob(pattern))
        if matches:
            return sorted(matches, key=lambda x: x.stat().st_mtime, reverse=True)[0]
    return None


class FireRedTTS2Adapter(BaseTTSModel):
    def __init__(self, model_config: dict, device: str):
        super().__init__(model_config, device)
        self.pretrained_dir = Path(model_config.get("pretrained_dir", "./pretrained_models/fireredtts2"))
        self.model_version = model_config.get("model_version", "v1.0.1")
    
    def is_clone_supported(self) -> bool:
        """FireRedTTS2 模型支持语音克隆"""
        return True
        
    def load(self) -> None:
        """加载 FireRedTTS2 模型"""
        try:
            from fireredtts2.fireredtts2 import FireRedTTS2
        except ImportError:
            logger.error("未安装 FireRedTTS2 库。请运行: pip install fireredtts2")
            raise
        
        # 自动查找配置文件
        config_patterns = ["config.yaml", "config.json", "*.yaml", "*.json"]
        config_path = find_model_file(self.pretrained_dir, config_patterns, "FireRedTTS2")
        
        if not config_path:
            logger.error(f"未找到配置文件: {self.pretrained_dir}")
            logger.error(f"尝试查找: {', '.join(config_patterns)}")
            sys.exit(1)
        
        logger.info(f"正在加载 FireRedTTS2 模型，版本: {self.model_version}")
        logger.info(f"配置: {config_path}, 目录: {self.pretrained_dir}")
        t0 = time.time()
        
        self.model = FireRedTTS2(
            config_path=str(config_path),
            pretrained_dir=str(self.pretrained_dir),
            device=self.device
        )
        self.model.load()
        
        elapsed = time.time() - t0
        logger.info(f"FireRedTTS2 模型加载完成，耗时 {elapsed:.2f}s")
        self.is_loaded = True
    
    def generate_dialogue(self, text_list: List[str], prompt_wav_list: Optional[List[str]], 
                         prompt_text_list: Optional[List[str]], temperature: float, topk: int) -> torch.Tensor:
        """生成对话"""
        if not self.is_loaded:
            raise RuntimeError("模型未加载，请先调用 load()")
        
        return self.model.generate_dialogue(
            text_list=text_list,
            prompt_wav_list=prompt_wav_list,
            prompt_text_list=prompt_text_list,
            temperature=temperature,
            topk=topk
        )
    
    def generate(self, text: str, speaker: str, context: list, max_audio_length_ms: int, 
                temperature: float, topk: int) -> torch.Tensor:
        """生成单句参考音频"""
        if not self.is_loaded:
            raise RuntimeError("模型未加载，请先调用 load()")
        
        return self.model.generate(
            text=text,
            speaker=speaker,
            context=context,
            max_audio_length_ms=max_audio_length_ms,
            temperature=temperature,
            topk=topk
        )
    
    def get_voice_for_role(self, role: str, voice_cache: dict) -> Optional[Path]:
        """FireRedTTS2 使用参考音频，不需要音色文件"""
        return None
