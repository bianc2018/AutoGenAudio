from abc import ABC, abstractmethod
from typing import Dict, List, Any


class BaseASRModel(ABC):
    def __init__(self, model_config: dict):
        self.model_config = model_config
        self.is_loaded = False
    
    @abstractmethod
    def load(self) -> None:
        """加载模型"""
        pass
    
    @abstractmethod
    def transcribe(self, audio_path: str) -> dict:
        """音频转写，返回包含转写文本和时间信息的字典"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """获取模型名称"""
        pass
    
    def get_type(self) -> str:
        """获取模型类型"""
        return self.model_config.get("type", "unknown")
