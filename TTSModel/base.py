from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from pathlib import Path
import torch


class BaseTTSModel(ABC):
    def __init__(self, model_config: dict, device: str):
        self.model_config = model_config
        self.device = device
        self.model = None
        self.is_loaded = False
    
    @abstractmethod
    def load(self) -> None:
        """加载模型"""
        pass
    
    @abstractmethod
    def generate_dialogue(self, text_list: List[str], prompt_wav_list: Optional[List[str]], 
                         prompt_text_list: Optional[List[str]], temperature: float, topk: int) -> torch.Tensor:
        """生成对话音频"""
        pass
    
    @abstractmethod
    def generate(self, text: str, speaker: str, context: list, max_audio_length_ms: int, 
                temperature: float, topk: int) -> torch.Tensor:
        """生成单句（用于参考音频）"""
        pass
    
    @abstractmethod
    def get_voice_for_role(self, role: str, voice_cache: Dict[str, Path]) -> Optional[Path]:
        """
        根据角色名获取音色文件路径（带缓存）
        
        Args:
            role: 角色名
            voice_cache: 会话级音色缓存字典 {role: voice_path}
        
        Returns:
            Optional[Path]: 音色文件路径
        """
        pass
    
    def is_clone_supported(self) -> bool:
        """是否支持音色克隆"""
        return True
