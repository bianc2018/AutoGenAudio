from pathlib import Path
from typing import List, Optional, Dict
import torch
import sys
import time
import logging
import torchaudio
from .base import BaseTTSModel

# 日志配置
logger = logging.getLogger("AutoGenAudio")


# ---------------- 文件扫描工具函数 ----------------
def find_model_file(directory: Path, patterns: List[str], model_name: str) -> Optional[Path]:
    """自动扫描匹配的模型文件"""
    for pattern in patterns:
        matches = list(directory.glob(pattern))
        if matches:
            return sorted(matches, key=lambda x: x.stat().st_mtime, reverse=True)[0]
    return None


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


class VibeVoiceAdapter(BaseTTSModel):
    def __init__(self, model_config: dict, device: str):
        super().__init__(model_config, device)
        self.pretrained_dir = Path(model_config.get("pretrained_dir", "./pretrained_models/vibevoice"))
        self.model_version = model_config.get("model_version", "1.5b")
    
    def is_clone_supported(self) -> bool:
        """VibeVoice 模型支持语音克隆"""
        return True
        
    def load(self) -> None:
        """加载 VibeVoice 模型"""
        try:
            from vibevoice import VibeVoiceModel, VibeVoiceConfig
            import yaml
        except ImportError:
            logger.error("未安装 VibeVoice 库。请运行: pip install git+https://github.com/vibevoice/vibevoice.git")
            raise
        
        # 自动查找模型文件
        model_patterns = [
            f"vibevoice_{self.model_version}.pt",
            f"vibevoice-{self.model_version}.pt",
            "*.pt", "*.pth"
        ]
        model_path = find_model_file(self.pretrained_dir, model_patterns, "VibeVoice")
        
        if not model_path:
            logger.error(f"未找到模型文件: {self.pretrained_dir}")
            logger.error(f"尝试查找: {', '.join(model_patterns)}")
            sys.exit(1)
        
        # 自动查找配置文件
        config_patterns = ["config.yaml", "config.json", "*.yaml"]
        config_path = find_model_file(self.pretrained_dir, config_patterns, "VibeVoice")
        
        if not config_path:
            logger.error(f"未找到配置文件: {self.pretrained_dir}")
            sys.exit(1)
        
        logger.info(f"正在加载 VibeVoice 模型:")
        logger.info(f"  版本: {self.model_version}")
        logger.info(f"  模型: {model_path.name}")
        logger.info(f"  配置: {config_path.name}")
        t0 = time.time()
        
        # 加载配置
        if config_path.suffix == '.yaml':
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
        else:
            import json
            config_dict = json.loads(config_path.read_text())
        
        config = VibeVoiceConfig(**config_dict)
        
        # 初始化模型
        self.model = VibeVoiceModel(config)
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # 处理不同的 checkpoint 格式
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        elapsed = time.time() - t0
        logger.info(f"VibeVoice 模型加载完成，耗时 {elapsed:.2f}s")
        self.is_loaded = True
    
    def generate_dialogue(self, text_list: List[str], prompt_wav_list: Optional[List[str]], 
                         prompt_text_list: Optional[List[str]], temperature: float, topk: int) -> torch.Tensor:
        """生成对话"""
        if not self.is_loaded:
            raise RuntimeError("模型未加载，请先调用 load()")
        
        audio_chunks = []
        SAMPLE_RATE = 24000
        
        for text in text_list:
            # 提取内容
            content = extract_content_from_dialogue(text)
            
            # 准备参考音频（如果有）
            reference_audio = None
            if prompt_wav_list and len(prompt_wav_list) > 0:
                ref_path = prompt_wav_list[0]
                ref_audio, sr = torchaudio.load(ref_path)
                if sr != SAMPLE_RATE:
                    ref_audio = torchaudio.functional.resample(ref_audio, sr, SAMPLE_RATE)
                reference_audio = ref_audio.to(self.device)
            
            # 生成
            with torch.no_grad():
                audio = self.model.generate(
                    text=content,
                    reference_audio=reference_audio,
                    temperature=temperature,
                    top_k=topk
                )
            
            if audio.ndim == 1:
                audio = audio.unsqueeze(0)
            audio_chunks.append(audio.cpu())
        
        return torch.cat(audio_chunks, dim=-1)
    
    def generate(self, text: str, speaker: str, context: list, max_audio_length_ms: int, 
                temperature: float, topk: int) -> torch.Tensor:
        """生成单句参考音频"""
        if not self.is_loaded:
            raise RuntimeError("模型未加载，请先调用 load()")
        
        SAMPLE_RATE = 24000
        
        # 生成
        with torch.no_grad():
            audio = self.model.generate(
                text=text,
                reference_audio=None,
                temperature=temperature,
                top_k=topk,
                max_length=int(max_audio_length_ms * SAMPLE_RATE / 1000)
            )
        
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        
        # 裁剪到指定长度
        max_length = int(max_audio_length_ms * SAMPLE_RATE / 1000)
        if audio.shape[-1] > max_length:
            audio = audio[..., :max_length]
        
        return audio.cpu()
    
    def get_voice_for_role(self, role: str, voice_cache: Dict[str, Path]) -> Optional[Path]:
        """VibeVoice 当前不支持音色选择"""
        return None
