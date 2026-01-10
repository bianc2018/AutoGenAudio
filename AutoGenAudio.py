#!/usr/bin/env python3
"""
AutoGenAudio - 多后端TTS批量生成器
最终版：支持角色音色配置、智能分段和会话音色一致性
"""

import os
import json
import csv
import torch
import torchaudio
import time
import sys
import random
import tempfile
import shutil
import atexit
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from abc import ABC, abstractmethod
from tqdm import tqdm
import logging
import warnings
import re
import glob
import inspect

warnings.filterwarnings('ignore')

# ---------------- 日志配置 ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("AutoGenAudio")

# ---------------- 读取 config.toml ----------------
try:
    import tomli as tomllib
except ModuleNotFoundError:
    import tomllib

with open("AutoGenAudio.toml", "rb") as f:
    CFG = tomllib.load(f)

MODEL_CFG      = CFG.get("model", {})
BACKEND        = MODEL_CFG.get("backend", "fireredtts2")
SCRIPTS_DIR    = CFG["paths"]["scripts_dir"]
OUT_DIR        = CFG["paths"]["output_dir"]
DEVICE         = CFG["device"]["device"]
CHECKPOINT     = "checkpoint.txt"
SAMPLE_RATE    = 24_000

# 参考音频配置
REF_SECS    = 8
REF_TEMP    = 0.4
REF_TOPK    = 10
MAX_SENTENCES = MODEL_CFG.get("audio_ref", {}).get("max_sentences", 20)

# ---------------- 模型基类 ----------------
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

# ---------------- 文件扫描工具函数 ----------------
def find_model_file(directory: Path, patterns: List[str], model_name: str) -> Optional[Path]:
    """自动扫描匹配的模型文件"""
    for pattern in patterns:
        matches = list(directory.glob(pattern))
        if matches:
            return sorted(matches, key=lambda x: x.stat().st_mtime, reverse=True)[0]
    return None

def find_voice_files(directory: Path, extensions: List[str] = [".pt", ".pth"]) -> List[Path]:
    """扫描音色文件"""
    voices = []
    for ext in extensions:
        voices.extend(directory.glob(f"*{ext}"))
    return voices

# ---------------- 文本工具函数 ----------------
def clean_text_for_kokoro(text: str) -> str:
    """清理文本以适应 Kokoro 模型"""
    if not text or not isinstance(text, str):
        return "..."
    
    # 移除 IPA 音标格式: /.../
    text = re.sub(r'\/[^\/]+\/', '', text)
    
    # 合并多个空格
    text = ' '.join(text.split())
    
    # 确保句子结尾有标点
    text = text.strip()
    if text and text[-1] not in '.。!！?？;；:：,，':
        text += '.'
    
    # 限制长度
    max_length = 500
    if len(text) > max_length:
        text = text[:max_length]
        if text[-1] not in '.。!！?？;；:：,，':
            last_punct = max(text.rfind(','), text.rfind('.'), text.rfind('，'), text.rfind('。'))
            if last_punct > len(text) * 0.8:
                text = text[:last_punct + 1]
            else:
                text += '.'
    
    return text

def extract_content_from_dialogue(text: str) -> str:
    """
    从对话文本中提取纯内容，移除角色标记
    
    输入: "[角色]文本内容"
    输出: "文本内容"
    """
    match = re.match(r'\[(.*?)\](.*)', text)
    if match:
        return match.group(2).strip()
    return text.strip()

def extract_role_from_dialogue(text: str) -> str:
    """
    从对话文本中提取角色名
    
    输入: "[角色]文本内容"
    输出: "角色"
    """
    match = re.match(r'\[(.*?)\]', text)
    if match:
        return match.group(1).strip()
    return "default"

# ---------------- FireRedTTS2 适配器 ----------------
class FireRedTTS2Adapter(BaseTTSModel):
    def __init__(self, model_config: dict, device: str):
        super().__init__(model_config, device)
        self.pretrained_dir = Path(model_config.get("pretrained_dir", "./pretrained_models/fireredtts2"))
        self.model_version = model_config.get("model_version", "v1.0.1")
        
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
    
    def get_voice_for_role(self, role: str, voice_cache: Dict[str, Path]) -> Optional[Path]:
        """FireRedTTS2 使用参考音频，不需要音色文件"""
        return None

# ---------------- Kokoro 适配器 ----------------
class KokoroAdapter(BaseTTSModel):
    def __init__(self, model_config: dict, device: str):
        super().__init__(model_config, device)
        self.pretrained_dir = Path(model_config.get("pretrained_dir", "./pretrained_models/kokoro"))
        self.model_file = model_config.get("model_file", "kokoro-v0_19.pth")
        self.lang_code = model_config.get("lang_code", "a")
        self.voice_mapping = model_config.get('voice_mapping', {})
        
    def load(self) -> None:
        """加载 Kokoro 模型"""
        try:
            from kokoro import KPipeline
        except ImportError:
            logger.error("未安装 Kokoro 库。请运行: pip install kokoro soundfile")
            raise
        
        # 查找模型文件
        model_path = self.pretrained_dir / self.model_file
        
        if not model_path.exists():
            logger.error(f"模型文件不存在: {model_path}")
            possible_models = find_model_file(self.pretrained_dir, ["*.pth", "*.pt"], "Kokoro")
            if possible_models:
                logger.info(f"找到其他模型文件: {possible_models.name}")
                logger.info(f"可在配置中指定: model_file = \"{possible_models.name}\"")
            sys.exit(1)
        
        # 扫描音色目录
        voices_dir = self.pretrained_dir / "voices"
        if not voices_dir.exists():
            possible_voice_dirs = list(self.pretrained_dir.glob("*/voices"))
            if possible_voice_dirs:
                voices_dir = possible_voice_dirs[0]
                logger.info(f"找到音色目录: {voices_dir}")
            else:
                voice_files = find_voice_files(self.pretrained_dir)
                if voice_files:
                    voices_dir = self.pretrained_dir
                    logger.info(f"在预训练目录找到音色文件，使用: {voices_dir}")
                else:
                    logger.error(f"音色目录不存在: {voices_dir}")
                    sys.exit(1)
        
        voice_files = find_voice_files(voices_dir)
        if not voice_files:
            logger.error(f"未找到音色文件: {voices_dir}")
            sys.exit(1)
        
        logger.info(f"正在加载 Kokoro 模型:")
        logger.info(f"  模型文件: {self.model_file}")
        logger.info(f"  语言代码: {self.lang_code}")
        logger.info(f"  音色数量: {len(voice_files)}")
        logger.info(f"  角色音色映射: {self.voice_mapping}")
        t0 = time.time()
        
        # 初始化 KPipeline
        try:
            self.model = KPipeline(
                lang_code=self.lang_code,
                device=self.device
            )
            
            # 加载模型权重
            if hasattr(self.model, 'load_ckpt'):
                self.model.load_ckpt(str(model_path))
                logger.debug("使用 load_ckpt 加载模型")
            elif hasattr(self.model, 'load_model'):
                self.model.load_model(str(model_path))
                logger.debug("使用 load_model 加载模型")
            else:
                logger.debug("KPipeline 将使用默认模型加载方式")
        
        except Exception as e:
            logger.error(f"加载 Kokoro 模型失败: {e}")
            logger.error(f"模型路径: {model_path}")
            logger.error("请检查模型文件是否正确，或尝试更新 kokoro 库")
            sys.exit(1)
        
        self.voices_dir = voices_dir
        self.voice_files = voice_files
        
        elapsed = time.time() - t0
        logger.info(f"Kokoro 模型加载完成，耗时 {elapsed:.2f}s")
        self.is_loaded = True
    
    def generate_dialogue(self, text_list: List[str], prompt_wav_list: Optional[List[str]], 
                         prompt_text_list: Optional[List[str]], temperature: float, topk: int) -> torch.Tensor:
        """生成对话"""
        if not self.is_loaded:
            raise RuntimeError("模型未加载，请先调用 load()")
        
        # 会话级音色缓存：role -> voice_path
        voice_cache = {}
        
        audio_chunks = []
        for i, text in enumerate(text_list):
            logger.debug(f"Kokoro 生成 {i+1}/{len(text_list)}: {text[:50]}...")
            
            # 提取角色和内容
            role = extract_role_from_dialogue(text)
            content = extract_content_from_dialogue(text)
            
            # 获取角色的音色（带缓存）
            voice_path = self.get_voice_for_role(role, voice_cache)
            if not voice_path:
                logger.warning(f"角色 {role} 未找到音色文件，使用默认")
                voice_path = self.voice_files[0] if self.voice_files else None
            
            # 清理文本
            content = clean_text_for_kokoro(content)
            logger.debug(f"清理后的文本: {content}")
            logger.debug(f"角色 {role} 使用音色: {voice_path.name if voice_path else '默认'}")
            
            # 生成音频（Kokoro 返回生成器）
            try:
                generator = self.model(
                    content,
                    voice=str(voice_path) if voice_path else None,
                    speed=1.0
                )
                
                # 收集所有音频片段
                chunk_audios = []
                for gs, ps, audio in generator:
                    if audio is not None and len(audio) > 0:
                        # 转换为 tensor
                        if isinstance(audio, np.ndarray):
                            audio_tensor = torch.from_numpy(audio).float()
                        else:
                            audio_tensor = torch.tensor(audio, dtype=torch.float32)
                        
                        # 确保二维
                        if audio_tensor.ndim == 1:
                            audio_tensor = audio_tensor.unsqueeze(0)
                        
                        chunk_audios.append(audio_tensor)
                
                if chunk_audios:
                    # 拼接当前句子的所有片段
                    sentence_audio = torch.cat(chunk_audios, dim=-1)
                    audio_chunks.append(sentence_audio)
                else:
                    logger.warning(f"生成音频为空，跳过此句: {content[:30]}...")
                    audio_chunks.append(torch.zeros(1, SAMPLE_RATE))
                
            except Exception as e:
                logger.error(f"生成失败 '{content[:30]}...': {e}")
                audio_chunks.append(torch.zeros(1, SAMPLE_RATE))
        
        # 拼接所有音频
        if audio_chunks:
            return torch.cat(audio_chunks, dim=-1)
        else:
            logger.error("所有音频生成失败，返回静音")
            return torch.zeros(1, SAMPLE_RATE)
    
    def generate(self, text: str, speaker: str, context: list, max_audio_length_ms: int, 
                temperature: float, topk: int) -> torch.Tensor:
        """生成单句参考音频"""
        if not self.is_loaded:
            raise RuntimeError("模型未加载，请先调用 load()")
        
        # 清理文本
        text = clean_text_for_kokoro(text)
        logger.debug(f"生成参考音频: {text}")
        
        # 使用第一个音色文件作为默认
        voice_path = self.voice_files[0] if self.voice_files else None
        
        try:
            # 生成音频（Kokoro 返回生成器）
            generator = self.model(
                text,
                voice=str(voice_path) if voice_path else None,
                speed=1.0
            )
            
            # 收集所有音频片段
            chunk_audios = []
            for gs, ps, audio in generator:
                if audio is not None and len(audio) > 0:
                    # 转换为 tensor
                    if isinstance(audio, np.ndarray):
                        audio_tensor = torch.from_numpy(audio).float()
                    else:
                        audio_tensor = torch.tensor(audio, dtype=torch.float32)
                    
                    # 确保二维
                    if audio_tensor.ndim == 1:
                        audio_tensor = audio_tensor.unsqueeze(0)
                    
                    chunk_audios.append(audio_tensor)
            
            if chunk_audios:
                # 拼接所有片段
                audio = torch.cat(chunk_audios, dim=-1)
            else:
                logger.warning("生成的参考音频为空")
                return torch.zeros(1, int(max_audio_length_ms * SAMPLE_RATE / 1000))
            
            # 裁剪到指定长度
            max_length = int(max_audio_length_ms * SAMPLE_RATE / 1000)
            if audio.shape[-1] > max_length:
                audio = audio[..., :max_length]
            
            return audio
            
        except Exception as e:
            logger.error(f"生成参考音频失败 '{text[:30]}...': {e}")
            return torch.zeros(1, int(max_audio_length_ms * SAMPLE_RATE / 1000))
    
    def get_voice_for_role(self, role: str, voice_cache: Dict[str, Path]) -> Optional[Path]:
        """
        根据角色名获取音色文件路径（带缓存，确保同一会话中音色一致）
        
        Args:
            role: 角色名
            voice_cache: 会话级音色缓存字典 {role: voice_path}
        
        Returns:
            Optional[Path]: 音色文件路径
        """
        # 检查缓存中是否已有该角色的音色
        if role in voice_cache:
            cached_voice = voice_cache[role]
            logger.debug(f"角色 {role} 使用缓存音色: {cached_voice.name}")
            return cached_voice
        
        # 加载角色音色映射配置
        voice_mapping = getattr(self, 'voice_mapping', {})
        
        selected_voice = None
        
        if role in voice_mapping and voice_mapping[role]:
            # 角色有配置音色列表，随机选择一个
            # 注意：这里随机选择一次后，后续调用会使用缓存
            voice_name = random.choice(voice_mapping[role])
            # 查找带扩展名的音色文件
            for ext in [".pt", ".pth"]:
                voice_path = self.voices_dir / f"{voice_name}{ext}"
                if voice_path.exists():
                    selected_voice = voice_path
                    break
            
            if selected_voice:
                logger.debug(f"角色 {role} 配置音色选择: {selected_voice.name}")
            else:
                logger.warning(f"角色 {role} 配置的音色列表 {voice_mapping[role]} 均未找到")
        
        # 如果未找到配置音色或角色未配置，从所有音色中随机
        if not selected_voice and self.voice_files:
            selected_voice = random.choice(self.voice_files)
            logger.debug(f"角色 {role} 随机选择音色: {selected_voice.name}")
        
        # 将选择的音色存入缓存
        if selected_voice:
            voice_cache[role] = selected_voice
        
        return selected_voice

# ---------------- Higgs 适配器 ----------------
class HiggsAdapter(BaseTTSModel):
    def __init__(self, model_config: dict, device: str):
        super().__init__(model_config, device)
        self.pretrained_dir = Path(model_config.get("pretrained_dir", "./pretrained_models/higgs"))
        self.model_version = model_config.get("model_version", "v2_base")
        
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

# ---------------- VibeVoice 适配器 ----------------
class VibeVoiceAdapter(BaseTTSModel):
    def __init__(self, model_config: dict, device: str):
        super().__init__(model_config, device)
        self.pretrained_dir = Path(model_config.get("pretrained_dir", "./pretrained_models/vibevoice"))
        self.model_version = model_config.get("model_version", "1.5b")
        
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

# ---------------- 模型工厂 ----------------
class ModelFactory:
    @staticmethod
    def create() -> BaseTTSModel:
        """根据配置创建模型实例"""
        backend = BACKEND
        model_config = MODEL_CFG.get(backend, {})
        pretrained_dir = Path(model_config.get("pretrained_dir", f"./pretrained_models/{backend}"))
        
        # 检查模型是否存在
        if not pretrained_dir.exists():
            logger.error(f"模型目录不存在: {pretrained_dir}")
            logger.error("请手动下载模型并放置在正确位置")
            _print_download_guide(backend, pretrained_dir)
            sys.exit(1)
        
        # 返回适配器
        if backend == "fireredtts2":
            return FireRedTTS2Adapter(model_config, DEVICE)
        elif backend == "kokoro":
            return KokoroAdapter(model_config, DEVICE)
        elif backend == "higgs":
            return HiggsAdapter(model_config, DEVICE)
        elif backend == "vibevoice":
            return VibeVoiceAdapter(model_config, DEVICE)
        else:
            raise ValueError(f"不支持的模型后端: {backend}")

def _print_download_guide(backend: str, save_dir: Path):
    """打印模型下载指南"""
    logger.info("=" * 60)
    logger.info("手动下载指南：")
    logger.info("=" * 60)
    
    guides = {
        "fireredtts2": "huggingface-cli download FireRedTeam/FireRedTTS2 --local-dir ./pretrained_models/fireredtts2",
        "kokoro": "huggingface-cli download hexgrad/Kokoro-82M --local-dir ./pretrained_models/kokoro",
        "higgs": "huggingface-cli download SWivid/Higgs-Audio --local-dir ./pretrained_models/higgs",
        "vibevoice": "huggingface-cli download vibevoice/VibeVoice-1.5B --local-dir ./pretrained_models/vibevoice"
    }
    
    if backend in guides:
        logger.info(f"运行命令:")
        logger.info(f"  {guides[backend]}")
    else:
        logger.info(f"请查看 {backend} 的官方文档获取下载方式")
    
    logger.info(f"模型将保存到: {save_dir}")
    logger.info("下载完成后，请确保必要的配置文件存在")
    logger.info("=" * 60)

# ---------------- 工具函数 ----------------
def load_json(p: str):
    for enc in ('utf-8', 'gbk', 'utf-16'):
        try:
            with open(p, 'r', encoding=enc) as f:
                return json.load(f)
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    raise RuntimeError(f'无法解析 {p}')

def load_refs(cfg_dir: str) -> dict:
    refs = {}
    ref_dir = Path(cfg_dir) / 'refs'
    if not ref_dir.exists():
        logger.info("未找到 refs/ 目录，将采用随机音色或生成临时参考")
        return {}
    
    for fname in os.listdir(ref_dir):
        base, ext = os.path.splitext(fname)
        if ext.lower() != '.wav':
            continue
        speaker = base.split('_ref')[0]
        wav_path = ref_dir / fname
        txt_path = ref_dir / f"{base}.txt"
        if txt_path.exists():
            with open(txt_path, 'r', encoding='utf-8') as f:
                refs[speaker] = {'wav': str(wav_path), 'txt': f.read().strip()}
    
    logger.info(f"加载固定参考完成，共 {len(refs)} 个角色")
    return refs

def ensure_dir(d: str):
    Path(d).mkdir(parents=True, exist_ok=True)

def get_resume_point(cfg_dir: str) -> int:
    if not Path(CHECKPOINT).exists():
        return 0
    with open(CHECKPOINT, 'r', encoding='utf-8') as f:
        name = f.read().strip()
    scripts = sorted([f for f in Path(cfg_dir).iterdir() if f.name.endswith('.json')])
    try:
        return scripts.index(Path(name))
    except ValueError:
        return 0

def save_checkpoint(name: str):
    with open(CHECKPOINT, 'w', encoding='utf-8') as f:
        f.write(name)

def save_temp_tensor(speaker: str, tensor: torch.Tensor) -> str:
    if not hasattr(save_temp_tensor, 'temp_dir'):
        save_temp_tensor.temp_dir = Path(tempfile.mkdtemp(prefix="autogen_"))
        atexit.register(lambda: shutil.rmtree(save_temp_tensor.temp_dir, ignore_errors=True))
    
    path = save_temp_tensor.temp_dir / f"{speaker}_ref.wav"
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    torchaudio.save(str(path), tensor, SAMPLE_RATE)
    logger.debug(f"临时参考落盘: {path}")
    return str(path)

def get_ref_path(speaker: str, wav_field) -> str:
    if isinstance(wav_field, torch.Tensor):
        return save_temp_tensor(speaker, wav_field)
    return str(wav_field)

# ---------------- 临时参考生成 ----------
def generate_temp_ref(model: BaseTTSModel, speaker: str, text: str) -> torch.Tensor:
    if not model.is_clone_supported():
        return torch.zeros(int(REF_SECS * SAMPLE_RATE))
    
    logger.debug(f"生成临时参考: speaker={speaker}, text={text[:30]}...")
    with torch.no_grad():
        audio = model.generate(
            text=text,
            speaker=speaker,
            context=[],
            max_audio_length_ms=int(REF_SECS * 1000),
            temperature=REF_TEMP,
            topk=REF_TOPK
        )
    return audio.cpu()

def build_temp_refs(model: BaseTTSModel, script: list, refs: dict, ref_dir: str) -> Dict[str, dict]:
    if not model.is_clone_supported():
        logger.info(f"当前模型 {BACKEND} 不支持音色克隆，跳过参考生成")
        return {}
    
    SAVE_TEMP_TO_REFS = MODEL_CFG.get("audio_ref", {}).get("save_temp_to_refs", True)
    temp_refs = {}
    roles_in_script = {utt['role'] for utt in script}
    
    for role in roles_in_script:
        if role in refs:
            continue
        
        if not SAVE_TEMP_TO_REFS:
            wav_path = Path(ref_dir) / f"{role}_ref.wav"
            txt_path = Path(ref_dir) / f"{role}_ref.txt"
            if wav_path.exists() and txt_path.exists():
                refs[role] = {'wav': str(wav_path), 'txt': txt_path.read_text(encoding='utf-8').strip()}
                logger.info(f"复用已有临时参考: {role}")
                continue
        
        cand_texts = [utt['text'] for utt in script if utt['role'] == role]
        if not cand_texts:
            continue
        
        txt = random.choice(cand_texts)
        wav_tensor = generate_temp_ref(model, role, txt)
        
        if SAVE_TEMP_TO_REFS:
            temp_refs[role] = {'wav': wav_tensor, 'txt': f"[{role}]{txt}"}
            logger.info(f"生成内存临时参考: {role}")
        else:
            ensure_dir(ref_dir)
            wav_path = Path(ref_dir) / f"{role}_ref.wav"
            txt_path = Path(ref_dir) / f"{role}_ref.txt"
            if wav_tensor.ndim == 1:
                wav_tensor = wav_tensor.unsqueeze(0)
            torchaudio.save(str(wav_path), wav_tensor, SAMPLE_RATE)
            txt_path.write_text(f"[{role}]{txt}", encoding='utf-8')
            refs[role] = {'wav': str(wav_path), 'txt': f"[{role}]{txt}"}
            logger.info(f"写入固定临时参考: {wav_path}")
    
    return temp_refs

# ---------------- 主流程 ----------
def AutoGenAudio(cfg_dir: str, out_root: str, device: str):
    logger.info("===== AutoGenAudio 批量对话生成启动 =====")
    
    model = ModelFactory.create()
    model.load()
    
    refs = load_refs(cfg_dir) if model.is_clone_supported() else {}
    
    scripts = sorted([f for f in os.listdir(cfg_dir) if f.endswith('.json')])
    if not scripts:
        logger.warning("目录内无 json 脚本，程序结束")
        return

    start_idx = get_resume_point(cfg_dir)
    logger.info(f"断点续跑：从第 {start_idx+1}/{len(scripts)} 个脚本开始")

    pbar_outer = tqdm(total=len(scripts), initial=start_idx,
                      desc='总进度', unit='脚本', dynamic_ncols=True)

    for idx in range(start_idx, len(scripts)):
        script_file = scripts[idx]
        name = os.path.splitext(script_file)[0]
        save_checkpoint(name)
        pbar_outer.set_postfix_str(f'当前：{name}')

        script = load_json(os.path.join(cfg_dir, script_file))
        ref_dir = os.path.join(cfg_dir, 'refs')
        ensure_dir(ref_dir)

        temp_refs = build_temp_refs(model, script, refs, ref_dir)

        t0 = time.time()
        audio = generate_long_dialogue(model, script, refs, temp_refs)
        elapsed = time.time() - t0
        logger.info(f"脚本 {name} 合成完成，耗时 {elapsed:.2f}s")

        ensure_dir(out_root)
        wav_path = os.path.join(out_root, f"{name}.wav")
        torchaudio.save(wav_path, audio, SAMPLE_RATE)
        csv_path = os.path.join(out_root, f"{name}.csv")
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for utt in script:
                writer.writerow(['', '', utt['role'], utt['text']])
        logger.info(f"已保存：{wav_path} + {csv_path}")

        pbar_outer.set_postfix_str(f'{name} 耗时 {elapsed:.1f}s')
        pbar_outer.update()

    pbar_outer.close()
    if Path(CHECKPOINT).exists():
        Path(CHECKPOINT).unlink()
    logger.info("===== 全部完成！=====")

# ---------------- 长对话合成 ----------
def generate_long_dialogue(model: BaseTTSModel, script: list, refs: dict, temp_refs: dict):
    # 智能分段：仅 FireRedTTS2 需要分段
    if BACKEND == "fireredtts2":
        chunks = optimal_chunks(script, MAX_SENTENCES)
        logger.info(f"FireRedTTS2 启用分段：{len(chunks)} 段")
    else:
        chunks = [script]
        logger.info(f"{BACKEND} 无需分段，直接合成")
    
    audio_chunks = []
    all_refs = {**refs, **temp_refs}
    
    for i, chunk in enumerate(chunks, 1):
        # 构建带角色标记的文本列表
        texts = [f"[{utt['role']}]{utt['text']}" for utt in chunk]
        
        # FireRedTTS2 使用参考音频，其他模型在内部处理音色选择
        if BACKEND == "fireredtts2":
            unique_roles = list(dict.fromkeys(utt['role'] for utt in chunk))
            wavs = [get_ref_path(role, all_refs[role]['wav']) for role in unique_roles if role in all_refs]
            txts = [all_refs[role]['txt'] for role in unique_roles if role in all_refs]
        else:
            wavs = None
            txts = None
        
        logger.debug(f"合成第 {i}/{len(chunks)} 段：{len(chunk)} 句")

        with torch.no_grad():
            audio = model.generate_dialogue(
                text_list=texts,
                prompt_wav_list=wavs,
                prompt_text_list=txts,
                temperature=0.75,
                topk=30
            )
        audio_chunks.append(audio.cpu())
        logger.info(f"段 {i}/{len(chunks)} 完成（{len(chunk)} 句）")

    return torch.cat(audio_chunks, dim=-1)

# ---------------- 分段合成 ----------
def optimal_chunks(script: list, max_sentences: int = MAX_SENTENCES) -> list:
    n = len(script)
    if n <= max_sentences:
        logger.info("整段未超，直接合成")
        return [script]
    chunks = [script[i:i + max_sentences] for i in range(0, n, max_sentences)]
    logger.info(f"分段完成：{len(chunks)} 段，共 {n} 句，每段 ≤ {max_sentences} 句")
    return chunks

if __name__ == '__main__':
    AutoGenAudio(SCRIPTS_DIR, OUT_DIR, DEVICE)
    input("运行完毕，按 Enter 键退出...")