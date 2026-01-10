from pathlib import Path
from typing import List, Optional, Dict
import torch
import sys
import time
import logging
import re
import numpy as np
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


class KokoroAdapter(BaseTTSModel):
    def __init__(self, model_config: dict, device: str):
        super().__init__(model_config, device)
        self.pretrained_dir = Path(model_config.get("pretrained_dir", "./pretrained_models/kokoro"))
        self.model_file = model_config.get("model_file", "kokoro-v0_19.pth")
        self.lang_code = model_config.get("lang_code", "a")
        self.voice_mapping = model_config.get('voice_mapping', {})
    
    def is_clone_supported(self) -> bool:
        """Kokoro 模型不支持语音克隆"""
        return False
        
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
        
        # 初始化 KPipeline，手动创建KModel实例避免重复下载
        try:
            from kokoro.model import KModel
            
            # 1. 手动创建KModel实例，直接使用本地模型文件，不触发hf_hub_download
            logger.debug(f"手动创建KModel，使用本地模型文件: {model_path}")
            local_kmodel = KModel(
                repo_id='hexgrad/Kokoro-82M',  # 传递repo_id抑制警告
                model=str(model_path)  # 直接指定本地模型文件，避免下载
            )
            local_kmodel.to(self.device).eval()
            logger.debug("KModel 创建完成，已加载本地模型")
            
            # 2. 将本地KModel实例传递给KPipeline，避免它再次创建和下载
            self.model = KPipeline(
                lang_code=self.lang_code,
                device=self.device,
                model=local_kmodel,  # 传递已加载的本地KModel实例
                repo_id='hexgrad/Kokoro-82M'  # 传递repo_id抑制警告
            )
            logger.debug("KPipeline 初始化完成，使用本地KModel实例")
            
            # 3. 不需要再次加载模型，因为已经在KModel初始化时加载了本地模型
            logger.debug("本地模型已加载，跳过重复加载")
        
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
                    audio_chunks.append(torch.zeros(1, 24000))
                
            except Exception as e:
                logger.error(f"生成失败 '{content[:30]}...': {e}")
                audio_chunks.append(torch.zeros(1, 24000))
        
        # 拼接所有音频
        if audio_chunks:
            return torch.cat(audio_chunks, dim=-1)
        else:
            logger.error("所有音频生成失败，返回静音")
            return torch.zeros(1, 24000)
    
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
                return torch.zeros(1, int(max_audio_length_ms * 24000 / 1000))
            
            # 裁剪到指定长度
            max_length = int(max_audio_length_ms * 24000 / 1000)
            if audio.shape[-1] > max_length:
                audio = audio[..., :max_length]
            
            return audio
            
        except Exception as e:
            logger.error(f"生成参考音频失败 '{text[:30]}...': {e}")
            return torch.zeros(1, int(max_audio_length_ms * 24000 / 1000))
    
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
            import random
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
            import random
            selected_voice = random.choice(self.voice_files)
            logger.debug(f"角色 {role} 随机选择音色: {selected_voice.name}")
        
        # 将选择的音色存入缓存
        if selected_voice:
            voice_cache[role] = selected_voice
        
        return selected_voice
