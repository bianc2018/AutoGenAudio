import os
import logging
from pathlib import Path
from typing import Dict, Any

from ASRModel.base import BaseASRModel

# 导入sherpa-onnx库
import sherpa_onnx

logger = logging.getLogger("AutoEvalAudio")


class SherpaONNXAdapter(BaseASRModel):
    def __init__(self, model_config: dict):
        super().__init__(model_config)
        self.model = None
        self.model_name = model_config.get("name", "sherpa_onnx")
        self.model_type = model_config.get("model_type", "asr")
        self.sample_rate = model_config.get("sample_rate", 16000)
        # 新增：记录模型是否为流式模型
        self.is_streaming_model = False
    
    def load(self) -> None:
        """加载Sherpa-ONNX模型"""
        try:
            logger.info(f"正在加载Sherpa-ONNX模型: {self.model_name}")
            
            # 根据模型类型创建不同的配置
            if self.model_type == "asr":
                # 检查是否提供了模型目录
                if "model_dir" in self.model_config:
                    model_dir = Path(self.model_config["model_dir"])
                    model_architecture = self.model_config.get("model_architecture", "zipformer")
                    
                    logger.info(f"从目录加载模型: {model_dir}, 架构: {model_architecture}")
                    
                    # 列出目录中的文件，方便调试
                    file_list = [f.name for f in model_dir.iterdir() if f.is_file()]
                    logger.info(f"模型目录中的文件: {file_list}")
                    
                    # 根据模型架构自动查找模型文件
                    if model_architecture in ["zipformer", "conformer"]:
                        # encoder-decoder架构
                        # 查找encoder文件
                        logger.info(f"正在查找encoder文件，目录: {model_dir}")
                        encoder_patterns = ["*encoder*.onnx", "encoder*.onnx"]
                        encoder_files = []
                        for pattern in encoder_patterns:
                            found = list(model_dir.glob(pattern))
                            logger.info(f"使用模式 '{pattern}' 找到的encoder文件: {[f.name for f in found]}")
                            encoder_files.extend(found)
                        encoder_path = encoder_files[0] if encoder_files else None
                        logger.info(f"最终选择的encoder路径: {encoder_path}")
                        
                        # 查找decoder文件
                        logger.info(f"正在查找decoder文件")
                        decoder_patterns = ["*decoder*.onnx", "decoder*.onnx"]
                        decoder_files = []
                        for pattern in decoder_patterns:
                            found = list(model_dir.glob(pattern))
                            logger.info(f"使用模式 '{pattern}' 找到的decoder文件: {[f.name for f in found]}")
                            decoder_files.extend(found)
                        decoder_path = decoder_files[0] if decoder_files else None
                        logger.info(f"最终选择的decoder路径: {decoder_path}")
                        
                        # 查找joiner文件
                        logger.info(f"正在查找joiner文件")
                        joiner_patterns = ["*joiner*.onnx", "joiner*.onnx"]
                        joiner_files = []
                        for pattern in joiner_patterns:
                            found = list(model_dir.glob(pattern))
                            logger.info(f"使用模式 '{pattern}' 找到的joiner文件: {[f.name for f in found]}")
                            joiner_files.extend(found)
                        joiner_path = joiner_files[0] if joiner_files else None
                        logger.info(f"最终选择的joiner路径: {joiner_path}")
                        
                        # 查找tokens文件
                        logger.info(f"正在查找tokens文件")
                        tokens_patterns = ["*tokens*.txt", "tokens*.txt", "*dict*.txt"]
                        tokens_files = []
                        for pattern in tokens_patterns:
                            found = list(model_dir.glob(pattern))
                            logger.info(f"使用模式 '{pattern}' 找到的tokens文件: {[f.name for f in found]}")
                            tokens_files.extend(found)
                        tokens_path = tokens_files[0] if tokens_files else None
                        logger.info(f"最终选择的tokens路径: {tokens_path}")
                        
                        # 检查必要的文件是否存在
                        if not encoder_path or not decoder_path or not joiner_path or not tokens_path:
                            logger.error(f"找不到必要的模型文件:")
                            logger.error(f"  encoder_path: {encoder_path}")
                            logger.error(f"  decoder_path: {decoder_path}")
                            logger.error(f"  joiner_path: {joiner_path}")
                            logger.error(f"  tokens_path: {tokens_path}")
                            raise FileNotFoundError(f"在模型目录 {model_dir} 中找不到必要的模型文件")
                        
                        logger.info(f"找到模型文件:")
                        logger.info(f"  encoder: {encoder_path.name}")
                        logger.info(f"  decoder: {decoder_path.name}")
                        logger.info(f"  joiner: {joiner_path.name}")
                        logger.info(f"  tokens: {tokens_path.name}")
                        
                        # 保存模型文件路径，以便在transcribe方法中使用
                        self.encoder_path = encoder_path
                        self.decoder_path = decoder_path
                        self.joiner_path = joiner_path
                        self.tokens_path = tokens_path
                        
                        # 对于zipformer/conformer模型，我们应该使用OnlineRecognizer而不是OfflineRecognizer
                        # 因为这些是流式模型
                        logger.info("使用OnlineRecognizer加载流式模型")
                        self.model = sherpa_onnx.OnlineRecognizer.from_transducer(
                            encoder=str(encoder_path),
                            decoder=str(decoder_path),
                            joiner=str(joiner_path),
                            tokens=str(tokens_path),
                            sample_rate=self.sample_rate,
                            feature_dim=self.model_config.get("feature_dim", 80),
                            num_threads=self.model_config.get("num_threads", 1),
                            provider=self.model_config.get("provider", "cpu"),
                            decoding_method=self.model_config.get("decoding_method", "greedy_search"),
                            max_active_paths=self.model_config.get("max_active_paths", 4),
                            hotwords_file=self.model_config.get("hotwords_file", ""),
                            hotwords_score=self.model_config.get("hotwords_score", 1.5),
                        )
                        self.is_streaming_model = True
                    elif model_architecture in ["whisper", "whisper-tiny", "whisper-base", "whisper-small", "whisper-medium", "whisper-large"]:
                        # encoder-only架构（Whisper）
                        # 查找model文件
                        logger.info(f"正在查找model文件")
                        model_patterns = ["*model*.onnx", "*.onnx"]
                        model_files = []
                        for pattern in model_patterns:
                            found = list(model_dir.glob(pattern))
                            logger.info(f"使用模式 '{pattern}' 找到的model文件: {[f.name for f in found]}")
                            model_files.extend(found)
                        # 过滤掉encoder/decoder/joiner文件，只保留whisper模型文件
                        whisper_model_files = [f for f in model_files if "encoder" not in f.name.lower() and "decoder" not in f.name.lower() and "joiner" not in f.name.lower()]
                        model_path = whisper_model_files[0] if whisper_model_files else None
                        logger.info(f"最终选择的model路径: {model_path}")
                        
                        # 查找tokens文件
                        logger.info(f"正在查找tokens文件")
                        tokens_patterns = ["*tokens*.txt", "tokens*.txt", "*dict*.txt"]
                        tokens_files = []
                        for pattern in tokens_patterns:
                            found = list(model_dir.glob(pattern))
                            logger.info(f"使用模式 '{pattern}' 找到的tokens文件: {[f.name for f in found]}")
                            tokens_files.extend(found)
                        tokens_path = tokens_files[0] if tokens_files else None
                        logger.info(f"最终选择的tokens路径: {tokens_path}")
                        
                        # 检查必要的文件是否存在
                        if not model_path or not tokens_path:
                            logger.error(f"找不到必要的模型文件:")
                            logger.error(f"  model_path: {model_path}")
                            logger.error(f"  tokens_path: {tokens_path}")
                            raise FileNotFoundError(f"在模型目录 {model_dir} 中找不到必要的模型文件")
                        
                        logger.info(f"找到模型文件:")
                        logger.info(f"  model: {model_path.name}")
                        logger.info(f"  tokens: {tokens_path.name}")
                        
                        # 直接使用from_whisper方法，传递所需的参数
                        self.model = sherpa_onnx.OfflineRecognizer.from_whisper(
                            model=str(model_path),
                            tokens=str(tokens_path),
                            sample_rate=self.sample_rate,
                            feature_dim=self.model_config.get("feature_dim", 80),
                            num_threads=self.model_config.get("num_threads", 1),
                            provider=self.model_config.get("provider", "cpu"),
                            decoding_method=self.model_config.get("decoding_method", "greedy_search"),
                            max_active_paths=self.model_config.get("max_active_paths", 4),
                            hotwords_file=self.model_config.get("hotwords_file", ""),
                            hotwords_score=self.model_config.get("hotwords_score", 1.5),
                        )
                        self.is_streaming_model = False
                    else:
                        raise ValueError(f"不支持的模型架构: {model_architecture}")
                elif "encoder" in self.model_config and "decoder" in self.model_config:
                    # 传统配置方式：直接提供模型文件路径
                    # encoder-decoder架构（如Zipformer）
                    # 使用OnlineRecognizer因为这些是流式模型
                    logger.info("使用OnlineRecognizer加载流式模型")
                    self.model = sherpa_onnx.OnlineRecognizer.from_transducer(
                        encoder=self.model_config["encoder"],
                        decoder=self.model_config["decoder"],
                        joiner=self.model_config.get("joiner"),
                        tokens=self.model_config["tokens"],
                        sample_rate=self.sample_rate,
                        feature_dim=self.model_config.get("feature_dim", 80),
                        num_threads=self.model_config.get("num_threads", 1),
                        provider=self.model_config.get("provider", "cpu"),
                        decoding_method=self.model_config.get("decoding_method", "greedy_search"),
                        max_active_paths=self.model_config.get("max_active_paths", 4),
                        hotwords_file=self.model_config.get("hotwords_file", ""),
                        hotwords_score=self.model_config.get("hotwords_score", 1.5),
                    )
                    self.is_streaming_model = True
                else:
                    # encoder-only架构或其他模型（如Whisper）
                    # 传统配置方式：直接提供模型文件路径
                    if "model" in self.model_config:
                        # 直接使用from_whisper方法，传递所需的参数
                        self.model = sherpa_onnx.OfflineRecognizer.from_whisper(
                            model=self.model_config["model"],
                            tokens=self.model_config["tokens"],
                            sample_rate=self.sample_rate,
                            feature_dim=self.model_config.get("feature_dim", 80),
                            num_threads=self.model_config.get("num_threads", 1),
                            provider=self.model_config.get("provider", "cpu"),
                            decoding_method=self.model_config.get("decoding_method", "greedy_search"),
                            max_active_paths=self.model_config.get("max_active_paths", 4),
                            hotwords_file=self.model_config.get("hotwords_file", ""),
                            hotwords_score=self.model_config.get("hotwords_score", 1.5),
                        )
                        self.is_streaming_model = False
                    else:
                        raise ValueError("未提供有效的模型配置")
            elif self.model_type == "kws":
                # KWS模型加载（暂时简化，只支持从模型目录加载）
                if "model_dir" in self.model_config:
                    model_dir = Path(self.model_config["model_dir"])
                    
                    logger.info(f"从目录加载KWS模型: {model_dir}")
                    
                    # 列出目录中的文件，方便调试
                    file_list = [f.name for f in model_dir.iterdir() if f.is_file()]
                    logger.info(f"模型目录中的文件: {file_list}")
                    
                    # 查找模型文件
                    logger.info(f"正在查找encoder文件")
                    encoder_patterns = ["*encoder*.onnx", "encoder*.onnx"]
                    encoder_files = []
                    for pattern in encoder_patterns:
                        found = list(model_dir.glob(pattern))
                        logger.info(f"使用模式 '{pattern}' 找到的encoder文件: {[f.name for f in found]}")
                        encoder_files.extend(found)
                    encoder_path = encoder_files[0] if encoder_files else None
                    logger.info(f"最终选择的encoder路径: {encoder_path}")
                    
                    logger.info(f"正在查找decoder文件")
                    decoder_patterns = ["*decoder*.onnx", "decoder*.onnx"]
                    decoder_files = []
                    for pattern in decoder_patterns:
                        found = list(model_dir.glob(pattern))
                        logger.info(f"使用模式 '{pattern}' 找到的decoder文件: {[f.name for f in found]}")
                        decoder_files.extend(found)
                    decoder_path = decoder_files[0] if decoder_files else None
                    logger.info(f"最终选择的decoder路径: {decoder_path}")
                    
                    logger.info(f"正在查找joiner文件")
                    joiner_patterns = ["*joiner*.onnx", "joiner*.onnx"]
                    joiner_files = []
                    for pattern in joiner_patterns:
                        found = list(model_dir.glob(pattern))
                        logger.info(f"使用模式 '{pattern}' 找到的joiner文件: {[f.name for f in found]}")
                        joiner_files.extend(found)
                    joiner_path = joiner_files[0] if joiner_files else None
                    logger.info(f"最终选择的joiner路径: {joiner_path}")
                    
                    logger.info(f"正在查找tokens文件")
                    tokens_patterns = ["*tokens*.txt", "tokens*.txt", "*dict*.txt"]
                    tokens_files = []
                    for pattern in tokens_patterns:
                        found = list(model_dir.glob(pattern))
                        logger.info(f"使用模式 '{pattern}' 找到的tokens文件: {[f.name for f in found]}")
                        tokens_files.extend(found)
                    tokens_path = tokens_files[0] if tokens_files else None
                    logger.info(f"最终选择的tokens路径: {tokens_path}")
                    
                    # 检查必要的文件是否存在
                    if not encoder_path or not decoder_path or not joiner_path or not tokens_path:
                        logger.error(f"找不到必要的模型文件:")
                        logger.error(f"  encoder_path: {encoder_path}")
                        logger.error(f"  decoder_path: {decoder_path}")
                        logger.error(f"  joiner_path: {joiner_path}")
                        logger.error(f"  tokens_path: {tokens_path}")
                        raise FileNotFoundError(f"在模型目录 {model_dir} 中找不到必要的模型文件")
                    
                    # 检查关键词文件
                    keywords_file = self.model_config.get("keywords_file")
                    if not keywords_file:
                        raise ValueError("KWS模型需要提供keywords_file")
                    
                    # 直接使用from_transducer方法，传递所需的参数
                    self.model = sherpa_onnx.KeywordSpotter.from_transducer(
                        encoder=str(encoder_path),
                        decoder=str(decoder_path),
                        joiner=str(joiner_path),
                        tokens=str(tokens_path),
                        sample_rate=self.sample_rate,
                        feature_dim=self.model_config.get("feature_dim", 80),
                        keywords_file=keywords_file,
                        keywords_score=self.model_config.get("keywords_score", 1.0),
                        num_threads=self.model_config.get("num_threads", 1),
                        provider=self.model_config.get("provider", "cpu"),
                    )
                else:
                    # 传统配置方式：直接提供模型文件路径
                    # 直接使用from_transducer方法，传递所需的参数
                    self.model = sherpa_onnx.KeywordSpotter.from_transducer(
                        encoder=self.model_config.get("encoder"),
                        decoder=self.model_config.get("decoder"),
                        joiner=self.model_config.get("joiner"),
                        tokens=self.model_config["tokens"],
                        sample_rate=self.sample_rate,
                        feature_dim=self.model_config.get("feature_dim", 80),
                        keywords_file=self.model_config["keywords_file"],
                        keywords_score=self.model_config.get("keywords_score", 1.0),
                        num_threads=self.model_config.get("num_threads", 1),
                        provider=self.model_config.get("provider", "cpu"),
                    )
            else:
                raise ValueError(f"不支持的模型类型: {self.model_type}")
            
            self.is_loaded = True
            logger.info(f"Sherpa-ONNX模型加载成功: {self.model_name}")
        except Exception as e:
            logger.error(f"加载Sherpa-ONNX模型失败: {e}")
            raise
    
    def transcribe(self, audio_path: str) -> dict:
        """音频转写，返回包含转写文本和时间信息的字典"""
        if not self.is_loaded:
            self.load()
        
        try:
            logger.debug(f"正在转写音频: {audio_path}")
            
            # 检查音频文件是否存在
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"音频文件不存在: {audio_path}")
            
            # 根据模型类型执行不同的转写
            if self.model_type == "asr":
                # 执行ASR转写
                import numpy as np
                import soundfile as sf
                
                # 使用soundfile库读取音频文件，它支持更多格式
                waveform, framerate = sf.read(audio_path, dtype='float32')
                
                # 如果是多通道，转换为单通道
                if waveform.ndim > 1:
                    waveform = np.mean(waveform, axis=1)
                
                # 根据模型类型执行不同的转写逻辑
                if self.is_streaming_model:
                    # 使用OnlineStream处理流式模型
                    stream = self.model.create_stream()
                    
                    # 接受波形数据
                    stream.accept_waveform(framerate, waveform.tolist())
                    
                    # 对于流式模型，我们需要输入EOF
                    stream.input_finished()
                    
                    # 解码
                    while self.model.is_ready(stream):
                        self.model.decode_stream(stream)
                    
                    # 获取结果 - 注意：OnlineStream的get_result()方法返回字符串
                    text_result = self.model.get_result(stream)
                else:
                    # 使用OfflineStream处理离线模型
                    stream = self.model.create_stream()
                    
                    # 接受波形数据
                    stream.accept_waveform(framerate, waveform.tolist())
                    
                    # 解码
                    self.model.decode_stream(stream)
                    
                    # 获取结果 - OfflineStream的result是对象，有text属性
                    text_result = stream.result.text
                
                # 构建返回结果
                return {
                    "text": text_result,
                    "model_name": self.model_name,
                    "sample_rate": self.sample_rate,
                    "audio_path": audio_path,
                    "success": True,
                }
            elif self.model_type == "kws":
                # 执行关键词检测
                result = self.model(audio_path)
                
                # 构建返回结果
                return {
                    "text": " | ".join([kw.keyword for kw in result.keywords]),
                    "keywords": [{
                        "keyword": kw.keyword,
                        "start_time": kw.start_time,
                        "end_time": kw.end_time,
                        "score": kw.score
                    } for kw in result.keywords],
                    "model_name": self.model_name,
                    "sample_rate": self.sample_rate,
                    "audio_path": audio_path,
                    "success": True,
                }
            else:
                raise ValueError(f"不支持的模型类型: {self.model_type}")
        except Exception as e:
            logger.error(f"转写音频失败: {e}")
            return {
                "text": "",
                "model_name": self.model_name,
                "sample_rate": self.sample_rate,
                "audio_path": audio_path,
                "success": False,
                "error": str(e)
            }
    
    def get_name(self) -> str:
        """获取模型名称"""
        return self.model_name
