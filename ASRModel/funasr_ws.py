import os
import logging
import asyncio
import websockets
import json
import base64
from pathlib import Path
from typing import Dict, Any

from ASRModel.base import BaseASRModel

logger = logging.getLogger("AutoEvalAudio")


class FunASRWSAdapter(BaseASRModel):
    def __init__(self, model_config: dict):
        super().__init__(model_config)
        self.model_name = model_config.get("name", "funasr_ws")
        self.websocket_url = model_config.get("websocket_url", "ws://localhost:8000")
        self.sample_rate = model_config.get("sample_rate", 16000)
        self.chunk_size = model_config.get("chunk_size", 8000)  # 500ms at 16kHz
        self.chunk_interval = model_config.get("chunk_interval", 0.5)  # 500ms
    
    def load(self) -> None:
        """加载模型（对于WebSocket后端，这里主要是初始化配置）"""
        logger.info(f"正在初始化FunASR WebSocket后端: {self.model_name}")
        logger.info(f"WebSocket URL: {self.websocket_url}")
        self.is_loaded = True
    
    def transcribe(self, audio_path: str) -> dict:
        """音频转写，返回包含转写文本和时间信息的字典"""
        if not self.is_loaded:
            self.load()
        
        try:
            logger.debug(f"正在转写音频: {audio_path}")
            
            # 检查音频文件是否存在
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"音频文件不存在: {audio_path}")
            
            # 运行异步转写函数
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(self._async_transcribe(audio_path))
            loop.close()
            
            return result
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
    
    async def _async_transcribe(self, audio_path: str) -> dict:
        """异步转写函数"""
        try:
            # 读取音频文件
            with open(audio_path, 'rb') as f:
                audio_data = f.read()
            
            # 连接到WebSocket服务器
            async with websockets.connect(self.websocket_url) as websocket:
                logger.debug(f"已连接到WebSocket服务器: {self.websocket_url}")
                
                # 发送配置信息
                config = {
                    "mode": "offline",
                    "audio_format": "wav",
                    "sample_rate": self.sample_rate,
                    "chunk_size": self.chunk_size,
                    "chunk_interval": self.chunk_interval,
                    "asr_result_type": "text",
                }
                await websocket.send(json.dumps(config))
                
                # 发送音频数据
                await websocket.send(audio_data)
                logger.debug(f"已发送音频数据，大小: {len(audio_data)}字节")
                
                # 接收转写结果
                result = await websocket.recv()
                logger.debug(f"已接收转写结果: {result}")
                
                # 解析结果
                result_dict = json.loads(result)
                
                # 构建返回结果
                return {
                    "text": result_dict.get("text", ""),
                    "model_name": self.model_name,
                    "sample_rate": self.sample_rate,
                    "audio_path": audio_path,
                    "success": True,
                    "raw_result": result_dict,
                }
        except websockets.exceptions.ConnectionClosedError as e:
            logger.error(f"WebSocket连接关闭错误: {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"解析转写结果失败: {e}")
            raise
    
    def get_name(self) -> str:
        """获取模型名称"""
        return self.model_name
