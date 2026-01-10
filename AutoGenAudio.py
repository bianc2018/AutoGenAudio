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
from tqdm import tqdm
import logging
import warnings
import re
import glob
import inspect

# 导入分离后的TTS模型模块
from TTSModel.base import BaseTTSModel
from TTSModel.fireredtts2 import FireRedTTS2Adapter
from TTSModel.kokoro import KokoroAdapter
from TTSModel.higgs import HiggsAdapter
from TTSModel.vibevoice import VibeVoiceAdapter

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

# ---------------- 文本工具函数 ----------------
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
