#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音频增强批量脚本 —— 支持 audiomentations 全部效果
pip install audiomentations soundfile tqdm toml joblib
"""
import os, sys, itertools, inspect, toml
from pathlib import Path
import soundfile as sf
import numpy as np
from tqdm import tqdm
import audiomentations as am
from joblib import Parallel, delayed

# ---------- 工具 ----------
def load_done(log_file):
    p = Path(log_file)
    return set(p.read_text(encoding='utf8').splitlines()) if p.exists() else set()

def append_done(log_file, name):
    with open(log_file, 'a', encoding='utf8') as f:
        f.write(name + '\n')

def scan_augmentations():
    """扫描 audiomentations 所有*可空参实例化*的增强类"""
    from audiomentations.core.transforms_interface import BaseWaveformTransform
    import inspect

    single = {}
    for name, obj in inspect.getmembers(am, inspect.isclass):
        if (issubclass(obj, BaseWaveformTransform) and
                not inspect.isabstract(obj) and
                not name.endswith(("Transform", "Compose"))):
            # 尝试空参构造，失败就跳过
            try:
                obj()
                single[name] = name
            except Exception:        # 缺失必填参数、文件找不到等
                continue
    return single

def build_chain(cls_names, cfg):
    """根据类名列表实例化 Compose"""
    transforms = []
    for cls_name in cls_names:
        cls = getattr(am, cls_name)
        kwargs = cfg.get("params", {}).get(cls_name, {})
        transforms.append(cls(**kwargs))
    return am.Compose(transforms)

def make_chains(singles: dict, combine: bool, max_combine: int):
    """返回 [ ( [alias..], [cls_name..] ) ]"""
    chains = []
    keys = list(singles.keys())          # 即英文类名
    # 单效果
    for k in keys:
        chains.append(([k], [k]))
    if not combine:
        return chains
    # 组合
    for r in range(2, max_combine + 1):
        for combo in itertools.combinations(keys, r):
            chains.append((list(combo), list(combo)))
    return chains

def process_one(audio_path, label_path, chain_aliases, chain_cls_names, cfg, done_set, log_file):
    """处理单个文件"""
    out_name = f"{audio_path.stem}_{'_'.join(chain_aliases)}"
    out_wav = Path(cfg["output_dir"]) / f"{out_name}.wav"
    out_csv = Path(cfg["output_dir"]) / f"{out_name}.csv"

    if str(out_wav) in done_set and out_wav.exists():
        return

    # 读音频
    wav, sr = sf.read(audio_path)
    if sr != cfg["sample_rate"]:
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=cfg["sample_rate"])
        sr = cfg["sample_rate"]

    # 增强
    chain = build_chain(chain_cls_names, cfg)
    aug_wav = chain(samples=wav, sample_rate=sr)

    # 写
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_wav, aug_wav, sr)
    if label_path and label_path.exists():
        import shutil
        shutil.copy(label_path, out_csv)

    append_done(log_file, str(out_wav))

def main():
    cfg = toml.load("AutoAugment.toml")
    Path(cfg["output_dir"]).mkdir(exist_ok=True)
    done_set = load_done(cfg["done_log"])

    # 扫描全部增强类
    singles = scan_augmentations()          # { "AddGaussianNoise":"AddGaussianNoise", ... }
    chains = make_chains(singles, cfg["combine"], cfg["max_combine"])

    # 扫描输入
    in_dir = Path(cfg["input_dir"])
    au_files = sorted(in_dir.glob(f"*.{cfg['audio_ext']}"))
    lab_files = {p.stem: p for p in in_dir.glob(f"*.{cfg['label_ext']}")}

    tasks = []
    for au in au_files:
        lab = lab_files.get(au.stem)
        for aliases, cls_names in chains:
            tasks.append((au, lab, aliases, cls_names))

    Parallel(n_jobs=cfg["num_workers"], backend="threading")(
        delayed(process_one)(au, lab, al, cl, cfg, done_set, cfg["done_log"])
        for au, lab, al, cl in tqdm(tasks, desc="Augment")
    )
    print("All done!")

if __name__ == "__main__":
    main()