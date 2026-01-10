#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量调用自建 OpenAI 接口生成多轮对话（TOML 配置版）
新增：①让大模型用 3~5 个汉字总结对话 → 文件名 = 总结_序号.json
     ②强制去掉 <think>...</think> 包裹内容
"""
import json
import logging
import random
import time
import re
from pathlib import Path
from openai import OpenAI
import tomllib
from tqdm import tqdm

# ---------- 加载配置 ----------
CFG = tomllib.load(open("AutoGenDialog.toml", "rb"))
OUT_DIR = Path(CFG["output_dir"])
OUT_DIR.mkdir(exist_ok=True)
logging.basicConfig(level="INFO", format="%(asctime)s | %(levelname)s | %(message)s")

client = OpenAI(base_url=CFG["api_base"], api_key=CFG["api_key"] or "dummy")

# ----------  Prompt 模板 ----------
SYSTEM_PROMPT = """
你是对话生成助手。我会给一句“聊天主题”，请你生成一段自然流畅、逻辑通顺的多人聊天。
要求：
1. 说话人从给定角色池中随机选择 2–3 人，每句仅出现一人。
2. 必须严格按时间顺序，交替发言，让对话完整、有逻辑。
3. 返回纯 JSON 数组，禁止任何解释，禁止 markdown 代码块。
4. 对话轮数是指生成多少句话，如格式示例则是俩轮对话，生成对话的长度必须严格按照要求，不得省略。
5. 对话需要完整，不允许中间节点，必须要将事情讲完。
格式示例：
[{"role":"S1","text":"你好，今天的天气真不错啊！"},
 {"role":"S2","text":"是啊，阳光明媚，适合出去走走。"}]
"""

SUMMARY_PROMPT = """
下面是一段多人对话的 JSON 数组，请用 **5～10 个汉字** 精准总结话题。
要求：
- 只能输出汉字，禁止标点、字母、数字、空格、换行。
- 禁止出现任何解释或多余文字。
- 如果无法总结，请输出 对话
输出示例：天气 电影 学习
---
{dialog}
"""

def extract_chinese(text: str) -> str:
    """只留汉字，超长截断，过短 fallback"""
    return text
    #ans = re.sub(r"[^\u4e00-\u9fa5]", "", text.strip())
    #ans = ans[:5]                       # 最长 5 字
    #return ans if 2 <= len(ans) <= 5 else "对话"

def strip_think(text: str) -> str:
    """去掉 <think>...</think> 及其内部所有内容"""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def call_llm(topic: str, roles: list, turns: int) -> list:
    role_str = "、".join(roles)
    user_prompt = f"""聊天主题：{topic}
角色：{role_str}
请生成至少 {turns} 轮对话（≥{turns} 句），可多不可少，必须聊完、有结尾。
返回纯 JSON 数组，禁止解释。"""

    dialog = []                       # 最终容器
    remaining = turns                 # 还差多少句
    attempt = 0                       # 总尝试次数（防真·死循环）
    max_attempt = 50                  # 硬上限，避免无限

    while remaining > 0 and attempt < max_attempt:
        attempt += 1
        try:
            resp = client.chat.completions.create(
                model=CFG["model"],
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=CFG["temperature"],
                max_tokens=CFG["max_tokens"]
            )
            content = strip_think(resp.choices[0].message.content)
            if content.startswith("```"):
                content = content.strip("`").strip()
                if content.startswith("json"):
                    content = content[4:].strip()
            chunk = json.loads(content)        # 这次拿到的数组
            if not isinstance(chunk, list):
                raise ValueError("非数组")
            dialog.extend(chunk)               # 累加
            remaining = turns - len(dialog)    # 更新缺口
            if remaining <= 0:
                logging.info("生成长度:(%d)", len(dialog))
                return dialog                    # 已达标，直接返回

           # 缺口还在 → 补刀提示（带完整上文）
            last_turn = dialog[-1]  # 最后一句，用来显式承接
            user_prompt = f"""上面这段对话共 {len(dialog)} 句，还差 {remaining} 句才到 {turns} 轮。
            已生成对话：
            {json.dumps(dialog, ensure_ascii=False, indent=2)}

            请继续往下编，再编恰好 {remaining} 句，要求：
            1. 必须承接最后一句（{last_turn['role']}：{last_turn['text']}）；
            2. 角色只能从 {role_str} 中选，保持人设一致；
            3. 话题、情绪、逻辑必须顺推，禁止跳新话题；
            4. 返回纯 JSON 数组，禁止任何解释。"""

        except Exception as e:
            logging.warning("补刀失败(%d)：%s", attempt, e)
            time.sleep(1)

    # 真到极限也凑不够，降权返回，保证流程不中断
    logging.error("死磕后仍只有 %d 句，低于要求 %d，先放行", len(dialog), turns)
    return dialog


def summarize(dialog: list) -> str:
    """让大模型用 3~5 个汉字总结对话"""
    try:
        text = json.dumps(dialog, ensure_ascii=False)
        resp = client.chat.completions.create(
            model=CFG["model"],
            messages=[{"role": "user", "content": SUMMARY_PROMPT.format(dialog=text)}],
            temperature=0.1,          # 越低越稳定
            #max_tokens=8              # 物理截断
        )
        raw = strip_think(resp.choices[0].message.content)
        summary = extract_chinese(raw)
        return summary
    except Exception as e:
        logging.warning("总结失败：%s", e)
        return "对话"


def main():
    topics = Path(CFG["prompt_file"]).read_text(encoding="utf8").splitlines()
    total = len(topics) * CFG["dialogue_per_topic"]
    pbar = tqdm(total=total, desc="生成进度", unit="条", ncols=80)

    counter = 0
    for topic in topics:
        topic = topic.strip()
        if not topic:
            continue
        for _ in range(CFG["dialogue_per_topic"]):
            # 随机角色 2~3 人
            k = random.randint(2, 3)
            roles = random.sample(CFG["role_pool"], k)
            turns = random.randint(*CFG["turn_range"])
            dialog = call_llm(topic, roles, turns)

            # 总结 → 文件名
            summary = summarize(dialog)
            counter += 1
            filename = f"{summary}_{counter:04d}.json"
            file = OUT_DIR / filename
            file.write_text(json.dumps(dialog, ensure_ascii=False, indent=2), encoding="utf8")
            pbar.update(1)
    pbar.close()
    logging.info("全部完成！共生成 %d 条对话，已保存至 %s", total, OUT_DIR.resolve())


if __name__ == "__main__":
    main()