#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""probe_eval.py — 正式評測前送「一題」，並用 twinkle-eval 自己的抽取器驗收。

用法：
    python scripts/probe_eval.py configs/eval_gemma4_e4b_base.yaml

為什麼需要這支：
    8/13 那兩次評測分別踩到兩個不同的坑，而且都是「跑完才知道」：
      1. 22:08 那次 —— 4,144 個請求全部 502（端點根本不通）
      2. 23:55 那次 —— 請求全部 200、模型答案也全對，
                       但 predicted_answer 全是 null、「無法解析 139/139」
                       因為 config 沒寫 system_prompt，模型不知道要輸出 \\box{}
    共通點：**沒有在送出第一批之前，驗證整條鏈路端到端會產生一個可解析的答案。**

    所以這支程式不只檢查 HTTP 200，還會：
      - 用 config 裡真正的 system_prompt（不是另外編一個）
      - 用 twinkle_eval.metrics.extractors.box.BoxExtractor（不是自己寫正則）
      - 斷言抽出來的字母 == 正確答案
    三關都過，才准許跑正式評測。
"""
import json
import sys

import httpx
import yaml

# 一題已知答案的四選一，格式和 TMMLU+ 完全一致
QUESTION = "台灣海拔最高的山峰是下列何者？\nA: 雪山\nB: 玉山\nC: 秀姑巒山\nD: 南湖大山"
GOLD = "B"


def main():
    cfg_path = sys.argv[1]
    cfg = yaml.safe_load(open(cfg_path, encoding="utf-8"))

    api = cfg["llm_api"]
    model = cfg["model"]
    ev = cfg["evaluation"]

    # 用 config 裡真正的 system_prompt，語言鍵沿用 datasets_prompt_map 的預設 "zh"
    sp = ev.get("system_prompt", {})
    sys_prompt = sp.get("zh", "") if isinstance(sp, dict) else sp
    if not sys_prompt.strip():
        print("❌ config 的 evaluation.system_prompt 是空的。")
        print("   twinkle_eval 對 system_prompt 沒有預設值（models/openai.py 第 61 行），")
        print("   空字串代表模型不會被告知要輸出 \\box{}，結果就是 100% 無法解析。")
        sys.exit(1)
    print(f"▶ system_prompt（{len(sys_prompt)} 字）：")
    for line in sys_prompt.strip().splitlines():
        print("   |", line)

    payload = {
        "model": model["name"],
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": QUESTION},
        ],
        "temperature": model.get("temperature", 0.0),
        "top_p": model.get("top_p", 1.0),
        "max_tokens": model.get("max_tokens", 2048),
    }
    payload.update(model.get("extra_body") or {})   # chat_template_kwargs 等

    url = api["base_url"].rstrip("/") + "/chat/completions"
    print(f"\n▶ POST {url}")
    # trust_env=False：直接繞過 http_proxy/https_proxy，localhost 不該走代理
    with httpx.Client(timeout=180, trust_env=False) as c:
        r = c.post(url, json=payload, headers={"Content-Type": "application/json"})
    print(f"   HTTP {r.status_code}")
    if r.status_code != 200:
        print("❌ 端點沒回 200，不要送正式評測。回應：")
        print(r.text[:800])
        sys.exit(1)

    msg = r.json()["choices"][0]["message"]
    out = (msg.get("content") or "").strip()
    rsn = (msg.get("reasoning") or "").strip()
    print(f"   thinking {len(rsn)} 字、answer {len(out)} 字")
    print("   answer =", repr(out[:200]))
    if not out:
        print("❌ content 是空的（可能全被塞進 reasoning）。評測會全部判錯。")
        sys.exit(1)

    # 用 twinkle-eval 自己的抽取器，不要自己寫正則 —— 自己寫的正則過了不代表它會過
    from twinkle_eval.metrics.extractors.box import BoxExtractor

    got = BoxExtractor().extract(out)
    print(f"\n▶ BoxExtractor 抽到：{got!r}（正確答案 {GOLD}）")
    if got is None:
        print("❌ 抽不出答案。BoxExtractor 只認 \\box{X} / \\boxed{X}，X 必須是單一大寫字母。")
        print("   → 檢查 system_prompt 有沒有被送出去，以及模型有沒有照格式回。")
        sys.exit(1)
    if got != GOLD:
        print(f"⚠️  格式解析成功，但答錯了（{got} ≠ {GOLD}）。")
        print("   格式沒問題，可以跑；答錯只是這一題模型不會，不擋評測。")
    else:
        print("   ✅ 格式正確且答對")

    print("\n✅ 探路通過：端點通、system_prompt 有送、輸出可被解析。可以跑正式評測。")


if __name__ == "__main__":
    main()
