#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
week4_eval_server.py — 對任何 OpenAI-compatible server 跑 TMMLU+ / MMLU 評測。

Week 4 的統一評測器：把 Week 3 notebook §7.1 的評測器（固定種子、嚴格/寬鬆、
macro/micro）搬到本機，改走 HTTP API，所以 mlx_lm.server、llama-server、
vLLM 都吃得下 —— AR（gemma-4-26B）和 DLM（diffusiongemma）可以用同一支程式評。

用法（先起 server，再跑這支）：
    mlx_lm.server --model mlx-community/gemma-4-26B-A4B-it-4bit --port 1234
    python scripts/week4_eval_server.py --tag ar26b_base \
        --model-name mlx-community/gemma-4-26B-A4B-it-4bit --thinking off

    # Week 2 真因重評（同一個模型 × thinking on/off 各一次）：
    mlx_lm.server --model out/gemma4-e4b-tw --port 1234
    python scripts/week4_eval_server.py --tag w2tuned_think_on  --model-name out/gemma4-e4b-tw \
        --thinking on --max-tokens 2048 --limit 100
    python scripts/week4_eval_server.py --tag w2tuned_think_off --model-name out/gemma4-e4b-tw \
        --thinking off --max-tokens 512 --limit 100

    # 英文對照組（MoE 方案 A/B 的關鍵量測）：
    python scripts/week4_eval_server.py --tag planA_en --model-name ... \
        --dataset mmlu --thinking off

繼承前幾週的教訓（不要拿掉）：
  1. 種子固定（twinkle-eval 的 shuffle 沒設種子 → 兩輪題目排列不同）。
  2. macro 與 micro 並列（三科題數 768/139/129，科目平均會失真）。
  3. 開跑前防呆：/v1/models 對 model id ＋ 先送 3 題斷言抽得出 \\box{}。
  4. localhost 繞過 proxy（Week 2 的 4,144 個 502 全滅事故）。
  5. system_prompt 一定要送（twinkle-eval 沒預設值 → 全部無法解析 100% 事故）。
  6. 吞吐量只在同一台機器、同一個 engine 內可比，跨硬體不可比。
"""

import argparse
import json
import os
import random
import re
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ---- 與 twinkle-eval BoxExtractor / Week 3 notebook 逐字相同 ----
BOX_PATTERNS = [r"\\{1,2}box{([A-Z])}", r"\\{1,2}boxed{([A-Z])}"]
LENIENT = [r"box\{\s*([ABCD])\s*\}",
           r"(?:答案是|答案為|正確答案是|應該是|選項)\s*[:：]?\s*([ABCD])",
           r"(?:answer is|Answer:|answer:)\s*\(?([ABCD])\)?"]

SYS_BOX_ZH = (
    "使用者將提供一個題目，並附上選項 A、B、C、D。\n"
    "請仔細閱讀題目要求，根據題意選出最符合的選項，並將選項以以下格式輸出：\n"
    "\\box{選項}\n"
    "請確保僅將選項包含在 { } 中，否則將不計算為有效答案。\n"
    "務必精確遵循輸出格式，避免任何多餘內容或錯誤格式。\n"
    "例如：答案是 A，就輸出 \\box{A}。\n"
)
SYS_BOX_EN = (
    "You will be given a question with options A, B, C, D.\n"
    "Read carefully, pick the single best option, and output it EXACTLY as:\n"
    "\\box{X}\n"
    "where X is the option letter. Any other format is not counted.\n"
    "Example: if the answer is A, output \\box{A}.\n"
)

TMMLU_SUBJECTS = ["geography_of_taiwan", "taiwanese_hokkien", "three_principles_of_people"]
MMLU_SUBJECTS = ["high_school_geography", "world_religions", "logical_fallacies"]


def extract_strict(s):
    if not s:
        return None
    for p in BOX_PATTERNS:
        m = re.search(p, s)
        if m:
            return m.group(1).strip()
    return None


def extract_lenient(s):
    a = extract_strict(s)
    if a:
        return a
    if not s:
        return None
    for p in LENIENT:
        m = re.search(p, s)
        if m:
            return m.group(1).strip().upper()
    return None


def strip_thinking(s):
    """Gemma 4 / DiffusionGemma 的 thinking channel：<|channel>thought ... <channel|>。
    評分前把 thought 區塊剝掉，只留最終回答 —— 否則 thought 裡出現的
    「答案是 X」會被寬鬆解析誤抓（Week 2 的嫌疑之一，這裡直接杜絕）。"""
    if not s:
        return s
    return re.sub(r"<\|channel>thought.*?(<channel\|>|$)", "", s, flags=re.S)


def _valid(v):
    # pandas 的缺值是 NaN（float），NaN != NaN
    return v is not None and v == v


def shuffle_options(row, rng):
    """與 twinkle-eval 同樣「靠選項文字對回正解」，但 rng 由外部傳入 → 可重現。"""
    opts = [(k, row[k]) for k in "ABCD" if k in row and _valid(row[k])]
    if len(opts) < 2:
        return None
    gold_key = str(row["answer"]).strip().upper()
    gold_text = row.get(gold_key)
    if not _valid(gold_text):
        return None
    rng.shuffle(opts)
    new = {"question": row["question"]}
    for (old, text), newk in zip(opts, "ABCD"):
        new[newk] = text
        if text == gold_text:
            new["answer"] = newk
    return new if "answer" in new else None


def build_prompt(q):
    return q["question"] + "\n" + "\n".join(
        f"{k}: {v}" for k, v in q.items() if k not in ("question", "answer"))


# ---------------------------------------------------------------- data loading
def load_tmmluplus(subjects, seed, limit):
    import pandas as pd
    out = {}
    for s in subjects:
        df = pd.read_parquet(ROOT / "datasets" / "ikala__tmmluplus" / f"{s}.parquet")
        rng = random.Random(seed)                      # 每一科都從同一個種子開始
        raw = [shuffle_options(dict(r), rng) for _, r in df.iterrows()]
        qs = [q for q in raw if q]
        dropped = len(raw) - len(qs)
        if dropped:
            print(f"    ⚠️ {s}: {dropped} 題選項文字重複/缺失無法對回正解，已剔除")
        out[s] = qs[:limit] if limit else qs
    return out


def load_mmlu(subjects, seed, limit):
    """cais/mmlu：choices 是 list、answer 是 int(0-3)。先轉成和 TMMLU+ 相同的 dict。"""
    from datasets import load_dataset
    out = {}
    for s in subjects:
        ds = load_dataset("cais/mmlu", s, split="test")
        rng = random.Random(seed)
        qs = []
        for ex in ds:
            row = {"question": ex["question"], "answer": "ABCD"[int(ex["answer"])]}
            for i, c in enumerate(ex["choices"][:4]):
                row["ABCD"[i]] = str(c)
            q = shuffle_options(row, rng)
            if q:
                qs.append(q)
        out[s] = qs[:limit] if limit else qs
    return out


# ---------------------------------------------------------------- HTTP client
def _post(url, payload, timeout=600):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def chat(base_url, model, sys_prompt, user, max_tokens, thinking, timeout):
    body = {
        "model": model,
        "messages": [{"role": "system", "content": sys_prompt},
                     {"role": "user", "content": user}],
        "temperature": 0.0, "top_p": 1.0, "max_tokens": max_tokens,
    }
    if thinking in ("on", "off"):
        body["chat_template_kwargs"] = {"enable_thinking": thinking == "on"}
    t0 = time.time()
    resp = _post(base_url.rstrip("/") + "/chat/completions", body, timeout)
    dt = time.time() - t0
    msg = resp["choices"][0]["message"]
    txt = msg.get("content") or ""
    usage = resp.get("usage") or {}
    return txt, usage.get("completion_tokens"), dt


# ---------------------------------------------------------------- pre-flight
def preflight(base_url, model_name, sys_prompt, sample_q, max_tokens, thinking,
              allow_unparsed=False):
    # 1. localhost 繞 proxy（Week 2 的 502 全滅事故）
    for k in ("NO_PROXY", "no_proxy"):
        os.environ[k] = os.environ.get(k, "") + ",127.0.0.1,localhost"
    # 2. /v1/models 回得來，且 id 對得上
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/models", timeout=10) as r:
            ids = [m.get("id", "") for m in json.loads(r.read().decode()).get("data", [])]
    except Exception as e:
        sys.exit(f"❌ server 沒回應（{e}）。先起 server 再跑這支。")
    if ids and not any(model_name in i or i in model_name for i in ids):
        print(f"  ⚠️ server 端 model id {ids} 和 --model-name '{model_name}' 對不上。\n"
              f"     多數 server 會忽略 name 直接用載入的模型 —— 請自己確認起的是對的模型！")
    # 3. 先送 3 題，斷言抽得出 \box{}（Week 2 的「全部成功、零解析」事故）
    ok = 0
    for q in sample_q[:3]:
        txt, _, _ = chat(base_url, model_name, sys_prompt, build_prompt(q),
                         max_tokens, thinking, 600)
        got = extract_strict(strip_thinking(txt))
        print(f"  [防呆] 正解 {q['answer']} | 抽到 {got} | 前 100 字: {txt[:100]!r}")
        ok += got is not None
    if ok == 0:
        if allow_unparsed:
            print("  ⚠️ 3 題都抽不出 \\box{}，但 --allow-unparsed-probe 已開 → 續跑。\n"
                  "     這個 flag 只該用在「已知會格式崩潰的模型」（例如 Week 2 tuned），\n"
                  "     而且必須先用同一套指令跑過 base 模型確認管線正常。\n")
        else:
            sys.exit("❌ 3 題一題都抽不出 \\box{} —— 兩種可能：\n"
                     "   (a) 管線/prompt/thinking 設定壞了 → 先用 base 模型跑同一套指令驗證；\n"
                     "   (b) 這個模型本來就格式崩潰（如 Week 2 tuned），無法解析率正是要量的東西\n"
                     "       → 確認 (a) 排除後，加 --allow-unparsed-probe 再跑。")
    else:
        print(f"  防呆通過（{ok}/3）\n")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:1234/v1")
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--dataset", choices=["tmmluplus", "mmlu"], default="tmmluplus")
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None, help="每科題數上限（快速評測用 100）")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--thinking", choices=["on", "off", "none"], default="none",
                    help="on/off 會送 chat_template_kwargs；none = 不送（gemma-3、llama-server 用）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--allow-unparsed-probe", action="store_true",
                    help="防呆 3 題全不中也續跑。只給「已知格式崩潰」的模型用，"
                         "且要先用 base 模型驗過管線")
    ap.add_argument("--differs-from", default=None,
                    help="參照 jsonl（例如 base 的逐題結果）。前 20 題輸出若與參照逐字全同"
                         "就中止 —— 抓「server 沒掛上 adapter、量到 base」的事故"
                         "（2026-08-25 踩過：4 輪 adapter 評測全是 base）")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--out-dir", default="results/week4")
    a = ap.parse_args()

    out_dir = ROOT / a.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"eval_{a.tag}.json"
    if summary_path.exists():
        sys.exit(f"❌ {summary_path} 已存在。換 --tag 或先移走 —— 不覆蓋既有結果。")

    if a.dataset == "tmmluplus":
        subjects = a.subjects or TMMLU_SUBJECTS
        data = load_tmmluplus(subjects, a.seed, a.limit)
        sys_prompt = SYS_BOX_ZH
    else:
        subjects = a.subjects or MMLU_SUBJECTS
        data = load_mmlu(subjects, a.seed, a.limit)
        sys_prompt = SYS_BOX_EN

    first_subject = subjects[0]
    preflight(a.server, a.model_name, sys_prompt, data[first_subject], a.max_tokens,
              a.thinking, allow_unparsed=a.allow_unparsed_probe)

    ref_outputs = None
    if a.differs_from:
        ref_outputs = [json.loads(l)["output"] for l in open(a.differs_from)]
        print(f"[差異防呆] 參照 {a.differs_from}（前 20 題輸出全同即中止）")

    per_subject, records = {}, []
    total_gen_tokens, total_gen_seconds = 0, 0.0
    n_same_as_ref = 0
    for s in subjects:
        qs = data[s]
        n_ok_s = n_ok_l = n_unparsed = 0
        for i, q in enumerate(qs):
            txt, ctok, dt = chat(a.server, a.model_name, sys_prompt, build_prompt(q),
                                 a.max_tokens, a.thinking, a.timeout)
            body = strip_thinking(txt)
            ps, pl = extract_strict(body), extract_lenient(body)
            ok_s, ok_l = ps == q["answer"], pl == q["answer"]
            n_ok_s += ok_s; n_ok_l += ok_l; n_unparsed += ps is None
            if ctok:
                total_gen_tokens += ctok; total_gen_seconds += dt
            records.append({"subject": s, "question": q["question"][:200],
                            "gold": q["answer"], "pred_strict": ps, "pred_lenient": pl,
                            "correct_strict": bool(ok_s), "correct_lenient": bool(ok_l),
                            "completion_tokens": ctok, "seconds": round(dt, 2),
                            "output": txt[:1500]})
            if ref_outputs is not None and len(records) <= 20:
                if len(ref_outputs) >= len(records) and \
                        txt[:1500] == ref_outputs[len(records) - 1]:
                    n_same_as_ref += 1
                if len(records) == 20 and n_same_as_ref == 20:
                    sys.exit("❌ 前 20 題輸出與參照逐字全同 —— server 在跑的是參照那個模型"
                             "（adapter 沒掛上 / 舊 server 還佔著 port）。\n"
                             "   先 `lsof -ti:1234 | xargs kill`，確認新 server 印出 adapter 載入"
                             "訊息再重跑。")
            if (i + 1) % 20 == 0:
                print(f"    {s} {i+1}/{len(qs)}  嚴格 {n_ok_s/(i+1):.3f}  "
                      f"無法解析 {n_unparsed/(i+1):.3f}", flush=True)
        n = len(qs)
        per_subject[s] = {"n": n, "acc_strict": n_ok_s / n, "acc_lenient": n_ok_l / n,
                          "unparsed_rate": n_unparsed / n}
        print(f"  {s:<32} n={n:<5} 嚴格 {n_ok_s/n:.4f}  寬鬆 {n_ok_l/n:.4f}  "
              f"無法解析 {n_unparsed/n:.4f}")

    tot = sum(v["n"] for v in per_subject.values())
    k = len(per_subject)
    res = {
        "tag": a.tag, "model_name": a.model_name, "server": a.server,
        "dataset": a.dataset, "subjects": subjects, "seed": a.seed,
        "thinking": a.thinking, "max_tokens": a.max_tokens, "n_questions": tot,
        "macro_acc_strict": sum(v["acc_strict"] for v in per_subject.values()) / k,
        "macro_acc_lenient": sum(v["acc_lenient"] for v in per_subject.values()) / k,
        "macro_unparsed": sum(v["unparsed_rate"] for v in per_subject.values()) / k,
        "micro_acc_strict": sum(v["acc_strict"] * v["n"] for v in per_subject.values()) / tot,
        "micro_acc_lenient": sum(v["acc_lenient"] * v["n"] for v in per_subject.values()) / tot,
        "micro_unparsed": sum(v["unparsed_rate"] * v["n"] for v in per_subject.values()) / tot,
        "throughput_tok_per_s": (total_gen_tokens / total_gen_seconds
                                 if total_gen_seconds else None),
        "throughput_note": "單一請求串行量測；只在同機同 engine 內可比，跨硬體不可比",
        "per_subject": per_subject,
    }
    with (out_dir / f"eval_{a.tag}.jsonl").open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary_path.write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(f"\n== {a.tag} ==  micro 嚴格 {res['micro_acc_strict']:.4f}  "
          f"macro 嚴格 {res['macro_acc_strict']:.4f}  無法解析(micro) {res['micro_unparsed']:.4f}")
    if res["throughput_tok_per_s"]:
        print(f"   吞吐 ≈ {res['throughput_tok_per_s']:.1f} tok/s（{res['throughput_note']}）")
    print(f"   → {summary_path}")


if __name__ == "__main__":
    main()
