#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dlm_cli_eval.py — 用 llama-diffusion-cli 逐題跑 DLM 的 TMMLU+ 品質評測（本機、免費）。

為什麼存在：DLM 的品質評測在本機沒有 server 可用（mlx_lm 不支援 diffusion_gemma、
llama-server 沒接 diffusion 解碼），但 llama-diffusion-cli 可以生成、而且會套 chat
template（2026-08-24 於 bench log 實證：有 thinking channel、繁中通順）。
所以這支把 week4_eval_server 的同一套計分（固定種子、strict/lenient、macro/micro、
剝 thinking channel）接到「每題 spawn 一次 CLI」的驅動上。

代價：每題含載入約 25–30 秒（權重 mmap 有 OS cache，載入不會每次從磁碟重讀）。
    每科 50 題 × 3 科 ≈ 1.2 小時；每科 100 題 ≈ 2.3 小時（掛機即可）。

用法：
    .venv/bin/python scripts/dlm_cli_eval.py --tag dlm26b_base_quick \
        --llama-bin ~/Projects/llama.cpp/build/bin/llama-diffusion-cli \
        --gguf models/dlm-gguf/diffusiongemma-26B-A4B-it-Q4_K_M.gguf \
        --limit 100 --allow-unparsed-probe

注意：
  * DLM base 沒被訓練過 \\box 格式，防呆 3 題可能全不中 —— 那是模型行為不是管線問題，
    和 Week 2 tuned 同一種情況，所以同樣提供 --allow-unparsed-probe。
  * engine 是 llama.cpp Q4_K_M：品質數字標註 engine，與 Colab（bf16）數字分開列。
"""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from week4_eval_server import (SYS_BOX_ZH, TMMLU_SUBJECTS, load_tmmluplus,  # noqa: E402
                               build_prompt, extract_strict, extract_lenient,
                               strip_thinking)


def cli_help(bin_):
    try:
        h = subprocess.run([bin_, "--help"], capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=30)
        return (h.stdout or "") + (h.stderr or "")
    except Exception:
        return ""


def ask(bin_, gguf, sys_prompt, user, max_tokens, use_sys, extra, timeout=900):
    base = [bin_, "-m", gguf, "-ngl", "99", "-c", "4096", "-n", str(max_tokens)]
    if use_sys:
        cmd = base + ["-sys", sys_prompt] + extra + ["-p", user]
    else:  # 沒有 -sys 就把指示併進 user turn（兩種模式擇一，整輪評測內固定）
        cmd = base + extra + ["-p", sys_prompt + "\n\n" + user]
    t0 = time.time()
    # errors="replace"：去噪中途的 token 可能組出非法 UTF-8 位元組，不能讓解碼炸掉
    r = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True,
                       text=True, encoding="utf-8", errors="replace", timeout=timeout)
    if r.returncode != 0:
        out_all = (r.stdout or "") + (r.stderr or "")
        if r.returncode < 0 and "total time:" in out_all:
            # 被 signal 殺在「收尾清理」階段（Metal cleanup 的 SIGTRAP，macOS 常見）：
            # 統計行都印出來了 = 生成已完整結束 → 數字可用，記 rc 供 summary 統計。
            return (r.stdout or ""), time.time() - t0, r.returncode
        # ⚠️ fail-fast：真的沒產出就立刻停（W2「全部成功零解析」與「300 題 0 秒空轉」教訓）
        tail = "\n".join(out_all.splitlines()[-12:])
        sys.exit(f"❌ CLI 退出碼 {r.returncode}，中止。最後輸出：\n{tail}\n"
                 f"   指令：{' '.join(cmd[:12])} …")
    return (r.stdout or ""), time.time() - t0, r.returncode


def extract_body(txt):
    """剝掉 thinking channel，回 (body, mode)。
    有關閉標記 → 正常剝；沒有（舊輸出/CLI 沒開 --special）→ fallback：
    取最後一個空行分隔的段落當回答候選（bench 樣本的答案就長在那裡）。"""
    def _drop_stats(s):
        return "\n".join(l for l in s.splitlines()
                         if not l.startswith(("total time:", "throughput:")))
    body = strip_thinking(txt)
    if body.strip():
        return _drop_stats(body), "tagged"
    raw = _drop_stats((txt or "").replace("<|channel>thought", ""))
    parts = [p for p in re.split(r"\n\s*\n", raw) if p.strip()]
    return (parts[-1] if parts else ""), "fallback-last-para"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llama-bin", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--max-tokens", type=int, default=512,
                    help="512 = 2 個 canvas：thinking 開著時 256 常被 thought 吃光，答案沒地方生")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--allow-unparsed-probe", action="store_true")
    ap.add_argument("--llama-extra", default="", help='附加 CLI 參數，如 "-ngl 26"')
    ap.add_argument("--out-dir", default="results/week4")
    a = ap.parse_args()

    import shlex
    extra = shlex.split(a.llama_extra or "")
    out_dir = ROOT / a.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"eval_{a.tag}.json"
    if summary_path.exists():
        sys.exit(f"❌ {summary_path} 已存在。換 --tag 或先移走 —— 不覆蓋既有結果。")
    assert Path(a.llama_bin).exists() and Path(a.gguf).exists()

    help_txt = cli_help(a.llama_bin)
    use_sys = "-sys" in help_txt
    print(f"system prompt 傳遞方式：{'-sys 旗標' if use_sys else '併入 user turn（CLI 無 -sys）'}")
    # 支援的話就把 thinking 關掉（和其他評測的 thinking-off 口徑對齊）；不支援就維持
    # template 預設（thinking on），results 會如實標註。
    thinking_mode = "template-default(on)"
    if "chat-template-kwargs" in help_txt:
        extra += ["--chat-template-kwargs", '{"enable_thinking":false}']
        thinking_mode = "off"
    print(f"thinking：{thinking_mode}")
    # --special（印出 <channel|> 等特殊 token）也要先探測 —— 這個 PR 的 CLI 參數表
    # 是精簡版，塞不認識的旗標會 rc=1 直接退出（2026-08-24 踩過：300 題 0 秒空轉）。
    if "--special" in help_txt:
        extra += ["--special"]
        print("特殊 token：--special 已開（thinking 關閉標記可見，剝除走 tagged 模式）")
    else:
        print("⚠️ CLI 不支援 --special：關閉標記可能被吞，剝除倚賴 fallback-last-para")

    subjects = a.subjects or TMMLU_SUBJECTS
    data = load_tmmluplus(subjects, a.seed, a.limit)

    # ---- 防呆 3 題 ----
    ok, probe_secs = 0, []
    for q in data[subjects[0]][:3]:
        txt, dt, rc = ask(a.llama_bin, a.gguf, SYS_BOX_ZH, build_prompt(q),
                          a.max_tokens, use_sys, extra)
        body, mode = extract_body(txt)
        got = extract_strict(body)
        print(f"  [防呆] 正解 {q['answer']} | 抽到 {got} | rc={rc} | {dt:.0f}s | "
              f"剝除模式={mode} | 前 80 字: {body.strip()[:80]!r}")
        ok += got is not None
        probe_secs.append(dt)
    if ok == 0 and not a.allow_unparsed_probe:
        sys.exit("❌ 3 題都抽不出 \\box{}。DLM base 本來就可能不套格式（量的就是這個）——"
                 "確認上面輸出是「通順但沒格式」而非亂碼後，加 --allow-unparsed-probe 重跑。")
    if ok == 0:
        print("  ⚠️ 3 題全不中但 --allow-unparsed-probe 已開（輸出應為通順散文，亂碼就停）\n")

    per_subject, records = {}, []
    n_total = sum(len(v) for v in data.values())
    per_q = sum(probe_secs) / len(probe_secs) if probe_secs else 30
    print(f"共 {n_total} 題，依防呆實測每題 ~{per_q:.0f}s，預估 {n_total*per_q/60:.0f} 分鐘。\n")
    done = 0
    for s in subjects:
        qs = data[s]
        n_ok_s = n_ok_l = n_unp = 0
        for q in qs:
            txt, dt, rc = ask(a.llama_bin, a.gguf, SYS_BOX_ZH, build_prompt(q),
                              a.max_tokens, use_sys, extra)
            body, mode = extract_body(txt)
            ps, pl = extract_strict(body), extract_lenient(body)
            n_ok_s += ps == q["answer"]; n_ok_l += pl == q["answer"]; n_unp += ps is None
            done += 1
            records.append({"subject": s, "gold": q["answer"], "pred_strict": ps,
                            "pred_lenient": pl, "seconds": round(dt, 1), "strip_mode": mode,
                            "returncode": rc, "output": txt[:1200]})
            if done % 10 == 0:
                i = len([r for r in records if r["subject"] == s])
                print(f"    {s} {i}/{len(qs)}  嚴格 {n_ok_s/i:.3f}  無法解析 {n_unp/i:.3f}"
                      f"  （總進度 {done}/{n_total}）", flush=True)
        n = len(qs)
        per_subject[s] = {"n": n, "acc_strict": n_ok_s / n, "acc_lenient": n_ok_l / n,
                          "unparsed_rate": n_unp / n}
        print(f"  {s:<32} n={n} 嚴格 {n_ok_s/n:.4f} 寬鬆 {n_ok_l/n:.4f} 無法解析 {n_unp/n:.4f}")

    tot = sum(v["n"] for v in per_subject.values()); k = len(per_subject)
    res = {"tag": a.tag, "engine": "llama.cpp " + (re.search(
               r"(Q\d[\w_]*|IQ\d[\w_]*)", Path(a.gguf).name) or [""])[0],
           "model": Path(a.gguf).name, "dataset": "tmmluplus", "subjects": subjects,
           "seed": a.seed, "thinking": thinking_mode, "sys_mode":
           "-sys" if use_sys else "merged-into-user", "n_questions": tot,
           "max_tokens": a.max_tokens,
           "macro_acc_strict": sum(v["acc_strict"] for v in per_subject.values()) / k,
           "macro_acc_lenient": sum(v["acc_lenient"] for v in per_subject.values()) / k,
           "macro_unparsed": sum(v["unparsed_rate"] for v in per_subject.values()) / k,
           "micro_acc_strict": sum(v["acc_strict"] * v["n"] for v in per_subject.values()) / tot,
           "micro_acc_lenient": sum(v["acc_lenient"] * v["n"] for v in per_subject.values()) / tot,
           "micro_unparsed": sum(v["unparsed_rate"] * v["n"] for v in per_subject.values()) / tot,
           "per_subject": per_subject,
           "n_signal_exits": sum(1 for r in records if r["returncode"] < 0),
           "note": ("逐題 spawn llama-diffusion-cli；品質數字標註 engine，與 Colab bf16 分開列。"
                    "n_signal_exits = 生成完成後在 Metal 清理階段被 signal 殺的次數（統計行齊全，"
                    "數字有效）")}
    with (out_dir / f"eval_{a.tag}.jsonl").open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary_path.write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(f"\n== {a.tag} ==  micro 嚴格 {res['micro_acc_strict']:.4f}  "
          f"macro 嚴格 {res['macro_acc_strict']:.4f}  無法解析 {res['micro_unparsed']:.4f}")
    print(f"→ {summary_path}")


if __name__ == "__main__":
    main()
