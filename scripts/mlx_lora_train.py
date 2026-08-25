#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mlx_lora_train.py — 帶 Router stop_gradient 補丁的 mlx_lm.lora 啟動器（Week 4）。

【為什麼需要這支】
mlx core 0.32.0 → 0.32.1 之後，gather 類算子對「indices 落在可微分路徑上」
從默默當作不可微，改成直接丟錯：
    ValueError: [gather] Cannot calculate VJP with respect to indices.
而 mlx-lm 0.31.3 的 gemma4 Router 裡，top_k_indices 沒有 stop_gradient 就被拿去
take_along_axis / per_expert_scale[...]。訓練 26B-A4B（任何會讓梯度流過 MoE 層的
LoRA，包括方案 A/B）就會炸。Week 2 能跑是因為當時 mlx 還是 0.32.0。

【這支做什麼】
1. import mlx_lm 的 gemma4_text，把 Router.__call__ 換成「逐行相同、只多一行
   mx.stop_gradient(top_k_indices)」的版本 —— indices 本來就不可微，這是 MoE
   訓練的標準寫法（switch_layers.py 裡其他模型都有做）；
   梯度照樣經由 top_k_weights（softmax 的 values 路徑）流回 router.proj，
   所以方案 B（router-only）的訓練不受影響。
2. 原封不動把 CLI 參數交給 mlx_lm.lora 的 main()。

【用法】＝ mlx_lm.lora 的替身：
    python scripts/mlx_lora_train.py --config configs/week4/lora_26b_planA.yaml --iters 30
    python scripts/mlx_lora_train.py --config configs/week4/lora_26b_planA.yaml
    python scripts/mlx_lora_train.py --config configs/week4/lora_26b_planB.yaml

【不patch行不行】可以改鎖版本：uv pip install "mlx==0.32.0" mlx-metal==0.32.0
但 DiffusionGemma 需要新 mlx，來回切版本比一個 20 行的 patch 更容易出事。
（E4B/gemma-3 等 dense 模型沒有 MoE gather，用原生 mlx_lm.lora 即可，不用這支。）
"""

import sys

import mlx.core as mx
from mlx_lm.models import gemma4_text


def _patched_router_call(self, x: mx.array):
    """gemma4_text.Router.__call__ 的逐行複製，只加一行 stop_gradient（有標註）。"""
    x = mx.fast.rms_norm(x, self.scale * self._root_size, self.eps)

    expert_scores = self.proj(x)

    top_k_indices = mx.argpartition(
        expert_scores, kth=-self.config.top_k_experts, axis=-1
    )
    top_k_indices = top_k_indices[..., -self.config.top_k_experts:]
    top_k_indices = mx.stop_gradient(top_k_indices)   # ← 唯一的新增行

    top_k_weights = mx.take_along_axis(expert_scores, top_k_indices, axis=-1)
    top_k_weights = mx.softmax(top_k_weights, axis=-1)
    top_k_weights = top_k_weights * self.per_expert_scale[top_k_indices]

    return top_k_indices, top_k_weights


def main():
    # 防呆：確認被 patch 的函式長得和我們以為的一樣（mlx-lm 升級後這裡會提醒你重看）
    import inspect
    src = inspect.getsource(gemma4_text.Router.__call__)
    for token in ("argpartition", "take_along_axis", "per_expert_scale"):
        assert token in src, (
            f"mlx-lm 的 Router.__call__ 已改版（找不到 {token}）——"
            "先確認新版是否已自帶 stop_gradient，再決定要不要繼續用這個 patch。")
    if "stop_gradient" in src:
        print("[patch] 注意：安裝版 Router 已含 stop_gradient，補丁可能不再必要（照樣可跑）。")

    gemma4_text.Router.__call__ = _patched_router_call
    print("[patch] gemma4 Router.__call__ 已加上 stop_gradient(top_k_indices) ✅")

    from mlx_lm.lora import main as lora_main
    lora_main()


if __name__ == "__main__":
    sys.argv[0] = "mlx_lm.lora"   # 讓 --help 與錯誤訊息顯示熟悉的名字
    main()
