# AR vs DLM 成本對照（M4 Pro / 單請求）

AR engine：MLX 4-bit｜DLM engine：llama.cpp Q4_K_M
⚠️ **兩邊 engine／量化格式不同**：比值僅供方向參考。要嚴謹比較，AR 也用同一個 engine 重量一次（例如兩邊都跑 llama.cpp 同一個 Q4_K_M）。（W2 教訓：MLX 4-bit = 4.50 bit/param，GGUF Q4_K_M 的位元組成不同。）

| 指標 | AR gemma-4-26B-A4B | DLM diffusiongemma-26B-A4B | 比值 DLM/AR |
|---|---:|---:|---:|
| 載入耗時 | 5.3 s | — | — |
| 權重佔用 | 13.23 GiB | — | — |
| 生成峰值 | 13.37 GiB | 18.62 GiB | 1.39× |
| 生成吞吐 | 55.8 tok/s | 13.5 tok/s | **0.24×** |

判讀（對回 Week 2 roofline）：AR 實測 53.6 tok/s ≈ 頻寬上限的 48%；
若 DLM 比值 <4×，落差來自 M4 Pro 沒有「閒置算力」可以吃（H5 已證 batch=1 就算力飽和），
官方 4× 是 GPU 算力閒置場景的數字 —— 這本身就是 roofline 的一次成功預測。
若比值 >1×，代表擴散把「每 token 讀一次全部權重」攤到 canvas 上，頻寬瓶頸被繞開了。