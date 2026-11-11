# GB10 フォーク実測リファレンス

English: [MEASUREMENTS.en.md](MEASUREMENTS.en.md)

測定環境: dgxspark1 / GB10 / 48 SM / 2190 MHz（上限 2418、実運用は 2200 cap）/ Triton 3.9.0 /
torch 2.15.0+cu132（2026-10-30）。設計は [DESIGN.md](DESIGN.md)、手順は [OPERATIONS.md](OPERATIONS.md)。

---

## 1. 測定の作法

- 同一構成でも実行間で **10〜15% 揺れる**（クロック）。必ず min-of-3〜5、可能なら構成をラウンドロビンで回す。
- **10% 未満の差は有意とみなさない。**
- 比較は 2 状態（素の comfy-kitchen = `switch_kernel_state.sh A` / 本フォーク = `gb10-optimized`）で
  同一スクリプトを流して取得する。
- 「GEMM 単体」と「`int8_linear` 全体（量子化込み）」を混ぜない。§8 は後者、§6〜§7 は前者。

---

## 2. H3 エンドツーエンド（M=15,787）

1 イテレーションの内訳。素の comfy-kitchen（上流 `cdab478`）との比較（`bench/bench_h3_breakdown.py`）。

| 段階 | 素 | 本フォーク | 短縮 | 変更元 |
| :--- | ---: | ---: | ---: | :--- |
| INT8 GEMM（4線形 × 50層） | 7.87 s | 3.82 s | **−4.05 s** | 本フォークの Triton 経路 |
| 変調 `scale/shift`（×100） | 0.49 s | 0.16 s | **−0.33 s** | `tensor/scale_shift.py` |
| 回転＋量子化（×200） | 0.32 s | 0.32 s | 0 | 変更なし（帯域天井） |
| attention（×50） | 2.69 s | 2.69 s | 0 | フォーク外（起動フラグ） |
| **合計** | **11.4 s** | **7.0 s** | **−4.4 s（1.63x）** | |

効果があるのは **INT8 GEMM と 変調**。attention はフォーク起因ではなく、ComfyUI の起動フラグ
`--use-sage-attention` による（SA2 53.9 ms/層、SDPA だと 96.6 ms/層でこの段階が 2.69 → 4.86 s）。

> 注: 本表の「素」は v0.2.40（`cdab478`）。フォークのベースは v0.2.41（`7d6a22a`）で、上流 #301
> （CUDA ブロック量子化の 64bit packed load/store 化と fused RMS-RoPE）を含むが、#301 は
> int8 convrot / `int8_linear` に触れないため本表の差分は不変。`0.2.41+gb10.1` で取り直した
> フォーク側は GEMM 3816 ms/forward、rotate+quant 285 ms、modulate 433 → 154 ms、
> attention 96.5 → 53.7 ms（合計 ~6.94 s）で、GEMM・変調・attention は本表と一致した。
>
> **変調の 0.16 s は ComfyUI 側フック適用時の値**（DESIGN §7）。フックを外すと torch 実装の 433 ms に戻り、
> 合計は 7.33 s / **1.55x**（GEMM だけで 4.05 s の短縮）。フックは 2026-11-11 以降 既定では適用しない。

---

## 3. レバーと効く条件

| # | レバー | 効く条件 | 大きさ | 効かない条件 |
| :--- | :--- | :--- | ---: | :--- |
| 1 | INT8 GEMM を Triton 経路へ | **N が広い**射影層（N ≳ 8192） | 1.5–2.4x | N ≤ 4096 では 1.05x |
| 2 | 変調の融合（`scale_shift`） | DiT の pre-norm 変調がある | 3.0x（その段階） | 変調の無いモデル |
| 3 | attention を SageAttention | ComfyUI の起動フラグ（コード変更なし） | 1.8x（その段階） | SA3 は 1.02x で不可 |

---

## 4. N 依存（M=15,787 固定・K=5,376）

| N | 素 comfy-kitchen | 本フォーク | 倍率 |
| ---: | ---: | ---: | ---: |
| 4096（K=4096） | 4.45 ms / 119 TOPS | 4.21 ms / 126 TOPS | 1.06x |
| 8192 | 22.24 ms / 62.5 TOPS | 9.38 ms / 148 TOPS | **2.37x** |
| 10752 | 24.44 ms / 74.7 TOPS | 11.88 ms / 154 TOPS | **2.06x** |
| 21504（H3 qkv） | 50.25 ms / 72.6 TOPS | 22.50 ms / 162 TOPS | **2.23x** |
| 28672（H3 fc1） | 69.06 ms / 70.5 TOPS | 29.21 ms / 167 TOPS | **2.36x** |

素の CUTLASS は N が広がると 119 → 62〜75 TOPS に低下し、Triton 側は 126 → 148〜167 に上がる。
射影層は N = hidden / 3·hidden / 2·ffn なのでこの帯に入る。

---

## 5. M 依存（N=K=4,096、重み 16MB = L2 圏）

### 5.1 素との比較

| M | 素 | 本フォーク | 倍率 |
| ---: | ---: | ---: | ---: |
| 24 | 32.1 µs | 33.1 | 0.97x（両方 CUTLASS） |
| 40 | 43.0 | 40.7 | 1.06x |
| 56 | 64.3 | 40.8 | **1.58x** |
| 64 | 88.3 | 40.4 | **2.18x** |
| 128 | 91.7 | 54.3 | **1.69x** |
| 512 | 230.3 | 141.5 | **1.63x** |
| 1024 | 289.7 | 304.0 | 0.95x（誤差圏） |
| 4096 | 1082.8 | 1079.6 | 1.00x |

M<40 は素と同じ CUTLASS に流す（Triton はランチャー固定費が高く M=24〜32 では 0.78〜0.82x と負ける）。
しきい値は `_TRITON_INT8_MIN_M = 40`。

### 5.2 本フォークの絶対性能（タイル別）

| M | TOPS | タイル |
| ---: | ---: | :--- |
| 32 | 43 | 64x64 |
| 64 | 83 | 64x64 |
| 160 | 119 | 64x128 |
| 192 | 135 | 64x128 |
| 256 | 128 | 128x64 |
| 384 | 171 | 128x128 |
| 512 | 155 | 128x128 |
| 1024 | 160 | 128x128 |
| 4096 | 178 | 128x128 |
| 27280 | 172 | 128x128 |

M=384 が 171 で M=512 が 155 なのは **ウェーブ量子化**（48 SM に対し CTA 数が 96 = 2 波ちょうど vs
128 = 2.67 波）。

---

## 6. 実用天井（GEMM 単体、量子化を含まない）

| 経路 | 2048×5120×5120 | 4096³ | 15787×21504×5376（H3 qkv） |
| :--- | ---: | ---: | ---: |
| 本カーネル INT8 | **176.5 TOPS** | 177.5 | **164.1** |
| CUTLASS INT8（素） | 137–181 | 137 | **72.1** |
| cuBLAS INT8 | 133 | 140 | — |
| cuBLAS FP8 | **178** | 162 | — |
| cuBLAS BF16 | 84 TFLOPS | 91 | — |

N が大きく重みが L2 を外れる形（H3 の qkv: 115MB）では素の CUTLASS が失速し、差が最も開く。

レガシー IMMA の理論天井 = 48 SM × 2.19 GHz × 1024 MAC/cycle ≈ **215 TOPS**（達成率 82%）。
cuBLAS 最速が 178 である以上、ライブラリ置換の余地はない。

注意: これは **GEMM 単体**の比較。エンドツーエンド（量子化込み）の差は **N の広さ**で決まり、
N=4096 ではほぼ差が出ない（1.06x）が、N ≥ 8192 の射影層では 1.5〜2.4x 開く（§4）。

---

## 7. レイヤー内訳（N=K=4096）

| M | 量子化 | GEMM | 合計 | GEMM 比率 |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 10.9 µs | 112.5 µs | 131.8 µs | 85% |
| 4096 | 212.6 µs | 783.4 µs | 1026 µs | 76% |
| 27280 | 1424 µs | 5334 µs | 6634 µs | 80% |

量子化は帯域天井（ピークの 83〜86%）に達しており、削る余地はない。

---

## 8. 上流（CUTLASS 経路）との差分 — `int8_linear` 全体、量子化込み

`COMFY_KITCHEN_INT8_MODE=cutlass`（＝上流相当）との比較。量子化パスは両者共通なので、比率は
GEMM 単体より圧縮されて見える。

| M (N=K=4096) | 本カーネル | 上流 CUTLASS | 倍率 | ビット一致 |
| ---: | ---: | ---: | ---: | :--- |
| 64 | 39.1 µs | 109.3 µs | 2.79x | ✓ |
| 160 | 54.7 µs | 137.0 µs | 2.50x | ✓ |
| 192 | 58.7 µs | 146.3 µs | 2.49x | ✓ |
| 256 | 80.6 µs | 150.0 µs | 1.86x | ✓ |
| 512 | 137.3 µs | 228.5 µs | 1.66x | ✓ |
| 4096 | 1013.9 µs | 1141.0 µs | 1.13x | ✓ |

---

## 9. 残差融合（`residual` 付き `int8_linear`）

CUTLASS の残差融合カーネル（`cutlass_int8_dequant_residual`）との比較。量子化込み。

| M (N=K=4096) | Triton | CUTLASS | 倍率 |
| ---: | ---: | ---: | ---: |
| 64 | 43.6 µs | 160.1 µs | **3.67x** |
| 160 | 63.1 µs | 199.5 µs | **3.16x** |
| 256 | 94.3 µs | 220.9 µs | **2.34x** |
| 512 | 159.2 µs | 288.8 µs | 1.81x |
| 1024 | 308.9 µs | 307.4 µs | 1.00x |
| 2048 | 642.8 µs | 702.9 µs | 1.09x |
| 4096 | 1430.8 µs | 1394.6 µs | 0.97x（誤差範囲） |

- 数値契約は `out = residual + rs[n] * round_out(acc * xs * ws + bias)`。**bias 加算で一旦出力 dtype に
  丸める**のが CUTLASS の `Add2 → Mul3 → Add4` ビジター列と同じ順序で、ここを省くと一致しない。
- epilogue が 2枚目の `[BM,BN]` タイルを読むためレジスタ圧が上がる。`m >= 288` の残差形は `BN=64`
  （`_INT8_GB10_RESIDUAL_TILE`）を使う（`128x128` 比で 8〜12% 速い）。
- M ≥ 1024 では CUTLASS 残差カーネルと同等。**小〜中 M で大きく勝ち、大 M で損はしない**のが位置づけ。

---

## 10. 再測定スクリプト

```bash
cd ~/comfy-kitchen && source .venv/bin/activate

python bench/bench_h3_breakdown.py fork     # H3 の内訳（GEMM/変調/回転/attention）
python bench/bench_int8_baseline.py fork    # 形状別 int8_linear
python bench/bench_thin_m.py fork           # M=16〜128 のクロスオーバー
python bench/bench_fp8_vs_int8.py fork      # FP8 との比較

~/archive/patches/switch_kernel_state.sh A  # 素に戻す
python bench/bench_h3_breakdown.py vanilla
```

`bench/` の生データ（`h3_breakdown_fork.json` / `h3_breakdown_vanilla.json`）はリポジトリに同梱。
上流が `_TRITON_INT8_MIN_M` の帯域やタイル表に触れたら、本稿の比較表を取り直す
（`~/archive/patches/bench_int8_baseline.py` を A / `gb10-optimized` の 2 状態で流す）。

---

## 11. 上流 #301 カーネルの帯域（sm_121 実測）

`0.2.41+gb10.1`、M=15792、bf16 入力。実効 GB/s は「読んだ量＋書いた量＋スケール」を時間で割った値。
天井は同サイズの bf16 copy（read+write）実測 **246.9 GB/s**（公称 273 の 90%）。

| カーネル | 実測 | 実効 GB/s | コピー天井比 |
| :--- | ---: | ---: | ---: |
| nvfp4 quantize | 1059.6 µs | 205.3 | 83% |
| mxfp8 quantize | 1067.0 µs | 241.3 | **98%** |
| int8 rowwise quantize | 1097.1 µs | 232.2 | 94% |
| fused rms_rope（in-place） | 4024.3 µs | 229.0 | 93% |
| rms_rope（out-of-place） | 4453.1 µs | 207.0 | 84% |

- MXFP8・int8 rowwise・fused rms_rope は実測天井に到達している。NVFP4 だけ 83% で、
  この形状で 0.18 ms/回の余地が残る（トラフィックは read 169.7 MB + packed 42.4 MB + scales 5.3 MB）。
  H3 の int8 経路は NVFP4 を使わないため、実運用への影響は小さい。
- rms_rope は H3 VAE が実際に呼ぶ経路（`rms_rope_split_half_`）。out-of-place 版は 84% で、
  推論時の in-place 経路より遅い。
