# GB10 フォーク設計根拠 — 何をどう変えたか

English: [DESIGN.en.md](DESIGN.en.md)

対象: `chai-md/comfy-kitchen` の `gb10-optimized`（`main..gb10-optimized` の7コミット）。
運用手順は [OPERATIONS.md](OPERATIONS.md)、数値は [MEASUREMENTS.md](MEASUREMENTS.md)。
**コードはリポジトリが正**（本稿は全文を貼らず、判断理由と落とせない制約だけを残す）。

---

## 1. 変更の全体像

| ファイル | 変更 | 詳細 |
| :--- | :--- | :--- |
| `comfy_kitchen/backends/triton/quantization.py` | 融合カーネル、タイル選択 `_int8_tile_config()`、`_ensure_python_headers()` のメモ化、`int8_linear()` を新カーネルへ | §2 |
| `comfy_kitchen/backends/cuda/__init__.py` | モード読取（env / ファイル）、`_should_use_triton_int8_dequant()`、`_cutlass_int8_dequant()` ラッパー、3 箇所の呼び出し置換、`_TRITON_INT8_MIN_M = 40` | §3 |
| `comfy_kitchen/__init__.py` | `int8_linear()` に `torch.compiler.is_compiling()` の早期リターン | §4 |
| `comfy_kitchen/tensor/scale_shift.py` | **新規**。per-segment 変調 `h*(1+scale[row])+shift[row]` を 1 パスで行う Triton カーネル | §5 |
| `tests/test_int8.py` | `TestTritonInt8GemmDequant`（INT8 の 92 ケース） | §6 |
| `tests/test_scale_shift.py` | scale_shift のビット一致（dtype 4 通り × セグメント数、非連続ビュー、カバレッジ、subnormal） | §6 |
| `patches/comfyui-minimax-modulation.patch` | ComfyUI 側の呼び出しフック | §7 |

変更は Python のみ。CUDA/C++ は触っていない（`_C.abi3.so` は再ビルドするが差分なし）。

---

## 2. Triton INT8 融合 GEMM（`triton/quantization.py`）

旧 `_int8_matmul_dequant_kernel` / `_int8_matmul_dequant_per_row_kernel` を削除し、**融合 GEMM +
dequant epilogue** 1 本に置き換えた。`int8_linear()` は `triton_int8_gemm_dequant()` を呼ぶ。

カーネルの要点（実装はリポジトリ参照）:

- int32 アキュムレータへの `tl.dot`、epilogue で `(int32 → fp32 * xs * ws + bias) → out.dtype`。
  演算順序は CUTLASS の visitor 列と同じで、`torch.equal` でビット一致する。
- アドレス計算は **int64**（`M*N` / `M*K` が 2^31 を超える CFG バッチ・長尺クリップで int32 が回り込む）。
- `EVEN` / `EVEN_K` の 2 段階でマスクを省く（全次元または K がタイル倍数のとき）。
- `G`（L2 スウィズル）でグループ内 M タイルを巡回させる。行優先にすると 176 → 105 TOPS（1.7 倍差）。
- `K == 0` は bias か 0 を書いて返す（CUTLASS は何も書かない。未初期化を返さない分だけ安全側）。

### 2.1 タイル表（`_int8_tile_config`）

GB10（48 SM）では 128×128 タイルが M=128 の倍数を要求し、端数 M で CTA 数が SM 数と合わず
大きく損する。実測帯域（min-of-N、N=K=4096、重みが L2 に載る領域）で次を選ぶ。

| M / 条件 | BM×BN, BK, G, warps, stages | 根拠 |
| :--- | :--- | :--- |
| M < 128 | 64×64, 64, 8, 4, 4 | 1.6〜1.7x |
| 128〜207 | 64×128, 64, 8, 4, 4 | 1.6x |
| 208〜287 | 128×64, 64, 8, 8, 4 | 1.15x |
| M ≥ 288 または `n*k > 20<<20` | 128×128, 64, 8, 8, 4 | 既定（大形状は帯域律速でタイル非依存） |
| 非 sm_121 | 128×128, **32**, 8, 8, 4 | 64KB 共有メモリ機（sm_75/RDNA1-4）で 4 段×16KB が 64KB ちょうどになりコンパイル失敗するため BK を半減 |

- L2 閾値は `_INT8_L2_WEIGHT_ELEMS = 20 << 20`。4096×4096（16MB）は薄いタイルで最大 2x、
  5120×5120（26MB）と 10240×4096（41MB）は ±4% で平坦。
- 既定タイルは `_INT8_GB10_DEFAULT_TILE = (128,128,64,8,8,4)`。M=160 では 128×128 だと 2×32=64 CTA =
  **1.33 波**になり、64 行タイルより 1.6x 遅い（これが帯域を切る理由）。
- **タイル選択は数値を変えない**（int32 累積は厳密、epilogue は出力要素ごと）。帯域ごとのビット一致は
  テストで固定している。
- `_ensure_python_headers()` は 1 プロセス 1 回にメモ化（旧実装は起動ごとに ~52 µs の stat を払っていた）。

> **罠**: ランチャーの `BM/BN/BK/G/warps/stages` は必ず変数で渡す。定数のままだとグリッドだけが
> タイル選択に従い、**結果が壊れる**（OPERATIONS §5.1 のビット一致検査で検出できる）。

### 2.2 残差融合（`residual` あり）

epilogue に `resid` / `rs` / `HAS_RESIDUAL` を足し、数値契約を
`out = residual + rs[n] * round_out(acc * xs * ws + bias)` とする。**bias 加算で一旦出力 dtype に丸める**
のが CUTLASS の `Add2 → Mul3 → Add4` と同じ順序で、ここを省くと一致しない（bias 無しでも Add2 は走る）。
epilogue が 2 枚目の `[BM,BN]` タイルを読むためレジスタ圧が上がり、`m >= 288` の残差形は `BN=64`
（`_INT8_GB10_RESIDUAL_TILE`）を使う（128×128 比 8〜12% 速い）。ランチャーは `residual is None` のとき
ダミーポインタ（`out`）を渡して `HAS_RESIDUAL=False` にする。

---

## 3. sm_121 ディスパッチ（`backends/cuda/__init__.py`）

### 3.1 ゲート

- `_TRITON_INT8_MIN_M = 40`。素の comfy-kitchen との実測クロスオーバー（`int8_linear`, 4096×4096）は
  M=24-32 0.78-0.82x / 40-48 1.06x / 56 1.59x / 64 2.17x で、**40 が「負けない下限」**。
  kernel 単体では M≤16 が CUTLASS 14.8〜24.5 µs / Triton 23〜28 µs、M=64 では Triton 25.7 µs vs
  CUTLASS 116.8 µs（4.5x）。分岐点は 24 だが、ランチャーの固定費が薄い M で効くため公開エントリで測る。
- モード: `COMFY_KITCHEN_INT8_MODE`（`off` / `cutlass` / `cublas` / `triton`）と
  `/etc/comfy-kitchen-int8.mode`。読取は `errors="replace"` ＋ `except (OSError, ValueError)` で
  import を壊さない。
- 実行時チェック: `m >= 40`、`k % 16 == 0`、`k < 131072`（int32 飽和）、2 次元・連続・dtype・形状、
  `x_scale`/`w_scale`/`bias` の連続性と要素数、`x.get_device() == torch.cuda.current_device()`、
  `stream_ptr == torch.cuda.current_stream().cuda_stream`、`_should_use_triton_int8_dequant()`。
  Triton は「現在のデバイス・現在のストリーム」で起動するため、最後の 2 つは誤デバイス起動・競合の防止。
- 例外時は `_triton_int8_broken` を立て、**stderr に 1 回だけ**理由を出して以降 CUTLASS に固定する
  （無言のフォールバックは使わない）。

### 3.2 呼び出し置換

`_int4_linear_via_int8_values()`、`int8_linear()`、`w4a8_int8_linear()` の 3 箇所を
`_cutlass_int8_dequant(...)` に置換し、int8 / int4(w4a8) / w4a8 の全経路が同じカーネルを使う。
CUTLASS へ落ちるときは `w_scale` を `reshape(-1)[:1].expand(n).contiguous()` で materialise する
（`(1,1)` スケールでのクラッシュ回避）。

---

## 4. `torch.compile` 経路（`comfy_kitchen/__init__.py`）

`int8_linear()` の冒頭で、`torch.compiler.is_compiling()` かつ `residual is None` かつ
`requires_grad` がどの入力にも立っていない場合だけ custom op の高速パスへ入る。
custom op には autograd 式が無く、meta 実装が residual のブロードキャストを無視するため、
それ以外は従来どおり Python 実装を trace させる（訓練で使うなら autograd 登録が別途必要）。

---

## 5. 変調融合 `tensor/scale_shift.py`（新規）

DiT の pre-norm ブロックが量子化線形の直前に行う per-segment 変調

```python
for a, b, row in segments:
    h[a:b].mul_(1.0 + scale[row].to(h.dtype)).add_(shift[row].to(h.dtype))
```

を 1 パスで実行する。torch は bf16 行列を 2 回 read-modify-write するため GB10 で 115〜150 GB/s しか
出ない（in-place は write-allocate で帯域が半減する）。カーネルは out-of-place で 222 GB/s、実測 2〜3 倍。

**ビット一致のために落とせない点が 3 つある。**

1. **Triton は `x*s + b` を FMA に融合し、`mul_()` が行う中間 bf16 丸めを消す。**
   整数演算で RNE を強制する（bf16: `(bits + 0x7FFF + ((bits>>16)&1)) & -65536`）。fp16 は指数部が
   5 bit で subnormal 域の丸めが変わるため `cvt.rn.f16.f32` を使う。
2. **変調テンソルは先に活性の dtype へ丸める**（`_mod_row` の `.to(h.dtype)`）。H3 の curve チェックポイントは
   adaLN を fp32 で持つので、これを飛ばすと **38% の要素がずれる**（bf16 変調では再現しない）。
3. **`scale`/`shift` は非連続ビューで来る。** ComfyUI は `[R, n*cols]` から `[R, cols]` を切り出すため
   行ストライドが `n*cols`。`row*cols` で索引すると mod 行 0 以外が全部ずれる（実測: 15787 行中 802 行が不一致）。
   呼び出し前に `.contiguous()` する。活性 `h` 側が非連続なら例外（誤ったオフセットで読み書きする）。

```python
from comfy_kitchen.tensor.scale_shift import scale_shift, row_mod_from_segments
row_mod = row_mod_from_segments(h.shape[0], segments, h.device)
out = scale_shift(h, scale, shift, row_mod)   # 新しいテンソルを返す（h は不変）
```

セグメントは `range(rows)` を隙間なく覆うこと（欠落は例外）。per-token の mod 行テンソルにも対応する。

---

## 6. テスト（回帰の安全網）

`tests/test_int8.py` の `TestTritonInt8GemmDequant` と `tests/test_scale_shift.py`。合格条件:

| テスト | 内容 |
| :--- | :--- |
| `test_triton_int8_gemm_dequant_matches_cutlass` | 3 形状 × dtype3 × bias2 × per-channel2 で CUTLASS と `torch.equal` |
| `test_tile_config_selection` | 帯域境界（24/127/128/207/208/256/288）、L2 ゲート、非 GB10 は既定タイル |
| `test_thin_m_tiles_match_cutlass` | M=24〜288 の 15 点で帯域ごとのビット一致 |
| `test_swizzle_and_even_paths_match_cutlass` | `gid>=1` のグループ分解（M=1100/1664）と EVEN 全開パス（384/1024） |
| `test_cuda_int8_linear_routes_match` | CUDA バックエンド経由（量子化＋GEMM）で cutlass と triton が一致 |
| `test_dispatch_gate_opens_below_128` | M=16/23 は CUTLASS、24/64/127 は Triton（スパイで検証） |
| `test_python_header_check_is_memoized` | ガードが 1 回しか stat しない（CPATH は monkeypatch で復元） |
| `test_sm121_dispatch_policy` | cap `(12,1)` 判定と env モード |
| `test_cutlass_int8_dequant_fail_open_fallback` | 例外時に CUTLASS へ落ち `_triton_int8_broken` が立つ |
| `test_edge_shapes_match_exact_reference` | N=1/7/33（不揃え）と K<BK。CUTLASS ではなく厳密参照と比較（§9.4） |
| `test_non_contiguous_scales_never_reach_triton` | `expand(n)` スケールは CUTLASS に落ち、連続版と同値 |
| `test_two_dim_weight_scale_falls_back` | `(1,1)` の weight_scale でクラッシュしない |
| `test_residual_epilogue_matches_cutlass` | 残差融合 epilogue が CUTLASS と `torch.equal`（4形状×dtype3×bias2） |
| `test_residual_routes_match_end_to_end` | `ck.int8_linear(residual=...)` の cutlass/triton 経路一致（スパイで Triton 実走を確認） |
| `test_residual_negative_zero_matches_cutlass` | bias 無しの `-0.0` 符号（CUTLASS は `+0.0` にする） |
| `test_cpu_residual_scale_does_not_poison_triton` | CPU スケールでゲートが落ち、`_triton_int8_broken` が立たない |

実装はリポジトリの `tests/test_int8.py` をそのまま参照（この文書の付録ではなくコードを正とする）。

---

## 7. ComfyUI 側フック（`patches/comfyui-minimax-modulation.patch`）

ComfyUI は上流クローンなので、フックはパッチとして本リポジトリに同梱している（適用は OPERATIONS §4）。
`_mod_scale_shift()` が comfy-kitchen のカーネルを使い、前提が崩れたときは stderr に 1 回だけ理由を出して
torch 実装（`_mod_scale_shift_ref`）に戻る。理由を出さないフォールバックは使わない
（以前、カーネル側の不具合を隠した）。

---

## 8. やらない最適化（実測済みの結論）

| 案 | 実測 / 根拠 | 判定 |
| :--- | :--- | :--- |
| tcgen05（第5世代TC）への書き直し | Triton 3.9 は sm_121 の INT8 を tcgen05 に lowering しない（TMA 記述子版でも `mma.sync.m16n8k32`）。cuBLAS の最速経路（FP8）が 178 TOPS ＝ 本カーネル 176.5 と同等で、上積みは最大2割・実質数% | **やらない** |
| FP8 へ切り替え | レガシー経路では INT8 と同レート（105.6 vs 105.3 TOPS）。カーネル単体でも 1.03〜1.05x（`bench/bench_fp8_vs_int8.py`）。重みは同じ 1 byte/要素で帯域差なし。仮数 e4m3 の 3 bit になりビット一致も失う | **やらない** |
| 活性量子化パスの融合 | 既に 226〜235 GB/s（ピーク 273 の 83〜86%）。融合すると A トラフィックが 2 倍になり逆効果 | **やらない** |
| 回転（256グループ Hadamard）の自前カーネル | 0.91x（cuBLAS が既に 228 GB/s の帯域天井） | **やらない** |
| 大形状向け自動チューナ | 形状ごとに最適が入れ替わり（2048×5120 は stages=3、512×4096 は stages=4）利得 ±5%。測定コストが上回る | **やらない** |
| `G`（L2 スウィズル）の削除 | 行優先順序にすると 176 → 105 TOPS（1.7 倍差）。このカーネルで最も効く部分 | やらない |
| `num_ctas=2` / `warp_specialize` / persistent / BK 変更 | −10〜−110% | 不可 |
| SA3 へ切替（attention） | 1.02x、cos 0.982（SA2 は 0.9992） | 不可 |
| VAE デコードを GPU バッファ + 一括コピーに | 103 フレーム (768×1344) で 23.39 → 23.52 s（差なし）。ページング罠自体は再現する（1.28GB の未タッチ領域への書き込み 10.1 s → `zero_()` 後 0.03 s）が、この VAE のチャンクコピーでは発生しない | 不要 |

---

## 9. レビュー記録（2026-10-31）

Claude Code（`--effort high`、Read のみ）と Antigravity（Gemini 4 Argon High、sandbox）の2系統で
事前レビューし、実コードで再現できた指摘だけ反映した。**両者の結論は割れた**（Claude: BLOCKING 0 /
agy: BLOCKING 4）ため、以下は1件ずつ検証した結果。

### 9.1 反映した指摘

| 指摘 | 検証 | 対応 |
| :--- | :--- | :--- |
| Triton は「現在のデバイス・現在のストリーム」で起動し、`x.device`/`stream_ptr` と食い違うと誤デバイス起動・競合になる | 妥当 | ゲートに `x.get_device() == torch.cuda.current_device()` と `stream_ptr == torch.cuda.current_stream().cuda_stream` を追加（不一致は CUTLASS） |
| **非連続スケールで OOB 読み出し**: `expand(n)` は `numel==n, stride==0` なので `reshape(-1)` が no-op になり、1要素バッファの先を読む | **実機で再現**（`stride=(0,)`, `is_contiguous()==False`） | ゲートに `x_scale/w_scale/bias` の `is_contiguous()` を追加＋ランチャ側で `.contiguous()` を materialize |
| `(1,1)` の `weight_scale` で CUTLASS フォールバックが `expand` に失敗してクラッシュ（`m < 24` で顕在化） | **実機で再現**（`RuntimeError: expand(...[1,1], size=[64])`） | `w_scale.reshape(-1)[:1].expand(n)` に修正（3箇所） |
| 非 `sm_121` で GB10 タイルを強制すると 64 KB 共有メモリ機（sm_75/RDNA）でコンパイル失敗 | 妥当（4段×16KB＝64KB） | `_INT8_PORTABLE_TILE = (128,128,32,8,8,4)`（32 KB）を非 GB10 に割当 |
| `torch.compile` 経路が autograd と residual ブロードキャストを壊す | 妥当（custom op に autograd 無し、fake 実装が residual を無視） | 高速パスは `residual is None` かつ `requires_grad` 無しのときだけに限定。それ以外は従来どおり Python を trace |
| `/opt/py312-include` から **別バージョンの** `Python.h` を借りる | 妥当（ABI 破壊リスク） | 同一 `major.minor` のみ許可（`python3.12` フォールバックを削除） |
| モードファイルの不正バイトで `UnicodeDecodeError` → import 失敗 | 妥当 | `errors="replace"` ＋ `except (OSError, ValueError)` |
| int32 飽和は K ≥ 131072 で分岐し得る | 妥当（理論値） | ゲートに `k < 131072` |
| テスト不足（`gid>=1`、EVEN、E2E、edge shape、スケール連続性） | 妥当 | 5テスト追加（§6） |

### 9.2 反映しなかった指摘（理由つき）

| 指摘 | 判断 |
| :--- | :--- |
| `rk * sbk` など K 方向のストライド積が int32 で溢れ得る | 閾値が `sbk >= 33,554,432`（＝転置重みで N が3300万行）。現実の形状で到達せず、int64 化はレジスタ圧と速度に影響するため見送り。コメントで明示 |
| スウィズルの端数グループで M タイル順が巡回する | 全単射で正しさに影響なし（L2 局所性のみ）。`pm = fm + ((pid % (G*npn)) % gs)` は等価なので変更せず |
| `_DISABLE_CUTLASS_INT8` が import 時固定で、後からの env 変更が効かない | 上流からの既存設計。挙動を変えると事故になるため保留（§10） |
| 残差融合パス（`cutlass_int8_dequant_residual`）が Triton を使わない | **対応した**（§2.2 / MEASUREMENTS §9）。epilogue に `residual + rs*round_out(...)` を実装し、CUTLASS とビット一致を24ケースで確認 |
| `w4a8_int8_linear` の `s_channel` 正規化・`_DISABLE_CUTLASS_INT8` 尊重 | 連続性はゲートで担保済み。それ以外は上流挙動を維持 |

### 9.3 ラウンド2レビュー（Claude Opus 5.5, 2026-10-31）

残差融合の追加分に対して再レビュー。**epilogue の数式は CUTLASS と全項目一致**（Mul0/Mul1 の順序、
Add2 の丸め、bias 無しでも丸める点、Mul3/Add4、rscale のブロードキャスト、fp32/fp16 の挙動、
residual ロードの stride とマスク）と確認された上で、5件の指摘。

| 指摘 | 対応 |
| :--- | :--- |
| CPU（他デバイス）の `residual_scale` がゲートを通り、Triton 内で例外→`_triton_int8_broken` が立ち**セッション中ずっと遅い経路**になる | ゲートに `residual.device == out.device` と `residual_scale.device == out.device` を追加 |
| `test_residual_routes_match_end_to_end` が「両方 CUTLASS」でも通ってしまう（自明な合格） | スパイで `residual is not None` 付きの Triton 呼び出しを検証 |
| ランチャーが `False` を返すと CUTLASS 残差パスも飛ばして融合を失う（size-1 次元の stride 差で起こり得る） | `if triton_int8_gemm_dequant(...): return True` にして False なら CUTLASS に落とす |
| bias 無しのとき `-0.0` の符号が CUTLASS と違う（CUTLASS は Bias ビジターで `+0.0` を足す） | `HAS_RESIDUAL` かつ bias 無しのとき `out = out + 0.0` を追加 |
| `_aligned16(resid_2d)` と `residual_scale.is_contiguous()` は CUTLASS の制約なのに Triton 経路まで CUTLASS に落としている | アラインメント検査を CUTLASS 分岐へ移動（非整列は `return False`）、rscale はランチャで `.contiguous()` |

テストは scalar `residual_scale` / scalar `weight_scale` / `-0.0` / CPU スケールの4ケースを追加
（当時 **263 passed**、現在は 290）。

### 9.4 上流のバグ（N が 16 の倍数でない場合）

**上流 CUTLASS カーネルは壊れた出力を返す。** 実測（M=24, K=256, bf16）:

| N | Triton の誤差（厳密参照比） | CUTLASS の誤差 |
| ---: | ---: | ---: |
| 1 | 0.000000 | 41.0 |
| 7 | 0.000000 | 5,275,648.5 |
| 33 | 0.000000 | **NaN** |
| 16 / 48 / 64 | 0.000000 | 0.000000 |

`torch._int_mm`（cuBLAS INT8）も `N % 8 != 0` を拒否するため、この領域は元々サポート外と見られる。
Triton 側は厳密参照と一致するので、**ゲートは N で制限しない**（CUTLASS より正しい方を選ぶ）。
テストは CUTLASS 比較ではなく厳密参照比較にした（`test_edge_shapes_match_exact_reference`）。

---

## 10. 既知の制限（意図的に変更しない）

- 残差融合は Triton の epilogue で対応済み（§2.2）。CUTLASS 残差カーネルとビット一致する。
- **非 GB10 では autotune が無い**。上流の autotuned カーネル2本は置換・削除した。ROCm 等で詰めるなら
  上流の `_int8_matmul_dequant_kernel` を復活させること。
- **`K==0` の挙動が CUTLASS と違う**（Triton は bias か 0 を書く、CUTLASS は何も書かない）。
  未初期化メモリを返さない分だけ安全側。
- **`COMFY_KITCHEN_INT8_MODE` は呼び出しごとに読むが、`_DISABLE_CUTLASS_INT8` は import 時固定**
  （上流由来の非対称）。

---

## 11. 謝辞 / Acknowledgements

本フォークの方向づけと運用の慣習は、次の公開ノートに負っています。

### kanthai / gb10-comfyui-tuning（MIT）

<https://github.com/kanthai/gb10-comfyui-tuning>

DGX Spark (GB10) 上で ComfyUI + comfy-kitchen を実測したノートとパッチ集。とくに参照した点:

| 参照した内容 | 本フォークでの対応 |
| :--- | :--- |
| `docs/02-int8-triton-gemm.md` — comfy-kitchen の CUTLASS 経路は sm_121 でピークの 37%、Triton の融合 dequant GEMM なら 88% | 本フォークの出発点。MEASUREMENTS §4 の「N が広い射影層で 1.5〜2.4 倍」は同じ現象を shape 別に測り直したもの |
| `docs/03-torch-compile-int8.md` — `torch.compile` 下で INT8 が落ちる原因 | `int8_linear()` の `torch.compiler.is_compiling()` 早期リターン（§4） |
| モード切替の慣習 `COMFY_KITCHEN_INT8_MODE` / `/etc/comfy-kitchen-int8.mode`（`off` で素に戻す） | `_read_int8_mode()` とファイル方式（§3） |
| Triton の実行時コンパイルに `Python.h` が要る件 | `_ensure_python_headers()` とそのメモ化（§2.1） |
| A/B・同一シード・ビット一致で確認する計測姿勢 | OPERATIONS §5 の検証手順と `bench/` のスクリプト群 |

### 本フォーク側で独自に実装・検証した部分

上記リポジトリのコードは取り込んでいません（向こうは `.pth` で読み込むモンキーパッチ、
こちらは comfy-kitchen 本体へのフォーク）。以下は本フォークで書いたもの:

- タイル表（M 帯ごとに 64/128 行を選ぶ）と、GB10 以外のデバイス向けフォールバックタイル
- residual 融合 epilogue（CUTLASS の型順序に合わせ、`torch.equal` でビット一致を検証）
- `comfy_kitchen/tensor/scale_shift.py`（変調の融合。FMA 融合による中間丸め消失、
  fp32 adaLN の事前丸め、非連続ビューの 3 点は実測で特定）
- 形状別の実測（MEASUREMENTS）と回帰テスト（`tests/test_int8.py`, `tests/test_scale_shift.py`）

本フォークの誤りは本フォーク側の責任です。原典のライセンスは上記リポジトリの `LICENSE`（MIT）を参照してください。
