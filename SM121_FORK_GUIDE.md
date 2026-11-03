# GB10 (`sm_121`) 向け `comfy-kitchen` フォーク指示書

上流 `Comfy-Org/comfy-kitchen` をフォークし、**GB10 (sm_121) 用の INT8 GEMM 最適化を再適用**するための手順書です。上流の新リリースに追従するたびに、この文書の §3→§5 を繰り返せば同じ状態に戻せます。

- 対象機: dgxspark1（GB10 / 48 SM / 2190 MHz、クロック上限 2200）
- 前提: CUDA 13.x、Python 3.13、`torch 2.15+cu132`、**作業は `source .venv/bin/activate` した venv の中で行う**（§4.2）
- ベース: `comfy-kitchen` v0.2.40、コミット `cdab478`
- 変更は **Python のみ**（CUDA/C++ の変更なし。`_C.abi3.so` は再ビルドで作り直すが、差分は無い）
- 参考: [kanthai/gb10-comfyui-tuning](https://github.com/kanthai/gb10-comfyui-tuning)（謝辞は §11）

---

## 0. 何が効くのか（結論から）

H3 の実測（M=15,787）では、1 イテレーションがこう変わる。数値はすべて**素の
comfy-kitchen（上流 `cdab478`）**との比較で、同一スクリプトを 2 状態（`switch_kernel_state.sh A`
と `main`）で流して取得（`bench/bench_h3_breakdown.py`）。

| 段階 | 素 | 本フォーク | 短縮 | 変更元 |
| :--- | ---: | ---: | ---: | :--- |
| INT8 GEMM（4線形 × 50層） | 7.87 s | 3.82 s | **−4.05 s** | 本フォークの Triton 経路 |
| 変調 `scale/shift`（×100） | 0.49 s | 0.16 s | **−0.33 s** | `tensor/scale_shift.py` |
| 回転＋量子化（×200） | 0.32 s | 0.32 s | 0 | 変更なし（帯域天井、§7） |
| attention（×50） | 2.69 s | 2.69 s | 0 | フォーク外（起動フラグ、下記） |
| **合計** | **11.4 s** | **7.0 s** | **−4.4 s（1.63x）** | |

効果があるのは **INT8 GEMM と 変調**。attention はフォーク起因ではなく、ComfyUI の起動フラグ
`--use-sage-attention` による（SA2 53.9 ms/層、SDPA だと 96.6 ms/層でこの段階が 2.69 → 4.86 s）。

### 0.1 レバーと効く条件

| # | レバー | 効く条件 | 大きさ | 効かない条件 |
| :--- | :--- | :--- | ---: | :--- |
| 1 | INT8 GEMM を Triton 経路へ | **N が広い**射影層（N ≳ 8192） | 1.5–2.4x | N ≤ 4096 では 1.05x |
| 2 | 変調の融合（`scale_shift`） | DiT の pre-norm 変調がある | 3.0x（その段階） | 変調の無いモデル |
| 3 | attention を SageAttention | ComfyUI の起動フラグ（コード変更なし） | 1.8x（その段階） | SA3 は 1.02x で不可 |

### 0.2 GEMM の効き目は M ではなく N で決まる

M=15,787 固定・K=5,376:

| N | 素 comfy-kitchen | 本フォーク | 倍率 |
| ---: | ---: | ---: | ---: |
| 4096（K=4096） | 4.45 ms / 119 TOPS | 4.21 ms / 126 TOPS | 1.06x |
| 8192 | 22.24 ms / 62.5 TOPS | 9.38 ms / 148 TOPS | **2.37x** |
| 10752 | 24.44 ms / 74.7 TOPS | 11.88 ms / 154 TOPS | **2.06x** |
| 21504（H3 qkv） | 50.25 ms / 72.6 TOPS | 22.50 ms / 162 TOPS | **2.23x** |
| 28672（H3 fc1） | 69.06 ms / 70.5 TOPS | 29.21 ms / 167 TOPS | **2.36x** |

素の CUTLASS は N が広がると 119 → 62〜75 TOPS に低下し、Triton 側は 126 → 148〜167 に上がる。
射影層は N = hidden / 3·hidden / 2·ffn なのでこの帯に入る。

### 0.3 細い M は別レバー（同じカーネルのタイル表）

N=K=4,096（重み 16MB = L2 圏）で M を振った場合:

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

M<40 は素と同じ CUTLASS に流す（Triton はランチャー固定費が高く M=24〜32 では 0.78〜0.82x と
負ける）。しきい値は `_TRITON_INT8_MIN_M = 40`。

### 0.4 効かないと確認済み（再試行しない）

| 案 | 実測 | 判定 |
| :--- | :--- | :--- |
| 回転（256グループ Hadamard）の自前カーネル | 0.91x（cuBLAS が既に 228 GB/s の帯域天井） | 不可 |
| 量子化パスの融合 | 既に 226〜235 GB/s（ピークの 83〜86%） | 不可 |
| SA3 へ切替 | 1.02x、cos 0.982（SA2 は 0.9992） | 不可 |
| `num_ctas=2` / `warp_specialize` / persistent / BK 変更 | −10〜−110% | 不可 |
| tcgen05（CUTLASS 書き直し） | cuBLAS FP8 が 178 TOPS ＝ 本カーネルと同等 | 不可 |
| FP8 へ切替 | カーネル単体で 1.03〜1.05x（H3 形状、`bench/bench_fp8_vs_int8.py`）。重みは同じ 1 byte/要素で帯域差なし。仮数は e4m3 の 3 bit になり、ビット一致も失う | 不可 |

### 0.5 再測定の仕方

```bash
cd ~/comfy-kitchen && source .venv/bin/activate

python bench/bench_h3_breakdown.py fork     # H3 の内訳（GEMM/変調/回転/attention）
python bench/bench_int8_baseline.py fork    # 形状別 int8_linear
python bench/bench_thin_m.py fork           # M=16〜128 のクロスオーバー

~/archive/patches/switch_kernel_state.sh A  # 素に戻す
python bench/bench_h3_breakdown.py vanilla
```

測定の作法: 同一構成でも実行間で 10〜15% 揺れる。min-of-3〜5、構成はラウンドロビン、
10% 未満の差は有意とみなさない。

---

## 1. フォークとブランチ

| 項目 | 実態 |
| :--- | :--- |
| フォーク | `chai-md/comfy-kitchen`（`origin`、HTTPS） |
| 上流 | `Comfy-Org/comfy-kitchen`（`upstream`） |
| 作業ブランチ | **`main`**（トピックブランチは切っていない。コミットは上流 `cdab478` の直上に載る） |
| 変更の実体 | `git log --oneline cdab478..origin/main` の 5 コミット（下記） |
| ローカル | このリポジトリ。ビルド/テストは直下の `.venv`（`source .venv/bin/activate`）、ComfyUI から使う場合は ComfyUI 側の venv にも editable で入れる（§4.2） |

```bash
# フォーク作成 + clone（初回のみ）
gh repo fork Comfy-Org/comfy-kitchen --clone
cd comfy-kitchen
git remote add upstream https://github.com/Comfy-Org/comfy-kitchen.git
git switch main                     # フォークの main がそのまま作業ブランチ
git submodule update --init --depth 1   # 必須（§4.1）
```

履歴はこの 5 つ。**ブランチ名で探さず、この範囲で追う**（フォークの `main` が最新）。

| コミット | 内容 |
| :--- | :--- |
| `3fba74f` | Triton INT8 融合 GEMM＋タイル表＋ディスパッチ＋residual 融合＋`tensor/scale_shift.py` 初版＋テスト |
| `fdc854d` | `scale_shift`: 変調テンソルを活性 dtype に丸める（fp32 adaLN 対応。これが無いと H3 の絵が変わる） |
| `33e16e8` | `scale_shift`: 非連続な変調ビューを `.contiguous()` 化、カバレッジ検証を追加 |
| `3dc2406` | ComfyUI 側フックを `patches/comfyui-minimax-modulation.patch` として同梱 |
| `4b14781` | 本指示書の比較基準を素の comfy-kitchen に修正、Triton ゲートを 40 へ、計測スクリプト `bench/` を同梱 |

```bash
# 同じ状態を別マシンで再現する場合
git fetch origin && git switch main && git reset --hard origin/main
```

`sm121_optimize.patch` は `git diff` のスナップショットで**追跡していない**（コミットが正）。必要なら `git diff cdab478..HEAD -- comfy_kitchen tests > sm121_optimize.patch` で作り直せます。

---

## 2. 変更対象ファイル

| ファイル | 変更 |
| :--- | :--- |
| `comfy_kitchen/backends/triton/quantization.py` | 融合カーネル、タイル選択 `_int8_tile_config()`、`_ensure_python_headers()` のメモ化、`int8_linear()` を新カーネルへ |
| `comfy_kitchen/backends/cuda/__init__.py` | モード読取（env / ファイル）、`_should_use_triton_int8_dequant()`、`_cutlass_int8_dequant()` ラッパー、3 箇所の呼び出し置換、`_TRITON_INT8_MIN_M = 40` |
| `comfy_kitchen/__init__.py` | `int8_linear()` に `torch.compiler.is_compiling()` の早期リターン |
| `comfy_kitchen/tensor/scale_shift.py` | **新規**。per-segment 変調 `h*(1+scale[row])+shift[row]` を 1 パスで行う Triton カーネル（§3.5） |
| `tests/test_int8.py` | `TestTritonInt8GemmDequant`（INT8 の 92 ケース） |
| `tests/test_scale_shift.py` | scale_shift のビット一致（dtype 4 通り × セグメント数、非連続ビュー、カバレッジ、subnormal） |
| `patches/comfyui-minimax-modulation.patch` | ComfyUI 側の呼び出しフック（§4.3） |

---

## 3. 変更内容（完全版）

### 3.1 `comfy_kitchen/backends/triton/quantization.py`

**旧 `_int8_matmul_dequant_kernel` / `_int8_matmul_dequant_per_row_kernel` は削除**し、以下を置きます。`int8_linear()` は `triton_int8_gemm_dequant()` を呼ぶように変更します。

```python
_PY_HEADERS_FALLBACK = "/opt/py312-include"
_PY_HEADERS_CHECKED = False


def _ensure_python_headers() -> None:
    """Make Python.h reachable for Triton's runtime C compile (once per process).

    The check itself costs ~52 us (sysconfig + stat) and used to run on every
    launch, which is a measurable share of a thin-M GEMM.
    """
    global _PY_HEADERS_CHECKED
    if _PY_HEADERS_CHECKED:
        return
    import os
    import sys
    import sysconfig

    inc = sysconfig.get_paths().get("include", "")
    if not os.path.exists(os.path.join(inc, "Python.h")):
        # Only a same-major.minor include dir is safe here: Triton's runtime driver
        # compiles against version-specific PyObject internals, so borrowing another
        # version's headers can corrupt the interpreter.
        alt = os.path.join(_PY_HEADERS_FALLBACK, f"python{sys.version_info.major}.{sys.version_info.minor}")
        if os.path.exists(os.path.join(alt, "Python.h")):
            os.environ["CPATH"] = os.pathsep.join(
                p for p in (alt, os.environ.get("CPATH")) if p
            )
    _PY_HEADERS_CHECKED = True


@triton.jit
def _int8_gemm_dequant_kernel(
    a,
    b,
    xs,
    ws,
    bias,
    c,
    M,
    N,
    K,
    sam,
    sak,
    sbn,
    sbk,
    scm,
    scn,
    ws_n,
    HAS_BIAS: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    G: tl.constexpr,
    EVEN: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    """Fused INT8 GEMM + dequantization epilogue:
    C[m, n] = (sum_k A[m, k] * B[n, k]) * xs[m] * ws[n] + bias[n]
    """
    pid = tl.program_id(0)
    npm = tl.cdiv(M, BM)
    npn = tl.cdiv(N, BN)
    gid = pid // (G * npn)
    fm = gid * G
    gs = min(npm - fm, G)
    pm = fm + (pid % gs)
    pn = (pid % (G * npn)) // gs
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    # int64 offsets: M*N or M*K past 2^31 elements (CFG batches, long clips) would wrap in int32
    rm64 = rm.to(tl.int64)
    rn64 = rn.to(tl.int64)
    A = a + rm64[:, None] * sam + rk[None, :] * sak
    B = b + rn64[None, :] * sbn + rk[:, None] * sbk
    acc = tl.zeros((BM, BN), dtype=tl.int32)
    for k0 in range(0, K, BK):
        if EVEN:  # M, N, K are tile multiples: unmasked loads
            acc += tl.dot(tl.load(A), tl.load(B))
        elif EVEN_K:  # only K is a tile multiple: row/column masks only
            acc += tl.dot(
                tl.load(A, mask=rm[:, None] < M, other=0),
                tl.load(B, mask=rn[None, :] < N, other=0),
            )
        else:
            km = (k0 + rk) < K
            acc += tl.dot(
                tl.load(A, mask=(rm[:, None] < M) & km[None, :], other=0),
                tl.load(B, mask=(rn[None, :] < N) & km[:, None], other=0),
            )
        A += BK * sak
        B += BK * sbk
    out = acc.to(tl.float32) * tl.load(xs + rm, mask=rm < M, other=0.0)[:, None]
    out = out * tl.load(ws + rn * ws_n, mask=rn < N, other=0.0)[None, :]
    if HAS_BIAS:
        out += tl.load(bias + rn, mask=rn < N, other=0.0).to(tl.float32)[None, :]
    C = c + rm64[:, None] * scm + rn64[None, :] * scn
    tl.store(C, out.to(c.dtype.element_ty), mask=(rm[:, None] < M) & (rn[None, :] < N))


# While the int8 weight fits in L2 it is re-read from cache once per M-tile, so the
# tile shape decides throughput; past that point the kernel is bandwidth-bound on B
# and splitting M only adds masked tail work. Measured on GB10: 4096x4096 (=16 MB)
# gains up to 2x from thin tiles, 5120x5120 (=26 MB) and 10240x4096 (=41 MB) stay
# flat within 4%.
_INT8_L2_WEIGHT_ELEMS = 20 << 20
_INT8_GB10_DEFAULT_TILE = (128, 128, 64, 8, 8, 4)
# 64 KB shared-memory devices (sm_75, RDNA1-4): 4 stages x (128*64 + 64*128) bytes
# is exactly 64 KB and fails to compile. Halving BK keeps the CTA shape at 32 KB.
_INT8_PORTABLE_TILE = (128, 128, 32, 8, 8, 4)
_DEVICE_CAPABILITY: dict[int, tuple[int, int]] = {}


def _device_capability(device_index: int = 0) -> tuple[int, int]:
    """Cached (major, minor) for the device, or (-1, -1) when it cannot be queried."""
    cap = _DEVICE_CAPABILITY.get(device_index)
    if cap is None:
        try:
            props = torch.cuda.get_device_properties(device_index)
            cap = (int(props.major), int(props.minor))
        except Exception:  # ROCm/XPU/CPU: no CUDA properties
            cap = (-1, -1)
        _DEVICE_CAPABILITY[device_index] = cap
    return cap


def _int8_tile_config(
    m: int, n: int, k: int, capability: tuple[int, int] | None = None
) -> tuple[int, int, int, int, int, int]:
    """Pick (BM, BN, BK, G, num_warps, num_stages) for this GEMM shape.

    GB10 (sm_121) has 48 SMs. The 128x128 tile needs M to be a multiple of 128;
    otherwise the last M-tile is partly empty and the CTA count stops matching the
    SM count (M=160 -> 2x32 = 64 CTAs = 1.33 waves, measured 1.6x slower than a
    64-row tile). Tile choice never changes the result: the int32 accumulation is
    exact and the dequant epilogue runs per output element, so every band below is
    bit-identical to the 128x128 kernel.

    Band boundaries are min-of-N measurements on GB10 (n=k=4096; the weight fits in
    L2 there, which is the regime where the tile matters at all). Other devices keep
    the generic tile: this Triton backend also serves ROCm/XPU, where the GB10 bands
    are unmeasured.
    """
    if capability is None:
        capability = _device_capability()
    if capability != (12, 1):
        return _INT8_PORTABLE_TILE
    if m >= 288 or n * k > _INT8_L2_WEIGHT_ELEMS:
        return _INT8_GB10_DEFAULT_TILE  # the shape-tuned default
    if m >= 208:
        return 128, 64, 64, 8, 8, 4   # 208-287: beats 128x128 by ~1.15x
    if m >= 128:
        return 64, 128, 64, 8, 4, 4   # 128-207: 1.6x
    return 64, 64, 64, 8, 4, 4        # <128: 1.6-1.7x


def triton_int8_gemm_dequant(
    x8: torch.Tensor,
    w8: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    bias: torch.Tensor | None,
    out: torch.Tensor,
) -> bool:
    """Launch fused INT8 GEMM + dequantization epilogue in Triton.

    Applies the same float operations in the same order as CUTLASS
    ``cutlass_int8_dequant``: ``(int32 -> fp32 * x_scale * w_scale + bias) -> out.dtype``.
    """
    _ensure_python_headers()
    M, K = x8.shape
    N = w8.shape[0]
    if M == 0 or N == 0:
        return True
    if K == 0:
        if bias is not None and bias.numel() > 0:
            out.copy_(bias.reshape(1, N).expand(M, N))
        else:
            out.zero_()
        return True
    xs = x_scale.reshape(-1).contiguous()
    ws = w_scale.reshape(-1).contiguous()
    ws_n = 0 if ws.numel() == 1 else 1
    # The kernel reads these as flat, stride-1 vectors: materialise them rather than
    # trusting a direct caller (expand()/slices produce numel == n with stride 0).
    if bias is not None and bias.numel() > 0:
        has_bias = True
        bias_arg = bias.reshape(-1).contiguous()
    else:
        has_bias = False
        bias_arg = out
    BM, BN, BK, G, num_warps, num_stages = _int8_tile_config(M, N, K)
    # Unmasked K loop when every dim (or K) is a tile multiple.
    even_k = K % BK == 0
    even = even_k and M % BM == 0 and N % BN == 0
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
    _int8_gemm_dequant_kernel[grid](
        x8,
        w8,
        xs,
        ws,
        bias_arg,
        out,
        M,
        N,
        K,
        x8.stride(0),
        x8.stride(1),
        w8.stride(0),
        w8.stride(1),
        out.stride(0),
        out.stride(1),
        ws_n,
        HAS_BIAS=has_bias,
        BM=BM,
        BN=BN,
        BK=BK,
        G=G,
        EVEN=even,
        EVEN_K=even_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return True
```

**残差融合（`residual` あり）の追加分** — カーネル引数に `resid`, `rs`, `rs_n`, `HAS_RESIDUAL` を足し、epilogue を以下に置き換えます。

```python
    out_ty = c.dtype.element_ty
    if HAS_RESIDUAL:
        # Add2 rounds to the output dtype even when bias is absent (CUTLASS always
        # runs the Bias visitor, with a zero default), then Mul3 + Add4 run in fp32.
        out = out.to(out_ty).to(tl.float32)
        out = out * tl.load(rs + rn * rs_n, mask=rn < N, other=0.0).to(tl.float32)[None, :]
        out = out + tl.load(
            resid + rm64[:, None] * scm + rn64[None, :] * scn,
            mask=(rm[:, None] < M) & (rn[None, :] < N),
            other=0.0,
        ).to(tl.float32)
    tl.store(C, out.to(out_ty), mask=(rm[:, None] < M) & (rn[None, :] < N))
```

ランチャー側は `triton_int8_gemm_dequant(..., residual=..., residual_scale=...)` を受け、`residual is None` のときはダミーポインタ（`out`）を渡して `HAS_RESIDUAL=False` にします。ディスパッチ側は `_cutlass_int8_dequant(..., residual=, residual_scale=)` に統合し、Triton が使えないときだけ `_C.cutlass_int8_dequant_residual` に落とします。

> **注意**: ランチャーの `BM=BN=BK=G=num_warps=num_stages` は変数で渡すこと。定数のままだと
> グリッドだけがタイル選択に従い、**結果が壊れます**（§5 の bit-identical 検査で検出できます）。

### 3.2 `comfy_kitchen/backends/cuda/__init__.py`

**(a) 定数とモード読取**（`_CONVROT_FUSED_MAX_K = 16384` の直後）

```python
_INT8_MODE_FILE = os.environ.get("COMFY_KITCHEN_INT8_MODE_FILE", "/etc/comfy-kitchen-int8.mode")
# Below this M the CUTLASS kernel wins (measured on GB10: 14.8-24.5 us for M<=16
# vs 23-28 us for Triton); at M>=24 Triton is ahead and at M=64 it is 4.5x ahead
# (25.7 us vs 116.8 us), so the gate used to send the worst part of the range to
# the slower kernel.
# 素の comfy-kitchen との実測クロスオーバー（int8_linear, 4096x4096）:
#   M=24-32 0.78-0.82x / 40-48 1.06x / 56 1.59x / 64 2.17x -> 40 が下限
_TRITON_INT8_MIN_M = 40
_triton_int8_broken = False


def _read_int8_mode() -> str:
    env_mode = os.environ.get("COMFY_KITCHEN_INT8_MODE", "").strip().lower()
    if env_mode:
        return env_mode
    try:
        with open(_INT8_MODE_FILE, encoding="utf-8", errors="replace") as f:
            return f.read().strip().lower()
    except (OSError, ValueError):
        # A missing file is the normal case; an unreadable or undecodable one must
        # not break `import comfy_kitchen`.
        return ""


_INT8_MODE = _read_int8_mode()

_DISABLE_CUTLASS_INT8 = (
    os.environ.get("COMFY_KITCHEN_DISABLE_CUTLASS", "0") == "1"
    or _INT8_MODE == "cublas"
)


def _should_use_triton_int8_dequant(device_index: int) -> bool:
    mode = os.environ.get("COMFY_KITCHEN_INT8_MODE", "").strip().lower() or _INT8_MODE
    if mode in ("off", "cutlass", "cublas", "0", "false"):
        return False
    if mode == "triton":
        return True
    # On NVIDIA GB10 (sm_121), the SM80-targeted CUTLASS INT8 kernel reaches
    # only ~37% of peak TOPS on large-M shapes, whereas the tuned Triton
    # fused-dequant INT8 GEMM reaches ~88% of peak while staying bit-identical.
    return _cuda_device_capability(device_index) == (12, 1)
```

**(b) ディスパッチ用ラッパー**

```python
def _cutlass_int8_dequant(
    x: torch.Tensor,
    w: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    bias: torch.Tensor,
    out: torch.Tensor,
    output_dtype_code: int,
    stream_ptr: int,
) -> bool:
    global _triton_int8_broken
    m, k = x.shape
    n = w.shape[0]
    # The Triton kernel launches on PyTorch's current device and stream, not on
    # (x.device, stream_ptr). Both checks keep a mismatched caller on CUTLASS
    # instead of launching against the wrong GPU or racing the producer. They sit
    # at the end of the gate so the short-circuit keeps the cheap path cheap.
    if (
        not _triton_int8_broken
        and m >= _TRITON_INT8_MIN_M
        and k % 16 == 0
        and k < 131072  # int32 accumulator: 127*127*131072 still fits; satfinite vs wrap diverges past it
        and w.dim() == 2
        and w.shape[1] == k
        and x.dtype == torch.int8
        and w.dtype == torch.int8
        and x.is_contiguous()
        and w.is_contiguous()
        and out.is_contiguous()
        and out.shape == (m, n)
        and out.dtype in (torch.bfloat16, torch.float16, torch.float32)
        and x_scale.dtype == torch.float32
        and x_scale.numel() == m
        and x_scale.is_contiguous()
        and w_scale.dtype == torch.float32
        and w_scale.numel() in (1, n)
        and w_scale.is_contiguous()
        and (
            bias.numel() == 0
            or (bias.numel() == n and bias.dtype == out.dtype and bias.is_contiguous())
        )
        and x.get_device() == torch.cuda.current_device()
        and stream_ptr == torch.cuda.current_stream().cuda_stream
        and _should_use_triton_int8_dequant(x.get_device())
    ):
        try:
            from comfy_kitchen.backends.triton.quantization import (
                triton_int8_gemm_dequant,
            )

            return triton_int8_gemm_dequant(x, w, x_scale, w_scale, bias, out)
        except Exception as e:
            _triton_int8_broken = True
            print(
                f"[comfy_kitchen] Triton INT8 GEMM failed ({type(e).__name__}: {e}); "
                f"falling back to CUTLASS for remainder of session",
                file=sys.stderr,
            )

    ws_cutlass = (
        w_scale.reshape(-1).contiguous()
        if w_scale.numel() == n
        else w_scale.reshape(-1)[:1].expand(n).contiguous()
    )
    return _C.cutlass_int8_dequant(
        _wrap_for_dlpack(x),
        _wrap_for_dlpack(w),
        _wrap_for_dlpack(x_scale),
        _wrap_for_dlpack(ws_cutlass),
        _wrap_for_dlpack(bias),
        _wrap_for_dlpack(out),
        output_dtype_code,
        stream_ptr,
    )
```

**(c) 呼び出し置換（3 箇所）** — 残差なしパスの `_C.cutlass_int8_dequant(...)` を `_cutlass_int8_dequant(...)` に置き換えます。

- `_int4_linear_via_int8_values()` 内
- `int8_linear()` 内（`if not used_cutlass:` のブロック）
- `w4a8_int8_linear()` 内

これで **int8 / int4(w4a8) / w4a8 の全経路**が同じカーネルを使います。

### 3.3 `comfy_kitchen/__init__.py`

`int8_linear()` の `if out_dtype is None: out_dtype = torch.bfloat16` の直後：

```python
    if out_dtype is None:
        out_dtype = torch.bfloat16
    # The custom op has no autograd formula and its meta function ignores residual
    # broadcasting, so only the plain inference call takes the fast path under
    # torch.compile; anything else keeps tracing the Python implementation.
    if torch.compiler.is_compiling() and residual is None and not any(
        getattr(t, "requires_grad", False)
        for t in (x, weight, bias, input_act_weight, residual_scale)
    ):
        if not isinstance(weight_scale, torch.Tensor):
            weight_scale = torch.tensor(weight_scale, device=x.device, dtype=torch.float32)
        return torch.ops.comfy_kitchen.int8_linear(
            x,
            weight,
            weight_scale,
            bias,
            DTYPE_TO_CODE[out_dtype],
            convrot,
            convrot_groupsize,
            input_act,
            input_act_weight,
            input_act_eps,
            residual,
            residual_scale,
        )
```

### 3.4 `tests/test_int8.py`

末尾に `TestTritonInt8GemmDequant` を追加（7 メソッド）。合格条件は以下。

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
| `test_edge_shapes_match_exact_reference` | N=1/7/33（不揃え）と K<BK。CUTLASS ではなく厳密参照と比較（§10.4） |
| `test_non_contiguous_scales_never_reach_triton` | `expand(n)` スケールは CUTLASS に落ち、連続版と同値 |
| `test_two_dim_weight_scale_falls_back` | `(1,1)` の weight_scale でクラッシュしない |
| `test_residual_epilogue_matches_cutlass` | 残差融合 epilogue が CUTLASS と `torch.equal`（4形状×dtype3×bias2） |
| `test_residual_routes_match_end_to_end` | `ck.int8_linear(residual=...)` の cutlass/triton 経路一致（スパイで Triton 実走を確認） |
| `test_residual_negative_zero_matches_cutlass` | bias 無しの `-0.0` 符号（CUTLASS は `+0.0` にする） |
| `test_cpu_residual_scale_does_not_poison_triton` | CPU スケールでゲートが落ち、`_triton_int8_broken` が立たない |

> 実装はリポジトリの `tests/test_int8.py` をそのまま参照（この文書の付録ではなくコードを正とする）。

---

### 3.5 `comfy_kitchen/tensor/scale_shift.py`（新規）

DiT の pre-norm ブロックが量子化線形の直前に行う per-segment 変調

```python
for a, b, row in segments:
    h[a:b].mul_(1.0 + scale[row].to(h.dtype)).add_(shift[row].to(h.dtype))
```

を 1 パスで実行する。torch は bf16 行列を 2 回 read-modify-write するため GB10 で
115〜150 GB/s しか出ない（in-place は write-allocate で帯域が半減する）。カーネルは
out-of-place で 222 GB/s、実測 **2〜3 倍**。

ビット一致のために落とせない点が 3 つある。

1. **Triton は `x*s + b` を FMA に融合し、`mul_()` が行う中間 bf16 丸めを消す。**
   整数演算で RNE を強制する（bf16: `(bits + 0x7FFF + ((bits>>16)&1)) & -65536`）。
   fp16 は指数部が 5 bit で subnormal 域の丸めが変わるため `cvt.rn.f16.f32` を使う。
2. **変調テンソルは先に活性の dtype へ丸める**（`_mod_row` の `.to(h.dtype)`）。
   H3 の curve チェックポイントは adaLN を fp32 で持つので、これを飛ばすと 38% の要素が
   ずれる（bf16 変調では再現しない）。
3. **`scale`/`shift` は非連続ビューで来る。** ComfyUI は `[R, n*cols]` から
   `[R, cols]` を切り出すため行ストライドが `n*cols`。`row*cols` で索引すると
   mod 行 0 以外が全部ずれる（実測: 15787 行中 802 行が不一致）。呼び出し前に
   `.contiguous()` する。活性 `h` 側が非連続なら例外（誤ったオフセットで読み書きする）。

```python
from comfy_kitchen.tensor.scale_shift import scale_shift, row_mod_from_segments
row_mod = row_mod_from_segments(h.shape[0], segments, h.device)
out = scale_shift(h, scale, shift, row_mod)   # 新しいテンソルを返す（h は不変）
```

セグメントは `range(rows)` を隙間なく覆うこと（欠落は例外）。per-token の
mod 行テンソルにも対応する。

### 3.6 ComfyUI 側の呼び出し（`comfy/ldm/minimax/model.py`）

ComfyUI は上流クローンなので、フックは `patches/comfyui-minimax-modulation.patch`
として本リポジトリに同梱している（適用は §4.3）。`_mod_scale_shift()` が
comfy-kitchen のカーネルを使い、前提が崩れたときは stderr に 1 回だけ理由を出して
torch 実装（`_mod_scale_shift_ref`）に戻る。理由を出さないフォールバックは使わない
（以前、カーネル側の不具合を隠した）。

---

## 4. ビルドとインストール

### 4.1 サブモジュール

```bash
cd ~/comfy-kitchen
git submodule update --init --depth 1
```

空のままビルドすると 2 つの事故が起きます。

1. `flash_decode.cu` が `flash_attn` ヘッダを見つけられずビルドが途中で失敗する
2. CMake が `COMFY_HAVE_CUTLASS` を無効化し、`cutlass_gemm_int8.cu` 等が **4KB のダミースタブ**としてコンパイルされる → ビルドは成功するが遅い状態になる

初期化前に一度ビルドしていた場合は `rm -rf build` で CMake キャッシュを捨ててください。

### 4.2 ビルド（editable）

**仮想環境は `source .venv/bin/activate` を前提にする。** ノードごとにパスが違うので、
`/home/chai/ComfyUI/.venv` のような絶対パスを手順に埋めない（他ノードで必ず失敗する）。

前提: その venv に **torch（2.15+cu132）と triton（3.9）が入っていること**。
`--no-build-isolation` はビルドに既存の torch を使うため、空の venv では失敗する。
ComfyUI が動いているノードなら、その ComfyUI の venv を使うのが最短。

```bash
cd ~/comfy-kitchen                 # ノードごとのリポジトリ位置

# リポジトリ直下に .venv がある場合
source .venv/bin/activate
# 無い場合は、そのノードの ComfyUI の venv を使う（torch/triton が揃っている）
#   source ~/ComfyUI/.venv/bin/activate

which python                      # venv を指していることを確認
uv pip install nanobind packaging wheel cmake ninja   # 不足時のみ
COMFY_CUDA_ARCHS=121 uv pip install --no-build-isolation -e .
```

まっさらな venv から始める場合だけ、先に依存を入れる（数 GB・数分）。

```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -U uv
uv pip install torch triton        # ComfyUI と同じバージョンに合わせる
```

- `COMFY_CUDA_ARCHS=121` で旧世代 arch を省略（20 コアで約 2.5 分）
- `-e`（editable）なので `_C.abi3.so` はリポジトリ内に置かれ、以後 **Python の修正は再ビルド不要**
- 確認: `ls -l comfy_kitchen/backends/cuda/_C.abi3.so` が 12MB 級であること（4KB なら §4.1）
- `uv pip install` は **activate 中の venv に入る**（`VIRTUAL_ENV` を見る）。空のまま実行すると
  別インタプリタに入って「import できない」になるので、`which python` で `.venv` を指しているか確認する
- この venv に `pip` モジュールが無い構成でも `uv pip` は動く。`python -m pip` は使わない

#### ComfyUI から使う場合

editable は **venv ごと**に入る。検証だけなら上の `.venv` で足りるが、ComfyUI に読ませるなら
そのノードの ComfyUI の venv を activate して同じコマンドを流す（`--python` を足すより事故が少ない）。

```bash
source ~/ComfyUI/.venv/bin/activate      # ノードごとのパス
cd ~/comfy-kitchen
COMFY_CUDA_ARCHS=121 uv pip install --no-build-isolation -e .
python -c "import comfy_kitchen, pathlib; print(pathlib.Path(comfy_kitchen.__file__).parent)"
# → <repo>/comfy_kitchen なら editable が効いている
```

再ビルド不要なのは Python 側だけ。**C++/CUDA（`_C.abi3.so`）を触ったら両方の venv で再ビルド**する。

### 4.3 ComfyUI 側フックの適用

ComfyUI は上流（`Comfy-Org/ComfyUI`）クローンなので push 先が無い。フックは本リポジトリの
`patches/comfyui-minimax-modulation.patch` を適用して使う。

```bash
git -C ~/ComfyUI apply ~/comfy-kitchen/patches/comfyui-minimax-modulation.patch

# 取り消し（上流に戻す）
git -C ~/ComfyUI apply -R ~/comfy-kitchen/patches/comfyui-minimax-modulation.patch

# 上流を pull した後は再適用（競合したら model.py の _mod_scale_shift だけ手で直す）
```

反映には **ComfyUI の再起動**が必要（Python の読み直しのみ）。

### 4.4 インストール直後の動作確認

```bash
python -c "
import torch
from comfy_kitchen.backends import cuda as cb
import comfy_kitchen.tensor.scale_shift as ss
print('cap', cb._cuda_device_capability(0), 'triton-int8', cb._should_use_triton_int8_dequant(0))
h = torch.randn(1024, 1024, dtype=torch.bfloat16, device='cuda')
s = torch.randn(2, 1024, dtype=torch.float32, device='cuda')
b = torch.randn(2, 1024, dtype=torch.float32, device='cuda')
segs = [(0, 512, 0), (512, 1024, 1)]
out = ss.scale_shift(h, s, b, ss.row_mod_from_segments(1024, segs, h.device))
ref = h.clone()
for a, z, r in segs:
    ref[a:z].mul_(1.0 + s[r].to(h.dtype)).add_(b[r].to(h.dtype))
print('scale_shift bit-identical:', torch.equal(out, ref))
"

---

## 5. 検証（変更したら必ず全部）

### 5.1 ディスパッチとビット一致

```bash
python -c "
import torch, comfy_kitchen
from comfy_kitchen.backends import cuda as cb
from comfy_kitchen.backends.eager.quantization import DTYPE_TO_CODE
from comfy_kitchen.backends.triton.quantization import triton_int8_gemm_dequant

print('cap', cb._cuda_device_capability(0), 'dispatch', cb._should_use_triton_int8_dequant(0))
assert cb._should_use_triton_int8_dequant(0) is True

for m in (24, 64, 96, 160, 192, 256, 512):
    n, k = 256, 256
    x = torch.randint(-127, 128, (m, k), dtype=torch.int8, device='cuda')
    w = torch.randint(-127, 128, (n, k), dtype=torch.int8, device='cuda')
    xs = torch.rand(m, 1, dtype=torch.float32, device='cuda') * 0.02
    ws = torch.rand(n, dtype=torch.float32, device='cuda') * 0.02
    bias = cb._empty_cuda_tensor(x.device, torch.bfloat16)
    a = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')
    b = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')
    assert triton_int8_gemm_dequant(x, w, xs, ws, bias, a)
    assert cb._C.cutlass_int8_dequant(cb._wrap_for_dlpack(x), cb._wrap_for_dlpack(w),
        cb._wrap_for_dlpack(xs), cb._wrap_for_dlpack(ws), cb._wrap_for_dlpack(bias),
        cb._wrap_for_dlpack(b), DTYPE_TO_CODE[torch.bfloat16],
        torch.cuda.current_stream().cuda_stream)
    assert torch.equal(a, b), f'M={m} mismatch'
    print('M', m, 'bit-identical OK')
"
```

### 5.2 タイル表

```bash
python -c "
from comfy_kitchen.backends.triton.quantization import _int8_tile_config as f
for m in (24, 64, 127, 128, 207, 208, 287, 288, 4096):
    print(m, f(m, 4096, 4096)[:2], '|', m, f(m, 5120, 5120)[:2])
"
```

期待値: `24..127 → (64,64)` / `128..207 → (64,128)` / `208..287 → (128,64)` / `288.. → (128,128)`、`5120×5120` は全て `(128,128)`。

### 5.3 単体テスト

```bash
uv pip install pytest        # 未導入なら
python -m pytest tests/test_int8.py tests/test_scale_shift.py -q
```

期待値: 全件パス（本リポジトリでは `290 passed` = 上流 + 追加 96 ケース）。
`test_scale_shift.py` は非連続ビュー・subnormal・カバレッジ欠落も落とすので、
カーネルを触ったら必ず両方流すこと。

### 5.4 実測（任意だが推奨）

**測定の作法**: 同一構成でも実行間で **10〜15% 揺れる**（クロック）。必ず min-of-3〜5、可能なら構成をラウンドロビンで回して比較する。10% 未満の差は有意とみなさない。

---

## 6. 上流追従

作業ブランチは `main`（§1）。上流の新リリースに載せ替えるときはこの順で。

```bash
cd /home/chai/comfy-kitchen
git fetch upstream
git rebase upstream/main                    # 上流のデフォルトブランチに合わせる
git submodule update --init --depth 1

source .venv/bin/activate                   # §4.2 の venv
COMFY_CUDA_ARCHS=121 uv pip install --no-build-isolation -e .
python -m pytest tests/test_int8.py tests/test_scale_shift.py -q
git push --force-with-lease origin main     # rebase したので force
```

- コンフリクトしやすいのは **(1) `_cutlass_int8_dequant()` の 3 呼び出し箇所**、**(2) triton 側 `int8_linear()` の末尾**、**(3) `__init__.py` の `int8_linear()` 冒頭**。
- 上流が同等機能（sm_121 判定や融合カーネル）を入れた場合は、**自分のゲートを捨てて上流に従う**。二重ディスパッチは事故のもと。
- 追従後は §5 を必ず再実行（特に 5.1 のビット一致。上流が epilogue の演算順序を変えたら一致しなくなる）。
- フォークを ComfyUI から使う場合、`pip install` の向き先が fork のパスであることを確認（§4.2 の確認コマンド）。
- 上流が `_TRITON_INT8_MIN_M` の帯域やタイル表に触れたら、§0 の比較表を取り直す
  （`~/archive/patches/bench_int8_baseline.py` を A / main の 2 状態で流すだけ）。
- ComfyUI 側は `git pull` でフックが外れるので、§4.3 のパッチを再適用する。

---

## 7. やってはいけない最適化（実測済みの結論）

| 案 | 判定 | 根拠 |
| :--- | :--- | :--- |
| tcgen05（第5世代TC）への書き直し | **やらない** | Triton 3.9 は sm_121 の INT8 を tcgen05 に lowering しない（TMA 記述子版でも `mma.sync.m16n8k32`）。cuBLAS の最速経路（FP8）が 178 TOPS ＝ 本カーネル 176.5 と同等で、上積みは最大2割・実質数% |
| FP8 へ切り替え | **やらない** | レガシー経路では INT8 と同レート（105.6 vs 105.3 TOPS）。ビット一致も失う |
| 活性量子化パスの融合 | **やらない** | 既に 226〜235 GB/s（ピーク 273 の 83〜86%）。融合すると A トラフィックが 2 倍になり逆効果 |
| 大形状向け自動チューナ | **やらない** | 形状ごとに最適が入れ替わり（2048×5120 は stages=3、512×4096 は stages=4）利得 ±5%。測定コストが上回る |
| `G`（L2 スウィズル）の削除 | やらない | 行優先順序にすると 176 → 105 TOPS（1.7 倍差）。このカーネルで最も効く部分 |

---

## 8. トラブルシューティング

| 症状 | 原因 / 対処 |
| :--- | :--- |
| `ModuleNotFoundError: comfy_kitchen` | editable の向き先（リポジトリ）が消えた。§4 を再実行 |
| ビルドは通るが遅い（CUTLASS と変わらない） | サブモジュール未初期化で `cutlass_gemm_int8.cu` がダミー化。`rm -rf build` → §4.1 から |
| `Kernel requires a runtime memory allocation, but no allocator was set` | TMA 記述子を使う実験時のみ。`triton.set_allocator(...)` を設定 |
| `stderr` に `Triton INT8 GEMM failed ... falling back to CUTLASS` | `_triton_int8_broken` が立ち、そのセッションは CUTLASS 固定。原因（多くは Python.h 不在や Triton のコンパイル失敗）を直してプロセス再起動 |
| `Python.h` が見つからない | `_ensure_python_headers()` のフォールバック（`/opt/py312-include`、コンテナ用）。ホスト venv では `sysconfig` の include が使われる |
| 数値が CUTLASS と一致しない | ランチャーの `BM/BN/BK/G/warps/stages` を定数のままにしている（§3.1）。ただし **N が16の倍数でない場合は上流 CUTLASS の方が壊れている**（§10.4）ので、厳密参照で判定する |
| `expand()` したスケールで値が壊れる / OOB | ゲートが `is_contiguous()` を要求する。呼び出し側で `.contiguous()` するか、CUTLASS に落とす |
| `torch.compile` で autograd エラー | `requires_grad` 付き入力は意図的に Python trace へ落とす（§10.1）。訓練で使うなら custom op に autograd 登録が必要 |

---

## 9. 実測リファレンス（dgxspark1, 2026-10-30）

環境: GB10 / 48 SM / 2190 MHz（上限 2418、実運用は 2200 cap）/ Triton 3.9.0 / torch 2.15.0+cu132。

> §9 は **GEMM カーネル単体**の数字（量子化を含まない）。ユーザーから見た
> エンドツーエンドの比較（`int8_linear` 全体、素の comfy-kitchen 基準）は §0。

### 9.1 実用天井

| 経路 | 2048×5120×5120 | 4096³ | 15787×21504×5376（H3 qkv） |
| :--- | ---: | ---: | ---: |
| 本カーネル INT8 | **176.5 TOPS** | 177.5 | **164.1** |
| CUTLASS INT8（素） | 137–181 | 137 | **72.1** |
| cuBLAS INT8 | 133 | 140 | — |
| cuBLAS FP8 | **178** | 162 | — |
| cuBLAS BF16 | 84 TFLOPS | 91 | — |

N が大きく重みが L2 を外れる形（H3 の qkv: 115MB）では素の CUTLASS が失速し、差が最も開く。

レガシー IMMA の理論天井 = 48 SM × 2.19 GHz × 1024 MAC/cycle ≈ **215 TOPS**（達成率 82%）。cuBLAS 最速が 178 である以上、ライブラリ置換の余地はない。

注意: これは **GEMM 単体**の比較。エンドツーエンド（量子化込み）の差は **N の広さ**で決まり、
N=4096 ではほぼ差が出ない（1.06x）が、N ≥ 8192 の射影層では 1.5〜2.4x 開く（§0.2）。

### 9.2 形状別（GEMM 単体、N=K=4096、本フォーク）

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

M=384 が 171 で M=512 が 155 なのは **ウェーブ量子化**（48 SM に対し CTA 数が 96 = 2 波ちょうど vs 128 = 2.67 波）。

### 9.3 レイヤー内訳（N=K=4096）

| M | 量子化 | GEMM | 合計 | GEMM 比率 |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 10.9 µs | 112.5 µs | 131.8 µs | 85% |
| 4096 | 212.6 µs | 783.4 µs | 1026 µs | 76% |
| 27280 | 1424 µs | 5334 µs | 6634 µs | 80% |

量子化は帯域天井（ピークの 83〜86%）に達しており、削る余地はない。

### 9.4 上流（CUTLASS 経路）との差分 — `int8_linear` 全体、量子化込み

`COMFY_KITCHEN_INT8_MODE=cutlass`（＝上流相当）との比較。量子化パスは両者共通なので、比率は GEMM 単体より圧縮されて見える。

| M (N=K=4096) | 本カーネル | 上流 CUTLASS | 倍率 | ビット一致 |
| ---: | ---: | ---: | ---: | :--- |
| 64 | 39.1 µs | 109.3 µs | 2.79x | ✓ |
| 160 | 54.7 µs | 137.0 µs | 2.50x | ✓ |
| 192 | 58.7 µs | 146.3 µs | 2.49x | ✓ |
| 256 | 80.6 µs | 150.0 µs | 1.86x | ✓ |
| 512 | 137.3 µs | 228.5 µs | 1.66x | ✓ |
| 4096 | 1013.9 µs | 1141.0 µs | 1.13x | ✓ |

### 9.5 残差融合（`residual` 付き `int8_linear`）

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

- 数値契約は `out = residual + rs[n] * round_out(acc * xs * ws + bias)`。**bias 加算で一旦出力 dtype に丸める**のが CUTLASS の `Add2 → Mul3 → Add4` ビジター列と同じ順序で、ここを省くと一致しない。
- epilogue が 2枚目の `[BM,BN]` タイルを読むためレジスタ圧が上がる。`m >= 288` の残差形は `BN=64`（`_INT8_GB10_RESIDUAL_TILE`）を使う（`128x128` 比で 8〜12% 速い）。
- M ≥ 1024 では CUTLASS 残差カーネルと同等。**小〜中 M で大きく勝ち、大 M で損はしない**のが位置づけ。


---

## 10. レビュー記録（2026-10-31）

Claude Code（`--effort high`、Read のみ）と Antigravity（Gemini 4 Argon High、sandbox）の2系統で事前レビューし、実コードで再現できた指摘だけ反映した。**両者の結論は割れた**（Claude: BLOCKING 0 / agy: BLOCKING 4）ため、以下は1件ずつ検証した結果。

### 10.1 反映した指摘

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
| テスト不足（`gid>=1`、EVEN、E2E、edge shape、スケール連続性） | 妥当 | 5テスト追加（下記 §3.4） |

### 10.2 反映しなかった指摘（理由つき）

| 指摘 | 判断 |
| :--- | :--- |
| `rk * sbk` など K 方向のストライド積が int32 で溢れ得る | 閾値が `sbk >= 33,554,432`（＝転置重みで N が3300万行）。現実の形状で到達せず、int64 化はレジスタ圧と速度に影響するため見送り。コメントで明示 |
| スウィズルの端数グループで M タイル順が巡回する | 全単射で正しさに影響なし（L2 局所性のみ）。`pm = fm + ((pid % (G*npn)) % gs)` は等価なので変更せず |
| `_DISABLE_CUTLASS_INT8` が import 時固定で、後からの env 変更が効かない | 上流からの既存設計。挙動を変えると事故になるため保留（§10.3 に記載） |
| 残差融合パス（`cutlass_int8_dequant_residual`）が Triton を使わない | **対応した**（§9.5）。epilogue に `residual + rs*round_out(...)` を実装し、CUTLASS とビット一致を24ケースで確認 |
| `w4a8_int8_linear` の `s_channel` 正規化・`_DISABLE_CUTLASS_INT8` 尊重 | 連続性はゲートで担保済み。それ以外は上流挙動を維持 |

### 10.3 既知の制限（意図的に変更しない）

- ~~残差融合パスは CUTLASS のまま~~ → **対応済み**（§9.5）。残差は Triton の epilogue で融合し、CUTLASS 残差カーネルとビット一致する。
- **非 GB10 では autotune が無い**。上流の autotuned カーネル2本は置換・削除した。ROCm 等で詰めるなら上流の `_int8_matmul_dequant_kernel` を復活させること。
- **`K==0` の挙動が CUTLASS と違う**（Triton は bias か 0 を書く、CUTLASS は何も書かない）。未初期化メモリを返さない分だけ安全側。
- **`COMFY_KITCHEN_INT8_MODE` は呼び出しごとに読むが、`_DISABLE_CUTLASS_INT8` は import 時固定**（上流由来の非対称）。

### 10.4 上流のバグ

**`N` が 16 の倍数でない場合、上流 CUTLASS カーネルは壊れた出力を返す。** 実測（M=24, K=256, bf16）:

| N | Triton の誤差（厳密参照比） | CUTLASS の誤差 |
| ---: | ---: | ---: |
| 1 | 0.000000 | 41.0 |
| 7 | 0.000000 | 5,275,648.5 |
| 33 | 0.000000 | **NaN** |
| 16 / 48 / 64 | 0.000000 | 0.000000 |

`torch._int_mm`（cuBLAS INT8）も `N % 8 != 0` を拒否するため、この領域は元々サポート外と見られる。Triton 側は厳密参照と一致するので、**ゲートは N で制限しない**（CUTLASS より正しい方を選ぶ）。テストは CUTLASS 比較ではなく厳密参照比較にした（`test_edge_shapes_match_exact_reference`）。

### 10.5 ラウンド2レビュー（Claude Opus 5.5, 2026-10-31）

残差融合の追加分に対して再レビュー。**epilogue の数式は CUTLASS と全項目一致**（Mul0/Mul1 の順序、Add2 の丸め、bias 無しでも丸める点、Mul3/Add4、rscale のブロードキャスト、fp32/fp16 の挙動、residual ロードの stride とマスク）と確認された上で、5件の指摘。

| 指摘 | 対応 |
| :--- | :--- |
| CPU（他デバイス）の `residual_scale` がゲートを通り、Triton 内で例外→`_triton_int8_broken` が立ち**セッション中ずっと遅い経路**になる | ゲートに `residual.device == out.device` と `residual_scale.device == out.device` を追加 |
| `test_residual_routes_match_end_to_end` が「両方 CUTLASS」でも通ってしまう（自明な合格） | スパイで `residual is not None` 付きの Triton 呼び出しを検証 |
| ランチャーが `False` を返すと CUTLASS 残差パスも飛ばして融合を失う（size-1 次元の stride 差で起こり得る） | `if triton_int8_gemm_dequant(...): return True` にして False なら CUTLASS に落とす |
| bias 無しのとき `-0.0` の符号が CUTLASS と違う（CUTLASS は Bias ビジターで `+0.0` を足す） | `HAS_RESIDUAL` かつ bias 無しのとき `out = out + 0.0` を追加 |
| `_aligned16(resid_2d)` と `residual_scale.is_contiguous()` は CUTLASS の制約なのに Triton 経路まで CUTLASS に落としている | アラインメント検査を CUTLASS 分岐へ移動（非整列は `return False`）、rscale はランチャで `.contiguous()` |

テストは scalar `residual_scale` / scalar `weight_scale` / `-0.0` / CPU スケールの4ケースを追加（合計 **263 passed**）。

---

## 11. 謝辞 / Acknowledgements

本フォークの方向づけと運用の慣習は、次の公開ノートに負っています。

### kanthai / gb10-comfyui-tuning（MIT）

<https://github.com/kanthai/gb10-comfyui-tuning>

DGX Spark (GB10) 上で ComfyUI + comfy-kitchen を実測したノートとパッチ集。とくに参照した点:

| 参照した内容 | 本フォークでの対応 |
| :--- | :--- |
| `docs/02-int8-triton-gemm.md` — comfy-kitchen の CUTLASS 経路は sm_121 でピークの 37%、Triton の融合 dequant GEMM なら 88% | 本フォークの出発点。§0 の「N が広い射影層で 1.5〜2.4 倍」は同じ現象を shape 別に測り直したもの |
| `docs/03-torch-compile-int8.md` — `torch.compile` 下で INT8 が落ちる原因 | `int8_linear()` の `torch.compiler.is_compiling()` 早期リターン（§3.3） |
| モード切替の慣習 `COMFY_KITCHEN_INT8_MODE` / `/etc/comfy-kitchen-int8.mode`（`off` で素に戻す） | `_read_int8_mode()` とファイル方式（§3.2） |
| Triton の実行時コンパイルに `Python.h` が要る件 | `_ensure_python_headers()` とそのメモ化 |
| A/B・同一シード・ビット一致で確認する計測姿勢 | §5 の検証手順と `bench/` のスクリプト群 |

### 本フォーク側で独自に実装・検証した部分

上記リポジトリのコードは取り込んでいません（向こうは `.pth` で読み込むモンキーパッチ、
こちらは comfy-kitchen 本体へのフォーク）。以下は本フォークで書いたもの:

- タイル表（M 帯ごとに 64/128 行を選ぶ）と、GB10 以外のデバイス向けフォールバックタイル
- residual 融合 epilogue（CUTLASS の型順序に合わせ、`torch.equal` でビット一致を検証）
- `comfy_kitchen/tensor/scale_shift.py`（変調の融合。FMA 融合による中間丸め消失、
  fp32 adaLN の事前丸め、非連続ビューの 3 点は実測で特定）
- 形状別の実測（§0）と回帰テスト（`tests/test_int8.py`, `tests/test_scale_shift.py`）

本フォークの誤りは本フォーク側の責任です。原典のライセンスは上記リポジトリの `LICENSE`（MIT）を参照してください。
