# GB10 fork design rationale — what changed and why

Scope: `gb10-optimized` in `chai-md/comfy-kitchen` (the seven commits in `main..gb10-optimized`).
Procedures are in [OPERATIONS.en.md](OPERATIONS.en.md); numbers in [MEASUREMENTS.en.md](MEASUREMENTS.en.md).
Japanese: [DESIGN.md](DESIGN.md)

**The repository is the source of truth for code** — this document keeps the reasoning and the
constraints that cannot be dropped, not full listings.

---

## 1. What changed

| File | Change | Details |
| :--- | :--- | :--- |
| `comfy_kitchen/backends/triton/quantization.py` | Fused kernel, tile selection `_int8_tile_config()`, memoised `_ensure_python_headers()`, `int8_linear()` moved onto the new kernel | §2 |
| `comfy_kitchen/backends/cuda/__init__.py` | Mode reading (env / file), `_should_use_triton_int8_dequant()`, `_cutlass_int8_dequant()` wrapper, three call-site replacements, `_TRITON_INT8_MIN_M = 40` | §3 |
| `comfy_kitchen/__init__.py` | Early return under `torch.compiler.is_compiling()` in `int8_linear()` | §4 |
| `comfy_kitchen/tensor/scale_shift.py` | **New.** Single-pass per-segment modulation `h*(1+scale[row])+shift[row]` | §5 |
| `tests/test_int8.py` | `TestTritonInt8GemmDequant` (92 INT8 cases) | §6 |
| `tests/test_scale_shift.py` | scale_shift bit-identity (4 dtypes × segment counts, non-contiguous views, coverage, subnormals) | §6 |
| `patches/comfyui-minimax-modulation.patch` | ComfyUI-side call hook | §7 |

Python only. CUDA/C++ untouched (`_C.abi3.so` is rebuilt but has no diff).

---

## 2. Triton INT8 fused GEMM (`triton/quantization.py`)

The old `_int8_matmul_dequant_kernel` / `_int8_matmul_dequant_per_row_kernel` were deleted and
replaced with a single **fused GEMM + dequant epilogue**. `int8_linear()` now calls
`triton_int8_gemm_dequant()`.

Kernel essentials (implementation in the repository):

- `tl.dot` into an int32 accumulator; the epilogue computes
  `(int32 → fp32 * xs * ws + bias) → out.dtype` in the same operation order as CUTLASS's visitor
  chain, so it is bit-identical (`torch.equal`).
- Address arithmetic is **int64** (`M*N` / `M*K` past 2^31 wraps in int32 for CFG batches and long
  clips).
- `EVEN` / `EVEN_K` skip masking in two steps (when every dimension, or K, is a tile multiple).
- `G` (L2 swizzle) rotates M tiles within a group. Row-major order drops 176 → 105 TOPS (1.7x worse).
- `K == 0` writes bias or zero (CUTLASS writes nothing; ours is the safe side by not returning
  uninitialised memory).

### 2.1 Tile table (`_int8_tile_config`)

On GB10 (48 SM) a 128×128 tile requires M to be a multiple of 128; a ragged M makes the CTA count
stop matching the SM count and loses badly. The measured bands (min-of-N, N=K=4096, weight resident
in L2) select:

| M / condition | BM×BN, BK, G, warps, stages | Basis |
| :--- | :--- | :--- |
| M < 128 | 64×64, 64, 8, 4, 4 | 1.6–1.7x |
| 128–207 | 64×128, 64, 8, 4, 4 | 1.6x |
| 208–287 | 128×64, 64, 8, 8, 4 | 1.15x |
| M ≥ 288 or `n*k > 20<<20` | 128×128, 64, 8, 8, 4 | default (large shapes are bandwidth-bound, tiles stop mattering) |
| non-sm_121 | 128×128, **32**, 8, 8, 4 | 4 stages × 16 KB is exactly the 64 KB shared memory of sm_75/RDNA1-4, which fails to compile; halve BK |

- The L2 threshold is `_INT8_L2_WEIGHT_ELEMS = 20 << 20`. 4096×4096 (16 MB) gains up to 2x from thin
  tiles; 5120×5120 (26 MB) and 10240×4096 (41 MB) stay flat within ±4%.
- The default is `_INT8_GB10_DEFAULT_TILE = (128,128,64,8,8,4)`. At M=160 a 128×128 tile gives
  2×32 = 64 CTAs = **1.33 waves** and is 1.6x slower than a 64-row tile (that is why the bands are
  cut).
- **Tile choice never changes the result** (exact int32 accumulation, per-output-element epilogue).
  Bit-identity per band is pinned by tests.
- `_ensure_python_headers()` is memoised once per process (the old code paid a ~52 µs stat on every
  launch).

> **Trap**: the launcher's `BM/BN/BK/G/warps/stages` must be passed as variables. Constants make the
> grid follow the tile selection while the kernel does not, and **the result breaks** (caught by the
> bit-identity check in OPERATIONS §5.1).

### 2.2 Residual fusion (`residual` present)

The epilogue takes `resid` / `rs` / `HAS_RESIDUAL` and implements the numerical contract
`out = residual + rs[n] * round_out(acc * xs * ws + bias)`. **Rounding to the output dtype after the
bias add** matches CUTLASS's `Add2 → Mul3 → Add4` order; skipping it breaks equality (Add2 runs even
with no bias). The epilogue reads a second `[BM,BN]` tile, which raises register pressure, so the
residual shape at `m >= 288` uses `BN=64` (`_INT8_GB10_RESIDUAL_TILE`, 8–12% faster than 128×128).
With `residual is None` the launcher passes a dummy pointer (`out`) and `HAS_RESIDUAL=False`.

---

## 3. sm_121 dispatch (`backends/cuda/__init__.py`)

### 3.1 Gate

- `_TRITON_INT8_MIN_M = 40`. Measured crossover against stock comfy-kitchen (`int8_linear`,
  4096×4096): M=24-32 0.78-0.82x / 40-48 1.06x / 56 1.59x / 64 2.17x — **40 is the lowest width that
  never loses**. Kernel-only, M≤16 is CUTLASS 14.8–24.5 µs vs Triton 23–28 µs, and M=64 is Triton
  25.7 µs vs CUTLASS 116.8 µs (4.5x). The kernel crossover sits at 24, but the Triton launcher's
  fixed cost dominates at the bottom of the band, so the threshold is measured through the public
  entry point.
- Modes: `COMFY_KITCHEN_INT8_MODE` (`off` / `cutlass` / `cublas` / `triton`) and
  `/etc/comfy-kitchen-int8.mode`. Reading uses `errors="replace"` plus
  `except (OSError, ValueError)` so a bad file cannot break `import`.
- Runtime checks: `m >= 40`, `k % 16 == 0`, `k < 131072` (int32 saturation), 2-D/contiguous/dtype/
  shape, contiguity and element counts for `x_scale`/`w_scale`/`bias`,
  `x.get_device() == torch.cuda.current_device()`,
  `stream_ptr == torch.cuda.current_stream().cuda_stream`, and `_should_use_triton_int8_dequant()`.
  Triton launches on the *current* device and stream, so the last two checks prevent wrong-device
  launches and races.
- On an exception it sets `_triton_int8_broken`, prints the reason **once** to stderr, and stays on
  CUTLASS for the rest of the session (no silent fallback).

### 3.2 Call-site replacement

`_int4_linear_via_int8_values()`, `int8_linear()` and `w4a8_int8_linear()` are routed through
`_cutlass_int8_dequant(...)`, so int8, int4(w4a8) and w4a8 all use the same kernel. The CUTLASS path
materialises `w_scale` as `reshape(-1)[:1].expand(n).contiguous()` (avoids the `(1,1)` crash).

---

## 4. `torch.compile` path (`comfy_kitchen/__init__.py`)

At the head of `int8_linear()`, the custom-op fast path is taken only when
`torch.compiler.is_compiling()` and `residual is None` and no input has `requires_grad`. The custom
op has no autograd formula and its meta implementation ignores residual broadcasting; everything else
keeps tracing the Python implementation (training needs autograd registered separately).

---

## 5. Fused modulation `tensor/scale_shift.py` (new)

The per-segment modulation a DiT pre-norm block applies right before a quantised linear

```python
for a, b, row in segments:
    h[a:b].mul_(1.0 + scale[row].to(h.dtype)).add_(shift[row].to(h.dtype))
```

runs in one pass. Torch read-modify-writes the bf16 matrix twice, reaching only 115–150 GB/s on GB10
(in-place halves bandwidth through write-allocate). The kernel is out-of-place at 222 GB/s, measured
2–3x.

**Three things cannot be dropped if bit-identity is to hold.**

1. **Triton fuses `x*s + b` into an FMA, removing the intermediate bf16 rounding `mul_()` performs.**
   Force RNE with integer arithmetic (bf16: `(bits + 0x7FFF + ((bits>>16)&1)) & -65536`). fp16 has a
   5-bit exponent and different subnormal rounding, so use `cvt.rn.f16.f32`.
2. **Round the modulation tensors to the activation dtype first** (`.to(h.dtype)` in `_mod_row`). H3's
   curve checkpoints hold adaLN in fp32; skipping this **mismatches 38% of elements** (it does not
   reproduce with bf16 modulation).
3. **`scale`/`shift` arrive as non-contiguous views.** ComfyUI slices `[R, cols]` out of `[R, n*cols]`,
   so the row stride is `n*cols`. Indexing with `row*cols` misplaces everything except mod row 0
   (measured: 802 of the 15787 rows mismatch). Call `.contiguous()` first. A non-contiguous activation
   `h` raises (otherwise the kernel reads/writes at the wrong offset).

```python
from comfy_kitchen.tensor.scale_shift import scale_shift, row_mod_from_segments
row_mod = row_mod_from_segments(h.shape[0], segments, h.device)
out = scale_shift(h, scale, shift, row_mod)   # returns a new tensor (h is unchanged)
```

Segments must cover `range(rows)` with no gaps (a gap raises). Per-token mod-row tensors are also
supported.

---

## 6. Tests (regression safety net)

`TestTritonInt8GemmDequant` in `tests/test_int8.py` plus `tests/test_scale_shift.py`. Pass criteria:

| Test | Content |
| :--- | :--- |
| `test_triton_int8_gemm_dequant_matches_cutlass` | 3 shapes × dtype3 × bias2 × per-channel2, `torch.equal` against CUTLASS |
| `test_tile_config_selection` | band boundaries (24/127/128/207/208/256/288), L2 gate, non-GB10 default tile |
| `test_thin_m_tiles_match_cutlass` | 15 points M=24–288, bit-identity per band |
| `test_swizzle_and_even_paths_match_cutlass` | `gid>=1` group decomposition (M=1100/1664) and the full EVEN path (384/1024) |
| `test_cuda_int8_linear_routes_match` | through the CUDA backend (quantise + GEMM), cutlass and triton agree |
| `test_dispatch_gate_opens_below_128` | M=16/23 take CUTLASS, 24/64/127 take Triton (verified with spies) |
| `test_python_header_check_is_memoized` | the guard stats only once (CPATH restored via monkeypatch) |
| `test_sm121_dispatch_policy` | cap `(12,1)` decision and env modes |
| `test_cutlass_int8_dequant_fail_open_fallback` | an exception falls back to CUTLASS and sets `_triton_int8_broken` |
| `test_edge_shapes_match_exact_reference` | N=1/7/33 (unaligned) and K<BK, compared against an exact reference rather than CUTLASS (§9.4) |
| `test_non_contiguous_scales_never_reach_triton` | `expand(n)` scales fall back to CUTLASS and match the contiguous version |
| `test_two_dim_weight_scale_falls_back` | a `(1,1)` weight_scale does not crash |
| `test_residual_epilogue_matches_cutlass` | residual epilogue is `torch.equal` to CUTLASS (4 shapes × dtype3 × bias2) |
| `test_residual_routes_match_end_to_end` | `ck.int8_linear(residual=...)` cutlass/triton routes agree (spy confirms Triton actually ran) |
| `test_residual_negative_zero_matches_cutlass` | `-0.0` sign with no bias (CUTLASS produces `+0.0`) |
| `test_cpu_residual_scale_does_not_poison_triton` | a CPU scale drops the gate and does not set `_triton_int8_broken` |

The implementation is `tests/test_int8.py` in the repository (the code, not this appendix, is
authoritative).

---

## 7. ComfyUI-side hook (optional — **not applied by default**)

`patches/comfyui-minimax-modulation.patch` swaps ComfyUI's `_mod_scale_shift()` onto the comfy-kitchen
kernel. When its preconditions fail it prints the reason once to stderr and returns to the torch
implementation (`_mod_scale_shift_ref`) — no silent fallback, which once hid a kernel defect.

**Policy: as of 2026-11-11 it is not applied.** Three reasons.

- The gain is small: modulation 433 → 154 ms, i.e. 0.28 s per forward — about 2.2 s over 8 steps, 1–2%
  of a 150–250 s five-second clip. It is 0.33 s of the fork's 4.4 s (7.5%).
- The maintenance cost is high: every upstream H3 change (#17655 rewrote 287 lines of `model.py`) moves
  the insertion point and breaks the patch, in the file upstream changes most often.
- It belongs upstream: ComfyUI's `AGENTS.md` makes "use an existing optimised comfy-kitchen operation
  whenever one supports the math" a default requirement, so this fusion is a natural PR. Once merged the
  patch disappears.

`tensor/scale_shift.py` and its tests stay in the fork (material for that PR; harmless while unused).
To re-apply, see OPERATIONS §4.

---

## 8. Optimizations we rejected (measured conclusions)

| Idea | Measurement / basis | Verdict |
| :--- | :--- | :--- |
| Rewrite for tcgen05 (5th-gen tensor cores) | Triton 3.9 does not lower sm_121 INT8 to tcgen05 (even with TMA descriptors it emits `mma.sync.m16n8k32`). cuBLAS's fastest path (FP8) is 178 TOPS ≈ our 176.5, so headroom is at most ~20% and realistically a few percent | **No** |
| Switch to FP8 | On the legacy path it is the same rate as INT8 (105.6 vs 105.3 TOPS). Kernel-only it is 1.03–1.05x (`bench/bench_fp8_vs_int8.py`); weights are the same 1 byte/element, so no bandwidth gain, and e4m3's 3-bit mantissa loses bit-identity | **No** |
| Fuse the activation quantisation pass | Already at 226–235 GB/s (83–86% of the 273 peak); fusing doubles A traffic and makes it worse | **No** |
| Custom rotation kernel (256-group Hadamard) | 0.91x (cuBLAS already at the 228 GB/s bandwidth roof) | **No** |
| Autotuner for large shapes | The optimum flips per shape (2048×5120 wants stages=3, 512×4096 wants stages=4) for ±5%; the measurement cost outweighs it | **No** |
| Remove `G` (L2 swizzle) | Row-major order drops 176 → 105 TOPS (1.7x). It is the most load-bearing part of this kernel | No |
| `num_ctas=2` / `warp_specialize` / persistent / changing BK | −10 to −110% | No |
| Switch attention to SA3 | 1.02x, cos 0.982 (SA2 is 0.9992) | No |
| GPU buffer + bulk copy for VAE decode | 103 frames (768×1344): 23.39 → 23.52 s (no difference). The pageable-memory trap does reproduce (writing 1.28 GB of untouched memory 10.1 s → 0.03 s after `zero_()`), but this VAE's chunked copy never hits it | Unnecessary |

---

## 9. Review record (2026-10-31)

Pre-reviewed by two systems — Claude Code (`--effort high`, read-only) and Antigravity (Gemini 4
Argon High, sandbox) — and only findings reproduced in the real code were applied. **The two
disagreed** (Claude: BLOCKING 0 / agy: BLOCKING 4), so each item below was verified individually.

### 9.1 Findings applied

| Finding | Verification | Action |
| :--- | :--- | :--- |
| Triton launches on the current device and stream; a mismatch with `x.device`/`stream_ptr` means a wrong-device launch or a race | Valid | Added `x.get_device() == torch.cuda.current_device()` and `stream_ptr == torch.cuda.current_stream().cuda_stream` to the gate (mismatch → CUTLASS) |
| **Non-contiguous scales cause OOB reads**: `expand(n)` has `numel==n, stride==0`, so `reshape(-1)` is a no-op and reads past a one-element buffer | **Reproduced on the machine** (`stride=(0,)`, `is_contiguous()==False`) | Added `is_contiguous()` checks for `x_scale/w_scale/bias` and materialise `.contiguous()` in the launcher |
| A `(1,1)` `weight_scale` crashes the CUTLASS fallback in `expand` (surfaces at `m < 24`) | **Reproduced** (`RuntimeError: expand(...[1,1], size=[64])`) | Fixed to `w_scale.reshape(-1)[:1].expand(n)` (3 sites) |
| Forcing GB10 tiles on non-`sm_121` fails to compile on 64 KB shared-memory devices (sm_75/RDNA) | Valid (4 stages × 16 KB = 64 KB) | Assign `_INT8_PORTABLE_TILE = (128,128,32,8,8,4)` (32 KB) on non-GB10 |
| The `torch.compile` path breaks autograd and residual broadcasting | Valid (custom op has no autograd; the fake implementation ignores residual) | Restrict the fast path to `residual is None` with no `requires_grad`; otherwise keep tracing Python |
| Borrowing a **different version's** `Python.h` from `/opt/py312-include` | Valid (ABI risk) | Allow the same `major.minor` only (dropped the `python3.12` fallback) |
| Invalid bytes in the mode file raise `UnicodeDecodeError` and break the import | Valid | `errors="replace"` + `except (OSError, ValueError)` |
| int32 saturation can diverge at K ≥ 131072 | Valid (theoretical) | Added `k < 131072` to the gate |
| Insufficient tests (`gid>=1`, EVEN, E2E, edge shapes, scale contiguity) | Valid | 5 tests added (§6) |

### 9.2 Findings not applied (with reasons)

| Finding | Decision |
| :--- | :--- |
| K-direction stride products such as `rk * sbk` can overflow int32 | The threshold is `sbk >= 33,554,432` (transposed weights with N of 33 million rows). Unreachable for real shapes, and int64 costs registers and speed, so it was left out. Documented in a comment |
| Ragged swizzle groups rotate the M-tile order | A bijection, so correctness is unaffected (only L2 locality). `pm = fm + ((pid % (G*npn)) % gs)` is equivalent, so unchanged |
| `_DISABLE_CUTLASS_INT8` is fixed at import, so later env changes do nothing | Pre-existing upstream design; changing the behaviour risks accidents, so deferred (§10) |
| The residual path (`cutlass_int8_dequant_residual`) does not use Triton | **Addressed** (§2.2 / MEASUREMENTS §9): the epilogue implements `residual + rs*round_out(...)`, verified bit-identical to CUTLASS over 24 cases |
| `w4a8_int8_linear`'s `s_channel` normalisation and `_DISABLE_CUTLASS_INT8` respect | Contiguity is covered by the gate; everything else keeps upstream behaviour |

### 9.3 Second-round review (Claude Opus 5.5, 2026-10-31)

A re-review of the residual addition. **The epilogue formula matched CUTLASS on every item**
(Mul0/Mul1 order, Add2 rounding, rounding even without bias, Mul3/Add4, rscale broadcast, fp32/fp16
behaviour, residual load stride and mask), with five findings.

| Finding | Action |
| :--- | :--- |
| A CPU (other-device) `residual_scale` passes the gate, raises inside Triton, sets `_triton_int8_broken`, and leaves the session on the slow path | Added `residual.device == out.device` and `residual_scale.device == out.device` to the gate |
| `test_residual_routes_match_end_to_end` also passes when both routes are CUTLASS (a trivial pass) | Use a spy to assert a Triton call with `residual is not None` |
| If the launcher returns `False`, the CUTLASS residual path is skipped too and the fusion is lost (possible with size-1 dimension stride differences) | `if triton_int8_gemm_dequant(...): return True`, otherwise fall back to CUTLASS |
| Without bias, the `-0.0` sign differs from CUTLASS (which adds `+0.0` in the Bias visitor) | Add `out = out + 0.0` when `HAS_RESIDUAL` and there is no bias |
| `_aligned16(resid_2d)` and `residual_scale.is_contiguous()` are CUTLASS constraints, yet they sent the Triton path to CUTLASS too | Move the alignment check into the CUTLASS branch (unaligned → `return False`); the launcher `.contiguous()`s rscale |

Four test cases were added (scalar `residual_scale`, scalar `weight_scale`, `-0.0`, CPU scale),
reaching **263 passed** at the time (290 now).

### 9.4 Upstream bug (N not a multiple of 16)

**The upstream CUTLASS kernel returns corrupt output.** Measured (M=24, K=256, bf16):

| N | Triton error (vs exact reference) | CUTLASS error |
| ---: | ---: | ---: |
| 1 | 0.000000 | 41.0 |
| 7 | 0.000000 | 5,275,648.5 |
| 33 | 0.000000 | **NaN** |
| 16 / 48 / 64 | 0.000000 | 0.000000 |

`torch._int_mm` (cuBLAS INT8) also rejects `N % 8 != 0`, so this region looks unsupported from the
start. Triton matches the exact reference, so **the gate does not restrict N** (we pick the correct
side rather than CUTLASS). The test compares against an exact reference instead of CUTLASS
(`test_edge_shapes_match_exact_reference`).

---

## 10. Known limits (deliberately unchanged)

- Residual fusion is handled by the Triton epilogue (§2.2) and is bit-identical to the CUTLASS
  residual kernel.
- **No autotune on non-GB10.** The two upstream autotuned kernels were replaced/deleted. To tune for
  ROCm and friends, restore upstream's `_int8_matmul_dequant_kernel`.
- **`K==0` differs from CUTLASS** (Triton writes bias or zero, CUTLASS writes nothing). Ours is the
  safe side by not returning uninitialised memory.
- **`COMFY_KITCHEN_INT8_MODE` is read per call, while `_DISABLE_CUTLASS_INT8` is fixed at import**
  (an asymmetry inherited from upstream).

---

## 11. Acknowledgements

The direction and operational conventions of this fork owe a debt to the following public notes.

### kanthai / gb10-comfyui-tuning (MIT)

<https://github.com/kanthai/gb10-comfyui-tuning>

Notes and patches measuring ComfyUI + comfy-kitchen on DGX Spark (GB10). Specific points we used:

| What we referenced | How this fork responds |
| :--- | :--- |
| `docs/02-int8-triton-gemm.md` — comfy-kitchen's CUTLASS path reaches 37% of peak on sm_121, while a Triton fused dequant GEMM reaches 88% | The starting point of this fork. The "1.5–2.4x on wide-N projections" in MEASUREMENTS §4 re-measures the same phenomenon per shape |
| `docs/03-torch-compile-int8.md` — why INT8 fails under `torch.compile` | The `torch.compiler.is_compiling()` early return in `int8_linear()` (§4) |
| The mode-switch convention `COMFY_KITCHEN_INT8_MODE` / `/etc/comfy-kitchen-int8.mode` (`off` returns to stock) | `_read_int8_mode()` and the file mechanism (§3) |
| Triton's runtime compilation needing `Python.h` | `_ensure_python_headers()` and its memoisation (§2.1) |
| The measurement discipline: A/B, same seed, bit-identity | The verification steps in OPERATIONS §5 and the `bench/` scripts |

### Written and verified in this fork

No code from that project is included (it ships import-time monkey-patches loaded via `.pth`; this
fork modifies comfy-kitchen itself). The following were written here:

- The tile table (64/128 rows per M band) and the fallback tile for non-GB10 devices
- The residual fusion epilogue (matching CUTLASS's type order, verified bit-identical with
  `torch.equal`)
- `comfy_kitchen/tensor/scale_shift.py` (fused modulation; the three pitfalls — FMA-contracted
  intermediate rounding, fp32 adaLN pre-rounding, non-contiguous views — were identified by
  measurement)
- The per-shape measurements (MEASUREMENTS) and the regression tests (`tests/test_int8.py`,
  `tests/test_scale_shift.py`)

Errors in this fork are this fork's responsibility. For the original licence see that repository's
`LICENSE` (MIT).
