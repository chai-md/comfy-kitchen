# GB10 fork measurement reference

Environment: dgxspark1 / GB10 / 48 SM / 2190 MHz (hardware max 2418, run at the 2200 cap) /
Triton 3.9.0 / torch 2.15.0+cu132 (2026-10-30). Design in [DESIGN.en.md](DESIGN.en.md), procedures in
[OPERATIONS.en.md](OPERATIONS.en.md). Japanese: [MEASUREMENTS.md](MEASUREMENTS.md)

---

## 1. Measurement discipline

- Run-to-run spread is **10–15%** even for an identical configuration (clocks). Always take min-of-3–5
  and rotate configurations round-robin where possible.
- **Differences below 10% are not significant.**
- Compare two states — stock comfy-kitchen (`switch_kernel_state.sh A`) and this fork
  (`gb10-optimized`) — running the same script.
- Do not mix "GEMM alone" with "`int8_linear` end to end (including quantisation)". §8 is the latter;
  §6–§7 are the former.

---

## 2. H3 end to end (M=15,787)

One iteration, broken down, against stock comfy-kitchen (upstream `cdab478`)
(`bench/bench_h3_breakdown.py`).

| Stage | Stock | This fork | Reduction | Source of change |
| :--- | ---: | ---: | ---: | :--- |
| INT8 GEMM (4 linears × 50 layers) | 7.87 s | 3.82 s | **−4.05 s** | This fork's Triton path |
| Modulation `scale/shift` (×100) | 0.49 s | 0.16 s | **−0.33 s** | `tensor/scale_shift.py` |
| Rotation + quantisation (×200) | 0.32 s | 0.32 s | 0 | Unchanged (bandwidth roof) |
| attention (×50) | 2.69 s | 2.69 s | 0 | Outside the fork (launch flag) |
| **Total** | **11.4 s** | **7.0 s** | **−4.4 s (1.63x)** | |

The gain is in **INT8 GEMM and modulation**. The attention difference does not come from the fork; it
comes from ComfyUI's `--use-sage-attention` launch flag (SA2 53.9 ms/layer; with SDPA it is
96.6 ms/layer and this stage becomes 2.69 → 4.86 s).

---

## 3. Levers and when they apply

| # | Lever | Applies when | Size | Does not apply when |
| :--- | :--- | :--- | ---: | :--- |
| 1 | Route INT8 GEMM through Triton | **wide-N** projections (N ≳ 8192) | 1.5–2.4x | 1.05x at N ≤ 4096 |
| 2 | Fuse modulation (`scale_shift`) | the DiT has pre-norm modulation | 3.0x (that stage) | models without modulation |
| 3 | SageAttention for attention | ComfyUI launch flag (no code change) | 1.8x (that stage) | SA3 is 1.02x, unusable |

---

## 4. N dependence (M=15,787 fixed, K=5,376)

| N | Stock comfy-kitchen | This fork | Ratio |
| ---: | ---: | ---: | ---: |
| 4096 (K=4096) | 4.45 ms / 119 TOPS | 4.21 ms / 126 TOPS | 1.06x |
| 8192 | 22.24 ms / 62.5 TOPS | 9.38 ms / 148 TOPS | **2.37x** |
| 10752 | 24.44 ms / 74.7 TOPS | 11.88 ms / 154 TOPS | **2.06x** |
| 21504 (H3 qkv) | 50.25 ms / 72.6 TOPS | 22.50 ms / 162 TOPS | **2.23x** |
| 28672 (H3 fc1) | 69.06 ms / 70.5 TOPS | 29.21 ms / 167 TOPS | **2.36x** |

Stock CUTLASS falls from 119 to 62–75 TOPS as N widens, while the Triton side rises from 126 to
148–167. Projections land at N = hidden / 3·hidden / 2·ffn, i.e. inside this band.

---

## 5. M dependence (N=K=4,096, weight 16 MB = L2-resident)

### 5.1 Against stock

| M | Stock | This fork | Ratio |
| ---: | ---: | ---: | ---: |
| 24 | 32.1 µs | 33.1 | 0.97x (both CUTLASS) |
| 40 | 43.0 | 40.7 | 1.06x |
| 56 | 64.3 | 40.8 | **1.58x** |
| 64 | 88.3 | 40.4 | **2.18x** |
| 128 | 91.7 | 54.3 | **1.69x** |
| 512 | 230.3 | 141.5 | **1.63x** |
| 1024 | 289.7 | 304.0 | 0.95x (noise) |
| 4096 | 1082.8 | 1079.6 | 1.00x |

M<40 goes to the same CUTLASS path as stock (Triton's launcher fixed cost dominates and it loses
0.78–0.82x at M=24–32). The threshold is `_TRITON_INT8_MIN_M = 40`.

### 5.2 Absolute performance of this fork (by tile)

| M | TOPS | Tile |
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

M=384 reaching 171 while M=512 reaches 155 is **wave quantisation** (against 48 SMs, 96 CTAs is
exactly 2 waves vs 128 CTAs = 2.67 waves).

---

## 6. Practical roof (GEMM alone, no quantisation)

| Path | 2048×5120×5120 | 4096³ | 15787×21504×5376 (H3 qkv) |
| :--- | ---: | ---: | ---: |
| This kernel INT8 | **176.5 TOPS** | 177.5 | **164.1** |
| CUTLASS INT8 (stock) | 137–181 | 137 | **72.1** |
| cuBLAS INT8 | 133 | 140 | — |
| cuBLAS FP8 | **178** | 162 | — |
| cuBLAS BF16 | 84 TFLOPS | 91 | — |

When N is wide and the weight falls out of L2 (H3's qkv: 115 MB), stock CUTLASS stalls and the gap is
widest.

The legacy IMMA theoretical roof is 48 SM × 2.19 GHz × 1024 MAC/cycle ≈ **215 TOPS** (82% achieved).
With cuBLAS's best at 178, there is no room left for a library swap.

Note: this is a **GEMM-alone** comparison. The end-to-end (including quantisation) difference is set
by **N width**: at N=4096 there is essentially no difference (1.06x), while N ≥ 8192 projections open
1.5–2.4x (§4).

---

## 7. Layer breakdown (N=K=4096)

| M | Quantisation | GEMM | Total | GEMM share |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 10.9 µs | 112.5 µs | 131.8 µs | 85% |
| 4096 | 212.6 µs | 783.4 µs | 1026 µs | 76% |
| 27280 | 1424 µs | 5334 µs | 6634 µs | 80% |

Quantisation is already at the bandwidth roof (83–86% of peak); there is nothing left to cut.

---

## 8. Difference against upstream (CUTLASS path) — `int8_linear` end to end, quantisation included

Compared with `COMFY_KITCHEN_INT8_MODE=cutlass` (upstream-equivalent). The quantisation pass is
common to both, so the ratios look compressed relative to GEMM alone.

| M (N=K=4096) | This kernel | Upstream CUTLASS | Ratio | Bit-identical |
| ---: | ---: | ---: | ---: | :--- |
| 64 | 39.1 µs | 109.3 µs | 2.79x | ✓ |
| 160 | 54.7 µs | 137.0 µs | 2.50x | ✓ |
| 192 | 58.7 µs | 146.3 µs | 2.49x | ✓ |
| 256 | 80.6 µs | 150.0 µs | 1.86x | ✓ |
| 512 | 137.3 µs | 228.5 µs | 1.66x | ✓ |
| 4096 | 1013.9 µs | 1141.0 µs | 1.13x | ✓ |

---

## 9. Residual fusion (`int8_linear` with `residual`)

Against CUTLASS's residual-fusion kernel (`cutlass_int8_dequant_residual`), quantisation included.

| M (N=K=4096) | Triton | CUTLASS | Ratio |
| ---: | ---: | ---: | ---: |
| 64 | 43.6 µs | 160.1 µs | **3.67x** |
| 160 | 63.1 µs | 199.5 µs | **3.16x** |
| 256 | 94.3 µs | 220.9 µs | **2.34x** |
| 512 | 159.2 µs | 288.8 µs | 1.81x |
| 1024 | 308.9 µs | 307.4 µs | 1.00x |
| 2048 | 642.8 µs | 702.9 µs | 1.09x |
| 4096 | 1430.8 µs | 1394.6 µs | 0.97x (noise) |

- The numerical contract is `out = residual + rs[n] * round_out(acc * xs * ws + bias)`. **Rounding to
  the output dtype after the bias add** matches CUTLASS's `Add2 → Mul3 → Add4` visitor chain;
  omitting it breaks equality.
- The epilogue reads a second `[BM,BN]` tile, raising register pressure. The residual shape at
  `m >= 288` uses `BN=64` (`_INT8_GB10_RESIDUAL_TILE`), 8–12% faster than `128x128`.
- At M ≥ 1024 it is on par with the CUTLASS residual kernel. The position is: **wins big at small and
  medium M, never loses at large M**.

---

## 10. Re-measurement scripts

```bash
cd ~/comfy-kitchen && source .venv/bin/activate

python bench/bench_h3_breakdown.py fork     # H3 breakdown (GEMM/modulation/rotation/attention)
python bench/bench_int8_baseline.py fork    # int8_linear per shape
python bench/bench_thin_m.py fork           # M=16-128 crossover
python bench/bench_fp8_vs_int8.py fork      # comparison against FP8

~/archive/patches/switch_kernel_state.sh A  # back to stock
python bench/bench_h3_breakdown.py vanilla
```

The raw runs (`h3_breakdown_fork.json` / `h3_breakdown_vanilla.json`) ship in the repository. If
upstream touches the `_TRITON_INT8_MIN_M` band or the tile table, re-take the comparison tables here
(run `~/archive/patches/bench_int8_baseline.py` in the two states, A and `gb10-optimized`).
