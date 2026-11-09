# GB10 (`sm_121`) fork of `comfy-kitchen` — index

A fork of upstream `Comfy-Org/comfy-kitchen` that adds a **Triton INT8 GEMM path** and **fused
modulation (scale/shift)** for GB10 (sm_121). The fork is the seven commits on
`gb10-optimized`; this document set is split by purpose (it replaces the former single-file
`SM121_FORK_GUIDE.md`).

Japanese: [SM121_FORK_GUIDE.md](SM121_FORK_GUIDE.md)

| Item | Value |
| :--- | :--- |
| Target machine | dgxspark1 (GB10 / 48 SM / 2190 MHz, clock cap 2200) |
| Prerequisites | CUDA 13.x, Python 3.13, `torch 2.15+cu132`, Triton 3.9 |
| Version | `0.2.41+gb10.1` (PEP 440 local version, to avoid confusion with the upstream release) |
| Scope of change | **Python only** (no CUDA/C++ diff; `_C.abi3.so` is rebuilt but identical) |
| Default branch | `gb10-optimized` (`main` is an upstream mirror) |

---

## 0. Document map

| What you need | Document | Section |
| :--- | :--- | :--- |
| Get it running / reinstall | [OPERATIONS.md](docs/gb10/OPERATIONS.en.md) | §3–§5 |
| Branch workflow, reproduce on another machine | OPERATIONS.md | §2 |
| Track upstream (rebase) | OPERATIONS.md | §6 |
| Troubleshooting, rollback | OPERATIONS.md | §7–§8 |
| What changed and why (design rationale) | [DESIGN.md](docs/gb10/DESIGN.en.md) | §1–§7 |
| Optimizations we rejected, review record, known limits | DESIGN.md | §8–§10 |
| Measurements, how to re-measure | [MEASUREMENTS.md](docs/gb10/MEASUREMENTS.en.md) | §1–§10 |
| Acknowledgements and sources | DESIGN.md | §11 |

---

## 1. Shortest path (details in OPERATIONS §3–§5)

```bash
cd ~/comfy-kitchen
git submodule update --init --depth 1
source ~/ComfyUI/.venv/bin/activate          # the venv ComfyUI will load from (path differs per node)
COMFY_CUDA_ARCHS=121 uv pip install --no-build-isolation -e .
python -m pytest tests/test_int8.py tests/test_scale_shift.py -q      # 290 passed
git -C ~/ComfyUI apply ~/comfy-kitchen/patches/comfyui-minimax-modulation.patch
# restart ComfyUI to pick it up
```

---

## 2. Fork state (the essentials)

- The default branch is `gb10-optimized`. The fork's substance is the seven commits in
  `main..gb10-optimized`; `git log --oneline main..origin/gb10-optimized` is authoritative
  (SHAs change on every rebase).
- `main` is a mirror carrying upstream commits only. **Never check out `main` in this clone** —
  `~/comfy-kitchen` is installed editable into ComfyUI's venv, so switching to `main` silently
  puts ComfyUI back on the stock kernels. Update the mirror without a checkout:
  `git branch -f main upstream/main` (OPERATIONS §2).
- If upstream ships an equivalent feature (an sm_121 check, a fused kernel), **drop our gate and
  follow upstream**. Double dispatch is how accidents happen (OPERATIONS §6).

---

## 3. Summary of changes

| File | Change | Effect | Details |
| :--- | :--- | :--- | :--- |
| `comfy_kitchen/backends/triton/quantization.py` | Fused INT8 GEMM + dequant epilogue, per-M-band tile table, memoised `_ensure_python_headers()` | 1.5–2.4x on wide-N projections | DESIGN §2 |
| `comfy_kitchen/backends/cuda/__init__.py` | sm_121 dispatch (`_TRITON_INT8_MIN_M = 40`), mode reading, CUTLASS fallback | Exposes the same path to ComfyUI | DESIGN §3 |
| `comfy_kitchen/__init__.py` | Early return under `torch.compiler.is_compiling()` | Avoids the INT8 path failing under `torch.compile` | DESIGN §4 |
| `comfy_kitchen/tensor/scale_shift.py` (new) | Single-pass per-segment modulation, bit-identical | 3.0x on the modulation stage | DESIGN §5 |
| `tests/test_int8.py`, `tests/test_scale_shift.py` | Bit-identity, tile, edge-shape and residual regressions | Safety net for changes | DESIGN §6 |
| `patches/comfyui-minimax-modulation.patch` | ComfyUI-side hook (`_mod_scale_shift`) | Only pays off in real workloads | DESIGN §7 |

---

## 4. Measurement summary

H3 (M=15,787), one iteration: **11.4 s → 7.0 s (−4.4 s / 1.63x)**. The breakdown is INT8 GEMM
7.87 → 3.82 s and modulation 0.49 → 0.16 s; rotation, quantisation and attention are unchanged.
The attention difference comes from ComfyUI's `--use-sage-attention` launch flag, not from the fork.

The lever applies to **wide-N** projections (N ≳ 8192); at N ≤ 4096 it is 1.05–1.06x. Numbers,
measurement conditions and the re-measurement procedure are in
[MEASUREMENTS.md](docs/gb10/MEASUREMENTS.en.md).

---

## 5. Rules before you touch it

- After any change, run the three checks in OPERATIONS §5 (dispatch + bit-identity, tile table,
  pytest).
- Measure min-of-3–5 with a round-robin order. **Differences below 10% are not significant**
  (run-to-run spread is 10–15%).
- Do not retry the ideas already measured and rejected (DESIGN §8).
