# GB10 fork operations — build, verify, track upstream, troubleshoot

Scope: the `gb10-optimized` branch of `chai-md/comfy-kitchen`. Design rationale is in
[DESIGN.en.md](DESIGN.en.md); measurements in [MEASUREMENTS.en.md](MEASUREMENTS.en.md).
Japanese: [OPERATIONS.md](OPERATIONS.md)

---

## 1. Prerequisites

- Target machine: dgxspark1 (GB10 / 48 SM / 2190 MHz, clock cap 2200)
- CUDA 13.x, Python 3.13, `torch 2.15+cu132`, Triton 3.9
- Work inside a venv you have activated with `source .venv/bin/activate` (§3.2). The venv path
  differs per node, so do not hard-code absolute paths into these steps.

---

## 2. Fork and branch

| Item | Reality |
| :--- | :--- |
| Fork | `chai-md/comfy-kitchen` (`origin`, HTTPS). **The default branch is `gb10-optimized`** |
| Upstream | `Comfy-Org/comfy-kitchen` (`upstream`, followed with `git fetch upstream`) |
| Work branch | **`gb10-optimized`** (7 commits: 5 patch commits, 1 docs/bench, 1 version. Always rebased directly onto `main`) |
| `main` | **Upstream mirror** (upstream commits only). Never commit to it, never check it out |
| Substance of the fork | The 7 commits in `git log --oneline main..origin/gb10-optimized` |
| Local | This repository. Build/test in the `.venv` at the repo root; to have ComfyUI use it, install it editable into ComfyUI's venv as well (§3.3) |

```bash
# Create the fork and clone (first time only)
gh repo fork Comfy-Org/comfy-kitchen --clone
cd comfy-kitchen
git remote add upstream https://github.com/Comfy-Org/comfy-kitchen.git
git fetch upstream
git switch -c gb10-optimized --track origin/gb10-optimized   # when pulling from an existing fork
git submodule update --init --depth 1   # mandatory (§3.1)
```

**Never check out `main`.** `~/comfy-kitchen` is installed editable into ComfyUI's venv, so the
moment you switch this clone to `main`, ComfyUI starts running the stock kernels (silently).
Update the mirror without a checkout:

```bash
git fetch upstream && git branch -f main upstream/main && git push origin main
```

The history is 7 commits (5 patch commits, 1 docs/bench, 1 version). **Track it by range, not by
branch name** (`main..gb10-optimized` is the substance of the fork).

| Commit | Content |
| :--- | :--- |
| `f5d2558` | Triton INT8 fused GEMM + tile table + dispatch + residual fusion + first `tensor/scale_shift.py` + tests |
| `dfcd33d` | `scale_shift`: round modulation tensors to the activation dtype (fp32 adaLN support; without it the H3 picture changes) |
| `b319605` | `scale_shift`: `.contiguous()` for non-contiguous modulation views, add coverage validation |
| `2ce8f0d` | Ship the ComfyUI-side hook as `patches/comfyui-minimax-modulation.patch` |
| `0f83d79` | Fix the comparison baseline to stock comfy-kitchen, raise the Triton gate to 40, ship the `bench/` scripts |
| `e0aa594` | One consolidation commit for guide/bench/docs (11 earlier commits squashed; original messages kept in the commit body) |
| `400aee0` | Set `pyproject.toml` version to `0.2.41+gb10.1` (to avoid confusion with the upstream release) |

The `+gb10.1` is a PEP 440 local version; installing `pip install git+https://github.com/chai-md/comfy-kitchen`
from the default branch produces exactly that. An existing editable environment keeps reporting
`0.2.41` in its dist-info until you reinstall (the import target does not change, so behaviour is
unaffected).

> Note: every rebase rewrites the SHAs. The SHAs above are as of 2026-11-09. **Track by range and
> content.**

```bash
# Reproduce the same state on another machine
git fetch origin && git switch gb10-optimized && git reset --hard origin/gb10-optimized
```

`sm121_optimize.patch` is a `git diff` snapshot and **is not kept in the repository** (it lives in
`~/archive/patches/`). If you need one:
`git diff main..gb10-optimized -- comfy_kitchen tests > ~/archive/patches/sm121_optimize.patch`.

---

## 3. Build and install

### 3.1 Submodules (do this first)

```bash
cd ~/comfy-kitchen
git submodule update --init --depth 1
```

Building with them empty causes two accidents:

1. `flash_decode.cu` cannot find the `flash_attn` headers and the build fails partway through.
2. CMake disables `COMFY_HAVE_CUTLASS`, so `cutlass_gemm_int8.cu` and friends compile as **4 KB
   dummy stubs** — the build succeeds but runs slowly.

If you built once before initialising them, drop the CMake cache with `rm -rf build`.

### 3.2 Build (editable)

Requires **torch (2.15+cu132) and triton (3.9) in that venv**. `--no-build-isolation` builds
against the existing torch, so it fails in an empty venv.

```bash
cd ~/comfy-kitchen                 # repository location per node

# if the repository has its own .venv
source .venv/bin/activate
# otherwise use that node's ComfyUI venv (it has torch/triton)
#   source ~/ComfyUI/.venv/bin/activate

which python                      # confirm it points at the venv
uv pip install nanobind packaging wheel cmake ninja   # only if missing
COMFY_CUDA_ARCHS=121 uv pip install --no-build-isolation -e .
```

Only for a brand-new venv, install the dependencies first (several GB, several minutes).

```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -U uv
uv pip install torch triton        # match the ComfyUI version
```

- `COMFY_CUDA_ARCHS=121` skips older architectures (about 2.5 minutes on 20 cores).
- With `-e` (editable) `_C.abi3.so` lands inside the repository, so **Python-only changes need no
  rebuild**.
- Check: `ls -l comfy_kitchen/backends/cuda/_C.abi3.so` should be around 12 MB (4 KB means §3.1).
- `uv pip install` installs into the **activated** venv (it reads `VIRTUAL_ENV`). Run it with none
  activated and it lands in another interpreter and "cannot import" — verify with `which python`
  that you point at `.venv`.
- `uv pip` works even when that venv has no `pip` module. Do not use `python -m pip`.

### 3.3 Install into ComfyUI's venv

Editable installs are **per venv**. For verification alone §3.2's `.venv` is enough; to have
ComfyUI load it, activate that node's ComfyUI venv and run the same command (safer than adding
`--python`).

```bash
source ~/ComfyUI/.venv/bin/activate      # path per node
cd ~/comfy-kitchen
COMFY_CUDA_ARCHS=121 uv pip install --no-build-isolation -e .
python -c "import comfy_kitchen, pathlib; print(pathlib.Path(comfy_kitchen.__file__).parent)"
# → <repo>/comfy_kitchen means the editable install is in effect
```

Only the Python side skips the rebuild. **If you touch C++/CUDA (`_C.abi3.so`), rebuild in both
venvs.**

### 3.4 Post-install smoke test

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
```

Expected: `cap (12, 1) triton-int8 True` and `scale_shift bit-identical: True`.

---

## 4. Applying the ComfyUI-side hook (optional)

ComfyUI is an upstream clone (`Comfy-Org/ComfyUI`) with no push target, so the hook ships as
`patches/comfyui-minimax-modulation.patch` in this repository.

**Not applied by default since 2026-11-11** (0.28 s per forward does not pay for the maintenance;
DESIGN §7). Run the following only when you want it.

```bash
git -C ~/ComfyUI apply ~/comfy-kitchen/patches/comfyui-minimax-modulation.patch

# revert (back to upstream behaviour)
git -C ~/ComfyUI apply -R ~/comfy-kitchen/patches/comfyui-minimax-modulation.patch

# re-apply after pulling upstream (if it conflicts, keep upstream's added lines and insert this
# patch's import block alongside them)
```

This patch is refreshed for ComfyUI `0228a9d73` (v0.45.0). Upstream #17655 (H3-image support) moved the
insertion point next to `_tag_runs`, so the previous revision conflicts in `model.py`.

### 4.1 H3 context-window fix (separate patch, optional)

The change to `PackedLayout` in `model.py` and `_window_payload` in `model_base.py` is kept as a
separate patch (`~/archive/patches/comfyui-h3-contextwindow-*.diff`). **It is also not applied by
default.** It is only needed for context windows (split generation beyond 10 s); single-shot runs do
not need it (windowing itself costs +47% step time and continuity is still imperfect). If you apply
it, do so after the hook and verify the positions with `check_h3_window_positions.py` (video err ≲1e-9).

```bash
git -C ~/ComfyUI apply ~/archive/patches/comfyui-h3-contextwindow-20261111.diff
python <skill>/scripts/check_h3_window_positions.py     # OK - windows sit at their real clip position
```

Applying it requires a **ComfyUI restart** (Python is re-read only).

---

## 5. Verification (all of it after any change)

### 5.1 Dispatch and bit-identity

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

### 5.2 Tile table

```bash
python -c "
from comfy_kitchen.backends.triton.quantization import _int8_tile_config as f
for m in (24, 64, 127, 128, 207, 208, 287, 288, 4096):
    print(m, f(m, 4096, 4096)[:2], '|', m, f(m, 5120, 5120)[:2])
"
```

Expected: `24..127 → (64,64)` / `128..207 → (64,128)` / `208..287 → (128,64)` / `288.. → (128,128)`;
`5120×5120` is `(128,128)` throughout.

### 5.3 Unit tests

```bash
uv pip install pytest        # if missing
python -m pytest tests/test_int8.py tests/test_scale_shift.py -q
```

Expected: everything passes (`290 passed` here = upstream + 96 added cases). `test_scale_shift.py`
also fails on non-contiguous views, subnormals and coverage gaps, so run both whenever you touch
the kernel. Re-measurement is covered in [MEASUREMENTS.en.md](MEASUREMENTS.en.md) §10.

---

## 6. Tracking upstream

The work branch is `gb10-optimized` (§2). To move onto a new upstream release, follow this order.

```bash
cd ~/comfy-kitchen
git fetch upstream
git switch gb10-optimized                   # never check out main (§2)
git rebase upstream/main                    # replay the 7-commit patch series onto the upstream tip
git submodule update --init --depth 1

source .venv/bin/activate                   # the venv from §3.2
COMFY_CUDA_ARCHS=121 uv pip install --no-build-isolation -e .
python -m pytest tests/test_int8.py tests/test_scale_shift.py -q
git push --force-with-lease origin gb10-optimized

# fast-forward the main mirror (no checkout)
git branch -f main upstream/main && git push origin main
```

- GitHub's "Sync fork" button should only be pressed **while `main` is open** (`gb10-optimized` has
  no same-named upstream branch, so the button does not appear there). `gh repo sync` also looks at
  the default branch — if you use it, be explicit: `gh repo sync chai-md/comfy-kitchen -b main`.
- If you tag sync points, avoid a `v` prefix: use `gb10-0.2.41-1`, because the fork's
  `build-wheels.yml` triggers a full wheel-build matrix on `tags: v*`.
- The conflict-prone spots are **(1) the three `_cutlass_int8_dequant()` call sites**, **(2) the tail
  of `int8_linear()` on the triton side**, **(3) the head of `int8_linear()` in `__init__.py`**.
- If upstream ships an equivalent feature (an sm_121 check or a fused kernel), **drop our gate and
  follow upstream**. Double dispatch is how accidents happen.
- Re-run §5 afterwards (especially 5.1 bit-identity; if upstream changes the epilogue's operation
  order it will no longer match).
- When ComfyUI uses the fork, confirm the install target is the fork path (§3.3 check command).
- If upstream touches the `_TRITON_INT8_MIN_M` band or the tile table, re-take the comparison tables
  in [MEASUREMENTS.en.md](MEASUREMENTS.en.md) (run `~/archive/patches/bench_int8_baseline.py` in the
  two states, A and `gb10-optimized`).
- A ComfyUI `git pull` drops the hook, so re-apply the patch from §4.

---

## 7. Troubleshooting

| Symptom | Cause / fix |
| :--- | :--- |
| `ModuleNotFoundError: comfy_kitchen` | The editable install's target (the repository) is gone. Re-run §3 |
| The build succeeds but is slow (no better than CUTLASS) | Submodules were not initialised, so `cutlass_gemm_int8.cu` became a dummy. `rm -rf build` → §3.1 |
| `Kernel requires a runtime memory allocation, but no allocator was set` | Only when experimenting with TMA descriptors. Set `triton.set_allocator(...)` |
| `Triton INT8 GEMM failed ... falling back to CUTLASS` on stderr | `_triton_int8_broken` is set and that session stays on CUTLASS. Fix the cause (usually a missing Python.h or a Triton compile failure) and restart the process |
| `Python.h` not found | `_ensure_python_headers()` falls back to `/opt/py312-include` (container use). On a host venv the `sysconfig` include is used |
| Numbers do not match CUTLASS | The launcher's `BM/BN/BK/G/warps/stages` are still constants (DESIGN §2). Note that **when N is not a multiple of 16 the upstream CUTLASS kernel is the broken one** (DESIGN §9.4), so judge against an exact reference |
| `expand()`ed scales produce wrong values / OOB | The gate requires `is_contiguous()`. Call `.contiguous()` on the caller side or fall back to CUTLASS |
| `torch.compile` autograd error | Inputs with `requires_grad` deliberately fall back to Python tracing (DESIGN §9.1). Training needs autograd registered on the custom op |
| ComfyUI's startup log still reports `comfy-kitchen version` `0.2.41` | You have not reinstalled per §3.3 (it should read `0.2.41+gb10.1`) |

---

## 8. Rollback (back to stock comfy-kitchen)

```bash
# 1) runtime only: turn the dispatch off (a process restart reverts it)
COMFY_KITCHEN_INT8_MODE=cutlass python main.py ...

# 2) persistent: the mode file (/etc/comfy-kitchen-int8.mode, write off / cutlass)

# 3) revert only the ComfyUI hook
git -C ~/ComfyUI apply -R ~/comfy-kitchen/patches/comfyui-minimax-modulation.patch

# 4) drop the fork entirely (upstream-equivalent state)
source ~/ComfyUI/.venv/bin/activate
uv pip install --force-reinstall "comfy-kitchen==0.2.41"
```

Note: `comfy-kitchen==0.2.41` matches `0.2.41+gb10.1` under PEP 440's public-version rule, so
without `--force-reinstall` the fork's editable install survives.
