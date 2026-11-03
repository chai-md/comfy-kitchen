"""One run, one protocol: everything that decides whether this fork helps H3.

Sections
  1. int8_linear end to end over the shapes that matter (thin / L2-resident / H3-wide).
  2. H3 per-iteration budget at M=15787: GEMM, rotate+quantize, modulate, attention.
  3. Which stage is bandwidth- or compute-bound, so the reader can tell what a rewrite
     can still buy.

Writes h3_breakdown_<label>.json next to the CWD.
"""

import json
import sys
import time

import torch
import torch.nn.functional as F

import comfy_kitchen as ck

LABEL = sys.argv[1] if len(sys.argv) > 1 else "run"
OUT = {}

# ---------------------------------------------------------------- helpers
def bench(fn, warmup=3, iters=10, rounds=4):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(rounds):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(iters):
            fn()
        e.record()
        torch.cuda.synchronize()
        best = min(best, s.elapsed_time(e) / iters)
    return best


def mk(m, n, k, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    w = torch.randint(-127, 127, (n, k), dtype=torch.int8, device="cuda")
    ws = torch.rand(n, dtype=torch.float32, device="cuda") * 0.02
    bias = torch.randn(n, dtype=torch.bfloat16, device="cuda")
    return x, w, ws, bias


# ---------------------------------------------------------------- 1. GEMM path
GEMM_SHAPES = [
    # (m, n, k, note)
    (64, 4096, 4096, "thin, weight 16MB (L2)"),
    (256, 4096, 4096, "thin-mid, weight 16MB (L2)"),
    (512, 4096, 4096, "mid, weight 16MB (L2)"),
    (4096, 4096, 4096, "large, weight 16MB (L2)"),
    (15787, 5376, 5376, "H3 out_proj, weight 115MB"),
    (15787, 21504, 5376, "H3 qkv, weight 115MB"),
    (15787, 28672, 5376, "H3 fc1, weight 154MB"),
    (15787, 5376, 14336, "H3 fc2, weight 77MB"),
]
print(f"== int8_linear ({LABEL}) ==")
print(f"{'shape (M x N x K)':>24} {'ms':>9} {'TOPS':>7}  note")
OUT["gemm"] = {}
for m, n, k, note in GEMM_SHAPES:
    x, w, ws, bias = mk(m, n, k)
    ms = bench(lambda: ck.int8_linear(x, w, ws, bias), iters=6 if m > 10000 else 15)
    tops = 2 * m * n * k / (ms * 1e-3) / 1e12
    key = f"{m}x{n}x{k}"
    OUT["gemm"][key] = {"ms": ms, "tops": tops, "note": note}
    print(f"{key:>24} {ms:9.3f} {tops:7.1f}  {note}")
    del x, w, ws, bias
    torch.cuda.empty_cache()

# ---------------------------------------------------------------- 2. H3 budget
M = 15787
print(f"\n== H3 per-iteration budget at M={M} (50 blocks) ==")
OUT["h3"] = {}

# 2a. the four int8 linears of one block, replicated x50
per_block = 0.0
for key in ("15787x21504x5376", "15787x5376x5376", "15787x28672x5376", "15787x5376x14336"):
    if key in OUT["gemm"]:
        per_block += OUT["gemm"][key]["ms"]
OUT["h3"]["gemm_per_block_ms"] = per_block
OUT["h3"]["gemm_per_forward_ms"] = per_block * 50
print(f"GEMM        {per_block*50:9.0f} ms/forward   ({per_block:.2f} ms x 50 blocks)")

# 2b. rotate + quantize (unchanged by this fork, kept for the budget)
try:
    from comfy_kitchen.tensor.int8_utils import _build_hadamard, _rotate_activation
    from comfy_kitchen.backends.triton.quantization import triton_quantize_rowwise

    h = _build_hadamard(256, device="cuda", dtype=torch.bfloat16)
    a = torch.randn(M, 5376, dtype=torch.bfloat16, device="cuda")
    t_rot = bench(lambda: _rotate_activation(a, h, 256), iters=8)
    rot = _rotate_activation(a, h, 256)
    t_q = bench(lambda: triton_quantize_rowwise(rot), iters=8)
    OUT["h3"]["rotate_ms"] = t_rot
    OUT["h3"]["quantize_ms"] = t_q
    print(f"rotate+quant {t_rot*200:9.0f} ms/forward   ({t_rot:.3f} + {t_q:.3f} ms x 200)")
    del a, rot
except Exception as exc:  # vanilla may not expose the triton quantizer
    print("rotate+quant n/a:", type(exc).__name__)

# 2c. modulation: torch reference vs the fused kernel (fork only)
try:
    from comfy_kitchen.tensor.scale_shift import scale_shift, row_mod_from_segments

    rows, cols = M, 5376
    big_s = torch.randn(3, 6, cols, dtype=torch.float32, device="cuda")
    big_b = torch.randn(3, 6, cols, dtype=torch.float32, device="cuda")
    s_view, b_view = big_s[:, 0], big_b[:, 0]
    hh = torch.randn(rows, cols, dtype=torch.bfloat16, device="cuda")
    segs = [(0, 388, 1), (388, 802, 2), (802, rows, 0)]
    rm = row_mod_from_segments(rows, segs, hh.device)

    def torch_mod():
        out = hh.clone()
        for a_, b_, r_ in segs:
            out[a_:b_].mul_(1.0 + s_view[r_].to(out.dtype)).add_(b_view[r_].to(out.dtype))
        return out

    t_mod_t = bench(torch_mod, iters=8)
    t_mod_f = bench(lambda: scale_shift(hh, s_view, b_view, rm), iters=8)
    got = scale_shift(hh, s_view, b_view, rm)
    ref = torch_mod()
    OUT["h3"]["modulate_torch_ms"] = t_mod_t
    OUT["h3"]["modulate_fused_ms"] = t_mod_f
    OUT["h3"]["modulate_equal"] = bool(torch.equal(got, ref))
    print(
        f"modulate     {t_mod_t*100:9.0f} -> {t_mod_f*100:.0f} ms/forward   "
        f"({t_mod_t:.3f} -> {t_mod_f:.3f} ms x 100, bit-identical={OUT['h3']['modulate_equal']})"
    )
    del hh, got, ref
except Exception as exc:
    print("modulate n/a:", type(exc).__name__)

# 2d. attention backend (config choice, not a fork change)
try:
    from sageattention import sageattn

    H, D = 56, 128
    q = torch.randn(1, H, M, D, dtype=torch.bfloat16, device="cuda")
    k_ = torch.randn_like(q)
    v = torch.randn_like(q)
    with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.FLASH_ATTENTION]):
        t_sdpa = bench(lambda: F.scaled_dot_product_attention(q, k_, v), iters=5)
    t_sage = bench(lambda: sageattn(q, k_, v, is_causal=False), iters=5)
    OUT["h3"]["attention_sdpa_ms"] = t_sdpa
    OUT["h3"]["attention_sage_ms"] = t_sage
    print(
        f"attention    {t_sdpa*50:9.0f} -> {t_sage*50:.0f} ms/forward   "
        f"(SDPA {t_sdpa:.2f} -> SageAttention {t_sage:.2f} ms x 50)"
    )
    del q, k_, v
except Exception as exc:
    print("attention n/a:", type(exc).__name__)

path = f"h3_breakdown_{LABEL}.json"
with open(path, "w") as f:
    json.dump(OUT, f, indent=2)
print("\nwrote", path)
