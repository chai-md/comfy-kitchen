"""FP8 (cuBLAS scaled_mm) vs INT8 (this fork's Triton kernel), kernel only.

Both are 1 byte per weight, so any difference is tensor-core rate + epilogue, not
bandwidth. quantize/rotate is excluded on both sides.
"""

import torch

from comfy_kitchen.backends.triton.quantization import triton_int8_gemm_dequant

SHAPES = [
    (15787, 5376, 5376, "H3 out_proj"),
    (15787, 21504, 5376, "H3 qkv"),
    (15787, 28672, 5376, "H3 fc1"),
    (15787, 5376, 14336, "H3 fc2"),
    (4096, 4096, 4096, "4096^3"),
]


def bench(fn, warmup=3, iters=8, rounds=4):
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


print(f"{'shape':>22} {'INT8':>18} {'FP8':>18} {'fp8/int8':>9}")
for m, n, k, note in SHAPES:
    x8 = torch.randint(-127, 127, (m, k), dtype=torch.int8, device="cuda")
    w8 = torch.randint(-127, 127, (n, k), dtype=torch.int8, device="cuda")
    xs = (torch.rand(m, 1, device="cuda") * 0.02).contiguous()
    ws = (torch.rand(n, device="cuda") * 0.02).contiguous()
    bias = torch.empty(0, dtype=torch.bfloat16, device="cuda")
    out = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
    t_i8 = bench(lambda: triton_int8_gemm_dequant(x8, w8, xs, ws, bias, out))

    a = (torch.randn(m, k, device="cuda") / 4).to(torch.float8_e4m3fn)
    b = (torch.randn(n, k, device="cuda") / 4).to(torch.float8_e4m3fn)
    sa = torch.tensor(1.0, device="cuda")
    sb = torch.tensor(1.0, device="cuda")
    o8 = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
    try:
        t_f8 = bench(
            lambda: torch._scaled_mm(a, b.t(), scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16, out=o8)
        )
    except TypeError:
        t_f8 = bench(
            lambda: torch._scaled_mm(a, b.t(), scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
        )
    fl = 2 * m * n * k
    print(
        f"{m}x{n}x{k:<6} {t_i8:8.2f} ms {fl/t_i8/1e9:6.0f} T "
        f"{t_f8:8.2f} ms {fl/t_f8/1e9:6.0f} T {t_i8/t_f8:9.2f}x   {note}"
    )
    del x8, w8, xs, ws, out, a, b, o8
    torch.cuda.empty_cache()
