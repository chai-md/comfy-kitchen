"""One H3-shaped int8_linear (qkv_proj proportions), fork vs vanilla."""
import sys

import torch
import comfy_kitchen as ck

M, N, K = 15787, 21504, 5376   # H3 packed rows x qkv_proj


def bench(fn, warmup=3, iters=8, rounds=3):
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


x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
w = torch.randint(-127, 127, (N, K), dtype=torch.int8, device="cuda")
ws = torch.rand(N, dtype=torch.float32, device="cuda") * 0.02
bias = torch.randn(N, dtype=torch.bfloat16, device="cuda")

ms = bench(lambda: ck.int8_linear(x, w, ws, bias))
tops = 2 * M * N * K / (ms * 1e-3) / 1e12
print(f"{sys.argv[1]}: M={M} N={N} K={K} -> {ms*1e3:.1f} us  {tops:.1f} TOPS", flush=True)
