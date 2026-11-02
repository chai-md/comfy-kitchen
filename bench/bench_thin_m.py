"""Thin-M sweep, same protocol in both states."""
import json
import sys

import torch
import comfy_kitchen as ck

MS = [16, 24, 32, 40, 48, 56, 64, 96, 128]


def bench(fn, warmup=5, iters=30, rounds=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(rounds):
        start = torch.cuda.Event(True)
        end = torch.cuda.Event(True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        best = min(best, start.elapsed_time(end) / iters)
    return best


def main(label):
    torch.manual_seed(0)
    out = {}
    n = k = 4096
    w = torch.randint(-127, 127, (n, k), dtype=torch.int8, device="cuda")
    ws = torch.rand(n, dtype=torch.float32, device="cuda") * 0.02
    bias = torch.randn(n, dtype=torch.bfloat16, device="cuda")
    for m in MS:
        x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
        ms = bench(lambda: ck.int8_linear(x, w, ws, bias))
        out[m] = ms
        print(f"M={m:>4} {ms*1e3:8.1f} us", flush=True)
        del x
    with open(f"thin_m_{label}.json", "w") as f:   # written to the current directory
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main(sys.argv[1])
