"""Vanilla comfy-kitchen vs this fork: same script, run in both states.

    python bench/bench_int8_baseline.py fork      # current tree
    switch_kernel_state.sh A && python bench/bench_int8_baseline.py vanilla

Writes int8_baseline_<label>.json in the current directory.
"""
import json
import sys
import time

import torch
import comfy_kitchen as ck

SHAPES = [
    (32, 4096, 4096),
    (64, 4096, 4096),
    (128, 4096, 4096),
    (160, 4096, 4096),
    (192, 4096, 4096),
    (256, 4096, 4096),
    (512, 4096, 4096),
    (1024, 4096, 4096),
    (4096, 4096, 4096),
    (2048, 5120, 5120),
]


def bench(fn, warmup=5, iters=20, rounds=5):
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


def cpu_per_call(fn, calls=200):
    fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(calls):
        fn()
    elapsed = (time.perf_counter() - t0) / calls * 1e6
    torch.cuda.synchronize()
    return elapsed


def main(label):
    torch.manual_seed(0)
    result = {"label": label, "torch": torch.__version__, "shapes": {}}
    dev = "cuda"
    for m, n, k in SHAPES:
        x = torch.randn(m, k, dtype=torch.bfloat16, device=dev)
        w = torch.randint(-127, 127, (n, k), dtype=torch.int8, device=dev)
        ws = (torch.rand(n, dtype=torch.float32, device=dev) * 0.02)
        bias = torch.randn(n, dtype=torch.bfloat16, device=dev)

        def run(x=x, w=w, ws=ws, bias=bias):
            return ck.int8_linear(x, w, ws, bias)

        ms = bench(run)
        tops = 2 * m * n * k / (ms * 1e-3) / 1e12
        result["shapes"][f"{m}x{n}x{k}"] = {"ms": ms, "tops": tops}
        print(f"{m:>6}x{n:<6}x{k:<6} {ms*1e3:9.1f} us  {tops:6.1f} TOPS", flush=True)
        del x, w, ws, bias
        torch.cuda.empty_cache()

    # per-call CPU cost on a thin shape (launch side only, GPU async)
    m, n, k = 32, 4096, 4096
    x = torch.randn(m, k, dtype=torch.bfloat16, device=dev)
    w = torch.randint(-127, 127, (n, k), dtype=torch.int8, device=dev)
    ws = torch.rand(n, dtype=torch.float32, device=dev) * 0.02
    bias = torch.randn(n, dtype=torch.bfloat16, device=dev)
    result["cpu_us_per_call"] = cpu_per_call(lambda: ck.int8_linear(x, w, ws, bias))
    print(f"cpu per call: {result['cpu_us_per_call']:.1f} us", flush=True)

    out = f"int8_baseline_{label}.json"   # written to the current directory
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1])
