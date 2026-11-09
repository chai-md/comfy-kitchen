"""Bandwidth-bound stages still on the slow path in H3 (torch / in-place ops).

Measures, at the real H3 shape (M=15787):
  - _mod_gate: x[a:b].addcmul_(other[a:b], gate)   x2 per layer, in place
  - ck.rms_rope_split_half_: fused qk-norm + rope, in place on the qkv buffer
  - a fused out-of-place equivalent for _mod_gate, to size the headroom
"""

import torch

import comfy.ldm.minimax.model as M
from comfy.quant_ops import ck

ROWS, HID = 15787, 5376
HEADS, HD = 56, 128


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


# ---------------------------------------------------------------- _mod_gate
x = torch.randn(ROWS, HID, dtype=torch.bfloat16, device="cuda")
other = torch.randn(ROWS, HID, dtype=torch.bfloat16, device="cuda")
gate = torch.randn(3, HID, dtype=torch.float32, device="cuda")
segs = [(0, 388, 1), (388, 802, 2), (802, ROWS, 0)]

t_gate = bench(lambda: M._mod_gate(x.clone(), gate, other, segs))
bytes_gate = ROWS * HID * 2 * 3          # read x, read other, write x
print(f"_mod_gate (addcmul_, in place)   {t_gate*1000:8.0f} us  x100 = {t_gate*100:7.0f} ms/fwd  "
      f"({bytes_gate/t_gate/1e6:5.0f} GB/s)")


# out-of-place fused equivalent, same arithmetic (x + other * gate[row])
def fused_gate(x, other, gate, segs):
    out = torch.empty_like(x)
    for a, b, r in segs:
        torch.addcmul(x[a:b], other[a:b], gate[r].to(x.dtype), out=out[a:b])
    return out


t_gate_o = bench(lambda: fused_gate(x, other, gate, segs))
print(f"_mod_gate (out of place)         {t_gate_o*1000:8.0f} us  x100 = {t_gate_o*100:7.0f} ms/fwd  "
      f"({bytes_gate/t_gate_o/1e6:5.0f} GB/s)  -> {t_gate/t_gate_o:.2f}x")

# ---------------------------------------------------------------- rms_rope
q = torch.randn(1, ROWS, HEADS, HD, dtype=torch.bfloat16, device="cuda")
k = torch.randn_like(q)
rot = 32
rope = torch.randn(1, ROWS, 1, rot // 2, 2, 2, dtype=torch.bfloat16, device="cuda")
qw = torch.ones(HD, dtype=torch.bfloat16, device="cuda")
kw = torch.ones(HD, dtype=torch.bfloat16, device="cuda")
eps = 1e-5

fn = getattr(ck, "rms_rope_split_half_", None)
if fn is not None:
    t_rope = bench(lambda: fn(q, k, rope, qw, kw, epsilon=eps, rot_dim=rot))
    n_qk = 2 * ROWS * HEADS * HD * 2
    print(f"rms_rope_split_half_ (in place)  {t_rope*1000:8.0f} us  x50  = {t_rope*50:7.0f} ms/fwd  "
          f"({n_qk*2/t_rope/1e6:5.0f} GB/s)")
else:
    print("rms_rope_split_half_: not found on cuda backend")
