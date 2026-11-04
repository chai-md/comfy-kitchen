"""One mode per process: stock vs gpu-buffer, so the CPU allocator state is comparable.

  python bench_vae_decode2.py stock|gpu
"""
import sys
import time

import torch

sys.path.insert(0, "/home/chai/ComfyUI")
import comfy.utils
import comfy.sd
import comfy.ldm.minimax.vae as vae_mod

VAE_PATH = "/home/chai/ComfyUI/models/vae/minimax_h3_video_vae_int8_convrot.safetensors"
B, C, T, H, W = 1, 24, 31, 48, 84

mode = sys.argv[1]
torch.manual_seed(0)

vae_mod._gpu_buffer_enabled = lambda: (mode == "gpu")

print("loading ...", flush=True)
sd = comfy.utils.load_torch_file(VAE_PATH, safe_load=True)
vae = comfy.sd.VAE(sd=sd)
model = vae.first_stage_model.to("cuda").eval()
dt = next(model.parameters()).dtype
print("frame plan:", model._decode_temporal_chunks(T), "dtype:", dt, flush=True)

# warmup on a small clip (does not touch the big output buffer)
with torch.inference_mode():
    model.decode(torch.zeros(B, C, 5, H, W, dtype=dt, device="cuda"), output_buffer=None)
torch.cuda.empty_cache()

z = (torch.randn(B, C, T, H, W, device="cuda") * 0.5).to(dt)
shape = model.decode_output_shape(z.shape)
buf = torch.empty(shape, dtype=torch.float32, device="cpu")   # fresh pageable memory

torch.cuda.synchronize()
t0 = time.time()
with torch.inference_mode():
    out = model.decode(z, output_buffer=buf)
torch.cuda.synchronize()
print(f"{mode}: {time.time() - t0:6.2f} s  out={tuple(out.shape)} cpu={out.device.type == 'cpu'} "
      f"finite={bool(torch.isfinite(out).all())}", flush=True)
