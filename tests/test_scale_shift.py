"""Bit-identity tests for the fused scale/shift modulation.

The reference is the torch expression the DiT blocks use:

    for a, b, row in segments:
        h[a:b].mul_(1.0 + vecs[row].to(h.dtype)).add_(shifts[row].to(h.dtype))

Note the ``.to(h.dtype)``: H3's curve checkpoints keep the adaLN output in fp32,
and dropping that cast moved ~38% of the outputs (max 0.125) while every bf16
modulation case still matched. Both modulation dtypes are therefore covered here.
"""

import pytest
import torch

from comfy_kitchen.tensor.scale_shift import (
    row_mod_from_segments,
    scale_shift,
    scale_shift_inplace_,
)


def _reference(h, shift, scale, segments):
    out = h.clone()
    for a, b, row in segments:
        out[a:b].mul_(1.0 + scale[row].to(h.dtype)).add_(shift[row].to(h.dtype))
    return out


def _segments(rows, nseg):
    bounds = [rows * i // nseg for i in range(nseg + 1)]
    return [(bounds[i], bounds[i + 1], i) for i in range(nseg)]


DEVICES = ["cuda"] if torch.cuda.is_available() else []


@pytest.mark.skipif(not DEVICES, reason="needs CUDA")
@pytest.mark.parametrize("h_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("mod_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("nseg", [1, 2, 3])
def test_matches_torch(h_dtype, mod_dtype, nseg):
    rows, cols = 512, 4096
    h = torch.randn(rows, cols, dtype=h_dtype, device="cuda")
    scale = torch.randn(nseg, cols, dtype=mod_dtype, device="cuda")
    shift = torch.randn(nseg, cols, dtype=mod_dtype, device="cuda")
    segs = _segments(rows, nseg)

    got = scale_shift(h, scale, shift, row_mod_from_segments(rows, segs, h.device))
    ref = _reference(h, shift, scale, segs)
    torch.cuda.synchronize()
    assert torch.equal(got, ref)


@pytest.mark.skipif(not DEVICES, reason="needs CUDA")
@pytest.mark.parametrize("h_dtype", [torch.bfloat16, torch.float16])
def test_wide_dynamic_range(h_dtype):
    """Tiny (subnormal) and large values round like torch's .to(dtype).

    Truncating the fp32 mantissa rounds subnormals at the wrong granularity for
    fp16, so the modulation tensors are pushed through the subnormal band here.
    """
    rows, cols = 256, 2048
    h = (torch.randn(rows, cols, dtype=torch.float32, device="cuda") * 2.0).to(h_dtype)
    scale = torch.randn(1, cols, dtype=torch.float32, device="cuda")
    scale[0, : cols // 2] *= 1e-5          # subnormal after the fp16/bf16 cast
    scale[0, cols // 2:] *= 1e3            # large but in range
    shift = torch.randn(1, cols, dtype=torch.float32, device="cuda")
    shift[0, : cols // 2] *= 1e-5
    segs = [(0, rows, 0)]

    got = scale_shift(h, scale, shift, row_mod_from_segments(rows, segs, h.device))
    ref = _reference(h, shift, scale, segs)
    torch.cuda.synchronize()
    assert torch.equal(got, ref)


@pytest.mark.skipif(not DEVICES, reason="needs CUDA")
def test_per_token_index():
    """segments may carry a per-token mod-row tensor (blended modulation)."""
    rows, cols = 512, 1024
    h = torch.randn(rows, cols, dtype=torch.bfloat16, device="cuda")
    scale = torch.randn(3, cols, dtype=torch.float32, device="cuda")
    shift = torch.randn(3, cols, dtype=torch.float32, device="cuda")
    idx = torch.arange(rows, device="cuda") % 3
    segs = [(0, rows, idx)]

    got = scale_shift(h, scale, shift, row_mod_from_segments(rows, segs, h.device))
    ref = _reference(h, shift, scale, segs)
    torch.cuda.synchronize()
    assert torch.equal(got, ref)


@pytest.mark.skipif(not DEVICES, reason="needs CUDA")
@pytest.mark.parametrize("cols", [255, 1000, 1024, 1025, 5376])
def test_masked_tail(cols):
    """Column counts that do not divide the 1024-wide program must mask exactly."""
    rows = 300
    h = torch.randn(rows, cols, dtype=torch.bfloat16, device="cuda")
    scale = torch.randn(2, cols, dtype=torch.bfloat16, device="cuda")
    shift = torch.randn(2, cols, dtype=torch.bfloat16, device="cuda")
    segs = _segments(rows, 2)

    got = scale_shift(h, scale, shift, row_mod_from_segments(rows, segs, h.device))
    ref = _reference(h, shift, scale, segs)
    torch.cuda.synchronize()
    assert torch.equal(got, ref)


@pytest.mark.skipif(not DEVICES, reason="needs CUDA")
def test_inplace_helper_and_input_untouched():
    rows, cols = 256, 2048
    h = torch.randn(rows, cols, dtype=torch.bfloat16, device="cuda")
    scale = torch.randn(2, cols, dtype=torch.float32, device="cuda")
    shift = torch.randn(2, cols, dtype=torch.float32, device="cuda")
    segs = _segments(rows, 2)
    row_mod = row_mod_from_segments(rows, segs, h.device)

    before = h.clone()
    out = scale_shift(h, scale, shift, row_mod)
    assert torch.equal(h, before), "out-of-place kernel must not touch its input"

    back = scale_shift_inplace_(h.clone(), scale, shift, row_mod)
    assert torch.equal(back, out)


@pytest.mark.skipif(not DEVICES, reason="needs CUDA")
def test_strided_modulation_view():
    """ComfyUI slices scale/shift out of a wider [R, n*cols] adaLN projection.

    A [R, cols] view with row stride n*cols is not contiguous; addressing it as
    row*cols reads the wrong row for every mod row but the first, which is exactly
    how a real H3 render diverged while every contiguous test passed.
    """
    rows, cols = 1024, 2048
    big_scale = torch.randn(3, 6, cols, dtype=torch.float32, device="cuda")
    big_shift = torch.randn(3, 6, cols, dtype=torch.float32, device="cuda")
    scale, shift = big_scale[:, 0], big_shift[:, 0]
    assert not scale.is_contiguous()

    h = torch.randn(rows, cols, dtype=torch.bfloat16, device="cuda")
    segs = [(0, 100, 1), (100, 400, 2), (400, rows, 0)]
    got = scale_shift(h, scale, shift, row_mod_from_segments(rows, segs, h.device))
    ref = _reference(h, shift, scale, segs)
    torch.cuda.synchronize()
    assert torch.equal(got, ref)


@pytest.mark.skipif(not DEVICES, reason="needs CUDA")
def test_partial_coverage_raises():
    with pytest.raises(ValueError, match="have no mod row"):
        row_mod_from_segments(10, [(0, 5, 0)], "cuda")


@pytest.mark.skipif(not DEVICES, reason="needs CUDA")
def test_index_length_mismatch_raises():
    idx = torch.zeros(4, dtype=torch.long, device="cuda")
    with pytest.raises(ValueError, match="mod-row indices"):
        row_mod_from_segments(10, [(0, 5, idx), (5, 10, 0)], "cuda")


@pytest.mark.skipif(not DEVICES, reason="needs CUDA")
def test_rejects_strided_activation():
    base = torch.randn(64, 512, dtype=torch.bfloat16, device="cuda")
    h = base[:, ::2]
    scale = torch.randn(1, 256, dtype=torch.bfloat16, device="cuda")
    shift = torch.randn(1, 256, dtype=torch.bfloat16, device="cuda")
    row_mod = row_mod_from_segments(64, [(0, 64, 0)], h.device)
    with pytest.raises(ValueError, match="contiguous"):
        scale_shift(h, scale, shift, row_mod)


@pytest.mark.skipif(not DEVICES, reason="needs CUDA")
def test_rejects_non_float_activation():
    h = torch.randint(0, 100, (64, 256), dtype=torch.int32, device="cuda")
    scale = torch.randn(1, 256, dtype=torch.float32, device="cuda")
    shift = torch.randn(1, 256, dtype=torch.float32, device="cuda")
    row_mod = row_mod_from_segments(64, [(0, 64, 0)], h.device)
    with pytest.raises(Exception):
        scale_shift(h, scale, shift, row_mod)
