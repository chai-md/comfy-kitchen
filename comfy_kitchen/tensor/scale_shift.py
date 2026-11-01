# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-segment affine modulation of an activation matrix, bit-identical to torch.

The pre-norm blocks of several DiTs (MiniMax H3, Krea2, Z-Image) apply

    for a, b, row in segments:
        h[a:b] = h[a:b] * (1 + scale[row]) + shift[row]

before the quantized linear. In torch that is two read-modify-write passes over a
bf16 matrix, which lands at ~115-150 GB/s on GB10. One out-of-place Triton pass
runs at ~222 GB/s and produces the same bits: measured at M=27280 (H3, K=4096),
4335 us -> 2028 us (2.1x); against a cold torch baseline up to 3.0x.

Reading and writing the *same* buffer caps at ~115 GB/s (write-allocate), so the
output is a fresh tensor; callers that need in-place use ``scale_shift_inplace_``.

The bit-exactness needs care: Triton contracts ``a * s + b`` into an FMA and drops
the intermediate bf16 rounding that ``Tensor.mul_()`` performs, so both rounds are
applied explicitly with round-to-nearest-even integer math.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _rne_bf16(v):
    """Round an fp32 value to bf16 precision, round-to-nearest-even, as fp32 bits.

    Triton contracts ``x * s + b`` into an FMA and silently drops the bf16 rounding
    that ``Tensor.mul_()`` performs between the two ops, so the rounding has to be
    re-imposed with something the compiler cannot fold away. Integer ops are safe
    from FP contraction.

    The bit trick is the standard one: add half an ulp (0x7FFF of the 16 discarded
    mantissa bits) plus the LSB of the kept part to break ties to even, then mask
    the low 16 bits. This costs nothing measurable here: the kernel is DRAM-bound
    (222 GB/s) and all four variants -- no rounding, RNE bit math, inline asm
    ``cvt.rn.bf16.f32``, plain copy -- land within 3% of each other at M=27280.
    """
    b = v.to(tl.int32, bitcast=True)
    b = (b + 0x7FFF + ((b >> 16) & 1)) & -65536
    return b.to(tl.float32, bitcast=True)


@triton.jit
def _rne_fp16(v):
    """Round fp32 to fp16 precision (RNE).

    The bf16-style integer trick does not work here: fp16 has a 5-bit exponent, so
    truncating the fp32 mantissa rounds the subnormal band (|x| < 6.1e-5) at the
    wrong granularity. A real ``cvt.rn.f16.f32`` is exact over the whole range and
    cannot be folded into an FMA.
    """
    h = tl.inline_asm_elementwise(
        "cvt.rn.f16.f32 $0, $1;", "=h,r", [v],
        dtype=tl.float16, is_pure=True, pack=1,
    )
    return h.to(tl.float32)


@triton.jit
def _scale_shift_kernel(x_ptr, s_ptr, shift_ptr, row_mod_ptr, y_ptr, n_elements,
                        BLOCK: tl.constexpr, DTYPE_CODE: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    mod = tl.load(row_mod_ptr + row)
    base = row.to(tl.int64) * n_elements
    m_base = mod.to(tl.int64) * n_elements

    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    s_raw = tl.load(s_ptr + m_base + offs, mask=mask, other=0.0).to(tl.float32)
    sh_raw = tl.load(shift_ptr + m_base + offs, mask=mask, other=0.0).to(tl.float32)

    # torch does _mod_row(scale, row, h.dtype) first, i.e. it rounds the modulation
    # tensors to the activation dtype before touching them. H3's curve checkpoints
    # keep the adaln output in fp32, so skipping this cast moves ~38% of the outputs
    # (max 0.125) even though the arithmetic below is right.
    if DTYPE_CODE == 2:
        s1 = _rne_bf16(1.0 + _rne_bf16(s_raw))
        sh = _rne_bf16(sh_raw)
        y = _rne_bf16(_rne_bf16(x * s1) + sh).to(tl.bfloat16)
    elif DTYPE_CODE == 1:
        s1 = _rne_fp16(1.0 + _rne_fp16(s_raw))
        sh = _rne_fp16(sh_raw)
        y = _rne_fp16(_rne_fp16(x * s1) + sh).to(tl.float16)
    else:
        y = x * (1.0 + s_raw) + sh_raw
    tl.store(y_ptr + base + offs, y, mask=mask)


def row_mod_from_segments(rows: int, segments, device) -> torch.Tensor:
    """Build the per-token modulation index from [(start, stop, row), ...].

    ``row`` may be an int or a per-token index tensor. The segments must tile
    ``range(rows)``: torch leaves uncovered rows untouched, whereas an uninitialised
    index would apply some other row's modulation, so a gap is an error.
    """
    row_mod = torch.full((rows,), -1, dtype=torch.int32, device=device)
    for a, b, r in segments:
        if isinstance(r, torch.Tensor):
            idx = r.to(torch.int32).reshape(-1)
            if idx.numel() != b - a:
                raise ValueError(
                    f"scale_shift: segment [{a}:{b}) has {idx.numel()} mod-row indices"
                )
            row_mod[a:b] = idx
        else:
            row_mod[a:b] = r
    gap = row_mod < 0
    if bool(gap.any()):
        raise ValueError(
            f"scale_shift: {int(gap.sum())} rows have no mod row "
            f"(first {int(gap.nonzero()[0])})"
        )
    return row_mod


def scale_shift(h: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor,
                row_mod: torch.Tensor, block: int = 1024) -> torch.Tensor:
    """Return ``h * (1 + scale[row_mod]) + shift[row_mod]``, bit-identical to torch.

    Args:
        h: [rows, cols] bf16/fp16/fp32 activation.
        scale: [R, cols] per-modulation-row scale (bf16/fp16 like ``h``).
        shift: [R, cols] per-modulation-row shift.
        row_mod: [rows] int32 modulation-row index (see ``row_mod_from_segments``).
        block: columns per program (1024 measured best on GB10).

    The result is a new tensor; ``h`` is left untouched.
    """
    rows, cols = h.shape
    if h.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        # The kernel has no rounding contract for other dtypes; refuse rather than
        # silently returning something that is not bit-identical to torch.
        raise TypeError(f"scale_shift: unsupported activation dtype {h.dtype}")
    if scale.shape[-1] != cols or shift.shape[-1] != cols:
        raise ValueError(
            f"scale_shift: modulation width {scale.shape[-1]}/{shift.shape[-1]} != {cols}"
        )
    if not h.is_contiguous():
        # The kernel addresses rows as row * cols; a strided view would be read and
        # written at the wrong offsets.
        raise ValueError("scale_shift: activation must be contiguous")
    # The modulation tensors are addressed the same way. ComfyUI slices them out of a
    # wider [R, n*cols] adaLN projection, which is a strided view -- a few hundred KB
    # of .contiguous() per call keeps the kernel address math trivial. (A strided
    # scale/shift silently read the wrong row for every mod row but the first.)
    if not scale.is_contiguous():
        scale = scale.contiguous()
    if not shift.is_contiguous():
        shift = shift.contiguous()
    if not row_mod.is_contiguous():
        row_mod = row_mod.contiguous()
    # empty_like would inherit a non-contiguous memory format; the kernel writes
    # row-major, so always allocate a dense buffer.
    out = torch.empty_like(h, memory_format=torch.contiguous_format)
    code = 1 if h.dtype == torch.float16 else 2 if h.dtype == torch.bfloat16 else 0
    grid = (rows, triton.cdiv(cols, block))
    _scale_shift_kernel[grid](h, scale, shift, row_mod, out, cols,
                              BLOCK=block, DTYPE_CODE=code, num_warps=4)
    return out


def scale_shift_inplace_(h: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor,
                         row_mod: torch.Tensor, block: int = 1024) -> torch.Tensor:
    """In-place variant. Same bits, but ~2x slower than the out-of-place kernel on GB10
    because read-modify-write of the same buffer caps at ~115 GB/s."""
    h.copy_(scale_shift(h, scale, shift, row_mod, block))
    return h
