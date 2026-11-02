# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for INT8 block-wise quantization."""

import pytest
import torch

import comfy_kitchen as ck
from comfy_kitchen.backends import cuda
from comfy_kitchen.tensor import TensorWiseINT8Layout

from .conftest import (
    assert_values_close,
    cuda_backend_available,
    get_capable_backends,
    rel_err,
    requires_cuda_backend,
)

COMFYUI_NVIDIA_16_SERIES = (
    "1660",
    "1650",
    "1630",
    "T500",
    "T550",
    "T600",
    "MX550",
    "MX450",
    "CMP 30HX",
    "T2000",
    "T1000",
    "T1200",
)


def test_cuda_int8_cublas_turing_n_alignment(monkeypatch):
    """CUDA cuBLAS fallback pads Turing skinny N to 32."""
    from comfy_kitchen.backends import cuda

    calls = []

    def fake_get_device_capability(device_index):
        calls.append(device_index)
        return (7, 5)

    cuda._turing_device_cache.clear()
    monkeypatch.setattr(torch.cuda, "get_device_capability", fake_get_device_capability)

    class FakeCudaTensor:
        is_cuda = True

        def get_device(self):
            return 0

    tensor = FakeCudaTensor()
    assert cuda._cublas_int8_n_alignment(tensor) == 32
    assert cuda._round_up(17, cuda._cublas_int8_n_alignment(tensor)) == 32
    assert calls == [0]


def test_nvidia_16_series_list_matches_comfyui():
    assert cuda._NVIDIA_16_SERIES == COMFYUI_NVIDIA_16_SERIES


@pytest.mark.parametrize("device_name", COMFYUI_NVIDIA_16_SERIES)
def test_turing_devices_without_tensor_cores_keep_fallback(device_name, monkeypatch):
    cuda._turing_device_cache.clear()
    cuda._nvidia_16_series_device_cache.clear()
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device_index: (7, 5))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _device_index: f"NVIDIA {device_name}")

    assert cuda._cuda_device_is_turing(0)
    assert cuda._cuda_device_is_nvidia_16_series(0)
    assert not cuda._cuda_device_should_use_turing_kernels(0)


@pytest.mark.parametrize(
    "device_name",
    [
        "NVIDIA GeForce RTX 2060",
        "NVIDIA GeForce RTX 2080 Ti",
        "NVIDIA Quadro RTX 5000",
        "NVIDIA TITAN RTX",
        "Tesla T4",
    ],
)
def test_turing_tensor_core_devices_use_native_kernels(device_name, monkeypatch):
    cuda._turing_device_cache.clear()
    cuda._nvidia_16_series_device_cache.clear()
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device_index: (7, 5))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _device_index: device_name)

    assert not cuda._cuda_device_is_nvidia_16_series(0)
    assert cuda._cuda_device_should_use_turing_kernels(0)


@pytest.mark.parametrize(
    "capability",
    [(8, 0), (8, 6), (8, 9), (9, 0), (10, 0), (12, 0)],
)
def test_non_turing_devices_never_use_turing_kernels(capability, monkeypatch):
    cuda._turing_device_cache.clear()
    cuda._nvidia_16_series_device_cache.clear()
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device_index: capability)

    assert not cuda._cuda_device_is_turing(0)
    assert not cuda._cuda_device_should_use_turing_kernels(0)


def test_eager_int8_matmul_turing_n_alignment(monkeypatch):
    """Eager torch INT8 matmul pads Turing skinny N to 32."""
    from comfy_kitchen.backends.eager import quantization

    calls = []

    def fake_get_device_capability(device_index):
        calls.append(device_index)
        return (7, 5)

    quantization._turing_device_cache.clear()
    monkeypatch.setattr(torch.cuda, "get_device_capability", fake_get_device_capability)

    class FakeCudaTensor:
        is_cuda = True

        def get_device(self):
            return 0

    tensor = FakeCudaTensor()
    assert quantization._int8_mm_n_alignment(tensor) == 32
    assert quantization._round_up(17, quantization._int8_mm_n_alignment(tensor)) == 32
    assert calls == [0]


def test_cuda_int8_linear_does_not_retain_scratch_tensors():
    """CUDA INT8 linear uses per-call temporaries instead of retained scratch caches."""
    if not cuda_backend_available():
        pytest.skip("compiled CUDA backend required")

    x = torch.randn(16, 128, device="cuda", dtype=torch.bfloat16)
    weight = torch.randint(-128, 127, (64, 128), device="cuda", dtype=torch.int8)
    weight_scale = torch.ones((64, 1), device="cuda", dtype=torch.float32)

    out = cuda.int8_linear(x, weight, weight_scale, out_dtype=torch.bfloat16)

    assert out.shape == (16, 64)
    assert not hasattr(cuda, "_int8_quant_scratch")
    assert not hasattr(cuda, "_int8_gemm_int32_scratch")
    assert not hasattr(cuda, "_int8_quant_scratch_tensors")
    assert not hasattr(cuda, "_int8_gemm_int32_scratch_tensor")


def test_turing_fused_int8_shape_selection():
    assert cuda._prefer_turing_fused_int8(128, 4096, 4096)
    assert cuda._prefer_turing_fused_int8(512, 2048, 1024)
    assert cuda._prefer_turing_fused_int8(1024, 4096, 2048)
    assert not cuda._prefer_turing_fused_int8(512, 4096, 4096)
    assert not cuda._prefer_turing_fused_int8(1024, 2048, 4096)


@pytest.mark.parametrize("m", [64, 128, 512, 1024])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("with_bias", [False, True])
@pytest.mark.parametrize("scalar_weight_scale", [False, True])
def test_turing_int8_fused_gemm_matches_reference(
    seed,
    m,
    dtype,
    with_bias,
    scalar_weight_scale,
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if not hasattr(cuda._C, "cutlass_turing_int8_dequant"):
        pytest.skip("CUDA extension was built without the Turing INT8 kernel")

    n, k = 128, 256
    activation = torch.randint(-127, 128, (m, k), device="cuda", dtype=torch.int8)
    weight = torch.randint(-127, 128, (n, k), device="cuda", dtype=torch.int8)
    activation_scale = torch.rand(m, device="cuda", dtype=torch.float32)
    weight_scale_shape = () if scalar_weight_scale else (n,)
    weight_scale = torch.rand(weight_scale_shape, device="cuda", dtype=torch.float32)
    bias = torch.randn(n, device="cuda", dtype=dtype) if with_bias else None

    actual = cuda._int8_linear_turing_quantized(
        activation,
        weight,
        activation_scale,
        weight_scale,
        bias,
        dtype,
    )
    accumulator = activation.cpu().to(torch.int32) @ weight.cpu().to(torch.int32).T
    expected = accumulator.to(device="cuda", dtype=torch.float32)
    expected *= activation_scale.reshape(-1, 1)
    expected *= weight_scale.reshape(1, -1)
    if bias is not None:
        expected += bias.float()
    expected = expected.to(dtype)

    assert actual is not None
    tolerance = 2 * torch.finfo(dtype).eps
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)


def test_int8_linear_routes_turing_to_fused_kernel(seed, monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if not hasattr(cuda._C, "cutlass_turing_int8_dequant"):
        pytest.skip("CUDA extension was built without the Turing INT8 kernel")

    x = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
    weight = torch.randint(-127, 128, (128, 256), device="cuda", dtype=torch.int8)
    weight_scale = torch.rand(128, device="cuda", dtype=torch.float32)
    original = cuda._int8_linear_turing_quantized
    calls = []

    def record_call(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(cuda, "_cuda_device_is_turing", lambda _device_index: True)
    monkeypatch.setattr(cuda, "_int8_linear_turing_quantized", record_call)
    output = cuda.int8_linear(x, weight, weight_scale, out_dtype=torch.bfloat16)

    assert output.shape == (128, 128)
    assert calls == [True]


# =============================================================================
# INT8 Quantization Tests
# =============================================================================


def test_eager_int8_stochastic_rounding_tensorwise(seed):
    """Eager INT8 stochastic rounding is seeded and unbiased for half values."""
    weight = torch.full((4096,), 0.5, dtype=torch.float32)
    scale = torch.tensor(1.0, dtype=torch.float32)

    with ck.registry.use_backend("eager"):
        q1, params = TensorWiseINT8Layout.quantize(weight, scale=scale, stochastic_rounding=123)
        q2, _ = TensorWiseINT8Layout.quantize(weight, scale=scale, stochastic_rounding=123)
        q3, _ = TensorWiseINT8Layout.quantize(weight, scale=scale, stochastic_rounding=124)

    assert q1.dtype == torch.int8
    assert params.scale.item() == 1.0
    assert torch.equal(q1, q2)
    assert not torch.equal(q1, q3)
    assert 0.45 < q1.float().mean().item() < 0.55


def test_cuda_int8_stochastic_rounding_seeded(seed):
    """CUDA stochastic INT8 rounding is seeded and unbiased."""
    if not cuda_backend_available():
        pytest.skip("compiled CUDA backend required")

    x = torch.full((4096,), 0.5, device="cuda", dtype=torch.float16)
    x[0] = 127.0

    with ck.registry.use_backend("cuda"):
        q1, scale1 = ck.quantize_int8_rowwise(x, stochastic_rounding=123)
        q2, scale2 = ck.quantize_int8_rowwise(x, stochastic_rounding=123)
        q3, scale3 = ck.quantize_int8_rowwise(x, stochastic_rounding=124)

    assert torch.equal(q1, q2)
    assert not torch.equal(q1, q3)
    assert torch.equal(scale1, scale2)
    assert torch.equal(scale1, scale3)
    assert scale1.item() == 1.0
    assert 0.45 < q1[1:].float().mean().item() < 0.55


def test_eager_int8_stochastic_rounding_convrot(seed):
    """ConvRot INT8 quantization supports seeded stochastic rounding."""
    torch.manual_seed(1234)
    weight = torch.randn(4, 256, dtype=torch.float32)

    with ck.registry.use_backend("eager"):
        q1, params = TensorWiseINT8Layout.quantize(
            weight,
            per_channel=True,
            convrot=True,
            convrot_groupsize=256,
            stochastic_rounding=123,
        )
        q2, _ = TensorWiseINT8Layout.quantize(
            weight,
            per_channel=True,
            convrot=True,
            convrot_groupsize=256,
            stochastic_rounding=123,
        )
        q3, _ = TensorWiseINT8Layout.quantize(
            weight,
            per_channel=True,
            convrot=True,
            convrot_groupsize=256,
            stochastic_rounding=124,
        )

    assert q1.dtype == torch.int8
    assert params.convrot
    assert params.scale.shape == (4, 1)
    assert torch.equal(q1, q2)
    assert not torch.equal(q1, q3)


class TestTensorWiseINT8Layout:
    """Tests for TensorWiseINT8Layout quantized tensor format."""

    @pytest.fixture(autouse=True)
    def cuda_only(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA required for TensorWiseINT8Layout tests")

    def test_weight_quantize_shape_dtype(self, seed):
        """Weight path: output INT8, scalar scale, shape preserved."""
        from comfy_kitchen.tensor import QuantizedTensor

        w = torch.randn(256, 512, device="cuda", dtype=torch.bfloat16)
        qt = QuantizedTensor.from_float(w, "TensorWiseINT8Layout")

        assert qt._qdata.dtype == torch.int8
        assert qt._qdata.shape == w.shape
        assert qt._params.scale.numel() == 1
        assert qt._params.scale.dtype == torch.float32

    def test_activation_quantize_shape_dtype(self, seed):
        """Activation path (is_weight=False): per-row scales [..., 1]."""
        from comfy_kitchen.tensor import TensorWiseINT8Layout

        x = torch.randn(32, 128, device="cuda", dtype=torch.float16)
        qdata, params = TensorWiseINT8Layout.quantize(x, is_weight=False)

        assert qdata.dtype == torch.int8
        assert qdata.shape == x.shape
        assert params.scale.shape == (32, 1)

    def test_weight_dequantize_dtype(self, seed):
        """Dequantize restores original dtype."""
        from comfy_kitchen.tensor import QuantizedTensor

        for dtype in (torch.float16, torch.bfloat16):
            w = torch.randn(64, 128, device="cuda", dtype=dtype)
            qt = QuantizedTensor.from_float(w, "TensorWiseINT8Layout")
            dq = qt.dequantize()
            assert dq.dtype == dtype
            assert dq.shape == w.shape

    @pytest.mark.parametrize("group_size", [64, 256])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    def test_convrot_dequant_kernel_matches_float32_reference(self, seed, group_size, dtype):
        """The fused ConvRot dequant rotates in float32 and rounds to the output dtype once, like the
        float32 rotate-then-cast reference, including rows whose scale is near zero."""
        if not cuda_backend_available():
            pytest.skip("compiled CUDA backend required")
        from comfy_kitchen.tensor.int8_utils import _build_hadamard, _rotate_weight

        rows, k = 257, 3 * group_size
        q = torch.randint(-127, 128, (rows, k), device="cuda", dtype=torch.int8)
        scale = torch.rand(rows, 1, device="cuda") * 1e-2
        scale[0] = 1e-9
        assert cuda._should_use_convrot_dequant_kernel(q, k, group_size)
        out = cuda.dequantize_int8_convrot_weight_dtype(q, scale, group_size, cuda.DTYPE_TO_CODE[dtype])
        h = _build_hadamard(group_size, device="cuda", dtype=torch.float32)
        ref = _rotate_weight(q.float() * scale, h, group_size)
        assert out.dtype == dtype
        # a single rounding of a value within float32 summation-order tolerance of the reference
        tol = 8 * torch.finfo(torch.float32).eps * ref.abs().amax(dim=1, keepdim=True)
        lo, hi = (ref - tol).to(dtype).float(), (ref + tol).to(dtype).float()
        assert ((out.float() >= lo) & (out.float() <= hi)).all()

    def test_weight_roundtrip_error(self, seed):
        """Roundtrip error stays within INT8 quantization tolerance."""
        from comfy_kitchen.tensor import QuantizedTensor

        w = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
        qt = QuantizedTensor.from_float(w, "TensorWiseINT8Layout")
        dq = qt.dequantize()

        rel_err = (w.float() - dq.float()).abs() / (w.float().abs().max() + 1e-8)
        assert rel_err.mean().item() < 0.02, f"Mean relative error too high: {rel_err.mean():.4f}"

    def test_state_dict_tensors_keys(self, seed):
        """state_dict_tensors returns '' and '_scale' keys."""
        from comfy_kitchen.tensor import QuantizedTensor, TensorWiseINT8Layout

        w = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
        qt = QuantizedTensor.from_float(w, "TensorWiseINT8Layout")
        sd = TensorWiseINT8Layout.state_dict_tensors(qt._qdata, qt._params)

        assert set(sd.keys()) == {"", "_scale"}
        assert sd[""].dtype == torch.int8
        assert sd["_scale"].numel() == 1

    @pytest.mark.parametrize(("n", "k"), [(128, 1024), (512, 1024), (11520, 3840)])
    def test_linear_never_relayouts_weight_storage(self, seed, n, k):
        """Dynamic VRAM owns the weight buffer: linear must read it in place."""
        import comfy_kitchen as ck
        from comfy_kitchen.tensor import QuantizedTensor, TensorWiseINT8Layout

        x = torch.randn(512, k, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
        qt = QuantizedTensor.from_float(
            w, "TensorWiseINT8Layout", per_channel=True
        )
        row_major, scale = TensorWiseINT8Layout.get_plain_tensors(qt)
        ptr = qt._qdata.data_ptr()
        snapshot = qt._qdata.clone()

        with torch.inference_mode():
            reference = ck.int8_linear(
                x, row_major.clone(), scale, out_dtype=x.dtype
            )
            candidate = torch.nn.functional.linear(x, qt)

        assert torch.equal(candidate, reference)
        assert qt._qdata.data_ptr() == ptr
        assert torch.equal(qt._qdata, snapshot)

    def test_supports_fast_matmul(self):
        """supports_fast_matmul returns True on CUDA SM >= 7.5."""
        from comfy_kitchen.tensor import TensorWiseINT8Layout

        result = TensorWiseINT8Layout.supports_fast_matmul()
        assert isinstance(result, bool)
        sm = torch.cuda.get_device_capability()
        if sm >= (7, 5):
            assert result is True

    def test_linear_dispatch(self, seed):
        """aten.linear dispatch fires and produces correct shape/dtype."""
        from comfy_kitchen.tensor import QuantizedTensor

        x = torch.randn(4, 128, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
        qt_w = QuantizedTensor.from_float(w, "TensorWiseINT8Layout")

        out = torch.nn.functional.linear(x, qt_w)

        assert out.shape == (4, 64)
        assert out.dtype == torch.bfloat16

    def test_linear_dispatch_uses_activation_dtype(self, seed):
        """aten.linear defaults output dtype to runtime activation dtype."""
        from comfy_kitchen.tensor import QuantizedTensor

        x = torch.randn(4, 128, device="cuda", dtype=torch.float16)
        w = torch.randn(64, 128, device="cuda", dtype=torch.float32)
        bias = torch.randn(64, device="cuda", dtype=torch.float32)
        qt_w = QuantizedTensor.from_float(w, "TensorWiseINT8Layout")

        out = torch.nn.functional.linear(x, qt_w, bias)

        assert out.shape == (4, 64)
        assert out.dtype == torch.float16

    def test_mm_dispatch(self, seed):
        """aten.mm dispatch fires and produces correct shape."""
        from comfy_kitchen.tensor import QuantizedTensor

        # mm: A [M,K] @ B [K,N] — store B as [K,N] so quantize/dequantize preserves shape
        a = torch.randn(8, 128, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(128, 64, device="cuda", dtype=torch.bfloat16)
        qt_b = QuantizedTensor.from_float(b, "TensorWiseINT8Layout")

        out = torch.mm(a, qt_b)
        assert out.shape == (8, 64)

    def test_mm_dispatch_uses_activation_dtype(self, seed):
        """aten.mm defaults output dtype to runtime activation dtype."""
        from comfy_kitchen.tensor import QuantizedTensor

        a = torch.randn(8, 128, device="cuda", dtype=torch.float16)
        b = torch.randn(128, 64, device="cuda", dtype=torch.float32)
        qt_b = QuantizedTensor.from_float(b, "TensorWiseINT8Layout")

        out = torch.mm(a, qt_b)

        assert out.shape == (8, 64)
        assert out.dtype == torch.float16

    def test_addmm_dispatch(self, seed):
        """aten.addmm dispatch fires and produces correct shape/dtype."""
        from comfy_kitchen.tensor import QuantizedTensor

        x = torch.randn(4, 128, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
        bias = torch.randn(64, device="cuda", dtype=torch.bfloat16)
        qt_w = QuantizedTensor.from_float(w, "TensorWiseINT8Layout")

        out = torch.nn.functional.linear(x, qt_w, bias)

        assert out.shape == (4, 64)
        assert out.dtype == torch.bfloat16

    def test_addmm_dispatch_uses_activation_dtype(self, seed):
        """aten.addmm defaults output dtype to runtime activation dtype."""
        from comfy_kitchen.tensor import QuantizedTensor

        bias = torch.randn(64, device="cuda", dtype=torch.float32)
        x = torch.randn(4, 128, device="cuda", dtype=torch.float16)
        w = torch.randn(128, 64, device="cuda", dtype=torch.float32)
        qt_w = QuantizedTensor.from_float(w, "TensorWiseINT8Layout")

        out = torch.addmm(bias, x, qt_w)

        assert out.shape == (4, 64)
        assert out.dtype == torch.float16

    @pytest.mark.parametrize("backend", get_capable_backends("int8_linear", "cuda"))
    def test_int8_linear_correctness(self, seed, backend):
        """Check int8_linear parity across all capable backends."""
        import comfy_kitchen as ck
        from comfy_kitchen.backends.eager.quantization import quantize_int8_tensorwise

        x = torch.randn(128, 256, device="cuda", dtype=torch.float16)
        w = torch.randn(64, 256, device="cuda", dtype=torch.float16)
        bias = torch.randn(64, device="cuda", dtype=torch.float16)

        w_int8, w_scale = quantize_int8_tensorwise(w)

        with ck.registry.use_backend("eager"):
            ref_out = ck.int8_linear(x, w_int8, w_scale, bias=bias, out_dtype=torch.float16)

        with ck.registry.use_backend(backend):
            out = ck.int8_linear(x, w_int8, w_scale, bias=bias, out_dtype=torch.float16)

        # cuBLAS INT8 GEMM output compared to eager may have slight differences due to rounding
        # However, eager vs triton vs cuda should be very close.
        assert_values_close(out, ref_out, rtol=1e-2, atol=1e-2, name=f"int8_linear_{backend}", max_mismatch_ratio=0.01)

    def test_int8_linear_cuda_single_row_gemv(self, seed):
        """CUDA int8_linear uses the single-row GEMV path correctly."""
        if not cuda_backend_available():
            pytest.skip("compiled CUDA backend required")
        import comfy_kitchen as ck
        from comfy_kitchen.backends.eager.quantization import quantize_int8_tensorwise

        x = torch.randn(1, 512, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(384, 512, device="cuda", dtype=torch.bfloat16)
        bias = torch.randn(384, device="cuda", dtype=torch.bfloat16)
        w_int8, w_scale = quantize_int8_tensorwise(w)

        with ck.registry.use_backend("eager"):
            ref_out = ck.int8_linear(x, w_int8, w_scale, bias=bias, out_dtype=torch.bfloat16)

        with ck.registry.use_backend("cuda"):
            out = ck.int8_linear(x, w_int8, w_scale, bias=bias, out_dtype=torch.bfloat16)

        assert out.shape == (1, 384)
        assert out.dtype == torch.bfloat16
        assert_values_close(out, ref_out, rtol=1e-2, atol=1e-2, name="int8_linear_cuda_single_row_gemv", max_mismatch_ratio=0.01)

    def test_public_api_quantize_tensorwise(self, seed):
        """comfy_kitchen.quantize_int8_tensorwise op is reachable."""
        import comfy_kitchen as ck

        x = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
        q, scale = ck.quantize_int8_tensorwise(x)

        assert q.dtype == torch.int8
        assert q.shape == x.shape
        assert scale.numel() == 1

    def test_public_api_quantize_rowwise(self, seed):
        """comfy_kitchen.quantize_int8_rowwise op is reachable."""
        import comfy_kitchen as ck

        x = torch.randn(32, 128, device="cuda", dtype=torch.bfloat16)
        q, scale = ck.quantize_int8_rowwise(x)

        assert q.dtype == torch.int8
        assert q.shape == x.shape
        assert scale.shape == (32, 1)

    def test_public_api_dequantize_simple(self, seed):
        """comfy_kitchen.dequantize_int8_simple op is reachable."""
        import comfy_kitchen as ck

        x = torch.randn(32, 64, device="cuda", dtype=torch.bfloat16)
        q, scale = ck.quantize_int8_tensorwise(x)
        dq = ck.dequantize_int8_simple(q, scale)

        assert dq.dtype == torch.float32
        assert dq.shape == x.shape

    @pytest.mark.parametrize("backend", get_capable_backends("dequantize_int8_simple", "cuda"))
    def test_dequantize_simple_backend_correctness(self, seed, backend):
        """CUDA INT8 dequantize matches eager for scalar and rowwise scales."""
        import comfy_kitchen as ck

        x = torch.randn(32, 64, device="cuda", dtype=torch.bfloat16)
        q_scalar, scale_scalar = ck.quantize_int8_tensorwise(x)
        q_row, scale_row = ck.quantize_int8_rowwise(x)

        with ck.registry.use_backend("eager"):
            ref_scalar = ck.dequantize_int8_simple(q_scalar, scale_scalar)
            ref_row = ck.dequantize_int8_simple(q_row, scale_row)

        with ck.registry.use_backend(backend):
            out_scalar = ck.dequantize_int8_simple(q_scalar, scale_scalar)
            out_row = ck.dequantize_int8_simple(q_row, scale_row)

        assert_values_close(out_scalar, ref_scalar, rtol=0, atol=0, name=f"dequant_scalar_{backend}")
        assert_values_close(out_row, ref_row, rtol=0, atol=0, name=f"dequant_rowwise_{backend}")

    def test_dequantize_direct_output_dtype_matches_final_cast(self, seed):
        """Direct fp16/bf16 dequant output matches the prior float32-then-cast behavior."""
        if not cuda_backend_available():
            pytest.skip("compiled CUDA backend required")
        import comfy_kitchen as ck

        x = torch.randn(64, 256, device="cuda", dtype=torch.bfloat16)

        with ck.registry.use_backend("cuda"):
            q_row, scale_row = ck.quantize_int8_rowwise(x)
            ref_row = torch.ops.comfy_kitchen.dequantize_int8_simple(q_row, scale_row)
            q_conv, scale_conv = torch.ops.comfy_kitchen.quantize_int8_convrot_weight(x, 256)
            ref_conv = torch.ops.comfy_kitchen.dequantize_int8_convrot_weight(q_conv, scale_conv, 256)

            for dtype, code in ((torch.float16, 1), (torch.bfloat16, 2)):
                out_row = torch.ops.comfy_kitchen.dequantize_int8_simple_dtype(q_row, scale_row, code)
                out_conv = torch.ops.comfy_kitchen.dequantize_int8_convrot_weight_dtype(q_conv, scale_conv, 256, code)

                assert out_row.dtype == dtype
                assert out_conv.dtype == dtype
                assert torch.equal(out_row, ref_row.to(dtype))
                assert torch.equal(out_conv, ref_conv.to(dtype))

    def test_public_api_int8_linear(self, seed):
        """comfy_kitchen.int8_linear op is reachable."""
        import comfy_kitchen as ck
        from comfy_kitchen.backends.eager.quantization import quantize_int8_tensorwise

        x = torch.randn(4, 128, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
        w_int8, w_scale = quantize_int8_tensorwise(w)

        out = ck.int8_linear(x, w_int8, w_scale)

        assert out.shape == (4, 64)
        assert out.dtype == torch.bfloat16

    def test_convrot_hadamard_properties(self):
        """Verify _build_hadamard constructs correct orthogonal, symmetric matrix."""
        from comfy_kitchen.tensor.int8_utils import _build_hadamard

        # Test valid sizes
        for size in [4, 16, 64, 256]:
            h = _build_hadamard(size, device="cuda", dtype=torch.float32)
            assert h.shape == (size, size)
            # Check symmetry: H^T = H
            assert torch.allclose(h, h.T, atol=1e-5)
            # Check orthogonality: H^T @ H = I
            identity = torch.eye(size, device="cuda", dtype=torch.float32)
            assert torch.allclose(torch.matmul(h.T, h), identity, atol=1e-4)

        # Test invalid sizes
        for size in [2, 8, 32, 128, 500]:
            with pytest.raises(ValueError, match="Regular Hadamard size must be a power of 4"):
                _build_hadamard(size, device="cuda")

    def test_convrot_param_validation(self):
        """Verify parameter combinations for convrot raise expected ValueErrors."""
        from comfy_kitchen.tensor import TensorWiseINT8Layout

        w = torch.randn(64, 256, device="cuda", dtype=torch.float16)

        # 1. convrot with is_weight=False -> ValueError
        with pytest.raises(ValueError, match="convrot is only supported when is_weight is True"):
            TensorWiseINT8Layout.quantize(w, is_weight=False, convrot=True)

        # 2. convrot with per_channel=False -> ValueError
        with pytest.raises(ValueError, match="convrot is only supported when per_channel is True"):
            TensorWiseINT8Layout.quantize(w, is_weight=True, per_channel=False, convrot=True)

    def test_convrot_weight_roundtrip(self, seed):
        """Verify weight roundtrip (quantize -> dequantize) with convrot=True preserves values."""
        from comfy_kitchen.tensor import QuantizedTensor

        w = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
        # Using default convrot_groupsize=256
        qt = QuantizedTensor.from_float(
            w, "TensorWiseINT8Layout", per_channel=True, convrot=True, convrot_groupsize=256
        )

        assert qt._params.convrot is True
        assert qt._params.convrot_groupsize == 256

        dq = qt.dequantize()
        assert dq.dtype == torch.bfloat16
        assert dq.shape == w.shape

        # Roundtrip error should stay within expected INT8 quantization limits
        rel_err = (w.float() - dq.float()).abs() / (w.float().abs().max() + 1e-8)
        assert rel_err.mean().item() < 0.02

    def test_convrot_weight_quantize_cuda_roundtrip(self, seed):
        """CUDA ConvRot weight quantization preserves the weight after inverse rotation."""
        if not cuda_backend_available():
            pytest.skip("compiled CUDA backend required")
        from comfy_kitchen.backends import cuda as cuda_backend
        from comfy_kitchen.tensor.int8_utils import _build_hadamard, _rotate_weight

        w = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
        h = _build_hadamard(256, device=w.device, dtype=w.dtype)

        with torch.cuda.device(w.device):
            q_cuda, scale_cuda = cuda_backend.quantize_int8_convrot_weight(w, 256)

        dq_rotated = q_cuda.float() * scale_cuda
        dq = _rotate_weight(dq_rotated, h.to(dtype=torch.float32), 256).to(w.dtype)

        rel_err = (w.float() - dq.float()).abs() / (w.float().abs().max() + 1e-8)
        assert rel_err.mean().item() < 0.02

    def test_convrot_weight_dequantize_torch_op_roundtrip(self, seed):
        """The ConvRot dequant torch op preserves weights within INT8 tolerance."""
        from comfy_kitchen.tensor import QuantizedTensor

        w = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
        qt = QuantizedTensor.from_float(
            w, "TensorWiseINT8Layout", per_channel=True, convrot=True, convrot_groupsize=256
        )

        dq = torch.ops.comfy_kitchen.dequantize_int8_convrot_weight(
            qt._qdata, qt._params.scale, qt._params.convrot_groupsize
        ).to(w.dtype)

        rel_err = (w.float() - dq.float()).abs() / (w.float().abs().max() + 1e-8)
        assert rel_err.mean().item() < 0.02

    def test_convrot_int8_supports_65536_rows(self):
        """INT8 and INT4 kernels put large row counts in CUDA's grid.x dimension."""
        if not cuda_backend_available():
            pytest.skip("compiled CUDA backend required")
        rows, cols = 1 << 16, 256
        row = torch.linspace(-1.0, 1.0, cols, device="cuda", dtype=torch.bfloat16)
        weight = row.expand(rows, cols).contiguous()

        q_rowwise, scale_rowwise = cuda.quantize_int8_rowwise(weight)
        dequantized_rowwise = cuda.dequantize_int8_simple_dtype(
            q_rowwise, scale_rowwise, 2
        )
        assert torch.equal(dequantized_rowwise[0], dequantized_rowwise[-1])

        rotated = cuda.rotate_int8_convrot_weight(weight, 256)
        assert torch.equal(rotated[0], rotated[-1])

        q, scale = cuda.quantize_int8_convrot_staged(weight, 256)
        assert torch.equal(q[0], q[-1])
        assert torch.equal(scale[0], scale[-1])

        dequantized = cuda.dequantize_int8_convrot_weight_dtype(q, scale, 256, 2)
        assert torch.equal(dequantized[0], dequantized[-1])

        q_int4, scale_int4 = cuda.quantize_int4_rowwise_convrot64(weight, 256)
        dequantized_int4 = cuda.dequantize_convrot_w4a4_weight(
            q_int4,
            scale_int4.reshape(-1),
            output_dtype=torch.bfloat16,
        )
        assert torch.equal(dequantized_int4[0], dequantized_int4[-1])

    def test_convrot_divisibility(self, seed):
        """Verify error when channels are not divisible by convrot_groupsize."""
        from comfy_kitchen.tensor import QuantizedTensor

        # in_features (250) not divisible by 256
        w = torch.randn(64, 250, device="cuda", dtype=torch.bfloat16)
        with pytest.raises(ValueError, match="not divisible by group_size"):
            QuantizedTensor.from_float(
                w, "TensorWiseINT8Layout", per_channel=True, convrot=True, convrot_groupsize=256
            )

    def test_convrot_linear_mm_addmm_dispatch(self, seed):
        """Verify linear, mm, and addmm dispatch with convrot=True works and is highly accurate."""
        from comfy_kitchen.tensor import QuantizedTensor

        # Shapes must be compatible with group_size 64 (which is a valid power of 4)
        group_size = 64
        x = torch.randn(4, 128, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
        bias = torch.randn(64, device="cuda", dtype=torch.bfloat16)

        # Baseline: normal INT8 Quantization (no rotation)
        qt_w_normal = QuantizedTensor.from_float(
            w, "TensorWiseINT8Layout", per_channel=True, convrot=False
        )
        out_linear_normal = torch.nn.functional.linear(x, qt_w_normal, bias)

        # Active rotation version
        qt_w_rot = QuantizedTensor.from_float(
            w, "TensorWiseINT8Layout", per_channel=True, convrot=True, convrot_groupsize=group_size
        )
        out_linear_rot = torch.nn.functional.linear(x, qt_w_rot, bias)

        # Output shapes & dtypes should match perfectly
        assert out_linear_rot.shape == out_linear_normal.shape
        assert out_linear_rot.dtype == out_linear_normal.dtype

        # Result with and without ConvRot should be extremely close (it is mathematically equivalent under exact math)
        # allowing for expected tiny differences in quantization noise.
        rel_err_linear = (out_linear_rot.float() - out_linear_normal.float()).abs() / (out_linear_normal.float().abs().max() + 1e-8)
        assert rel_err_linear.mean().item() < 0.02

        # Test mm dispatch through the common linear decomposition shape:
        # a [M, K] @ weight.t() [K, N].
        x_mm = torch.randn(4, 128, device="cuda", dtype=torch.bfloat16)
        w_mm = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
        qt_w_mm_rot = QuantizedTensor.from_float(
            w_mm, "TensorWiseINT8Layout", per_channel=True, convrot=True, convrot_groupsize=group_size
        )
        out_mm_rot = torch.mm(x_mm, qt_w_mm_rot.t())
        assert out_mm_rot.shape == (4, 64)
        assert out_mm_rot.dtype == torch.bfloat16

        # Test addmm dispatch
        # addmm: bias + a [M, K] @ weight.t() [K, N]
        bias_mm = torch.randn(64, device="cuda", dtype=torch.bfloat16)
        out_addmm_rot = torch.addmm(bias_mm, x_mm, qt_w_mm_rot.t())
        assert out_addmm_rot.shape == (4, 64)
        assert out_addmm_rot.dtype == torch.bfloat16

    def test_convrot_triton_fused_correctness(self, seed):
        """Verify that fused Triton ConvRot+Quantization matches the eager baseline."""
        import comfy_kitchen as ck
        from comfy_kitchen.tensor import QuantizedTensor

        group_size = 64
        x = torch.randn(32, 128, device="cuda", dtype=torch.float16)
        w = torch.randn(64, 128, device="cuda", dtype=torch.float16)
        bias = torch.randn(64, device="cuda", dtype=torch.float16)

        # Quantize weight with convrot
        qt_w = QuantizedTensor.from_float(
            w, "TensorWiseINT8Layout", per_channel=True, convrot=True, convrot_groupsize=group_size
        )
        weight_qdata, weight_scale = qt_w._qdata, qt_w._params.scale

        # Run with Eager backend
        with ck.registry.use_backend("eager"):
            out_eager = ck.int8_linear(
                x, weight_qdata, weight_scale, bias=bias, out_dtype=torch.float16,
                convrot=True, convrot_groupsize=group_size
            )

        # Run with Triton backend
        with ck.registry.use_backend("triton"):
            out_triton = ck.int8_linear(
                x, weight_qdata, weight_scale, bias=bias, out_dtype=torch.float16,
                convrot=True, convrot_groupsize=group_size
            )

        # Triton and Eager outputs must be extremely close
        assert_values_close(out_triton, out_eager, rtol=1.0e-1, atol=1.0e-1, name="convrot_triton_vs_eager", max_mismatch_ratio=0.02)





class TestTensorWisePublicAPI:
    @pytest.fixture
    def seed(self):
        torch.manual_seed(42)

    def test_public_api_quantize_tensorwise(self, seed, device):
        """comfy_kitchen.quantize_int8_tensorwise op is reachable."""
        import torch

        import comfy_kitchen as ck

        x = torch.randn(64, 128, device=device, dtype=torch.bfloat16)
        q, scale = ck.quantize_int8_tensorwise(x)

        assert q.dtype == torch.int8
        assert q.shape == x.shape
        assert scale.numel() == 1

    def test_public_api_quantize_rowwise(self, seed, device):
        """comfy_kitchen.quantize_int8_rowwise op is reachable."""
        import torch

        import comfy_kitchen as ck

        x = torch.randn(32, 128, device=device, dtype=torch.bfloat16)
        q, scale = ck.quantize_int8_rowwise(x)

        assert q.dtype == torch.int8
        assert q.shape == x.shape
        assert scale.shape == (32, 1)

    def test_public_api_dequantize_simple(self, seed, device):
        """comfy_kitchen.dequantize_int8_simple op is reachable."""
        import torch

        import comfy_kitchen as ck

        x = torch.randn(32, 64, device=device, dtype=torch.bfloat16)
        q, scale = ck.quantize_int8_tensorwise(x)
        dq = ck.dequantize_int8_simple(q, scale)

        assert dq.dtype == torch.float32
        assert dq.shape == x.shape

    def test_public_api_int8_linear(self, seed, device):
        """comfy_kitchen.int8_linear op is reachable."""
        import torch

        import comfy_kitchen as ck
        from comfy_kitchen.backends.eager.quantization import quantize_int8_tensorwise

        x = torch.randn(4, 128, device=device, dtype=torch.bfloat16)
        w = torch.randn(64, 128, device=device, dtype=torch.bfloat16)
        w_int8, w_scale = quantize_int8_tensorwise(w)

        out = ck.int8_linear(x, w_int8, w_scale)

        assert out.shape == (4, 64)
        assert out.dtype == torch.bfloat16

    def test_eager_int8_linear_single_row(self, seed, device):
        """Eager int8_linear supports single-row batches."""
        import torch

        import comfy_kitchen as ck
        from comfy_kitchen.backends.eager.quantization import quantize_int8_tensorwise

        x = torch.randn(1, 128, device=device, dtype=torch.bfloat16)
        w = torch.randn(64, 128, device=device, dtype=torch.bfloat16)
        w_int8, w_scale = quantize_int8_tensorwise(w)

        with ck.registry.use_backend("eager"):
            out = ck.int8_linear(x, w_int8, w_scale)

        assert out.shape == (1, 64)
        assert out.dtype == torch.bfloat16

    def test_eager_int8_linear_pads_k_to_int8_mm_tile(self, seed, device):
        """Eager int8_linear pads K to int8 matmul's tile size."""
        import torch

        import comfy_kitchen as ck
        from comfy_kitchen.backends.eager.quantization import quantize_int8_tensorwise

        x = torch.randn(17, 12, device=device, dtype=torch.bfloat16)
        w = torch.randn(64, 12, device=device, dtype=torch.bfloat16)
        w_int8, w_scale = quantize_int8_tensorwise(w)

        with ck.registry.use_backend("eager"):
            out = ck.int8_linear(x, w_int8, w_scale)

        assert out.shape == (17, 64)
        assert out.dtype == torch.bfloat16

    def test_eager_int8_linear_pads_n_to_int8_mm_tile(self, seed, device):
        """Eager int8_linear pads N to int8 matmul's tile size."""
        import torch

        import comfy_kitchen as ck
        from comfy_kitchen.backends.eager.quantization import quantize_int8_tensorwise

        x = torch.randn(17, 16, device=device, dtype=torch.bfloat16)
        w = torch.randn(1, 16, device=device, dtype=torch.bfloat16)
        w_int8, w_scale = quantize_int8_tensorwise(w)

        with ck.registry.use_backend("eager"):
            out = ck.int8_linear(x, w_int8, w_scale)

        assert out.shape == (17, 1)
        assert out.dtype == torch.bfloat16


def test_fused_rms_modulated_declines_batched_modulation():
    """A per-sample (B > 1) modulation falls back to the unfused composition."""
    width = 256
    x = torch.randn(2, 4, width, dtype=torch.bfloat16)
    norm_weight = torch.ones(width, dtype=torch.bfloat16)
    modulation_scale = torch.zeros(2, width, dtype=torch.bfloat16)

    # The shape checks run before any weight is inspected.
    assert (
        TensorWiseINT8Layout.fused_rms_modulated(
            x, None, None, norm_weight, 1.0e-6, modulation_scale
        )
        is NotImplemented
    )


def test_fused_rms_modulated_declines_without_native_backend():
    """Without a native kernel the fused method leaves the unfused path to the caller."""
    from comfy_kitchen.tensor import QuantizedTensor

    torch.manual_seed(0)
    width = 256
    x = torch.randn(1, 4, width, dtype=torch.bfloat16)
    norm_weight = torch.rand(width, dtype=torch.bfloat16) + 0.5
    modulation_scale = torch.randn(1, width, dtype=torch.bfloat16) * 0.25

    def weight(n, k):
        return QuantizedTensor.from_float(
            torch.randn(n, k, dtype=torch.bfloat16), "TensorWiseINT8Layout",
            per_channel=True, convrot=True, convrot_groupsize=256,
        )

    qkv = weight(3 * width, width)

    # CPU tensors only match the eager backend.
    assert TensorWiseINT8Layout.fused_rms_modulated(
        x, qkv, None, norm_weight, 1.0e-6, modulation_scale
    ) is NotImplemented

    # An explicit eager override still runs the reference composition.
    with ck.use_backend("eager"):
        out = TensorWiseINT8Layout.fused_rms_modulated(
            x, qkv, None, norm_weight, 1.0e-6, modulation_scale
        )
    assert out is not NotImplemented
    assert out.shape == (1, 4, 3 * width)
class TestBandedStreamK:
    """The banded stream-K tile order must give bit-identical output to the identity order."""

    @requires_cuda_backend
    def test_captured_workspace_survives_larger_gemms(self, seed):
        """Later GEMMs and captures must not replace a live graph's scratch storage."""
        if torch.cuda.get_device_capability()[0] < 8:
            pytest.skip("CUTLASS stream-K requires SM80 or newer")

        from comfy_kitchen.backends import cuda as cuda_backend

        wrap = cuda_backend._wrap_for_dlpack
        stream = torch.cuda.Stream()
        cases = []
        for k, config in [(128, 12), (1024, 12), (4096, 13), (8192, 13)]:
            a = torch.randint(-8, 8, (128, k), dtype=torch.int8, device="cuda")
            b = torch.randint(-8, 8, (256, k), dtype=torch.int8, device="cuda")
            xs = torch.full((128, 1), 0.125, device="cuda")
            ws = torch.full((256,), 0.0625, device="cuda")
            out = torch.empty((128, 256), dtype=torch.bfloat16, device="cuda")
            expected = ((a.float() @ b.float().t()) * xs * ws).bfloat16()
            cases.append((a, b, xs, ws, out, expected, config))
        stream.wait_stream(torch.cuda.current_stream())

        def run(case):
            a, b, xs, ws, out, _, config = case
            assert cuda_backend._C.cutlass_int8_dequant_config(
                wrap(a), wrap(b), wrap(xs), wrap(ws), wrap(out), 2, config, stream.cuda_stream
            )

        first, second = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
        with torch.cuda.stream(stream):
            # No workspace warmup: capture must handle the first allocation and growth.
            with torch.cuda.graph(first, stream=stream):
                run(cases[0])
                run(cases[1])
            with torch.cuda.graph(second, stream=stream):
                run(cases[2])
            run(cases[3])
            for _ in range(5):
                for case in cases[:3]:
                    case[4].fill_(float("nan"))
                second.replay()
                first.replay()
                stream.synchronize()
                for case in cases:
                    torch.testing.assert_close(case[4], case[5], rtol=0, atol=0)
            second.reset()
            run(cases[3])
            first.replay()
            stream.synchronize()
            for case in cases[:2]:
                torch.testing.assert_close(case[4], case[5], rtol=0, atol=0)

    @staticmethod
    def _run(m, n, k, config):
        from comfy_kitchen.backends import cuda as cuda_backend
        c, wrap = cuda_backend._C, cuda_backend._wrap_for_dlpack
        stream = torch.cuda.current_stream().cuda_stream
        a = torch.randint(-127, 128, (m, k), dtype=torch.int8, device="cuda")
        b = torch.randint(-127, 128, (n, k), dtype=torch.int8, device="cuda")
        xs = torch.rand(m, 1, device="cuda") * 0.01
        ws = torch.rand(n, device="cuda") * 0.01
        outs = {}
        for cfg in config:
            out = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
            if cfg == "auto":
                empty = cuda_backend._empty_cuda_tensor(a.device, torch.bfloat16)
                assert c.cutlass_int8_dequant(wrap(a), wrap(b), wrap(xs), wrap(ws), wrap(empty), wrap(out), 2, stream)
            else:
                assert c.cutlass_int8_dequant_config(wrap(a), wrap(b), wrap(xs), wrap(ws), wrap(out), 2, cfg, stream)
            outs[cfg] = out
        ref = (a.float() @ b.float().t()) * xs * ws
        return outs, ref

    # 37 / 74 N tiles: a short last band; M has enough tiles that both configs band
    @requires_cuda_backend
    @pytest.mark.parametrize("m,n,k", [(9600, 9408, 128), (19100, 9408, 128)])
    def test_bands_match_identity_order(self, m, n, k, seed):
        outs, ref = self._run(m, n, k, config=[0, 12, 13])
        assert torch.equal(outs[12], outs[0]) and torch.equal(outs[13], outs[0])
        assert rel_err(outs[0].float(), ref) < 1e-2

    @requires_cuda_backend
    def test_large_activation_reroutes_to_banded_stream_k(self):
        """Config 0 shapes move to banded stream-K past the L2-derived activation size."""
        from comfy_kitchen.backends import cuda as cuda_backend
        pick = cuda_backend._C.cutlass_int8_selected_config
        l2 = torch.cuda.get_device_properties(0).L2_cache_size
        threshold = max(l2 * 3 // 4, 48 << 20)
        n, k = 16384, 8192
        assert pick(2048, n, k) in (0, 1)                 # 16.8 MB: below any threshold
        m = 9216                                          # 75.5 MB
        assert pick(m, n, k) == (13 if m * k > threshold else 0)
        assert pick(4096, 24576, 32768) in (0, 1)         # fewer M tiles than N: no reroute


class TestTritonInt8GemmDequant:
    """Tests for GB10 (sm_121) Triton fused-dequant INT8 GEMM and dispatch."""

    @requires_cuda_backend
    @pytest.mark.parametrize(
        "m,n,k",
        [
            (256, 256, 256),   # EVEN (M%128==0, N%128==0, K%64==0)
            (160, 320, 256),   # EVEN_K (K%64==0, odd M/N)
            (150, 200, 176),   # general masked (odd M, N, K%16==0)
        ],
    )
    @pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16, torch.float32])
    @pytest.mark.parametrize("has_bias", [False, True])
    @pytest.mark.parametrize("per_channel_w", [True, False])
    def test_triton_int8_gemm_dequant_matches_cutlass(
        self, m, n, k, out_dtype, has_bias, per_channel_w, seed
    ):
        from comfy_kitchen.backends import cuda as cuda_backend
        from comfy_kitchen.backends.eager.quantization import DTYPE_TO_CODE
        from comfy_kitchen.backends.triton.quantization import triton_int8_gemm_dequant

        x = torch.randint(-127, 128, (m, k), dtype=torch.int8, device="cuda")
        w = torch.randint(-127, 128, (n, k), dtype=torch.int8, device="cuda")
        xs = torch.rand(m, 1, dtype=torch.float32, device="cuda") * 0.02
        ws = (
            torch.rand(n, dtype=torch.float32, device="cuda") * 0.02
            if per_channel_w
            else torch.rand(1, dtype=torch.float32, device="cuda") * 0.02
        )
        bias = (
            torch.randn(n, dtype=out_dtype, device="cuda")
            if has_bias
            else cuda_backend._empty_cuda_tensor(x.device, out_dtype)
        )

        out_triton = torch.empty((m, n), dtype=out_dtype, device="cuda")
        assert triton_int8_gemm_dequant(x, w, xs, ws, bias, out_triton)

        ws_cutlass = ws if ws.numel() == n else ws.expand(n).contiguous()
        out_cutlass = torch.empty((m, n), dtype=out_dtype, device="cuda")
        wrap = cuda_backend._wrap_for_dlpack
        stream = torch.cuda.current_stream().cuda_stream
        assert cuda_backend._C.cutlass_int8_dequant(
            wrap(x),
            wrap(w),
            wrap(xs),
            wrap(ws_cutlass),
            wrap(bias),
            wrap(out_cutlass),
            DTYPE_TO_CODE[out_dtype],
            stream,
        )
        assert torch.equal(out_triton, out_cutlass)

    def test_tile_config_selection(self):
        """Thin-M tiles are chosen only where GB10 measurements show a win."""
        pytest.importorskip("triton")
        from comfy_kitchen.backends.triton.quantization import _int8_tile_config

        gb10 = (12, 1)
        # weight fits in L2 (16 MB): thin tiles
        assert _int8_tile_config(24, 4096, 4096, gb10)[:2] == (64, 64)
        assert _int8_tile_config(64, 4096, 4096, gb10)[:2] == (64, 64)
        assert _int8_tile_config(127, 4096, 4096, gb10)[:2] == (64, 64)
        assert _int8_tile_config(128, 4096, 4096, gb10)[:2] == (64, 128)
        assert _int8_tile_config(207, 4096, 4096, gb10)[:2] == (64, 128)
        assert _int8_tile_config(208, 4096, 4096, gb10)[:2] == (128, 64)
        assert _int8_tile_config(256, 4096, 4096, gb10)[:2] == (128, 64)

        # large M, or a weight that no longer fits in L2: keep the default tile
        assert _int8_tile_config(288, 4096, 4096, gb10)[:2] == (128, 128)
        assert _int8_tile_config(4096, 4096, 4096, gb10)[:2] == (128, 128)
        assert _int8_tile_config(64, 5120, 5120, gb10)[:2] == (128, 128)
        assert _int8_tile_config(192, 10240, 4096, gb10)[:2] == (128, 128)

        # other architectures (this backend also serves ROCm/XPU) keep the generic tile
        for cap in [(9, 0), (8, 0), (-1, -1)]:
            assert _int8_tile_config(64, 4096, 4096, cap)[:2] == (128, 128)

        # BK/warps/stages follow the tile
        assert _int8_tile_config(64, 4096, 4096, gb10)[2:] == (64, 8, 4, 4)
        assert _int8_tile_config(192, 4096, 4096, gb10)[2:] == (64, 8, 4, 4)
        assert _int8_tile_config(256, 4096, 4096, gb10)[2:] == (64, 8, 8, 4)
        assert _int8_tile_config(4096, 4096, 4096, gb10)[2:] == (64, 8, 8, 4)

    @requires_cuda_backend
    @pytest.mark.parametrize("m", [24, 32, 63, 64, 96, 112, 127, 128, 160, 192, 207, 208, 256, 287, 288])
    @pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
    def test_thin_m_tiles_match_cutlass(self, m, out_dtype, seed):
        """Every tile band must stay bit-identical to CUTLASS (n=256,k=256 = thin band)."""
        from comfy_kitchen.backends import cuda as cuda_backend
        from comfy_kitchen.backends.eager.quantization import DTYPE_TO_CODE
        from comfy_kitchen.backends.triton.quantization import triton_int8_gemm_dequant

        n, k = 256, 256
        x = torch.randint(-127, 128, (m, k), dtype=torch.int8, device="cuda")
        w = torch.randint(-127, 128, (n, k), dtype=torch.int8, device="cuda")
        xs = torch.rand(m, 1, dtype=torch.float32, device="cuda") * 0.02
        ws = torch.rand(n, dtype=torch.float32, device="cuda") * 0.02
        bias = cuda_backend._empty_cuda_tensor(x.device, out_dtype)

        out_triton = torch.empty((m, n), dtype=out_dtype, device="cuda")
        assert triton_int8_gemm_dequant(x, w, xs, ws, bias, out_triton)

        out_cutlass = torch.empty((m, n), dtype=out_dtype, device="cuda")
        wrap = cuda_backend._wrap_for_dlpack
        stream = torch.cuda.current_stream().cuda_stream
        assert cuda_backend._C.cutlass_int8_dequant(
            wrap(x), wrap(w), wrap(xs), wrap(ws), wrap(bias), wrap(out_cutlass),
            DTYPE_TO_CODE[out_dtype], stream,
        )
        assert torch.equal(out_triton, out_cutlass)

    @requires_cuda_backend
    @pytest.mark.parametrize(
        "m,n,k",
        [
            (1100, 320, 176),   # npm=9 > G=8: exercises gid>=1 and a masked K tail
            (1664, 384, 320),   # npm=13: two groups, partial last group
            (384, 256, 256),    # EVEN on the 128x128 tile (M%128, N%128, K%64 all 0)
            (1024, 512, 512),   # EVEN on a thin-tile boundary
        ],
    )
    def test_swizzle_and_even_paths_match_cutlass(self, m, n, k, seed):
        """Grouped pid decomposition (gid>=1) and the fully unmasked path."""
        from comfy_kitchen.backends import cuda as cuda_backend
        from comfy_kitchen.backends.eager.quantization import DTYPE_TO_CODE
        from comfy_kitchen.backends.triton.quantization import triton_int8_gemm_dequant

        x = torch.randint(-127, 128, (m, k), dtype=torch.int8, device="cuda")
        w = torch.randint(-127, 128, (n, k), dtype=torch.int8, device="cuda")
        xs = torch.rand(m, 1, dtype=torch.float32, device="cuda") * 0.02
        ws = torch.rand(n, dtype=torch.float32, device="cuda") * 0.02
        bias = cuda_backend._empty_cuda_tensor(x.device, torch.bfloat16)
        got = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
        ref = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")

        assert triton_int8_gemm_dequant(x, w, xs, ws, bias, got)
        wrap = cuda_backend._wrap_for_dlpack
        assert cuda_backend._C.cutlass_int8_dequant(
            wrap(x), wrap(w), wrap(xs), wrap(ws), wrap(bias), wrap(ref),
            DTYPE_TO_CODE[torch.bfloat16], torch.cuda.current_stream().cuda_stream,
        )
        assert torch.equal(got, ref)

    @requires_cuda_backend
    @pytest.mark.parametrize("m", [64, 192, 320])
    def test_cuda_int8_linear_routes_match(self, m, monkeypatch, seed):
        """End to end through the CUDA backend (quantise + GEMM) on both routes."""
        from comfy_kitchen.backends import cuda as cuda_backend

        n, k = 512, 512
        x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
        w = torch.randint(-127, 128, (n, k), dtype=torch.int8, device="cuda")
        ws = torch.rand(n, dtype=torch.float32, device="cuda") * 0.02
        monkeypatch.setattr(cuda_backend, "_triton_int8_broken", False)

        monkeypatch.setenv("COMFY_KITCHEN_INT8_MODE", "cutlass")
        ref = cuda_backend.int8_linear(x, w, ws)
        monkeypatch.setenv("COMFY_KITCHEN_INT8_MODE", "triton")
        got = cuda_backend.int8_linear(x, w, ws)
        assert torch.equal(got, ref)

    @requires_cuda_backend
    @pytest.mark.parametrize("m,expect_triton", [(16, False), (24, False), (39, False), (40, True), (64, True), (127, True)])
    def test_dispatch_gate_opens_below_128(self, m, expect_triton, monkeypatch, seed):
        """M in [40,128) is routed to Triton; below 40 the CUTLASS path is faster.

        Measured end-to-end (int8_linear, 4096x4096, vanilla comfy-kitchen as the
        baseline): 0.78-0.82x at M=24-32, 1.06x at 40-48, 1.59x at 56, 2.17x at 64.
        """
        from comfy_kitchen.backends import cuda as cuda_backend
        from comfy_kitchen.backends.eager.quantization import DTYPE_TO_CODE
        from comfy_kitchen.backends.triton import quantization as triton_quant

        monkeypatch.setenv("COMFY_KITCHEN_INT8_MODE", "triton")
        monkeypatch.setattr(cuda_backend, "_triton_int8_broken", False)

        calls = []

        def spy(x8, w8, xs, ws, bias, out, **kwargs):
            calls.append((tuple(x8.shape), tuple(w8.shape), sorted(kwargs)))
            out.zero_()
            return True

        monkeypatch.setattr(triton_quant, "triton_int8_gemm_dequant", spy)

        n, k = 256, 256
        x = torch.randint(-127, 128, (m, k), dtype=torch.int8, device="cuda")
        w = torch.randint(-127, 128, (n, k), dtype=torch.int8, device="cuda")
        xs = torch.rand(m, 1, dtype=torch.float32, device="cuda") * 0.01
        ws = torch.rand(n, dtype=torch.float32, device="cuda") * 0.01
        bias = cuda_backend._empty_cuda_tensor(x.device, torch.bfloat16)
        out = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
        stream = torch.cuda.current_stream().cuda_stream

        assert cuda_backend._cutlass_int8_dequant(
            x, w, xs, ws, bias, out, DTYPE_TO_CODE[torch.bfloat16], stream
        )
        assert bool(calls) is expect_triton

    def test_python_header_check_is_memoized(self, monkeypatch):
        """The header guard used to cost ~52 us per launch; it must run once."""
        import os

        pytest.importorskip("triton")
        from comfy_kitchen.backends.triton import quantization as triton_quant

        monkeypatch.setenv("CPATH", "")  # restored by monkeypatch, not leaked to other tests
        real_exists = os.path.exists
        hits = []

        def counting_exists(path):
            hits.append(path)
            return real_exists(path)

        monkeypatch.setattr(triton_quant, "_PY_HEADERS_CHECKED", False)
        monkeypatch.setattr(os.path, "exists", counting_exists)
        triton_quant._ensure_python_headers()
        first = len(hits)
        triton_quant._ensure_python_headers()
        triton_quant._ensure_python_headers()
        assert len(hits) == first
        assert triton_quant._PY_HEADERS_CHECKED is True

    def test_sm121_dispatch_policy(self, monkeypatch):
        from comfy_kitchen.backends import cuda as cuda_backend

        monkeypatch.setattr(cuda_backend, "_INT8_MODE", "")
        monkeypatch.delenv("COMFY_KITCHEN_INT8_MODE", raising=False)

        monkeypatch.setattr(cuda_backend, "_cuda_device_capability", lambda idx: (12, 1))
        assert cuda_backend._should_use_triton_int8_dequant(0) is True

        monkeypatch.setattr(cuda_backend, "_cuda_device_capability", lambda idx: (12, 0))
        assert cuda_backend._should_use_triton_int8_dequant(0) is False

        monkeypatch.setattr(cuda_backend, "_cuda_device_capability", lambda idx: (9, 0))
        assert cuda_backend._should_use_triton_int8_dequant(0) is False

        monkeypatch.setenv("COMFY_KITCHEN_INT8_MODE", "triton")
        assert cuda_backend._should_use_triton_int8_dequant(0) is True

        monkeypatch.setenv("COMFY_KITCHEN_INT8_MODE", "off")
        monkeypatch.setattr(cuda_backend, "_cuda_device_capability", lambda idx: (12, 1))
        assert cuda_backend._should_use_triton_int8_dequant(0) is False

    @requires_cuda_backend
    def test_cutlass_int8_dequant_fail_open_fallback(self, monkeypatch, seed):
        from comfy_kitchen.backends import cuda as cuda_backend
        from comfy_kitchen.backends.eager.quantization import DTYPE_TO_CODE
        from comfy_kitchen.backends.triton import quantization as triton_quant

        monkeypatch.setenv("COMFY_KITCHEN_INT8_MODE", "triton")
        monkeypatch.setattr(cuda_backend, "_triton_int8_broken", False)

        def boom(*args, **kwargs):
            raise RuntimeError("simulated Triton failure")

        monkeypatch.setattr(triton_quant, "triton_int8_gemm_dequant", boom)

        m, n, k = 128, 128, 128
        x = torch.randint(-127, 128, (m, k), dtype=torch.int8, device="cuda")
        w = torch.randint(-127, 128, (n, k), dtype=torch.int8, device="cuda")
        xs = torch.rand(m, 1, dtype=torch.float32, device="cuda") * 0.01
        ws = torch.rand(n, dtype=torch.float32, device="cuda") * 0.01
        bias = cuda_backend._empty_cuda_tensor(x.device, torch.bfloat16)
        out = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
        stream = torch.cuda.current_stream().cuda_stream

        assert cuda_backend._cutlass_int8_dequant(
            x, w, xs, ws, bias, out, DTYPE_TO_CODE[torch.bfloat16], stream
        )
        assert cuda_backend._triton_int8_broken is True

    @requires_cuda_backend
    @pytest.mark.parametrize("m,n,k", [
        (24, 1, 256), (24, 7, 256), (24, 33, 256),   # N unaligned: CUTLASS is *wrong* here
        (24, 64, 16), (24, 64, 32), (24, 64, 48),    # K < BK
        (300, 7, 64), (320, 129, 96),
    ])
    def test_edge_shapes_match_exact_reference(self, m, n, k, seed):
        """Unaligned N / K < BK must match an exact int reference, not CUTLASS.

        N % 16 != 0 makes the upstream CUTLASS kernel emit garbage (measured: 5.3e6
        absolute error at N=7, NaN at N=33), so the reference here is the exact
        float32 matmul of int8 values (|sum| <= 127*127*K, exact in fp32 for small K)
        followed by the same epilogue order.
        """
        from comfy_kitchen.backends import cuda as cuda_backend
        from comfy_kitchen.backends.triton.quantization import triton_int8_gemm_dequant

        x = torch.randint(-128, 128, (m, k), dtype=torch.int8, device="cuda")
        w = torch.randint(-128, 128, (n, k), dtype=torch.int8, device="cuda")
        xs = torch.rand(m, 1, dtype=torch.float32, device="cuda") * 0.02
        ws = torch.rand(n, dtype=torch.float32, device="cuda") * 0.02
        bias = cuda_backend._empty_cuda_tensor(x.device, torch.bfloat16)

        got = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
        assert triton_int8_gemm_dequant(x, w, xs, ws, bias, got)

        acc = x.to(torch.float32) @ w.to(torch.float32).T
        ref = ((acc * xs) * ws).to(torch.bfloat16)
        assert torch.equal(got, ref)

    @requires_cuda_backend
    def test_non_contiguous_scales_never_reach_triton(self, monkeypatch, seed):
        """expand()/strided scales must fall back: the kernel reads stride-1 floats."""
        from comfy_kitchen.backends import cuda as cuda_backend
        from comfy_kitchen.backends.eager.quantization import DTYPE_TO_CODE

        monkeypatch.setenv("COMFY_KITCHEN_INT8_MODE", "triton")
        monkeypatch.setattr(cuda_backend, "_triton_int8_broken", False)

        m, n, k = 64, 128, 256
        x = torch.randint(-128, 128, (m, k), dtype=torch.int8, device="cuda")
        w = torch.randint(-128, 128, (n, k), dtype=torch.int8, device="cuda")
        xs = torch.rand(m, 1, dtype=torch.float32, device="cuda") * 0.02
        bias = cuda_backend._empty_cuda_tensor(x.device, torch.bfloat16)
        stream = torch.cuda.current_stream().cuda_stream

        expanded = torch.tensor([0.01], device="cuda").expand(n)   # numel == n, stride 0
        assert not expanded.is_contiguous()
        out_exp = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
        out_con = torch.empty_like(out_exp)
        assert cuda_backend._cutlass_int8_dequant(
            x, w, xs, expanded, bias, out_exp, DTYPE_TO_CODE[torch.bfloat16], stream
        )
        assert cuda_backend._cutlass_int8_dequant(
            x, w, xs, expanded.contiguous(), bias, out_con, DTYPE_TO_CODE[torch.bfloat16], stream
        )
        assert torch.equal(out_exp, out_con)

    @requires_cuda_backend
    def test_two_dim_weight_scale_falls_back(self, monkeypatch, seed):
        """A (1, 1) weight_scale must not crash the CUTLASS fallback (upstream did)."""
        from comfy_kitchen.backends import cuda as cuda_backend
        from comfy_kitchen.backends.eager.quantization import DTYPE_TO_CODE

        monkeypatch.setenv("COMFY_KITCHEN_INT8_MODE", "cutlass")
        m, n, k = 16, 64, 256          # m < _TRITON_INT8_MIN_M: pure CUTLASS path
        x = torch.randint(-128, 128, (m, k), dtype=torch.int8, device="cuda")
        w = torch.randint(-128, 128, (n, k), dtype=torch.int8, device="cuda")
        xs = torch.rand(m, 1, dtype=torch.float32, device="cuda") * 0.02
        bias = cuda_backend._empty_cuda_tensor(x.device, torch.bfloat16)
        out = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
        assert cuda_backend._cutlass_int8_dequant(
            x, w, xs, torch.tensor([[0.01]], device="cuda"), bias, out,
            DTYPE_TO_CODE[torch.bfloat16], torch.cuda.current_stream().cuda_stream,
        )
        assert torch.isfinite(out.float()).all()

    @requires_cuda_backend
    @pytest.mark.parametrize("m,n,k", [(256, 256, 256), (1024, 512, 256), (384, 320, 176), (150, 200, 176)])
    @pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16, torch.float32])
    @pytest.mark.parametrize("has_bias", [False, True])
    @pytest.mark.parametrize("scalar_ws,scalar_rs", [(False, False), (True, True)])
    def test_residual_epilogue_matches_cutlass(
        self, m, n, k, out_dtype, has_bias, scalar_ws, scalar_rs, seed
    ):
        """out = residual + rscale * round_out(acc*xs*ws + bias), bit for bit."""
        from comfy_kitchen.backends import cuda as cuda_backend
        from comfy_kitchen.backends.eager.quantization import DTYPE_TO_CODE
        from comfy_kitchen.backends.triton.quantization import triton_int8_gemm_dequant

        x = torch.randint(-127, 128, (m, k), dtype=torch.int8, device="cuda")
        w = torch.randint(-127, 128, (n, k), dtype=torch.int8, device="cuda")
        xs = torch.rand(m, 1, dtype=torch.float32, device="cuda") * 0.02
        ws = (
            torch.rand(1, dtype=torch.float32, device="cuda") * 0.02
            if scalar_ws
            else torch.rand(n, dtype=torch.float32, device="cuda") * 0.02
        )
        bias = (
            torch.randn(n, dtype=out_dtype, device="cuda")
            if has_bias
            else cuda_backend._empty_cuda_tensor(x.device, out_dtype)
        )
        resid = (torch.randn(m, n, device="cuda") * 0.5).to(out_dtype)
        rs_full = (torch.rand(n, device="cuda") * 0.2 + 0.9).to(out_dtype)
        rs = rs_full[:1] if scalar_rs else rs_full

        got = torch.empty((m, n), dtype=out_dtype, device="cuda")
        assert triton_int8_gemm_dequant(
            x, w, xs, ws, bias, got, residual=resid, residual_scale=rs
        )
        ref = torch.empty_like(got)
        assert cuda_backend._C.cutlass_int8_dequant_residual(
            cuda_backend._wrap_for_dlpack(x),
            cuda_backend._wrap_for_dlpack(w),
            cuda_backend._wrap_for_dlpack(xs),
            cuda_backend._wrap_for_dlpack(ws if ws.numel() == n else ws.expand(n).contiguous()),
            cuda_backend._wrap_for_dlpack(bias),
            cuda_backend._wrap_for_dlpack(rs if rs.numel() == n else rs.expand(n).contiguous()),
            cuda_backend._wrap_for_dlpack(resid),
            cuda_backend._wrap_for_dlpack(ref),
            DTYPE_TO_CODE[out_dtype],
            torch.cuda.current_stream().cuda_stream,
        )
        assert torch.equal(got, ref)

    @requires_cuda_backend
    @pytest.mark.parametrize("m", [64, 192, 512])
    def test_residual_routes_match_end_to_end(self, m, monkeypatch, seed):
        """ck.int8_linear(residual=...) is identical on the CUTLASS and Triton routes."""
        from comfy_kitchen.backends import cuda as cuda_backend

        n, k = 512, 512
        x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
        w = torch.randint(-127, 128, (n, k), dtype=torch.int8, device="cuda")
        ws = torch.rand(n, dtype=torch.float32, device="cuda") * 0.02
        resid = torch.randn(m, n, dtype=torch.bfloat16, device="cuda")
        rs = (torch.rand(n, device="cuda") * 0.3 + 0.8).to(torch.bfloat16)
        from comfy_kitchen.backends.triton import quantization as triton_quant

        monkeypatch.setattr(cuda_backend, "_triton_int8_broken", False)
        calls = []
        real = triton_quant.triton_int8_gemm_dequant

        def spy(*args, **kwargs):
            calls.append(kwargs.get("residual") is not None)
            return real(*args, **kwargs)

        monkeypatch.setattr(triton_quant, "triton_int8_gemm_dequant", spy)

        monkeypatch.setenv("COMFY_KITCHEN_INT8_MODE", "cutlass")
        ref = cuda_backend.int8_linear(x, w, ws, residual=resid, residual_scale=rs)
        monkeypatch.setenv("COMFY_KITCHEN_INT8_MODE", "triton")
        got = cuda_backend.int8_linear(x, w, ws, residual=resid, residual_scale=rs)
        assert torch.equal(got, ref)
        # otherwise both sides could have taken CUTLASS and the equality is trivial
        assert calls and all(calls), "the fused Triton residual path must actually run"

    @requires_cuda_backend
    def test_residual_negative_zero_matches_cutlass(self, seed):
        """CUTLASS's bias visitor adds +0.0, so a -0.0 product must not keep its sign."""
        from comfy_kitchen.backends import cuda as cuda_backend
        from comfy_kitchen.backends.eager.quantization import DTYPE_TO_CODE
        from comfy_kitchen.backends.triton.quantization import triton_int8_gemm_dequant

        m, n, k = 64, 128, 256
        x = torch.zeros((m, k), dtype=torch.int8, device="cuda")          # acc == 0
        w = torch.ones((n, k), dtype=torch.int8, device="cuda")
        xs = torch.full((m, 1), -0.5, dtype=torch.float32, device="cuda")  # 0 * -0.5 = -0.0
        ws = torch.ones(n, dtype=torch.float32, device="cuda")
        bias = cuda_backend._empty_cuda_tensor(x.device, torch.bfloat16)
        resid = torch.full((m, n), -0.0, dtype=torch.bfloat16, device="cuda")
        rs = torch.full((n,), 0.5, dtype=torch.bfloat16, device="cuda")

        got = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
        assert triton_int8_gemm_dequant(x, w, xs, ws, bias, got, residual=resid, residual_scale=rs)
        ref = torch.empty_like(got)
        assert cuda_backend._C.cutlass_int8_dequant_residual(
            cuda_backend._wrap_for_dlpack(x), cuda_backend._wrap_for_dlpack(w),
            cuda_backend._wrap_for_dlpack(xs), cuda_backend._wrap_for_dlpack(ws),
            cuda_backend._wrap_for_dlpack(bias), cuda_backend._wrap_for_dlpack(rs),
            cuda_backend._wrap_for_dlpack(resid), cuda_backend._wrap_for_dlpack(ref),
            DTYPE_TO_CODE[torch.bfloat16], torch.cuda.current_stream().cuda_stream,
        )
        assert torch.equal(got, ref)

    @requires_cuda_backend
    def test_cpu_residual_scale_does_not_poison_triton(self, monkeypatch, seed):
        """A CPU scale must be rejected by the gate, not raise inside Triton and latch."""
        from comfy_kitchen.backends import cuda as cuda_backend

        monkeypatch.setattr(cuda_backend, "_triton_int8_broken", False)
        monkeypatch.setenv("COMFY_KITCHEN_INT8_MODE", "triton")

        m, n, k = 128, 512, 512
        x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
        w = torch.randint(-127, 128, (n, k), dtype=torch.int8, device="cuda")
        ws = torch.rand(n, dtype=torch.float32, device="cuda") * 0.02
        resid = torch.randn(m, n, dtype=torch.bfloat16, device="cuda")
        rs_cpu = (torch.rand(n) * 0.3 + 0.8).to(torch.bfloat16)

        out = cuda_backend.int8_linear(x, w, ws, residual=resid, residual_scale=rs_cpu)
        assert cuda_backend._triton_int8_broken is False
        ref = cuda_backend.int8_linear(x, w, ws, residual=resid, residual_scale=rs_cpu.cuda())
        assert torch.equal(out, ref)
