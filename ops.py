import pathlib
from typing import Any

import torch
from torch.autograd import Function

_CSRC_DIR = pathlib.Path(__file__).parent / "csrc"
_gelu_cuda_ext = None


def _load_gelu_cuda_ext():
    """JIT-compile the GELU CUDA extension on first use (torch.utils.cpp_extension.load
    invokes nvcc + a C++ compiler and caches the .so under ~/.cache/torch_extensions)."""
    global _gelu_cuda_ext
    if _gelu_cuda_ext is None:
        from torch.utils.cpp_extension import load

        _gelu_cuda_ext = load(
            name="gelu_cuda_ext",
            sources=[
                str(_CSRC_DIR / "gelu_ext.cpp"),
                str(_CSRC_DIR / "gelu_kernel.cu"),
                str(_CSRC_DIR / "gelu_kernel_optimized.cu"),
            ],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=False,
        )
    return _gelu_cuda_ext


class ReLU(Function):
    @staticmethod
    def forward(ctx, x) -> Any:
        ctx.save_for_backward(x)
        return x * (x > 0)

    @staticmethod
    def backward(ctx, *grad_output) -> Any:
        (grad_output,) = grad_output
        (x,) = ctx.saved_tensors
        return grad_output * (x > 0)


class SoftMax(Function):
    @staticmethod
    def forward(ctx, x):
        # subtract max for numerical stability (softmax is shift-invariant)
        exp = torch.exp(x - x.max(dim=-1, keepdim=True).values)
        exp_sum = exp.sum(dim=-1, keepdim=True)
        out = exp / exp_sum
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, *grad_output):
        (grad_output,) = grad_output
        (out,) = ctx.saved_tensors
        dot = (grad_output * out).sum(dim=-1, keepdim=True)
        return out * (grad_output - dot)


class LayerNorm(Function):
    @staticmethod
    def forward(ctx, x, weight, bias):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        rstd = torch.rsqrt(var + 1e-5)  # 1/sqrt(var + eps)
        x_norm = (x - mean) * rstd  # normalized, before scale/shift
        out = x_norm * weight + bias

        ctx.save_for_backward(x_norm, rstd, weight)
        return out

    @staticmethod
    def backward(ctx, *grad_output):
        (grad_output,) = grad_output
        x_norm, rstd, weight = ctx.saved_tensors
        N = x_norm.shape[-1]

        grad_bias = grad_output.sum(dim=0)
        grad_weight = (grad_output * x_norm).sum(dim=0)

        grad_x_norm = grad_output * weight
        grad_x = (
            rstd
            / N
            * (
                N * grad_x_norm
                - grad_x_norm.sum(dim=-1, keepdim=True)
                - x_norm * (grad_x_norm * x_norm).sum(dim=-1, keepdim=True)
            )
        )

        return grad_x, grad_weight, grad_bias


class GELU(Function):
    """Exact GELU (via erf), forward and backward computed by a hand-written CUDA
    kernel in csrc/gelu_kernel.cu instead of composed torch ops. CUDA-only.

    Autograd is wired here at the Python level (this class). See gelu_native() below
    for the same kernel exposed as a native ATen op with autograd wired in C++ instead,
    which skips this class's Function.apply overhead."""

    @staticmethod
    def forward(ctx, x) -> Any:
        ext = _load_gelu_cuda_ext()
        x = x.contiguous()
        ctx.save_for_backward(x)
        return ext.forward(x)

    @staticmethod
    def backward(ctx, *grad_output) -> Any:
        (grad_output,) = grad_output
        (x,) = ctx.saved_tensors
        ext = _load_gelu_cuda_ext()
        return ext.backward(grad_output.contiguous(), x)


class GELUOptimized(Function):
    """Same math as GELU, but the kernel (csrc/gelu_kernel_optimized.cu) loads/stores
    128 bits per thread (float4 for float32, double2 for float64) instead of one scalar,
    matching the vectorization ATen's own elementwise kernels use. See README for the
    profiling that showed the naive GELU kernel is slower than torch's on the GPU
    itself, not just at the launch/dispatch layer, and that this is why."""

    @staticmethod
    def forward(ctx, x) -> Any:
        ext = _load_gelu_cuda_ext()
        x = x.contiguous()
        ctx.save_for_backward(x)
        return ext.forward_optimized(x)

    @staticmethod
    def backward(ctx, *grad_output) -> Any:
        (grad_output,) = grad_output
        (x,) = ctx.saved_tensors
        ext = _load_gelu_cuda_ext()
        return ext.backward_optimized(grad_output.contiguous(), x)


def gelu_native(x):
    """Same CUDA kernel as GELU, but registered as a native ATen op (TORCH_LIBRARY in
    csrc/gelu_ext.cpp) with autograd wired in C++ via torch::autograd::Function, instead
    of a Python torch.autograd.Function. Calling it skips Function.apply's Python-level
    bookkeeping (ctx creation, save_for_backward as Python objects, building the graph
    node in Python) entirely -- that bookkeeping still happens, just in C++. CUDA-only.
    """
    _load_gelu_cuda_ext()  # loading the .so runs its TORCH_LIBRARY static initializers
    return torch.ops.custom_autograd.gelu(x)
