from typing import Any

import torch
from torch.autograd import Function


class ReLU(Function):
    @staticmethod
    def forward(ctx, x) -> Any:
        ctx.save_for_backward(x)
        return x * (x > 0)

    @staticmethod
    def backward(ctx, grad_output) -> Any:
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
    def backward(ctx, grad_output):
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
    def backward(ctx, grad_output):
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
