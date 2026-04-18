import torch
from torch.autograd import Function

from typing import Any


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
