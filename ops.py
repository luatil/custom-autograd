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
