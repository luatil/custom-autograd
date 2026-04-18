import torch
from torch.autograd import gradcheck

from ops import LayerNorm, ReLU, SoftMax


def main():
    # gradcheck requires float64 and requires_grad=True
    x = torch.randn(4, 8, dtype=torch.float64, requires_grad=True)

    assert gradcheck(ReLU.apply, (x,), eps=1e-6, atol=1e-4), "ReLU"
    assert gradcheck(SoftMax.apply, (x,), eps=1e-6, atol=1e-4), "SoftMax"

    # layernorm needs weight and bias too
    weight = torch.ones(8, dtype=torch.float64, requires_grad=True)
    bias = torch.zeros(8, dtype=torch.float64, requires_grad=True)
    assert gradcheck(LayerNorm.apply, (x, weight, bias), eps=1e-6, atol=1e-4), (
        "LayerNorm"
    )


if __name__ == "__main__":
    main()
