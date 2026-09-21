import torch
from torch.autograd import gradcheck

from ops import GELU, GELUOptimized, LayerNorm, ReLU, SoftMax, gelu_native


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

    # GELU is a CUDA kernel, so it only runs where a GPU is available
    if torch.cuda.is_available():
        x_cuda = x.detach().to("cuda").requires_grad_()
        assert gradcheck(GELU.apply, (x_cuda,), eps=1e-6, atol=1e-4), "GELU"
        assert gradcheck(gelu_native, (x_cuda,), eps=1e-6, atol=1e-4), "gelu_native"
        assert gradcheck(GELUOptimized.apply, (x_cuda,), eps=1e-6, atol=1e-4), (
            "GELUOptimized"
        )

        # element count not a multiple of the vector width (4 for float32/float64's 2),
        # to exercise GELUOptimized's scalar tail-kernel path
        x_odd = torch.randn(4, 7, dtype=torch.float64, device="cuda", requires_grad=True)
        assert gradcheck(GELUOptimized.apply, (x_odd,), eps=1e-6, atol=1e-4), (
            "GELUOptimized (tail path)"
        )


if __name__ == "__main__":
    main()
