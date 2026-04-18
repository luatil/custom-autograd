import torch

from ops import LayerNorm


def main():
    # gradcheck requires float64 and requires_grad=True
    x = torch.randn(4, 8, dtype=torch.float64, requires_grad=True)
    weight = torch.ones(8, dtype=torch.float64, requires_grad=True)
    bias = torch.zeros(8, dtype=torch.float64, requires_grad=True)
    out = LayerNorm.apply(x, weight, bias)
    out.sum().backward()
    print(x.grad)

    # assert gradcheck(ReLU.apply, (x,), eps=1e-6, atol=1e-4), "ReLU grad"


if __name__ == "__main__":
    main()
