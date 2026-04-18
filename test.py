import torch
from torch.autograd import gradcheck
from ops import ReLU

# gradcheck requires float64 and requires_grad=True
x = torch.randn(4, 8, dtype=torch.float64, requires_grad=True)

assert gradcheck(ReLU.apply, (x,), eps=1e-6, atol=1e-4), "ReLU grad"
