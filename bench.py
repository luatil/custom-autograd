"""
Benchmark custom implementation against builtin torch implementation.

| op | cpu custom | gpu custom | gpu torch | gpu slowdown |
|---|---|---|---|---|
| relu | 435 us | 39 us | 18 us | 2x |
| softmax | 684 us | 62 us | 22 us | 3x |
| layernorm | 4490 us | 100 us | 25 us | 4x |
"""

import torch
import torch.utils.benchmark as benchmark

from ops import LayerNorm, ReLU, SoftMax


def bench(fn, *args, label):
    t = benchmark.Timer(
        stmt="fn(*args); torch.cuda.synchronize()",
        globals={"fn": fn, "args": args, "torch": torch},
        label=label,
    )
    return t.blocked_autorange(min_run_time=1.0)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"running on: {device}")

    x = torch.randn(512, 768, device=device)
    weight = torch.ones(768, device=device)
    bias = torch.zeros(768, device=device)

    # compare relu
    r1 = bench(ReLU.apply, x, label="custom relu")
    r2 = bench(torch.relu, x, label="torch relu")

    # compare softmax
    s1 = bench(SoftMax.apply, x, label="custom softmax")
    s2 = bench(lambda x: torch.softmax(x, dim=-1), x, label="torch softmax")

    # compare layernorm
    l1 = bench(LayerNorm.apply, x, weight, bias, label="custom layernorm")
    l2 = bench(
        torch.nn.functional.layer_norm, x, (768,), weight, bias, label="torch layernorm"
    )

    print(r1)
    print(r2)
    print(s1)
    print(s2)
    print(l1)
    print(l2)


if __name__ == "__main__":
    main()
