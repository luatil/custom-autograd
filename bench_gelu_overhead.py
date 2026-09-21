"""
Decompose GELU's slowdown vs. torch into (1) autograd.Function/Python overhead and
(2) actual kernel cost, by comparing three call paths at multiple tensor sizes:

  - the raw CUDA extension call, bypassing torch.autograd.Function entirely
  - GELU.apply (goes through torch.autograd.Function.apply)
  - torch.nn.functional.gelu (builtin ATen op, no custom Function wrapper)
"""

import torch
import torch.utils.benchmark as benchmark

from ops import GELU, _load_gelu_cuda_ext


def bench(stmt, globals_, label):
    t = benchmark.Timer(stmt=stmt, globals=globals_, label=label)
    return t.blocked_autorange(min_run_time=0.5)


def main():
    if not torch.cuda.is_available():
        print("GELU is CUDA-only; skipping (no GPU available)")
        return

    ext = _load_gelu_cuda_ext()

    # fixed-size comparison: raw extension call vs GELU.apply vs torch gelu
    x = torch.randn(512, 768, device="cuda")
    r1 = bench(
        "ext.forward(x); torch.cuda.synchronize()",
        {"ext": ext, "x": x, "torch": torch},
        "raw extension call (no autograd)",
    )
    r2 = bench(
        "GELU.apply(x); torch.cuda.synchronize()",
        {"GELU": GELU, "x": x, "torch": torch},
        "GELU.apply (autograd.Function)",
    )
    r3 = bench(
        "torch.nn.functional.gelu(x); torch.cuda.synchronize()",
        {"x": x, "torch": torch},
        "torch gelu",
    )
    print(r1)
    print(r2)
    print(r3)

    # scaling comparison: raw extension call vs torch gelu as n grows, to see whether
    # the gap is fixed per-launch overhead (shrinks as a fraction of total time) or
    # genuinely slower compute (stays proportional)
    print()
    for n in [512 * 768, 8_000_000, 64_000_000]:
        x = torch.randn(n, device="cuda")
        custom = bench(
            "ext.forward(x); torch.cuda.synchronize()",
            {"ext": ext, "x": x, "torch": torch},
            f"custom n={n}",
        )
        torch_builtin = bench(
            "torch.nn.functional.gelu(x); torch.cuda.synchronize()",
            {"x": x, "torch": torch},
            f"torch n={n}",
        )
        ratio = custom.median / torch_builtin.median
        print(
            f"n={n:>10}  custom={custom.median * 1e6:8.2f}us  "
            f"torch={torch_builtin.median * 1e6:8.2f}us  ratio={ratio:.2f}x"
        )


if __name__ == "__main__":
    main()
