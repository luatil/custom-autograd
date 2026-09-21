"""
Decompose GELU's slowdown vs. torch by comparing call paths at multiple tensor sizes:

  - the raw CUDA extension call (naive kernel), bypassing autograd entirely
  - GELU.apply (torch.autograd.Function.apply, Python-level autograd, naive kernel)
  - gelu_native (a native ATen op via TORCH_LIBRARY, C++-level autograd, naive kernel)
  - the raw extension call for the vectorized kernel, bypassing autograd entirely
  - GELUOptimized.apply (Python-level autograd, vectorized kernel)
  - torch.nn.functional.gelu (builtin ATen op, no custom wrapper at all)

gelu_native was written on the assumption that registering a real dispatcher op with
autograd wired in C++ would skip GELU.apply's Python-side bookkeeping and close the gap
to torch's builtin. It doesn't: going through torch.ops.* pays for the dispatcher's own
key-set computation and calling convention, which turns out to cost more than the
Python bookkeeping it was meant to avoid, at least for an op this small (see README).

GELUOptimized was written after bench_gelu_profile.py showed the naive kernel is also
genuinely slower on the GPU itself (not just around it): its kernel does one scalar
load/store per thread, while torch's builtin loads 4 floats per thread as a single
128-bit transaction. GELUOptimized's kernel (csrc/gelu_kernel_optimized.cu) does the
same. That closes most of the actual GPU kernel-time gap (see bench_gelu_profile.py),
but the Python autograd.Function.apply overhead (points 1-2 above) is orthogonal to the
kernel and isn't affected by it, so it still shows up here in wall-clock time.
"""

import torch
import torch.utils.benchmark as benchmark

from ops import GELU, GELUOptimized, _load_gelu_cuda_ext, gelu_native


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
        "GELU.apply (Python autograd.Function)",
    )
    r3 = bench(
        "gelu_native(x); torch.cuda.synchronize()",
        {"gelu_native": gelu_native, "x": x, "torch": torch},
        "gelu_native (native ATen op, C++ autograd)",
    )
    r4 = bench(
        "ext.forward_optimized(x); torch.cuda.synchronize()",
        {"ext": ext, "x": x, "torch": torch},
        "raw extension call, vectorized kernel (no autograd)",
    )
    r5 = bench(
        "GELUOptimized.apply(x); torch.cuda.synchronize()",
        {"GELUOptimized": GELUOptimized, "x": x, "torch": torch},
        "GELUOptimized.apply (Python autograd.Function, vectorized kernel)",
    )
    r6 = bench(
        "torch.nn.functional.gelu(x); torch.cuda.synchronize()",
        {"x": x, "torch": torch},
        "torch gelu",
    )
    print(r1)
    print(r2)
    print(r3)
    print(r4)
    print(r5)
    print(r6)

    # scaling comparison across all paths, to see whether gaps are fixed per-launch
    # overhead (shrinks as a fraction of total time) or genuinely slower compute
    # (stays proportional)
    print()
    for n in [512 * 768, 8_000_000, 64_000_000]:
        x = torch.randn(n, device="cuda")
        raw = bench(
            "ext.forward(x); torch.cuda.synchronize()", {"ext": ext, "x": x, "torch": torch}, "raw"
        )
        py = bench(
            "GELU.apply(x); torch.cuda.synchronize()", {"GELU": GELU, "x": x, "torch": torch}, "py"
        )
        native = bench(
            "gelu_native(x); torch.cuda.synchronize()",
            {"gelu_native": gelu_native, "x": x, "torch": torch},
            "native",
        )
        raw_opt = bench(
            "ext.forward_optimized(x); torch.cuda.synchronize()",
            {"ext": ext, "x": x, "torch": torch},
            "raw_opt",
        )
        py_opt = bench(
            "GELUOptimized.apply(x); torch.cuda.synchronize()",
            {"GELUOptimized": GELUOptimized, "x": x, "torch": torch},
            "py_opt",
        )
        torch_builtin = bench(
            "torch.nn.functional.gelu(x); torch.cuda.synchronize()",
            {"x": x, "torch": torch},
            f"torch n={n}",
        )
        print(
            f"n={n:>10}  raw={raw.median * 1e6:7.2f}us  "
            f"GELU.apply={py.median * 1e6:7.2f}us  "
            f"gelu_native={native.median * 1e6:7.2f}us  "
            f"raw_opt={raw_opt.median * 1e6:7.2f}us  "
            f"GELUOptimized.apply={py_opt.median * 1e6:7.2f}us  "
            f"torch={torch_builtin.median * 1e6:7.2f}us"
        )


if __name__ == "__main__":
    main()
