"""
Profile GELU forward with torch.profiler to isolate actual GPU kernel time (self CUDA
time) from CPU-side launch/dispatch overhead, across:

  - the naive kernel (csrc/gelu_kernel.cu): one scalar load/store per thread
  - the vectorized kernel (csrc/gelu_kernel_optimized.cu): one float4/double2 load per
    thread, matching the pattern behind ATen's own elementwise kernels
  - torch.nn.functional.gelu (builtin)

This is what motivated writing the optimized kernel in the first place: bench.py and
bench_gelu_overhead.py showed GELU's custom kernel was slower than torch's builtin, and
this script's job is to check whether that gap is CPU-side overhead (launch/dispatch) or
the GPU kernel itself being slower. It's the latter, by a wide margin: torch's builtin
uses a kernel named `vectorized_elementwise_kernel<4, ...>` -- 4 elements per thread via
a single wide memory transaction, instead of one thread per element. See README.
"""

import torch
from torch.profiler import ProfilerActivity, profile

from ops import _load_gelu_cuda_ext


def profile_call(fn, x, label, iters=300, warmup=30):
    for _ in range(warmup):
        fn(x)
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn(x)
        torch.cuda.synchronize()

    events = prof.key_averages()
    total_cuda_us = sum(e.self_device_time_total for e in events) / iters
    print(f"{label}: {total_cuda_us:.3f} us/call (self CUDA time)")
    for e in events:
        if e.self_device_time_total > 0:
            print(f"  {e.key[:100]}: {e.self_device_time_total / iters:.3f} us/call")


def main():
    if not torch.cuda.is_available():
        print("GELU is CUDA-only; skipping (no GPU available)")
        return

    ext = _load_gelu_cuda_ext()
    x = torch.randn(512, 768, device="cuda")

    profile_call(ext.forward, x, "naive kernel (gelu_kernel.cu)")
    profile_call(ext.forward_optimized, x, "vectorized kernel (gelu_kernel_optimized.cu)")
    profile_call(torch.nn.functional.gelu, x, "torch gelu (builtin)")


if __name__ == "__main__":
    main()
