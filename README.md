# Custom Autograd

Hand-written forward/backward implementations of common neural-net ops, wired into
PyTorch as `torch.autograd.Function` subclasses, benchmarked against PyTorch's builtins.

## Ops

| op | file | forward/backward implemented with |
|---|---|---|
| `ReLU` | `ops.py` | composed `torch` tensor ops |
| `SoftMax` | `ops.py` | composed `torch` tensor ops |
| `LayerNorm` | `ops.py` | composed `torch` tensor ops |
| `GELU` | `ops.py` + `csrc/` | a hand-written CUDA kernel, JIT-compiled as a PyTorch extension |

`ReLU`, `SoftMax`, and `LayerNorm` derive their backward pass by hand from the forward
math, but the forward/backward bodies themselves are just ordinary `torch` tensor
operations — they run on CPU or GPU and dispatch into whatever kernels ATen already
ships (elementwise ops, reductions, etc).

`GELU` goes one level deeper: forward and backward are `__global__` CUDA kernels that
this project owns, not ATen ops composed together. See [CUDA extension](#cuda-extension)
below for how that's built and wired in.

## Layout

```
ops.py          torch.autograd.Function subclasses (ReLU, SoftMax, LayerNorm, GELU)
csrc/
  gelu_kernel.h   launcher declarations shared between the two files below
  gelu_kernel.cu  the actual CUDA kernels (__global__) + launchers, compiled by nvcc
  gelu_ext.cpp    tensor checks, dtype dispatch, and the pybind11 module, compiled by g++
test.py         gradcheck against PyTorch's numerical differencing
bench.py        latency comparison: custom op vs. torch builtin
example.py      minimal usage example
```

## CUDA extension

`GELU` (`ops.py`) is backed by a real CUDA kernel instead of composed tensor ops. It's
built as a PyTorch C++/CUDA extension and JIT-compiled the first time `GELU.apply` runs,
via `torch.utils.cpp_extension.load`:

```python
_gelu_cuda_ext = load(
    name="gelu_cuda_ext",
    sources=["csrc/gelu_ext.cpp", "csrc/gelu_kernel.cu"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)
```

This shells out to `nvcc`/`g++` and `ninja`, caches the compiled `.so` under
`~/.cache/torch_extensions/`, and only recompiles when the sources change — no separate
build step or `setup.py` is needed.

**Why two source files instead of one.** The natural thing is a single `.cu` file with
kernels, tensor checks, dispatch, and the pybind11 module together. That works with an
older/matched toolchain, but on a newer host GCC (16 here) nvcc's own front end — not
the host compiler picked via `-ccbin` — fails to parse a template in one of ATen's
headers (`ATen/core/List_inl.h`) that gets pulled in by `<torch/extension.h>`. Splitting
the file sidesteps it entirely:

- `gelu_kernel.cu` — only `__global__` kernels and their launchers. Includes just
  `<cuda_runtime.h>` and `<math.h>`, so nvcc's front end never sees the ATen headers
  that trip it up. Compiled by `nvcc`.
- `gelu_ext.cpp` — `torch::Tensor` checks, `AT_DISPATCH_FLOATING_TYPES` dtype dispatch,
  and the `PYBIND11_MODULE` that exposes `forward`/`backward` to Python. Includes
  `<torch/extension.h>` and is compiled by the plain host C++ compiler, which handles
  that header fine.

This split (kernel-only `.cu`, torch-aware `.cpp`) is also just a reasonable way to
structure any nontrivial PyTorch CUDA extension, independent of the toolchain issue.

`GELU` only runs on CUDA tensors — there's no CPU fallback, so `test.py` and `bench.py`
skip it when `torch.cuda.is_available()` is `False`.

## Running

```
uv run test.py                   # gradcheck all four ops (GELU only if a GPU is available)
uv run bench.py                  # latency: custom vs. torch builtin, on CPU/GPU as available
uv run bench_gelu_overhead.py    # decomposes GELU's slowdown vs. torch (see below)
uv run example.py                # minimal LayerNorm usage
```

## Benchmarks

Measured on an RTX 3060, `x = torch.randn(512, 768)`:

| op | cpu custom | gpu custom | gpu torch | gpu slowdown |
|---|---|---|---|---|
| relu | 435 us | 39 us | 19 us | 2x |
| softmax | 684 us | 67 us | 22 us | 3x |
| layernorm | 4490 us | 97 us | 26 us | 4x |
| gelu (raw CUDA kernel) | n/a | 33 us | 19 us | 2x |

The composed-tensor-op implementations (`relu`/`softmax`/`layernorm`) are each several
separate kernel launches under the hood, which is most of the gap to torch's fused
builtins. `GELU`'s hand-written kernel is a single launch — on par with `ReLU`'s
slowdown despite doing more math per element, since kernel-launch overhead (not compute)
dominates at this tensor size.

### Where GELU's slowdown actually comes from

Writing a real CUDA kernel didn't close the gap to `torch.nn.functional.gelu`, which is
initially surprising — the whole point was to stop being a composition of ATen calls.
`bench_gelu_overhead.py` isolates two separate, stacking causes by comparing three call
paths (raw extension call, `GELU.apply`, and torch's builtin) and by varying tensor size:

```
raw extension call (no autograd)   25.6 us
GELU.apply (autograd.Function)     34.4 us
torch gelu                         19.3 us

n=    393216  custom=  25.8us  torch=19.3us  ratio=1.34x
n=   8000000  custom= 216.2us  torch=210.6us  ratio=1.03x
n=  64000000  custom=1620.3us  torch=1599.4us  ratio=1.01x
```

**1. `torch.autograd.Function.apply` overhead (~9 us, the bigger piece here).**
`GELU.apply(x)` builds a graph node, runs `ctx.save_for_backward`, and goes through
autograd/dispatcher bookkeeping before the C++ call happens at all. Calling the compiled
extension's `forward` directly (skipping `Function.apply`) drops 34.4us to 25.6us.
`torch.nn.functional.gelu` has no such wrapper — its backward is a natively registered
ATen op, not a Python-level `Function`. This is also why the composed-tensor-op ops
(`ReLU`, `SoftMax`, `LayerNorm`) show similar 2-4x slowdowns despite doing nothing but
calling into ATen: the overhead is inherent to wrapping any op in a custom
`autograd.Function`, not specific to hand-writing CUDA.

**2. Fixed per-launch overhead, not slower compute (~6 us at this size, vanishes at scale).**
Even bypassing autograd, the raw extension call is slower than torch's builtin at 393K
elements (25.8 vs 19.3us) but nearly identical at 64M elements (1.01x). GELU is memory-
bound and this tensor is tiny, so at 393K elements you're mostly measuring fixed
per-call cost — the pybind11 call boundary (marshaling the `torch::Tensor` argument,
`TORCH_CHECK` validation) plus kernel launch, versus torch's builtin going through an
already-warm, C++-resident dispatch path. The 1% gap at 64M elements shows the actual
memory-bandwidth-bound kernel work is essentially the same; the naive one-element-per-
thread kernel isn't meaningfully worse than torch's at doing the real work, it's just
worse at the fixed cost of getting there.
