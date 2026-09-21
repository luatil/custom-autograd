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
| `gelu_native` | `ops.py` + `csrc/` | the same CUDA kernel, exposed as a native ATen op (`TORCH_LIBRARY`) with autograd wired in C++ instead of Python |

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

### Native ATen op (`gelu_native`)

The same kernel is also exposed as a native dispatcher op, registered in
`gelu_ext.cpp`:

```cpp
class GeluFunction : public torch::autograd::Function<GeluFunction> {
 public:
  static torch::Tensor forward(torch::autograd::AutogradContext* ctx, torch::Tensor x) { ... }
  static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,
                                                  torch::autograd::variable_list grad_outputs) { ... }
};

TORCH_LIBRARY(custom_autograd, m) { m.def("gelu(Tensor x) -> Tensor"); }
TORCH_LIBRARY_IMPL(custom_autograd, Autograd, m) { m.impl("gelu", gelu_native); }
```

`TORCH_LIBRARY`'s static initializers run as soon as the `.so` is loaded, so
`torch.ops.custom_autograd.gelu` exists the moment `cpp_extension.load()` returns — no
separate pybind exposure needed. This was written specifically to test whether wiring
autograd in C++ (`torch::autograd::Function`, not Python's `torch.autograd.Function`)
would bypass the overhead described below. **It doesn't** — see the benchmark section.

`gelu_ext.cpp` also has to rename its own `gelu_forward`/`gelu_backward` helpers to
`custom_gelu_forward`/`custom_gelu_backward`: unqualified names collided with ATen's own
`at::gelu_forward`/`at::gelu_backward` (pulled in transitively by `<torch/extension.h>`),
making the calls ambiguous.

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
`bench_gelu_overhead.py` isolates the causes by comparing four call paths at multiple
tensor sizes: the raw extension call (no autograd at all), `GELU.apply` (Python
autograd), `gelu_native` (C++ autograd via a native ATen op), and torch's builtin:

```
raw extension call (no autograd)             26.2 us
GELU.apply (Python autograd.Function)        35.4 us
gelu_native (native ATen op, C++ autograd)   44.8 us
torch gelu                                   18.7 us

n=    393216  raw= 26.3us  GELU.apply= 35.3us  gelu_native= 44.7us  torch=18.8us
n=   8000000  raw=213.7us  GELU.apply=222.9us  gelu_native=226.9us  torch=207.6us
n=  64000000  raw=1582us   GELU.apply=1591us   gelu_native=1602us   torch=1568us
```

**1. `torch.autograd.Function.apply` overhead (~9 us, fixed per call).**
`GELU.apply(x)` builds a graph node, runs `ctx.save_for_backward`, and goes through
autograd bookkeeping in Python before the C++ call happens at all. Calling the compiled
extension's `forward` directly (skipping `Function.apply`) drops 35.4us to 26.2us.
`torch.nn.functional.gelu` has no such wrapper — its backward is a natively registered
ATen op. This is also why the composed-tensor-op ops (`ReLU`, `SoftMax`, `LayerNorm`)
show similar 2-4x slowdowns despite doing nothing but calling into ATen: the overhead is
inherent to wrapping any op in a custom `autograd.Function`, not specific to hand-writing
CUDA.

**2. `gelu_native` doesn't fix this — it makes it worse (~9 us more on top).** The
hypothesis behind writing `gelu_native` was that moving autograd into C++
(`torch::autograd::Function` + `TORCH_LIBRARY`) would skip `Function.apply`'s Python
bookkeeping and get closer to torch's builtin. Measured, it's the *slowest* of the four
paths at small size (44.8us, worse than `GELU.apply`'s 35.4us). Going through
`torch.ops.custom_autograd.gelu` pays for the dispatcher's own machinery — computing the
dispatch key set, boxed/unboxed calling convention, kernel table lookup — every call,
and for a tiny single-input op that machinery costs more than the Python-level
bookkeeping it was meant to replace. `torch.nn.functional.gelu` also goes through the
dispatcher, but as a built-in op it's on a more optimized path than a
`TORCH_LIBRARY`-registered custom op gets by default. The lesson: dispatcher
registration isn't automatically "closer to the metal" than a plain pybind11 call — for
tiny ops, a direct C++ function call beats going through `torch.ops.*` even when the
Python-side `Function.apply` wrapper is removed entirely.

**3. Fixed per-launch overhead, not slower compute (vanishes at scale).** All four paths
converge to within ~2% of each other at 64M elements, while at 393K elements the raw
extension call is already 1.4x torch's time. GELU is memory-bound and 393K elements is
tiny, so at that size every path is dominated by its fixed per-call cost (Python
bookkeeping, dispatcher machinery, or the pybind11 call boundary and kernel launch
itself) rather than actual kernel work. The naive one-element-per-thread kernel isn't
meaningfully worse than torch's at doing the real compute — it's worse at the fixed cost
of getting there, and that fixed cost is what every one of these wrapper layers adds to.
