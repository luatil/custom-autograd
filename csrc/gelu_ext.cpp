// Torch-aware glue: tensor checks, dtype dispatch, the pybind module, and a native
// ATen op (see the TORCH_LIBRARY block below). Compiled by the host C++ compiler only
// (see gelu_kernel.cu for why this is split out).
#include <torch/extension.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>

#include "gelu_kernel.h"
#include "gelu_kernel_optimized.h"

torch::Tensor custom_gelu_forward(torch::Tensor x) {
  TORCH_CHECK(x.is_cuda(), "custom_gelu_forward: x must be a CUDA tensor");
  TORCH_CHECK(x.is_contiguous(), "custom_gelu_forward: x must be contiguous");

  auto y = torch::empty_like(x);
  const int64_t n = x.numel();

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "gelu_forward_cuda", ([&] {
    gelu_forward_launcher<scalar_t>(x.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(), n);
  }));

  return y;
}

torch::Tensor custom_gelu_backward(torch::Tensor grad_output, torch::Tensor x) {
  TORCH_CHECK(grad_output.is_cuda(), "custom_gelu_backward: grad_output must be a CUDA tensor");
  TORCH_CHECK(x.is_cuda(), "custom_gelu_backward: x must be a CUDA tensor");
  TORCH_CHECK(grad_output.is_contiguous(), "custom_gelu_backward: grad_output must be contiguous");
  TORCH_CHECK(x.is_contiguous(), "custom_gelu_backward: x must be contiguous");

  auto grad_input = torch::empty_like(x);
  const int64_t n = x.numel();

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "gelu_backward_cuda", ([&] {
    gelu_backward_launcher<scalar_t>(grad_output.data_ptr<scalar_t>(), x.data_ptr<scalar_t>(),
                                      grad_input.data_ptr<scalar_t>(), n);
  }));

  return grad_input;
}

torch::Tensor custom_gelu_forward_optimized(torch::Tensor x) {
  TORCH_CHECK(x.is_cuda(), "custom_gelu_forward_optimized: x must be a CUDA tensor");
  TORCH_CHECK(x.is_contiguous(), "custom_gelu_forward_optimized: x must be contiguous");

  auto y = torch::empty_like(x);
  const int64_t n = x.numel();

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "gelu_forward_cuda_optimized", ([&] {
    gelu_forward_launcher_optimized<scalar_t>(x.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(), n);
  }));

  return y;
}

torch::Tensor custom_gelu_backward_optimized(torch::Tensor grad_output, torch::Tensor x) {
  TORCH_CHECK(grad_output.is_cuda(), "custom_gelu_backward_optimized: grad_output must be a CUDA tensor");
  TORCH_CHECK(x.is_cuda(), "custom_gelu_backward_optimized: x must be a CUDA tensor");
  TORCH_CHECK(grad_output.is_contiguous(), "custom_gelu_backward_optimized: grad_output must be contiguous");
  TORCH_CHECK(x.is_contiguous(), "custom_gelu_backward_optimized: x must be contiguous");

  auto grad_input = torch::empty_like(x);
  const int64_t n = x.numel();

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "gelu_backward_cuda_optimized", ([&] {
    gelu_backward_launcher_optimized<scalar_t>(grad_output.data_ptr<scalar_t>(), x.data_ptr<scalar_t>(),
                                                grad_input.data_ptr<scalar_t>(), n);
  }));

  return grad_input;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &custom_gelu_forward, "GELU forward (CUDA)");
  m.def("backward", &custom_gelu_backward, "GELU backward (CUDA)");
  m.def("forward_optimized", &custom_gelu_forward_optimized, "GELU forward, vectorized (CUDA)");
  m.def("backward_optimized", &custom_gelu_backward_optimized, "GELU backward, vectorized (CUDA)");
}

// --- Native ATen op --------------------------------------------------------------
// Registers gelu as a real dispatcher op (callable as torch.ops.custom_autograd.gelu)
// with autograd wired in C++ instead of a Python torch.autograd.Function. The static
// TORCH_LIBRARY initializers below run as soon as this shared library is loaded into
// the process, so torch.ops.custom_autograd.gelu exists the moment
// torch.utils.cpp_extension.load() returns -- no separate pybind exposure needed.

namespace {

class GeluFunction : public torch::autograd::Function<GeluFunction> {
 public:
  static torch::Tensor forward(torch::autograd::AutogradContext* ctx, torch::Tensor x) {
    x = x.contiguous();
    ctx->save_for_backward({x});
    return custom_gelu_forward(x);
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx, torch::autograd::variable_list grad_outputs) {
    auto x = ctx->get_saved_variables()[0];
    auto grad_x = custom_gelu_backward(grad_outputs[0].contiguous(), x);
    return {grad_x};
  }
};

torch::Tensor gelu_native(const torch::Tensor& x) { return GeluFunction::apply(x); }

}  // namespace

TORCH_LIBRARY(custom_autograd, m) { m.def("gelu(Tensor x) -> Tensor"); }

TORCH_LIBRARY_IMPL(custom_autograd, Autograd, m) { m.impl("gelu", gelu_native); }
