// Torch-aware glue: tensor checks, dtype dispatch, and the pybind module. Compiled by
// the host C++ compiler only (see gelu_kernel.cu for why this is split out).
#include <torch/extension.h>

#include "gelu_kernel.h"

torch::Tensor gelu_forward(torch::Tensor x) {
  TORCH_CHECK(x.is_cuda(), "gelu_forward: x must be a CUDA tensor");
  TORCH_CHECK(x.is_contiguous(), "gelu_forward: x must be contiguous");

  auto y = torch::empty_like(x);
  const int64_t n = x.numel();

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "gelu_forward_cuda", ([&] {
    gelu_forward_launcher<scalar_t>(x.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(), n);
  }));

  return y;
}

torch::Tensor gelu_backward(torch::Tensor grad_output, torch::Tensor x) {
  TORCH_CHECK(grad_output.is_cuda(), "gelu_backward: grad_output must be a CUDA tensor");
  TORCH_CHECK(x.is_cuda(), "gelu_backward: x must be a CUDA tensor");
  TORCH_CHECK(grad_output.is_contiguous(), "gelu_backward: grad_output must be contiguous");
  TORCH_CHECK(x.is_contiguous(), "gelu_backward: x must be contiguous");

  auto grad_input = torch::empty_like(x);
  const int64_t n = x.numel();

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "gelu_backward_cuda", ([&] {
    gelu_backward_launcher<scalar_t>(grad_output.data_ptr<scalar_t>(), x.data_ptr<scalar_t>(),
                                      grad_input.data_ptr<scalar_t>(), n);
  }));

  return grad_input;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &gelu_forward, "GELU forward (CUDA)");
  m.def("backward", &gelu_backward, "GELU backward (CUDA)");
}
