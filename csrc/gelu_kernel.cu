// Raw CUDA kernels + launchers for GELU. Deliberately free of ATen/torch headers:
// nvcc's own front end (independent of the host compiler picked via -ccbin) chokes on
// a template in ATen/core/List_inl.h under newer GCC, so all torch-aware code (tensor
// checks, dispatch, pybind glue) lives in gelu_ext.cpp instead, compiled by g++ directly.
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <cstdint>

#include "gelu_kernel.h"

namespace {

constexpr double kInvSqrt2 = 0.7071067811865476;
constexpr double kInvSqrt2Pi = 0.3989422804014327;

template <typename scalar_t>
__global__ void gelu_forward_kernel(const scalar_t* __restrict__ x,
                                     scalar_t* __restrict__ y, int64_t n) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    scalar_t xi = x[i];
    scalar_t cdf = scalar_t(0.5) * (scalar_t(1.0) + erf(xi * scalar_t(kInvSqrt2)));
    y[i] = xi * cdf;
  }
}

template <typename scalar_t>
__global__ void gelu_backward_kernel(const scalar_t* __restrict__ grad_out,
                                      const scalar_t* __restrict__ x,
                                      scalar_t* __restrict__ grad_in, int64_t n) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    scalar_t xi = x[i];
    scalar_t cdf = scalar_t(0.5) * (scalar_t(1.0) + erf(xi * scalar_t(kInvSqrt2)));
    scalar_t pdf = scalar_t(kInvSqrt2Pi) * exp(scalar_t(-0.5) * xi * xi);
    grad_in[i] = grad_out[i] * (cdf + xi * pdf);
  }
}

}  // namespace

template <typename scalar_t>
void gelu_forward_launcher(const scalar_t* x, scalar_t* y, int64_t n) {
  const int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  gelu_forward_kernel<scalar_t><<<blocks, threads>>>(x, y, n);
}

template <typename scalar_t>
void gelu_backward_launcher(const scalar_t* grad_out, const scalar_t* x,
                             scalar_t* grad_in, int64_t n) {
  const int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  gelu_backward_kernel<scalar_t><<<blocks, threads>>>(grad_out, x, grad_in, n);
}

template void gelu_forward_launcher<float>(const float*, float*, int64_t);
template void gelu_forward_launcher<double>(const double*, double*, int64_t);
template void gelu_backward_launcher<float>(const float*, const float*, float*, int64_t);
template void gelu_backward_launcher<double>(const double*, const double*, double*, int64_t);
