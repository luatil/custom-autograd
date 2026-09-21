// Vectorized GELU kernels: each thread loads/stores a 128-bit chunk (float4 for
// float32, double2 for float64) instead of one scalar, matching the pattern behind
// ATen's vectorized_elementwise_kernel. See README for the profiling that motivated
// this (torch's builtin kernel is faster on-GPU, not just cheaper to launch, because it
// does this). Falls back to the plain scalar kernels in gelu_kernel.cu when a tensor's
// data pointer isn't aligned for wide loads, and handles any remainder elements (n not
// a multiple of the vector width) with a small scalar tail kernel.
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <cstdint>
#include <algorithm>

#include "gelu_kernel.h"
#include "gelu_kernel_optimized.h"

namespace {

constexpr double kInvSqrt2 = 0.7071067811865476;
constexpr double kInvSqrt2Pi = 0.3989422804014327;

template <typename scalar_t>
__device__ __forceinline__ scalar_t gelu_elem(scalar_t xi) {
  scalar_t cdf = scalar_t(0.5) * (scalar_t(1.0) + erf(xi * scalar_t(kInvSqrt2)));
  return xi * cdf;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t gelu_grad_elem(scalar_t xi, scalar_t grad_out) {
  scalar_t cdf = scalar_t(0.5) * (scalar_t(1.0) + erf(xi * scalar_t(kInvSqrt2)));
  scalar_t pdf = scalar_t(kInvSqrt2Pi) * exp(scalar_t(-0.5) * xi * xi);
  return grad_out * (cdf + xi * pdf);
}

// Maps each scalar type to a 128-bit-wide vector type/width.
template <typename scalar_t>
struct VecTraits;
template <>
struct VecTraits<float> {
  using vec_t = float4;
  static constexpr int width = 4;
};
template <>
struct VecTraits<double> {
  using vec_t = double2;
  static constexpr int width = 2;
};

template <typename scalar_t>
__global__ void gelu_forward_kernel_vec(const scalar_t* __restrict__ x,
                                         scalar_t* __restrict__ y, int64_t n_vec) {
  using vec_t = typename VecTraits<scalar_t>::vec_t;
  constexpr int width = VecTraits<scalar_t>::width;
  const vec_t* xv = reinterpret_cast<const vec_t*>(x);
  vec_t* yv = reinterpret_cast<vec_t*>(y);

  int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n_vec;
       i += stride) {
    vec_t v = xv[i];
    scalar_t* vp = reinterpret_cast<scalar_t*>(&v);
#pragma unroll
    for (int j = 0; j < width; ++j) vp[j] = gelu_elem(vp[j]);
    yv[i] = v;
  }
}

template <typename scalar_t>
__global__ void gelu_forward_kernel_tail(const scalar_t* __restrict__ x,
                                          scalar_t* __restrict__ y, int64_t start, int64_t n) {
  int64_t i = start + static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) y[i] = gelu_elem(x[i]);
}

template <typename scalar_t>
__global__ void gelu_backward_kernel_vec(const scalar_t* __restrict__ grad_out,
                                          const scalar_t* __restrict__ x,
                                          scalar_t* __restrict__ grad_in, int64_t n_vec) {
  using vec_t = typename VecTraits<scalar_t>::vec_t;
  constexpr int width = VecTraits<scalar_t>::width;
  const vec_t* gov = reinterpret_cast<const vec_t*>(grad_out);
  const vec_t* xv = reinterpret_cast<const vec_t*>(x);
  vec_t* giv = reinterpret_cast<vec_t*>(grad_in);

  int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n_vec;
       i += stride) {
    vec_t gv = gov[i];
    vec_t xvv = xv[i];
    vec_t rv;
    const scalar_t* gp = reinterpret_cast<const scalar_t*>(&gv);
    const scalar_t* xp = reinterpret_cast<const scalar_t*>(&xvv);
    scalar_t* rp = reinterpret_cast<scalar_t*>(&rv);
#pragma unroll
    for (int j = 0; j < width; ++j) rp[j] = gelu_grad_elem(xp[j], gp[j]);
    giv[i] = rv;
  }
}

template <typename scalar_t>
__global__ void gelu_backward_kernel_tail(const scalar_t* __restrict__ grad_out,
                                           const scalar_t* __restrict__ x,
                                           scalar_t* __restrict__ grad_in, int64_t start,
                                           int64_t n) {
  int64_t i = start + static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) grad_in[i] = gelu_grad_elem(x[i], grad_out[i]);
}

template <typename scalar_t>
bool is_aligned(const void* ptr) {
  constexpr size_t align = sizeof(typename VecTraits<scalar_t>::vec_t);
  return reinterpret_cast<uintptr_t>(ptr) % align == 0;
}

}  // namespace

template <typename scalar_t>
void gelu_forward_launcher_optimized(const scalar_t* x, scalar_t* y, int64_t n) {
  if (!is_aligned<scalar_t>(x) || !is_aligned<scalar_t>(y)) {
    gelu_forward_launcher<scalar_t>(x, y, n);  // scalar fallback (gelu_kernel.cu)
    return;
  }

  constexpr int width = VecTraits<scalar_t>::width;
  const int64_t n_vec = n / width;
  const int64_t tail_start = n_vec * width;
  const int threads = 256;

  if (n_vec > 0) {
    const int blocks = static_cast<int>((n_vec + threads - 1) / threads);
    gelu_forward_kernel_vec<scalar_t><<<blocks, threads>>>(x, y, n_vec);
  }
  const int64_t tail_n = n - tail_start;
  if (tail_n > 0) {
    const int tail_blocks = static_cast<int>((tail_n + threads - 1) / threads);
    gelu_forward_kernel_tail<scalar_t><<<tail_blocks, threads>>>(x, y, tail_start, n);
  }
}

template <typename scalar_t>
void gelu_backward_launcher_optimized(const scalar_t* grad_out, const scalar_t* x,
                                       scalar_t* grad_in, int64_t n) {
  if (!is_aligned<scalar_t>(grad_out) || !is_aligned<scalar_t>(x) || !is_aligned<scalar_t>(grad_in)) {
    gelu_backward_launcher<scalar_t>(grad_out, x, grad_in, n);  // scalar fallback
    return;
  }

  constexpr int width = VecTraits<scalar_t>::width;
  const int64_t n_vec = n / width;
  const int64_t tail_start = n_vec * width;
  const int threads = 256;

  if (n_vec > 0) {
    const int blocks = static_cast<int>((n_vec + threads - 1) / threads);
    gelu_backward_kernel_vec<scalar_t><<<blocks, threads>>>(grad_out, x, grad_in, n_vec);
  }
  const int64_t tail_n = n - tail_start;
  if (tail_n > 0) {
    const int tail_blocks = static_cast<int>((tail_n + threads - 1) / threads);
    gelu_backward_kernel_tail<scalar_t><<<tail_blocks, threads>>>(grad_out, x, grad_in, tail_start,
                                                                   n);
  }
}

template void gelu_forward_launcher_optimized<float>(const float*, float*, int64_t);
template void gelu_forward_launcher_optimized<double>(const double*, double*, int64_t);
template void gelu_backward_launcher_optimized<float>(const float*, const float*, float*, int64_t);
template void gelu_backward_launcher_optimized<double>(const double*, const double*, double*,
                                                         int64_t);
