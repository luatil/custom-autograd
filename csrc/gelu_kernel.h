#pragma once
#include <cstdint>

template <typename scalar_t>
void gelu_forward_launcher(const scalar_t* x, scalar_t* y, int64_t n);

template <typename scalar_t>
void gelu_backward_launcher(const scalar_t* grad_out, const scalar_t* x,
                             scalar_t* grad_in, int64_t n);
