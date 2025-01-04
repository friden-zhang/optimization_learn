#pragma once

namespace layernorm {

/**
 * @brief input is [N,M] tensor, output is [N,M] tensor, we layernorm each feature of each sample. which mean we calculate mean and var according M
 * dimension
 *
 * @tparam T: the data type of input and output
 * @tparam U: the data type of middle calculation
 * @param input: the input tensor
 * @param output: the output tensor
 * @param alpha: the scale parameter
 * @param beta: the bias parameter
 * @param N: the number of samples in the input tensor
 * @param M: the number of features in the input tensor
 */
template <typename T, typename U = T>
bool lauch_layernorm_kernel(const T* input, T* output, const T* alpha, const T* beta, int N, int M);

}  // namespace layernorm