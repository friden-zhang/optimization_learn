#include <gtest/gtest.h>
#include <thrust/device_vector.h>

#include <random>

#include "layernorm/cpu_layernorm.hpp"
#include "layernorm/cuda_layernorm.hpp"

static std::vector<float> generate_random_data(int N, int M) {
    std::vector<float> data(N * M);
    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_real_distribution<float> dis(0.0f, 10.0f);

    for (int i = 0; i < N * M; ++i) {
        data[i] = dis(gen);
    }

    return data;
}

TEST(LayerNormTest, CorrectTest) {
    int N = 32;
    int M = 32;

    auto generated_data = generate_random_data(N, M);

    auto cpu_output = std::vector<float>(N * M);

    layernorm::cpu_layernorm<float, float>(generated_data.data(), cpu_output.data(), nullptr, nullptr, N, M);

    thrust::device_vector<float> input_dev(generated_data.data(), generated_data.data() + N * M);
    thrust::device_vector<float> output_dev(generated_data.data(), generated_data.data() + N * M);

    float* input_dev_raw_ptr = thrust::raw_pointer_cast(input_dev.data());
    float* output_dev_raw_ptr = thrust::raw_pointer_cast(output_dev.data());

    layernorm::lauch_layernorm_kernel<float, float>(input_dev_raw_ptr, output_dev_raw_ptr, nullptr, nullptr, N, M);

    std::vector<float> cuda_output_host(N * M);
    thrust::copy(output_dev.begin(), output_dev.end(), cuda_output_host.begin());

    float sqrt = 0;
    float diff = 0;

    for (int i = 0; i < N * M; ++i) {
        sqrt += cuda_output_host[i] * cuda_output_host[i];
        diff += std::abs(cuda_output_host[i] - cpu_output[i]);
    }

    auto error = std::sqrt(diff / sqrt);

    EXPECT_LT(error, 1e-3);
}