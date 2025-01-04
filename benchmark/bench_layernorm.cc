#include <benchmark/benchmark.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include <vector>
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

static void BM_cpu_layernorm(benchmark::State& state) {
    int N = state.range(0);
    int M = state.range(1);

    auto data = generate_random_data(N, M);

    auto output = std::vector<float>(N * M);

    for (auto _ : state) {
        layernorm::cpu_layernorm<float, float>(data.data(), output.data(), nullptr, nullptr, N, M);
    }
}

static void BM_cuda_layernorm(benchmark::State& state) {
    int N = state.range(0);
    int M = state.range(1);

    auto data = generate_random_data(N, M);

    thrust::device_vector<float> input_dev(data.data(), data.data() + N * M);
    thrust::device_vector<float> output_dev(data.data(), data.data() + N * M);

    float* input_dev_raw_ptr = thrust::raw_pointer_cast(input_dev.data());
    float* output_dev_raw_ptr = thrust::raw_pointer_cast(output_dev.data());

    for (auto _ : state) {
        layernorm::lauch_layernorm_kernel<float, float>(input_dev_raw_ptr, output_dev_raw_ptr, nullptr, nullptr, N, M);
    }
}


BENCHMARK(BM_cpu_layernorm)->Args({4096, 40960})->Unit(benchmark::kMillisecond)->Iterations(10);
BENCHMARK(BM_cuda_layernorm)->Args({4096, 40960})->Unit(benchmark::kMillisecond)->Iterations(10);
