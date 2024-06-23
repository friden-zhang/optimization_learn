#pragma once

#include <array>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>
namespace utilities {

template <typename T, size_t N, typename SizeType = int>
auto generate_random_matrix(T lower_bound, T upper_bound, const std::array<SizeType, N>& sizes) {
    static_assert(N > 0, "N must be greater than 0");
    auto size = sizes[0];

    if constexpr (N > 1) {
        std::array<SizeType, N - 1> next_sizes;
        std::copy(sizes.begin() + 1, sizes.end(), next_sizes.begin());
        std::vector<decltype(generate_random_matrix<T, N - 1>(lower_bound, upper_bound,
                                                              next_sizes))>
            array(size);
        for (int i = 0; i < size; ++i) {
            array[i] = generate_random_matrix<T, N - 1>(lower_bound, upper_bound, next_sizes);
        }
        return array;
    } else {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<T> dis(lower_bound, upper_bound);
        std::vector<T> array(size);
        for (int i = 0; i < size; ++i) {
            array[i] = dis(gen);
        }
        return array;
    }
}

}  // namespace utilities