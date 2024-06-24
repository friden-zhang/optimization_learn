#pragma once

#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

namespace utilities {

template <typename T>
static std::vector<std::vector<T>> transpose_matrix(const std::vector<std::vector<T>>& data) {
    std::vector<std::vector<T>> result(data[0].size(), std::vector<T>(data.size()));
    for (size_t i = 0; i < data.size(); i++) {
        for (size_t j = 0; j < data[0].size(); j++) {
            result[j][i] = data[i][j];
        }
    }
    return result;
}
}  // namespace utilities