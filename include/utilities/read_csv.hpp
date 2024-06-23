#pragma once

#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

namespace utilities {

template <typename T>
static std::vector<std::vector<T>> read_csv(const std::string& filename,
                                            const char delimiter = ',') {
    std::vector<std::vector<T>> data;
    std::ifstream file(filename);
    std::string line, cell;
    while (std::getline(file, line)) {
        std::stringstream line_stream(line);
        std::vector<T> row;
        while (getline(line_stream, cell, delimiter)) {
            row.push_back(stod(cell));
        }
        data.push_back(row);
    }
    return data;
}

}  // namespace utilities
