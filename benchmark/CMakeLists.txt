

find_package(benchmark REQUIRED)
include_directories(${benchmark_INCLUDE_DIRS})

aux_source_directory(. BENCHMARK_SRCS)

add_executable(bench_all ${BENCHMARK_SRCS})
target_link_libraries(bench_all benchmark::benchmark optimization_cuda_lib)
link_tbb_to_target(bench_all)