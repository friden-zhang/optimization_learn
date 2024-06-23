

find_package(benchmark REQUIRED)
include_directories(${benchmark_INCLUDE_DIRS})

add_executable(bench_all bench_gemm.cc)
target_link_libraries(bench_all benchmark::benchmark)
link_tbb_to_target(bench_all)