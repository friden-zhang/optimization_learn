

find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})


add_executable(test_all test_gemm.cc test_generate_matrix.cc test_all.cc)

target_link_libraries(test_all ${GTEST_LIBRARIES} pthread optimization_cuda_lib)
link_tbb_to_target(test_all)