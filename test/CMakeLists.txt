

find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

aux_source_directory(./ test_sources)

add_executable(test_all ${test_sources})

target_link_libraries(test_all ${GTEST_LIBRARIES} pthread optimization_cuda_lib)
link_tbb_to_target(test_all)