

add_library(optimization_cuda_lib cuda_gemm.cu)
set_target_properties(optimization_cuda_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(optimization_cuda_lib ${CUDA_LIBRARIES})
