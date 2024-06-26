cmake_minimum_required(VERSION 3.10)
project(optimization_learn LANGUAGES CXX CUDA)


if(POLICY CMP0074)
    cmake_policy(SET CMP0074 NEW)
endif()

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -g")


find_package(OpenMP REQUIRED)
find_package(TBB REQUIRED)
find_package(CUDA REQUIRED)

if (OpenMP_CXX_FOUND)
    message(STATUS "Found OpenMP")
    message(STATUS "OpenMP_CXX_FLAGS: ${OpenMP_CXX_FLAGS}")
    message(STATUS "OpenMP_EXE_LINKER_FLAGS: ${OpenMP_EXE_LINKER_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else ()
    message(WARNING "OpenMP not found!")
endif ()

include_directories(${TBB_INCLUDE_DIRS})

# 链接 TBB
if(TBB_FOUND)
    message(STATUS "Found TBB")
    add_compile_options(${TBB_CXX_FLAGS})
    add_link_options(${TBB_LINK_FLAGS})
    message(STATUS "TBB_CXX_FLAGS: ${TBB_CXX_FLAGS}")
    message(STATUS "TBB_LINK_FLAGS: ${TBB_LINK_FLAGS}")
    function(link_tbb_to_target target)
    if(TBB_FOUND)
        target_link_libraries(${target} TBB::tbb)
    endif()
    endfunction()
else()
    message(WARNING "TBB not found!")
endif()


include_directories(include)


add_subdirectory(src/cuda)
add_subdirectory(test)
add_subdirectory(benchmark)


