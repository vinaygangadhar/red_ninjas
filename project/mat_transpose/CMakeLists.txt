#set the required version of CMake
cmake_minimum_required(VERSION 2.8)
# Set your project title
project(ME759_FINAL_PROJ)
find_package(CUDA QUIET REQUIRED)
set(CUDA_NVCC_FLAGS -O3 -arch=compute_20 -code=sm_20)
#The following will but all binary files in a "bin" directory
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})

#Add subdirectories for each homework
message("cmake_bin_dir: ${CMAKE_BINARY_DIR}")
message("cmake_current_bin_dir: ${CMAKE_CURRENT_BINARY_DIR}")
message("cmake_source_dir: ${CMAKE_SOURCE_DIR}")
message("cmake_current_source_dir: ${CMAKE_CURRENT_SOURCE_DIR}")
cuda_add_executable(transpose_ii  stdio-wrapper.h image.h haar.h gpu_integral.cuh gpu_transpose_kernel.cuh gpu_integral_kernel.cuh main.cpp image.c rectangles.cpp stdio-wrapper.c haar.cu)
