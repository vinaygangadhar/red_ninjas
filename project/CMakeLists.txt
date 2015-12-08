# Set the required version of CMake
cmake_minimum_required(VERSION 2.8)
# Set your project title
project(me759_homework)

#The following will but all binary files in a "bin" directory
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_BUILD_TYPE RELEASE)
#set(CMAKE_BUILD_TYPE DEBUG)

##Look for CUDA package and related by calling macros in Parallelutils
include (ParallelUtils.cmake)

##Enable the CUDA Support
enable_cuda_support()

##Enable the OpenMP Support
enable_openmp_support()

##Enable the MPI Support
#enable_mpi_support()

#Add subdirectories for each homework
add_subdirectory(nearest_neighbor)
add_subdirectory(prefix_scan)
add_subdirectory(runcascade)
add_subdirectory(integral_image)

