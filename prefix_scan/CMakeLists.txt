# Minimum version of CMake required. Don't touch.
cmake_minimum_required(VERSION 2.8)

# Include macros from the SBEL utils library
include(SBELUtils.cmake)

##########################################################
# If using Makefiles, be sure to set CMAKE_BUILD_TYPE to # 
# select different build configurations.                 #
# Possible values:                                       #
# - RELEASE - fully optimized for speed                  #
# - DEBUG - unoptimized, with debugging symbols          #
# - RELWITHDEBUGINFO - optimized, with debugging symbols #
# - MINSIZEREL - optimized for size                      #
# - [not set] - unoptimized                              #
##########################################################


#enable_mpi_support()
enable_openmp_support()
enable_cuda_support()

cuda_add_executable(scan scan_gold.cpp  scan_largearray.cu)
