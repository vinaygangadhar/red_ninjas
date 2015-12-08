set(CUDA_PROFILE 1)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_20,code=sm_20 --maxrregcount=20)

cuda_add_executable(scan  scan_largearray.h scan_gold.cpp scan_kernel.cuh scan_largearray.cu)


