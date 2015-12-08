set(CUDA_PROFILE 1)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_20,code=sm_20 --maxrregcount=20)

cuda_add_executable(ii  stdio-wrapper.h image.h haar.h gpu_integral.cuh main.cpp image.c rectangles.cpp stdio-wrapper.c haar.cu)


