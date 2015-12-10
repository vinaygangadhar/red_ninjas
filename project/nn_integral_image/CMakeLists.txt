set(CUDA_PROFILE 1)
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_20,code=sm_20 -g -G --maxrregcount=20)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_20,code=sm_20  --maxrregcount=20)

cuda_add_executable(nn_ii  stdio-wrapper.h image.h haar.h gpu_nn_integral.cuh gpu_nn_integral_kernel.cuh gpu_transpose_kernel.cuh optim.cuh image.c main.cpp rectangles.cpp stdio-wrapper.c haar.cu)


