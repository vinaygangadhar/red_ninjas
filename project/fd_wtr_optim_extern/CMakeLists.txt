set(CUDA_PROFILE 1)
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_20,code=sm_20 -use_fast_math --maxrregcount=20)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_20,code=sm_20 -use_fast_math )
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_20,code=sm_20 -g -G --maxrregcount=20)

cuda_add_executable(fd_wtr_optim_extern stdio-wrapper.h image.h haar.h cuda_util.h gpu_nn_integral.cuh gpu_nn_integral_kernel.cuh gpu_transpose_kernel.cuh haar_stage.cuh haar_stage_kernel.cuh optim.cuh main.cpp image.c rectangles.cpp stdio-wrapper.c haar.cu)


