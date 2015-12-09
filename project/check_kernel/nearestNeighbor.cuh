#include "haar.h"
#include "image.h"
#include <stdio.h>
#include "stdio-wrapper.h"
#include "cuda_util.h"

#include "nearestNeighbor_kernel.cuh"

////DEBUG Varibales
//#ifdef LOG
//      static const bool PRINT_LOG = true;
//#else
//      static const bool PRINT_LOG = false;
//#endif
//
//#ifdef DEVICE
//      static const bool PRINT_GPU = true;
//#else
//      static const bool PRINT_GPU = false;
//#endif



//Setting up the kernel for device
void nearestNeighborOnDevice(MyImage *src, MyImage *dst)
{

   //Image Characteristics
   int w1 = src->width;
   int h1 = src->height;
   int w2 = dst->width;
   int h2 = dst->height;

   unsigned char* src_data = src->data;
   unsigned char* dst_data = dst->data;
      
   //Allocate Device Array for dst image in gloabl memory for downsampled image
   int check;

   char *deviceSrcImage, *deviceDstImage;
   int imageDstSize = dst->width * dst->height;
   int imageSrcSize = src->width * src->height;

   if(PRINT_LOG){
       printf("\tSrc size: %d x %d\n", src->width, src->height);
       printf("\tDst size: %d x %d\n", dst->width, dst->height);
   }

   //Pinned memory
   check = CUDA_CHECK_RETURN(cudaMalloc((void**)&deviceSrcImage, imageSrcSize));
   if( check != 0){
        std::cerr << "Error: CudaMallocHost not successfull for device source image" << std::endl;
        exit(1);
    }
   
   check = CUDA_CHECK_RETURN(cudaMalloc((void**)&deviceDstImage, imageDstSize));
   if( check != 0){
        std::cerr << "Error: CudaMallocHost not successfull for device dest image" << std::endl;
        exit(1);
    }

   //Copy the source image to device
   check = CUDA_CHECK_RETURN(cudaMemcpy(deviceSrcImage, src_data, sizeof(char) * imageSrcSize, cudaMemcpyHostToDevice));      
   if( check != 0){
       std::cerr << "Error: CudaMemCpy from Host To Device Failed" << std::endl;
       exit(1);
   }

   //Get the scaling ratio of src and dest image
   int x_ratio = (int)((w1<<16)/w2) +1;
   int y_ratio = (int)((h1<<16)/h2) +1;

   //Execution Configuration
   //dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);  
   //dim3 blocksPerGrid( (dst->width + BLOCK_SIZE - 1)/BLOCK_SIZE, (dst->height + BLOCK_SIZE - 1)/BLOCK_SIZE );
     
   int threadsPerBlock = getSmallestPower2(w2) ;  
   int blocksPerGrid = h2;

   //GPU CALL
   nn_kernel<<<blocksPerGrid, threadsPerBlock/2>>>(deviceSrcImage, deviceDstImage, 
                                                 w1, h1, w2, h2,
                                                 x_ratio, y_ratio, imageDstSize);
   if(PRINT_GPU){
      std::cerr << "\t/***************GPU-LOG****************/" << std::endl;
      std::cerr << "\tThreads Per Block: " << threadsPerBlock/2 << std::endl;
      std::cerr << "\tTBS per Grid: " << blocksPerGrid << std::endl;
      std::cerr << "\t/**************************************/" << std::endl;
   }

   //Copy the dst image result back to host ptr
   check = CUDA_CHECK_RETURN(cudaMemcpy(dst_data, deviceDstImage, sizeof(char) * imageDstSize, cudaMemcpyDeviceToHost));      
   if( check != 0){
       std::cerr << "Error: CudaMemCpy from Host To Device Failed" << std::endl;
       exit(1);
   }

   cudaFree(deviceSrcImage);
   cudaFree(deviceDstImage);
}

