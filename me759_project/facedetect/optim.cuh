#include "haar.h"
#include "image.h"
#include <stdio.h>
#include "stdio-wrapper.h"
#include "cuda_util.h"

////////////////////////////
//Optimization Functions //
//////////////////////////

void createImagePinned(int width, int height, MyImage *image)
{
	image->width = width;
	image->height = height;
	image->flag = 1;
   int check;

   check = CUDA_CHECK_RETURN(cudaMallocHost((void**)&(image->data), sizeof(unsigned char) * width * height), __FILE__, __LINE__);
   if( check != 0){
           std::cerr << "Error: CudaMallocHost not successfull for host source image" << std::endl;
           exit(1);
   }

}

//free the image pinned memory
int freeImagePinned(MyImage* image)
{
	if (image->flag == 0)
	{
		printf("no image to delete\n");
		return -1;
	}
	else
	{
		cudaFreeHost(image->data); 
		return 0;
	}
}

//bit vector creation on host
void createBitVectorPinned(bool** hvector, int width, int height){

   int check;
   check = CUDA_CHECK_RETURN(cudaMallocHost((void**)hvector, sizeof(bool) * width * height), __FILE__, __LINE__);
   if( check != 0){
           std::cerr << "Error: CudaMallocHost not successfull for host bit vector" << std::endl;
           exit(1);
   }
}

//Free the bit vector
void freeBitVectorPinned(bool* hvector){
   
   cudaFreeHost(hvector);
}
