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

int freeImagePinned(MyImage* image)
{
	if (image->flag == 0)
	{
		printf("no image to delete\n");
		return -1;
	}
	else
	{
//		printf("image deleted\n");
		cudaFreeHost(image->data); 
		return 0;
	}
}


///Functions with int16_t 
void createSumImage16_t(int width, int height, MyIntImage16_t *image)
{
	image->width = width;
	image->height = height;
	image->flag = 1;
	image->data = (int16_t *)malloc(sizeof(int16_t)*(height*width));
}

int freeSumImage16_t(MyIntImage16_t* image)
{
	if (image->flag == 0)
	{
		printf("no image to delete\n");
		return -1;
	}
	else
	{
//		printf("image deleted\n");
		free(image->data); 
		return 0;
	}
}

void setSumImage16_t(int width, int height, MyIntImage16_t *image)
{
	image->width = width;
	image->height = height;
}
