#include "haar.h"
#include "image.h"
#include <stdio.h>
#include "stdio-wrapper.h"
#include "cuda_util.h"

#include "gpu_nn_integral_kernel.cuh"
#include "gpu_transpose_kernel.cuh"

//Allocation for device img
void nniiAllocateImgOnDevice(unsigned char** d_img, int srcSize){

    int check;
    //Allocate device src
    
    check = CUDA_CHECK_RETURN(cudaMalloc((void**)d_img, sizeof(unsigned char) * srcSize), __FILE__, __LINE__);
    if( check != 0){
          std::cerr << "Error: CudaMalloc not successfull for device source image" << std::endl;
          exit(1);
    }
}

//Allocate sum and sqsum and its tranpsoes on device
void nniiAllocateOnDevice(int32_t** d_sum, int32_t** d_sqsum, 
                           int** transpose_dsum, int** transpose_dsqsum , 
                           int dstSize
                          )
{
    
    // ALLOCATION FOR SUM/SQSUM IMAGES   //
    int check;
    
    //Malloc for sum and sqsum 
    check = CUDA_CHECK_RETURN(cudaMalloc((void**)d_sum, sizeof(int32_t) * dstSize), __FILE__, __LINE__);
    if( check != 0){
          std::cerr << "Error: CudaMalloc not successfull for device dest sum image" << std::endl;
          exit(1);
    }

    check = CUDA_CHECK_RETURN(cudaMalloc((void**)d_sqsum, sizeof(int32_t) * dstSize), __FILE__, __LINE__);
    if( check != 0){
          std::cerr << "Error: CudaMalloc not successfull for device dest sqsum image" << std::endl;
          exit(1);
    }

   //Malloc for transpose sum and sqsum 
   check = CUDA_CHECK_RETURN(cudaMalloc((void**)transpose_dsum, sizeof(int32_t) * dstSize), __FILE__, __LINE__);
   if( check != 0){
        std::cerr << "Error: CudaMalloc not successfull for device transpose sum image" << std::endl;
        exit(1);
   }

   check = CUDA_CHECK_RETURN(cudaMalloc((void**)transpose_dsqsum, sizeof(int32_t) * dstSize), __FILE__, __LINE__);
   if( check != 0){
        std::cerr << "Error: CudaMalloc not successfull for device tranpsoe sqsum image" << std::endl;
        exit(1);
   }

}

//Copy the src image to device
void nniiCopyImgToDevice(unsigned char* srcimg, unsigned char* dstimg, int srcSize){

   int check;
   //Copy source contents to a src on device
   
   check = CUDA_CHECK_RETURN(cudaMemcpy(dstimg, srcimg, sizeof(unsigned char) * srcSize, cudaMemcpyHostToDevice), __FILE__, __LINE__);
   //if( check != 0){
   //   std::cerr << "Error: CudaMemCpy not successfull for device source image" << std::endl;
   //   exit(1);
   //}
}

//Setting up the kernel for device -- 32bit version
void nn_integralImageOnDevice(MyImage *src, unsigned char* deviceimg, 
                              int32_t *d_sum, int32_t *d_sqsum,
                              int32_t *transpose_dsum, int32_t *transpose_dsqsum,
                              int dstWidth, int dstHeight)
{
     /**************************************/
     //Timing related
     cudaError_t error;
     cudaEvent_t gpu_exc_start;
     cudaEvent_t gpu_exc_stop;
     float exc_msecTotal;
   
     float mt2plusrs_excmsecTotal;
     float gpu_excmsecTotal;
    
     //CUDA Events 
     error = cudaEventCreate(&gpu_exc_start);
     if (error != cudaSuccess)
     {
         fprintf(stderr, "Failed to create start event (error code %s)!\n", cudaGetErrorString(error));
         exit(EXIT_FAILURE);
     }
     
     error = cudaEventCreate(&gpu_exc_stop);
     if (error != cudaSuccess)
     {
         fprintf(stderr, "Failed to create stop event (error code %s)!\n", cudaGetErrorString(error));
         exit(EXIT_FAILURE);
     
     }

     /**************************************/
     //Image Characteristics
     int src_w = src->width;
     int src_h = src->height;
     int dst_w = dstWidth;
     int dst_h = dstHeight;

     printf("\n\tNN and II on GPU Started\n");

     if(PRINT_LOG){
        printf("\tSrc size: %d x %d\n", src->width, src->height);
        printf("\tDst size: %d x %d\n", dst_w, dst_h);
     }
      
     //Get the scaling ratio of src and dest image
     int x_ratio = (int)((src_w<<16)/dst_w) +1;
     int y_ratio = (int)((src_h<<16)/dst_h) +1;
 
     // Execution Configuration for Orig ROW SCAN //
     int threadsPerBlock_rs = getSmallestPower2(dst_w);
     int threadsPerBlock_cs = getSmallestPower2(dst_h);
     
     int blocksPerGrid_rs = dst_h;
     int blocksPerGrid_cs = dst_w;
     
     if (threadsPerBlock_rs > 1024 || threadsPerBlock_cs > 1024)
     {
       printf("\tII: Supported only for Downsample Image width & height < 1024\n");
       printf("\tII: Currently passed Downsampled Image[w]: %d Image[h]: %d\n", dst_w, dst_h);
       
       cudaFree(deviceimg);
       cudaFree(d_sum);
       cudaFree(d_sqsum);
     }

     if(PRINT_GPU){
        std::cerr << "\t/***************GPU-LOG****************/" << std::endl;                              
        std::cerr << "\tThreads Per Block for RowScan: " << threadsPerBlock_rs << std::endl;
        std::cerr << "\tTBS per Grid for RowScan: " << blocksPerGrid_rs << std::endl;         
        std::cerr << "\tThreads Per Block for ColumnScan: " << threadsPerBlock_cs << std::endl;
        std::cerr << "\tTBS per Grid for ColumnScan: " << blocksPerGrid_cs << std::endl;         
        std::cerr << "\t/**************************************/" << std::endl;                              
     }                                                                                                       

      /*****************************************************************
     *rowscan does row-wise inclusive prefix scan for sum and sqsum
     *colscan does column-wise inclusive prefix scan for sum and sqsum*/

      //////////////////////////
     // ORIG ROW SCAN KERNEL //
     /////////////////////////
     int check;

     //--Exclusive Timing Only Start

     error = cudaEventRecord(gpu_exc_start, NULL);
     if (error != cudaSuccess)
     {
         fprintf(stderr, "Failed to record start event (error code %s)!\n", cudaGetErrorString(error));
         exit(EXIT_FAILURE);
     }

     //*******************************************//
     //Neareast Neighbor and Row Scan Kernel Call //
     //*******************************************//
     rowscan_nn_kernel<<<blocksPerGrid_rs, (threadsPerBlock_rs / 2)>>>(
                                                                        deviceimg,
                                                                        d_sum, d_sqsum, 
                                                                        src_w, src_h, dst_w, dst_h,
                                                                        x_ratio, y_ratio, 
                                                                        threadsPerBlock_rs
                                                                       );
     
     check = CUDA_CHECK_RETURN( cudaPeekAtLastError(), __FILE__, __LINE__ );
     if( check != 0){
           std::cerr << "Error: CudaPeek on Row Scan not successfull for device dest image" << std::endl;
           exit(1);
     }
     
     check = CUDA_CHECK_RETURN( cudaDeviceSynchronize(), __FILE__, __LINE__ );
     if( check != 0){
           std::cerr << "Error: CudaSynchronize not successfull for device dest image" << std::endl;
           exit(1);
     }    

     // Record the stop event
     error = cudaEventRecord(gpu_exc_stop, NULL);
     if (error != cudaSuccess)
     {
         fprintf(stderr, "Failed to record stop event (error code %s)!\n", cudaGetErrorString(error));
         exit(EXIT_FAILURE);
     }

     // Wait for the stop event to complete
     error = cudaEventSynchronize(gpu_exc_stop);
     if (error != cudaSuccess)
     {
         fprintf(stderr, "Failed to synchronize on the stop event (error code %s)!\n", cudaGetErrorString(error));
         exit(EXIT_FAILURE);
     }

     error = cudaEventElapsedTime(&exc_msecTotal, gpu_exc_start, gpu_exc_stop);
     if (error != cudaSuccess)
     {
         fprintf(stderr, "Failed to get time elapsed between events (error code %s)!\n", cudaGetErrorString(error));
         exit(EXIT_FAILURE);
     }

     printf("\tNN: Rowscan Done on GPU-->: Exclusive Time: %f ms\n", exc_msecTotal);
     gpu_excmsecTotal = exc_msecTotal;


      /////////////////////////////////
     //  MATRIX TRANSPOSE 1 KERNEL  //
     ////////////////////////////////

     // Execution Configuration for  Matrix Transpose 1//
     int tx = NN_II_BLOCK_SIZE;
     int ty = NN_II_BLOCK_SIZE;
     int bx = (dst_w + NN_II_BLOCK_SIZE - 1)/NN_II_BLOCK_SIZE;
     int by = (dst_h + NN_II_BLOCK_SIZE - 1)/NN_II_BLOCK_SIZE;
     
     dim3 blocks(tx,ty);
     dim3 grid(bx,by);

     error = cudaEventRecord(gpu_exc_start, NULL);
     if (error != cudaSuccess)
     {
          fprintf(stderr, "Failed to record start event (error code %s)!\n", cudaGetErrorString(error));
          exit(EXIT_FAILURE);
     }
 
     //**********************************//
     // matrix transpose 1  Kernel Call //
     //********************************//
     transpose_kernel<<<grid,blocks>>>(d_sum, transpose_dsum,  
                                       d_sqsum, transpose_dsqsum, 
                                       dst_w, dst_h);
     
     check = CUDA_CHECK_RETURN( cudaPeekAtLastError(), __FILE__, __LINE__ );
     if( check != 0){
          std::cerr << "Error: CudaPeek on Row Scan not successfull for device dest image" << std::endl;
          exit(1);
     }

     check = CUDA_CHECK_RETURN( cudaDeviceSynchronize(), __FILE__, __LINE__ );
     if( check != 0){
          std::cerr << "Error: CudaSynchronize not successfull for device dest image" << std::endl;
          exit(1);
     }    

     // Record the stop event
     error = cudaEventRecord(gpu_exc_stop, NULL);
     if (error != cudaSuccess)
     {
          fprintf(stderr, "Failed to record stop event (error code %s)!\n", cudaGetErrorString(error));
          exit(EXIT_FAILURE);
     }

     // Wait for the stop event to complete
     error = cudaEventSynchronize(gpu_exc_stop);
     if (error != cudaSuccess)
     {
        fprintf(stderr, "Failed to synchronize on the stop event (error code %s)!\n", cudaGetErrorString(error));
          exit(EXIT_FAILURE);
     }

     error = cudaEventElapsedTime(&exc_msecTotal, gpu_exc_start, gpu_exc_stop);
     if (error != cudaSuccess)
     {
        fprintf(stderr, "Failed to get time elapsed between events (error code %s)!\n", cudaGetErrorString(error));
          exit(EXIT_FAILURE);
     }

     printf("\tII: Matrix Transpose1 Done on GPU--> Exclusive Time: %f ms\n", exc_msecTotal);
     gpu_excmsecTotal += exc_msecTotal;
     mt2plusrs_excmsecTotal = exc_msecTotal;

      
     /////////////////////////////////////
     // ROW SCAN ONLY (w/o NN) KERNEL  //
     ////////////////////////////////////

     error = cudaEventRecord(gpu_exc_start, NULL);
     if (error != cudaSuccess)
     {
          fprintf(stderr, "Failed to record start event (error code %s)!\n", cudaGetErrorString(error));
          exit(EXIT_FAILURE);
     }

      //***************************//
     // row scan only Kernel Call //
     //**************************//
     rowscan_only_kernel<<<blocksPerGrid_cs, (threadsPerBlock_cs / 2)>>>(transpose_dsum, 
                                                                  transpose_dsqsum, 
                                                                  dst_h, threadsPerBlock_cs);
     
     check = CUDA_CHECK_RETURN( cudaPeekAtLastError(), __FILE__, __LINE__ );
     if( check != 0){
          std::cerr << "Error: CudaPeek on Row Scan not successfull for device dest image" << std::endl;
          exit(1);
     }

     check = CUDA_CHECK_RETURN( cudaDeviceSynchronize(), __FILE__, __LINE__ );
     if( check != 0){
          std::cerr << "Error: CudaSynchronize not successfull for device dest image" << std::endl;
          exit(1);
     }    

     // Record the stop event
     error = cudaEventRecord(gpu_exc_stop, NULL);
     if (error != cudaSuccess)
     {
          fprintf(stderr, "Failed to record stop event (error code %s)!\n", cudaGetErrorString(error));
          exit(EXIT_FAILURE);
     }

     // Wait for the stop event to complete
     error = cudaEventSynchronize(gpu_exc_stop);
     if (error != cudaSuccess)
     {
          fprintf(stderr, "Failed to synchronize on the stop event (error code %s)!\n", cudaGetErrorString(error));
          exit(EXIT_FAILURE);
     }

     error = cudaEventElapsedTime(&exc_msecTotal, gpu_exc_start, gpu_exc_stop);
     if (error != cudaSuccess)
     {
          fprintf(stderr, "Failed to get time elapsed between events (error code %s)!\n", cudaGetErrorString(error));
          exit(EXIT_FAILURE);
     }

     printf("\tII: RowScan Only on GPU Done--> Exclusive Time: %f ms\n", exc_msecTotal);
     gpu_excmsecTotal += exc_msecTotal;
     mt2plusrs_excmsecTotal += exc_msecTotal;
     
      /////////////////////////////////
     //  MATRIX TRANSPOSE 2 KERNEL  //
     ////////////////////////////////
   
     // Execution Configuration for  Matrix Transpose 2//
     bx = (dst_h + NN_II_BLOCK_SIZE - 1)/NN_II_BLOCK_SIZE;
     by = (dst_w + NN_II_BLOCK_SIZE)/NN_II_BLOCK_SIZE;
     dim3 blocks2(tx,ty);
     dim3 grid2(bx,by);
     
     error = cudaEventRecord(gpu_exc_start, NULL);
     if (error != cudaSuccess)
     {
          fprintf(stderr, "Failed to record start event (error code %s)!\n", cudaGetErrorString(error));
          exit(EXIT_FAILURE);
     }

     //**********************************//
     // matrix transpose 2  Kernel Call //
     //********************************//
     transpose_kernel<<<grid2, blocks2>>>(transpose_dsum, d_sum, 
                                          transpose_dsqsum, d_sqsum, 
                                          dst_h, dst_w);

     check = CUDA_CHECK_RETURN( cudaPeekAtLastError(), __FILE__, __LINE__ );
     if( check != 0){
          std::cerr << "Error: CudaPeek on Row Scan not successfull for device dest image" << std::endl;
          exit(1);
     }

     check = CUDA_CHECK_RETURN( cudaDeviceSynchronize(), __FILE__, __LINE__ );
     if( check != 0){
          std::cerr << "Error: CudaSynchronize not successfull for device dest image" << std::endl;
          exit(1);
     }    

     // Record the stop event
     error = cudaEventRecord(gpu_exc_stop, NULL);
     if (error != cudaSuccess)
     {
          fprintf(stderr, "Failed to record stop event (error code %s)!\n", cudaGetErrorString(error));
          exit(EXIT_FAILURE);
     }

     // Wait for the stop event to complete
     error = cudaEventSynchronize(gpu_exc_stop);
     if (error != cudaSuccess)
     {
          fprintf(stderr, "Failed to synchronize on the stop event (error code %s)!\n", cudaGetErrorString(error));
          exit(EXIT_FAILURE);
     }

     error = cudaEventElapsedTime(&exc_msecTotal, gpu_exc_start, gpu_exc_stop);
     if (error != cudaSuccess)
     {
          fprintf(stderr, "Failed to get time elapsed between events (error code %s)!\n", cudaGetErrorString(error));
          exit(EXIT_FAILURE);
     }

     printf("\tII:Matrix Transpose2 Done on GPU- Exclusive Time: %f ms\n", exc_msecTotal);
     gpu_excmsecTotal += exc_msecTotal;
     mt2plusrs_excmsecTotal += exc_msecTotal;
     
     
     //printf("\t2 Matrix Transposes  + RowScan Only Exclusive Time: %f ms\n", mt2plusrs_excmsecTotal);
     printf("\tNN and II on GPU complete--> Combined Exclusive Time: %f ms\n", gpu_excmsecTotal);

     //Destroy Events
     cudaEventDestroy(gpu_exc_start);
     cudaEventDestroy(gpu_exc_stop);
     
}


//Free resources on GPU
void nniiFree(int32_t *d_sum, int32_t *d_sqsum,
              int32_t *transpose_dsum, int32_t *transpose_dsqsum
             )
{

     cudaFree(d_sum);
     cudaFree(d_sqsum);
     cudaFree(transpose_dsum);
     cudaFree(transpose_dsqsum);
}


//Free the Device image
void nniiFreeImg(unsigned char* deviceimg){
   
     cudaFree(deviceimg);

}

