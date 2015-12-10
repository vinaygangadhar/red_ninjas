#include "haar.h"
#include "image.h"
#include <stdio.h>
#include "stdio-wrapper.h"
#include "cuda_util.h"

#include "gpu_integral_kernel.cuh"
#include "gpu_transpose_kernel.cuh"

//CUDA Error Checker -- If return value is -1 then there is an error
//#define CUDA_CHECK_RETURN(func) {cuda_check((func),__FILE__,__LINE__);}

int CUDA_CHECK_RETURN(cudaError_t err_ret, const char* file, int line)
{
     int val = 0;
     if (err_ret != cudaSuccess) {
          fprintf(stderr, "Error %s at line %d in file %s\n", cudaGetErrorString(err_ret), file, line); 
          val = 1;
     }
     return val;    
}                                                                                                             

//Setting up the kernel for device
void nn_integralImageOnDevice(MyImage *src, MyImage *dst, MyIntImage *sum, MyIntImage *sqsum )
{
     /**************************************/
     //Timing related
     cudaError_t error;
     cudaEvent_t gpu_inc_start;
     cudaEvent_t gpu_inc_stop;
     cudaEvent_t gpu_exc_start;
     cudaEvent_t gpu_exc_stop;
     float inc_msecTotal;
     float exc_msecTotal;
     float gpu_excmsecTotal;
     float mt2plusrs_excmsecTotal;

     //CUDA Events 
     error = cudaEventCreate(&gpu_inc_start);
     if (error != cudaSuccess)
     {
          fprintf(stderr, "Failed to create start event (error code %s)!\n", cudaGetErrorString(error));
          exit(EXIT_FAILURE);
     }

     error = cudaEventCreate(&gpu_inc_stop);
     if (error != cudaSuccess)
     {
          fprintf(stderr, "Failed to create stop event (error code %s)!\n", cudaGetErrorString(error));
          exit(EXIT_FAILURE);

     }

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
     int dst_w = sum->width;
     int dst_h = sum->height;

     //Device Source Image
     MyImage device_srcimg;
     device_srcimg.height = src->height;
     device_srcimg.width =  src->width;
     int srcSize = device_srcimg.height * device_srcimg.width;

     //Downsample device image 
     MyImage device_nnimg;
     device_nnimg.height = dst->height;
     device_nnimg.width =  dst->width;
     int dstSize = device_nnimg.height * device_nnimg.width;

     int check = 0;

     printf("\n\tNN and II on GPU Started\n");

     //allocate device src
     check = CUDA_CHECK_RETURN(cudaMalloc((void**)&(device_srcimg.data), sizeof(unsigned char) * srcSize), __FILE__, __LINE__);
     if( check != 0){
          std::cerr << "Error: CudaMalloc not successfull for device source image" << std::endl;
          exit(1);
     }

     //allocate device dst
     check = CUDA_CHECK_RETURN(cudaMalloc((void**)&(device_nnimg.data), sizeof(unsigned char) * dstSize), __FILE__, __LINE__);
     if( check != 0){
          std::cerr << "Error: CudaMalloc not successfull for device dest image" << std::endl;
          exit(1);
     }

     //allocate space for sum and sqsum pixels only on device//
     MyIntImage d_sum, d_sqsum;
     d_sum.width = sum->width; d_sum.height = sum->height; 
     d_sqsum.width = sqsum->width; d_sqsum.height = sqsum->height;

     //Malloc for sum and sqsum 
     check = CUDA_CHECK_RETURN(cudaMalloc((void**)&(d_sum.data), sizeof(int)*dstSize), __FILE__, __LINE__);
     if( check != 0){
          std::cerr << "Error: CudaMalloc not successfull for device dest sum image" << std::endl;
          exit(1);
     }

     check = CUDA_CHECK_RETURN(cudaMalloc((void**)&(d_sqsum.data), sizeof(int)*dstSize), __FILE__, __LINE__);
     if( check != 0){
          std::cerr << "Error: CudaMalloc not successfull for device dest sqsum image" << std::endl;
          exit(1);
     }  

     if(PRINT_LOG){
          printf("\tSrc size: %d x %d\n", src->width, src->height);
          printf("\tDst size: %d x %d\n", sum->width, sum->height);
     }

     //Get the scaling ratio of src and dest image
     int x_ratio = (int)((src_w<<16)/dst_w) +1;
     int y_ratio = (int)((src_h<<16)/dst_h) +1;

     //Execution COnfiguration 
     int threadsPerBlock_rs = getSmallestPower2(dst_w);
     int threadsPerBlock_cs = getSmallestPower2(dst_h);

     int blocksPerGrid_rs = dst_h;
     int blocksPerGrid_cs = dst_w;

     if (threadsPerBlock_rs > 1024 || threadsPerBlock_cs > 1024)
     {
          printf("\tII: Supported only for Downsample Image width & height < 1024\n");
          printf("\tII: Currently passed Downsampled Image[w]: %d Image[h]: %d\n", dst_w, dst_h);
          cudaFree(device_srcimg.data);
          cudaFree(device_nnimg.data);
          cudaFree(d_sum.data);
          cudaFree(d_sqsum.data);
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


     //--Timing Start
     error = cudaEventRecord(gpu_inc_start, NULL);
     if (error != cudaSuccess)
     {
          fprintf(stderr, "Failed to record start event (error code %s)!\n", cudaGetErrorString(error));
          exit(EXIT_FAILURE);
     }

     //Copy source contents to a src on device
     check = CUDA_CHECK_RETURN(cudaMemcpy(device_srcimg.data, src->data, sizeof(unsigned char)*srcSize, cudaMemcpyHostToDevice), __FILE__, __LINE__);
     if( check != 0){
          std::cerr << "Error: CudaMemCpy not successfull for device source image" << std::endl;
          exit(1);
     }

     error = cudaEventRecord(gpu_exc_start, NULL);
     if (error != cudaSuccess)
     {
          fprintf(stderr, "Failed to record start event (error code %s)!\n", cudaGetErrorString(error));
          exit(EXIT_FAILURE);
     }

     //Neareast Neighbor and ROw Scan
     // rowscan_nn_kernel<<<blocksPerGrid_rs, (threadsPerBlock_rs / 2)>>>(device_srcimg.data, device_nnimg.data,                                                                   d_sum.data, d_sqsum.data, 
     //                                                                    src_w, src_h, dst_w, dst_h,
     //                                                                    x_ratio, y_ratio, 
     //                                                                    threadsPerBlock_rs);
     // 

     rowscan_nn_kernel<<<blocksPerGrid_rs, (threadsPerBlock_rs / 2)>>>(device_srcimg.data,                                                                    d_sum.data, d_sqsum.data, 
               src_w, src_h, dst_w, dst_h,
               x_ratio, y_ratio, 
               threadsPerBlock_rs);




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

     printf("\tII: Rowscan Done on GPU-%d: Exclusive Time: %f ms\n", __LINE__, exc_msecTotal);

     gpu_excmsecTotal = exc_msecTotal;
     
     //allocate space for transpose sum and sqsum pixels only on device//
     MyIntImage transposed_dsum, transposed_dsqsum;
     transposed_dsum.width = sum->width; transposed_dsum.height = sum->height; 
     transposed_dsqsum.width = sqsum->width; transposed_dsqsum.height = sqsum->height;

     //Malloc for sum and sqsum 
     check = CUDA_CHECK_RETURN(cudaMalloc((void**)&(transposed_dsum.data), sizeof(int)*dstSize), __FILE__, __LINE__);
     if( check != 0){
          std::cerr << "Error: CudaMalloc not successfull for device dest sum image" << std::endl;
          exit(1);
     }

     check = CUDA_CHECK_RETURN(cudaMalloc((void**)&(transposed_dsqsum.data), sizeof(int)*dstSize), __FILE__, __LINE__);
     if( check != 0){
          std::cerr << "Error: CudaMalloc not successfull for device dest sqsum image" << std::endl;
          exit(1);
     }  
     //matrix transpose for sum and sqsum before colscan
     int tx = 16;
     int ty = 16;
     int bx = (dst_w + 15)/16;
     int by = (dst_h + 15)/16;
     dim3 blk(tx,ty);
     dim3 grid(bx,by);
     
     error = cudaEventRecord(gpu_exc_start, NULL);
     if (error != cudaSuccess)
     {
          fprintf(stderr, "Failed to record start event (error code %s)!\n", cudaGetErrorString(error));
          exit(EXIT_FAILURE);
     }
 
     transpose_kernel<<<grid,blk>>>(d_sum.data,transposed_dsum.data,  d_sqsum.data, transposed_dsqsum.data, dst_w, dst_h);
     
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

     gpu_excmsecTotal += exc_msecTotal;
     mt2plusrs_excmsecTotal = exc_msecTotal;

     printf("\tII: Matrix Transpose1 Done on GPU- Exclusive Time: %f ms\n", exc_msecTotal);

     //Timing for column scan
     error = cudaEventRecord(gpu_exc_start, NULL);
     if (error != cudaSuccess)
     {
          fprintf(stderr, "Failed to record start event (error code %s)!\n", cudaGetErrorString(error));
          exit(EXIT_FAILURE);
     }

     //Column Scan
     //colscan<<<blocksPerGrid_cs, (threadsPerBlock_cs/2)>>>(d_sum.data, d_sqsum.data, dst_h, threadsPerBlock_cs);
     rowscan_only<<<blocksPerGrid_cs, (threadsPerBlock_cs/2)>>>(transposed_dsum.data, transposed_dsqsum.data, dst_h, threadsPerBlock_cs);
     
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

     gpu_excmsecTotal += exc_msecTotal;
     mt2plusrs_excmsecTotal += exc_msecTotal;
     
     printf("\tII: RowScan Only on Transpose1 Done on GPU- Exclusive Time: %f ms\n", exc_msecTotal);

    //launch transpose again
     tx = 16;
     ty = 16;
     bx = (dst_h + 15)/16;
     by = (dst_w + 15)/16;
     dim3 blk2(tx,ty);
     dim3 grid2(bx,by);
     
     error = cudaEventRecord(gpu_exc_start, NULL);
     if (error != cudaSuccess)
     {
          fprintf(stderr, "Failed to record start event (error code %s)!\n", cudaGetErrorString(error));
          exit(EXIT_FAILURE);
     }

     transpose_kernel<<<grid2, blk2>>>(transposed_dsum.data, d_sum.data, transposed_dsqsum.data, d_sqsum.data, dst_h, dst_w);

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

     gpu_excmsecTotal += exc_msecTotal;
     mt2plusrs_excmsecTotal += exc_msecTotal;
     
     printf("\tII: Matrix Transpose2 Done on GPU- Exclusive Time: %f ms\n", exc_msecTotal);

     
     //Copy back the sum from device
     check = CUDA_CHECK_RETURN(cudaMemcpy(sum->data, d_sum.data, sizeof(int)*dstSize, cudaMemcpyDeviceToHost), __FILE__, __LINE__);
     if( check != 0){
          std::cerr << "Error: CudaMemCpy not successfull for device dest sum image" << std::endl;
          exit(1);
     }

     //Copy back the sq_sum from device
     check = CUDA_CHECK_RETURN(cudaMemcpy(sqsum->data, d_sqsum.data, sizeof(int)*dstSize, cudaMemcpyDeviceToHost), __FILE__, __LINE__);
     if( check != 0){
          std::cerr << "Error: CudaMemcpy not successfull for device dest sqsum image" << std::endl;
          exit(1);
     }

     //     //Copy back the downsampled image to host
     //     check = CUDA_CHECK_RETURN(cudaMemcpy(dst->data, device_nnimg.data, sizeof(unsigned char)*dstSize, cudaMemcpyDeviceToHost), __FILE__, __LINE__);
     //     if( check != 0){
     //            std::cerr << "Error: CudaMemCpy not successfull for device dst image" << std::endl;
     //            exit(1);
     //     }

     // Record the stop event
     error = cudaEventRecord(gpu_inc_stop, NULL);
     if (error != cudaSuccess)
     {
          fprintf(stderr, "Failed to record stop event (error code %s)!\n", cudaGetErrorString(error));
          exit(EXIT_FAILURE);
     }

     // Wait for the stop event to complete
     error = cudaEventSynchronize(gpu_inc_stop);
     if (error != cudaSuccess)
     {
          fprintf(stderr, "Failed to synchronize on the stop event (error code %s)!\n", cudaGetErrorString(error));
          exit(EXIT_FAILURE);
     }

     error = cudaEventElapsedTime(&inc_msecTotal, gpu_inc_start, gpu_inc_stop);
     if (error != cudaSuccess)
     {
          fprintf(stderr, "Failed to get time elapsed between events (error code %s)!\n", cudaGetErrorString(error));
          exit(EXIT_FAILURE);
     }

     printf("\tII: Matrix Transpose2  + RowScan Only Exclusive Time: %f ms\n", mt2plusrs_excmsecTotal);
     printf("\tNN and II on GPU complete--> Combined Exclusive Time: %f ms, Total Inclusive time: %f ms\n", gpu_excmsecTotal, inc_msecTotal);

     //Destroy Events
     cudaEventDestroy(gpu_exc_start);
     cudaEventDestroy(gpu_exc_stop);
     cudaEventDestroy(gpu_inc_start);
     cudaEventDestroy(gpu_inc_stop);

     //Free resources
     cudaFree(device_srcimg.data);
     cudaFree(device_nnimg.data);
     cudaFree(d_sum.data);
     cudaFree(d_sqsum.data);



}

