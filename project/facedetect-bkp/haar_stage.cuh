#include "haar.h"
#include "image.h"
#include <stdio.h>
#include "stdio-wrapper.h"
#include "cuda_util.h"

#include "haar_stage_kernel.cuh"

//Setting up the kernel for device 
void haarAllocateOnDevice(uint16_t** dindex_x, uint16_t** dindex_y,
                          uint16_t** dwidth,   uint16_t** dheight,
                          int16_t** dweights_array, int16_t** dalpha1_array,
                          int16_t** dalpha2_array,  int16_t** dtree_thresh_array,
                          int16_t** dstages_thresh_array, 
                          int** dstages_array
                        )
{

  /****************************************************
   Setting up the data for GPU Kernels
  ***************************************************/
   int check;

   check = CUDA_CHECK_RETURN( cudaMalloc((void**)dindex_x, NUM_RECT * TOTAL_HAAR * sizeof(uint16_t)), __FILE__, __LINE__) ;
   if( check != 0){
      std::cerr << "Error: CudaMalloc not successfull for device dindex_x" << std::endl;
      exit(1);
   }
    
   check = CUDA_CHECK_RETURN( cudaMalloc((void**)dindex_y, NUM_RECT * TOTAL_HAAR * sizeof(uint16_t)), __FILE__, __LINE__) ;
   if( check != 0){
      std::cerr << "Error: CudaMalloc not successfull for device dindex_y" << std::endl;
      exit(1);
   }
   
   check = CUDA_CHECK_RETURN( cudaMalloc((void**)dwidth, NUM_RECT * TOTAL_HAAR * sizeof(uint16_t)) , __FILE__, __LINE__);
   if( check != 0){
      std::cerr << "Error: CudaMalloc not successfull for device dwidth" << std::endl;
      exit(1);
   }  

   check = CUDA_CHECK_RETURN( cudaMalloc((void**)dheight, NUM_RECT * TOTAL_HAAR * sizeof(uint16_t)), __FILE__, __LINE__ );
   if( check != 0){
      std::cerr << "Error: CudaMalloc not successfull for device dheight" << std::endl;
      exit(1);
   }

   check = CUDA_CHECK_RETURN( cudaMalloc((void**)dweights_array, NUM_RECT * TOTAL_HAAR * sizeof(int16_t)), __FILE__, __LINE__ );
   if( check != 0){
      std::cerr << "Error: CudaMalloc not successfull for device dweights" << std::endl;
      exit(1);
   }  

   check = CUDA_CHECK_RETURN( cudaMalloc((void**)dtree_thresh_array, TOTAL_HAAR * sizeof(int16_t)), __FILE__, __LINE__ );
   if( check != 0){
      std::cerr << "Error: CudaMalloc not successfull for device dtree_thresh" << std::endl;
      exit(1);
   }   

   check = CUDA_CHECK_RETURN( cudaMalloc((void**)dalpha1_array, TOTAL_HAAR * sizeof(int16_t)), __FILE__, __LINE__ );
   if( check != 0){
      std::cerr << "Error: CudaMalloc not successfull for device dalpha1" << std::endl;
      exit(1);
   } 
   
   check = CUDA_CHECK_RETURN( cudaMalloc((void**)dalpha2_array, TOTAL_HAAR * sizeof(int16_t)), __FILE__, __LINE__ );
   if( check != 0){
      std::cerr << "Error: CudaMalloc not successfull for device dalpha2" << std::endl;
      exit(1);
   }
 
   check = CUDA_CHECK_RETURN( cudaMalloc((void**)dstages_thresh_array, TOTAL_STAGES * sizeof(int16_t)), __FILE__, __LINE__);
   if( check != 0){
      std::cerr << "Error: CudaMalloc not successfull for device dstage_thresh" << std::endl;
      exit(1);
   }
   
   check = CUDA_CHECK_RETURN( cudaMalloc((void**)dstages_array, TOTAL_STAGES * sizeof(int)), __FILE__, __LINE__ );
   if( check != 0){
      std::cerr << "Error: CudaMalloc not successfull for device stages_array" << std::endl;
      exit(1);
   }
}


//Setting up the kernel data structures by copying from host 
void haarCopyToDevice(uint16_t* dindex_x, uint16_t* hindex_x,       
                          uint16_t* dindex_y, uint16_t* hindex_y,       
                          uint16_t* dwidth, uint16_t* hwidth,           
                          uint16_t* dheight, uint16_t* hheight,
                          int16_t* dweights_array, int16_t* hweights_array, 
                          int16_t* dalpha1_array, int16_t* halpha1_array,
                          int16_t* dalpha2_array, int16_t* halpha2_array,
                          int16_t* dtree_thresh_array, int16_t* htree_thresh_array,
                          int16_t* dstages_thresh_array, int16_t* hstages_thresh_array,
                          int* dstages_array, int* hstages_array
                     )
{
   int check;

   //MemCopy to device
   check = CUDA_CHECK_RETURN( cudaMemcpy(dindex_x, hindex_x, NUM_RECT * TOTAL_HAAR * sizeof(uint16_t), cudaMemcpyHostToDevice), __FILE__, __LINE__) ;
   //if( check != 0){
   //   std::cerr << "Error: CudaMemCpy not successfull for device dindex_x" << std::endl;
   //   exit(1);
   //}

   check = CUDA_CHECK_RETURN( cudaMemcpy(dindex_y, hindex_y, NUM_RECT*TOTAL_HAAR * sizeof(uint16_t), cudaMemcpyHostToDevice), __FILE__, __LINE__);
   //if( check != 0){
   //   std::cerr << "Error: CudaMemCpy not successfull for device dindex_y" << std::endl;
   //   exit(1);
   //}

   check = CUDA_CHECK_RETURN( cudaMemcpy(dwidth, hwidth, NUM_RECT*TOTAL_HAAR * sizeof(uint16_t), cudaMemcpyHostToDevice), __FILE__, __LINE__ );
   //if( check != 0){
   //   std::cerr << "Error: CudaMalloc not successfull for device dwidth" << std::endl;
   //   exit(1);
   //}

   check = CUDA_CHECK_RETURN( cudaMemcpy(dheight, hheight, NUM_RECT*TOTAL_HAAR * sizeof(uint16_t), cudaMemcpyHostToDevice), __FILE__, __LINE__ );
   //if( check != 0){
   //   std::cerr << "Error: CudaMalloc not successfull for device dheight" << std::endl;
   //   exit(1);
   //}

   check = CUDA_CHECK_RETURN( cudaMemcpy(dweights_array, hweights_array, NUM_RECT * TOTAL_HAAR*sizeof(int16_t), cudaMemcpyHostToDevice), __FILE__, __LINE__ );
   //if( check != 0){
   //   std::cerr << "Error: CudaMalloc not successfull for device dweights_array" << std::endl;
   //   exit(1);
   //}
   
   check = CUDA_CHECK_RETURN( cudaMemcpy(dtree_thresh_array, htree_thresh_array, TOTAL_HAAR * sizeof(int16_t), cudaMemcpyHostToDevice), __FILE__, __LINE__ );
   //if( check != 0){
   //   std::cerr << "Error: CudaMalloc not successfull for device dtree_thresh" << std::endl;
   //   exit(1);
   //}
   
   check = CUDA_CHECK_RETURN( cudaMemcpy(dalpha1_array, halpha1_array, TOTAL_HAAR * sizeof(int16_t), cudaMemcpyHostToDevice), __FILE__, __LINE__ );
   //if( check != 0){
   //   std::cerr << "Error: CudaMalloc not successfull for device dalpha1" << std::endl;
   //   exit(1);
   //}
   
   check = CUDA_CHECK_RETURN( cudaMemcpy(dalpha2_array, halpha2_array, TOTAL_HAAR * sizeof(int16_t), cudaMemcpyHostToDevice), __FILE__, __LINE__ );
   //if( check != 0){
   //   std::cerr << "Error: CudaMalloc not successfull for device dalpha2" << std::endl;
   //   exit(1);
   //}
   
   check = CUDA_CHECK_RETURN( cudaMemcpy(dstages_thresh_array, hstages_thresh_array, TOTAL_STAGES * sizeof(int16_t), cudaMemcpyHostToDevice), __FILE__, __LINE__ );
   //if( check != 0){
   //   std::cerr << "Error: CudaMalloc not successfull for device dstage_thresh" << std::endl;
   //   exit(1);
   //}
 
   check = CUDA_CHECK_RETURN( cudaMemcpy(dstages_array, hstages_array, TOTAL_STAGES * sizeof(int), cudaMemcpyHostToDevice), __FILE__,  __LINE__ );
   //if( check != 0){
   //   std::cerr << "Error: CudaMalloc not successfull for device dhstages_array" << std::endl;
   //   exit(1);
   //}
 
}

////ENTIRE CASCADE CLASSIFIER ON GPU ////
void cascadeClassifierOnDevice(MyImage* img1, 
                               int bitvec_width, int bitvec_height,
                               uint16_t* dindex_x, uint16_t* dindex_y,
                               uint16_t* dwidth,   uint16_t* dheight,
                               int16_t* dweights_array, int16_t* dtree_thresh_array,
                               int16_t* dalpha1_array, int16_t* dalpha2_array,  
                               int16_t* dstages_thresh_array,
                               int32_t* devicesum1, int32_t* devicesqsum1, 
                               int* dstages_array,
                               bool* dbit_vector
                              )
{

      /**************************************/
      //Timing related
      cudaError_t error;
      cudaEvent_t gpu_exc_start;
      cudaEvent_t gpu_exc_stop;
      
      float exc_msecTotal;
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
   
      
      // Eexcution Configuratuion Kernel 0
      dim3 numThreads(HAAR_BLOCK_SIZE, HAAR_BLOCK_SIZE);
      dim3 numBlocks((bitvec_width + HAAR_BLOCK_SIZE - 1) / HAAR_BLOCK_SIZE, 
                      (bitvec_height + HAAR_BLOCK_SIZE - 1) / HAAR_BLOCK_SIZE);


      printf("\n\tCascade Classifier on GPU Started\n");

      /*-------------------------------------------------------------------
        Starting timer for Runcascade Kernels comparison
      -------------------------------------------------------------------*/
      int check;

      //if (error != cudaSuccess)
      //{
      //     fprintf(stderr, "Failed to record start event (error code %s)!\n", cudaGetErrorString(error));
      //     exit(EXIT_FAILURE);
      //}

       //**************//
       // HAAR KERNEL 1 //
       //*************//

      cudaEventRecord(gpu_exc_start, 0);
      haar_stage_kernel0<<<numBlocks, numThreads>>>(dindex_x, dindex_y, dwidth, dheight, 
                                                     dweights_array, dtree_thresh_array, 
                                                     dalpha1_array, dalpha2_array, 
                                                     dstages_thresh_array, 
                                                     devicesum1, devicesqsum1, 
                                                     dstages_array, 
                                                     HAAR_KERN_0, NUMSTG_KERN_0, 
                                                     img1->width, img1->height, 
                                                     dbit_vector); 

       
       //cudaPeekAtLastError();
       cudaDeviceSynchronize();
       cudaEventRecord(gpu_exc_stop, 0);
       cudaEventSynchronize(gpu_exc_stop);
       
       cudaEventElapsedTime(&exc_msecTotal, gpu_exc_start, gpu_exc_stop);
       printf("\tCC: Kernel 1 Complete--> Exclusive Time: %f ms\n", exc_msecTotal);
       
       //if( check != 0){
       //      std::cerr << "Error: CudaPeek on Row Scan not successfull for device dest image" << std::endl;
       //      exit(1);
       //}
       
       //if( check != 0){
       //      std::cerr << "Error: CudaSynchronize not successfull for device dest image" << std::endl;
       //      exit(1);
       //}    

       // Record the stop event
       //if (error != cudaSuccess)
       //{
       //    fprintf(stderr, "Failed to record stop event (error code %s)!\n", cudaGetErrorString(error));
       //    exit(EXIT_FAILURE);
       //}

       //// Wait for the stop event to complete
       //if (error != cudaSuccess)
       //{
       //    fprintf(stderr, "Failed to synchronize on the stop event (error code %s)!\n", cudaGetErrorString(error));
       //    exit(EXIT_FAILURE);
       //}

       //if (error != cudaSuccess)
       //{
       //    fprintf(stderr, "Failed to get time elapsed between events (error code %s)!\n", cudaGetErrorString(error));
       //    exit(EXIT_FAILURE);
       //}

       gpu_excmsecTotal = exc_msecTotal;
     

      //**************//
      // HAAR KERNEL 2 //
      //*************//
       
      int haar_prev_stage = HAAR_KERN_0;
      int num_prev_stage = NUMSTG_KERN_0;
      
      error = cudaEventRecord(gpu_exc_start, NULL);
      if (error != cudaSuccess)
      {
          fprintf(stderr, "Failed to record start event (error code %s)!\n", cudaGetErrorString(error));
          exit(EXIT_FAILURE);
      }

      haar_stage_kernel0<<<numBlocks, numThreads>>>(dindex_x + 3 * haar_prev_stage, dindex_y + 3 * haar_prev_stage, 
                                                     dwidth + 3 * haar_prev_stage, dheight + 3 * haar_prev_stage, 
                                                     dweights_array + 3 * haar_prev_stage, 
                                                     dtree_thresh_array + haar_prev_stage, dalpha1_array + haar_prev_stage, 
                                                     dalpha2_array + haar_prev_stage, dstages_thresh_array + num_prev_stage, 
                                                     devicesum1, devicesqsum1, 
                                                     dstages_array + num_prev_stage, 
                                                     HAAR_KERN_1, NUMSTG_KERN_1, 
                                                     img1->width, img1->height, 
                                                     dbit_vector
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

      gpu_excmsecTotal += exc_msecTotal;
      printf("\tCC: Kernel 2 Complete--> Exclusive Time: %f ms\n", exc_msecTotal );
    
      //**************//
      // HAAR KERNEL 2//
      //*************//
       
      haar_prev_stage += HAAR_KERN_1;
      num_prev_stage += NUMSTG_KERN_1;
      
      error = cudaEventRecord(gpu_exc_start, NULL);
      if (error != cudaSuccess)
      {
          fprintf(stderr, "Failed to record start event (error code %s)!\n", cudaGetErrorString(error));
          exit(EXIT_FAILURE);
      }

      haar_stage_kernel0<<<numBlocks, numThreads>>>(dindex_x + 3 * haar_prev_stage, dindex_y + 3 * haar_prev_stage, 
                                                     dwidth + 3 * haar_prev_stage, dheight + 3 * haar_prev_stage, 
                                                     dweights_array + 3 * haar_prev_stage, 
                                                     dtree_thresh_array + haar_prev_stage, dalpha1_array + haar_prev_stage, 
                                                     dalpha2_array + haar_prev_stage, dstages_thresh_array + num_prev_stage, 
                                                     devicesum1, devicesqsum1, 
                                                     dstages_array + num_prev_stage, 
                                                     HAAR_KERN_2, NUMSTG_KERN_2, 
                                                     img1->width, img1->height, 
                                                     dbit_vector
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

      gpu_excmsecTotal += exc_msecTotal;
      printf("\tCC: Kernel 3 Complete--> Exclusive Time: %f ms\n", exc_msecTotal );
    
      //**************//
      // HAAR KERNEL 3//
      //*************//
       
      haar_prev_stage += HAAR_KERN_2;
      num_prev_stage += NUMSTG_KERN_2;
      
      error = cudaEventRecord(gpu_exc_start, NULL);
      if (error != cudaSuccess)
      {
          fprintf(stderr, "Failed to record start event (error code %s)!\n", cudaGetErrorString(error));
          exit(EXIT_FAILURE);
      }

      haar_stage_kernel0<<<numBlocks, numThreads>>>(dindex_x + 3 * haar_prev_stage, dindex_y + 3 * haar_prev_stage, 
                                                     dwidth + 3 * haar_prev_stage, dheight + 3 * haar_prev_stage, 
                                                     dweights_array + 3 * haar_prev_stage, 
                                                     dtree_thresh_array + haar_prev_stage, dalpha1_array + haar_prev_stage, 
                                                     dalpha2_array + haar_prev_stage, dstages_thresh_array + num_prev_stage, 
                                                     devicesum1, devicesqsum1, 
                                                     dstages_array + num_prev_stage, 
                                                     HAAR_KERN_3, NUMSTG_KERN_3, 
                                                     img1->width, img1->height, 
                                                     dbit_vector
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

      gpu_excmsecTotal += exc_msecTotal;
      printf("\tCC: Kernel 3 Complete--> Exclusive Time: %f ms\n", exc_msecTotal );
    

      //**************//
      // HAAR KERNEL 4//
      //*************//
       
      haar_prev_stage += HAAR_KERN_3;
      num_prev_stage += NUMSTG_KERN_3;
      
      error = cudaEventRecord(gpu_exc_start, NULL);
      if (error != cudaSuccess)
      {
          fprintf(stderr, "Failed to record start event (error code %s)!\n", cudaGetErrorString(error));
          exit(EXIT_FAILURE);
      }

      haar_stage_kernel0<<<numBlocks, numThreads>>>(dindex_x + 3 * haar_prev_stage, dindex_y + 3 * haar_prev_stage, 
                                                     dwidth + 3 * haar_prev_stage, dheight + 3 * haar_prev_stage, 
                                                     dweights_array + 3 * haar_prev_stage, 
                                                     dtree_thresh_array + haar_prev_stage, dalpha1_array + haar_prev_stage, 
                                                     dalpha2_array + haar_prev_stage, dstages_thresh_array + num_prev_stage, 
                                                     devicesum1, devicesqsum1, 
                                                     dstages_array + num_prev_stage, 
                                                     HAAR_KERN_4, NUMSTG_KERN_4, 
                                                     img1->width, img1->height, 
                                                     dbit_vector
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

      gpu_excmsecTotal += exc_msecTotal;
      printf("\tCC: Kernel 4 Complete--> Exclusive Time: %f ms\n", exc_msecTotal );
    
      //**************//
      // HAAR KERNEL 5//
      //*************//
       
      haar_prev_stage += HAAR_KERN_4;
      num_prev_stage += NUMSTG_KERN_4;
      
      error = cudaEventRecord(gpu_exc_start, NULL);
      if (error != cudaSuccess)
      {
          fprintf(stderr, "Failed to record start event (error code %s)!\n", cudaGetErrorString(error));
          exit(EXIT_FAILURE);
      }

      haar_stage_kernel0<<<numBlocks, numThreads>>>(dindex_x + 3 * haar_prev_stage, dindex_y + 3 * haar_prev_stage, 
                                                     dwidth + 3 * haar_prev_stage, dheight + 3 * haar_prev_stage, 
                                                     dweights_array + 3 * haar_prev_stage, 
                                                     dtree_thresh_array + haar_prev_stage, dalpha1_array + haar_prev_stage, 
                                                     dalpha2_array + haar_prev_stage, dstages_thresh_array + num_prev_stage, 
                                                     devicesum1, devicesqsum1, 
                                                     dstages_array + num_prev_stage, 
                                                     HAAR_KERN_5, NUMSTG_KERN_5, 
                                                     img1->width, img1->height, 
                                                     dbit_vector
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

      gpu_excmsecTotal += exc_msecTotal;
      printf("\tCC: Kernel 5 Complete--> Exclusive Time: %f ms\n", exc_msecTotal );
    
      //**************//
      // HAAR KERNEL 6//
      //*************//
       
      haar_prev_stage += HAAR_KERN_5;
      num_prev_stage += NUMSTG_KERN_5;
      
      error = cudaEventRecord(gpu_exc_start, NULL);
      if (error != cudaSuccess)
      {
          fprintf(stderr, "Failed to record start event (error code %s)!\n", cudaGetErrorString(error));
          exit(EXIT_FAILURE);
      }

      haar_stage_kernel0<<<numBlocks, numThreads>>>(dindex_x + 3 * haar_prev_stage, dindex_y + 3 * haar_prev_stage, 
                                                     dwidth + 3 * haar_prev_stage, dheight + 3 * haar_prev_stage, 
                                                     dweights_array + 3 * haar_prev_stage, 
                                                     dtree_thresh_array + haar_prev_stage, dalpha1_array + haar_prev_stage, 
                                                     dalpha2_array + haar_prev_stage, dstages_thresh_array + num_prev_stage, 
                                                     devicesum1, devicesqsum1, 
                                                     dstages_array + num_prev_stage, 
                                                     HAAR_KERN_6, NUMSTG_KERN_6, 
                                                     img1->width, img1->height, 
                                                     dbit_vector
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

      gpu_excmsecTotal += exc_msecTotal;
      printf("\tCC: Kernel 6 Complete--> Exclusive Time: %f ms\n", exc_msecTotal );
    

      //**************//
      // HAAR KERNEL 7//
      //*************//
       
      haar_prev_stage += HAAR_KERN_6;
      num_prev_stage += NUMSTG_KERN_6;
      
      error = cudaEventRecord(gpu_exc_start, NULL);
      if (error != cudaSuccess)
      {
          fprintf(stderr, "Failed to record start event (error code %s)!\n", cudaGetErrorString(error));
          exit(EXIT_FAILURE);
      }

      haar_stage_kernel0<<<numBlocks, numThreads>>>(dindex_x + 3 * haar_prev_stage, dindex_y + 3 * haar_prev_stage, 
                                                     dwidth + 3 * haar_prev_stage, dheight + 3 * haar_prev_stage, 
                                                     dweights_array + 3 * haar_prev_stage, 
                                                     dtree_thresh_array + haar_prev_stage, dalpha1_array + haar_prev_stage, 
                                                     dalpha2_array + haar_prev_stage, dstages_thresh_array + num_prev_stage, 
                                                     devicesum1, devicesqsum1, 
                                                     dstages_array + num_prev_stage, 
                                                     HAAR_KERN_7, NUMSTG_KERN_7, 
                                                     img1->width, img1->height, 
                                                     dbit_vector
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

      gpu_excmsecTotal += exc_msecTotal;
      printf("\tCC: Kernel 7 Complete--> Exclusive Time: %f ms\n", exc_msecTotal );
    
      //**************//
      // HAAR KERNEL 8//
      //*************//
       
      haar_prev_stage += HAAR_KERN_7;
      num_prev_stage += NUMSTG_KERN_7;
      
      error = cudaEventRecord(gpu_exc_start, NULL);
      if (error != cudaSuccess)
      {
          fprintf(stderr, "Failed to record start event (error code %s)!\n", cudaGetErrorString(error));
          exit(EXIT_FAILURE);
      }

      haar_stage_kernel0<<<numBlocks, numThreads>>>(dindex_x + 3 * haar_prev_stage, dindex_y + 3 * haar_prev_stage, 
                                                     dwidth + 3 * haar_prev_stage, dheight + 3 * haar_prev_stage, 
                                                     dweights_array + 3 * haar_prev_stage, 
                                                     dtree_thresh_array + haar_prev_stage, dalpha1_array + haar_prev_stage, 
                                                     dalpha2_array + haar_prev_stage, dstages_thresh_array + num_prev_stage, 
                                                     devicesum1, devicesqsum1, 
                                                     dstages_array + num_prev_stage, 
                                                     HAAR_KERN_8, NUMSTG_KERN_8, 
                                                     img1->width, img1->height, 
                                                     dbit_vector
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

      gpu_excmsecTotal += exc_msecTotal;
      printf("\tCC: Kernel 8 Complete--> Exclusive Time: %f ms\n", exc_msecTotal );
    
      
      //**************//
      // HAAR KERNEL 9//
      //*************//
       
      haar_prev_stage += HAAR_KERN_8;
      num_prev_stage += NUMSTG_KERN_8;
      
      error = cudaEventRecord(gpu_exc_start, NULL);
      if (error != cudaSuccess)
      {
          fprintf(stderr, "Failed to record start event (error code %s)!\n", cudaGetErrorString(error));
          exit(EXIT_FAILURE);
      }

      haar_stage_kernel0<<<numBlocks, numThreads>>>(dindex_x + 3 * haar_prev_stage, dindex_y + 3 * haar_prev_stage, 
                                                     dwidth + 3 * haar_prev_stage, dheight + 3 * haar_prev_stage, 
                                                     dweights_array + 3 * haar_prev_stage, 
                                                     dtree_thresh_array + haar_prev_stage, dalpha1_array + haar_prev_stage, 
                                                     dalpha2_array + haar_prev_stage, dstages_thresh_array + num_prev_stage, 
                                                     devicesum1, devicesqsum1, 
                                                     dstages_array + num_prev_stage, 
                                                     HAAR_KERN_9, NUMSTG_KERN_9, 
                                                     img1->width, img1->height, 
                                                     dbit_vector
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

      gpu_excmsecTotal += exc_msecTotal;
      printf("\tCC: Kernel 9 Complete--> Exclusive Time: %f ms\n", exc_msecTotal );
    
      //**************//
      // HAAR KERNEL 10//
      //*************//
       
      haar_prev_stage += HAAR_KERN_9;
      num_prev_stage += NUMSTG_KERN_9;
      
      error = cudaEventRecord(gpu_exc_start, NULL);
      if (error != cudaSuccess)
      {
          fprintf(stderr, "Failed to record start event (error code %s)!\n", cudaGetErrorString(error));
          exit(EXIT_FAILURE);
      }

      haar_stage_kernel0<<<numBlocks, numThreads>>>(dindex_x + 3 * haar_prev_stage, dindex_y + 3 * haar_prev_stage, 
                                                     dwidth + 3 * haar_prev_stage, dheight + 3 * haar_prev_stage, 
                                                     dweights_array + 3 * haar_prev_stage, 
                                                     dtree_thresh_array + haar_prev_stage, dalpha1_array + haar_prev_stage, 
                                                     dalpha2_array + haar_prev_stage, dstages_thresh_array + num_prev_stage, 
                                                     devicesum1, devicesqsum1, 
                                                     dstages_array + num_prev_stage, 
                                                     HAAR_KERN_10, NUMSTG_KERN_10, 
                                                     img1->width, img1->height, 
                                                     dbit_vector
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

      gpu_excmsecTotal += exc_msecTotal;
      printf("\tCC: Kernel 10 Complete--> Exclusive Time: %f ms\n", exc_msecTotal );
    
      //**************//
      // HAAR KERNEL 11//
      //*************//
       
      haar_prev_stage += HAAR_KERN_10;
      num_prev_stage += NUMSTG_KERN_10;
      
      error = cudaEventRecord(gpu_exc_start, NULL);
      if (error != cudaSuccess)
      {
          fprintf(stderr, "Failed to record start event (error code %s)!\n", cudaGetErrorString(error));
          exit(EXIT_FAILURE);
      }

      haar_stage_kernel0<<<numBlocks, numThreads>>>(dindex_x + 3 * haar_prev_stage, dindex_y + 3 * haar_prev_stage, 
                                                     dwidth + 3 * haar_prev_stage, dheight + 3 * haar_prev_stage, 
                                                     dweights_array + 3 * haar_prev_stage, 
                                                     dtree_thresh_array + haar_prev_stage, dalpha1_array + haar_prev_stage, 
                                                     dalpha2_array + haar_prev_stage, dstages_thresh_array + num_prev_stage, 
                                                     devicesum1, devicesqsum1, 
                                                     dstages_array + num_prev_stage, 
                                                     HAAR_KERN_11, NUMSTG_KERN_11, 
                                                     img1->width, img1->height, 
                                                     dbit_vector
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

      gpu_excmsecTotal += exc_msecTotal;
      printf("\tCC: Kernel 11 Complete--> Exclusive Time: %f ms\n", exc_msecTotal );
    

      /***********************************************************************************/
      printf("\tCascade Classifier on GPU Complete--> Combined Exclusive Time: %f ms\n", gpu_excmsecTotal);

}