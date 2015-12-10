///MATRIX TRANPSOSE KERNEL FOR PREFIX SCAN
#ifndef __TRANSPOSE_KERNEL_H__                                                                        
#define __TRANSPOSE_KERNEL_H__                                                                        

//32 bit version
__global__ void transpose_kernel(int* sum, int *transpose_sum, int* sqsum, int *transpose_sqsum, int w, int h)
{

     volatile __shared__ int sum_sMem[BLOCK_SIZE][BLOCK_SIZE + 1];
     volatile __shared__ int sqsum_sMem[BLOCK_SIZE][BLOCK_SIZE + 1];

     int tx = threadIdx.x;
     int ty = threadIdx.y;
     int bx = blockIdx.x;
     int by = blockIdx.y;
     
     int row = by * BLOCK_SIZE + ty ;
     int col = bx * BLOCK_SIZE + tx;
    //condition for row and column bound check
     
     sum_sMem[ty][tx] = (row < h && col < w) ? sum[row * w + col] : 0;
     sqsum_sMem[ty][tx] = (row < h && col < w) ? sqsum[row * w + col] : 0;

     __syncthreads();
   
     row = by * BLOCK_SIZE + tx;
     col = bx * BLOCK_SIZE + ty;

     if (row < h && col < w)
     {
          transpose_sum[col * h + row] = sum_sMem[tx][ty];
          transpose_sqsum[col * h + row] = sqsum_sMem[tx][ty];
     }

}


//16 bit version
__global__ void transpose_kernel16_t(int16_t* sum, int16_t *transpose_sum, 
                                    int16_t* sqsum, int16_t *transpose_sqsum, 
                                    int w, int h)
{

     volatile __shared__ int16_t sum_sMem[BLOCK_SIZE][BLOCK_SIZE + 1];
     volatile __shared__ int16_t sqsum_sMem[BLOCK_SIZE][BLOCK_SIZE + 1];

     int tx = threadIdx.x;
     int ty = threadIdx.y;
     int bx = blockIdx.x;
     int by = blockIdx.y;
     
     int row = by * BLOCK_SIZE + ty ;
     int col = bx * BLOCK_SIZE + tx;
    //condition for row and column bound check
     
     sum_sMem[ty][tx] = (row < h && col < w) ? sum[row * w + col] : 0;
     sqsum_sMem[ty][tx] = (row < h && col < w) ? sqsum[row * w + col] : 0;

     __syncthreads();
   
     row = by * BLOCK_SIZE + tx;
     col = bx * BLOCK_SIZE + ty;

     if (row < h && col < w)
     {
          transpose_sum[col * h + row] = sum_sMem[tx][ty];
          transpose_sqsum[col * h + row] = sqsum_sMem[tx][ty];
     }

}



#endif
