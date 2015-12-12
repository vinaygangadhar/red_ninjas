///MATRIX TRANPSOSE KERNEL FOR PREFIX SCAN
#ifndef __TRANSPOSE_KERNEL_H__                                                                        
#define __TRANSPOSE_KERNEL_H__                                                                        

//32 bit version
__global__ void transpose_kernel(int* sum, int *transpose_sum, int* sqsum, int *transpose_sqsum, int w, int h)
{

     volatile __shared__ int sum_sMem[NN_II_BLOCK_SIZE][NN_II_BLOCK_SIZE + 1];
     volatile __shared__ int sqsum_sMem[NN_II_BLOCK_SIZE][NN_II_BLOCK_SIZE + 1];

     int tx = threadIdx.x;
     int ty = threadIdx.y;
     int bx = blockIdx.x;
     int by = blockIdx.y;
     
     int row = by * NN_II_BLOCK_SIZE + ty ;
     int col = bx * NN_II_BLOCK_SIZE + tx;
    //condition for row and column bound check
     
     sum_sMem[ty][tx] = (row < h && col < w) ? sum[row * w + col] : 0;
     sqsum_sMem[ty][tx] = (row < h && col < w) ? sqsum[row * w + col] : 0;

     __syncthreads();
   
     row = by * NN_II_BLOCK_SIZE + tx;
     col = bx * NN_II_BLOCK_SIZE + ty;

     if (row < h && col < w)
     {
          transpose_sum[col * h + row] = sum_sMem[tx][ty];
          transpose_sqsum[col * h + row] = sqsum_sMem[tx][ty];
     }

}


#endif
