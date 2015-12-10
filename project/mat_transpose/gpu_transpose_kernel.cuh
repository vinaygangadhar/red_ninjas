__global__ void transpose_kernel(int* sum, int *trans_sum, int* sqsum, int *trans_sqsum, int w, int h)
{

     volatile __shared__ int sum_temp[16][16];
     volatile __shared__ int sqsum_temp[16][16];

     int tx = threadIdx.x;
     int ty = threadIdx.y;
     int bx = blockIdx.x;
     int by = blockIdx.y;
     //int blk_offset = bx * 16 + by * w * 16;
     //int t_offset = ty * w + tx;
     int row = by * 16 + ty ;
     int col = bx * 16 + tx;
    //condition for row and column bound check
     
     sum_temp[ty][tx] = (row < h && col < w) ? sum[row * w + col] : 0;
     sqsum_temp[ty][tx] = (row < h && col < w) ? sqsum[row * w + col] : 0;

     __syncthreads();

     if (row < h && col < w)
     {
     trans_sum[col * h + row] = sum_temp[ty][tx];
     trans_sqsum[col * h + row] = sqsum_temp[ty][tx];
     }

}
