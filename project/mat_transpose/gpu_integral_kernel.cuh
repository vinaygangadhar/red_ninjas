#ifndef __NN_RS_KERNEL_H__
#define __NN_RS_KERNEL_H__

#include <stdlib.h>

//__global__ void rowscan_nn_kernel(unsigned char* src, unsigned char* dst, int* sum, int* sqsum, 
//                                   int src_w, int src_h, int dst_w, int dst_h, 
//                                   int x_ratio, int y_ratio, int row_elems)
//
__global__ void rowscan_nn_kernel(unsigned char* src, int* sum, int* sqsum, 
                                   int src_w, int src_h, int dst_w, int dst_h, 
                                   int x_ratio, int y_ratio, int row_elems)
{
     volatile __shared__ int row[SHARED];

     //Nearest Neighbor Kernel
     int tId = threadIdx.x;
     int g_Idx = blockIdx.x * dst_w + tId;
  
     int offset = 1;
     int sq_start = row_elems + 1;
     int offset_tId = tId + blockDim.x;

     //Compute the TB start in the global scope of dst image
     int TB_START_ADDR = blockIdx.x * dst_w;                     
          
     //Populating 2 elemnts by each thread
     int row_src = (blockIdx.x * y_ratio) >> 16;
     int col1_src = (tId * x_ratio)  >> 16 ;
     int col2_src = ( (tId + blockDim.x) * x_ratio)  >> 16 ;

////--DELETE LATER
//     dst[TB_START_ADDR + tId] = src[row_src * src_w + col1_src];                    
//     if( (offset_tId) < dst_w){                                                                  
//          dst[TB_START_ADDR + offset_tId] = src[row_src * src_w + col2_src];
//     }
////--DELETE LATER

     //Populate shared memory row with down sampled pixels 
     row[tId] = src[row_src * src_w + col1_src];
     
     if( (offset_tId) < dst_w){                                                                  
          row[offset_tId] = src[row_src * src_w + col2_src];
     }else{
          row[offset_tId] = 0;
     }

     //Square of each elements in SMem
     row[sq_start + tId] = row[tId] * row[tId];
     row[sq_start + offset_tId] = row[offset_tId] * row[offset_tId];
     
     //Upsweep
     for (int stage_threads = row_elems>>1; stage_threads > 0; stage_threads >>= 1)
     {
       __syncthreads();

       if (tId < stage_threads)
       {
         int ai = offset * (2*tId+1) - 1;         //Each thread start
         int bi = offset * (2*tId+2) - 1;         //Neighbor or mate thread
         
         row[bi] += row[ai];
         row[sq_start + bi] += row[sq_start + ai];
       }
       
       offset <<= 1;
     }

     //DownSweep
     if (tId == 0) 
     { 
       row[row_elems] = row[row_elems - 1]; 
       row[row_elems -1] = 0; 
       row[sq_start + row_elems] = row[sq_start + row_elems - 1]; 
       row[sq_start + row_elems - 1] = 0; 
     }

     for (int stage_threads = 1; stage_threads < row_elems; stage_threads *= 2 )
     {

       offset >>=1 ;
       __syncthreads();

       if (tId < stage_threads)
       {

         int ai = offset * (2*tId+1) - 1;
         int bi = offset * (2*tId+2) - 1;

         int dummy = row[ai];
         row[ai] = row[bi];
         row[bi] += dummy;
         
         int dummy_sq = row[sq_start + ai];
         row[sq_start + ai] = row[sq_start + bi];
         row[sq_start + bi] += dummy_sq;
       }
     }

     __syncthreads();

     sum[g_Idx] = row[tId + 1];
     sqsum[g_Idx] = row[sq_start + tId + 1];
     
     if (offset_tId < dst_w)
     {
       sum[g_Idx + blockDim.x] = row[offset_tId + 1];
       sqsum[g_Idx + blockDim.x] = row[sq_start + offset_tId + 1];

     }
}

//row only Scan

__global__ void rowscan_only(int* sum, int* sqsum, int w, int n)
{
  volatile __shared__ int temp1[2050];

  int tid = threadIdx.x;
  int offset =1;
  int g_idx = blockIdx.x * w + tid;
  int sq_start = n + 1;
  int new_tid = tid + blockDim.x;

  temp1[tid] = sum[g_idx] ;
  temp1[new_tid] = new_tid < w ? sum[g_idx + blockDim.x] : 0;
  temp1[sq_start + tid] =  sqsum[g_idx];
  temp1[sq_start + new_tid] =  new_tid < w ? sqsum[g_idx + blockDim.x] : 0;

  for (int d = n>>1; d > 0; d >>= 1)
  {
    __syncthreads();

    if (tid < d)
    {

      int ai = offset*(2*tid+1)-1;
      int bi = offset*(2*tid+2)-1;
      temp1[bi] += temp1[ai];
      temp1[sq_start+bi] += temp1[sq_start+ai];

    }
    offset <<= 1;
  }

  if (tid==0) 
  { 
    temp1[n] = temp1[n-1]; temp1[n-1] = 0; 
    temp1[sq_start+n] = temp1[sq_start+n-1]; temp1[sq_start+n-1] = 0; 
  }

  for (int d = 1; d < n; d *= 2 )
  {

    offset >>= 1;
    __syncthreads();

    if (tid < d)
    {

      int ai = offset*(2*tid+1)-1;
      int bi = offset*(2*tid+2)-1;

      int t = temp1[ai];
      temp1[ai] = temp1[bi];
      temp1[bi] += t;
      int t_sq = temp1[sq_start+ai];
      temp1[sq_start+ai] = temp1[sq_start+bi];
      temp1[sq_start+bi] += t_sq;
    }
  }

  __syncthreads();

    sum[g_idx] = temp1[tid+1];
    sqsum[g_idx] = temp1[sq_start+tid+1];
  if (new_tid < w)
  {
    sum[g_idx+blockDim.x] = temp1[new_tid+1];
    sqsum[g_idx+blockDim.x] = temp1[sq_start+new_tid+1];
  }

}

#endif
