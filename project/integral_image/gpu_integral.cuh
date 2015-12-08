#include <stdlib.h>

__global__ void rowscan(unsigned char* src, int* sum, int* sqsum, int w)
{
  volatile __shared__ int temp[2050];

  int tid = threadIdx.x;
  int offset =1;
  int g_idx = blockIdx.x * w + tid;
  int sq_start = blockDim.x+1;
  int n = blockDim.x;

  temp[tid] = tid < w ? src[g_idx] : 0 ;
  temp[sq_start+tid] = temp[tid] * temp[tid];

  for (int d = n>>1; d > 0; d >>= 1)
  {
    __syncthreads();

    if (tid < d)
    {

      int ai = offset*(2*tid+1)-1;
      int bi = offset*(2*tid+2)-1;
      temp[bi] += temp[ai];
      temp[sq_start+bi] += temp[sq_start+ai];

    }
    offset <<= 1;
  }

  if (tid==0) 
  { 
    temp[n] = temp[n-1]; temp[n-1] = 0; 
    temp[sq_start+n] = temp[sq_start+n-1]; temp[sq_start+n-1] = 0; 
  }

  for (int d = 1; d < n; d *= 2 )
  {

    offset >>=1 ;
    __syncthreads();

    if (tid < d)
    {

      int ai = offset*(2*tid+1)-1;
      int bi = offset*(2*tid+2)-1;

      int t = temp[ai];
      temp[ai] = temp[bi];
      temp[bi] += t;
      int t_sq = temp[sq_start+ai];
      temp[sq_start+ai] = temp[sq_start+bi];
      temp[sq_start+bi] += t_sq;
    }
  }

  __syncthreads();

  if (tid < w)
  {
    sum[g_idx] = temp[tid+1];
    sqsum[g_idx] = temp[sq_start+tid+1];
  }
}


__global__ void colscan(int* sum, int* sqsum, int h)
{
  volatile __shared__ int temp1[2050];

  int tid = threadIdx.x;
  int offset =1;
  int g_idx = tid*gridDim.x + blockIdx.x;
  int sq_start = blockDim.x+1;
  int n = blockDim.x;

  temp1[tid] = tid < h ? sum[g_idx] : 0;
  temp1[sq_start + tid] =  tid < h ? sqsum[g_idx] : 0;

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

  if (tid < h)
  {
    sum[g_idx] = temp1[tid+1];
    sqsum[g_idx] = temp1[sq_start+tid+1];
  }

}
