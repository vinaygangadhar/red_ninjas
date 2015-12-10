#include <stdio.h>
#include <stdlib.h>

#include "gpu_integral.cuh"

/*cuda return value check function*/
#define devErrChk(func) {devAssert((func),__FILE__,__LINE__);}

inline void devAssert(cudaError_t code, const char *file, int line)
{
  if (code != cudaSuccess)
  {
    printf("devAssert: %s %s %d \n", cudaGetErrorString(code), file,line);

  }
}


int main()
{
int w =4;
int h =2;
int num_pixels = 10;
int a[8] = {0,1,2,3,4,5,6,7};
int *d_src;
int *d_sum, *d_sqsum;
  devErrChk(cudaMalloc((void**)&(d_src), sizeof(int)*num_pixels));
  devErrChk(cudaMemcpy(d_src, a, sizeof(int)*num_pixels, cudaMemcpyHostToDevice));

  devErrChk(cudaMalloc((void**)&(d_sum), sizeof(int)*num_pixels));
  devErrChk(cudaMalloc((void**)&(d_sqsum), sizeof(int)*num_pixels));

  int *h_sum = (int*)malloc(sizeof(int)*num_pixels);
  if (!h_sum) {printf("h_sum malloc failed/n");}
  int *h_sqsum = (int*)malloc(sizeof(int)*num_pixels);
  if (!h_sqsum) {printf("h_sqsum malloc failed/n");}
  int q = (w+1)/2;
  int threads = q*2;
  int blocks = h;
  printf("height=%d width=%d\n",h,w);
  rowscan<<<blocks,threads,sizeof(int)*2*(threads+1)>>>(d_src,d_sum,d_sqsum,w);
 // colscan<<<threads,blocks,sizeof(int)*2*(blocks+1)>>>(d_sum,d_sqsum);
  devErrChk( cudaPeekAtLastError() );
  devErrChk( cudaDeviceSynchronize() );

  devErrChk(cudaMemcpy(h_sum, d_sum, sizeof(int)*num_pixels, cudaMemcpyDeviceToHost));
  devErrChk(cudaMemcpy(h_sqsum, d_sqsum, sizeof(int)*num_pixels, cudaMemcpyDeviceToHost));

  int sum_check = 0;
  int sqsum_check = 0;
  for (int i=0; i < num_pixels ; i++)

  {
    printf("h_sum[%d]=%d h_sqsum[%d]=%d\n",i,h_sum[i],i,h_sqsum[i]);
    //sum_check += (sum->data[i] == h_sum[i]) ? 0 : 1;
    //sqsum_check += (sqsum->data[i] == h_sqsum[i]) ? 0 : 1;
  }

  printf("\n********check-vals*********\n");
  printf("sum_check=%d\n sqsum_check=%d\n",sum_check, sqsum_check);
  printf("********check-vals-end*********\n");
  
  cudaFree(d_src);
  cudaFree(d_sum);
  cudaFree(d_sqsum);
  free(h_sum);
  free(h_sqsum);
}
