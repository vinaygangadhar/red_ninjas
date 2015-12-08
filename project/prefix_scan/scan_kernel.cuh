#ifndef _SCAN_KERNEL_H

#define BLOCK_SIZE 1024
__global__ void prefixscan_perblock(float *g_idata, float *blockSum, int n)
{
    volatile __shared__ float temp[2*BLOCK_SIZE];// allocated on invocation

    int thid = threadIdx.x;
    unsigned int blockStart = blockIdx.x*(blockDim.x*2);
    unsigned int index = blockStart + threadIdx.x; 
    int offset = 1;

    if(blockIdx.x == 0 && threadIdx.x == 0) {
        blockSum[0] = 0.0f;
        //printf("FK: blockSum[%d] = %f\n", blockIdx.x, blockSum[blockIdx.x]);
    }

    // load input into shared memory
    // Avoid bank conflicts
    if(thid < n) {
        temp[thid] = g_idata[index];
    }
    if(thid+blockDim.x < n) {
        temp[thid+blockDim.x] = g_idata[index+blockDim.x];
    }
    __syncthreads();

    // Up-Stream
    for (int d = (2*BLOCK_SIZE)>>1; d > 0; d >>= 1) // build sum in place up the tree
    {
        __syncthreads();
        if (thid < d)
        {
            int ai = offset*(2*thid+1)-1;
            int bi = offset*(2*thid+2)-1;
            temp[bi] += temp[ai];
        }
        offset <<= 1; //multiply by 2 implemented as bitwise operation
    }
    __syncthreads();
    if (thid == 0) { 
        blockSum[blockIdx.x+1] = temp[(2*BLOCK_SIZE)-1];
        temp[(2*BLOCK_SIZE)-1] = 0;  // clear the last element
    }
    
    // Down-stream
    for (int d = 1; d < (2*BLOCK_SIZE); d <<= 1) // traverse down tree & build scan
    {
        offset >>= 1;
        __syncthreads();
        if (thid < d)
        {
            int ai = offset*(2*thid+1)-1;
            int bi = offset*(2*thid+2)-1;

            float t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }

    __syncthreads();

    //printf("Blockstar %d: temp[%d] = %.3f\n", blockIdx.x, 2*thid, temp[2*thid]);
    __syncthreads();
    if(blockStart+2*thid < n) {
        g_idata[blockStart+2*thid] = temp[2*thid]; // write results to device memory
    }
    if(blockStart+2*thid+1 < n) {
        g_idata[blockStart+2*thid+1] = temp[2*thid+1];
    }
}

__global__ void prefixscan_allblocks(float* g_idata, float* blockSum, int num_elements) 
{
    volatile __shared__ float thisBlockSum;
    int index = blockIdx.x * (blockDim.x*2) + threadIdx.x; 
    if(threadIdx.x == 0) {
        thisBlockSum = blockSum[blockIdx.x];
        //printf("SK: blockSum[%d] = %f = %f\n", blockIdx.x, blockSum[blockIdx.x], thisBlockSum);
    }
    __syncthreads();
    if(index < num_elements) {
        g_idata[index] = g_idata[index] + thisBlockSum;   
    }
    if((index+blockDim.x) < num_elements) {
        g_idata[index+blockDim.x] = g_idata[index+blockDim.x] + thisBlockSum;
    }
}

#endif // #ifndef _SCAN_KERNEL_H
