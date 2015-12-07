#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <iostream>
#include <math.h>
#include <stdint.h>

#include "haar_stage_kernel.cu"

#define BLOCKSIZE 1024

void checkError();

using namespace std;

void checkError() {
    // check for error
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        // print the CUDA error message and exit
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char** argv) {
    uint16_t* index_x =        (uint16_t*) malloc(3*MAX_HAAR*sizeof(uint16_t));
    uint16_t* index_y =        (uint16_t*) malloc(3*MAX_HAAR*sizeof(uint16_t));
    uint16_t* width =          (uint16_t*) malloc(3*MAX_HAAR*sizeof(uint16_t));
    uint16_t* height =         (uint16_t*) malloc(3*MAX_HAAR*sizeof(uint16_t));
    uint16_t* weight =        (uint16_t*) malloc(MAX_HAAR*sizeof(uint16_t));
    uint16_t* tree_threshold =(uint16_t*) malloc(MAX_HAAR*sizeof(uint16_t));
    uint16_t* alpha1 =        (uint16_t*) malloc(MAX_HAAR*sizeof(uint16_t));
    uint16_t* alpha2 =        (uint16_t*) malloc(MAX_HAAR*sizeof(uint16_t));
    uint16_t* thr_per_stg =   (uint16_t*) malloc(MAX_STAGE*sizeof(uint16_t));
    uint32_t* sum =           (uint32_t*) malloc(55*55*sizeof(uint32_t));
    uint32_t* sqsum =         (uint32_t*) malloc(55*55*sizeof(uint32_t));
    uint8_t* haar_per_stage = (uint8_t*) malloc(9*sizeof(uint8_t));

    uint16_t* dindex_x;      
    uint16_t* dindex_y; 
    uint16_t* dwidth;   
    uint16_t* dheight; 
    uint16_t* dweight; 
    uint16_t* dtree_th;
    uint16_t* dalpha1; 
    uint16_t* dalpha2; 
    uint16_t* dthr_per_stg;
    uint32_t* dsum;
    uint32_t* dsqsum; 
    uint8_t* dhaar_per_stg;
    bool* dbit_vector;

    cudaMalloc(&dindex_x, 3*MAX_HAAR*sizeof(uint16_t));
    cudaMalloc(&dindex_y, 3*MAX_HAAR*sizeof(uint16_t));
    cudaMalloc(&dwidth, 3*MAX_HAAR*sizeof(uint16_t));
    cudaMalloc(&dheight, 3*MAX_HAAR*sizeof(uint16_t));
    
    cudaMalloc(&dweight, MAX_HAAR*sizeof(uint16_t));
    cudaMalloc(&dtree_th, MAX_HAAR*sizeof(uint16_t));
    cudaMalloc(&dalpha1, MAX_HAAR*sizeof(uint16_t));
    cudaMalloc(&dalpha2, MAX_HAAR*sizeof(uint16_t));
    cudaMalloc(&dthr_per_stg, MAX_STAGE*sizeof(uint16_t));
    
    cudaMalloc(&dsum, 55*55*sizeof(uint32_t));
    cudaMalloc(&dsqsum, 55*55*sizeof(uint32_t));
    cudaMalloc(&dhaar_per_stg, 9*sizeof(uint8_t));
    cudaMalloc(&dbit_vector, 616*456*sizeof(bool));

    int i;
    for(i=0; i<3*MAX_HAAR; i++) {
        index_x[i] = rand();
        index_y[i] = rand();
        width[i] = rand();
        height[i] = rand();
    }
    for(i=0; i<MAX_HAAR; i++) {
        weight[i] = rand();
        tree_threshold[i] = rand();
        alpha1[i] = rand();
        alpha2[i] = rand();
    }
    
    cudaMemcpy(dindex_x, index_x, 3*MAX_HAAR*sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dindex_y, index_y, 3*MAX_HAAR*sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dwidth, width, 3*MAX_HAAR*sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dheight, height, 3*MAX_HAAR*sizeof(uint16_t), cudaMemcpyHostToDevice);
    
    cudaMemcpy(dweight, weight, MAX_HAAR*sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dtree_th, tree_threshold, MAX_HAAR*sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dalpha1, alpha1, MAX_HAAR*sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dalpha2, alpha2, MAX_HAAR*sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dthr_per_stg, thr_per_stg, MAX_STAGE*sizeof(uint16_t), cudaMemcpyHostToDevice);

    cudaMemcpy(dsum, sum, 55*55*sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dsqsum, sqsum, 55*55*sizeof(uint32_t), cudaMemcpyHostToDevice);
    
    cudaMemcpy(dhaar_per_stg, haar_per_stage, 9*sizeof(uint8_t), cudaMemcpyHostToDevice);

    dim3 block(32, 32);

    haar_stage_kernel0<<<275, block>>>(dindex_x, dindex_y, dwidth, dheight, dweight, dtree_th, dalpha1, dalpha2, dthr_per_stg, dsum, dsqsum, dhaar_per_stg, 323, 9, dbit_vector);

}
