#ifndef __ERROR_HANDLE__H_
#define __ERROR_HANDLE__H_

#include <sys/time.h>
#include <stdio.h>
#include <stdint.h>
#include <iostream>
#include <cuda.h> //CUDA library


#define BLOCK_SIZE (16)             //16 x 16
using namespace std;

//CUDA Error Checker -- If return value is -1 then there is an error
int CUDA_CHECK_RETURN(cudaError_t err_ret){

	int val = 0;
    if (err_ret != cudaSuccess) {
        fprintf(stderr, "Error %s at line %d in file %s\n", cudaGetErrorString(err_ret), __LINE__, __FILE__);   
        val = 1;
    }
    return val;    
} 																													

//compare the data stored in two arrays on the host
bool CompareResults(unsigned char* A, unsigned char* B, int elements){
   
   int diff = 0;   
   for(unsigned int i = 0; i < elements; i++){
       int error = abs(A[i]-B[i]);
       
       if(error > 0){
         diff++;
       }
   }
   
   if(diff > 0)
      return false;
   else
      return true;
}




#endif
