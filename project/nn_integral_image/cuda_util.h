#ifndef __ERROR_HANDLE__H_
#define __ERROR_HANDLE__H_

#include <sys/time.h>
#include <stdio.h>
#include <stdint.h>
#include <iostream>
#include <fstream>
#include <cuda.h> //CUDA library

//DEBUG FLAGS
#define LOG 1
#define DEVICE 1

#define MAX_IMG_SIZE (1024)

#define BLOCK_SIZE (16)             //16 x 16
#define SHARED     (MAX_IMG_SIZE * 2 + 2)

using namespace std;


//DEBUG Varibales
#ifdef LOG
      static const bool PRINT_LOG = true;
#else
      static const bool PRINT_LOG = false;
#endif

#ifdef DEVICE
      static const bool PRINT_GPU = true;
#else
      static const bool PRINT_GPU = false;
#endif

//CUDA CHECK RETURN                                                                                                      
int CUDA_CHECK_RETURN(cudaError_t err_ret, const char* file, int line)                                
{                                                                                                     
    int val = 0;                                                                                      
    if (err_ret != cudaSuccess) {                                                                     
        fprintf(stderr, "Error %s at line %d in file %s\n", cudaGetErrorString(err_ret), file, line); 
        val = 1;                                                                                      
    }                                                                                                 
    return val;                                                                                       
}                                                                                                     

//compare the data stored in two arrays on the host -- characters
bool CompareResultsChar(unsigned char* A, unsigned char* B, int elements){
   
   int diff = 0;   
   for(unsigned int i = 0; i < elements; i++){
       int error = abs(A[i]-B[i]);
       
       if(error > 0)
         diff++;
   }
   
   if(diff > 0)
      return false;
   else
      return true;
}


//compare the data stored in two arrays on the host -- 32bit integers
bool CompareResultsInt(int* hA, int* dA, int* hsq_A, int* dsq_A, int elements){
   
   int sum_check = 0;  
   int sqsum_check = 0;   
   
   for(unsigned int i = 0; i < elements; i++){
      
      sum_check += (hA[i] == dA[i]) ? 0 : 1;      
      sqsum_check += (hsq_A[i] == dsq_A[i]) ? 0 : 1;
   
   }

   if( (sum_check != 0) || (sqsum_check != 0)){
        return false;
   }else{
        return true;
   }

}

//compare the data stored in two arrays on the host -- 16bit integers
bool CompareResultsInt16_t(int16_t* hA, int16_t* dA, int16_t* hsq_A, int16_t* dsq_A, int elements){
   
   int sum_check = 0;  
   int sqsum_check = 0;   
   
   for(unsigned int i = 0; i < elements; i++){
      
      sum_check += (hA[i] == dA[i]) ? 0 : 1;      
      sqsum_check += (hsq_A[i] == dsq_A[i]) ? 0 : 1;
   
   }

   if( (sum_check != 0) || (sqsum_check != 0)){
        return false;
   }else{
        return true;
   }

}



// Write a 16x16 floating point matrix to file
void WriteFileChar(unsigned char* data, int elements, std::fstream& ofs){

   for(unsigned int i = 0; i < elements; i++){
      ofs<<(data[i] - '0')<<" ";
   }
}

//Write contens of int data structure
void WriteFileInt(int* data, int elements, std::fstream& ofs){

   for(unsigned int i = 0; i < elements; i++){
      ofs<<data[i]<<" ";
   }
}



//Get smallest power 2 of a number
int getSmallestPower2(int num) {                                  
  int result = 1;                                                 
  while(result < num && result > 0)                               
    result <<= 1;                                                 
  if(result <= 0 || num <= 0) {                                   
    fprintf(stderr, "The size requested might be two large!\n");  
    exit(-1);                                                     
  }                                                               
  return result;                                                  
}                                                                 

#endif
