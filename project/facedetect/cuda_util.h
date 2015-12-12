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
//#define DEVICE 0

//NN_II Define
#define MAX_IMG_SIZE (1024)
#define NN_II_BLOCK_SIZE (16)             //16 x 16
#define SHARED     (MAX_IMG_SIZE * 2 + 2)

//HAAR Defines
#define HAAR_BLOCK_SIZE 32
#define NUM_RECT  3

//#define TOTAL_HAAR 2913
//#define TOTAL_STAGES 25
//#define MAX_HAAR 325
//#define WINDOW_WIDTH 24
//#define WINDOW_HEIGHT 24
//#define MAX_STAGE 7

// defines for separate kernels
#define HAAR_KERN_0 323
#define NUMSTG_KERN_0 8
#define HAAR_KERN_1 273
#define NUMSTG_KERN_1 3
#define HAAR_KERN_2 242
#define NUMSTG_KERN_2 2
#define HAAR_KERN_3 271
#define NUMSTG_KERN_3 2
#define HAAR_KERN_4 296
#define NUMSTG_KERN_4 2
#define HAAR_KERN_5 324
#define NUMSTG_KERN_5 2
#define HAAR_KERN_6 196
#define NUMSTG_KERN_6 1
#define HAAR_KERN_7 197
#define NUMSTG_KERN_7 1
#define HAAR_KERN_8 181
#define NUMSTG_KERN_8 1
#define HAAR_KERN_9 199
#define NUMSTG_KERN_9 1
#define HAAR_KERN_10 211
#define NUMSTG_KERN_10 1
#define HAAR_KERN_11 200
#define NUMSTG_KERN_11 1

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

//cuda error last
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

//Comparison for bit vector function
void CompareBits(bool* ref, bool* data, int n) {
    int i;
    int counter = 0;
    for(i=0; i<n; i++) {
        if(ref[i] == true) {
            printf("True: %d = %d\n", ref[i], data[i]);
        }
        if(ref[i] != data[i]) {
            printf("%d: Failed: %d != %d\n", i, ref[i], data[i]);
            counter++;
        }
    }
    if(counter == 0) {
        printf("Test Passed\n-----------------------------------------------------\n");
    }
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
