/*
 * Copyright 1993-2006 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:   
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and 
 * international Copyright laws.  
 *
 * This software and the information contained herein is PROPRIETARY and 
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and 
 * conditions of a Non-Disclosure Agreement.  Any reproduction or 
 * disclosure to any third party without the express written consent of 
 * NVIDIA is prohibited.     
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF 
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.   
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS 
 * OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE 
 * OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE 
 * OR PERFORMANCE OF THIS SOURCE CODE.  
 *
 * U.S. Government End Users.  This source code is a "commercial item" as 
 * that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of 
 * "commercial computer software" and "commercial computer software 
 * documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995) 
 * and is provided to the U.S. Government only as a commercial end item.  
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the 
 * source code with only those rights set forth herein.
 */

#ifndef __NN_KERNEL_H_
#define __NN_KERNEL_H_

#include "cuda_util.h"
// **===----------------- nn_kernel ---------------------===**
//! @param src_data  input data in global memory of src image
//! @param dst_data  output downsampled iamge
// **===------------------------------------------------------------------===**

__global__ void nn_kernel(char* deviceSrc, char* deviceDst,
                          int w1, int h1, int w2, int h2,
                          int x_ratio, int y_ratio, int dst_elems)
{

     int tId = threadIdx.x;
     int g_Idx = blockIdx.x * w2 + tId;

     //Compute the TB start in the global scope of dst image
     int TB_START_ADDR = blockIdx.x * w2;                     //Starting address of data structure  (or each row of dst) --delete
     
     //To access the src image, the indices
     //need to be scaled based on the scale factor

     //For each thread id based on the ratio get the nearest neighbor
     //if( g_Idx < dst_elems/2){
           
     //Populating 2 elemnts by each thread
     int row = (blockIdx.x * y_ratio) >> 16;
     int col1 = (tId * x_ratio)  >> 16 ;

     int col2 = ( (tId + blockDim.x) * x_ratio)  >> 16 ;

     deviceDst[TB_START_ADDR + tId] = deviceSrc[row * w1 + col1];
     
     if( (tId + blockDim.x) < w2){
          deviceDst[TB_START_ADDR + tId + blockDim.x] = deviceSrc[row * w1 + col2];
     }
    
}

#endif 
