#ifndef __NN_RS_KERNEL_H__                                                                        
#define __NN_RS_KERNEL_H__                                                                        
                                                                                                  
#include <stdlib.h>                                                                               
                                                                                                  
__global__ void rowscan_nn_kernel(unsigned char* src, unsigned char* dst, int* sum, int* sqsum,   
                                   int src_w, int src_h, int dst_w, int dst_h,                    
                                   int x_ratio, int y_ratio, int tb_elems)                        
                                                                                                  
//__global__ void rowscan_nn_kernel(unsigned char* src, int* sum, int* sqsum,                     
//                                   int src_w, int src_h, int dst_w, int dst_h,                  
//                                   int x_ratio, int y_ratio, int tb_elems)                      
{                                                                                                 
   volatile __shared__ int row[SHARED];                                                         
                                                                                                  
   //Nearest Neighbor Kernel                                                                    
   int tId = threadIdx.x;                                                                       
   int g_Idx = blockIdx.x * dst_w + tId;                                                        
                                                                                                
   int offset = 1;                                                                              
   int sq_start = tb_elems + 1;           //For square sum in Smem                              
   int offset_tId = tId + blockDim.x;                                                           
                                                                                                
   int row_src = (blockIdx.x * y_ratio) >> 16;                                                  
   int col1_src = (tId * x_ratio)  >> 16 ;                                                      
   int col2_src = ( (tId + blockDim.x) * x_ratio)  >> 16 ;                                      
                                                                                                
//--DELETE LATER                                                                                
                                                                                                
   int TB_START_ADDR = blockIdx.x * dst_w;                                                     
                                                                                               
   //Populating 2 elemnts by each thread                                                       
                                                                                               
   dst[TB_START_ADDR + tId] = src[row_src * src_w + col1_src];                                 
   if( (offset_tId) < dst_w){                                                                  
        dst[TB_START_ADDR + offset_tId] = src[row_src * src_w + col2_src];                     
   }                                                                                           
//--DELETE LATER                                                                                  
                                                                                                
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
   for (int stage_threads = tb_elems>>1; stage_threads > 0; stage_threads >>= 1)              
   {                                                                                          
     __syncthreads();                                                                         
                                                                                              
     if (tId < stage_threads)                                                                 
     {                                                                                        
       int bi = offset * (2*tId+2) - 1;         //Each thread start                           
       int ai = offset * (2*tId+1) - 1;         //Neighbor or mate thread                     
                                                                                              
       row[bi] += row[ai];                                                                    
       row[sq_start + bi] += row[sq_start + ai];                                              
     }                                                                                        
                                                                                              
     offset <<= 1;                                                                            
   }                                                                                          
                                                                                              
   //DownSweep                                                                                
   if (tId == 0)                                                                              
   {                                                                                          
     row[tb_elems] = row[tb_elems - 1];          //For inclusive scan                         
     row[tb_elems - 1] = 0;                                                                   
     row[sq_start + tb_elems] = row[sq_start + tb_elems - 1];                                 
     row[sq_start + tb_elems - 1] = 0;                                                        
   }                                                                                          
                                                                                              
   for (int stage_threads = 1; stage_threads < tb_elems; stage_threads *= 2 )                 
   {                                                                                          
                                                                                              
     offset >>=1 ;                                                                            
     __syncthreads();                                                                         
                                                                                              
     if (tId < stage_threads)                                                                 
     {                                                                                        
                                                                                              
       int ai = offset * (2*tId+1) - 1;                                                       
       int bi = offset * (2*tId+2) - 1;                                                       
                                                                                              
       int dummy = row[ai];                      //Mate elements to dummy                     
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
                                                                                      
//Column Scan                                                                         
__global__ void column_scan_kernel(int* sum, int* sqsum, int dst_h, int tb_elems)     
{                                                                                     
   volatile __shared__ int column[SHARED];                                             
                                                                                      
   int tId = threadIdx.x;                                                              
   int g_Idx = tId * gridDim.x + blockIdx.x;    //To index into memory data structure  
                                                                                       
   int offset = 1;                                                                     
   int offset_gIdx = g_Idx + blockDim.x * gridDim.x; //To index into offset element    
   int sq_start = tb_elems + 1;                                                        
   int offset_tId = tId + blockDim.x;                                                  
                                                                                       
   column[tId] = sum[g_Idx] ;                                                          
   column[offset_tId] = (offset_tId < dst_h) ? sum[offset_gIdx] : 0;                   
   column[sq_start + tId] =  sqsum[g_Idx];                                             
   column[sq_start + offset_tId] =  (offset_tId < dst_h) ? sqsum[offset_gIdx] : 0;     
                                                                                       
   //Upsweep                                                                           
   for (int stage_threads = tb_elems>>1; stage_threads > 0; stage_threads >>= 1)       
   {                                                                                   
      __syncthreads();                                                                
                                                                                    
      if (tId < stage_threads)                                                        
      {                                                                               
                                                                                      
        int ai = offset * (2*tId+1) - 1;    //neighbor mate thread                    
        int bi = offset * (2*tId+2) - 1;   //Each thread                              
                                                                                      
        column[bi] += column[ai];                                                     
        column[sq_start + bi] += column[sq_start + ai];
      }
                                 
      offset <<= 1;                                                                   
   }                                                                                 
                                                                                    
   if (tId == 0)                                                                     
   {                                                                                 
     column[tb_elems] = column[tb_elems-1];                                          
     column[tb_elems - 1] = 0;                                                       
     column[sq_start + tb_elems] = column[sq_start + tb_elems - 1];                  
     column[sq_start + tb_elems - 1] = 0;                                            
   }                                                                                 
                                                                                     
   //DownSweep                                                                       
   for (int stage_threads = 1; stage_threads < tb_elems; stage_threads *= 2 )        
   {                                                                                 
                                                                                     
     offset >>= 1;                                                                   
                                                                                     
     __syncthreads();                                                                
                                                                                     
     if (tId < stage_threads)                                                        
     {                                                                               
                                                                                     
       int ai = offset * (2*tId+1) - 1;       //Mate thread                          
       int bi = offset * (2*tId+2) - 1;       //Each thread                          
                                                                                     
       int dummy = column[ai];                                                       
       column[ai] = column[bi];                                                      
       column[bi] += dummy;                                                          
      int dummy_sq = column[sq_start + ai];                          
      column[sq_start + ai] = column[sq_start + bi];                 
      column[sq_start + bi] += dummy_sq;                             
    }                                                                
  }                                                                  
                                                                     
   __syncthreads();                                                   
                                                                      
   sum[g_Idx] = column[tId + 1];                                      
   sqsum[g_Idx] = column[sq_start + tId + 1];                         
                                                                      
   if (offset_tId < dst_h)                                            
   {                                                                  
     sum[offset_gIdx] = column[offset_tId + 1];                       
     sqsum[offset_gIdx] = column[sq_start + offset_tId + 1];          
   }                                                                  
                                                                     
}                                                                    
                                                                     
#endif                                                               
                                                                     
                                                                     
                                                                                    
                                                                                          


