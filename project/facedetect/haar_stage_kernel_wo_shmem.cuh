// These kernels correspond to setImageForClassifier, scaleImage_invoke
// and runCascadeClassifier. These kernels are written under the 
// assumption that classifier.txt is changed to store structure of arrays
// instead of array of structures


__global__ void haar_stage_kernel0(uint16_t* haar_index_x, uint16_t* haar_index_y, 
                                    uint16_t* width, uint16_t* height, 
                                    int16_t* weight, int16_t* tree_threshold, 
                                    int16_t* alpha1, int16_t* alpha2, 
                                    int16_t* threshold_per_stage, 
                                    int32_t* sum_data, int32_t* sqsum_data, 
                                    int* haar_per_stage, 
                                    int16_t haar_num, int16_t num_stages, 
                                    int image_width, int image_height, 
                                    bool* bit_vector
                                   )
{

   
   //int tId = threadIdx.y * blockDim.x + threadIdx.x;
   int row = blockIdx.y * blockDim.y + threadIdx.y;
   int col = blockIdx.x * blockDim.x + threadIdx.x;
   int offset = row * image_width + col;

   // Execute remaining section only if row and col are valid
   if( (row < (image_height - WINDOW_HEIGHT)) && (col < (image_width - WINDOW_WIDTH)) ) { //&&
            //(bit_vector[row * (image_width - WINDOW_WIDTH) + col] == true) ) {

       int sum = 0, result;
       int stage_sum = 0;
       int i, j;
       int num_haars = 0;

       double variance;
       int t;
       int i2, i3, i4;

       i2 = offset + WINDOW_WIDTH - 1;
       i3 = offset + image_width * (WINDOW_HEIGHT - 1);
       i4 = i3 + WINDOW_WIDTH - 1;
       int var = sqsum_data[offset] - sqsum_data[i2] - sqsum_data[i3] + sqsum_data[i4];
       int mean = sum_data[offset] - sum_data[i2] - sum_data[i3] + sum_data[i4];

       variance = var * WINDOW_WIDTH * WINDOW_HEIGHT - mean * mean;
       if((int)variance <= 0) {
           variance = 1.0f;
       }
       variance = sqrtf(variance);

       // Four corners for each rectangle
       int32_t index0[4], index1[4], index2[4];
       
       for(i = 0; i < num_stages; i++) {
           stage_sum = 0;

           for(j = 0; j < haar_per_stage[i]; j++) {
               // Four corners of first rectangle
               index0[0] = image_width * haar_index_y[3*num_haars] + haar_index_x[3*num_haars] + offset; 
               index0[1] = image_width * haar_index_y[3*num_haars] + haar_index_x[3*num_haars] + width[3*num_haars] + offset; 
               index0[2] = image_width * haar_index_y[3*num_haars] + haar_index_x[3*num_haars] + image_width*height[3*num_haars] + offset;
               index0[3] = image_width * haar_index_y[3*num_haars] + haar_index_x[3*num_haars] + image_width*height[3*num_haars] + width[3*num_haars] + offset; 

               // Four corners of second rectangle
               index1[0] = image_width * haar_index_y[3*num_haars+1] + haar_index_x[3*num_haars+1] + offset; 
               index1[1] = image_width * haar_index_y[3*num_haars+1] + haar_index_x[3*num_haars+1] + width[3*num_haars+1] + offset; 
               index1[2] = image_width * haar_index_y[3*num_haars+1] + haar_index_x[3*num_haars+1] + image_width*height[3*num_haars+1] + offset;
               index1[3] = image_width * haar_index_y[3*num_haars+1] + haar_index_x[3*num_haars+1] + image_width*height[3*num_haars+1] + width[3*num_haars+1] + offset;

               if((haar_index_x[3*num_haars+2] == 0) && (haar_index_y[3*num_haars+2] ==0) && 
                       (width[3*num_haars+2] == 0) && (height[3*num_haars+2] == 0) ) {
                   index2[0] = -1;
               } else {
                   index2[0] = image_width * haar_index_y[3*num_haars+2] + haar_index_x[3*num_haars+2] + offset; 
                   index2[1] = image_width * haar_index_y[3*num_haars+2] + haar_index_x[3*num_haars+2] + width[3*num_haars+2] + offset; 
                   index2[2] = image_width * haar_index_y[3*num_haars+2] + haar_index_x[3*num_haars+2] + image_width*height[3*num_haars+2] + offset;
                   index2[3] = image_width * haar_index_y[3*num_haars+2] + haar_index_x[3*num_haars+2] + image_width*height[3*num_haars+2] + width[3*num_haars+2] + offset;
               }

               t = (int)variance * tree_threshold[num_haars];

               result = sum_data[index0[0]] - sum_data[index0[1]] -
                           sum_data[index0[2]] + sum_data[index0[3]];
               sum = result * weight[3 * num_haars];
             
               result = sum_data[index1[0]] - sum_data[index1[1]] -
                           sum_data[index1[2]] + sum_data[index1[3]];
               sum += result*weight[3 * num_haars + 1];

              // Branch Divergence for Haar features that do not have 3rd rectangle
               if(index2[0] == -1)  
               {
                   result = 0;
               } else 
               {
                   result = sum_data[index2[0]] - sum_data[index2[1]] -   
                              sum_data[index2[2]] + sum_data[index2[3]];
               }

               sum += result * weight[3 * num_haars + 2];

               if(sum >= t) 
                   stage_sum += alpha2[num_haars];
               else 
                   stage_sum += alpha1[num_haars];
               num_haars++;
           }// end filter for
           
           if(stage_sum < (0.4*threshold_per_stage[i])) { 
               bit_vector[row * (image_width - WINDOW_WIDTH) + col] = false;
           }

       }//end stages for
   }//end if cond
}

// Corner cases:
/*
   1. 3rd rectangle may not be present. Should try to handle it without branch divergence.- Now implemented with branch div
   2. Should we really consider sqsum? Waste of 55x55 pixels- considering anyways
   3. How to indicate some x,y passed all the stages in this kernel? - Using a huge array of size around 32kB
 */













