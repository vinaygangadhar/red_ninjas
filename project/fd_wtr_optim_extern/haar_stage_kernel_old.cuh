// These kernels correspond to setImageForClassifier, scaleImage_invoke
// and runCascadeClassifier. These kernels are written under the 
// assumption that classifier.txt is changed to store structure of arrays
// instead of array of structures

#define TOTAL_HAAR 2913
#define TOTAL_STAGES 25
#define MAX_HAAR 326
#define WINDOW_WIDTH 24
#define WINDOW_HEIGHT 24
#define MAX_STAGE 7

__global__ void haar_stage_kernel0(uint16_t* haar_index_x, uint16_t* haar_index_y, uint16_t* width, 
        uint16_t* height, int16_t* weight, int16_t* tree_threshold, int16_t* alpha1, int16_t* alpha2, 
        int16_t* threshold_per_stage, int32_t* sum_data, int32_t* sqsum_data, int* haar_per_stage, 
        int16_t haar_num, int16_t num_stages, int image_width, int image_height, bool* bit_vector) {

    int tid = threadIdx.y*blockDim.x + threadIdx.x;
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int offset = row*image_width + col;

    volatile __shared__ int32_t index0_1[MAX_HAAR], index0_2[MAX_HAAR], index0_3[MAX_HAAR], index0_4[MAX_HAAR];
    volatile __shared__ int32_t index1_1[MAX_HAAR], index1_2[MAX_HAAR], index1_3[MAX_HAAR], index1_4[MAX_HAAR];
    volatile __shared__ int32_t index2_1[MAX_HAAR], index2_2[MAX_HAAR], index2_3[MAX_HAAR], index2_4[MAX_HAAR];

    volatile __shared__ int16_t sweight[3*MAX_HAAR];
    volatile __shared__ int16_t stree_threshold[MAX_HAAR];
    volatile __shared__ int16_t salpha1[MAX_HAAR];
    volatile __shared__ int16_t salpha2[MAX_HAAR];
    volatile __shared__ int16_t sthreshold[MAX_STAGE];

    // Index values+offset should be less than (image_height * image_width)
    if(tid < haar_num) { //some branch divergence
        index0_1[tid] = image_width*haar_index_y[3*tid] + haar_index_x[3*tid];
        index0_2[tid] = image_width*haar_index_y[3*tid] + haar_index_x[3*tid] + width[3*tid];
        index0_3[tid] = image_width*haar_index_y[3*tid] + haar_index_x[3*tid] + 
            image_width*height[3*tid];
        index0_4[tid] = image_width*haar_index_y[3*tid] + haar_index_x[3*tid] + 
            image_width*height[3*tid] + width[3*tid];

        index1_1[tid] = image_width*haar_index_y[3*tid+1] + haar_index_x[3*tid+1];
        index1_2[tid] = image_width*haar_index_y[3*tid+1] + haar_index_x[3*tid+1] + width[3*tid+1];
        index1_3[tid] = image_width*haar_index_y[3*tid+1] + haar_index_x[3*tid+1] + 
            image_width*height[3*tid+1];
        index1_4[tid] = image_width*haar_index_y[3*tid+1] + haar_index_x[3*tid+1] + 
            image_width*height[3*tid+1] + width[3*tid+1];

        // This should be done only when third rectangle is present
        // Optimizing by setting only index2_1 to -1
        if(haar_index_x[3*tid+2]==0 && haar_index_y[3*tid+2]==0 && 
                width[3*tid+2]==0 && height[3*tid+2]==0) {
            index2_1[tid] = -1;
        }
        else {
            index2_1[tid] = image_width*haar_index_y[3*tid+2] + haar_index_x[3*tid+2];
            index2_2[tid] = image_width*haar_index_y[3*tid+2] + haar_index_x[3*tid+2] + width[3*tid+2];
            index2_3[tid] = image_width*haar_index_y[3*tid+2] + haar_index_x[3*tid+2] + 
                image_width*height[3*tid+2];
            index2_4[tid] = image_width*haar_index_y[3*tid+2] + haar_index_x[3*tid+2] + 
                image_width*height[3*tid+2] + width[3*tid+2];
        }

        sweight[3*tid] = weight[3*tid];
        sweight[3*tid+1] = weight[3*tid+1];
        sweight[3*tid+2] = weight[3*tid+2];

        stree_threshold[tid] = tree_threshold[tid];
        salpha1[tid] = alpha1[tid];
        salpha2[tid] = alpha2[tid];
    }

    if(tid < num_stages) {
        sthreshold[tid] = threshold_per_stage[tid];
    }
    __syncthreads();

    // Execute remaining section only if row and col are valid
    if((row < (image_height-WINDOW_HEIGHT)) && (col < (image_width-WINDOW_WIDTH))) {
        int sum = 0, result;
        int stage_sum = 0;
        int i, j;
        int num_haars = 0;

        double variance;
        int t;
        int i2, i3, i4;
        i2 = offset + WINDOW_WIDTH - 1;
        i3 = offset + image_width*(WINDOW_HEIGHT-1);
        i4 = i3 + WINDOW_WIDTH - 1;
        long int var = sqsum_data[offset] - sqsum_data[i2] - sqsum_data[i3] + sqsum_data[i4];
        long int mean = sum_data[offset] - sum_data[i2] - sum_data[i3] + sum_data[i4];

        variance = var*WINDOW_WIDTH*WINDOW_HEIGHT - mean*mean;
        if((long int)variance <= 0) {
            variance = 1.0f;
        }
        variance = sqrtf(variance);
        //printf("Kernel: Row: %d, Col: %d, ID: %d: Variance = %d\n", row, col, offset, (int)variance);

        for(i=0; i<num_stages; i++) {
            stage_sum = 0;
            for(j=0; j<haar_per_stage[i]; j++) {
                t = (long int)variance * stree_threshold[num_haars];

                result = sum_data[index0_1[num_haars]+offset] - sum_data[index0_2[num_haars]+offset] -
                    sum_data[index0_3[num_haars]+offset] + sum_data[index0_4[num_haars]+offset];
                sum = result*sweight[3*num_haars];
              
                /*
                if(row == 16 && col == 8) {
                    printf("GPU Stage %d, Haar %d: %d - %d - %d + %d = %d\nweight0 = %d, sum = %d\n", i, j, sum_data[index0_1[num_haars]+offset], 
                            sum_data[index0_2[num_haars]+offset], sum_data[index0_3[num_haars]+offset],
                            sum_data[index0_4[num_haars]+offset], result, sweight[num_haars], sum);
                }*/

                result = sum_data[index1_1[num_haars]+offset] - sum_data[index1_2[num_haars]+offset] -
                    sum_data[index1_3[num_haars]+offset] + sum_data[index1_4[num_haars]+offset];
                sum += result*sweight[3*num_haars+1];

                /*
                if(row == 16 && col == 8) {
                    printf("GPU Stage %d, Haar %d: %d - %d - %d + %d = %d\nweight1 = %d, sum = %d\n", i, j, sum_data[index1_1[num_haars]+offset], 
                            sum_data[index1_2[num_haars]+offset], sum_data[index1_3[num_haars]+offset],
                            sum_data[index1_4[num_haars]+offset], result, sweight[3*num_haars+1], sum);
                }*/

                // Branch Divergence for Haar features that do not have 3rd rectangle
                if(index2_1[num_haars] == -1)  {
                    result = 0;
                }
                else  {
                    result = sum_data[index2_1[num_haars]+offset] - sum_data[index2_2[num_haars]+offset] -
                        sum_data[index2_3[num_haars]+offset] + sum_data[index2_4[num_haars]+offset];
                }

                sum += result*sweight[3*num_haars+2];

                if(sum >= t) 
                    stage_sum += salpha2[num_haars];
                else 
                    stage_sum += salpha1[num_haars];
                num_haars++;
            }
            if(stage_sum < 0.4*sthreshold[i]) { 
                bit_vector[row*(image_width-WINDOW_WIDTH)+col] = false;
                //printf("False: Vec ID = %d, Kernel: Row = %d, Col = %d: stage_sum = %ld < %d\n", row*(image_width-WINDOW_WIDTH)+col, row, col, stage_sum, (int)(0.4*sthreshold[i]));
            }
        }
    }
}
// Corner cases:
/*
   1. 3rd rectangle may not be present. Should try to handle it without branch divergence.- Now implemented with branch div
   2. Should we really consider sqsum? Waste of 55x55 pixels- considering anyways
   3. How to indicate some x,y passed all the stages in this kernel? - Using a huge array of size around 32kB
 */













