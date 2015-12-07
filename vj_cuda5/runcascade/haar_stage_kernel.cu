// These kernels correspond to setImageForClassifier, scaleImage_invoke
// and runCascadeClassifier. These kernels are written under the 
// assumption that classifier.txt is changed to store structure of arrays
// instead of array of structures

#define MAX_HAAR 324
#define WINDOW_WIDTH 24
#define WINDOW_HEIGHT 24
#define MAX_STAGE 7

__global__ void haar_stage_kernel0(uint16_t* haar_index_x, uint16_t* haar_index_y, uint16_t* width, uint16_t* height, uint16_t* weight, uint16_t* tree_threshold, uint16_t* alpha1, uint16_t* alpha2, uint16_t* threshold_per_stage, uint32_t* sum_data, uint32_t* sqsum_data, uint8_t* haar_per_stage, uint16_t haar_num, uint16_t num_stages, bool* bit_vector) {

    int tid = threadIdx.x*blockDim.x + threadIdx.y;
    int offset = blockIdx.x*blockDim.x*blockDim.y + tid;
    __shared__ int16_t index0_1[MAX_HAAR], index0_2[MAX_HAAR], index0_3[MAX_HAAR], index0_4[MAX_HAAR];
    __shared__ int16_t index1_1[MAX_HAAR], index1_2[MAX_HAAR], index1_3[MAX_HAAR], index1_4[MAX_HAAR];
    __shared__ int16_t index2_1[MAX_HAAR], index2_2[MAX_HAAR], index2_3[MAX_HAAR], index2_4[MAX_HAAR];

    __shared__ uint16_t sweight[3*MAX_HAAR];
    __shared__ uint16_t stree_threshold[MAX_HAAR];
    __shared__ uint16_t salpha1[MAX_HAAR];
    __shared__ uint16_t salpha2[MAX_HAAR];
    __shared__ uint16_t sthreshold[MAX_STAGE];

    if(tid < haar_num) { //some branch divergence
        index0_1[tid] = offset + WINDOW_WIDTH*haar_index_y[3*tid] + haar_index_x[3*tid];
        index0_2[tid] = offset + index0_1[tid] + width[3*tid] - 1;
        index0_3[tid] = offset + index0_1[tid] + WINDOW_WIDTH*(height[3*tid]-1);
        index0_4[tid] = offset + index0_3[tid] + width[3*tid] - 1;

        index1_1[tid] = offset + WINDOW_WIDTH*haar_index_y[3*tid+1] + haar_index_x[3*tid+1];
        index1_2[tid] = offset + index1_1[tid] + width[3*tid+1] - 1;
        index1_3[tid] = offset + index1_1[tid] + WINDOW_WIDTH*(height[3*tid+1]-1);
        index1_4[tid] = offset + index1_3[tid] + width[3*tid+1] - 1;

        // This should be done only when third rectangle is present
        // Optimizing by setting only index2_1 to -1
        if(haar_index_x[3*tid+2]==0 && haar_index_y[3*tid+2]==0 && width[3*tid+2]==0 && height[3*tid+2]==0) {
            index2_1[tid] = -1;
        }
        else {
            index2_1[tid] = WINDOW_WIDTH*haar_index_y[3*tid+2] + haar_index_x[3*tid+2];
            index2_2[tid] = index2_1[tid] + width[3*tid+2] - 1;
            index2_3[tid] = index2_1[tid] + WINDOW_WIDTH*(height[3*tid+2]-1);
            index2_4[tid] = index2_3[tid] + width[3*tid+2] - 1;
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

    long int sum = 0, result;
    long int stage_sum = 0;
    int i, j;
    int num_haars = 0;

    float t, variance;
    int i2, i3, i4;
    i2 = offset + WINDOW_WIDTH - 1;
    i3 = offset + WINDOW_WIDTH*(WINDOW_HEIGHT-1);
    i4 = i3 + WINDOW_WIDTH - 1;
    int var = sqsum_data[offset] - sqsum_data[i2] - sqsum_data[i3] + sqsum_data[i4];
    int mean = sum_data[offset] - sum_data[i2] - sum_data[i3] + sum_data[i4];

    variance = sqrtf(var*WINDOW_WIDTH*WINDOW_HEIGHT - mean*mean);
    if(variance <= 0.0f) {
        variance = 1.0f;
    }

    for(i=0; i<num_stages; i++) {
        for(j=0; j<haar_per_stage[i]; j++) {
            t = variance * stree_threshold[num_haars];
            result = sum_data[index0_4[num_haars]] + sum_data[index0_1[num_haars]] - sum_data[index0_2[num_haars]] - sum_data[index0_3[num_haars]];
            sum += result*sweight[3*num_haars];
            result = sum_data[index1_4[num_haars]] + sum_data[index1_1[num_haars]] - sum_data[index1_2[num_haars]] - sum_data[index1_3[num_haars]];
            sum += sum*sweight[3*num_haars+1];

            // Branch Divergence for Haar features that do not have 3rd rectangle
            if(index2_1[num_haars] == -1) 
                result = 0;
            else 
                result = sum_data[index2_4[num_haars]] + sum_data[index2_1[num_haars]] - sum_data[index2_2[num_haars]] - sum_data[index2_3[num_haars]];
            sum += result*sweight[3*num_haars+1];

            if(sum >= (int)t) 
                stage_sum += salpha2[num_haars];
            else 
                stage_sum += salpha1[num_haars];
            num_haars++;
        }
        if(stage_sum < 0.4*sthreshold[i]) {
            bit_vector[offset] = true;
        }
    }
}
// Corner cases:
/*
   1. 3rd rectangle may not be present. Should try to handle it without branch divergence.- Now implemented with branch div
   2. Should we really consider sqsum? Waste of 55x55 pixels- considering anyways
   3. How to indicate some x,y passed all the stages in this kernel? - Using a huge array of size around 32kB
 */













