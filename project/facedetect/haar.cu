/*
 *  TU Eindhoven
 *  Eindhoven, The Netherlands
 *
 *  Name            :   haar.cpp
 *
 *  Author          :   Francesco Comaschi (f.comaschi@tue.nl)
 *
 *  Date            :   November 12, 2012
 *
 *  Function        :   Haar features evaluation for face detection
 *
 *  History         :
 *      12-11-12    :   Initial version.
 *
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the
 * Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program;  If not, see <http://www.gnu.org/licenses/>
 *
 * In other words, you are welcome to use, share and improve this program.
 * You are forbidden to forbid anyone else to use, share and improve
 * what you give them.   Happy coding!
 */

#include "haar.h"
#include "image.h"
#include <stdio.h>
#include <stdint.h>
#include "stdio-wrapper.h"

/* include the gpu functions */
#include "haar_stage.cuh"
#include "gpu_nn_integral.cuh"
#include "optim.cuh"
#include "cuda_util.h"

/* TODO: use matrices */
/* classifier parameters */
/************************************
 * Notes:
 * To paralleism the filter,
 * these monolithic arrays may
 * need to be splitted or duplicated
 ***********************************/
static int *stages_array;
static int *rectangles_array;
static int *weights_array;
static int *alpha1_array;
static int *alpha2_array;
static int *tree_thresh_array;
static int *stages_thresh_array;
static int **scaled_rectangles_array;

static uint16_t *hindex_x;
static uint16_t *hindex_y;
static uint16_t *hwidth;
static uint16_t *hheight;
static int16_t *hweights_array;
static int16_t *halpha1_array;
static int16_t *halpha2_array;
static int16_t *htree_thresh_array;
static int16_t *hstages_thresh_array;
static int *hstages_array;
static bool *bit_vector;

int clock_counter = 0;
float n_features = 0;

int iter_counter = 0;

///FUNCTION DECLARATIONS//
/* compute integral images */
void integralImageOnHost( MyImage *src, MyIntImage *sum, MyIntImage *sqsum );

/* scale down the image */
void ScaleImage_Invoker( myCascade* _cascade, float _factor, int sum_row, int sum_col, std::vector<MyRect>& _vec);

/* compute scaled image */
void nearestNeighborOnHost(MyImage *src, MyImage *dst);

/* rounding function */
inline  int  myRound( float value )
{
    return (int)(value + (value >= 0 ? 0.5 : -0.5));
}

/*******************************************************
 * Function: detectObjects
 * Description: It calls all the major steps
 ******************************************************/

std::vector<MyRect> detectObjects( MyImage* _img, MySize minSize, MySize maxSize, myCascade* cascade,
        float scaleFactor, int minNeighbors, std::fstream& ofs)
{

    /* group overlaping windows */
    const float GROUP_EPS = 0.4f;
    /* pointer to input image */
    MyImage *img = _img;
    /***********************************
     * create structs for images
     * see haar.h for details 
     * img1: normal image (unsigned char)
     * sum1: integral image (int)
     * sqsum1: square integral image (int)
     **********************************/
    MyImage image1Obj;
    MyIntImage sum1Obj;
    MyIntImage sqsum1Obj;
   
    /* pointers for the created structs */
    MyImage *img1 = &image1Obj;
    MyIntImage *sum1 = &sum1Obj;
    MyIntImage *sqsum1 = &sqsum1Obj;

    /**************************************/
    //Timing related
    cudaError_t error;
    cudaEvent_t cpu_start;
    cudaEvent_t cpu_stop;
    cudaEvent_t gpu_cpy_start;
    cudaEvent_t gpu_cpy_stop;

    float gpu_cpyTime;
    float gpu_cpyTotal;

    float gpu_kernTime;
    float cpu_kernTime;
    float cpu_stageTotal;
    float gpu_stageTotal;
   
    float cpu_msecTotal;
    float gpu_msecTotal;

    //CUDA Events 
    error = cudaEventCreate(&cpu_start);
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to create start event (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
    
    error = cudaEventCreate(&cpu_stop);
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to create stop event (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    
    }

    error = cudaEventCreate(&gpu_cpy_start);
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to create start event (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
    
    error = cudaEventCreate(&gpu_cpy_stop);
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to create stop event (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    
    }

    /**************************************/

    /********************************************************
     * allCandidates is the preliminaray face candidate,
     * which will be refined later.
     *
     * std::vector is a sequential container 
     * http://en.wikipedia.org/wiki/Sequence_container_(C++) 
     *
     * Each element of the std::vector is a "MyRect" struct 
     * MyRect struct keeps the info of a rectangle (see haar.h)
     * The rectangle contains one face candidate 
     *****************************************************/
    std::vector<MyRect> allCandidates;
    std::vector<MyRect> faces; // For data from GPU

    /* scaling factor */
    float factor;

    /* maxSize */
    if( maxSize.height == 0 || maxSize.width == 0 )
    {
        maxSize.height = img->height;
        maxSize.width = img->width;
    }

    /* window size of the training set */
    MySize winSize0 = cascade->orig_window_size;

    /* malloc for img1: unsigned char */
    //createImage(img->width, img->height, img1);
    createImagePinned(img->width, img->height, img1);
    /* malloc for sum1: unsigned char */
    createSumImage(img->width, img->height, sum1);
    /* malloc for sqsum1: unsigned char */
    createSumImage(img->width, img->height, sqsum1);
    
    /****************************************************
      Setting up the data for GPU Kernels -- NN-II
    ***************************************************/
    //NN_II RELATED

    //Src image on Device
    MyImage deviceimg;
    deviceimg.width = img->width; deviceimg.height = img->height; 
    deviceimg.maxgrey = img->maxgrey; deviceimg.flag = img->flag;

    //Allocation for nn_ii device image
    nniiAllocateImgOnDevice(&deviceimg.data, deviceimg.width * deviceimg.height);

    //Copy the src image to device
    
    cudaEventRecord(gpu_cpy_start, 0);
    
    nniiCopyImgToDevice(img->data, deviceimg.data, deviceimg.width * deviceimg.height); 
 
    cudaEventRecord(gpu_cpy_stop, 0);
    cudaEventSynchronize(gpu_cpy_stop);

    cudaEventElapsedTime(&gpu_cpyTime, gpu_cpy_start, gpu_cpy_stop);
    gpu_cpyTotal = gpu_cpyTime;
    printf("\n\tHost to Device Src Image Copy Time: %f ms\n", gpu_cpyTime);

    /****************************************************
      Setting up the data for GPU Kernels -- HAAR
    ***************************************************/

    //HAAR RELATED
    uint16_t* dindex_x; uint16_t* dindex_y;
    uint16_t* dwidth;  uint16_t* dheight;
    int16_t* dweights_array;
    int16_t* dalpha1_array;  int16_t* dalpha2_array;
    int16_t* dtree_thresh_array; int16_t* dstages_thresh_array;
    int* dstages_array;

    //Bit vector creation
    bit_vector = (bool*) malloc(img->width * img->height * sizeof(bool));

    ////Allocation for GPU Data Structures
    
    haarAllocateOnDevice(&dindex_x, &dindex_y, 
                         &dwidth, &dheight,
                         &dweights_array, &dalpha1_array,
                         &dalpha2_array, &dtree_thresh_array,
                         &dstages_thresh_array, &dstages_array  
                        );
   

   //Copy the contents read from class.txt to device structures
    cudaEventRecord(gpu_cpy_start, 0);
    
    haarCopyToDevice(dindex_x, hindex_x, dindex_y, hindex_y,
                     dwidth, hwidth, dheight, hheight, 
                     dweights_array, hweights_array,   
                     dalpha1_array, halpha1_array, 
                     dalpha2_array, halpha2_array,
                     dtree_thresh_array,  htree_thresh_array,
                     dstages_thresh_array, hstages_thresh_array,
                     dstages_array, hstages_array
                    );

    
    cudaEventRecord(gpu_cpy_stop, 0);
    cudaEventSynchronize(gpu_cpy_stop);

    cudaEventElapsedTime(&gpu_cpyTime, gpu_cpy_start, gpu_cpy_stop);
    gpu_cpyTotal += gpu_cpyTime;
    printf("\tClassifier Info Copy Time: %f ms\n", gpu_cpyTime);

    /****************************************************
      Setting up DONE 
    ***************************************************/

    /* initial scaling factor */
    factor = 1;

    /* iterate over the image pyramid */
    for( factor = 1; ; factor *= scaleFactor )
    {
        /* iteration counter */
        iter_counter++;

        /* size of the image scaled up */
        MySize winSize = { myRound(winSize0.width*factor), myRound(winSize0.height*factor) };

        /* size of the image scaled down (from bigger to smaller) */
        MySize sz = { ( img->width/factor ), ( img->height/factor ) };

        /* difference between sizes of the scaled image and the original detection window */
        MySize sz1 = { sz.width - winSize0.width, sz.height - winSize0.height };

        /* if the actual scaled image is smaller than the original detection window, break */
        if( sz1.width < 0 || sz1.height < 0 )
            break;

        /* if a minSize different from the original detection window is specified, continue to the next scaling */
        if( winSize.width < minSize.width || winSize.height < minSize.height )
            continue;

        /*************************************
         * Set the width and height of 
         * img1: normal image (unsigned char)
         * sum1: integral image (int)
         * sqsum1: squared integral image (int)
         * see image.c for details
         ************************************/
        setImage(sz.width, sz.height, img1);
        setSumImage(sz.width, sz.height, sum1);
        setSumImage(sz.width, sz.height, sqsum1);

        //Sum and SqSum on Device
        MyIntImage devicesum1, devicesqsum1;
        devicesum1.width = sum1->width; devicesum1.height = sum1->height; 
        devicesqsum1.width = sqsum1->width; devicesqsum1.height = sqsum1->height;

        //Transpose sum and sqsum on Device
        MyIntImage transpose_dsum1, transpose_dsqsum1;
        transpose_dsum1.width = sum1->width; transpose_dsum1.height = sum1->height; 
        transpose_dsqsum1.width = sqsum1->width; transpose_dsqsum1.height = sqsum1->height;

        //Allocate Sum and Sqsum and transposes on device
        nniiAllocateOnDevice(&devicesum1.data, &devicesqsum1.data, 
                              &transpose_dsum1.data, &transpose_dsqsum1.data, 
                              devicesum1.width * devicesum1.height
                            );

        printf("\n\tIteration:= %d\n \tDownsampling-->  New Image Size:   Width: %d, Height: %d\n",
                  iter_counter, sz.width, sz.height);
   
        //CPU CALL
        printf("\tNN and II on CPU Started\n");
        
        error = cudaEventRecord(cpu_start, NULL);
        if (error != cudaSuccess)
        {
            fprintf(stderr, "Failed to record start event (error code %s)!\n", cudaGetErrorString(error));
            exit(EXIT_FAILURE);
        }
     
        //NN's downsampled image is passed to II 
        nearestNeighborOnHost(img, img1);
        integralImageOnHost(img1, sum1, sqsum1);
        
        // Record the stop event
        error = cudaEventRecord(cpu_stop, NULL);
        if (error != cudaSuccess)
        {
            fprintf(stderr, "Failed to record stop event (error code %s)!\n", cudaGetErrorString(error));
            exit(EXIT_FAILURE);
        }

        // Wait for the stop event to complete
        error = cudaEventSynchronize(cpu_stop);
        if (error != cudaSuccess)
        {
            fprintf(stderr, "Failed to synchronize on the stop event (error code %s)!\n", cudaGetErrorString(error));
            exit(EXIT_FAILURE);
        }

        error = cudaEventElapsedTime(&cpu_kernTime, cpu_start, cpu_stop);
        if (error != cudaSuccess)
        {
            fprintf(stderr, "Failed to get time elapsed between events (error code %s)!\n", cudaGetErrorString(error));
            exit(EXIT_FAILURE);
        }

        cpu_stageTotal = cpu_kernTime;
        printf("\tNN and II on CPU complete--> Execution time: %f ms\n", cpu_kernTime);

        /***************************************************
        * Compute-intensive step:
        * At each scale of the image pyramid,
        * compute a new integral and squared integral image
        ***************************************************/
        //GPU CALL

        float kernelTime = nn_integralImageOnDevice(img, deviceimg.data, 
                                                      devicesum1.data, devicesqsum1.data,
                                                      transpose_dsum1.data, transpose_dsqsum1.data,
                                                      devicesum1.width, devicesum1.height
                                                    );

        gpu_kernTime = kernelTime;
        gpu_stageTotal = gpu_kernTime;
        ///////////////////////////////////////////////////////////
        ///// RUNCASCADE KERNEL RELATED //////////////////////////
        ///////////////////////////////////////////////////////////
        
        /* sets images for haar classifier cascade */
        /**************************************************
         * Note:
         * Summing pixels within a haar window is done by
         * using four corners of the integral image:
         * http://en.wikipedia.org/wiki/Summed_area_table
         * 
         * This function loads the four corners,
         * but does not do compuation based on four coners.
         * The computation is done next in ScaleImage_Invoker
         *************************************************/

        /*-------------------------------------------------------------------
          Starting timer for Runcascade Kernels comparison
          -------------------------------------------------------------------*/
        
        //// Calculate CPU time////////
       
        printf("\n\tCascade Classifier on CPU Started\n");
        
        error = cudaEventRecord(cpu_start, NULL);
        if (error != cudaSuccess)
        {
            fprintf(stderr, "Failed to record start event (error code %s)!\n", cudaGetErrorString(error));
            exit(EXIT_FAILURE);
        }
           
        setImageForCascadeClassifier( cascade, sum1, sqsum1);
      
        /* print out for each scale of the image pyramid */

        /****************************************************
         * Process the current scale with the cascaded fitler.
         * The main computations are invoked by this function.
         * Optimization oppurtunity:
         * the same cascade filter is invoked each time
         ***************************************************/
        ScaleImage_Invoker(cascade, factor, sum1->height, sum1->width,
                allCandidates);
       
        // Record the stop event
        error = cudaEventRecord(cpu_stop, NULL);
        if (error != cudaSuccess)
        {
            fprintf(stderr, "Failed to record stop event (error code %s)!\n", cudaGetErrorString(error));
            exit(EXIT_FAILURE);
        }

        // Wait for the stop event to complete
        error = cudaEventSynchronize(cpu_stop);
        if (error != cudaSuccess)
        {
            fprintf(stderr, "Failed to synchronize on the stop event (error code %s)!\n", cudaGetErrorString(error));
            exit(EXIT_FAILURE);
        }

        error = cudaEventElapsedTime(&cpu_kernTime, cpu_start, cpu_stop);
        if (error != cudaSuccess)
        {
            fprintf(stderr, "Failed to get time elapsed between events (error code %s)!\n", cudaGetErrorString(error));
            exit(EXIT_FAILURE);
        }
    
        cpu_stageTotal += cpu_kernTime;
        printf("\tCascade Classifier on CPU Complete--> Execution time: %f ms\n", cpu_kernTime);
       
        /*--------------------------------------------------------------------------------------------*/

        ////////////////////////////////
        // CASCADE CLASSIIER ON GPU  //
        ///////////////////////////////
       
        ///Initial Data Setting//
        int bitvec_width = img1->width - cascade->orig_window_size.width;
        int bitvec_height = img1->height - cascade->orig_window_size.height;
  
        //Allocate host bit vector
        bool* hbit_vector;
         
        createBitVector(&hbit_vector, bitvec_width, bitvec_height);
        //hbit_vector = (bool*) malloc(bitvec_width * bitvec_height * sizeof(bool));
         
        int check;

        bool* dbit_vector;
        check = CUDA_CHECK_RETURN(cudaMalloc((void**)&dbit_vector, bitvec_width * bitvec_height * sizeof(bool)), __FILE__, __LINE__);
        if( check != 0){
           std::cerr << "Error: CudaMalloc not successfull for device bit vector" << std::endl;
           exit(1);
        }

        int i;
        for(i = 0; i < (bitvec_width * bitvec_height); i++) {
            hbit_vector[i] = true;
        }
       
        //Copy the Host bitvector to Device bit vector
        cudaEventRecord(gpu_cpy_start, 0);
       
        cudaMemcpy(dbit_vector, hbit_vector, bitvec_width * bitvec_height * sizeof(bool), cudaMemcpyHostToDevice);
         
        cudaEventRecord(gpu_cpy_stop, 0);
        cudaEventSynchronize(gpu_cpy_stop);

        cudaEventElapsedTime(&gpu_cpyTime, gpu_cpy_start, gpu_cpy_stop);

        gpu_stageTotal += gpu_cpyTime;
        printf("\n\tCC: To Device BitVector Copy Time: %f ms", gpu_cpyTime);
        
        //Call the Classifier on GPU Now
        kernelTime = cascadeClassifierOnDevice(img1,  
                                   bitvec_width, bitvec_height,
                                   dindex_x, dindex_y, dwidth, dheight, 
                                   dweights_array, dtree_thresh_array, 
                                   dalpha1_array, dalpha2_array, 
                                   dstages_thresh_array, 
                                   devicesum1.data, devicesqsum1.data, 
                                   dstages_array, 
                                   dbit_vector
                                 );
        

        gpu_stageTotal += kernelTime;

        //CUDA MemCPy back to host vector
        cudaEventRecord(gpu_cpy_start, 0);
       
        cudaMemcpy(hbit_vector, dbit_vector, bitvec_width * bitvec_height * sizeof(bool), cudaMemcpyDeviceToHost);
         
        cudaEventRecord(gpu_cpy_stop, 0);
        cudaEventSynchronize(gpu_cpy_stop);

        cudaEventElapsedTime(&gpu_cpyTime, gpu_cpy_start, gpu_cpy_stop);
        gpu_stageTotal += gpu_cpyTime;

        printf("\tCC: To Host BitVector Copy Time: %f ms\n", gpu_cpyTime);
        
        //Recgnize the rectangles and push to Faces Structure
        int x, y;
        for(y = 0;  y < bitvec_height; y++) {
            for(x = 0; x < bitvec_width; x++) {
                if(hbit_vector[y * bitvec_width + x] == true) {
                    MyRect r = {myRound(x * factor), myRound(y * factor), winSize.width, winSize.height};
                    faces.push_back(r);
                }
            }
        }
        
        printf("\n\tStage Complete--> CPU Time: %f ms, GPU Time: %f ms\n", cpu_stageTotal, gpu_stageTotal);
        /*--------------------------------------------------------------------------------------------*/

        printf("\tGPU detection--> Scaling Factor: %f, Number of faces: %d\n", factor, faces.size());
       
        printf("\n\t--------------------------------------------------------------------------------------------\n");
         ////////////////////////////////////////////////
        //Freee resources for each downsample iteration//
        ///////////////////////////////////////////////

        cudaFree(dbit_vector);
        
        freeBitVector(hbit_vector);
        //free(hbit_vector);
        
        nniiFree(devicesum1.data, devicesqsum1.data,
                 transpose_dsum1.data, transpose_dsqsum1.data);

      
        //Add the overall time of each stage
        gpu_msecTotal += gpu_stageTotal;
        cpu_msecTotal += cpu_stageTotal;
    } /* end of the factor loop, finish all scales in pyramid*/


    if( minNeighbors != 0)
    {
        //groupRectangles(allCandidates, minNeighbors, GROUP_EPS);
        groupRectangles(faces, minNeighbors, GROUP_EPS);
    }

    printf("\n-- GPU Face Detection Done--> Number of faces: %d\n", faces.size());
    printf("\n-- Overall Timing: Total CPU Time: %f ms, Total GPU Time: %f ms\n", cpu_msecTotal, gpu_msecTotal);

    freeImagePinned(img1);
    freeSumImage(sum1);
    freeSumImage(sqsum1);
   
     //////////////////////////
    //Free all GPU resources //
    //////////////////////////
   
    haarFreeOnDevice(dindex_x, dindex_y, 
                      dwidth, dheight,
                      dweights_array, dalpha1_array, 
                      dalpha2_array, dtree_thresh_array, 
                      dstages_thresh_array, dstages_array
                     );

    nniiFreeImg(deviceimg.data);

    //Destroy all CUDA Events
    cudaEventDestroy(cpu_start);
    cudaEventDestroy(cpu_stop);
    cudaEventDestroy(gpu_cpy_start);
    cudaEventDestroy(gpu_cpy_stop);

    //return allCandidates;
    return faces;

}


/***********************************************
 * Note:
 * The int_sqrt is softwar integer squre root.
 * GPU has hardware for floating squre root (sqrtf).
 * In GPU, it is wise to convert an int variable
 * into floating point, and use HW sqrtf function.
 * More info:
 * http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#standard-functions
 **********************************************/
/*****************************************************
 * The int_sqrt is only used in runCascadeClassifier
 * If you want to replace int_sqrt with HW sqrtf in GPU,
 * simple look into the runCascadeClassifier function.
 *****************************************************/
unsigned int int_sqrt (unsigned int value)
{
    int i;
    unsigned int a = 0, b = 0, c = 0;
    for (i=0; i < (32 >> 1); i++)
    {
        c<<= 2;
#define UPPERBITS(value) (value>>30)
        c += UPPERBITS(value);
#undef UPPERBITS
        value <<= 2;
        a <<= 1;
        b = (a<<1) | 1;
        if (c >= b)
        {
            c -= b;
            a++;
        }
    }
    return a;
}


void setImageForCascadeClassifier( myCascade* _cascade, MyIntImage* _sum, MyIntImage* _sqsum)
{
    MyIntImage *sum = _sum;
    MyIntImage *sqsum = _sqsum;
    myCascade* cascade = _cascade;
    int i, j, k;
    MyRect equRect;
    int r_index = 0;
    int w_index = 0;
    MyRect tr;

    cascade->sum = *sum;
    cascade->sqsum = *sqsum;

    equRect.x = equRect.y = 0;
    equRect.width = cascade->orig_window_size.width;
    equRect.height = cascade->orig_window_size.height;

    cascade->inv_window_area = equRect.width*equRect.height;

    cascade->p0 = (sum->data) ;
    cascade->p1 = (sum->data +  equRect.width - 1) ;
    cascade->p2 = (sum->data + sum->width*(equRect.height - 1));
    cascade->p3 = (sum->data + sum->width*(equRect.height - 1) + equRect.width - 1);
    cascade->pq0 = (sqsum->data);
    cascade->pq1 = (sqsum->data +  equRect.width - 1) ;
    cascade->pq2 = (sqsum->data + sqsum->width*(equRect.height - 1));
    cascade->pq3 = (sqsum->data + sqsum->width*(equRect.height - 1) + equRect.width - 1);

    /****************************************
     * Load the index of the four corners 
     * of the filter rectangle
     **************************************/

    /* loop over the number of stages */
    for( i = 0; i < cascade->n_stages; i++ )
    {
        /* loop over the number of haar features */
        for( j = 0; j < stages_array[i]; j++ )
        {
            int nr = 3;
            /* loop over the number of rectangles */
            for( k = 0; k < nr; k++ )
            {
                //Haar Feature indices  
                tr.x = rectangles_array[r_index + k*4];
                tr.y = rectangles_array[r_index + 1 + k*4];
                
                tr.width = rectangles_array[r_index + 2 + k*4];
                tr.height = rectangles_array[r_index + 3 + k*4];

                if (k < 2)
                {
                    scaled_rectangles_array[r_index + k*4] = (sum->data + sum->width*(tr.y ) + (tr.x )) ;
                    scaled_rectangles_array[r_index + k*4 + 1] = (sum->data + sum->width*(tr.y ) + (tr.x  + tr.width)) ;
                    scaled_rectangles_array[r_index + k*4 + 2] = (sum->data + sum->width*(tr.y  + tr.height) + (tr.x ));
                    scaled_rectangles_array[r_index + k*4 + 3] = (sum->data + sum->width*(tr.y  + tr.height) + (tr.x  + tr.width));
                }
                else   //for 3rd rect
                {
                    if ((tr.x == 0)&& (tr.y == 0) &&(tr.width == 0) &&(tr.height == 0))
                    {
                        scaled_rectangles_array[r_index + k*4] = NULL ;
                        scaled_rectangles_array[r_index + k*4 + 1] = NULL ;
                        scaled_rectangles_array[r_index + k*4 + 2] = NULL;
                        scaled_rectangles_array[r_index + k*4 + 3] = NULL;
                    }
                    else
                    {
                        scaled_rectangles_array[r_index + k*4] = (sum->data + sum->width*(tr.y ) + (tr.x )) ;
                        scaled_rectangles_array[r_index + k*4 + 1] = (sum->data + sum->width*(tr.y ) + (tr.x  + tr.width)) ;
                        scaled_rectangles_array[r_index + k*4 + 2] = (sum->data + sum->width*(tr.y  + tr.height) + (tr.x ));
                        scaled_rectangles_array[r_index + k*4 + 3] = (sum->data + sum->width*(tr.y  + tr.height) + (tr.x  + tr.width));
                    }
                } /* end of branch if(k<2) */
            } /* end of k loop*/

            r_index+=12;
            w_index+=3;

        } /* end of j loop */
    } /* end i loop */
}


/****************************************************
 * evalWeakClassifier:
 * the actual computation of a haar filter.
 * More info:
 * http://en.wikipedia.org/wiki/Haar-like_features
 ***************************************************/
inline int evalWeakClassifier(int variance_norm_factor, int p_offset, int tree_index, int w_index, int r_index )
{

    /* the node threshold is multiplied by the standard deviation of the image */
    int t = tree_thresh_array[tree_index] * variance_norm_factor;               //Filter threshold

    int sum = (*(scaled_rectangles_array[r_index] + p_offset)
            - *(scaled_rectangles_array[r_index + 1] + p_offset)
            - *(scaled_rectangles_array[r_index + 2] + p_offset)
            + *(scaled_rectangles_array[r_index + 3] + p_offset))
        * weights_array[w_index];

    /* 
       if(p_offset == 648) {
       printf("CPU: %d - %d - %d + %d = %d\nweight0 = %d, sum = %d\n", *(scaled_rectangles_array[r_index] + p_offset),
     *(scaled_rectangles_array[r_index+1] + p_offset),
     *(scaled_rectangles_array[r_index+2] + p_offset),
     *(scaled_rectangles_array[r_index+3] + p_offset), sum, weights_array[w_index], sum);
     }*/

    sum += (*(scaled_rectangles_array[r_index+4] + p_offset)
            - *(scaled_rectangles_array[r_index + 5] + p_offset)
            - *(scaled_rectangles_array[r_index + 6] + p_offset)
            + *(scaled_rectangles_array[r_index + 7] + p_offset))
        * weights_array[w_index + 1];

    /*
       if(p_offset == 648) {
       printf("CPU: %d - %d - %d + %d = %d\nweight0 = %d, sum = %d\n", *(scaled_rectangles_array[r_index+4] + p_offset),
     *(scaled_rectangles_array[r_index+5] + p_offset),
     *(scaled_rectangles_array[r_index+6] + p_offset),
     *(scaled_rectangles_array[r_index+7] + p_offset), sum, weights_array[w_index+1], sum);
     }*/

    if ((scaled_rectangles_array[r_index+8] != NULL)){
        sum += (*(scaled_rectangles_array[r_index+8] + p_offset)
                - *(scaled_rectangles_array[r_index + 9] + p_offset)
                - *(scaled_rectangles_array[r_index + 10] + p_offset)
                + *(scaled_rectangles_array[r_index + 11] + p_offset))
            * weights_array[w_index + 2];
    }
    if(sum >= t)
        return alpha2_array[tree_index];
    else
        return alpha1_array[tree_index];
}



int runCascadeClassifier( myCascade* _cascade, MyPoint pt, int start_stage )
{

    int p_offset, pq_offset;
    int i, j;
    unsigned int mean;
    unsigned int variance_norm_factor;
    int haar_counter = 0;
    int w_index = 0;
    int r_index = 0;
    int stage_sum;
    myCascade* cascade;
    cascade = _cascade;

    p_offset = pt.y * (cascade->sum.width) + pt.x;    //shifted widnow
    pq_offset = pt.y * (cascade->sqsum.width) + pt.x;

    /**************************************************************************
     * Image normalization
     * mean is the mean of the pixels in the detection window
     * cascade->pqi[pq_offset] are the squared pixel values (using the squared integral image)
     * inv_window_area is 1 over the total number of pixels in the detection window
     *************************************************************************/

    variance_norm_factor =  (cascade->pq0[pq_offset] - cascade->pq1[pq_offset] - cascade->pq2[pq_offset] + cascade->pq3[pq_offset]);
    mean = (cascade->p0[p_offset] - cascade->p1[p_offset] - cascade->p2[p_offset] + cascade->p3[p_offset]);

    //printf("CPU Row %d: Col %d: var: %d - %d - %d + %d = %d\nCol %d: mean: %d - %d - %d + %d = %d\n", pt.y, pt.x, cascade->pq0[pq_offset], cascade->pq1[pq_offset], cascade->pq2[pq_offset], cascade->pq3[pq_offset], variance_norm_factor, pt.x, cascade->p0[p_offset], cascade->p1[p_offset], cascade->p2[p_offset], cascade->p3[p_offset], mean);

    variance_norm_factor = (variance_norm_factor * cascade->inv_window_area);
    variance_norm_factor =  variance_norm_factor - mean*mean;

    /***********************************************
     * Note:
     * The int_sqrt is softwar integer squre root.
     * GPU has hardware for floating squre root (sqrtf).
     * In GPU, it is wise to convert the variance norm
     * into floating point, and use HW sqrtf function.
     * More info:
     * http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#standard-functions
     **********************************************/
    if( variance_norm_factor > 0 )
        variance_norm_factor = int_sqrt(variance_norm_factor);
    else
        variance_norm_factor = 1;

    //printf("CPU: Row: %d, Col: %d, ID: %d: Variance = %d\n", pt.y, pt.x, p_offset, variance_norm_factor);
    /**************************************************
     * The major computation happens here.
     * For each scale in the image pyramid,
     * and for each shifted step of the filter,
     * send the shifted window through cascade filter.
     *
     * Note:
     *
     * Stages in the cascade filter are independent.
     * However, a face can be rejected by any stage.
     * Running stages in parallel delays the rejection,
     * which induces unnecessary computation.
     *
     * Filters in the same stage are also independent,
     * except that filter results need to be merged,
     * and compared with a per-stage threshold.
     *************************************************/
    for( i = start_stage; i < 25; i++) //cascade->n_stages; i++ ) Change here- Sharmila
    {

        /****************************************************
         * A shared variable that induces false dependency
         * 
         * To avoid it from limiting parallelism,
         * we can duplicate it multiple times,
         * e.g., using stage_sum_array[number_of_threads].
         * Then threads only need to sync at the end
         ***************************************************/
        stage_sum = 0;

        for( j = 0; j < stages_array[i]; j++ )
        {
            /**************************************************
             * Send the shifted window to a haar filter.
             **************************************************/
            stage_sum += evalWeakClassifier(variance_norm_factor, p_offset, haar_counter, w_index, r_index);
            n_features++;
            haar_counter++;
            w_index+=3;
            r_index+=12;
        } /* end of j loop */

        /**************************************************************
         * threshold of the stage. 
         * If the sum is below the threshold, 
         * no faces are detected, 
         * and the search is abandoned at the i-th stage (-i).
         * Otherwise, a face is detected (1)
         **************************************************************/

        /* the number "0.4" is empirically chosen for 5kk73 */
        if( stage_sum <  0.4 * stages_thresh_array[i] ){
            return -i;
        } /* end of the per-stage thresholding */
    } /* end of i loop */

    //printf("True: Vec ID = %d, Stage = %d, CPU: Row = %d, Col = %d: stage_sum = %ld < %d\n", pt.y*(cascade->sum.width-cascade->orig_window_size.width)+pt.x, i, pt.y, pt.x, stage_sum, (int)(0.4*stages_thresh_array[i]));
    return 1;
}


void ScaleImage_Invoker( myCascade* _cascade, float _factor, int sum_row, int sum_col, std::vector<MyRect>& _vec)
{

    myCascade* cascade = _cascade;

    float factor = _factor;
    MyPoint p;
    int result;
    int y1, y2, x2, x, y, step;
    std::vector<MyRect> *vec = &_vec;

    MySize winSize0 = cascade->orig_window_size;
    MySize winSize;

    winSize.width =  myRound(winSize0.width*factor);
    winSize.height =  myRound(winSize0.height*factor);
    y1 = 0;

    /********************************************
     * When filter window shifts to image boarder,
     * some margin need to be kept
     *********************************************/
    y2 = sum_row - winSize0.height;
    x2 = sum_col - winSize0.width;

    /********************************************
     * Step size of filter window shifting
     * Reducing step makes program faster,
     * but decreases quality of detection.
     * example:
     * step = factor > 2 ? 1 : 2;
     * 
     * For 5kk73, 
     * the factor and step can be kept constant,
     * unless you want to change input image.
     *
     * The step size is set to 1 for 5kk73,
     * i.e., shift the filter window by 1 pixel.
     *******************************************/	
    step = 1;

    /**********************************************
     * Shift the filter window over the image.
     * Each shift step is independent.
     * Shared data structure may limit parallelism.
     *
     * Some random hints (may or may not work):
     * Split or duplicate data structure.
     * Merge functions/loops to increase locality
     * Tiling to increase computation-to-memory ratio
     *********************************************/
    int i;
    for(i = 0; i < (x2 * y2); i++) {
        bit_vector[i] = true;
    }
    
    for( x = 0; x <= x2; x += step )
        for( y = y1; y <= y2; y += step )
        {
            p.x = x;
            p.y = y;

            /*********************************************
             * Optimization Oppotunity:
             * The same cascade filter is used each time
             ********************************************/
            result = runCascadeClassifier( cascade, p, 0 );

            /*******************************************************
             * If a face is detected,
             * record the coordinates of the filter window
             * the "push_back" function is from std:vec, more info:
             * http://en.wikipedia.org/wiki/Sequence_container_(C++)
             *
             * Note that, if the filter runs on GPUs,
             * the push_back operation is not possible on GPUs.
             * The GPU may need to use a simpler data structure,
             * e.g., an array, to store the coordinates of face,
             * which can be later memcpy from GPU to CPU to do push_back
             *******************************************************/
            int index = y*x2+x;
            if( result > 0 )
            {
                //printf("Result is greater than zero\n");
                MyRect r = {myRound(x*factor), myRound(y*factor), winSize.width, winSize.height};
                vec->push_back(r);
            }
            else
                bit_vector[index] = false;
        }
}

/*****************************************************
 * Compute the integral image (and squared integral)
 * Integral image helps quickly sum up an area.
 * More info:
 * http://en.wikipedia.org/wiki/Summed_area_table
 ****************************************************/
void integralImageOnHost( MyImage *src, MyIntImage *sum, MyIntImage *sqsum )
{
    int x, y, s, sq, t, tq;
    unsigned char it;
    int height = src->height;
    int width = src->width;
    unsigned char *data = src->data;
    int * sumData = sum->data;
    int * sqsumData = sqsum->data;

    for( y = 0; y < height; y++)
    {
        s = 0;
        sq = 0;
        /* loop over the number of columns */
        for( x = 0; x < width; x ++)
        {
            it = data[y*width+x];
            /* sum of the current row (integer)*/
            s += it; 
            sq += it*it;

            t = s;
            tq = sq;
            if (y != 0)
            {
                t += sumData[(y-1)*width+x];
                tq += sqsumData[(y-1)*width+x];
            }
            sumData[y*width+x]=t;
            sqsumData[y*width+x]=tq;
        }
    }
}

/***********************************************************
 * This function downsample an image using nearest neighbor
 * It is used to build the image pyramid
 **********************************************************/
void nearestNeighborOnHost(MyImage *src, MyImage *dst)
{

    int y;
    int j;
    int x;
    int i;
    unsigned char* t;
    unsigned char* p;
    int w1 = src->width;
    int h1 = src->height;
    int w2 = dst->width;
    int h2 = dst->height;

    int rat = 0;

    unsigned char* src_data = src->data;
    unsigned char* dst_data = dst->data;


    int x_ratio = (int)((w1<<16)/w2) +1;
    int y_ratio = (int)((h1<<16)/h2) +1;

    for (i=0;i<h2;i++)
    {
        t = dst_data + i*w2;       //Pointer to next row in dst image
        y = ((i*y_ratio)>>16);
        p = src_data + y*w1;
        rat = 0;

        for (j=0;j<w2;j++)
        {
            x = (rat>>16);
            *t++ = p[x];
            rat += x_ratio;
        }
    }
}

void readTextClassifierForGPU()//(myCascade * cascade)
{
    /*number of stages of the cascade classifier*/
    int stages;
    /*total number of weak classifiers (one node each)*/
    int total_nodes = 0;
    int i, j, k;
    char mystring [12];
    int w_index = 0;
    int tree_index = 0;
    FILE *finfo = fopen("info.txt", "r");

    /**************************************************
     * how many stages are in the cascaded filter? 
     * the first line of info.txt is the number of stages 
     * (in the 5kk73 example, there are 25 stages)
     **************************************************/
    if ( fgets (mystring , 12 , finfo) != NULL )
    {
        stages = atoi(mystring);
    }
    i = 0;

    hstages_array = (int *)malloc(sizeof(int)*stages);

    /**************************************************
     * how many filters in each stage? 
     * They are specified in info.txt,
     * starting from second line.
     * (in the 5kk73 example, from line 2 to line 26)
     *************************************************/
    while ( fgets (mystring , 12 , finfo) != NULL )
    {
        hstages_array[i] = atoi(mystring);
        total_nodes += hstages_array[i];
        i++;
    }
    fclose(finfo);

    /* TODO: use matrices where appropriate */
    /***********************************************
     * Allocate a lot of array structures
     * Note that, to increase parallelism,
     * some arrays need to be splitted or duplicated
     **********************************************/
    //rectangles_array = (int *)malloc(sizeof(int)*total_nodes*12);
    //scaled_rectangles_array = (int **)malloc(sizeof(int*)*total_nodes*12);

    hindex_x = (uint16_t *)malloc(sizeof(uint16_t)*total_nodes*3);
    hindex_y = (uint16_t *)malloc(sizeof(uint16_t)*total_nodes*3);
    hwidth = (uint16_t *)malloc(sizeof(uint16_t)*total_nodes*3);
    hheight = (uint16_t *)malloc(sizeof(uint16_t)*total_nodes*3);
    hweights_array = (int16_t *)malloc(sizeof(int16_t)*total_nodes*3);
    halpha1_array = (int16_t*)malloc(sizeof(int16_t)*total_nodes);
    halpha2_array = (int16_t*)malloc(sizeof(int16_t)*total_nodes);
    htree_thresh_array = (int16_t*)malloc(sizeof(int16_t)*total_nodes);
    hstages_thresh_array = (int16_t*)malloc(sizeof(int16_t)*stages);
    FILE *fp = fopen("class.txt", "r");

    /******************************************
     * Read the filter parameters in class.txt
     *
     * Each stage of the cascaded filter has:
     * 18 parameter per filter x tilter per stage
     * + 1 threshold per stage
     *
     * For example, in 5kk73, 
     * the first stage has 9 filters,
     * the first stage is specified using
     * 18 * 9 + 1 = 163 parameters
     * They are line 1 to 163 of class.txt
     *
     * The 18 parameters for each filter are:
     * 1 to 4: coordinates of rectangle 1
     * 5: weight of rectangle 1
     * 6 to 9: coordinates of rectangle 2
     * 10: weight of rectangle 2
     * 11 to 14: coordinates of rectangle 3
     * 15: weight of rectangle 3
     * 16: threshold of the filter
     * 17: alpha 1 of the filter
     * 18: alpha 2 of the filter
     ******************************************/

    /* loop over n of stages */
    for (i = 0; i < stages; i++)
    {    /* loop over n of trees */
        for (j = 0; j < hstages_array[i]; j++)
        {	/* loop over n of rectangular features */
            for(k = 0; k < 3; k++)
            {	/* loop over the n of vertices */
                //for (l = 0; l <4; l++)
                //{
                if (fgets (mystring , 12 , fp) != NULL)
                    hindex_x[w_index] = atoi(mystring);
                else
                    break;
                if (fgets (mystring , 12 , fp) != NULL)
                    hindex_y[w_index] = atoi(mystring);
                else
                    break;
                if (fgets (mystring , 12 , fp) != NULL)
                    hwidth[w_index] = atoi(mystring);
                else
                    break;
                if (fgets (mystring , 12 , fp) != NULL)
                    hheight[w_index] = atoi(mystring);
                else
                    break;
                //r_index++;
                //} /* end of l loop */

                if (fgets (mystring , 12 , fp) != NULL)
                {
                    hweights_array[w_index] = atoi(mystring);
                    /* Shift value to avoid overflow in the haar evaluation */
                    /*TODO: make more general */
                    /*weights_array[w_index]>>=8; */
                }
                else
                    break;
                w_index++;
            } /* end of k loop */

            if (fgets (mystring , 12 , fp) != NULL)
                htree_thresh_array[tree_index]= atoi(mystring);
            else
                break;
            if (fgets (mystring , 12 , fp) != NULL)
                halpha1_array[tree_index]= atoi(mystring);
            else
                break;
            if (fgets (mystring , 12 , fp) != NULL)
                halpha2_array[tree_index]= atoi(mystring);
            else
                break;
            tree_index++;

            if (j == hstages_array[i]-1)
            {
                if (fgets (mystring , 12 , fp) != NULL)
                    hstages_thresh_array[i] = atoi(mystring);
                else
                    break;
            }
        } /* end of j loop */
    } /* end of i loop */
    fclose(fp);
}

void readTextClassifier()//(myCascade * cascade)
{
    /*number of stages of the cascade classifier*/
    int stages;
    /*total number of weak classifiers (one node each)*/
    int total_nodes = 0;
    int i, j, k, l;
    char mystring [12];
    int r_index = 0;
    int w_index = 0;
    int tree_index = 0;
    FILE *finfo = fopen("info.txt", "r");

    /**************************************************
    /* how many stages are in the cascaded filter? 
    /* the first line of info.txt is the number of stages 
    /* (in the 5kk73 example, there are 25 stages)
     **************************************************/
    if ( fgets (mystring , 12 , finfo) != NULL )
    {
        stages = atoi(mystring);
    }
    i = 0;

    stages_array = (int *)malloc(sizeof(int)*stages);

    /**************************************************
     * how many filters in each stage? 
     * They are specified in info.txt,
     * starting from second line.
     * (in the 5kk73 example, from line 2 to line 26)
     *************************************************/
    while ( fgets (mystring , 12 , finfo) != NULL )
    {
        stages_array[i] = atoi(mystring);
        total_nodes += stages_array[i];
        i++;
    }
    fclose(finfo);


    /* TODO: use matrices where appropriate */
    /***********************************************
     * Allocate a lot of array structures
     * Note that, to increase parallelism,
     * some arrays need to be splitted or duplicated
     **********************************************/
    rectangles_array = (int *)malloc(sizeof(int)*total_nodes*12);
    scaled_rectangles_array = (int **)malloc(sizeof(int*)*total_nodes*12);
    weights_array = (int *)malloc(sizeof(int)*total_nodes*3);
    alpha1_array = (int*)malloc(sizeof(int)*total_nodes);
    alpha2_array = (int*)malloc(sizeof(int)*total_nodes);
    tree_thresh_array = (int*)malloc(sizeof(int)*total_nodes);
    stages_thresh_array = (int*)malloc(sizeof(int)*stages);
    FILE *fp = fopen("class.txt", "r");

    /******************************************
     * Read the filter parameters in class.txt
     *
     * Each stage of the cascaded filter has:
     * 18 parameter per filter x tilter per stage
     * + 1 threshold per stage
     *
     * For example, in 5kk73, 
     * the first stage has 9 filters,
     * the first stage is specified using
     * 18 * 9 + 1 = 163 parameters
     * They are line 1 to 163 of class.txt
     *
     * The 18 parameters for each filter are:
     * 1 to 4: coordinates of rectangle 1
     * 5: weight of rectangle 1
     * 6 to 9: coordinates of rectangle 2
     * 10: weight of rectangle 2
     * 11 to 14: coordinates of rectangle 3
     * 15: weight of rectangle 3
     * 16: threshold of the filter
     * 17: alpha 1 of the filter
     * 18: alpha 2 of the filter
     ******************************************/

    /* loop over n of stages */
    for (i = 0; i < stages; i++)
    {    /* loop over n of trees */
        for (j = 0; j < stages_array[i]; j++)
        {	/* loop over n of rectangular features */
            for(k = 0; k < 3; k++)
            {	/* loop over the n of vertices */
                for (l = 0; l <4; l++)
                {
                    if (fgets (mystring , 12 , fp) != NULL)
                        rectangles_array[r_index] = atoi(mystring);
                    else
                        break;
                    r_index++;
                } /* end of l loop */

                if (fgets (mystring , 12 , fp) != NULL)
                {
                    weights_array[w_index] = atoi(mystring);
                    /* Shift value to avoid overflow in the haar evaluation */
                    /*TODO: make more general */
                    /*weights_array[w_index]>>=8; */
                }
                else
                    break;
                w_index++;
            } /* end of k loop */

            if (fgets (mystring , 12 , fp) != NULL)
                tree_thresh_array[tree_index]= atoi(mystring);
            else
                break;
            if (fgets (mystring , 12 , fp) != NULL)
                alpha1_array[tree_index]= atoi(mystring);
            else
                break;
            if (fgets (mystring , 12 , fp) != NULL)
                alpha2_array[tree_index]= atoi(mystring);
            else
                break;
            tree_index++;

            if (j == stages_array[i]-1)
            {
                if (fgets (mystring , 12 , fp) != NULL)
                    stages_thresh_array[i] = atoi(mystring);
                else
                    break;
            }
        } /* end of j loop */
    } /* end of i loop */
    fclose(fp);
}


void releaseTextClassifier()
{
    free(stages_array);
    free(rectangles_array);
    free(scaled_rectangles_array);
    free(weights_array);
    free(tree_thresh_array);
    free(alpha1_array);
    free(alpha2_array);
    free(stages_thresh_array);
}

void releaseTextClassifierGPU()
{
    free(hstages_array);
    free(hindex_x);
    free(hindex_y);
    free(hweights_array);
    free(htree_thresh_array);
    free(halpha1_array);
    free(halpha2_array);
    free(hstages_thresh_array);
    free(bit_vector);
}

/* End of file. */
