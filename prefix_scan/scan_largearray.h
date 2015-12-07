void checkError(char* display) {
    // check for error
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        // print the CUDA error message and exit
        printf("%s: CUDA error: %s\n", display, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}


