/*
 *A simple CUDA program to list properties of the devices.
 *    Copyright (C) 2015  Tim Haines (thaines.astro@gmail.com)
 *
 *    This program is free software: you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation, either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
 
#include <iostream>
#include <iomanip>
#include <cuda.h>
 
#define CUDA_CHECK_RETURN(value) {                              \
    cudaError_t _m_cudaStat = (value);                          \
    if (_m_cudaStat != cudaSuccess) {                           \
        std::cout << "Error " << cudaGetErrorString(_m_cudaStat)\
             << " at line " << __LINE__                         \
             << " in file " __FILE__ << "\n";                   \
        exit(-1);                                               \
    } }
 
/* Make sure we are compiling for compute capability 2.0 or later.
 *
 * NOTE: The __CUDA_ARCH__ macro is only available during the device code
 *       trajectory with nvcc steering compilation (i.e., __CUDACC__ is defined).
 *
 *      This is just an example. The proceeding code does not require a specific
 *      compute capability and is available on all CUDA runtime environments.
 */
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 200
#error "Must have CUDA compute capability of at least 2.0!"
#endif
 
int main() {
    int nDevices;
    const unsigned int width = 25;
 
    /* Get the runtime version */
    int version;
    CUDA_CHECK_RETURN(cudaRuntimeGetVersion(&version));
    std::cout << std::left << std::setw(width) << "CUDA runtime version: " << version << std::endl;
 
    /* Get the driver version */
    CUDA_CHECK_RETURN(cudaDriverGetVersion(&version));
    std::cout << std::left << std::setw(width) << "CUDA driver version: " << version << std::endl << std::endl;
 
    CUDA_CHECK_RETURN(cudaGetDeviceCount(&nDevices));
 
    /* Get the device properties */
    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        CUDA_CHECK_RETURN(cudaGetDeviceProperties(&prop, i));
 
        std::cout << std::left
             << std::setw(width) << "Device Number:" << i << std::endl
             << std::setw(width) << "Device Name:" << prop.name << std::endl
             << std::setw(width) << "Compute Capability:" << prop.major << "." << prop.minor << std::endl
             << std::setw(width) << "Global Memory:" << prop.totalGlobalMem / (1 << 20) << "MB (" << prop.totalGlobalMem << ")" << std::endl
             << std::setw(width) << "Total Const Memory:" << prop.totalConstMem / (1 << 10) << "KB (" << prop.totalConstMem << ")" << std::endl
             << std::setw(width) << "Shared Memory per Block:" << prop.sharedMemPerBlock / (1 << 10) << "KB (" << prop.sharedMemPerBlock << ")" << std::endl
             << std::setw(width) << "Shared Memory per MP:" << prop.sharedMemPerMultiprocessor / (1 << 10) << "KB (" << prop.sharedMemPerMultiprocessor << ")" << std::endl
             << std::setw(width) << "L2 Cache Size:" << prop.l2CacheSize / (1 << 10) << "KB (" << prop.l2CacheSize << ")" << std::endl
             << std::setw(width) << "Global L1 Cache:" << ((prop.globalL1CacheSupported==1)?"yes":"no") << std::endl
             << std::setw(width) << "Local L1 Cache:" << ((prop.localL1CacheSupported==1)?"yes":"no") << std::endl
             << std::setw(width) << "Registers per Block" << prop.regsPerBlock << std::endl
             << std::setw(width) << "Registers per SM" << prop.regsPerMultiprocessor << std::endl
             << std::setw(width) << "SM Count: " << prop.multiProcessorCount << std::endl
             << std::setw(width) << "max threads per SM: " << prop.maxThreadsPerMultiProcessor << std::endl
             << std::setw(width) << "Max threads per block: " << prop.maxThreadsPerBlock << std::endl
             << std::setw(width) << "Max thread dims: " << "(" << prop.maxThreadsDim[0] << "," << prop.maxThreadsDim[1] << "," << prop.maxThreadsDim[2] << ")\n"
             << std::setw(width) << "Max grid size: " << "(" << prop.maxGridSize[0] << "," << prop.maxGridSize[1] << ","    << prop.maxGridSize[2] << ")" << std::endl
             << std::setw(width) << "Concurrent Kernels: " << ((prop.concurrentKernels==1) ? "yes" : "no") << std::endl
             << std::setw(width) << "Num copy engines: " << prop.asyncEngineCount << std::endl << std::endl;
    }
}
