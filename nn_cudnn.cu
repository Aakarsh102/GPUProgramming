#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <random>


#define checkCUDNN(expression)                              \
  {                                                         \
    cudnnStatus_t status = (expression);                    \
    if (status != CUDNN_STATUS_SUCCESS) {                   \
      std::cerr << "Error on line " << __LINE__ << ": "     \
                << cudnnGetErrorString(status) << std::endl;\
      std::exit(EXIT_FAILURE);                              \
    }                                                       \
  }


void matmul(cublaHandle_t handle, float *a, float *b) {
	
}


__global__ void Linear() {

}
