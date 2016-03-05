#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>

#include <cutil.h>
#include "util.h"
#include "ref_2dhisto.h"
#include "opt_2dhisto.h"
#include "test.h"
#include <cuda.h>
#include "helper_cuda.h"
#include "helper_string.h"

void opt_2dhisto(uint32_t * result, uint32_t ** input, int height,
                 int width)
{
  dim3 dim_block(1024, 1, 1);
  dim3 dim_grid(1, 16, 1);

  //std::cout << "calling the CUDA kernel...\n";
  cudaMemset(result, 0, HISTO_WIDTH * HISTO_HEIGHT * sizeof(uint32_t));

  histo_kernel<<<dim_grid, dim_block>>>(result, input, height, width);

  cudaDeviceSynchronize();

}

/* Include below the implementation of any other functions you need */

__global__ void histo_kernel(uint32_t * result, uint32_t ** input,
                             int height, int width)
{
  //__shared__ uint8_t hist[HISTO_HEIGHT * HISTO_WIDTH];
  __shared__ uint32_t hist[HISTO_HEIGHT * HISTO_WIDTH][NUM_HISTO];
  //__shared__ uint32_t hist[NUM_HISTO][HISTO_HEIGHT * HISTO_WIDTH];

 
  // Clear shared memory buffer 
  for (size_t j = threadIdx.y; j < HISTO_HEIGHT; j += blockDim.y)
    for (size_t i = threadIdx.x; i < HISTO_WIDTH; i += blockDim.x) {
      size_t tid = j * HISTO_WIDTH + i;
      for (size_t k = 0; k < NUM_HISTO; ++k)
        hist[tid][k] = 0;
      //if (blockIdx.x == 0 && blockIdx.y == 0)
      //  result[tid] = 0;  // coalased memroy write
    }

  __syncthreads();
  
  for (size_t j = blockIdx.y * blockDim.y + threadIdx.y; j < height;
       j += blockDim.y * gridDim.y)
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < width;
         i += blockDim.x * gridDim.x) {
      const uint32_t value = input[j][i];
      size_t hid = (j * HISTO_WIDTH + i) % NUM_HISTO;
      //if (hist[hid][value] < UINT8_MAX)
        atomicAdd(hist[value] + hid, 1);
    }

  __syncthreads();

  for (size_t i = threadIdx.x; i < HISTO_WIDTH; i += blockDim.x)
    for (size_t j = threadIdx.y; j < HISTO_HEIGHT; j += blockDim.y)
      for (size_t k = 0; k < NUM_HISTO; ++k)
        atomicAdd(result + j * HISTO_WIDTH + i, hist[j * HISTO_WIDTH + i][k]);
}

/*
__device__ void atomic_inc_uint8 (uint8_t * addr)
{
  unsigned int * base_addr = (unsigned int *)(addr - ((size_t)addr & 0x3));
  unsigned int value = 1;
  switch ((size_t)addr & 0x3) {
    case 3:
      value = value << 24;
      break;
    case 2:
      value = value << 16;
      break;
    case 1:
      value = value << 8;
      break;
  }
  atomicAdd(base_addr, value);
}
*/

uint32_t ** setup_data(uint32_t ** input, int width, int height)
{
  const size_t width_padded = (width + 128) & 0xFFFFFF80;
  size_t size = width_padded * height * sizeof(uint32_t);

  uint32_t * p;
  checkCudaErrors(cudaMalloc((void**)&p, size));
  //std::cout << "size  : " << size << "\n";
  //std::cout << "addr p: " << p << "\n";
  checkCudaErrors(cudaMemcpy(p, &input[0][0], size, cudaMemcpyHostToDevice));
  
  uint32_t * data[height];
  for(int  i = 0; i < height; ++i) {
    data[i] = p + (i * width_padded);
    //std::cout << "data[" << i << "]: " << data[i] << "\n";
  }

  uint32_t * p2;
  checkCudaErrors(cudaMalloc((void**)&p2, sizeof(data)));
  //std::cout << "size   : " << sizeof(data) << "\n";
  //std::cout << "addr p2: " << p2 << "\n";
  checkCudaErrors(cudaMemcpy(p2, data, sizeof(data), cudaMemcpyHostToDevice));

  return (uint32_t **) p2;
}

uint32_t * setup_histo()
{
  uint32_t * p;
  cudaMalloc((void**)&p, HISTO_WIDTH * HISTO_HEIGHT * sizeof(uint32_t));
  cudaMemset(p, 0, HISTO_WIDTH * HISTO_HEIGHT * sizeof(uint32_t));
  //std::cout << "size      : " << HISTO_WIDTH * HISTO_HEIGHT * sizeof(uint32_t) << "\n";
  //std::cout << "histo addr: " << (uint *)p << "\n";

  return p;
}

void copy_final(uint32_t **data, uint32_t *histo_bin, uint32_t *histo)
{
  int size = HISTO_WIDTH * HISTO_HEIGHT * sizeof(uint32_t);
  checkCudaErrors(cudaMemcpy(histo_bin, histo, size, cudaMemcpyDeviceToHost));


  cudaFree(histo);
  cudaFree(data);
}
