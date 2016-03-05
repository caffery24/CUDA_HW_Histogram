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

void opt_2dhisto(uint32_t * result, uint32_t * input, int height,
                 int width)
{
  dim3 dim_block(1024, 1, 1);
  dim3 dim_grid(16, 1, 1);

  //std::cout << "calling the CUDA kernel...\n";
  cudaMemset(result, 0, HISTO_WIDTH * HISTO_HEIGHT * sizeof(uint32_t));

  histo_kernel<<<dim_grid, dim_block>>>(result, input, height, width);

  cudaDeviceSynchronize();
}

uint8_t * setup_histo()
{
  uint8_t * p;
  cudaMalloc((void**)&p, HISTO_WIDTH * HISTO_HEIGHT * sizeof(uint8_t));
  std::cout << "size: " << HISTO_WIDTH * HISTO_HEIGHT * sizeof(uint8_t) << "\n";
  cudaMemset(p, 0, HISTO_WIDTH * HISTO_HEIGHT * sizeof(uint8_t));
  return p;
}

void copy_final(uint32_t *data, uint32_t *histo_bin, uint32_t *histo)
{
  printf("copy_final\n");
  int size = HISTO_WIDTH * HISTO_HEIGHT * sizeof(uint32_t);
  checkCudaErrors(cudaMemcpy(histo_bin, histo, size, cudaMemcpyDeviceToHost));

  cudaFree(histo);
  cudaFree(data);
}

uint32_t * newdata_collect(uint32_t ** input, int height,int width){
  uint32_t * newdata = (uint32_t* ) malloc(height * width * sizeof(uint32_t));
  
  for(int i= 0; i < height;  i++) {
    for(int j=0; j<width; j++){	
      newdata[i * width + j] = input[i][j];  
    }
  }
  return  newdata;
}

uint32_t * setup(uint32_t * input, int  width, int  height){
  uint32_t *p;
  cudaMalloc((void **)&p, width  * height * sizeof(uint32_t));
  checkCudaErrors(cudaMemcpy(p, input , width * height * sizeof(uint32_t), cudaMemcpyHostToDevice));
  return p;
}



__global__ void histo_kernel(uint32_t * result, uint32_t * input,
                             int height, int width)
{
  __shared__ uint32_t hist[HISTO_HEIGHT * HISTO_WIDTH][6];
 
    for (size_t j = threadIdx.x; j <HISTO_HEIGHT * HISTO_WIDTH ; j += blockDim.x) {
      hist[j][0] = 0;
      hist[j][1] = 0;
      hist[j][2] = 0;
      hist[j][3] = 0;
      hist[j][4] = 0;
      hist[j][5] = 0;
   
    
    }

    __syncthreads();

   for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < width*height;
     i += blockDim.x * gridDim.x) {
     const uint32_t value = input[i];
     int index = i % 6;
     atomicAdd(&hist[value][index], 1);
   }

    __syncthreads();

   for (size_t i = threadIdx.x; i < HISTO_HEIGHT * HISTO_WIDTH; i += blockDim.x)       
     atomicAdd(result + i, hist[i][0]+hist[i][1]+hist[i][2]+hist[i][3]+hist[i][4]+hist[i][5]);
}


uint32_t * setuphisto()
{
  uint32_t * p;
  cudaMalloc((void**)&p, HISTO_WIDTH * HISTO_HEIGHT * sizeof(uint32_t));
  std::cout << "size      : " << HISTO_WIDTH * HISTO_HEIGHT * sizeof(uint32_t) << "\n";
  cudaMemset(p,0,HISTO_WIDTH * HISTO_HEIGHT * sizeof(uint32_t));
  checkCudaErrors(cudaGetLastError());
  return p;
}
