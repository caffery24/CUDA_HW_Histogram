__global__ void histo_kernel(uint32_t * result, uint32_t * input[], int height, int width);

__device__ void atomic_inc_uint8(uint8_t * addr);
