#include <stdio.h>
#include <cuda_runtime.h>

__global__ void accessMemory(unsigned int *count, char *data) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    char value = data[idx % 1024]; // Accessing the same 1024 bytes repeatedly

    unsigned long long start = clock64();
    while (clock64() - start < 64000) {
        value = data[idx % 1024];
        atomicAdd(count, 1);
    }
}

int main() {
    char *d_data;
    unsigned int *d_count;
    unsigned int h_count = 0;

    cudaMalloc(&d_data, 1024); // Allocate 1KB on device to ensure the same row access
    cudaMalloc(&d_count, sizeof(unsigned int));
    cudaMemset(d_count, 0, sizeof(unsigned int));

    int blocks = 256;
    int threadsPerBlock = 256;

    // Launch the kernel
    accessMemory<<<blocks, threadsPerBlock>>>(d_count, d_data);
    cudaDeviceSynchronize();

    // Copy back the result
    cudaMemcpy(&h_count, d_count, sizeof(unsigned int), cudaMemcpyDeviceToHost);

    printf("Total accesses in 64 milliseconds: %u\n", h_count);

    // Cleanup
    cudaFree(d_data);
    cudaFree(d_count);
    return 0;
}
