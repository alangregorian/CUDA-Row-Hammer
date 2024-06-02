#include <stdio.h>
#include <cuda_runtime.h>

__global__ void accessMemory(unsigned int *count, volatile char *data, int dataSize, int stride) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    idx = (idx * stride) % dataSize;  // Calculate index based on stride

    unsigned long long start = clock64();
    while (clock64() - start < 64000) {
        volatile char value = data[idx];  // Ensure each access bypasses L1
        idx = (idx + stride) % dataSize;  // Move index by stride each loop iteration
        atomicAdd(count, 1);
    }
}

int main() {
    int dataSize = 1024 * 1024 * 10; // 10 MB of data to exceed L1 and L2 cache sizes
    int stride = 1024; // Stride larger than the typical cache line size

    char *d_data;
    unsigned int *d_count;
    unsigned int h_count = 0;

    cudaMalloc(&d_data, dataSize); // Allocate 10 MB on device
    cudaMemset(d_data, 0, dataSize);

    cudaMalloc(&d_count, sizeof(unsigned int));
    cudaMemset(d_count, 0, sizeof(unsigned int));

    // Cast the allocated memory to volatile char* for kernel invocation
    volatile char *v_data = reinterpret_cast<volatile char*>(d_data);

    int blocks = 256;
    int threadsPerBlock = 256;

    // Launch the kernel with additional parameters for dataSize and stride
    accessMemory<<<blocks, threadsPerBlock>>>(d_count, v_data, dataSize, stride);
    cudaDeviceSynchronize();

    // Copy back the result
    cudaMemcpy(&h_count, d_count, sizeof(unsigned int), cudaMemcpyDeviceToHost);

    printf("Total accesses in 64 milliseconds: %u\n", h_count);

    // Cleanup
    cudaFree(d_data);
    cudaFree(d_count);
    return 0;
}
