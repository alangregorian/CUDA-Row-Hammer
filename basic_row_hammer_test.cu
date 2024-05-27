#include <cuda_runtime.h>
#include <iostream>

#define N 1024 * 1024  // Size of the memory allocation
#define ITERATIONS 1000000  // Number of hammering iterations
#define PATTERN 0xDEADBEEF  // Known pattern to initialize memory

__global__ void initMemoryKernel(int *data, int value) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        data[idx] = value;
    }
}

__global__ void hammerKernel(int *data) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i = 0; i < ITERATIONS; ++i) {
        data[idx] ^= 1;  // Example operation to induce hammering
    }
}

bool checkMemory(int *data, int value) {
    for (int i = 0; i < N; ++i) {
        if (data[i] != value) {
            std::cerr << "Bit flip detected at index " << i << ": "
                      << std::hex << data[i] << std::dec << std::endl;
            return true;
        }
    }
    return false;
}

int main() {
    // Allocate memory on the GPU
    int *d_data;
    cudaError_t err = cudaMalloc(&d_data, N * sizeof(int));
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate memory on GPU: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    // Initialize memory with a known pattern
    initMemoryKernel<<<(N + 255) / 256, 256>>>(d_data, PATTERN);
    cudaDeviceSynchronize();

    // Perform hammering
    hammerKernel<<<(N + 255) / 256, 256>>>(d_data);
    cudaDeviceSynchronize();

    // Allocate host memory to copy data back
    int *h_data = (int*)malloc(N * sizeof(int));
    if (h_data == nullptr) {
        std::cerr << "Failed to allocate host memory" << std::endl;
        cudaFree(d_data);
        return -1;
    }

    // Copy data back to host
    cudaMemcpy(h_data, d_data, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Check for bit flips
    bool success = checkMemory(h_data, PATTERN);

    if (success) {
        std::cout << "Bit flips detected!" << std::endl;
    } else {
        std::cout << "No bit flips detected." << std::endl;
    }

    // Free allocated memory
    free(h_data);
    cudaFree(d_data);

    return 0;
}
