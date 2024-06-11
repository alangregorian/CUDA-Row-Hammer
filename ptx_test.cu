#include <cuda_runtime.h>
#include <iostream>
#include <cstring> // For memset

// Kernel to perform repeated writes to a specific row in memory and reset it
__global__ void rowhammerKernel(volatile float *data, int N, int hammerIdx, int hammerCount) {
    for (int i = 0; i < hammerCount; ++i) {
        // Perform repeated writes to the hammer index
        asm volatile("st.global.cg.f32 [%0], %1;" : : "l"(&data[hammerIdx]), "f"(data[hammerIdx] + 1.0f));
        // Reset the bit back to its original value (0)
        asm volatile("st.global.cg.f32 [%0], %1;" : : "l"(&data[hammerIdx]), "f"(0.0f));
    }
}

// Function to check for bit flips in memory
bool checkForBitFlips(float *data, float *originalData, int N) {
    bool bitFlipsDetected = false;
    for (int i = 0; i < N; ++i) {
        if (data[i] != originalData[i]) {
            std::cout << "Bit flip detected at index " << i << ": " << originalData[i] << " -> " << data[i] << std::endl;
            bitFlipsDetected = true;
        }
    }
    return bitFlipsDetected;
}

int main() {
    int N = 1024 * 1024; // 1 million elements
    size_t size = N * sizeof(float);
    int hammerIdx = 512 * 1024; // Middle of the array
    int hammerCount = 1000000; // Number of hammering iterations

    // Allocate and initialize host memory
    float *h_data = (float *)malloc(size);
    float *originalData = (float *)malloc(size);
    memset(h_data, 0, size);
    memcpy(originalData, h_data, size);

    // Allocate device memory
    float *d_data;
    cudaMalloc(&d_data, size);
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);

    // Launch kernel to perform Rowhammer attack
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    rowhammerKernel<<<numBlocks, blockSize>>>(d_data, N, hammerIdx, hammerCount);
    cudaDeviceSynchronize();

    // Copy data back to host for verification
    cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);

    // Check for bit flips
    bool bitFlipsDetected = checkForBitFlips(h_data, originalData, N);

    if (bitFlipsDetected) {
        std::cout << "Rowhammer bit flips detected!" << std::endl;
    } else {
        std::cout << "No Rowhammer bit flips detected." << std::endl;
    }

    // Cleanup
    cudaFree(d_data);
    free(h_data);
    free(originalData);

    return 0;
}
