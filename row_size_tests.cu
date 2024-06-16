#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

// Kernel to access global memory
__global__ void accessGlobalMemory(float* d_array, int stride, int accesses) {
    int idx = threadIdx.x;
    float sum = 0.0f;

    for (int i = 0; i < accesses; i++) {
        sum += d_array[(idx + i * stride) % accesses];
    }

    // Prevent compiler optimization
    if (sum > 0) {
        d_array[idx] = sum;
    }
}

void measureAccessTime(float* d_array, int stride, int accesses) {
    auto start = std::chrono::high_resolution_clock::now();
    accessGlobalMemory<<<1, 1>>>(d_array, stride, accesses);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> diff = end - start;
    std::cout << diff.count()*1000 << std::endl;
}

int main() {
    const int size = 1024 * 1024; // Size of the array (1M elements)
    const int accesses = 1024; // Number of accesses to test
    float* h_array = new float[size];
    float* d_array;

    // Initialize host array
    for (int i = 0; i < size; i++) {
        h_array[i] = static_cast<float>(i);
    }

    // Allocate device memory
    cudaMalloc(&d_array, size * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_array, h_array, size * sizeof(float), cudaMemcpyHostToDevice);

    // Test with different strides
    for (int stride = 1; stride <= 1024; stride *= 2) {
        measureAccessTime(d_array, stride, accesses);
    }

    // Free device memory
    cudaFree(d_array);

    // Free host memory
    delete[] h_array;

    return 0;
}
