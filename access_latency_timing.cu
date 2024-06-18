//#include <cuda_runtime.h>
//#include <iostream>
//#include <fstream>
//#include <chrono>
//
//// Kernel to access global memory
//__global__ void accessGlobalMemory(float* d_array, int stride, int accesses) {
//    int idx = threadIdx.x;
//    float sum = 0.0f;
//
//    for (int i = 0; i < accesses; i++) {
//        sum += d_array[(idx + i * stride) % accesses];
//    }
//
//    // Prevent compiler optimization
//    if (sum > 0) {
//        d_array[idx] = sum;
//    }
//}
//
//void measureAccessTime(float* d_array, int stride, int accesses, std::ofstream& file) {
//    auto start = std::chrono::high_resolution_clock::now();
//    accessGlobalMemory<<<1, 1>>>(d_array, stride, accesses);
//    cudaDeviceSynchronize();
//    auto end = std::chrono::high_resolution_clock::now();
//
//    std::chrono::duration<double> diff = end - start;
//    file << diff.count() << std::endl;
//}
//
//int main() {
//    const int size = 1024 * 1024; // Size of the array (1M elements)
//    const int accesses = 1024; // Number of accesses to test
//    float* h_array = new float[size];
//    float* d_array;
//
//    // Initialize host array
//    for (int i = 0; i < size; i++) {
//        h_array[i] = static_cast<float>(i);
//    }
//
//    // Allocate device memory
//    cudaMalloc(&d_array, size * sizeof(float));
//
//    // Copy data from host to device
//    cudaMemcpy(d_array, h_array, size * sizeof(float), cudaMemcpyHostToDevice);
//
//    // Open the file to write results
//    std::ofstream file("access_latency_timing.txt");
//
//    // Test with different strides
//    for (int stride = 1; stride <= 8192; stride *= 2) {
//        measureAccessTime(d_array, stride, accesses, file);
//    }
//
//    // Close the file
//    file.close();
//
//    // Free device memory
//    cudaFree(d_array);
//
//    // Free host memory
//    delete[] h_array;
//
//    return 0;
//}

#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <chrono>

// Kernel to access global memory
__global__ void accessGlobalMemory(float* d_array, int index) {
    int idx = threadIdx.x;
    float sum = 0.0f;

    // Access a specific index
    sum += d_array[index];

    // Prevent compiler optimization
    if (sum > 0) {
        d_array[idx] = sum;
    }
}

void measureAccessTime(float* d_array, int index, std::ofstream& file) {
    auto start = std::chrono::high_resolution_clock::now();
    accessGlobalMemory<<<1, 1>>>(d_array, index);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> diff = end - start;
    file << diff.count() << std::endl;
}

int main() {
    const int size = 1024 * 1024; // Size of the array (1M elements)
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

    // Open the file to write results
    std::ofstream file("access_latency_timing.txt");

    // Test with different indices from 1 to 8192
    for (int index = 1; index <= 8192; ++index) {
        measureAccessTime(d_array, index, file);
    }

    // Close the file
    file.close();

    // Free device memory
    cudaFree(d_array);

    // Free host memory
    delete[] h_array;

    return 0;
}

