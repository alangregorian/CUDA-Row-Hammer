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
#include <vector>

// Kernel to access global memory in bytes and measure time
__global__ void accessGlobalMemory(char* d_array, int zero_index, int offset_index, unsigned long long* zero_time, unsigned long long* offset_time) {
    int idx = threadIdx.x;
    char sum = 0;

    // Measure time for the fifth access to zero_index
    unsigned long long start, end;

    start = clock64();
    sum += d_array[zero_index];
    end = clock64();
    zero_time[0] = end - start;

    __syncthreads();

    // Measure time for accessing the offset index
    start = clock64();
    sum += d_array[offset_index];
    end = clock64();
    offset_time[0] = end - start;

    // Prevent compiler optimization
    if (sum > 0) {
        d_array[idx] = sum;
    }
}

void measureAccessTime(char* d_array, int zero_index, int offset_index, unsigned long long& zero_time, unsigned long long& offset_time) {
    unsigned long long* d_zero_time;
    unsigned long long* d_offset_time;

    // Allocate device memory for timing results
    cudaMalloc(&d_zero_time, sizeof(unsigned long long));
    cudaMalloc(&d_offset_time, sizeof(unsigned long long));

    // Launch kernel to measure access times
    accessGlobalMemory<<<1, 1>>>(d_array, zero_index, offset_index, d_zero_time, d_offset_time);
    cudaDeviceSynchronize();

    // Copy timing results back to host
    cudaMemcpy(&zero_time, d_zero_time, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaMemcpy(&offset_time, d_offset_time, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_zero_time);
    cudaFree(d_offset_time);
}

void warmUpMemory(char* d_array, const std::vector<int>& indices) {
    for (int i = 0; i < 1000; ++i) {  // Access 1000 random indices for warm-up
        accessGlobalMemory<<<1, 1>>>(d_array, indices[i], indices[i], nullptr, nullptr);
    }
    cudaDeviceSynchronize();
}

int main() {
    const int total_size = 1024 * 1024; // Size of the array (1M elements)
    const int measure_size = 8192; // Measure up to 8K bytes
    char* h_array = new char[total_size];
    char* d_array;

    // Initialize host array
    for (int i = 0; i < total_size; i++) {
        h_array[i] = static_cast<char>(i % 256);  // Initialize with byte values
    }

    // Allocate device memory
    cudaMalloc(&d_array, total_size * sizeof(char));

    // Copy data from host to device
    cudaMemcpy(d_array, h_array, total_size * sizeof(char), cudaMemcpyHostToDevice);

    // Create a vector of indices
    std::vector<int> indices;
    for (int i = 0; i <= measure_size; i += 100) {
        indices.push_back(i);
    }

    // Warm-up memory
    warmUpMemory(d_array, indices);

    // Open the file to write results
    std::ofstream file("access_latency_timing.csv");

    // Write the header for CSV
    file << "BytesFromZero,TimeZero,TimeN100\n";

    // Measure and write access times
    for (int i = 0; i < indices.size(); ++i) {
        // Measure access time for zero and the offset index
        unsigned long long zero_time, offset_time;
        measureAccessTime(d_array, 0, indices[i], zero_time, offset_time);

        // Write the results to the file
        file << indices[i] << "," << zero_time << "," << offset_time << "\n";
    }

    // Close the file
    file.close();

    // Free device memory
    cudaFree(d_array);

    // Free host memory
    delete[] h_array;

    return 0;
}
