//
// Created by aj on 6/26/24.
//

#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <assert.h>

#include "dram_read_time_helpers.cu"

// Definitions of constants
#define GPU_MAX_OUTER_LOOP 10
#define GPU_L2_CACHE_LINE_SIZE 128

// Function declarations
double device_find_dram_read_time(void *_a, void *_b, double threshold);

// Utility functions to initialize arrays
void initialize_array(uint64_t *array, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        array[i] = (i + 1) % size;
    }
}

// Test function
void test_dram_read_speed() {
    const size_t array_size = 1024; // Example size, can be adjusted
    uint64_t *h_a = new uint64_t[array_size];
    uint64_t *h_b = new uint64_t[array_size];

    initialize_array(h_a, array_size);
    initialize_array(h_b, array_size);

    uint64_t *d_a, *d_b, *d_refresh_v;
    double *d_ticks;
    uint64_t *d_sum;
    size_t refresh_size = array_size;

    // Allocate device memory
    cudaMalloc((void**)&d_a, array_size * sizeof(uint64_t));
    cudaMalloc((void**)&d_b, array_size * sizeof(uint64_t));
    cudaMalloc((void**)&d_refresh_v, refresh_size * sizeof(uint64_t));
    cudaMalloc((void**)&d_ticks, GPU_MAX_OUTER_LOOP * sizeof(double));
    cudaMalloc((void**)&d_sum, GPU_MAX_OUTER_LOOP * sizeof(uint64_t));

    // Copy data to device
    cudaMemcpy(d_a, h_a, array_size * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, array_size * sizeof(uint64_t), cudaMemcpyHostToDevice);

    // Set threshold for the test
    double threshold = 1000.0; // Example threshold, can be adjusted

    // Call the function to measure DRAM read time
    double min_ticks = device_find_dram_read_time(d_a, d_b, threshold);

    // Print the result
    std::cout << "Minimum DRAM read time (ticks): " << min_ticks << std::endl;

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_refresh_v);
    cudaFree(d_ticks);
    cudaFree(d_sum);

    // Free host memory
    delete[] h_a;
    delete[] h_b;
}

int main() {
    // Run the test
    test_dram_read_speed();

    return 0;
}
