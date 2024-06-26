#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <chrono>

#define N 1024*1024  // Array size

__global__ void write_latency_test(int* d_array, unsigned long long* d_time, int value) {
//    int idx = blockIdx.x * blockDim.x + threadIdx.x;
//
//    if (idx < N) {
//        unsigned long long start_time = clock64();
//
//        // Write operation
//        d_array[idx] = value;
//
//        unsigned long long end_time = clock64();
//
//        // Record the latency
//        d_time[idx] = end_time - start_time;
//    }
    for (int idx = 0; idx < 8192*2; idx++) {
        unsigned long long start_time = clock64();

        // Write operation
        d_array[idx] = value;

        unsigned long long end_time = clock64();

        // Record the latency
        d_time[idx] = end_time - start_time;
    }
}

int main() {
    int *d_array;
    unsigned long long *d_time;
    cudaMalloc(&d_array, N * sizeof(int));
    cudaMalloc(&d_time, N * sizeof(unsigned long long));

    int value = 42;
//    int threadsPerBlock = 256;
//    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
//
//    write_latency_test<<<blocksPerGrid, threadsPerBlock>>>(d_array, d_time, value);
    write_latency_test<<<1, 1>>>(d_array, d_time, value);
    cudaDeviceSynchronize();

    // Copy the timing data back to the host
    unsigned long long *h_time = new unsigned long long[N];
    cudaMemcpy(h_time, d_time, N * sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    // Save the timings to a CSV file
    std::ofstream csv_file("write_latencies.csv");
    csv_file << "Iteration,Time\n";
    for (int i = 0; i < 8192*2; i++) {
        csv_file << i << "," << h_time[i] << "\n";
    }
    csv_file.close();

    delete[] h_time;
    cudaFree(d_array);
    cudaFree(d_time);
    return 0;
}
