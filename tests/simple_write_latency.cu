#include <cuda_runtime.h>
#include <iostream>
#include <fstream>

#define N 1024*1024  // Number of iterations

__global__ void write_latency_test(volatile int* d_array, unsigned long long* d_time, int value) {
    for (int idx = 0; idx < 1000; idx++) {
        unsigned long long start_time = clock64();

        // Write operation
        d_array[idx] = value;

        unsigned long long end_time = clock64();

        // Record the latency
        d_time[idx] = end_time - start_time;
    }
}

int main() {
    volatile int *d_array;
    unsigned long long *d_time;
    cudaMalloc((void**)&d_array, N * sizeof(volatile int));
    cudaMalloc(&d_time, N * sizeof(unsigned long long));

    int value = 42;

    // Launch a single thread
    write_latency_test<<<1, 1>>>(d_array, d_time, value);
    cudaDeviceSynchronize();

    // Copy the timing data back to the host
    unsigned long long *h_time = new unsigned long long[N];
    cudaMemcpy(h_time, d_time, N * sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    // Save the timings to a CSV file
    std::ofstream csv_file("write_latencies.csv");
    csv_file << "Iteration,Time\n";
    for (int i = 0; i < 1000; i++) {
        csv_file << i << "," << h_time[i] << "\n";
    }
    csv_file.close();

    delete[] h_time;
    cudaFree((void*)d_array);
    cudaFree(d_time);
    return 0;
}
