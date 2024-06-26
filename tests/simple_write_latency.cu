#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>

#define N 1024*1024  // Number of iterations

__global__ void write_latency_test(volatile int* d_array, const int* d_random_values, unsigned long long* d_time) {
    for (int idx = 0; idx < 5000; idx++) {
        unsigned long long start_time = clock();

        int temp = d_random_values[idx];
        // Write operation
        d_array[idx] = temp;

        unsigned long long end_time = clock();

        // Record the latency
        d_time[idx] = end_time - start_time;
    }
}

int main() {
    srand(time(0));

    volatile int *d_array;
    int *d_random_values;
    unsigned long long *d_time;
    cudaMalloc((void**)&d_array, N * sizeof(volatile int));
    cudaMalloc(&d_random_values, N * sizeof(int));
    cudaMalloc(&d_time, N * sizeof(unsigned long long));

    int *h_random_values = new int[N];
    for (int i = 0; i < N; i++) {
        h_random_values[i] = rand() % 1000;  // Random values between 0 and 999
    }

    // Copy random values to device
    cudaMemcpy(d_random_values, h_random_values, N * sizeof(int), cudaMemcpyHostToDevice);

    // Launch a single thread
    write_latency_test<<<1, 1>>>(d_array, d_random_values, d_time);
    cudaDeviceSynchronize();

    // Copy the timing data back to the host
    unsigned long long *h_time = new unsigned long long[N];
    cudaMemcpy(h_time, d_time, N * sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    // Save the timings to a CSV file
    std::ofstream csv_file("write_latencies.csv");
    csv_file << "Iteration,Time\n";
    for (int i = 0; i < 5000; i++) {
        csv_file << i << "," << h_time[i] << "\n";
    }
    csv_file.close();

    delete[] h_random_values;
    delete[] h_time;
    cudaFree((void*)d_array);
    cudaFree(d_random_values);
    cudaFree(d_time);
    return 0;
}
