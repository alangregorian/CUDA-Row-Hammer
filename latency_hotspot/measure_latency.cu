#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

#define N 1024 * 1024  // Size of the array
#define NUM_ACCESSES 10000  // Number of repeated measurements

__global__ void measure_initial_latency(volatile int *array, unsigned long long *latencies, int index, int stride) {
    unsigned long long start, end;

    // Measure latency for two accesses separated by the stride
    int value1 = array[index];
    start = clock64();
    int value2 = array[index + stride];
    end = clock64();

    latencies[index] = end - start;
}

__global__ void measure_repeated_latency(volatile int *array, unsigned long long *latencies, int index) {
    for (int i = 0; i < NUM_ACCESSES; ++i) {
        unsigned long long start, end;

        // Measure latency for repeated accesses to the same index
        start = clock64();
        int value1 = array[index];
        end = clock64();

        latencies[i] = end - start;
    }
}

int main() {
    int *d_array;
    unsigned long long *d_latencies_initial;
    unsigned long long *d_latencies_repeated;
    int *d_indices;
    int *indices = (int *)malloc(N * sizeof(int));
    unsigned long long *latencies_initial = (unsigned long long *)malloc(N * sizeof(unsigned long long));
    unsigned long long *latencies_repeated = (unsigned long long *)malloc(NUM_ACCESSES * sizeof(unsigned long long));
    int stride = 128;  // Example stride value, can be modified as needed

    // Initialize random indices
    for (int i = 0; i < N; ++i) {
        indices[i] = rand() % (N - stride);  // Ensure index + stride is within bounds
    }

    // Allocate memory on the GPU
    cudaMalloc((void **)&d_array, N * sizeof(int));
    cudaMalloc((void **)&d_latencies_initial, N * sizeof(unsigned long long));
    cudaMalloc((void **)&d_latencies_repeated, NUM_ACCESSES * sizeof(unsigned long long));
    cudaMalloc((void **)&d_indices, N * sizeof(int));

    // Copy data to the GPU
    cudaMemcpy(d_indices, indices, N * sizeof(int), cudaMemcpyHostToDevice);

    // Initialize the array on the device
    cudaMemset(d_array, 1, N * sizeof(int));

    // Launch the initial latency measurement kernel with one thread at a time
    for (int i = 0; i < N; ++i) {
        measure_initial_latency<<<1, 1>>>((volatile int *)d_array, d_latencies_initial, i, stride);
    }

    // Synchronize to ensure all initial measurements are completed
    cudaDeviceSynchronize();

    // Copy one index for repeated latency measurement
    int single_index = indices[0]; // Use the first index as an example

    // Launch the repeated latency measurement kernel with one thread and one block
    measure_repeated_latency<<<1, 1>>>((volatile int *)d_array, d_latencies_repeated, single_index);

    // Synchronize to ensure all repeated measurements are completed
    cudaDeviceSynchronize();

    // Copy results back to the host
    cudaMemcpy(latencies_initial, d_latencies_initial, N * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaMemcpy(latencies_repeated, d_latencies_repeated, NUM_ACCESSES * sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    // Write the results to a CSV file
    FILE *f = fopen("latencies.csv", "w");
    fprintf(f, "Index,Initial Latency,Repeated Latency\n");
    for (int i = 0; i < N; ++i) {
        fprintf(f, "%d,%llu\n", indices[i], latencies_initial[i]);
    }
    for (int i = 0; i < NUM_ACCESSES; ++i) {
        fprintf(f, "%d,%llu\n", single_index, latencies_repeated[i]);
    }
    fclose(f);

    // Free memory
    cudaFree(d_array);
    cudaFree(d_latencies_initial);
    cudaFree(d_latencies_repeated);
    cudaFree(d_indices);
    free(indices);
    free(latencies_initial);
    free(latencies_repeated);

    return 0;
}
