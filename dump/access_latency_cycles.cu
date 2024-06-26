#include <cuda_runtime.h>
#include <iostream>
#include <fstream>

__global__ void measureAccessLatency(int *data, int stride, int iterations, unsigned long long *timings) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned long long start, end;
    for (int i = 0; i < iterations; i++) {
        // Memory fence to ensure all previous operations are complete
        asm volatile("membar.gl;":::"memory");

        // Use inline PTX to read the global timer for start time
        asm volatile ("mov.u64 %0, %%globaltimer;" : "=l"(start) :: "memory");
        int temp = data[idx * stride];

        // Memory fence to ensure the load operation is complete
        asm volatile("membar.gl;":::"memory");

        // Use inline PTX to read the global timer for end time
        asm volatile ("mov.u64 %0, %%globaltimer;" : "=l"(end) :: "memory");
        timings[idx * iterations + i] = end - start;
    }
}

int main() {
    int *d_data;
    unsigned long long *d_timings;
    int size = 1024;
    int iterations = 100;

    cudaMalloc(&d_data, size * sizeof(int));
    cudaMemset(d_data, 0, size * sizeof(int));
    cudaMalloc(&d_timings, size * iterations * sizeof(unsigned long long));

    int numThreads = 256;
    int numBlocks = (size + numThreads - 1) / numThreads;

    measureAccessLatency<<<numBlocks, numThreads>>>(d_data, 256, iterations, d_timings);
    cudaDeviceSynchronize();

    unsigned long long *h_timings = new unsigned long long[size * iterations];
    cudaMemcpy(h_timings, d_timings, size * iterations * sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    // Save timings to a file
    std::ofstream outFile("access_cycles_cycles.txt");
    if (outFile.is_open()) {
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < iterations; j++) {
                outFile << "Timing[" << i << "][" << j << "]: " << h_timings[i * iterations + j] << " cycles" << std::endl;
            }
        }
        outFile.close();
    } else {
        std::cerr << "Unable to open file for writing." << std::endl;
    }

    cudaFree(d_data);
    cudaFree(d_timings);
    delete[] h_timings;

    return 0;
}
