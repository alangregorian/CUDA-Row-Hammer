#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Kernel to measure memory access time with high-resolution timer
__global__ void measure_access_time(int *data, int stride, long long *time) {
    int index = threadIdx.x;
    unsigned long long start, end;

    int temp = data[0];
    start = clock64();
    temp = *(int*)((char*)data + index * stride); // Accessing the data at byte offset
    end = clock64();

    time[index] = end - start;
}

// Host function to test memory access with random strides
void test_memory_access(int *d_data, int dataSize, int numThreads, int *strides, int numStrides, long long *h_time) {
    long long *d_time;
    cudaMalloc(&d_time, numThreads * sizeof(long long));

    // Initialize random seed
    srand(time(NULL));

    // Array to keep track of accessed strides
    bool *accessed = (bool*)malloc(numStrides * sizeof(bool));
    for (int i = 0; i < numStrides; i++) {
        accessed[i] = false;
    }

    for (int i = 0; i < numStrides; i++) {
        int strideIndex;
        // Select a random stride that has not been accessed yet
        do {
            strideIndex = rand() % numStrides;
        } while (accessed[strideIndex]);
        accessed[strideIndex] = true;

        int stride = strides[strideIndex];
        printf("Testing stride: %d bytes\n", stride);

        // Launch kernel
        measure_access_time<<<1, numThreads>>>(d_data, stride, d_time);

        // Copy time results back to host
        cudaMemcpy(h_time + i * numThreads, d_time, numThreads * sizeof(long long), cudaMemcpyDeviceToHost);
    }

    cudaFree(d_time);
    free(accessed);
}

int main() {
    int dataSize = 1024 * 1024 * 2; // 2 MB of data
    int numThreads = 1; // Number of threads per block

    int *d_data;
    cudaMalloc(&d_data, dataSize * sizeof(int));

    // Initialize data (for demonstration purposes)
    int *h_data = (int*)malloc(dataSize * sizeof(int));
    for (int i = 0; i < dataSize; i++) {
        h_data[i] = i;
    }
    cudaMemcpy(d_data, h_data, dataSize * sizeof(int), cudaMemcpyHostToDevice);

    // Array of stride values in bytes
    int strides[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192};
    int numStrides = sizeof(strides) / sizeof(strides[0]);

    long long *h_time = (long long*)malloc(numThreads * numStrides * sizeof(long long));
    test_memory_access(d_data, dataSize, numThreads, strides, numStrides, h_time);

    // Write results to file
    FILE *file = fopen("access_times.txt", "w");
    for (int i = 0; i < numStrides; i++) {
        for (int j = 0; j < numThreads; j++) {
            fprintf(file, "%d %lld\n", strides[i], h_time[i * numThreads + j]);
        }
    }
    fclose(file);

    cudaFree(d_data);
    free(h_data);
    free(h_time);

    return 0;
}
