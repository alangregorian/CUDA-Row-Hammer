#include <stdio.h>
#include <cuda.h>
#include <fstream>

#define N 10 // Number of kernels

__global__ void kernel0(int *data, float *time) {
    clock_t start, end;
    start = clock();
    int value0 = data[0];
    end = clock();
    time[0] = (float)(end - start);

    start = clock();
    int valueN = data[N * 500 / sizeof(int)];
    end = clock();
    time[1] = (float)(end - start);
}

__global__ void kernel1(int *data, float *time) {
    clock_t start, end;
    start = clock();
    int value0 = data[0];
    end = clock();
    time[0] = (float)(end - start);

    start = clock();
    int valueN = data[N * 500 / sizeof(int)];
    end = clock();
    time[1] = (float)(end - start);
}

__global__ void kernel2(int *data, float *time) {
    clock_t start, end;
    start = clock();
    int value0 = data[0];
    end = clock();
    time[0] = (float)(end - start);

    start = clock();
    int valueN = data[N * 500 / sizeof(int)];
    end = clock();
    time[1] = (float)(end - start);
}

__global__ void kernel3(int *data, float *time) {
    clock_t start, end;
    start = clock();
    int value0 = data[0];
    end = clock();
    time[0] = (float)(end - start);

    start = clock();
    int valueN = data[N * 500 / sizeof(int)];
    end = clock();
    time[1] = (float)(end - start);
}

__global__ void kernel4(int *data, float *time) {
    clock_t start, end;
    start = clock();
    int value0 = data[0];
    end = clock();
    time[0] = (float)(end - start);

    start = clock();
    int valueN = data[N * 500 / sizeof(int)];
    end = clock();
    time[1] = (float)(end - start);
}

__global__ void kernel5(int *data, float *time) {
    clock_t start, end;
    start = clock();
    int value0 = data[0];
    end = clock();
    time[0] = (float)(end - start);

    start = clock();
    int valueN = data[N * 500 / sizeof(int)];
    end = clock();
    time[1] = (float)(end - start);
}

__global__ void kernel6(int *data, float *time) {
    clock_t start, end;
    start = clock();
    int value0 = data[0];
    end = clock();
    time[0] = (float)(end - start);

    start = clock();
    int valueN = data[N * 500 / sizeof(int)];
    end = clock();
    time[1] = (float)(end - start);
}

__global__ void kernel7(int *data, float *time) {
    clock_t start, end;
    start = clock();
    int value0 = data[0];
    end = clock();
    time[0] = (float)(end - start);

    start = clock();
    int valueN = data[N * 500 / sizeof(int)];
    end = clock();
    time[1] = (float)(end - start);
}

__global__ void kernel8(int *data, float *time) {
    clock_t start, end;
    start = clock();
    int value0 = data[0];
    end = clock();
    time[0] = (float)(end - start);

    start = clock();
    int valueN = data[N * 500 / sizeof(int)];
    end = clock();
    time[1] = (float)(end - start);
}

__global__ void kernel9(int *data, float *time) {
    clock_t start, end;
    start = clock();
    int value0 = data[0];
    end = clock();
    time[0] = (float)(end - start);

    start = clock();
    int valueN = data[N * 500 / sizeof(int)];
    end = clock();
    time[1] = (float)(end - start);
}

int main() {
    int *d_data;
    float *d_time[N], h_time[N][2];

    // Allocate memory
    cudaMalloc((void **)&d_data, (N * 500 + 1) * sizeof(int));
    for (int i = 0; i < N; i++) {
        cudaMalloc((void **)&d_time[i], 2 * sizeof(float));
    }

    // Initialize data
    int *h_data = (int *)malloc((N * 500 + 1) * sizeof(int));
    for (int i = 0; i < N * 500 + 1; i++) {
        h_data[i] = i;
    }
    cudaMemcpy(d_data, h_data, (N * 500 + 1) * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernels
    kernel0<<<1, 1>>>(d_data, d_time[0]);
    kernel1<<<1, 1>>>(d_data, d_time[1]);
    kernel2<<<1, 1>>>(d_data, d_time[2]);
    kernel3<<<1, 1>>>(d_data, d_time[3]);
    kernel4<<<1, 1>>>(d_data, d_time[4]);
    kernel5<<<1, 1>>>(d_data, d_time[5]);
    kernel6<<<1, 1>>>(d_data, d_time[6]);
    kernel7<<<1, 1>>>(d_data, d_time[7]);
    kernel8<<<1, 1>>>(d_data, d_time[8]);
    kernel9<<<1, 1>>>(d_data, d_time[9]);

    // Copy results back to host
    for (int i = 0; i < N; i++) {
        cudaMemcpy(h_time[i], d_time[i], 2 * sizeof(float), cudaMemcpyDeviceToHost);
    }

    // Save results to CSV
    std::ofstream file("results.csv");
    file << "Index,ZeroTime,IndexTime\n";
    for (int i = 0; i < N; i++) {
        file << i << "," << h_time[i][0] << "," << h_time[i][1] << "\n";
    }
    file.close();

    // Free memory
    cudaFree(d_data);
    for (int i = 0; i < N; i++) {
        cudaFree(d_time[i]);
    }
    free(h_data);

    return 0;
}
