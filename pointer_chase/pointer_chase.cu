#include <ctime>
#include <iostream>
#include <limits>

#include <getopt.h>

#include <cuda_runtime.h>

const struct option LongOptions[] = {
    {"help", no_argument, 0, 'h'},
    {"iterations", required_argument, 0, 'i'},
    {"random", no_argument, 0, 'r'},
    {"size", required_argument, 0, 'n'},
    {"stride", required_argument, 0, 's'},
    {0, 0, 0, 0}
};

void initPattern(unsigned int *pattern, const size_t size,
    const size_t stride, const bool random);
void printResults(const unsigned int *indices, const unsigned int *latencies,
    const size_t iterations, const size_t size);
void shufflePattern(unsigned int *pattern, const size_t size,
    const size_t stride);
void usage(const char *program);

__global__ void pointerChaseKernel(volatile unsigned int *pattern,
    unsigned int *indices, unsigned int *latencies, const size_t iterations);

void initPattern(unsigned int *pattern, const size_t size,
    const size_t stride, const bool random) {
    for (size_t i = 0; i < size; i += stride) {
        pattern[i] = i + stride;
    }
    pattern[size-stride] = 0;

    if (random) {
        shufflePattern(pattern, size, stride);
    }
}

void printResults(const unsigned int *indices, const unsigned int *latencies,
    const size_t iterations, const size_t size) {
    unsigned int currIndex = 0;

    std::cout << "Size,Index,Cycles" << std::endl;
    for (size_t i = 0; i < iterations; i++) {
        std::cout << size << "," << currIndex << "," << latencies[i]
                  << std::endl;

        currIndex = indices[i];
    }
}

// Durstenfeld shuffle
// https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle#The_modern_algorithm
void shufflePattern(unsigned int *pattern, const size_t size,
    const size_t stride) {
    for (size_t i = (size / stride) - 1; i > 0; i--) {
        unsigned int j = rand() % (i + 1);

        std::swap(pattern[j*stride], pattern[i*stride]);
    }
}

void usage(const char *program) {
    std::cout << "Usage: " << program << " [options]" << std::endl
              << "Options:" << std::endl
              << "  -h (--help)       \tShow help message" << std::endl
              << "  -i (--iterations) \tSet number of iterations" << std::endl
              << "  -n (--size)       \tSet array size" << std::endl
              << "  -r (--random)     \tEnable random indices" << std::endl
              << "  -s (--stride)     \tSet stride value" << std::endl;
}

// Fine-grained pointer chasing
// https://arxiv.org/pdf/1509.02308
__global__ void pointerChaseKernel(volatile unsigned int *pattern,
    unsigned int *indices, unsigned int *latencies, const size_t iterations) {
    clock_t start, stop;
    unsigned int j = 0;

    for (size_t i = 0; i < iterations; i++) {
        start = clock();
        j = pattern[j];
        indices[i] = j;
        stop = clock();

        latencies[i] = stop - start;
    }
}

int main(int argc, char *argv[]) {
    size_t iterations = 0;
    size_t size = 0;
    size_t stride = 0;
    bool random = false;

    try {
        int opt;
        int64_t optValue;
        while ((opt = getopt_long(argc, argv, "hi:n:rs:",
                LongOptions, nullptr)) != -1) {
            switch (opt) {
                case 'h':
                    usage(argv[0]);
                    return 0;
                case 'i':
                    if (optarg) {
                        optValue = std::stoll(optarg);
                        if (optValue > 0) {
                            iterations = static_cast<size_t>(optValue);
                        } else {
                            std::cerr << "Error: Invalid iterations value"
                                      << std::endl;
                            return -1;
                        }
                    } else {
                        usage(argv[0]);
                        return -1;
                    }
                    break;
                case 'n':
                    if (optarg) {
                        optValue = std::stoll(optarg);
                        if ((optValue > 0) && ((optValue % 2) == 0) &&
                            (optValue <
                                std::numeric_limits<unsigned int>::max())) {
                            size = static_cast<size_t>(optValue);
                        } else {
                            std::cerr << "Error: Invalid size value"
                                      << std::endl;
                            return -1;
                        }
                    } else {
                        usage(argv[0]);
                        return -1;
                    }
                    break;
                case 'r':
                    random = true;
                    break;
                case 's':
                    if (optarg) {
                        optValue = std::stoll(optarg);
                        if (optValue > 0) {
                            stride = static_cast<size_t>(optValue);
                        } else {
                            std::cerr << "Error: Invalid stride value"
                                      << std::endl;
                            return -1;
                        }
                    } else {
                        usage(argv[0]);
                        return -1;
                    }
                    break;
                case '?':
                    usage(argv[0]);
                    return -1;
                default:
                    usage(argv[0]);
                    return -1;
            }
        }
    } catch (const std::invalid_argument &ia) {
        std::cerr << "Invalid argument: " << ia.what() << std::endl;
        return -1;
    } catch (const std::out_of_range &oor) {
        std::cerr << "Out of range error: " << oor.what() << std::endl;
        return -1;
    }

    if (iterations == 0) {
        std::cerr << "Error: No iterations value provided" << std::endl;
        return -1;
    } else if (size == 0) {
        std::cerr << "Error: No size value provided" << std::endl;
        return -1;
    } else if (stride == 0) {
        std::cerr << "Error: No stride value provided" << std::endl;
        return -1;
    }

    if (stride >= size) {
        std::cerr << "Error: Invalid stride value" << std::endl;
        return -1;
    }

    srand(static_cast<unsigned int>(time(nullptr)));
    
    unsigned int *hostLatencies, *hostPattern, *hostIndices;
    hostIndices = static_cast<unsigned int*>(
        malloc(iterations * sizeof(unsigned int)));
    hostLatencies = static_cast<unsigned int*>(
        malloc(iterations * sizeof(unsigned int)));
    hostPattern = static_cast<unsigned int*>(
        malloc(size * sizeof(unsigned int)));

    memset(hostIndices, 0, iterations * sizeof(unsigned int));
    memset(hostLatencies, 0, iterations * sizeof(unsigned int));
    memset(hostPattern, 0, size * sizeof(unsigned int));
    initPattern(hostPattern, size, stride, random);

    unsigned int *deviceIndices, *deviceLatencies, *devicePattern;
    cudaMalloc(reinterpret_cast<void**>(&deviceIndices),
        iterations * sizeof(unsigned int));
    cudaMalloc(reinterpret_cast<void**>(&deviceLatencies),
        iterations * sizeof(unsigned int));
    cudaMalloc(reinterpret_cast<void**>(&devicePattern),
        size * sizeof(unsigned int));

    cudaMemset(deviceIndices, 0, iterations * sizeof(unsigned int));
    cudaMemset(deviceLatencies, 0, iterations * sizeof(unsigned int));
    cudaMemcpy(devicePattern, hostPattern, size * sizeof(unsigned int),
        cudaMemcpyHostToDevice);

    cudaError_t ret;

    dim3 blockDim(1, 1, 1);
    dim3 gridDim(1, 1, 1);

    void *args[] = {&devicePattern, &deviceIndices, &deviceLatencies,
        &iterations};

    ret = cudaLaunchKernel((const void*)pointerChaseKernel, gridDim, blockDim,
                            args, 0, nullptr);
    cudaDeviceSynchronize();

    if (ret != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(ret) << std::endl;

        cudaFree(deviceIndices);
        cudaFree(devicePattern);
        cudaFree(deviceLatencies);
        free(hostIndices);
        free(hostPattern);
        free(hostLatencies);

        return -1;
    }

    cudaMemcpy(hostIndices, deviceIndices, iterations * sizeof(unsigned int),
        cudaMemcpyDeviceToHost);
    cudaMemcpy(hostLatencies, deviceLatencies,
         iterations * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    printResults(hostIndices, hostLatencies, iterations, size);

    cudaFree(deviceIndices);
    cudaFree(deviceLatencies);
    cudaFree(devicePattern);
    free(hostIndices);
    free(hostLatencies);
    free(hostPattern);

    return 0;
}
