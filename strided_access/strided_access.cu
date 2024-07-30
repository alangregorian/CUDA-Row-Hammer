#include <ctime>
#include <iostream>
#include <limits>

#include <getopt.h>

#include <cuda_runtime.h>
#include <cuda.h>

const char *FILENAME = "strided_access.cubin";
const char *KERNEL_NAME = "strided_access";
const size_t ITERATIONS = 128;

int main(int argc, char *argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <array-size> <start-offset> <stride>" << std::endl;
        return -1;
    }

    size_t size;
    unsigned int offset, stride;
    unsigned int offsetBytes, strideBytes;
    try {
        int64_t argValue;
        for (int i = 1; i < argc; i++) {
            switch(i) {
                case 1:
                    argValue = std::stoll(argv[i]);
                    if ((argValue < std::numeric_limits<unsigned int>::max()) &&
                        (argValue > 0) && ((argValue % 2) == 0)) {
                        size = static_cast<size_t>(argValue);
                    } else {
                        std::cerr << "Error: Invalid size value" << std::endl;
                        return -1;
                    }
                    break;
                case 2:
                    argValue = std::stoll(argv[i]);
                    if ((argValue < static_cast<int64_t>(size)) &&
                        (argValue >= 0)) {
                        offset = static_cast<unsigned int>(argValue);
                        offsetBytes = offset * sizeof(unsigned int);
                    } else {
                        std::cerr << "Error: Invalid offset value" << std::endl;
                        return -1;
                    }
                    break;
                case 3:
                    argValue = std::stoll(argv[i]);
                    if (((offset + (argValue * (ITERATIONS - 1))) < size) &&
                        (argValue > 0)) {
                        stride = static_cast<unsigned int>(argValue);
                        strideBytes = stride * sizeof(unsigned int);
                    } else {
                        std::cerr << "Error: Invalid stride value" << std::endl;
                        return -1;
                    }
                    break;
                default:
                    std::cerr << "Usage: " << argv[0]
                              << " <array-size> <start-offset> <stride>"
                              << std::endl;
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

    unsigned int *hostInput, *hostClock;
    hostInput = static_cast<unsigned int*>(
        malloc(size * sizeof(unsigned int)));
    hostClock = static_cast<unsigned int*>(
        malloc(ITERATIONS * sizeof(unsigned int)));

    srand(static_cast<unsigned int>(time(nullptr)));
    for (size_t i = 0; i < size; i++) {
        hostInput[i] = rand();
    }

    unsigned int *deviceInput, *deviceClock;
    cudaMalloc(reinterpret_cast<void**>(&deviceInput),
        size * sizeof(unsigned int));
    cudaMalloc(reinterpret_cast<void**>(&deviceClock),
        ITERATIONS * sizeof(unsigned int));
    cudaMemcpy(deviceInput, hostInput, size * sizeof(unsigned int),
        cudaMemcpyHostToDevice);
    cudaMemset(deviceClock, 0, ITERATIONS * sizeof(unsigned int));

    CUmodule module;
    CUfunction kernel;

    cuModuleLoad(&module, FILENAME);
    cuModuleGetFunction(&kernel, module, KERNEL_NAME);

    void *args[] = {&deviceInput, &deviceClock, &offsetBytes, &strideBytes};

    cuLaunchKernel(kernel, 1, 1, 1, 1, 1, 1, 0, 0, args, 0);
    cudaDeviceSynchronize();

    cudaMemcpy(hostClock, deviceClock, ITERATIONS * sizeof(unsigned int),
        cudaMemcpyDeviceToHost);

    std::cout << "Size,Index,Cycles" << std::endl;
    for (size_t i = 0; i < ITERATIONS; i++) {
        std::cout << size << "," << (offset + (stride * i)) << ","
                  << hostClock[i] << std::endl;
    }
 
    cudaFree(deviceInput);
    cudaFree(deviceClock);
    free(hostInput);
    free(hostClock);

    return 0;
}
