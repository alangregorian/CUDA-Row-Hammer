#include <CL/cl.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>

// Size of the data array
const size_t DATA_SIZE = 1024 * 1024;
const size_t MAX_BYTES = 6000;

// OpenCL kernel to read from global memory byte by byte
const char* kernelSource = R"(
__kernel void readGlobalMemory(__global float* input, __global float* output) {
    int id = get_global_id(0);
    if (id < get_global_size(0)) {
        output[id] = input[id];
    }
}
)";

void checkError(cl_int err, const char* operation) {
    if (err != CL_SUCCESS) {
        std::cerr << "Error during operation '" << operation << "': " << err << std::endl;
        exit(1);
    }
}

int main() {
    // Initialize data
    std::vector<float> data(DATA_SIZE, 1.0f);
    std::vector<float> result(DATA_SIZE, 0.0f);

    // Open output file
    std::ofstream outputFile("latency_results.csv");
    if (!outputFile.is_open()) {
        std::cerr << "Failed to open output file." << std::endl;
        return 1;
    }
    outputFile << "Index,Latency\n";

    // Get platform and device information
    cl_platform_id platform;
    cl_device_id device;
    cl_uint numPlatforms;
    cl_uint numDevices;
    checkError(clGetPlatformIDs(1, &platform, &numPlatforms), "clGetPlatformIDs");
    checkError(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, &numDevices), "clGetDeviceIDs");

    // Create an OpenCL context
    cl_int err;
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    checkError(err, "clCreateContext");

    // Create a command queue
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
    checkError(err, "clCreateCommandQueue");

    // Create memory buffers on the device
    cl_mem inputBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, DATA_SIZE * sizeof(float), NULL, &err);
    checkError(err, "clCreateBuffer(input)");
    cl_mem outputBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, DATA_SIZE * sizeof(float), NULL, &err);
    checkError(err, "clCreateBuffer(output)");

    // Copy the data to the input buffer
    checkError(clEnqueueWriteBuffer(queue, inputBuffer, CL_TRUE, 0, DATA_SIZE * sizeof(float), data.data(), 0, NULL, NULL), "clEnqueueWriteBuffer");

    // Create and build the program
    cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, &err);
    checkError(err, "clCreateProgramWithSource");
    checkError(clBuildProgram(program, 1, &device, NULL, NULL, NULL), "clBuildProgram");

    // Create the OpenCL kernel
    cl_kernel kernel = clCreateKernel(program, "readGlobalMemory", &err);
    checkError(err, "clCreateKernel");

    // Measure latency for accessing each byte
    for (size_t index = 0; index < MAX_BYTES; index++) {
        // Set the kernel arguments
        checkError(clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputBuffer), "clSetKernelArg(0)");
        checkError(clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputBuffer), "clSetKernelArg(1)");

        // Execute the kernel and measure the time
        auto start = std::chrono::high_resolution_clock::now();
        size_t globalWorkSize = 1;
        checkError(clEnqueueNDRangeKernel(queue, kernel, 1, &index, &globalWorkSize, NULL, 0, NULL, NULL), "clEnqueueNDRangeKernel");
        checkError(clFinish(queue), "clFinish");
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsedTime = end - start;

        // Write results to the file
        outputFile << index << "," << elapsedTime.count() << "\n";
        std::cout << "Index: " << index << ", Latency: " << elapsedTime.count() << " seconds" << std::endl;
    }

    // Close the output file
    outputFile.close();

    // Clean up
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(inputBuffer);
    clReleaseMemObject(outputBuffer);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
