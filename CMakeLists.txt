cmake_minimum_required(VERSION 3.27)
project(CUDA_Row_Hammer CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(CUDA_Row_Hammer dump/basic_row_hammer_test.cu
        dump/row_access_timer.cu
        dump/ptx_test.cu
        dump/access_latency_cycles.cu
        dump/access_latency_timing.cu
        dump/open_closed_policy_test.cu
        last_attempt_timing/individual_kernals.cu
        tests/dram_read_time_helpers.cu
        reverse_engineering.hpp
        tests/test_read_latency.cu
        tests/simple_write_latency.cu
        OpenCL_tests/latency_test.cpp
        OpenCL_tests/latency_test_2.cpp
        latency_hotspot/measure_latency.cu)

set_target_properties(CUDA_Row_Hammer PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
