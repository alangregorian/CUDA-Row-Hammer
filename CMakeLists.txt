cmake_minimum_required(VERSION 3.27)
project(CUDA_Row_Hammer CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(CUDA_Row_Hammer basic_row_hammer_test.cu)

set_target_properties(CUDA_Row_Hammer PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
