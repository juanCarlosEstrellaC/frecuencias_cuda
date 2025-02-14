cmake_minimum_required(VERSION 3.0)
project(frecuencias CUDA)

set(CMAKE_CUDA_STANDARD 17)
enable_language(CXX CUDA)


#add_executable(frecuencias main.cpp kernel.cu)
#add_executable(frecuencias main_histograma.cpp kernel_histograma.cu)
add_executable(frecuencias main_prodint.cpp kernel_prodint.cu)
add_executable(frecuencias main_blur.cpp kernel_blur.cu)


target_include_directories(untitled PUBLIC
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

set_target_properties(frecuencias PROPERTIES
                      CUDA_SEPARABLE_COMPILATION ON)
