cmake_minimum_required(VERSION 3.22)
project(matmul)

set(CMAKE_CXX_STANDARD 17)

add_compile_options(-O3)
add_compile_options(-lopenblas)

include_directories(/usr/local/include)
link_libraries(/usr/local/lib/libopenblas.a)

add_executable(matmul matrix.h matrix.c benchmark.c matmul.h matmul.c)

find_package(OpenMP)
if(OpenMP_C_FOUND)
    target_link_libraries(matmul PUBLIC OpenMP::OpenMP_C)
endif()