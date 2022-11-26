cmake_minimum_required(VERSION 3.22)
project(matmul)

set(CMAKE_CXX_STANDARD 17)

add_compile_options(-O2)
add_compile_options(-mavx2)
add_compile_options(-lopenblas)

include_directories(/usr/local/include ./inc)
link_libraries(/usr/local/lib/libopenblas.a)

aux_source_directory(./src DIR_SRC)

add_executable(matmul ${DIR_SRC})

find_package(OpenMP)
if(OpenMP_C_FOUND)
    target_link_libraries(matmul PUBLIC OpenMP::OpenMP_C)
endif()