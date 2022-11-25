cmake_minimum_required(VERSION 3.22)
project(Project04)

set(CMAKE_CXX_STANDARD 17)

add_executable(Project04 matrix.h matrix.c benchmark.c matmul.h matmul.c)
