cmake_minimum_required(VERSION 3.23)
project(cppbench)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CXX_FLAGS "-march=native")

set(BENCHMARK_DOWNLOAD_DEPENDENCIES ON)
add_subdirectory(vendor/benchmark)

add_executable(cppbench main.cpp)
target_include_directories(cppbench PRIVATE benchmark::benchmark)
target_link_libraries(cppbench benchmark::benchmark)
