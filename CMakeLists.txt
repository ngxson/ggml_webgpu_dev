cmake_minimum_required(VERSION 3.13)
project(app)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_STANDARD_REQUIRED true)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED true)
set(THREADS_PREFER_PTHREAD_FLAG ON)

find_package(Threads REQUIRED)

# ggml
add_library(ggml OBJECT
            ggml.c
            ggml.h
            ggml-alloc.c
            ggml-alloc.h
            ggml-backend.c
            ggml-backend.h
            ggml-quants.c
            ggml-quants.h
            ggml-wgpu.cpp
            ggml-wgpu.h)
target_include_directories(ggml PUBLIC . ${LLAMA_EXTRA_INCLUDES})
target_compile_features   (ggml PUBLIC c_std_11) # don't bump
target_link_libraries(ggml PUBLIC Threads::Threads ${LLAMA_EXTRA_LIBS})

# app
add_executable(main "main.cpp")
add_executable(demo "demo.cpp")
add_library(ggml_static STATIC $<TARGET_OBJECTS:ggml>)

#include_directories("${CMAKE_CURRENT_LIST_DIR}/includes")
set_target_properties(main PROPERTIES SUFFIX ".html")
set_target_properties(demo PROPERTIES SUFFIX ".html")
target_link_options(main PRIVATE "-sUSE_WEBGPU=1" "-sUSE_GLFW=3" "-sALLOW_MEMORY_GROWTH=1" "-sASYNCIFY=1")
target_link_options(demo PRIVATE "-sUSE_WEBGPU=1" "-sUSE_GLFW=3" "-sALLOW_MEMORY_GROWTH=1" "-sASYNCIFY=1")
