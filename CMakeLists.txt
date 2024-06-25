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
            ggml-quants.h)
target_include_directories(ggml PUBLIC . ${LLAMA_EXTRA_INCLUDES})
target_compile_features   (ggml PUBLIC c_std_11) # don't bump
target_link_libraries(ggml PUBLIC Threads::Threads ${LLAMA_EXTRA_LIBS})

# app
add_executable(app "main.cpp")
add_library(ggml_static STATIC $<TARGET_OBJECTS:ggml>)

#include_directories("${CMAKE_CURRENT_LIST_DIR}/includes")
set_target_properties(app PROPERTIES SUFFIX ".html")
target_link_options(app PRIVATE "-sUSE_WEBGPU=1" "-sUSE_GLFW=3")
