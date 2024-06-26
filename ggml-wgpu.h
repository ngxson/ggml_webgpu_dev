#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

// backend API
GGML_API ggml_backend_t ggml_backend_wgpu_init(void);
GGML_API bool ggml_backend_is_wgpu(ggml_backend_t backend);

// devide buffer
//GGML_API ggml_backend_buffer_type_t ggml_backend_wgpu_buffer_type(int device);

// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
//GGML_API ggml_backend_buffer_type_t ggml_backend_wgpu_host_buffer_type(void);

#ifdef  __cplusplus
}
#endif
