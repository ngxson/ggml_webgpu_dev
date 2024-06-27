#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

struct ggml_wgpu_tensor_params {
    int32_t ne00 = 0;
    int32_t ne01 = 0;
    int32_t ne02 = 0;
    int32_t ne03 = 0;

    uint32_t nb00 = 0;
    uint32_t nb01 = 0;
    uint32_t nb02 = 0;
    uint32_t nb03 = 0;

    int32_t ne10 = 0;
    int32_t ne11 = 0;
    int32_t ne12 = 0;
    int32_t ne13 = 0;

    uint32_t nb10 = 0;
    uint32_t nb11 = 0;
    uint32_t nb12 = 0;
    uint32_t nb13 = 0;

    int32_t ne0 = 0;
    int32_t ne1 = 0;
    int32_t ne2 = 0;
    int32_t ne3 = 0;

    uint32_t nb0 = 0;
    uint32_t nb1 = 0;
    uint32_t nb2 = 0;
    uint32_t nb3 = 0;

    uint32_t offs_src0 = 0;
    uint32_t offs_src1 = 0;
    uint32_t offs_dest = 0;
};

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
