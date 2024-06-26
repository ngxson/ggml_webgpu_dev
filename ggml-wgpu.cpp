#include <GLFW/glfw3.h>
#include <iostream>
#include <vector>
#include <emscripten/emscripten.h>

#define WEBGPU_CPP_IMPLEMENTATION
#include "ggml-wgpu/webgpu.hpp"

#include "ggml-wgpu.h"
#include "ggml.h"
#include "ggml-backend-impl.h"

#define BUF_ALIGN 256
#define N_TENSOR_PARAMS 32
#define UNUSED GGML_UNUSED

struct ggml_wgpu_context {
    wgpu::Instance        instance;
    wgpu::Adapter         adapter;
    wgpu::Device          device;

    wgpu::SupportedLimits limits;
    wgpu::Queue           queue;
    wgpu::ShaderModule    shader_module;
    wgpu::BindGroupLayout bind_group_layout;
    wgpu::PipelineLayout  pipeline_layout;

    // one pipeline per kernel
    wgpu::ComputePipeline pipeline_add;

    ggml_backend_buffer_type_t buft = nullptr;
    bool                  buft_initialized = false;
    wgpu::Buffer          buf_tensor_params;
    int32_t               tensor_params_host[N_TENSOR_PARAMS];
    wgpu::BindGroupEntry  bgEntries[4];
    size_t                totalAlloc;

    ggml_wgpu_context() {
        // instance = wgpu::createInstance(&instanceDesc);
        // descriptor not implemented yet in emscripten
        instance = wgpuCreateInstance(nullptr);
        wgpu::RequestAdapterOptions reqAdaptOpts = wgpu::Default;
        adapter = instance.requestAdapter(reqAdaptOpts);
        wgpu::DeviceDescriptor deviceDesc = wgpu::Default;
        device = adapter.requestDevice(deviceDesc);
        // TODO: load kernels, bind group layout, etc
    }

    // TODO: add destructor
};

// we only support single device for now
using ggml_wgpu_buffer_type_context = ggml_wgpu_context;

struct ggml_wgpu_buffer_context {
    ggml_wgpu_buffer_type_context * ctx;
    wgpu::Buffer               buffer;
    size_t                     size;
    ggml_wgpu_buffer_context(ggml_wgpu_buffer_type_context * _ctx, size_t _size): size(_size), ctx(_ctx) {
        wgpu::BufferDescriptor desc0 = wgpu::Default;
        {
            desc0.label = "storage_buffer";
            desc0.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::CopySrc;
            desc0.size = _size;
            desc0.mappedAtCreation = false;
        };
        buffer = ctx->device.createBuffer(desc0);
        GGML_ASSERT(buffer && "cannot create buffer");
        ctx->totalAlloc += size;
    }
    ~ggml_wgpu_buffer_context() {
        buffer.unmap();
        ctx->totalAlloc -= size;
    }
};


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// backend buffer interface

static const char * ggml_backend_wgpu_buffer_get_name(ggml_backend_buffer_t buffer) {
    UNUSED(buffer);
    return "webgpu_buffer";
}

static void ggml_backend_wgpu_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    delete (ggml_wgpu_buffer_context *)buffer->context;
    UNUSED(buffer);
}

static void * ggml_backend_wgpu_buffer_get_base(ggml_backend_buffer_t buffer) {
    return (void *)buffer->context;
}


static void ggml_backend_wgpu_buffer_init_tensor(ggml_backend_buffer_t buffer,
                                     ggml_tensor *tensor) {
}

static void ggml_backend_wgpu_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                                ggml_tensor *tensor,
                                                const void *data, size_t offset,
                                                size_t size) {
}

static void ggml_backend_wgpu_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                                const ggml_tensor *tensor,
                                                void *data, size_t offset,
                                                size_t size) {
}

static void ggml_backend_wgpu_buffer_clear(ggml_backend_buffer_t buffer,
                                           uint8_t value) {
}

static struct ggml_backend_buffer_i ggml_backend_wgpu_buffer_interface = {
    /* .get_name        = */ ggml_backend_wgpu_buffer_get_name,
    /* .free_buffer     = */ ggml_backend_wgpu_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_wgpu_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_wgpu_buffer_init_tensor,
    /* .set_tensor      = */ ggml_backend_wgpu_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_wgpu_buffer_get_tensor,
    /* .cpy_tensor      = */ NULL,
    /* .clear           = */ ggml_backend_wgpu_buffer_clear,
    /* .reset           = */ NULL,
};

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// backend buffer type interface

static const char * ggml_backend_wgpu_buffer_type_name(ggml_backend_buffer_type_t buft) {
    return "wgpu_buffer_type";
    UNUSED(buft);
}

static ggml_backend_buffer_t ggml_backend_wgpu_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_wgpu_buffer_type_context * buft_ctx = (ggml_wgpu_buffer_type_context *)buft->context;
    ggml_wgpu_buffer_context * ctx = new ggml_wgpu_buffer_context(buft_ctx, size);
    return ggml_backend_buffer_init(buft, ggml_backend_wgpu_buffer_interface, ctx, size);
}

static size_t ggml_backend_wgpu_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return BUF_ALIGN;
    UNUSED(buft);
}

static size_t ggml_backend_wgpu_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    ggml_wgpu_buffer_type_context * buft_ctx = (ggml_wgpu_buffer_type_context *)buft->context;
    return buft_ctx->totalAlloc;
}

GGML_CALL static bool ggml_backend_wgpu_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    return false;
    UNUSED(buft);
}

static ggml_backend_buffer_type_i ggml_backend_wgpu_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_wgpu_buffer_type_name,
    /* .alloc_buffer     = */ ggml_backend_wgpu_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_wgpu_buffer_type_get_alignment,
    /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
    /* .get_alloc_size   = */ ggml_backend_wgpu_buffer_type_get_alloc_size,
    /* .is_host          = */ ggml_backend_wgpu_buffer_type_is_host,
};

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// backend interface

static const char * ggml_backend_wgpu_get_name(ggml_backend_t backend) {
    UNUSED(backend);
    return "webgpu";
}

static void ggml_backend_wgpu_free(ggml_backend_t backend) {
    ggml_wgpu_context * ctx = (ggml_wgpu_context *)backend->context;
    delete ctx;
    delete backend;
}

static ggml_backend_buffer_type_t ggml_backend_wgpu_get_default_buffer_type(ggml_backend_t backend) {
    ggml_wgpu_context * ctx = (ggml_wgpu_context *)backend->context;
    if (ctx->buft == nullptr) {
        ctx->buft = new ggml_backend_buffer_type{
            /* .iface    = */ ggml_backend_wgpu_buffer_type_interface,
            /* .context  = */ ctx,
        };
    }
    return ctx->buft;
}

static void ggml_backend_wgpu_synchronize(ggml_backend_t backend) {
    UNUSED(backend);
}

static ggml_status ggml_backend_wgpu_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    struct ggml_ggml_wgpu_context * wgpu_ctx = (struct ggml_ggml_wgpu_context *)backend->context;
    return GGML_STATUS_SUCCESS;
}

static bool ggml_backend_wgpu_supports_op(ggml_backend_t backend, const struct ggml_tensor * op) {
    return true;
    UNUSED(backend);
    UNUSED(op);
}

static bool ggml_backend_wgpu_supports_buft(ggml_backend_t backend, ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_wgpu_buffer_type_name;
}

static bool ggml_backend_wgpu_offload_op(ggml_backend_t backend, const ggml_tensor * op) {
    return true;
    GGML_UNUSED(backend);
}

static ggml_backend_i ggml_backend_wgpu_interface = {
    /* .get_name                = */ ggml_backend_wgpu_get_name,
    /* .free                    = */ ggml_backend_wgpu_free,
    /* .get_default_buffer_type = */ ggml_backend_wgpu_get_default_buffer_type,
    /* .set_tensor_async        = */ NULL, // ggml_backend_wgpu_set_tensor_async,
    /* .get_tensor_async        = */ NULL, // ggml_backend_wgpu_get_tensor_async,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ ggml_backend_wgpu_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_wgpu_graph_compute,
    /* .supports_op             = */ ggml_backend_wgpu_supports_op,
    /* .supports_buft           = */ ggml_backend_wgpu_supports_buft,
    /* .offload_op              = */ ggml_backend_wgpu_offload_op,
    /* .event_new               = */ NULL,
    /* .event_free              = */ NULL,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .event_synchronize       = */ NULL,
};

static ggml_guid_t ggml_backend_wgpu_guid() {
    static const char * guid_str = "__ggml_webgpu :)";
    return reinterpret_cast<ggml_guid_t>((void *)guid_str);
}

GGML_API ggml_backend_t ggml_backend_wgpu_init(void) {
    ggml_wgpu_context * ctx = new ggml_wgpu_context;
    ggml_backend_t wgpu_backend = new ggml_backend{
        /* .guid      = */ ggml_backend_wgpu_guid(),
        /* .interface = */ ggml_backend_wgpu_interface,
        /* .context   = */ ctx,
    };
    return wgpu_backend;
}

GGML_API bool ggml_backend_is_wgpu(ggml_backend_t backend) {
    return backend->iface.get_name == ggml_backend_wgpu_get_name;
}
