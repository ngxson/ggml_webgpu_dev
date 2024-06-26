#include <GLFW/glfw3.h>
#include <iostream>
#include <vector>
#include <emscripten/emscripten.h>

#define WEBGPU_CPP_IMPLEMENTATION
#include "ggml-wgpu/webgpu.hpp"

#define BUF_ALIGN 256
#define N_TENSOR_PARAMS 4

const int rows_A = 4, cols_A = 2;
const int rows_B = 4, cols_B = 2;
float demo_mat_A[rows_A * cols_A] = {
    2, 8,
    -5, 1,
    4, 2,
    8, -6
};
float demo_mat_B[rows_B * cols_B] = {
    1, 6,
    3, -7,
    9, 0,
    -1, 5
};
float demo_mat_RESULT[rows_A * cols_A] = {
    3, 14,
    -2, -6,
    13, 2,
    7, -1
};
const int demo_mat_nbytes = rows_A * cols_A * sizeof(float);
const int demo_mat_nelements = rows_A * cols_A;

#define GGML_ASSERT(x) \
    do { \
        if (!(x)) { \
            fflush(stdout); \
            fprintf(stderr, "GGML_ASSERT: %s:%d: %s\n", __FILE__, __LINE__, #x); \
            abort(); \
        } \
    } while (0)

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

    wgpu::Buffer          buffer;
    wgpu::Buffer          buf_tensor_params;
    int32_t               tensor_params_host[N_TENSOR_PARAMS];
    wgpu::BindGroupEntry  bgEntries[4];
};

ggml_wgpu_context * ctx;

const char SHADER_CODE[] = R"(
struct TensorParams {
    offs_src0 : u32,
    offs_src1 : u32,
    offs_dest : u32,
}

@group(0) @binding(0)
var<storage,read_write> src0: array<f32>;

@group(0) @binding(1)
var<storage,read_write> src1: array<f32>;

@group(0) @binding(2)
var<storage,read_write> dest: array<f32>;

@group(0) @binding(3)
var<uniform> tensor_params: TensorParams;

@compute
@workgroup_size(1)
fn kernel_add(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = src0[global_id.x + tensor_params.offs_src0/4u];
    let y = src1[global_id.x + tensor_params.offs_src1/4u];
            dest[global_id.x + tensor_params.offs_dest/4u] = x + y;
}
)";

void Start() {
    ctx->device.getLimits(&ctx->limits);
    ctx->queue = ctx->device.getQueue();

    // init bind group layout
    wgpu::BindGroupLayoutEntry bgLayoutEntries[4];
    {
        bgLayoutEntries[0].setDefault();
        bgLayoutEntries[0].binding = 0;
        bgLayoutEntries[0].visibility = wgpu::ShaderStage::Compute;
        bgLayoutEntries[0].buffer.type = wgpu::BufferBindingType::Storage;
        bgLayoutEntries[0].buffer.hasDynamicOffset = false;
        bgLayoutEntries[0].buffer.minBindingSize = 0;

        bgLayoutEntries[1].setDefault();
        bgLayoutEntries[1].binding = 1;
        bgLayoutEntries[1].visibility = wgpu::ShaderStage::Compute;
        bgLayoutEntries[1].buffer.type = wgpu::BufferBindingType::Storage;
        bgLayoutEntries[1].buffer.hasDynamicOffset = false;
        bgLayoutEntries[1].buffer.minBindingSize = 0;

        bgLayoutEntries[2].setDefault();
        bgLayoutEntries[2].binding = 2;
        bgLayoutEntries[2].visibility = wgpu::ShaderStage::Compute;
        bgLayoutEntries[2].buffer.type = wgpu::BufferBindingType::Storage;
        bgLayoutEntries[2].buffer.hasDynamicOffset = false;
        bgLayoutEntries[2].buffer.minBindingSize = 0;

        bgLayoutEntries[3].setDefault();
        bgLayoutEntries[3].binding = 3;
        bgLayoutEntries[3].visibility = wgpu::ShaderStage::Compute;
        bgLayoutEntries[3].buffer.type = wgpu::BufferBindingType::Uniform;
        bgLayoutEntries[3].buffer.hasDynamicOffset = false;
        bgLayoutEntries[3].buffer.minBindingSize = N_TENSOR_PARAMS*sizeof(int32_t);
    }
    wgpu::BindGroupLayoutDescriptor bglDesc = wgpu::Default;
    {
        bglDesc.label = "ggml-wgpu-bind-group-layout";
        bglDesc.entryCount = 4;
        bglDesc.entries = bgLayoutEntries;
    };
    ctx->bind_group_layout = ctx->device.createBindGroupLayout(bglDesc);
    GGML_ASSERT(ctx->bind_group_layout);

    // load shader
    {
        wgpu::ShaderModuleWGSLDescriptor wgslDesc = wgpu::Default;
        wgslDesc.code = SHADER_CODE;
        wgpu::ShaderModuleDescriptor shaderModuleDescriptor;
        shaderModuleDescriptor.nextInChain = (const WGPUChainedStruct *) &wgslDesc;
        ctx->shader_module = ctx->device.createShaderModule(shaderModuleDescriptor);
        GGML_ASSERT(ctx->shader_module);
    }

    // create pipeline from shader
    {
        wgpu::PipelineLayoutDescriptor plDesc = wgpu::Default;
        {
            plDesc.label = "ggml-wgpu-pipeline-layout";
            plDesc.bindGroupLayoutCount = 1;
            plDesc.bindGroupLayouts = (const WGPUBindGroupLayout *) &ctx->bind_group_layout;
        };
        ctx->pipeline_layout = ctx->device.createPipelineLayout(plDesc);
        GGML_ASSERT(ctx->pipeline_layout);

        wgpu::ComputePipelineDescriptor cpDesc = wgpu::Default;
        {
            cpDesc.label = "pipeline_add";
            cpDesc.layout = ctx->pipeline_layout;
            wgpu::ProgrammableStageDescriptor psDesc = wgpu::Default;
            {
                psDesc.module = ctx->shader_module;
                psDesc.entryPoint = "kernel_add";
            }
            cpDesc.compute = psDesc;
        };
        ctx->pipeline_add = ctx->device.createComputePipeline(cpDesc);
        GGML_ASSERT(ctx->pipeline_add);
    }

}

void Alloc() {
    printf("%s\n", __func__);
    // alloc buffers
    {
        wgpu::BufferDescriptor desc0 = wgpu::Default;
        {
            desc0.label = "storage_buffer";
            desc0.usage = wgpu::BufferUsage::Storage
                | wgpu::BufferUsage::CopyDst
                | wgpu::BufferUsage::CopySrc;
            desc0.size = BUF_ALIGN*16u;
            desc0.mappedAtCreation = false;
        };
        ctx->buffer = ctx->device.createBuffer(desc0);
        GGML_ASSERT(ctx->buffer);

        wgpu::BufferDescriptor desc1 = wgpu::Default;
        {
            desc1.label = "params_buffer";
            desc1.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
            desc1.size = BUF_ALIGN;
            desc1.mappedAtCreation = false;
        };
        ctx->buf_tensor_params = ctx->device.createBuffer(desc1);
        GGML_ASSERT(ctx->buffer);
    }

}

void Compute() {
    printf("%s\n", __func__);
    // set matrix data and params
    size_t offs_A = 0;
    ctx->queue.writeBuffer(ctx->buffer, offs_A, demo_mat_A, demo_mat_nbytes);
    size_t offs_B = 1u*BUF_ALIGN;
    ctx->queue.writeBuffer(ctx->buffer, offs_B, demo_mat_B, demo_mat_nbytes);
    size_t offs_dest = 2u*BUF_ALIGN;
    //size_t offs_params = 3u*BUF_ALIGN;
    size_t offs_params = 0; // buf_tensor_params
    ctx->tensor_params_host[0] = 0;
    ctx->tensor_params_host[1] = 0;
    ctx->tensor_params_host[2] = 0;
    ctx->tensor_params_host[3] = 0; // unused for now
    ctx->queue.writeBuffer(ctx->buf_tensor_params, offs_params, ctx->tensor_params_host, N_TENSOR_PARAMS*sizeof(int32_t));

    // set bind group entry
    auto & bgEntries = ctx->bgEntries;
    {
        bgEntries[0].binding = 0;
        bgEntries[0].buffer = ctx->buffer;
        bgEntries[0].offset = 0;
        bgEntries[0].size = BUF_ALIGN;

        bgEntries[1].binding = 1;
        bgEntries[1].buffer = ctx->buffer;
        bgEntries[1].offset = offs_B;
        bgEntries[1].size = BUF_ALIGN;

        bgEntries[2].binding = 2;
        bgEntries[2].buffer = ctx->buffer;
        bgEntries[2].offset = offs_dest;
        bgEntries[2].size = BUF_ALIGN;

        bgEntries[3].binding = 3;
        bgEntries[3].buffer = ctx->buf_tensor_params;
        bgEntries[3].offset = offs_params;
        bgEntries[3].size = BUF_ALIGN;
    }
    wgpu::BindGroupDescriptor bgDesc = wgpu::Default;
    {
        bgDesc.label = "bind_group";
        bgDesc.layout = ctx->bind_group_layout;
        bgDesc.entryCount = 4;
        bgDesc.entries = ctx->bgEntries;
    };
    auto bindGroup = ctx->device.createBindGroup(bgDesc);
    GGML_ASSERT(bindGroup);

    // compute
    wgpu::CommandEncoderDescriptor desc0 = wgpu::Default;
    desc0.label = "ggml_command_encoder";
    auto commandEncoder = ctx->device.createCommandEncoder(desc0);
    GGML_ASSERT(commandEncoder);
    auto computePassEncoder = commandEncoder.beginComputePass();
    GGML_ASSERT(computePassEncoder);
    computePassEncoder.setPipeline(ctx->pipeline_add);
    computePassEncoder.setBindGroup(0, bindGroup, 0, NULL);
    computePassEncoder.dispatchWorkgroups(demo_mat_nelements, 1, 1);
    computePassEncoder.end();

    auto cmdBuffer = commandEncoder.finish();
    GGML_ASSERT(cmdBuffer);
    ctx->queue.submit(1, &cmdBuffer);
}

void GetResult(void * output) {
    printf("%s\n", __func__);
    // create tmp buffer
    wgpu::BufferDescriptor desc0 = wgpu::Default;
    {
        desc0.label = "map_read_buffer";
        desc0.usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst;
        desc0.size = BUF_ALIGN;
        desc0.mappedAtCreation = false;
    };
    auto tmpbuf = ctx->device.createBuffer(desc0);

    // command encoder
    wgpu::CommandEncoderDescriptor desc1 = wgpu::Default;
    desc1.label = "ggml_command_encoder_get_tensor";
    auto commandEncoder = ctx->device.createCommandEncoder(desc1);
    size_t offs_dest = 2u*BUF_ALIGN;
    commandEncoder.copyBufferToBuffer(ctx->buffer, offs_dest, tmpbuf, 0, demo_mat_nbytes);

    // run cmd
    auto cmdBuffer = commandEncoder.finish();
    ctx->queue.submit(1, &cmdBuffer);
    bool ready = false;
    auto ret = tmpbuf.mapAsync(wgpu::MapMode::Read, 0, demo_mat_nbytes, [&ready](WGPUBufferMapAsyncStatus status) {
        printf("buffer_map status=%#.8x\n", status);
        if (status == WGPUBufferMapAsyncStatus_Success) {
            ready = true;
        }
    });
    while (!ready) {
        // https://eliemichel.github.io/LearnWebGPU/basic-3d-rendering/input-geometry/playing-with-buffers.html
        emscripten_sleep(1);
    }

    // get output buf
    const void * buf = tmpbuf.getConstMappedRange(0, demo_mat_nbytes);
    GGML_ASSERT(buf);
    memcpy(output, buf, demo_mat_nbytes);

    // free
    tmpbuf.unmap();
}

int main() {
    printf("Start\n");
    ctx = new ggml_wgpu_context;
    ctx->instance = wgpuCreateInstance(nullptr); // descriptor not implemented yet in emscripten
    //ctx->instance = wgpu::createInstance(&instanceDesc);

    wgpu::RequestAdapterOptions opts = wgpu::Default;
    auto adapter = ctx->instance.requestAdapter(opts);

    wgpu::DeviceDescriptor deviceDesc = wgpu::Default;
    ctx->device = adapter.requestDevice(deviceDesc);

    Start();
    Alloc();
    Compute();

    std::vector<float> result(demo_mat_nelements);
    GetResult(result.data());

    printf("Result:\n");
    for (int ir = 0; ir < rows_A; ir++) {
        for (int ic = 0; ic < cols_A; ic++) {
            printf("%f, ", result[ir*cols_A + ic]);
        }
        printf("\n");
    }
}
