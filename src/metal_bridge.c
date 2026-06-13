// Metal C 桥接：使用 Objective-C Runtime 动态调用 Metal API
// 避免 Objective-C 编译器依赖，纯 C + dlopen + objc_msgSend

#include <dlfcn.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// Objective-C 类型简写
typedef void* id;
typedef void* SEL;

// MTLSize 结构体（与 Objective-C 版本二进制兼容）
typedef struct { uintptr_t width; uintptr_t height; uintptr_t depth; } MTLSize;

// Objective-C Runtime 函数（macOS 上默认链接 libobjc）
extern id objc_getClass(const char* name);
extern SEL sel_registerName(const char* name);
extern id objc_msgSend(id self, SEL op, ...);

// Metal C 入口
typedef id (*MTLCreateSystemDefaultDevice_fn)(void);
static MTLCreateSystemDefaultDevice_fn mtlCreateDevice = NULL;

// 缓存常用 selector
static SEL selAlloc = NULL;
static SEL selInit = NULL;
static SEL selNewCommandQueue = NULL;
static SEL selNewBufferWithBytesLengthOptions = NULL;
static SEL selContents = NULL;
static SEL selNewLibraryWithSourceOptionsError = NULL;
static SEL selNewFunctionWithName = NULL;
static SEL selNewComputePipelineStateWithFunctionError = NULL;
static SEL selCommandBuffer = NULL;
static SEL selComputeCommandEncoder = NULL;
static SEL selSetComputePipelineState = NULL;
static SEL selSetBufferOffsetAtIndex = NULL;
static SEL selSetBytesLengthAtIndex = NULL;
static SEL selDispatchThreadgroupsThreadsPerThreadgroup = NULL;
static SEL selEndEncoding = NULL;
static SEL selCommit = NULL;
static SEL selWaitUntilCompleted = NULL;
static SEL selRelease = NULL;
static SEL selLength = NULL;

// MTLResourceOptions
typedef enum { MTLResourceStorageModeShared = 0 } MTLResourceOptions;

static char last_error[512] = {0};

const char* metal_get_last_error(void) { return last_error; }

int metal_init(void) {
    void* handle = dlopen("/System/Library/Frameworks/Metal.framework/Metal", RTLD_LAZY);
    if (!handle) {
        snprintf(last_error, sizeof(last_error), "dlopen Metal failed: %s", dlerror());
        return -1;
    }
    mtlCreateDevice = (MTLCreateSystemDefaultDevice_fn)dlsym(handle, "MTLCreateSystemDefaultDevice");
    if (!mtlCreateDevice) {
        snprintf(last_error, sizeof(last_error), "dlsym MTLCreateSystemDefaultDevice failed");
        return -2;
    }

    selAlloc = sel_registerName("alloc");
    selInit = sel_registerName("init");
    selNewCommandQueue = sel_registerName("newCommandQueue");
    selNewBufferWithBytesLengthOptions = sel_registerName("newBufferWithBytes:length:options:");
    selContents = sel_registerName("contents");
    selNewLibraryWithSourceOptionsError = sel_registerName("newLibraryWithSource:options:error:");
    selNewFunctionWithName = sel_registerName("newFunctionWithName:");
    selNewComputePipelineStateWithFunctionError = sel_registerName("newComputePipelineStateWithFunction:error:");
    selCommandBuffer = sel_registerName("commandBuffer");
    selComputeCommandEncoder = sel_registerName("computeCommandEncoder");
    selSetComputePipelineState = sel_registerName("setComputePipelineState:");
    selSetBufferOffsetAtIndex = sel_registerName("setBuffer:offset:atIndex:");
    selSetBytesLengthAtIndex = sel_registerName("setBytes:length:atIndex:");
    selDispatchThreadgroupsThreadsPerThreadgroup = sel_registerName("dispatchThreadgroups:threadsPerThreadgroup:");
    selEndEncoding = sel_registerName("endEncoding");
    selCommit = sel_registerName("commit");
    selWaitUntilCompleted = sel_registerName("waitUntilCompleted");
    selRelease = sel_registerName("release");
    selLength = sel_registerName("length");

    return 0;
}

id metal_create_device(void) {
    if (!mtlCreateDevice) return NULL;
    return mtlCreateDevice();
}

void metal_release(id obj) {
    if (obj) objc_msgSend(obj, selRelease);
}

id metal_new_command_queue(id device) {
    return objc_msgSend(device, selNewCommandQueue);
}

id metal_new_buffer(id device, const void* bytes, size_t length) {
    return objc_msgSend(device, selNewBufferWithBytesLengthOptions,
                        bytes, (uintptr_t)length, (uintptr_t)MTLResourceStorageModeShared);
}

void* metal_buffer_contents(id buffer) {
    return objc_msgSend(buffer, selContents);
}

size_t metal_buffer_length(id buffer) {
    return (size_t)(uintptr_t)objc_msgSend(buffer, selLength);
}

id metal_create_library(id device, const char* source) {
    id options = objc_msgSend(objc_msgSend(objc_getClass("MTLCompileOptions"), selAlloc), selInit);
    id error = NULL;
    id library = objc_msgSend(device, selNewLibraryWithSourceOptionsError, source, options, &error);
    metal_release(options);
    if (error) {
        snprintf(last_error, sizeof(last_error), "Metal compile error occurred");
        return NULL;
    }
    return library;
}

id metal_create_function(id library, const char* name) {
    return objc_msgSend(library, selNewFunctionWithName, name);
}

id metal_create_pipeline(id device, id function) {
    id error = NULL;
    id pipeline = objc_msgSend(device, selNewComputePipelineStateWithFunctionError, function, &error);
    if (error) {
        snprintf(last_error, sizeof(last_error), "Metal pipeline creation failed");
        return NULL;
    }
    return pipeline;
}

id metal_command_buffer(id queue) {
    return objc_msgSend(queue, selCommandBuffer);
}

id metal_compute_encoder(id cmdbuf) {
    return objc_msgSend(cmdbuf, selComputeCommandEncoder);
}

void metal_set_pipeline(id encoder, id pipeline) {
    objc_msgSend(encoder, selSetComputePipelineState, pipeline);
}

void metal_set_buffer(id encoder, id buffer, size_t offset, size_t index) {
    objc_msgSend(encoder, selSetBufferOffsetAtIndex, buffer, (uintptr_t)offset, (uintptr_t)index);
}

void metal_set_bytes(id encoder, const void* bytes, size_t length, size_t index) {
    objc_msgSend(encoder, selSetBytesLengthAtIndex, bytes, (uintptr_t)length, (uintptr_t)index);
}

void metal_dispatch(id encoder, size_t gx, size_t gy, size_t gz, size_t tx, size_t ty, size_t tz) {
    MTLSize grid = {gx, gy, gz};
    MTLSize threads = {tx, ty, tz};
    objc_msgSend(encoder, selDispatchThreadgroupsThreadsPerThreadgroup, grid, threads);
}

void metal_end_encoding(id encoder) {
    objc_msgSend(encoder, selEndEncoding);
}

void metal_commit(id cmdbuf) {
    objc_msgSend(cmdbuf, selCommit);
}

void metal_wait_completed(id cmdbuf) {
    objc_msgSend(cmdbuf, selWaitUntilCompleted);
}
