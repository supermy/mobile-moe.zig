# Mobile MoE GPU 后端设计文档

## 目标

为 `mobile-moe.zig` 推理引擎提供跨平台 GPU 加速能力，目标平台覆盖：
- **iOS/macOS**: Apple Metal (Apple Silicon / A-series)
- **Android**: OpenCL / Vulkan / Qualcomm Adreno SDK
- **未来扩展**: WebGPU (浏览器端推理)

## 架构原则

1. **算子抽象层（Kernel Abstraction Layer）**: 定义与硬件无关的算子接口，后端只需实现这些接口
2. **CPU 回退**: 所有 GPU 内核必须有对应的 CPU 参考实现（当前已有），用于精度验证和低端设备
3. **零拷贝内存**: 尽可能使用统一内存（Unified Memory）或内存映射，减少 CPU↔GPU 拷贝
4. **延迟加载**: GPU 内核按需编译/加载，避免启动时初始化所有后端

## 算子抽象层

```zig
// src/gpu/gpu_backend.zig
pub const ComputeBackend = enum {
    cpu,
    metal,
    opencl,
    vulkan,
};

pub const KernelOp = enum {
    rms_norm,
    mat_vec_mul,
    mat_mul,
    silu,
    softmax,
    rope_apply,
    attention,      // FlashAttention / FlashDecoding
    moe_route,      // Top-K gating
    moe_expert_ffn, // 单个 expert 的 FFN
};

/// 后端接口：所有 GPU 后端必须实现
pub const Backend = struct {
    vtable: *const VTable,

    pub const VTable = struct {
        /// 初始化后端（设备选择、上下文创建）
        init: *const fn (ctx: *anyopaque, device_id: u32) anyerror!void,
        /// 释放后端资源
        deinit: *const fn (ctx: *anyopaque) void,
        /// 分配设备内存
        allocDevice: *const fn (ctx: *anyopaque, size: usize) anyerror![*]u8,
        /// 释放设备内存
        freeDevice: *const fn (ctx: *anyopaque, ptr: [*]u8) void,
        /// 上传数据到设备（异步）
        upload: *const fn (ctx: *anyopaque, dst: [*]u8, src: []const u8) anyerror!void,
        /// 下载数据到主机（异步）
        download: *const fn (ctx: *anyopaque, dst: []u8, src: [*]const u8) anyerror!void,
        /// 执行内核
        dispatch: *const fn (ctx: *anyopaque, op: KernelOp, args: *const anyopaque) anyerror!void,
        /// 同步等待所有操作完成
        synchronize: *const fn (ctx: *anyopaque) void,
    };
};
```

## Metal 后端（iOS/macOS 优先）

### 技术选型理由
- Apple 平台唯一的低开销 GPU API
- A17 Pro / M4 支持硬件加速的矩阵乘法（Apple AMX）
- Unified Memory：CPU 和 GPU 共享同一物理内存，天然零拷贝

### 内核实现策略

1. **内核源码管理**
   - `.metal` 文件放在 `kernels/metal/`
   - 编译时预编译为 `.metallib`（使用 `xcrun -sdk iphoneos metal`）
   - 运行时加载预编译库，避免着色器编译延迟

2. **关键内核**
   ```metal
   // kernels/metal/attention.metal
   kernel void flash_attention(
       device const float* q [[buffer(0)]],
       device const float* k [[buffer(1)]],
       device const float* v [[buffer(2)]],
       device float* out [[buffer(3)]],
       constant uint& seq_len [[buffer(4)]],
       constant uint& head_dim [[buffer(5)]],
       uint3 tid [[threadgroup_position_in_grid]],
       uint3 lid [[thread_position_in_threadgroup]]
   ) {
       // Tile-based FlashAttention
       // Br = 64, Bc = 64 tile size for A17/M4 cache
   }
   ```

3. **内存管理**
   - `MTLBuffer` 使用 `MTLResourceOptions.storageModeShared`（Unified Memory）
   - 权重数据：GGUF mmap 后直接使用 `MTLBuffer` 包装（无需拷贝）
   - KV Cache：`MTLBuffer` 分配，Metal 内核直接读写

4. **性能优化**
   - 使用 `simdgroup_matrix` (SIMD-scoped matrix) 做 8×8 或 16×16 tile GEMM
   - Attention 使用 threadgroup memory (SRAM) 缓存 K/V tile
   - MoE 专家并行：使用 `dispatchThreadgroups` 并发调度多个 expert 内核

## OpenCL 后端（Android 通用）

### 技术选型理由
- Android 上最广泛支持的 GPU 计算 API（Qualcomm Adreno、Mali、PowerVR）
- 不需要 NDK 中的私有 API

### 内核实现策略

1. **内核源码**
   - `.cl` 文件放在 `kernels/opencl/`
   - 运行时编译（OpenCL JIT），缓存二进制到磁盘

2. **关键优化**
   - 使用 `local memory` (__local) 做 tile cache
   - Adreno GPU：优先使用 image2d_t（纹理缓存）存储权重
   - Mali GPU：使用 `arm_thread_group_hint` 扩展优化 warp 大小

3. **内存管理**
   - `CL_MEM_ALLOC_HOST_PTR` + `clEnqueueMapBuffer` 实现零拷贝（如果驱动支持）
   - 否则使用 `CL_MEM_COPY_HOST_PTR` 上传权重（一次性）

## Vulkan 后端（未来/跨平台）

### 技术选型理由
- Android 上逐渐替代 OpenCL（Vulkan Compute Shader）
- 更好的工具链和调试支持（RenderDoc）
- 可以复用 SPIR-V 中间表示

### 实现策略
- 使用 `vkCmdDispatch` 调度 compute shader
- SPIR-V 从 GLSL/HLSL/MLIR 编译
- 权重存储为 `VkBuffer`（`VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT`）

## 与 InferenceEngine 的集成

```zig
// src/inference.zig
pub const InferenceEngine = struct {
    // ... 现有字段 ...
    backend: ?Backend,
    device_weights: ?DeviceWeightCache, // GPU 上的权重缓存

    pub fn init(...) !Self {
        // ...
        // 尝试初始化 Metal（macOS/iOS）
        self.backend = try metal_backend.init(allocator);
        // 如果失败，回退到 CPU
        if (self.backend == null) {
            std.log.info("GPU backend not available, using CPU", .{});
        }
    }

    fn mlaAttention(self: *Self, layer: u32, pos: u32) !void {
        if (self.backend) |*be| {
            // 上传 Q/KV 到 GPU（如果尚未上传）
            // 调用 FlashAttention 内核
            try be.dispatch(.attention, &.{ .q = q_buf, .k = k_buf, .v = v_buf, .out = out_buf, .seq_len = pos + 1 });
            // 下载结果（或直接在 GPU 上继续 FFN）
        } else {
            // CPU 参考路径
            try self.mlaAttentionCpu(layer, pos);
        }
    }
};
```

## 精度验证策略

1. **单元测试**: 每个 GPU 内核的输出与 CPU 参考实现对比，RMSE < 1e-4
2. **集成测试**: 完整 forward 的 logits 对比，Top-5 token 重合率 > 99%
3. **量化感知**: GPU 内核直接消费 Q4_K/Q8_0 量化权重，避免 f32 反量化

## 开发路线图

| 阶段 | 时间 | 目标 |
|------|------|------|
| Phase 1 | 2 周 | Metal 后端框架 + matVecMul + RMSNorm |
| Phase 2 | 4 周 | Metal FlashAttention + MoE FFN |
| Phase 3 | 6 周 | OpenCL 后端（Android） |
| Phase 4 | 8 周 | 量化直接计算（Q4_K/Q8_0 GPU GEMM） |
| Phase 5 | 12 周 | Vulkan 后端 + WebGPU 原型 |

## 参考资源

- [Metal Performance Shaders](https://developer.apple.com/documentation/metalperformanceshaders)
- [OpenCL 2.0 Reference](https://www.khronos.org/registry/OpenCL/specs/opencl-2.0.pdf)
- [Vulkan Compute Shader Tutorial](https://vulkan-tutorial.com/)
- llama.cpp `ggml-metal.m` / `ggml-opencl.cpp`
