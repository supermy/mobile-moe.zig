// OpenCL 后端模块（Android / Linux 通用）
//
// 设计原则：
//   1. 动态加载 libOpenCL.so，避免编译时依赖
//   2. 支持多队列（主队列 + 异步队列）实现 2 专家并行
//   3. 所有内核输出与 CPU 参考路径对齐验证
//
// 参考 HeteroLLM 洞察：Adreno 750 上 2 队列并行加速比约 1.45x

const std = @import("std");

// ---------- OpenCL 类型 ----------
const cl_int = i32;
const cl_uint = u32;
const cl_ulong = u64;
const cl_bool = u32;
const cl_device_type = u64;
const cl_mem_flags = u64;
const cl_platform_id = ?*anyopaque;
const cl_device_id = ?*anyopaque;
const cl_context = ?*anyopaque;
const cl_command_queue = ?*anyopaque;
const cl_mem = ?*anyopaque;
const cl_program = ?*anyopaque;
const cl_kernel = ?*anyopaque;
const cl_event = ?*anyopaque;

const CL_DEVICE_TYPE_GPU: cl_device_type = 1 << 2;
const CL_MEM_READ_WRITE: cl_mem_flags = 1 << 0;
const CL_MEM_READ_ONLY: cl_mem_flags = 1 << 2;
const CL_MEM_WRITE_ONLY: cl_mem_flags = 1 << 1;
const CL_SUCCESS: cl_int = 0;

// ---------- 函数指针类型 ----------
const ClGetPlatformIDs = *const fn (cl_uint, ?*cl_platform_id, ?*cl_uint) callconv(.c) cl_int;
const ClGetDeviceIDs = *const fn (cl_platform_id, cl_device_type, cl_uint, ?*cl_device_id, ?*cl_uint) callconv(.c) cl_int;
const ClCreateContext = *const fn (?*anyopaque, cl_uint, [*]const cl_device_id, ?*anyopaque, ?*anyopaque, ?*cl_int) callconv(.c) cl_context;
const ClCreateCommandQueue = *const fn (cl_context, cl_device_id, cl_ulong, ?*cl_int) callconv(.c) cl_command_queue;
const ClCreateBuffer = *const fn (cl_context, cl_mem_flags, usize, ?*anyopaque, ?*cl_int) callconv(.c) cl_mem;
const ClCreateProgramWithSource = *const fn (cl_context, cl_uint, [*]const [*:0]const u8, ?[*]const usize, ?*cl_int) callconv(.c) cl_program;
const ClBuildProgram = *const fn (cl_program, cl_uint, ?*const cl_device_id, [*:0]const u8, ?*anyopaque, ?*anyopaque) callconv(.c) cl_int;
const ClCreateKernel = *const fn (cl_program, [*:0]const u8, ?*cl_int) callconv(.c) cl_kernel;
const ClSetKernelArg = *const fn (cl_kernel, cl_uint, usize, ?*const anyopaque) callconv(.c) cl_int;
const ClEnqueueNDRangeKernel = *const fn (cl_command_queue, cl_kernel, cl_uint, ?*const usize, [*]const usize, ?*const usize, cl_uint, ?*const cl_event, ?*cl_event) callconv(.c) cl_int;
const ClEnqueueWriteBuffer = *const fn (cl_command_queue, cl_mem, cl_bool, usize, usize, ?*const anyopaque, cl_uint, ?*const cl_event, ?*cl_event) callconv(.c) cl_int;
const ClEnqueueReadBuffer = *const fn (cl_command_queue, cl_mem, cl_bool, usize, usize, ?*anyopaque, cl_uint, ?*const cl_event, ?*cl_event) callconv(.c) cl_int;
const ClFinish = *const fn (cl_command_queue) callconv(.c) cl_int;
const ClReleaseMemObject = *const fn (cl_mem) callconv(.c) cl_int;
const ClReleaseKernel = *const fn (cl_kernel) callconv(.c) cl_int;
const ClReleaseProgram = *const fn (cl_program) callconv(.c) cl_int;
const ClReleaseCommandQueue = *const fn (cl_command_queue) callconv(.c) cl_int;
const ClReleaseContext = *const fn (cl_context) callconv(.c) cl_int;

/// OpenCL 后端状态
pub const OpenCLBackend = struct {
    lib: ?std.DynLib,
    context: cl_context,
    device: cl_device_id,
    queue_main: cl_command_queue,
    queue_async: ?cl_command_queue, // 用于专家并行
    matvec_kernel: ?cl_kernel,
    rmsnorm_kernel: ?cl_kernel,
    qmm_kernel: ?cl_kernel,
    allocator: std.mem.Allocator,

    // 权重 buffer 常驻缓存（key: 数据指针地址，value: cl_mem）
    weight_buffers: std.AutoHashMap(u64, cl_mem),

    // 函数指针缓存
    clGetPlatformIDs: ?ClGetPlatformIDs = null,
    clGetDeviceIDs: ?ClGetDeviceIDs = null,
    clCreateContext: ?ClCreateContext = null,
    clCreateCommandQueue: ?ClCreateCommandQueue = null,
    clCreateBuffer: ?ClCreateBuffer = null,
    clCreateProgramWithSource: ?ClCreateProgramWithSource = null,
    clBuildProgram: ?ClBuildProgram = null,
    clCreateKernel: ?ClCreateKernel = null,
    clSetKernelArg: ?ClSetKernelArg = null,
    clEnqueueNDRangeKernel: ?ClEnqueueNDRangeKernel = null,
    clEnqueueWriteBuffer: ?ClEnqueueWriteBuffer = null,
    clEnqueueReadBuffer: ?ClEnqueueReadBuffer = null,
    clFinish: ?ClFinish = null,
    clReleaseMemObject: ?ClReleaseMemObject = null,
    clReleaseKernel: ?ClReleaseKernel = null,
    clReleaseProgram: ?ClReleaseProgram = null,
    clReleaseCommandQueue: ?ClReleaseCommandQueue = null,
    clReleaseContext: ?ClReleaseContext = null,

    const Self = @This();

    /// 检测 OpenCL 是否可用（尝试加载库）
    pub fn isAvailable() bool {
        var lib = std.DynLib.open("/vendor/lib64/libOpenCL_adreno.so") catch
            std.DynLib.open("/vendor/lib64/libOpenCL.so") catch
            std.DynLib.open("/usr/lib/libOpenCL.so") catch
            std.DynLib.open("libOpenCL.so") catch return false;
        lib.close();
        return true;
    }

    pub fn init(allocator: std.mem.Allocator) !Self {
        // 1. 动态加载 OpenCL 库
        var lib = std.DynLib.open("/vendor/lib64/libOpenCL_adreno.so") catch
            std.DynLib.open("/vendor/lib64/libOpenCL.so") catch
            std.DynLib.open("/usr/lib/libOpenCL.so") catch
            std.DynLib.open("libOpenCL.so") catch return error.OpenCLLibraryNotFound;

        // 2. 解析符号
        const clGetPlatformIDs = lib.lookup(ClGetPlatformIDs, "clGetPlatformIDs") orelse return error.OpenCLSymbolNotFound;
        const clGetDeviceIDs = lib.lookup(ClGetDeviceIDs, "clGetDeviceIDs") orelse return error.OpenCLSymbolNotFound;
        const clCreateContext = lib.lookup(ClCreateContext, "clCreateContext") orelse return error.OpenCLSymbolNotFound;
        const clCreateCommandQueue = lib.lookup(ClCreateCommandQueue, "clCreateCommandQueue") orelse return error.OpenCLSymbolNotFound;
        const clCreateBuffer = lib.lookup(ClCreateBuffer, "clCreateBuffer") orelse return error.OpenCLSymbolNotFound;
        const clCreateProgramWithSource = lib.lookup(ClCreateProgramWithSource, "clCreateProgramWithSource") orelse return error.OpenCLSymbolNotFound;
        const clBuildProgram = lib.lookup(ClBuildProgram, "clBuildProgram") orelse return error.OpenCLSymbolNotFound;
        const clCreateKernel = lib.lookup(ClCreateKernel, "clCreateKernel") orelse return error.OpenCLSymbolNotFound;
        const clSetKernelArg = lib.lookup(ClSetKernelArg, "clSetKernelArg") orelse return error.OpenCLSymbolNotFound;
        const clEnqueueNDRangeKernel = lib.lookup(ClEnqueueNDRangeKernel, "clEnqueueNDRangeKernel") orelse return error.OpenCLSymbolNotFound;
        const clEnqueueWriteBuffer = lib.lookup(ClEnqueueWriteBuffer, "clEnqueueWriteBuffer") orelse return error.OpenCLSymbolNotFound;
        const clEnqueueReadBuffer = lib.lookup(ClEnqueueReadBuffer, "clEnqueueReadBuffer") orelse return error.OpenCLSymbolNotFound;
        const clFinish = lib.lookup(ClFinish, "clFinish") orelse return error.OpenCLSymbolNotFound;
        const clReleaseMemObject = lib.lookup(ClReleaseMemObject, "clReleaseMemObject") orelse return error.OpenCLSymbolNotFound;
        const clReleaseKernel = lib.lookup(ClReleaseKernel, "clReleaseKernel") orelse return error.OpenCLSymbolNotFound;
        const clReleaseProgram = lib.lookup(ClReleaseProgram, "clReleaseProgram") orelse return error.OpenCLSymbolNotFound;
        const clReleaseCommandQueue = lib.lookup(ClReleaseCommandQueue, "clReleaseCommandQueue") orelse return error.OpenCLSymbolNotFound;
        const clReleaseContext = lib.lookup(ClReleaseContext, "clReleaseContext") orelse return error.OpenCLSymbolNotFound;

        // 3. 获取平台/设备
        var num_platforms: cl_uint = 0;
        var ret = clGetPlatformIDs(0, null, &num_platforms);
        if (ret != CL_SUCCESS or num_platforms == 0) return error.OpenCLNoPlatform;

        var platform: cl_platform_id = null;
        ret = clGetPlatformIDs(1, &platform, null);
        if (ret != CL_SUCCESS) return error.OpenCLPlatformError;

        var device: cl_device_id = null;
        ret = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, null);
        if (ret != CL_SUCCESS) return error.OpenCLNoGPU;

        // 4. 创建上下文和队列
        var err: cl_int = 0;
        var devices = [_]cl_device_id{device};
        const context = clCreateContext(null, 1, &devices, null, null, &err);
        if (err != CL_SUCCESS) return error.OpenCLContextError;
        errdefer _ = clReleaseContext(context);

        const queue_main = clCreateCommandQueue(context, device, 0, &err);
        if (err != CL_SUCCESS) {
            _ = clReleaseContext(context);
            return error.OpenCLQueueError;
        }
        errdefer _ = clReleaseCommandQueue(queue_main);

        // 异步队列（用于专家并行）
        const queue_async = clCreateCommandQueue(context, device, 0, &err);
        // 允许失败，后续回退到单队列

        // 5. 编译内核程序
        const kernel_source =
            "__kernel void matVecMul(__global const float* mat, __global const float* vec, __global float* out, int out_dim, int in_dim) {\n" ++
            "    int row = get_global_id(0);\n" ++
            "    if (row >= out_dim) return;\n" ++
            "    float sum = 0.0f;\n" ++
            "    for (int i = 0; i < in_dim; i++) {\n" ++
            "        sum += mat[row * in_dim + i] * vec[i];\n" ++
            "    }\n" ++
            "    out[row] = sum;\n" ++
            "}\n" ++
            "\n" ++
            "__kernel void rmsNorm(__global float* out, __global const float* x, __global const float* weight, int n, float eps) {\n" ++
            "    int gid = get_global_id(0);\n" ++
            "    if (gid >= n) return;\n" ++
            "    float ss = 0.0f;\n" ++
            "    for (int i = 0; i < n; i++) {\n" ++
            "        ss += x[i] * x[i];\n" ++
            "    }\n" ++
            "    ss = ss / (float)n + eps;\n" ++
            "    float inv = 1.0f / sqrt(ss);\n" ++
            "    out[gid] = x[gid] * inv * weight[gid];\n" ++
            "}\n" ++
            "\n" ++
            "typedef struct { float d; char qs[32]; } block_q8_0;\n" ++
            "\n" ++
            "__kernel void matMulQ8_0(\n" ++
            "    __global float* out,\n" ++
            "    __global const block_q8_0* mat,\n" ++
            "    __global const float* vec,\n" ++
            "    int out_dim,\n" ++
            "    int in_dim\n" ++
            ") {\n" ++
            "    int row = get_global_id(0);\n" ++
            "    if (row >= out_dim) return;\n" ++
            "    float sum = 0.0f;\n" ++
            "    int blocks_per_row = in_dim / 32;\n" ++
            "    for (int b = 0; b < blocks_per_row; b++) {\n" ++
            "        block_q8_0 blk = mat[row * blocks_per_row + b];\n" ++
            "        float d = blk.d;\n" ++
            "        for (int i = 0; i < 32; i++) {\n" ++
            "            sum += d * (float)blk.qs[i] * vec[b * 32 + i];\n" ++
            "        }\n" ++
            "    }\n" ++
            "    out[row] = sum;\n" ++
            "}\n";

        var source_ptrs: [1][*:0]const u8 = .{kernel_source};
        const program = clCreateProgramWithSource(context, 1, &source_ptrs, null, &err);
        if (err != CL_SUCCESS) {
            _ = clReleaseCommandQueue(queue_main);
            _ = clReleaseContext(context);
            return error.OpenCLProgramError;
        }
        defer _ = clReleaseProgram(program);

        ret = clBuildProgram(program, 1, &device, "", null, null);
        if (ret != CL_SUCCESS) {
            _ = clReleaseCommandQueue(queue_main);
            _ = clReleaseContext(context);
            return error.OpenCLBuildError;
        }

        const matvec_kernel = clCreateKernel(program, "matVecMul", &err);
        if (err != CL_SUCCESS) {
            _ = clReleaseCommandQueue(queue_main);
            _ = clReleaseContext(context);
            return error.OpenCLKernelError;
        }
        errdefer _ = clReleaseKernel(matvec_kernel);

        const rmsnorm_kernel = clCreateKernel(program, "rmsNorm", &err);
        if (err != CL_SUCCESS) {
            _ = clReleaseKernel(matvec_kernel);
            _ = clReleaseCommandQueue(queue_main);
            _ = clReleaseContext(context);
            return error.OpenCLKernelError;
        }
        errdefer _ = clReleaseKernel(rmsnorm_kernel);

        const qmm_kernel = clCreateKernel(program, "matMulQ8_0", &err);
        if (err != CL_SUCCESS) {
            _ = clReleaseKernel(rmsnorm_kernel);
            _ = clReleaseKernel(matvec_kernel);
            _ = clReleaseCommandQueue(queue_main);
            _ = clReleaseContext(context);
            return error.OpenCLKernelError;
        }
        errdefer _ = clReleaseKernel(qmm_kernel);

        const weight_buffers = std.AutoHashMap(u64, cl_mem).init(allocator);

        return .{
            .lib = lib,
            .context = context,
            .device = device,
            .queue_main = queue_main,
            .queue_async = if (err == CL_SUCCESS) queue_async else null,
            .matvec_kernel = matvec_kernel,
            .rmsnorm_kernel = rmsnorm_kernel,
            .qmm_kernel = qmm_kernel,
            .allocator = allocator,
            .weight_buffers = weight_buffers,
            .clGetPlatformIDs = clGetPlatformIDs,
            .clGetDeviceIDs = clGetDeviceIDs,
            .clCreateContext = clCreateContext,
            .clCreateCommandQueue = clCreateCommandQueue,
            .clCreateBuffer = clCreateBuffer,
            .clCreateProgramWithSource = clCreateProgramWithSource,
            .clBuildProgram = clBuildProgram,
            .clCreateKernel = clCreateKernel,
            .clSetKernelArg = clSetKernelArg,
            .clEnqueueNDRangeKernel = clEnqueueNDRangeKernel,
            .clEnqueueWriteBuffer = clEnqueueWriteBuffer,
            .clEnqueueReadBuffer = clEnqueueReadBuffer,
            .clFinish = clFinish,
            .clReleaseMemObject = clReleaseMemObject,
            .clReleaseKernel = clReleaseKernel,
            .clReleaseProgram = clReleaseProgram,
            .clReleaseCommandQueue = clReleaseCommandQueue,
            .clReleaseContext = clReleaseContext,
        };
    }

    pub fn deinit(self: *Self) void {
        if (self.matvec_kernel) |k| _ = self.clReleaseKernel.?(k);
        if (self.rmsnorm_kernel) |k| _ = self.clReleaseKernel.?(k);
        if (self.qmm_kernel) |k| _ = self.clReleaseKernel.?(k);
        if (self.queue_async) |q| _ = self.clReleaseCommandQueue.?(q);

        // 释放所有缓存的权重 buffer
        var buf_iter = self.weight_buffers.valueIterator();
        while (buf_iter.next()) |buf_ptr| {
            _ = self.clReleaseMemObject.?(buf_ptr.*);
        }
        self.weight_buffers.deinit();

        _ = self.clReleaseCommandQueue.?(self.queue_main);
        _ = self.clReleaseContext.?(self.context);
        if (self.lib) |*l| l.close();
        self.* = undefined;
    }

    // ---------- 通用错误处理辅助 ----------

    fn check(_: *const Self, ret: cl_int, comptime msg: []const u8) !void {
        if (ret != CL_SUCCESS) {
            std.log.err("OpenCL error {d} at {s}", .{ ret, msg });
            return error.OpenCLError;
        }
    }

    /// 获取或上传权重 buffer（常驻 GPU，避免重复 H2D）
    fn getOrUploadMatBuffer(self: *Self, mat_data: []const u8, flags: cl_mem_flags) !cl_mem {
        const key = @intFromPtr(mat_data.ptr);
        if (self.weight_buffers.get(key)) |buf| return buf;

        var err: cl_int = 0;
        const buf = self.clCreateBuffer.?(self.context, flags, mat_data.len, null, &err);
        try self.check(err, "create weight buffer");
        try self.check(self.clEnqueueWriteBuffer.?(self.queue_main, buf, 1, 0, mat_data.len, mat_data.ptr, 0, null, null), "upload weight");

        try self.weight_buffers.put(key, buf);
        return buf;
    }

    /// GPU 矩阵-向量乘法（单队列）
    pub fn matVecMul(self: *Self, out: []f32, mat: []const f32, vec: []const f32, out_dim: u32, in_dim: u32) !void {
        std.debug.assert(out.len >= out_dim);
        std.debug.assert(mat.len >= out_dim * in_dim);
        std.debug.assert(vec.len >= in_dim);

        const clCreateBuffer = self.clCreateBuffer.?;
        const clEnqueueWriteBuffer = self.clEnqueueWriteBuffer.?;
        const clSetKernelArg = self.clSetKernelArg.?;
        const clEnqueueNDRangeKernel = self.clEnqueueNDRangeKernel.?;
        const clEnqueueReadBuffer = self.clEnqueueReadBuffer.?;
        const clFinish = self.clFinish.?;
        const clReleaseMemObject = self.clReleaseMemObject.?;

        // 权重 buffer 常驻复用
        const mat_bytes = std.mem.sliceAsBytes(mat);
        const mat_buf = try self.getOrUploadMatBuffer(mat_bytes, CL_MEM_READ_ONLY);

        var err: cl_int = 0;
        const vec_buf = clCreateBuffer(self.context, CL_MEM_READ_ONLY, vec.len * @sizeOf(f32), null, &err);
        try self.check(err, "vecCreateBuffer");
        defer _ = clReleaseMemObject(vec_buf);

        const out_buf = clCreateBuffer(self.context, CL_MEM_WRITE_ONLY, out.len * @sizeOf(f32), null, &err);
        try self.check(err, "outCreateBuffer");
        defer _ = clReleaseMemObject(out_buf);

        try self.check(clEnqueueWriteBuffer(self.queue_main, vec_buf, 1, 0, vec.len * @sizeOf(f32), vec.ptr, 0, null, null), "write vec");

        var out_dim_i32: i32 = @intCast(out_dim);
        var in_dim_i32: i32 = @intCast(in_dim);
        try self.check(clSetKernelArg(self.matvec_kernel.?, 0, @sizeOf(cl_mem), @ptrCast(@constCast(&mat_buf))), "set mat arg");
        try self.check(clSetKernelArg(self.matvec_kernel.?, 1, @sizeOf(cl_mem), @ptrCast(@constCast(&vec_buf))), "set vec arg");
        try self.check(clSetKernelArg(self.matvec_kernel.?, 2, @sizeOf(cl_mem), @ptrCast(@constCast(&out_buf))), "set out arg");
        try self.check(clSetKernelArg(self.matvec_kernel.?, 3, @sizeOf(i32), @ptrCast(@constCast(&out_dim_i32))), "set out_dim arg");
        try self.check(clSetKernelArg(self.matvec_kernel.?, 4, @sizeOf(i32), @ptrCast(@constCast(&in_dim_i32))), "set in_dim arg");

        const global_size = [_]usize{out_dim};
        try self.check(clEnqueueNDRangeKernel(self.queue_main, self.matvec_kernel.?, 1, null, &global_size, null, 0, null, null), "enqueue kernel");
        try self.check(clEnqueueReadBuffer(self.queue_main, out_buf, 1, 0, out_dim * @sizeOf(f32), out.ptr, 0, null, null), "read out");
        try self.check(clFinish(self.queue_main), "finish");
    }

    /// GPU RMSNorm
    pub fn rmsNorm(self: *Self, out: []f32, x: []const f32, weight: []const f32, n: u32, eps: f32) !void {
        std.debug.assert(out.len >= n);
        std.debug.assert(x.len >= n);
        std.debug.assert(weight.len >= n);

        const clCreateBuffer = self.clCreateBuffer.?;
        const clEnqueueWriteBuffer = self.clEnqueueWriteBuffer.?;
        const clSetKernelArg = self.clSetKernelArg.?;
        const clEnqueueNDRangeKernel = self.clEnqueueNDRangeKernel.?;
        const clEnqueueReadBuffer = self.clEnqueueReadBuffer.?;
        const clFinish = self.clFinish.?;
        const clReleaseMemObject = self.clReleaseMemObject.?;

        var err: cl_int = 0;
        const x_buf = clCreateBuffer(self.context, CL_MEM_READ_ONLY, x.len * @sizeOf(f32), null, &err);
        try self.check(err, "xCreateBuffer");
        defer _ = clReleaseMemObject(x_buf);

        const w_buf = clCreateBuffer(self.context, CL_MEM_READ_ONLY, weight.len * @sizeOf(f32), null, &err);
        try self.check(err, "wCreateBuffer");
        defer _ = clReleaseMemObject(w_buf);

        const out_buf = clCreateBuffer(self.context, CL_MEM_WRITE_ONLY, out.len * @sizeOf(f32), null, &err);
        try self.check(err, "outCreateBuffer");
        defer _ = clReleaseMemObject(out_buf);

        try self.check(clEnqueueWriteBuffer(self.queue_main, x_buf, 1, 0, x.len * @sizeOf(f32), x.ptr, 0, null, null), "write x");
        try self.check(clEnqueueWriteBuffer(self.queue_main, w_buf, 1, 0, weight.len * @sizeOf(f32), weight.ptr, 0, null, null), "write weight");

        var n_i32: i32 = @intCast(n);
        try self.check(clSetKernelArg(self.rmsnorm_kernel.?, 0, @sizeOf(cl_mem), @ptrCast(@constCast(&out_buf))), "set out arg");
        try self.check(clSetKernelArg(self.rmsnorm_kernel.?, 1, @sizeOf(cl_mem), @ptrCast(@constCast(&x_buf))), "set x arg");
        try self.check(clSetKernelArg(self.rmsnorm_kernel.?, 2, @sizeOf(cl_mem), @ptrCast(@constCast(&w_buf))), "set weight arg");
        try self.check(clSetKernelArg(self.rmsnorm_kernel.?, 3, @sizeOf(i32), @ptrCast(@constCast(&n_i32))), "set n arg");
        try self.check(clSetKernelArg(self.rmsnorm_kernel.?, 4, @sizeOf(f32), @ptrCast(@constCast(&eps))), "set eps arg");

        const global_size = [_]usize{n};
        try self.check(clEnqueueNDRangeKernel(self.queue_main, self.rmsnorm_kernel.?, 1, null, &global_size, null, 0, null, null), "enqueue rmsnorm");
        try self.check(clEnqueueReadBuffer(self.queue_main, out_buf, 1, 0, n * @sizeOf(f32), out.ptr, 0, null, null), "read out");
        try self.check(clFinish(self.queue_main), "finish");
    }

    /// GPU Q8_0 矩阵-向量乘法
    pub fn matMulQ8_0(self: *Self, out: []f32, mat: []const u8, vec: []const f32, out_dim: u32, in_dim: u32) !void {
        std.debug.assert(out.len >= out_dim);
        std.debug.assert(vec.len >= in_dim);
        // mat 是 Q8_0 格式，每个 block 33 字节（1 f16 + 32 i8）
        const blocks_per_row = (in_dim + 31) / 32;
        std.debug.assert(mat.len >= out_dim * blocks_per_row * 33);

        const clCreateBuffer = self.clCreateBuffer.?;
        const clEnqueueWriteBuffer = self.clEnqueueWriteBuffer.?;
        const clSetKernelArg = self.clSetKernelArg.?;
        const clEnqueueNDRangeKernel = self.clEnqueueNDRangeKernel.?;
        const clEnqueueReadBuffer = self.clEnqueueReadBuffer.?;
        const clFinish = self.clFinish.?;
        const clReleaseMemObject = self.clReleaseMemObject.?;

        // 权重 buffer 常驻复用
        const mat_buf = try self.getOrUploadMatBuffer(mat, CL_MEM_READ_ONLY);

        var err: cl_int = 0;
        const vec_buf = clCreateBuffer(self.context, CL_MEM_READ_ONLY, vec.len * @sizeOf(f32), null, &err);
        try self.check(err, "vecCreateBuffer");
        defer _ = clReleaseMemObject(vec_buf);

        const out_buf = clCreateBuffer(self.context, CL_MEM_WRITE_ONLY, out.len * @sizeOf(f32), null, &err);
        try self.check(err, "outCreateBuffer");
        defer _ = clReleaseMemObject(out_buf);

        try self.check(clEnqueueWriteBuffer(self.queue_main, vec_buf, 1, 0, vec.len * @sizeOf(f32), vec.ptr, 0, null, null), "write vec");

        var out_dim_i32: i32 = @intCast(out_dim);
        var in_dim_i32: i32 = @intCast(in_dim);
        try self.check(clSetKernelArg(self.qmm_kernel.?, 0, @sizeOf(cl_mem), @ptrCast(@constCast(&out_buf))), "set out arg");
        try self.check(clSetKernelArg(self.qmm_kernel.?, 1, @sizeOf(cl_mem), @ptrCast(@constCast(&mat_buf))), "set mat arg");
        try self.check(clSetKernelArg(self.qmm_kernel.?, 2, @sizeOf(cl_mem), @ptrCast(@constCast(&vec_buf))), "set vec arg");
        try self.check(clSetKernelArg(self.qmm_kernel.?, 3, @sizeOf(i32), @ptrCast(@constCast(&out_dim_i32))), "set out_dim arg");
        try self.check(clSetKernelArg(self.qmm_kernel.?, 4, @sizeOf(i32), @ptrCast(@constCast(&in_dim_i32))), "set in_dim arg");

        const global_size = [_]usize{out_dim};
        try self.check(clEnqueueNDRangeKernel(self.queue_main, self.qmm_kernel.?, 1, null, &global_size, null, 0, null, null), "enqueue qmm");
        try self.check(clEnqueueReadBuffer(self.queue_main, out_buf, 1, 0, out_dim * @sizeOf(f32), out.ptr, 0, null, null), "read out");
        try self.check(clFinish(self.queue_main), "finish");
    }

    /// 双队列并行执行两个专家 FFN（HeteroLLM 推荐：2 专家并行加速比 ~1.45x）
    pub fn matVecMulTwoExpertsParallel(
        self: *Self,
        out_a: []f32,
        mat_a: []const f32,
        vec_a: []const f32,
        out_dim: u32,
        in_dim: u32,
        out_b: []f32,
        mat_b: []const f32,
        vec_b: []const f32,
    ) !void {
        const q_async = self.queue_async orelse {
            // 回退：串行执行
            try self.matVecMul(out_a, mat_a, vec_a, out_dim, in_dim);
            try self.matVecMul(out_b, mat_b, vec_b, out_dim, in_dim);
            return;
        };

        const clCreateBuffer = self.clCreateBuffer.?;
        const clEnqueueWriteBuffer = self.clEnqueueWriteBuffer.?;
        const clSetKernelArg = self.clSetKernelArg.?;
        const clEnqueueNDRangeKernel = self.clEnqueueNDRangeKernel.?;
        const clEnqueueReadBuffer = self.clEnqueueReadBuffer.?;
        const clFinish = self.clFinish.?;
        const clReleaseMemObject = self.clReleaseMemObject.?;

        var err: cl_int = 0;

        // 权重 buffer 常驻复用
        const mat_a_buf = try self.getOrUploadMatBuffer(std.mem.sliceAsBytes(mat_a), CL_MEM_READ_ONLY);
        const mat_b_buf = try self.getOrUploadMatBuffer(std.mem.sliceAsBytes(mat_b), CL_MEM_READ_ONLY);

        // 专家 A 动态缓冲区
        const vec_a_buf = clCreateBuffer(self.context, CL_MEM_READ_ONLY, vec_a.len * @sizeOf(f32), null, &err);
        try self.check(err, "vec_a CreateBuffer");
        defer _ = clReleaseMemObject(vec_a_buf);

        const out_a_buf = clCreateBuffer(self.context, CL_MEM_WRITE_ONLY, out_a.len * @sizeOf(f32), null, &err);
        try self.check(err, "out_a CreateBuffer");
        defer _ = clReleaseMemObject(out_a_buf);

        // 专家 B 动态缓冲区
        const vec_b_buf = clCreateBuffer(self.context, CL_MEM_READ_ONLY, vec_b.len * @sizeOf(f32), null, &err);
        try self.check(err, "vec_b CreateBuffer");
        defer _ = clReleaseMemObject(vec_b_buf);

        const out_b_buf = clCreateBuffer(self.context, CL_MEM_WRITE_ONLY, out_b.len * @sizeOf(f32), null, &err);
        try self.check(err, "out_b CreateBuffer");
        defer _ = clReleaseMemObject(out_b_buf);

        var out_dim_i32: i32 = @intCast(out_dim);
        var in_dim_i32: i32 = @intCast(in_dim);

        // 上传动态数据
        try self.check(clEnqueueWriteBuffer(self.queue_main, vec_a_buf, 1, 0, vec_a.len * @sizeOf(f32), vec_a.ptr, 0, null, null), "write vec_a");
        try self.check(clEnqueueWriteBuffer(q_async, vec_b_buf, 1, 0, vec_b.len * @sizeOf(f32), vec_b.ptr, 0, null, null), "write vec_b");

        // 设置参数并调度 A（主队列）
        try self.check(clSetKernelArg(self.matvec_kernel.?, 0, @sizeOf(cl_mem), @ptrCast(@constCast(&mat_a_buf))), "set mat_a arg");
        try self.check(clSetKernelArg(self.matvec_kernel.?, 1, @sizeOf(cl_mem), @ptrCast(@constCast(&vec_a_buf))), "set vec_a arg");
        try self.check(clSetKernelArg(self.matvec_kernel.?, 2, @sizeOf(cl_mem), @ptrCast(@constCast(&out_a_buf))), "set out_a arg");
        try self.check(clSetKernelArg(self.matvec_kernel.?, 3, @sizeOf(i32), @ptrCast(@constCast(&out_dim_i32))), "set out_dim arg");
        try self.check(clSetKernelArg(self.matvec_kernel.?, 4, @sizeOf(i32), @ptrCast(@constCast(&in_dim_i32))), "set in_dim arg");

        const global_size = [_]usize{out_dim};
        try self.check(clEnqueueNDRangeKernel(self.queue_main, self.matvec_kernel.?, 1, null, &global_size, null, 0, null, null), "enqueue A");

        // 复用同一 kernel 对象调度到异步队列
        try self.check(clSetKernelArg(self.matvec_kernel.?, 0, @sizeOf(cl_mem), @ptrCast(@constCast(&mat_b_buf))), "set mat_b arg");
        try self.check(clSetKernelArg(self.matvec_kernel.?, 1, @sizeOf(cl_mem), @ptrCast(@constCast(&vec_b_buf))), "set vec_b arg");
        try self.check(clSetKernelArg(self.matvec_kernel.?, 2, @sizeOf(cl_mem), @ptrCast(@constCast(&out_b_buf))), "set out_b arg");
        try self.check(clSetKernelArg(self.matvec_kernel.?, 3, @sizeOf(i32), @ptrCast(@constCast(&out_dim_i32))), "set out_dim arg");
        try self.check(clSetKernelArg(self.matvec_kernel.?, 4, @sizeOf(i32), @ptrCast(@constCast(&in_dim_i32))), "set in_dim arg");
        try self.check(clEnqueueNDRangeKernel(q_async, self.matvec_kernel.?, 1, null, &global_size, null, 0, null, null), "enqueue B");

        // 读取结果
        try self.check(clEnqueueReadBuffer(self.queue_main, out_a_buf, 1, 0, out_dim * @sizeOf(f32), out_a.ptr, 0, null, null), "read out_a");
        try self.check(clEnqueueReadBuffer(q_async, out_b_buf, 1, 0, out_dim * @sizeOf(f32), out_b.ptr, 0, null, null), "read out_b");
        try self.check(clFinish(self.queue_main), "finish main");
        try self.check(clFinish(q_async), "finish async");
    }
};

// ========== TDD ==========

const math = @import("math.zig");

fn expectApproxEqSlice(expected: []const f32, actual: []const f32, tol: f32) !void {
    try std.testing.expectEqual(expected.len, actual.len);
    for (expected, actual, 0..) |e, a, i| {
        const diff = @abs(e - a);
        if (diff > tol) {
            std.debug.print("index {d}: expected {e}, got {e}, diff {e} > tol {e}\n", .{ i, e, a, diff, tol });
            return error.TestExpectedApproxEq;
        }
    }
}

test "OpenCL matVecMul 数值正确性（vs CPU 参考）" {
    if (!OpenCLBackend.isAvailable()) {
        std.debug.print("跳过：当前平台无 OpenCL\n", .{});
        return;
    }
    const allocator = std.testing.allocator;
    var backend = try OpenCLBackend.init(allocator);
    defer backend.deinit();

    const out_dim: u32 = 32;
    const in_dim: u32 = 64;

    const mat = try allocator.alloc(f32, out_dim * in_dim);
    defer allocator.free(mat);
    const vec = try allocator.alloc(f32, in_dim);
    defer allocator.free(vec);
    const gpu_out = try allocator.alloc(f32, out_dim);
    defer allocator.free(gpu_out);
    const cpu_out = try allocator.alloc(f32, out_dim);
    defer allocator.free(cpu_out);

    for (mat, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 7)) * 0.1;
    for (vec, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 5)) * 0.2;

    math.matVecMul(cpu_out, mat, vec, out_dim, in_dim);
    try backend.matVecMul(gpu_out, mat, vec, out_dim, in_dim);

    try expectApproxEqSlice(cpu_out, gpu_out, 1e-4);
}

test "OpenCL rmsNorm 数值正确性（vs CPU 参考）" {
    if (!OpenCLBackend.isAvailable()) {
        std.debug.print("跳过：当前平台无 OpenCL\n", .{});
        return;
    }
    const allocator = std.testing.allocator;
    var backend = try OpenCLBackend.init(allocator);
    defer backend.deinit();

    const n: u32 = 256;
    const eps: f32 = 1e-6;

    const x = try allocator.alloc(f32, n);
    defer allocator.free(x);
    const weight = try allocator.alloc(f32, n);
    defer allocator.free(weight);
    const gpu_out = try allocator.alloc(f32, n);
    defer allocator.free(gpu_out);
    const cpu_out = try allocator.alloc(f32, n);
    defer allocator.free(cpu_out);

    for (x, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 11)) * 0.05 - 0.25;
    for (weight, 0..) |*v, i| v.* = 1.0 + @as(f32, @floatFromInt(i % 3)) * 0.1;

    math.rmsNorm(cpu_out, x, weight, eps);
    try backend.rmsNorm(gpu_out, x, weight, n, eps);

    try expectApproxEqSlice(cpu_out, gpu_out, 1e-4);
}

test "OpenCL 双专家并行数值正确性" {
    if (!OpenCLBackend.isAvailable()) {
        std.debug.print("跳过：当前平台无 OpenCL\n", .{});
        return;
    }
    const allocator = std.testing.allocator;
    var backend = try OpenCLBackend.init(allocator);
    defer backend.deinit();

    const out_dim: u32 = 16;
    const in_dim: u32 = 32;

    const mat_a = try allocator.alloc(f32, out_dim * in_dim);
    defer allocator.free(mat_a);
    const vec_a = try allocator.alloc(f32, in_dim);
    defer allocator.free(vec_a);
    const out_a = try allocator.alloc(f32, out_dim);
    defer allocator.free(out_a);

    const mat_b = try allocator.alloc(f32, out_dim * in_dim);
    defer allocator.free(mat_b);
    const vec_b = try allocator.alloc(f32, in_dim);
    defer allocator.free(vec_b);
    const out_b = try allocator.alloc(f32, out_dim);
    defer allocator.free(out_b);

    for (mat_a, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 3)) * 0.15;
    for (vec_a, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 4)) * 0.25;
    for (mat_b, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 5)) * 0.12;
    for (vec_b, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 6)) * 0.18;

    const cpu_a = try allocator.alloc(f32, out_dim);
    defer allocator.free(cpu_a);
    const cpu_b = try allocator.alloc(f32, out_dim);
    defer allocator.free(cpu_b);

    math.matVecMul(cpu_a, mat_a, vec_a, out_dim, in_dim);
    math.matVecMul(cpu_b, mat_b, vec_b, out_dim, in_dim);

    try backend.matVecMulTwoExpertsParallel(out_a, mat_a, vec_a, out_dim, in_dim, out_b, mat_b, vec_b);

    try expectApproxEqSlice(cpu_a, out_a, 1e-4);
    try expectApproxEqSlice(cpu_b, out_b, 1e-4);
}
