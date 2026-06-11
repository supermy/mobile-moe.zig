const std = @import("std");

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

const MATRIX_SIZE: usize = 512;
const ITERATIONS: usize = 100;

const kernel_source =
    "__kernel void matMul(__global const float* A, __global const float* B, __global float* C, int n) {\n" ++
    "    int row = get_global_id(0);\n" ++
    "    int col = get_global_id(1);\n" ++
    "    float sum = 0.0f;\n" ++
    "    for (int k = 0; k < n; k++) {\n" ++
    "        sum += A[row * n + k] * B[k * n + col];\n" ++
    "    }\n" ++
    "    C[row * n + col] = sum;\n" ++
    "}\n";

fn getTimeNs() u64 {
    var ts: std.os.linux.timespec = undefined;
    _ = std.os.linux.clock_gettime(std.os.linux.CLOCK.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

pub fn main() !void {
    const allocator = std.heap.page_allocator;
    const n = MATRIX_SIZE;
    const total_elements = n * n;

    std.debug.print("[0] Preloading EGL/GLES to init GPU driver...\n", .{});
    _ = std.DynLib.open("/vendor/lib64/libEGL.so") catch {};
    _ = std.DynLib.open("/vendor/lib64/libGLESv2.so") catch {};
    _ = std.DynLib.open("/vendor/lib64/libGLESv3.so") catch {};

    std.debug.print("[1] Loading OpenCL...\n", .{});
    var lib = std.DynLib.open("/vendor/lib64/libOpenCL_adreno.so") catch std.DynLib.open("/vendor/lib64/libOpenCL.so") catch |e| {
        std.debug.print("Failed to load OpenCL library: {any}\n", .{e});
        return;
    };
    defer lib.close();

    const clGetPlatformIDs = lib.lookup(ClGetPlatformIDs, "clGetPlatformIDs").?;
    const clGetDeviceIDs = lib.lookup(ClGetDeviceIDs, "clGetDeviceIDs").?;
    const clCreateContext = lib.lookup(ClCreateContext, "clCreateContext").?;
    const clCreateCommandQueue = lib.lookup(ClCreateCommandQueue, "clCreateCommandQueue").?;
    const clCreateBuffer = lib.lookup(ClCreateBuffer, "clCreateBuffer").?;
    const clCreateProgramWithSource = lib.lookup(ClCreateProgramWithSource, "clCreateProgramWithSource").?;
    const clBuildProgram = lib.lookup(ClBuildProgram, "clBuildProgram").?;
    const clCreateKernel = lib.lookup(ClCreateKernel, "clCreateKernel").?;
    const clSetKernelArg = lib.lookup(ClSetKernelArg, "clSetKernelArg").?;
    const clEnqueueNDRangeKernel = lib.lookup(ClEnqueueNDRangeKernel, "clEnqueueNDRangeKernel").?;
    const clEnqueueWriteBuffer = lib.lookup(ClEnqueueWriteBuffer, "clEnqueueWriteBuffer").?;
    const clEnqueueReadBuffer = lib.lookup(ClEnqueueReadBuffer, "clEnqueueReadBuffer").?;
    _ = clEnqueueReadBuffer;
    const clFinish = lib.lookup(ClFinish, "clFinish").?;
    const clReleaseMemObject = lib.lookup(ClReleaseMemObject, "clReleaseMemObject").?;
    const clReleaseKernel = lib.lookup(ClReleaseKernel, "clReleaseKernel").?;
    const clReleaseProgram = lib.lookup(ClReleaseProgram, "clReleaseProgram").?;
    const clReleaseCommandQueue = lib.lookup(ClReleaseCommandQueue, "clReleaseCommandQueue").?;
    const clReleaseContext = lib.lookup(ClReleaseContext, "clReleaseContext").?;

    std.debug.print("[2] Getting platform count...\n", .{});
    var num_platforms: cl_uint = 0;
    var ret = clGetPlatformIDs(0, null, &num_platforms);
    std.debug.print("Platform count ret={d} num={d}\n", .{ret, num_platforms});
    if (ret != CL_SUCCESS or num_platforms == 0) {
        std.debug.print("No OpenCL platforms found\n", .{});
        return;
    }

    std.debug.print("[3] Getting platform...\n", .{});
    var platform: cl_platform_id = null;
    ret = clGetPlatformIDs(1, &platform, null);
    std.debug.print("Platform ret={d} ptr={*}\n", .{ret, platform});
    if (ret != CL_SUCCESS) {
        std.debug.print("Failed to get OpenCL platform: {d}\n", .{ret});
        return;
    }

    std.debug.print("[4] Getting device...\n", .{});
    var device: cl_device_id = null;
    ret = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, null);
    std.debug.print("Device ret={d} ptr={*}\n", .{ret, device});
    if (ret != CL_SUCCESS) {
        std.debug.print("Failed to get GPU device: {d}\n", .{ret});
        return;
    }

    // 创建上下文和命令队列
    var err: cl_int = 0;
    var device_arr = [_]cl_device_id{device};
    const context = clCreateContext(null, 1, &device_arr, null, null, &err);
    if (err != CL_SUCCESS) {
        std.debug.print("Failed to create context: {d}\n", .{err});
        return;
    }
    defer _ = clReleaseContext(context);

    std.debug.print("[5] Creating queue...\n", .{});
    const queue = clCreateCommandQueue(context, device, 0, &err);
    if (err != CL_SUCCESS) {
        std.debug.print("Failed to create command queue: {d}\n", .{err});
        return;
    }
    defer _ = clReleaseCommandQueue(queue);

    std.debug.print("[6] Allocating host memory...\n", .{});
    const A_host = try allocator.alloc(f32, total_elements);
    const B_host = try allocator.alloc(f32, total_elements);
    const C_host = try allocator.alloc(f32, total_elements);
    defer allocator.free(A_host);
    defer allocator.free(B_host);
    defer allocator.free(C_host);

    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();
    for (A_host) |*a| a.* = rng.float(f32);
    for (B_host) |*b| b.* = rng.float(f32);

    std.debug.print("[7] Creating buffers...\n", .{});
    const A_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, total_elements * @sizeOf(f32), null, &err);
    const B_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, total_elements * @sizeOf(f32), null, &err);
    const C_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, total_elements * @sizeOf(f32), null, &err);
    defer _ = clReleaseMemObject(A_buf);
    defer _ = clReleaseMemObject(B_buf);
    defer _ = clReleaseMemObject(C_buf);

    std.debug.print("[8] Writing buffers...\n", .{});
    _ = clEnqueueWriteBuffer(queue, A_buf, 1, 0, total_elements * @sizeOf(f32), A_host.ptr, 0, null, null);
    _ = clEnqueueWriteBuffer(queue, B_buf, 1, 0, total_elements * @sizeOf(f32), B_host.ptr, 0, null, null);
    _ = clFinish(queue);

    std.debug.print("[9] Creating program...\n", .{});
    var source_ptrs: [1][*:0]const u8 = .{kernel_source};
    const program = clCreateProgramWithSource(context, 1, &source_ptrs, null, &err);
    if (err != CL_SUCCESS) {
        std.debug.print("Failed to create program: {d}\n", .{err});
        return;
    }
    defer _ = clReleaseProgram(program);

    std.debug.print("[10] Building program...\n", .{});
    ret = clBuildProgram(program, 1, &device, "", null, null);
    if (ret != CL_SUCCESS) {
        std.debug.print("Failed to build program: {d}\n", .{ret});
        return;
    }

    std.debug.print("[11] Creating kernel...\n", .{});
    const kernel = clCreateKernel(program, "matMul", &err);
    if (err != CL_SUCCESS) {
        std.debug.print("Failed to create kernel: {d}\n", .{err});
        return;
    }
    defer _ = clReleaseKernel(kernel);

    std.debug.print("[12] Setting args...\n", .{});
    var n_i32: i32 = @intCast(n);
    _ = clSetKernelArg(kernel, 0, @sizeOf(cl_mem), @ptrCast(@constCast(&A_buf)));
    _ = clSetKernelArg(kernel, 1, @sizeOf(cl_mem), @ptrCast(@constCast(&B_buf)));
    _ = clSetKernelArg(kernel, 2, @sizeOf(cl_mem), @ptrCast(@constCast(&C_buf)));
    _ = clSetKernelArg(kernel, 3, @sizeOf(i32), @ptrCast(@constCast(&n_i32)));

    std.debug.print("[13] Warmup...\n", .{});
    const global_size = [_]usize{ n, n };
    _ = clEnqueueNDRangeKernel(queue, kernel, 2, null, &global_size, null, 0, null, null);
    _ = clFinish(queue);

    std.debug.print("[14] Benchmarking...\n", .{});
    const start = getTimeNs();
    for (0..ITERATIONS) |_| {
        _ = clEnqueueNDRangeKernel(queue, kernel, 2, null, &global_size, null, 0, null, null);
    }
    _ = clFinish(queue);
    const end = getTimeNs();

    const elapsed_ns = end - start;
    const elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0;
    const elapsed_s = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000_000.0;
    const total_flops = @as(f64, @floatFromInt(ITERATIONS)) * 2.0 * @as(f64, @floatFromInt(n * n * n));
    const gflops = total_flops / (elapsed_s * 1e9);

    std.debug.print("=== GPU (OpenCL) Benchmark ===\n", .{});
    std.debug.print("Matrix size: {d}x{d}\n", .{n, n});
    std.debug.print("Iterations: {d}\n", .{ITERATIONS});
    std.debug.print("Total time: {d:.2} ms\n", .{elapsed_ms});
    std.debug.print("Per iteration: {d:.2} ms\n", .{elapsed_ms / @as(f64, @floatFromInt(ITERATIONS))});
    std.debug.print("GFLOPS: {d:.2}\n", .{gflops});
}
