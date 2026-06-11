const std = @import("std");

// ========== EGL types & constants ==========
const EGLDisplay = ?*anyopaque;
const EGLContext = ?*anyopaque;
const EGLSurface = ?*anyopaque;
const EGLConfig = ?*anyopaque;
const EGLint = i32;
const EGLBoolean = u32;

const EGL_DEFAULT_DISPLAY: ?*anyopaque = null;
const EGL_NO_CONTEXT: EGLContext = null;
const EGL_NO_SURFACE: EGLSurface = null;
const EGL_TRUE: EGLBoolean = 1;
const EGL_FALSE: EGLBoolean = 0;

const EGL_SURFACE_TYPE = 0x3033;
const EGL_PBUFFER_BIT = 0x0001;
const EGL_RENDERABLE_TYPE = 0x3040;
const EGL_OPENGL_ES2_BIT = 0x0004;
const EGL_RED_SIZE = 0x3024;
const EGL_GREEN_SIZE = 0x3023;
const EGL_BLUE_SIZE = 0x3022;
const EGL_ALPHA_SIZE = 0x3021;
const EGL_NONE = 0x3038;
const EGL_WIDTH = 0x3057;
const EGL_HEIGHT = 0x3056;
const EGL_CONTEXT_CLIENT_VERSION = 0x3098;

const EglGetDisplay = *const fn (?*anyopaque) callconv(.c) EGLDisplay;
const EglInitialize = *const fn (EGLDisplay, ?*EGLint, ?*EGLint) callconv(.c) EGLBoolean;
const EglChooseConfig = *const fn (EGLDisplay, ?[*]const EGLint, ?*EGLConfig, EGLint, ?*EGLint) callconv(.c) EGLBoolean;
const EglCreateContext = *const fn (EGLDisplay, EGLConfig, EGLContext, ?[*]const EGLint) callconv(.c) EGLContext;
const EglCreatePbufferSurface = *const fn (EGLDisplay, EGLConfig, ?[*]const EGLint) callconv(.c) EGLSurface;
const EglMakeCurrent = *const fn (EGLDisplay, EGLSurface, EGLSurface, EGLContext) callconv(.c) EGLBoolean;
const EglTerminate = *const fn (EGLDisplay) callconv(.c) EGLBoolean;

// ========== OpenCL types & constants ==========
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

fn loadFunc(comptime T: type, lib: *std.DynLib, name: [:0]const u8) T {
    return lib.lookup(T, name).?;
}

pub fn main() !void {
    const allocator = std.heap.page_allocator;
    const n = MATRIX_SIZE;
    const total_elements = n * n;

    // ====== Step 0: Init EGL to trigger GPU driver ======
    std.debug.print("[0] Initializing EGL...\n", .{});
    var egl_lib = std.DynLib.open("/system/lib64/libEGL.so") catch std.DynLib.open("/vendor/lib64/libEGL.so") catch |e| {
        std.debug.print("Failed to load libEGL.so: {any}\n", .{e});
        return;
    };
    defer egl_lib.close();

    const eglGetDisplay = loadFunc(EglGetDisplay, &egl_lib, "eglGetDisplay");
    const eglInitialize = loadFunc(EglInitialize, &egl_lib, "eglInitialize");
    const eglChooseConfig = loadFunc(EglChooseConfig, &egl_lib, "eglChooseConfig");
    const eglCreateContext = loadFunc(EglCreateContext, &egl_lib, "eglCreateContext");
    const eglCreatePbufferSurface = loadFunc(EglCreatePbufferSurface, &egl_lib, "eglCreatePbufferSurface");
    const eglMakeCurrent = loadFunc(EglMakeCurrent, &egl_lib, "eglMakeCurrent");
    const eglTerminate = loadFunc(EglTerminate, &egl_lib, "eglTerminate");

    const display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (display == null) {
        std.debug.print("eglGetDisplay failed\n", .{});
        return;
    }

    var major: EGLint = 0;
    var minor: EGLint = 0;
    if (eglInitialize(display, &major, &minor) != EGL_TRUE) {
        std.debug.print("eglInitialize failed\n", .{});
        return;
    }
    std.debug.print("EGL version: {d}.{d}\n", .{major, minor});

    const attribs = [_]EGLint{
        EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
        EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT,
        EGL_RED_SIZE, 8,
        EGL_GREEN_SIZE, 8,
        EGL_BLUE_SIZE, 8,
        EGL_ALPHA_SIZE, 8,
        EGL_NONE,
    };
    var config: EGLConfig = null;
    var num_config: EGLint = 0;
    if (eglChooseConfig(display, &attribs, &config, 1, &num_config) != EGL_TRUE or num_config == 0) {
        std.debug.print("eglChooseConfig failed\n", .{});
        _ = eglTerminate(display);
        return;
    }

    const ctx_attribs = [_]EGLint{
        EGL_CONTEXT_CLIENT_VERSION, 2,
        EGL_NONE,
    };
    const ctx = eglCreateContext(display, config, EGL_NO_CONTEXT, &ctx_attribs);
    if (ctx == null) {
        std.debug.print("eglCreateContext failed\n", .{});
        _ = eglTerminate(display);
        return;
    }

    const surf_attribs = [_]EGLint{
        EGL_WIDTH, 1,
        EGL_HEIGHT, 1,
        EGL_NONE,
    };
    const surf = eglCreatePbufferSurface(display, config, &surf_attribs);
    if (surf == null) {
        std.debug.print("eglCreatePbufferSurface failed\n", .{});
        _ = eglTerminate(display);
        return;
    }

    if (eglMakeCurrent(display, surf, surf, ctx) != EGL_TRUE) {
        std.debug.print("eglMakeCurrent failed\n", .{});
        _ = eglTerminate(display);
        return;
    }
    std.debug.print("EGL context active\n", .{});

    // ====== Step 1: Load OpenCL ======
    std.debug.print("[1] Loading OpenCL...\n", .{});
    var cl_lib = std.DynLib.open("/vendor/lib64/libOpenCL_adreno.so") catch std.DynLib.open("/vendor/lib64/libOpenCL.so") catch |e| {
        std.debug.print("Failed to load OpenCL library: {any}\n", .{e});
        return;
    };
    defer cl_lib.close();

    const clGetPlatformIDs = loadFunc(ClGetPlatformIDs, &cl_lib, "clGetPlatformIDs");
    const clGetDeviceIDs = loadFunc(ClGetDeviceIDs, &cl_lib, "clGetDeviceIDs");
    const clCreateContext = loadFunc(ClCreateContext, &cl_lib, "clCreateContext");
    const clCreateCommandQueue = loadFunc(ClCreateCommandQueue, &cl_lib, "clCreateCommandQueue");
    const clCreateBuffer = loadFunc(ClCreateBuffer, &cl_lib, "clCreateBuffer");
    const clCreateProgramWithSource = loadFunc(ClCreateProgramWithSource, &cl_lib, "clCreateProgramWithSource");
    const clBuildProgram = loadFunc(ClBuildProgram, &cl_lib, "clBuildProgram");
    const clCreateKernel = loadFunc(ClCreateKernel, &cl_lib, "clCreateKernel");
    const clSetKernelArg = loadFunc(ClSetKernelArg, &cl_lib, "clSetKernelArg");
    const clEnqueueNDRangeKernel = loadFunc(ClEnqueueNDRangeKernel, &cl_lib, "clEnqueueNDRangeKernel");
    const clEnqueueWriteBuffer = loadFunc(ClEnqueueWriteBuffer, &cl_lib, "clEnqueueWriteBuffer");
    const clFinish = loadFunc(ClFinish, &cl_lib, "clFinish");
    const clReleaseMemObject = loadFunc(ClReleaseMemObject, &cl_lib, "clReleaseMemObject");
    const clReleaseKernel = loadFunc(ClReleaseKernel, &cl_lib, "clReleaseKernel");
    const clReleaseProgram = loadFunc(ClReleaseProgram, &cl_lib, "clReleaseProgram");
    const clReleaseCommandQueue = loadFunc(ClReleaseCommandQueue, &cl_lib, "clReleaseCommandQueue");
    const clReleaseContext = loadFunc(ClReleaseContext, &cl_lib, "clReleaseContext");

    // ====== Step 2: OpenCL platform/device ======
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
    if (ret != CL_SUCCESS) {
        std.debug.print("Failed to get OpenCL platform: {d}\n", .{ret});
        return;
    }

    std.debug.print("[4] Getting device...\n", .{});
    var device: cl_device_id = null;
    ret = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, null);
    if (ret != CL_SUCCESS) {
        std.debug.print("Failed to get GPU device: {d}\n", .{ret});
        return;
    }

    std.debug.print("[5] Creating context...\n", .{});
    var err: cl_int = 0;
    var device_arr = [_]cl_device_id{device};
    const cl_ctx = clCreateContext(null, 1, &device_arr, null, null, &err);
    if (err != CL_SUCCESS) {
        std.debug.print("Failed to create context: {d}\n", .{err});
        return;
    }
    defer _ = clReleaseContext(cl_ctx);

    std.debug.print("[6] Creating queue...\n", .{});
    const queue = clCreateCommandQueue(cl_ctx, device, 0, &err);
    if (err != CL_SUCCESS) {
        std.debug.print("Failed to create command queue: {d}\n", .{err});
        return;
    }
    defer _ = clReleaseCommandQueue(queue);

    // ====== Step 3: Data & buffers ======
    std.debug.print("[7] Allocating host memory...\n", .{});
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

    std.debug.print("[8] Creating buffers...\n", .{});
    const A_buf = clCreateBuffer(cl_ctx, CL_MEM_READ_ONLY, total_elements * @sizeOf(f32), null, &err);
    const B_buf = clCreateBuffer(cl_ctx, CL_MEM_READ_ONLY, total_elements * @sizeOf(f32), null, &err);
    const C_buf = clCreateBuffer(cl_ctx, CL_MEM_WRITE_ONLY, total_elements * @sizeOf(f32), null, &err);
    defer _ = clReleaseMemObject(A_buf);
    defer _ = clReleaseMemObject(B_buf);
    defer _ = clReleaseMemObject(C_buf);

    std.debug.print("[9] Writing buffers...\n", .{});
    _ = clEnqueueWriteBuffer(queue, A_buf, 1, 0, total_elements * @sizeOf(f32), A_host.ptr, 0, null, null);
    _ = clEnqueueWriteBuffer(queue, B_buf, 1, 0, total_elements * @sizeOf(f32), B_host.ptr, 0, null, null);
    _ = clFinish(queue);

    // ====== Step 4: Program & kernel ======
    std.debug.print("[10] Creating program...\n", .{});
    var source_ptrs: [1][*:0]const u8 = .{kernel_source};
    const program = clCreateProgramWithSource(cl_ctx, 1, &source_ptrs, null, &err);
    if (err != CL_SUCCESS) {
        std.debug.print("Failed to create program: {d}\n", .{err});
        return;
    }
    defer _ = clReleaseProgram(program);

    std.debug.print("[11] Building program...\n", .{});
    ret = clBuildProgram(program, 1, &device, "", null, null);
    if (ret != CL_SUCCESS) {
        std.debug.print("Failed to build program: {d}\n", .{ret});
        return;
    }

    std.debug.print("[12] Creating kernel...\n", .{});
    const kernel = clCreateKernel(program, "matMul", &err);
    if (err != CL_SUCCESS) {
        std.debug.print("Failed to create kernel: {d}\n", .{err});
        return;
    }
    defer _ = clReleaseKernel(kernel);

    std.debug.print("[13] Setting args...\n", .{});
    var n_i32: i32 = @intCast(n);
    _ = clSetKernelArg(kernel, 0, @sizeOf(cl_mem), @ptrCast(@constCast(&A_buf)));
    _ = clSetKernelArg(kernel, 1, @sizeOf(cl_mem), @ptrCast(@constCast(&B_buf)));
    _ = clSetKernelArg(kernel, 2, @sizeOf(cl_mem), @ptrCast(@constCast(&C_buf)));
    _ = clSetKernelArg(kernel, 3, @sizeOf(i32), @ptrCast(@constCast(&n_i32)));

    // ====== Step 5: Benchmark ======
    std.debug.print("[14] Warmup...\n", .{});
    const global_size = [_]usize{ n, n };
    _ = clEnqueueNDRangeKernel(queue, kernel, 2, null, &global_size, null, 0, null, null);
    _ = clFinish(queue);

    std.debug.print("[15] Benchmarking...\n", .{});
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
