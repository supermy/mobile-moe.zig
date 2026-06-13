// Metal 后端 Zig 包装器
// 封装 metal_bridge.c 的 C API，提供类型安全的 GPU 算子调用

const std = @import("std");

// C 桥接函数声明
extern fn metal_init() c_int;
extern fn metal_get_last_error() [*:0]const u8;
extern fn metal_create_device() ?*anyopaque;
extern fn metal_release(obj: ?*anyopaque) void;
extern fn metal_new_command_queue(device: ?*anyopaque) ?*anyopaque;
extern fn metal_new_buffer(device: ?*anyopaque, bytes: ?*const anyopaque, length: usize) ?*anyopaque;
extern fn metal_buffer_contents(buffer: ?*anyopaque) ?*anyopaque;
extern fn metal_buffer_length(buffer: ?*anyopaque) usize;
extern fn metal_create_library(device: ?*anyopaque, source: [*:0]const u8) ?*anyopaque;
extern fn metal_create_function(library: ?*anyopaque, name: [*:0]const u8) ?*anyopaque;
extern fn metal_create_pipeline(device: ?*anyopaque, function: ?*anyopaque) ?*anyopaque;
extern fn metal_command_buffer(queue: ?*anyopaque) ?*anyopaque;
extern fn metal_compute_encoder(cmdbuf: ?*anyopaque) ?*anyopaque;
extern fn metal_set_pipeline(encoder: ?*anyopaque, pipeline: ?*anyopaque) void;
extern fn metal_set_buffer(encoder: ?*anyopaque, buffer: ?*anyopaque, offset: usize, index: usize) void;
extern fn metal_set_bytes(encoder: ?*anyopaque, bytes: ?*const anyopaque, length: usize, index: usize) void;
extern fn metal_dispatch(encoder: ?*anyopaque, gx: usize, gy: usize, gz: usize, tx: usize, ty: usize, tz: usize) void;
extern fn metal_end_encoding(encoder: ?*anyopaque) void;
extern fn metal_commit(cmdbuf: ?*anyopaque) void;
extern fn metal_wait_completed(cmdbuf: ?*anyopaque) void;

/// 检测当前平台是否支持 Metal（macOS/iOS）
pub fn isAvailable() bool {
    return metal_init() == 0;
}

pub const MetalBackend = struct {
    device: ?*anyopaque,
    queue: ?*anyopaque,
    matvec_pipeline: ?*anyopaque,
    rmsnorm_pipeline: ?*anyopaque,
    allocator: std.mem.Allocator,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator) !Self {
        if (metal_init() != 0) {
            std.log.err("Metal init failed: {s}", .{metal_get_last_error()});
            return error.MetalInitFailed;
        }
        const device = metal_create_device() orelse return error.MetalDeviceNotFound;
        const queue = metal_new_command_queue(device) orelse return error.MetalQueueCreateFailed;

        // 编译 matVecMul shader
        const matvec_source =
            \\#include <metal_stdlib>
            \\using namespace metal;
            \\kernel void matVecMul(
            \\    device float* out        [[buffer(0)]],
            \\    device const float* mat  [[buffer(1)]],
            \\    device const float* vec  [[buffer(2)]],
            \\    constant uint& out_dim   [[buffer(3)]],
            \\    constant uint& in_dim    [[buffer(4)]],
            \\    uint gid                 [[thread_position_in_grid]]
            \\) {
            \\    if (gid >= out_dim) return;
            \\    float sum = 0.0f;
            \\    for (uint i = 0; i < in_dim; i++) {
            \\        sum += mat[gid * in_dim + i] * vec[i];
            \\    }
            \\    out[gid] = sum;
            \\}
        ;
        const matvec_lib = metal_create_library(device, matvec_source) orelse {
            std.log.err("Metal matVecMul compile failed: {s}", .{metal_get_last_error()});
            return error.MetalCompileFailed;
        };
        defer metal_release(matvec_lib);
        const matvec_func = metal_create_function(matvec_lib, "matVecMul") orelse return error.MetalFunctionNotFound;
        defer metal_release(matvec_func);
        const matvec_pipeline = metal_create_pipeline(device, matvec_func) orelse {
            std.log.err("Metal matVecMul pipeline failed: {s}", .{metal_get_last_error()});
            return error.MetalPipelineFailed;
        };

        // 编译 rmsNorm shader
        const rmsnorm_source =
            \\#include <metal_stdlib>
            \\using namespace metal;
            \\kernel void rmsNorm(
            \\    device float* out        [[buffer(0)]],
            \\    device const float* x    [[buffer(1)]],
            \\    device const float* weight [[buffer(2)]],
            \\    constant uint& n         [[buffer(3)]],
            \\    constant float& eps      [[buffer(4)]],
            \\    uint gid                 [[thread_position_in_grid]]
            \\) {
            \\    if (gid >= n) return;
            \\    float ss = 0.0f;
            \\    for (uint i = 0; i < n; i++) {
            \\        ss += x[i] * x[i];
            \\    }
            \\    ss = ss / float(n) + eps;
            \\    float inv = 1.0f / sqrt(ss);
            \\    out[gid] = x[gid] * inv * weight[gid];
            \\}
        ;
        const rmsnorm_lib = metal_create_library(device, rmsnorm_source) orelse {
            std.log.err("Metal rmsNorm compile failed: {s}", .{metal_get_last_error()});
            return error.MetalCompileFailed;
        };
        defer metal_release(rmsnorm_lib);
        const rmsnorm_func = metal_create_function(rmsnorm_lib, "rmsNorm") orelse return error.MetalFunctionNotFound;
        defer metal_release(rmsnorm_func);
        const rmsnorm_pipeline = metal_create_pipeline(device, rmsnorm_func) orelse {
            std.log.err("Metal rmsNorm pipeline failed: {s}", .{metal_get_last_error()});
            return error.MetalPipelineFailed;
        };

        return .{
            .device = device,
            .queue = queue,
            .matvec_pipeline = matvec_pipeline,
            .rmsnorm_pipeline = rmsnorm_pipeline,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        metal_release(self.matvec_pipeline);
        metal_release(self.rmsnorm_pipeline);
        metal_release(self.queue);
        metal_release(self.device);
    }

    /// GPU 矩阵-向量乘法
    pub fn matVecMul(self: *Self, out: []f32, mat: []const f32, vec: []const f32, out_dim: u32, in_dim: u32) void {
        std.debug.assert(out.len >= out_dim);
        std.debug.assert(mat.len >= out_dim * in_dim);
        std.debug.assert(vec.len >= in_dim);

        const mat_buf = metal_new_buffer(self.device, mat.ptr, mat.len * @sizeOf(f32)) orelse return;
        defer metal_release(mat_buf);
        const vec_buf = metal_new_buffer(self.device, vec.ptr, vec.len * @sizeOf(f32)) orelse return;
        defer metal_release(vec_buf);
        const out_buf = metal_new_buffer(self.device, out.ptr, out.len * @sizeOf(f32)) orelse return;
        defer metal_release(out_buf);

        const cmdbuf = metal_command_buffer(self.queue) orelse return;
        defer metal_release(cmdbuf);
        const encoder = metal_compute_encoder(cmdbuf) orelse return;

        metal_set_pipeline(encoder, self.matvec_pipeline);
        metal_set_buffer(encoder, out_buf, 0, 0);
        metal_set_buffer(encoder, mat_buf, 0, 1);
        metal_set_buffer(encoder, vec_buf, 0, 2);
        metal_set_bytes(encoder, &out_dim, @sizeOf(u32), 3);
        metal_set_bytes(encoder, &in_dim, @sizeOf(u32), 4);

        const threads_per_group: usize = 256;
        const groups = (out_dim + threads_per_group - 1) / threads_per_group;
        metal_dispatch(encoder, groups, 1, 1, threads_per_group, 1, 1);
        metal_end_encoding(encoder);
        metal_commit(cmdbuf);
        metal_wait_completed(cmdbuf);

        const result_ptr: [*]f32 = @ptrCast(@alignCast(metal_buffer_contents(out_buf)));
        @memcpy(out[0..out_dim], result_ptr[0..out_dim]);
    }

    /// GPU RMSNorm
    pub fn rmsNorm(self: *Self, out: []f32, x: []const f32, weight: []const f32, n: u32, eps: f32) void {
        std.debug.assert(out.len >= n);
        std.debug.assert(x.len >= n);
        std.debug.assert(weight.len >= n);

        const x_buf = metal_new_buffer(self.device, x.ptr, x.len * @sizeOf(f32)) orelse return;
        defer metal_release(x_buf);
        const w_buf = metal_new_buffer(self.device, weight.ptr, weight.len * @sizeOf(f32)) orelse return;
        defer metal_release(w_buf);
        const out_buf = metal_new_buffer(self.device, out.ptr, out.len * @sizeOf(f32)) orelse return;
        defer metal_release(out_buf);

        const cmdbuf = metal_command_buffer(self.queue) orelse return;
        defer metal_release(cmdbuf);
        const encoder = metal_compute_encoder(cmdbuf) orelse return;

        metal_set_pipeline(encoder, self.rmsnorm_pipeline);
        metal_set_buffer(encoder, out_buf, 0, 0);
        metal_set_buffer(encoder, x_buf, 0, 1);
        metal_set_buffer(encoder, w_buf, 0, 2);
        metal_set_bytes(encoder, &n, @sizeOf(u32), 3);
        metal_set_bytes(encoder, &eps, @sizeOf(f32), 4);

        const threads_per_group: usize = 256;
        const groups = (n + threads_per_group - 1) / threads_per_group;
        metal_dispatch(encoder, groups, 1, 1, threads_per_group, 1, 1);
        metal_end_encoding(encoder);
        metal_commit(cmdbuf);
        metal_wait_completed(cmdbuf);

        const result_ptr: [*]f32 = @ptrCast(@alignCast(metal_buffer_contents(out_buf)));
        @memcpy(out[0..n], result_ptr[0..n]);
    }
};

// ========== TDD: GPU 数值正确性验证（与 CPU 参考路径逐元素对比） ==========

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

test "Metal matVecMul 数值正确性（vs CPU 参考）" {
    if (!isAvailable()) {
        std.debug.print("跳过：当前平台不支持 Metal\n", .{});
        return;
    }
    const allocator = std.testing.allocator;
    var backend = MetalBackend.init(allocator) catch |err| {
        if (err == error.MetalDeviceNotFound) {
            std.debug.print("跳过：当前平台无 Metal 设备\n", .{});
            return;
        }
        return err;
    };
    defer backend.deinit();

    const out_dim: u32 = 64;
    const in_dim: u32 = 128;

    const mat = try allocator.alloc(f32, out_dim * in_dim);
    defer allocator.free(mat);
    const vec = try allocator.alloc(f32, in_dim);
    defer allocator.free(vec);
    const gpu_out = try allocator.alloc(f32, out_dim);
    defer allocator.free(gpu_out);
    const cpu_out = try allocator.alloc(f32, out_dim);
    defer allocator.free(cpu_out);

    // 填充确定性数据
    for (mat, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 7)) * 0.1;
    for (vec, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 5)) * 0.2;

    // CPU 参考
    math.matVecMul(cpu_out, mat, vec, out_dim, in_dim);

    // GPU 计算
    backend.matVecMul(gpu_out, mat, vec, out_dim, in_dim);

    // 对比（RMSE 容忍度 1e-4）
    try expectApproxEqSlice(cpu_out, gpu_out, 1e-4);
}

test "Metal rmsNorm 数值正确性（vs CPU 参考）" {
    if (!isAvailable()) {
        std.debug.print("跳过：当前平台不支持 Metal\n", .{});
        return;
    }
    const allocator = std.testing.allocator;
    var backend = MetalBackend.init(allocator) catch |err| {
        if (err == error.MetalDeviceNotFound) {
            std.debug.print("跳过：当前平台无 Metal 设备\n", .{});
            return;
        }
        return err;
    };
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
    backend.rmsNorm(gpu_out, x, weight, n, eps);

    try expectApproxEqSlice(cpu_out, gpu_out, 1e-4);
}
