const std = @import("std");

const MATRIX_SIZE: usize = 512;
const ITERATIONS: usize = 10;

fn matMul(A: []const f32, B: []const f32, C: []f32, n: usize) void {
    for (0..n) |i| {
        for (0..n) |j| {
            var sum: f32 = 0.0;
            for (0..n) |k| {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

fn getTimeNs() u64 {
    var ts: std.os.linux.timespec = undefined;
    _ = std.os.linux.clock_gettime(std.os.linux.CLOCK.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

pub fn main() !void {
    const allocator = std.heap.page_allocator;
    const n = MATRIX_SIZE;
    const total_elements = n * n;

    const A = try allocator.alloc(f32, total_elements);
    const B = try allocator.alloc(f32, total_elements);
    const C = try allocator.alloc(f32, total_elements);
    defer allocator.free(A);
    defer allocator.free(B);
    defer allocator.free(C);

    // 初始化随机矩阵
    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();
    for (A) |*a| a.* = rng.float(f32);
    for (B) |*b| b.* = rng.float(f32);

    // Warmup
    matMul(A, B, C, n);

    // Benchmark
    const start = getTimeNs();
    for (0..ITERATIONS) |_| {
        matMul(A, B, C, n);
    }
    const end = getTimeNs();

    const elapsed_ns = end - start;
    const elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0;
    const elapsed_s = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000_000.0;
    const total_flops = @as(f64, @floatFromInt(ITERATIONS)) * 2.0 * @as(f64, @floatFromInt(n * n * n));
    const gflops = total_flops / (elapsed_s * 1e9);

    std.debug.print("=== CPU Benchmark ===\n", .{});
    std.debug.print("Matrix size: {d}x{d}\n", .{ n, n });
    std.debug.print("Iterations: {d}\n", .{ITERATIONS});
    std.debug.print("Total time: {d:.2} ms\n", .{elapsed_ms});
    std.debug.print("Per iteration: {d:.2} ms\n", .{elapsed_ms / @as(f64, @floatFromInt(ITERATIONS))});
    std.debug.print("GFLOPS: {d:.2}\n", .{gflops});
}
