// 数学原语：RMSNorm、RoPE、SiLU、点积、Softmax
// 所有函数均为 CPU 参考实现，后续 GPU/NPU 内核需与此对齐验证

const std = @import("std");

/// RMSNorm: x_norm = x / sqrt(mean(x²) + eps) * weight
pub fn rmsNorm(out: []f32, x: []const f32, weight: []const f32, eps: f32) void {
    std.debug.assert(out.len == x.len);
    std.debug.assert(out.len == weight.len);

    var ss: f32 = 0;
    for (x) |v| {
        ss += v * v;
    }
    ss = ss / @as(f32, @floatFromInt(x.len)) + eps;
    const inv = 1.0 / @sqrt(ss);

    for (out, x, weight) |*o, v, w| {
        o.* = v * inv * w;
    }
}

/// SiLU 激活函数: x * sigmoid(x) = x / (1 + exp(-x))
pub fn silu(x: f32) f32 {
    return x / (1.0 + @exp(-x));
}

/// 逐元素 SiLU
pub fn siluInplace(x: []f32) void {
    for (x) |*v| {
        v.* = silu(v.*);
    }
}

/// 逐元素乘法: out = a * b
pub fn mul(out: []f32, a: []const f32, b: []const f32) void {
    for (out, a, b) |*o, va, vb| {
        o.* = va * vb;
    }
}

/// 向量点积
pub fn dot(a: []const f32, b: []const f32) f32 {
    std.debug.assert(a.len == b.len);
    var sum: f32 = 0;
    for (a, b) |va, vb| {
        sum += va * vb;
    }
    return sum;
}

/// Softmax: out[i] = exp(x[i]) / sum(exp(x))
pub fn softmax(out: []f32, x: []const f32) void {
    std.debug.assert(out.len == x.len);
    if (x.len == 0) return;

    // 数值稳定：减去最大值
    var max_val: f32 = x[0];
    for (x[1..]) |v| {
        if (v > max_val) max_val = v;
    }

    var sum: f32 = 0;
    for (out, x) |*o, v| {
        const e = @exp(v - max_val);
        o.* = e;
        sum += e;
    }

    const inv_sum = 1.0 / sum;
    for (out) |*o| {
        o.* *= inv_sum;
    }
}

/// Top-K：从 scores 中选出前 k 个最大值的索引
/// 返回排序后的 (index, score) 对，降序排列
pub const TopKEntry = struct { index: u32, score: f32 };

pub fn topK(
    allocator: std.mem.Allocator,
    scores: []const f32,
    k: usize,
) ![]TopKEntry {
    std.debug.assert(k <= scores.len);

    var entries = try allocator.alloc(TopKEntry, scores.len);
    defer allocator.free(entries);

    for (scores, 0..) |s, i| {
        entries[i] = .{ .index = @intCast(i), .score = s };
    }

    // Sort the ENTIRE array first (the actual top-k elements might be anywhere)
    std.sort.block(TopKEntry, entries, {}, struct {
        fn lessThan(_: void, a: TopKEntry, b: TopKEntry) bool {
            return a.score > b.score; // 降序
        }
    }.lessThan);

    const result = try allocator.dupe(TopKEntry, entries[0..k]);
    return result;
}

/// RoPE (Rotary Position Embedding) 预计算
/// 生成 cos/sin 表，用于 MLA 中 qk_rope_head_dim 维度的旋转位置编码
pub const RoPECache = struct {
    cos: []f32,
    sin: []f32,
    dim: u32, // rope_dimension_count

    pub fn init(allocator: std.mem.Allocator, max_seq_len: u32, dim: u32, freq_base: f32) !RoPECache {
        const cache = try allocator.alloc(f32, max_seq_len * dim);
        const sin_cache = try allocator.alloc(f32, max_seq_len * dim);

        for (0..max_seq_len) |pos| {
            for (0..dim) |i| {
                const freq = 1.0 / std.math.pow(f32, freq_base, @as(f32, @floatFromInt(2 * i)) / @as(f32, @floatFromInt(dim)));
                const theta = @as(f32, @floatFromInt(pos)) * freq;
                cache[pos * dim + i] = @cos(theta);
                sin_cache[pos * dim + i] = @sin(theta);
            }
        }

        return .{
            .cos = cache,
            .sin = sin_cache,
            .dim = dim,
        };
    }

    pub fn deinit(self: *RoPECache, allocator: std.mem.Allocator) void {
        allocator.free(self.cos);
        allocator.free(self.sin);
    }

    /// 对向量应用 RoPE（仅前 dim 维度）
    /// q: [head_dim]，前 rope_dim 维度应用旋转
    pub fn apply(self: *const RoPECache, q: []f32, pos: u32) void {
        const dim = self.dim;
        for (0..dim / 2) |i| {
            const cos_val = self.cos[pos * dim + i];
            const sin_val = self.sin[pos * dim + i];
            const q0 = q[i * 2];
            const q1 = q[i * 2 + 1];
            q[i * 2] = q0 * cos_val - q1 * sin_val;
            q[i * 2 + 1] = q0 * sin_val + q1 * cos_val;
        }
    }
};

// ========== 测试 ==========

test "rmsNorm basic" {
    const weight = [_]f32{ 1.0, 1.0, 1.0, 1.0 };
    const x = [_]f32{ 2.0, 2.0, 2.0, 2.0 };
    var out: [4]f32 = undefined;
    rmsNorm(&out, &x, &weight, 1e-5);

    // mean(x²) = 4, inv = 1/sqrt(4+eps) ≈ 0.5
    for (out) |v| {
        try std.testing.expectApproxEqAbs(@as(f32, 2.0) * 0.5, v, 1e-4);
    }
}

test "silu" {
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), silu(0.0), 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), silu(-100.0), 1e-2);
}

test "softmax" {
    const x = [_]f32{ 1.0, 2.0, 3.0 };
    var out: [3]f32 = undefined;
    softmax(&out, &x);

    var sum: f32 = 0;
    for (out) |v| sum += v;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum, 1e-5);
    // 最大值对应的概率最大
    try std.testing.expect(out[2] > out[1]);
    try std.testing.expect(out[1] > out[0]);
}

test "topK finds global top-k" {
    const scores = [_]f32{ 1.0, 5.0, 3.0, 10.0, 2.0, 8.0, 4.0, 7.0 };
    const result = try topK(std.testing.allocator, &scores, 3);
    defer std.testing.allocator.free(result);
    // Top 3 should be indices 3(10.0), 5(8.0), 7(7.0) in descending order
    try std.testing.expectEqual(@as(u32, 3), result[0].index);
    try std.testing.expectEqual(@as(u32, 5), result[1].index);
    try std.testing.expectEqual(@as(u32, 7), result[2].index);
}

test "dot product" {
    const a = [_]f32{ 1.0, 2.0, 3.0 };
    const b = [_]f32{ 4.0, 5.0, 6.0 };
    try std.testing.expectApproxEqAbs(@as(f32, 32.0), dot(&a, &b), 1e-4);
}
