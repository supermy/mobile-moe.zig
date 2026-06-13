// 量化矩阵乘法（Quantized Matrix Multiplication）
//
// 目标：直接消费 Q4_K/Q8_0/IQ2_XXS 等量化权重，避免反量化为 f32
// 当前：Q8_0 参考实现，其他类型待补充

const std = @import("std");
const dequant = @import("dequant.zig");

/// Q8_0 矩阵-向量乘法
/// blocks: [out_dim, blocks_per_row] BlockQ8_0，按行优先存储
/// x: [in_dim] 输入向量
/// out: [out_dim] 输出向量
pub fn matVecMulQ8_0(out: []f32, blocks: []const dequant.BlockQ8_0, x: []const f32, out_dim: usize, in_dim: usize) void {
    std.debug.assert(blocks.len >= out_dim * ((in_dim + 31) / 32));
    std.debug.assert(x.len >= in_dim);
    std.debug.assert(out.len >= out_dim);

    const blocks_per_row = (in_dim + 31) / 32;
    for (0..out_dim) |row| {
        const row_blocks = blocks[row * blocks_per_row ..][0..blocks_per_row];
        var sum_vec = @as(@Vector(4, f32), @splat(0.0));
        var col: usize = 0;
        for (row_blocks) |blk| {
            const d = dequant.f16ToF32(blk.d);

            // SIMD 内层 32 元素（分 8 组 @Vector(4, f32)）
            var qi: usize = 0;
            while (qi + 4 <= 32 and col + 4 <= in_dim) : (qi += 4) {
                const w_f: @Vector(4, f32) = .{
                    d * @as(f32, @floatFromInt(blk.qs[qi])),
                    d * @as(f32, @floatFromInt(blk.qs[qi + 1])),
                    d * @as(f32, @floatFromInt(blk.qs[qi + 2])),
                    d * @as(f32, @floatFromInt(blk.qs[qi + 3])),
                };
                const x_vec: @Vector(4, f32) = x[col..][0..4].*;
                sum_vec += w_f * x_vec;
                col += 4;
            }
            // 尾部处理
            while (qi < 32 and col < in_dim) : ({
                qi += 1;
                col += 1;
            }) {
                const w = d * @as(f32, @floatFromInt(blk.qs[qi]));
                sum_vec[0] += w * x[col];
            }
        }
        out[row] = @reduce(.Add, sum_vec);
    }
}

/// Q8_0 矩阵-矩阵乘法（用于 Prompt Prefill 批量 attention）
/// A: [M, K] Q8_0 blocks，按行优先存储
/// B: [N, K] f32 行优先（每行是一个输入向量）
/// C: [M, N] f32 行优先（每列对应一个输出维度，每行对应一个 batch 项）
/// 分块策略：对 M 和 N 分块，使 B 的子块常驻 L2 cache；
/// 内层权重向量复用于多个 N，减少 Q8_0 权重读取带宽。
pub fn matMulQ8_0(
    out: []f32,
    a_blocks: []const dequant.BlockQ8_0,
    b: []const f32,
    m: usize,
    n: usize,
    k: usize,
) void {
    std.debug.assert(a_blocks.len >= m * ((k + 31) / 32));
    std.debug.assert(b.len >= n * k);
    std.debug.assert(out.len >= m * n);

    const blocks_per_row = (k + 31) / 32;
    @memset(out, 0);

    // 分块尺寸：根据典型 L2 cache 大小（256-512KB）选择
    // M_BLOCK × K 的权重约 64 × 2048 × 34/32 ≈ 136KB
    // N_BLOCK × K 的 B 约 64 × 2048 × 4 ≈ 512KB
    const M_BLOCK: usize = 64;
    const N_BLOCK: usize = 64;

    var mi: usize = 0;
    while (mi < m) : (mi += M_BLOCK) {
        const m_end = @min(mi + M_BLOCK, m);
        var ni: usize = 0;
        while (ni < n) : (ni += N_BLOCK) {
            const n_end = @min(ni + N_BLOCK, n);

            for (mi..m_end) |mi_| {
                const row_blocks = a_blocks[mi_ * blocks_per_row ..][0..blocks_per_row];
                var col: usize = 0;
                for (row_blocks) |blk| {
                    const d = dequant.f16ToF32(blk.d);
                    var qi: usize = 0;
                    // SIMD 内层：4-wide 权重 × 4-wide B 元素
                    while (qi + 4 <= 32 and col + 4 <= k) : (qi += 4) {
                        const w_vec = @Vector(4, f32){
                            d * @as(f32, @floatFromInt(blk.qs[qi])),
                            d * @as(f32, @floatFromInt(blk.qs[qi + 1])),
                            d * @as(f32, @floatFromInt(blk.qs[qi + 2])),
                            d * @as(f32, @floatFromInt(blk.qs[qi + 3])),
                        };
                        var ni_: usize = ni;
                        while (ni_ + 4 <= n_end) : (ni_ += 4) {
                            // 加载 4 个 B 行的 4 个元素
                            const b0 = b[(ni_ + 0) * k + col ..];
                            const b1 = b[(ni_ + 1) * k + col ..];
                            const b2 = b[(ni_ + 2) * k + col ..];
                            const b3 = b[(ni_ + 3) * k + col ..];
                            const b_vec0: @Vector(4, f32) = b0[0..4].*;
                            const b_vec1: @Vector(4, f32) = b1[0..4].*;
                            const b_vec2: @Vector(4, f32) = b2[0..4].*;
                            const b_vec3: @Vector(4, f32) = b3[0..4].*;

                            out[mi_ * n + ni_ + 0] += @reduce(.Add, w_vec * b_vec0);
                            out[mi_ * n + ni_ + 1] += @reduce(.Add, w_vec * b_vec1);
                            out[mi_ * n + ni_ + 2] += @reduce(.Add, w_vec * b_vec2);
                            out[mi_ * n + ni_ + 3] += @reduce(.Add, w_vec * b_vec3);
                        }
                        while (ni_ < n_end) : (ni_ += 1) {
                            const b_row = b[ni_ * k + col ..];
                            const b_vec: @Vector(4, f32) = b_row[0..4].*;
                            out[mi_ * n + ni_] += @reduce(.Add, w_vec * b_vec);
                        }
                        col += 4;
                    }
                    // 尾部处理
                    while (qi < 32 and col < k) : ({
                        qi += 1;
                        col += 1;
                    }) {
                        const w = d * @as(f32, @floatFromInt(blk.qs[qi]));
                        for (ni..n_end) |ni_| {
                            out[mi_ * n + ni_] += w * b[ni_ * k + col];
                        }
                    }
                }
            }
        }
    }
}

// ========== 测试 ==========

test "matMulQ8_0 batch matches f32 reference" {
    const m: usize = 4;
    const n: usize = 8;
    const k: usize = 64;
    const blocks_per_row = (k + 31) / 32;
    const n_blocks = m * blocks_per_row;

    var blocks = try std.testing.allocator.alloc(dequant.BlockQ8_0, n_blocks);
    defer std.testing.allocator.free(blocks);

    // 构造简单权重：所有值为 1.0
    for (0..n_blocks) |i| {
        blocks[i].d = dequant.f32ToF16(1.0 / 127.0);
        for (0..32) |j| {
            blocks[i].qs[j] = 127;
        }
    }

    // B: [n, k] 行优先，所有值为 1.0
    const b = try std.testing.allocator.alloc(f32, n * k);
    defer std.testing.allocator.free(b);
    @memset(b, 1.0);

    const out = try std.testing.allocator.alloc(f32, m * n);
    defer std.testing.allocator.free(out);
    matMulQ8_0(out, blocks, b, m, n, k);

    // 每个输出应为 64 * 1.0 = 64.0
    for (out) |v| {
        try std.testing.expectApproxEqAbs(@as(f32, 64.0), v, 0.1);
    }
}

test "matMulQ8_0 vs repeated matVecMulQ8_0" {
    const m: usize = 3;
    const n: usize = 5;
    const k: usize = 64;
    const blocks_per_row = (k + 31) / 32;
    const n_blocks = m * blocks_per_row;

    var blocks = try std.testing.allocator.alloc(dequant.BlockQ8_0, n_blocks);
    defer std.testing.allocator.free(blocks);

    // 构造随机-ish 权重
    for (0..n_blocks) |i| {
        blocks[i].d = dequant.f32ToF16(0.5 / 127.0);
        for (0..32) |j| {
            blocks[i].qs[j] = @intCast((i * 32 + j) % 127);
        }
    }

    var b = try std.testing.allocator.alloc(f32, n * k);
    defer std.testing.allocator.free(b);
    for (b, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(i % 7)) * 0.3;
    }

    const out_batch = try std.testing.allocator.alloc(f32, m * n);
    defer std.testing.allocator.free(out_batch);
    matMulQ8_0(out_batch, blocks, b, m, n, k);

    // 逐列验证与 matVecMulQ8_0 一致
    for (0..n) |ni| {
        var expected: [3]f32 = undefined;
        matVecMulQ8_0(&expected, blocks, b[ni * k ..], m, k);
        for (0..m) |mi| {
            try std.testing.expectApproxEqAbs(expected[mi], out_batch[mi * n + ni], 0.01);
        }
    }
}

test "matVecMulQ8_0 matches f32 reference" {
    const in_dim: usize = 64;
    const out_dim: usize = 4;
    const n_blocks = out_dim * (in_dim / 32);

    var blocks = try std.testing.allocator.alloc(dequant.BlockQ8_0, n_blocks);
    defer std.testing.allocator.free(blocks);

    // 构造简单权重：所有值为 1.0
    for (0..n_blocks) |i| {
        blocks[i].d = dequant.f32ToF16(1.0 / 127.0);
        for (0..32) |j| {
            blocks[i].qs[j] = 127;
        }
    }

    const x = [_]f32{1.0} ** in_dim;
    var out: [out_dim]f32 = undefined;
    matVecMulQ8_0(&out, blocks, &x, out_dim, in_dim);

    // 每个输出应为 64 * 1.0 = 64.0（因为 127 * (1/127) = 1.0）
    // 容差放宽到 0.1：f16 scale 转换引入约 0.006% 相对误差，64 元素累加后约 ±0.05
    for (out) |v| {
        try std.testing.expectApproxEqAbs(@as(f32, 64.0), v, 0.1);
    }
}
