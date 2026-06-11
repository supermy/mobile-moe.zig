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
                while (qi < 32 and col < in_dim) : ({ qi += 1; col += 1; }) {
                    const w = d * @as(f32, @floatFromInt(blk.qs[qi]));
                    sum_vec[0] += w * x[col];
                }
            }
            out[row] = @reduce(.Add, sum_vec);
        }
}

/// Q8_0 矩阵-矩阵乘法（用于 Prompt Prefill 批量 attention）
/// A: [M, K] Q8_0 blocks
/// B: [N, K] f32（B 转置后实际为 [K, N]）
/// C: [M, N] f32
/// TODO: 实现分块 GEMM，利用 L2 cache
pub fn matMulQ8_0(
    out: []f32,
    a_blocks: []const dequant.BlockQ8_0,
    b: []const f32,
    m: usize,
    n: usize,
    k: usize,
) void {
    _ = out;
    _ = a_blocks;
    _ = b;
    _ = m;
    _ = n;
    _ = k;
    // TODO: 分块 GEMM 实现
}

// ========== 测试 ==========

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
