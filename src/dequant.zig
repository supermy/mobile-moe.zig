// IQ2_XXS 反量化内核
//
// IQ2_XXS 格式：每 256 个元素为一个超级块，占 66 字节
//   struct block_iq2_xxs {
//       ggml_half d;          // 2 字节 - 全局缩放因子 (FP16)
//       uint16_t qs[32];      // 64 字节 - 量化索引 (8 子组 × 4 uint16)
//   };
//
// 解码结构（参考 llama.cpp ggml-quants.c dequantize_row_iq2_xxs）：
//   - 256 元素分为 8 个子组（ib32），每个子组 32 元素
//   - 每个子组由 4 个 uint16_t（8 字节）编码
//   - 前 4 字节（aux32[0]）：4 个 grid 索引，每个 8 位
//   - 后 4 字节（aux32[1]）：高 4 位 = level shift，低 28 位 = 4 × 7-bit sign 索引
//   - 最终值 = d * (0.5 + ls) * 0.25 * grid_byte * sign

const std = @import("std");

// FP16 → F32 转换（IEEE 754 位构造）
pub fn f16ToF32(h: u16) f32 {
    const sign: u32 = @as(u32, h >> 15);
    const exp_raw: u32 = @as(u32, (h >> 10) & 0x1F);
    const mant: u32 = @as(u32, h & 0x3FF);

    if (exp_raw == 0) {
        // Subnormal or zero
        if (mant == 0) return @bitCast(@as(u32, sign << 31));
        // Subnormal: normalize
        var m = mant;
        var e: u32 = 0;
        while (m & 0x400 == 0) : (e += 1) {
            m <<= 1;
        }
        m &= 0x3FF; // Remove implicit bit
        const f32_exp = 127 - 14 - e; // Subnormal f16 exponent is -14 (1 - bias), not -15
        const f32_bits = (sign << 31) | (f32_exp << 23) | (m << 13);
        return @bitCast(f32_bits);
    }
    if (exp_raw == 31) {
        // Inf or NaN
        if (mant == 0) return @bitCast(@as(u32, (sign << 31) | 0x7F800000));
        return @bitCast(@as(u32, (sign << 31) | 0x7F800000 | (mant << 13)));
    }
    // Normalized
    const f32_exp: u32 = @intCast(@as(i32, @intCast(exp_raw)) - 15 + 127);
    const f32_bits = (sign << 31) | (f32_exp << 23) | (mant << 13);
    return @bitCast(f32_bits);
}

/// F32 → FP16 转换（IEEE 754 舍入到最近偶数）
pub fn f32ToF16(v: f32) u16 {
    const bits: u32 = @bitCast(v);
    const sign: u32 = (bits >> 31) & 0x1;
    const exp_raw: u32 = (bits >> 23) & 0xFF;
    const mant: u32 = bits & 0x7FFFFF;

    var e: u32 = undefined;
    var m: u32 = undefined;

    if (exp_raw == 0) {
        e = 0;
        m = 0;
    } else if (exp_raw == 0xFF) {
        e = 0x1F;
        m = mant >> 13;
        if (mant != 0 and m == 0) m = 1;
    } else {
        const exp16 = @as(i32, @intCast(exp_raw)) - 127 + 15;
        if (exp16 >= 31) {
            e = 0x1F;
            m = 0;
        } else if (exp16 <= 0) {
            if (exp16 < -10) {
                e = 0;
                m = 0;
            } else {
                e = 0;
                m = (mant | 0x800000) >> @as(u5, @intCast(1 - exp16));
            }
        } else {
            e = @intCast(exp16);
            m = mant >> 13;
            const round_bits = mant & 0x1FFF;
            const half = 0x1000;
            if (round_bits > half or (round_bits == half and (m & 1) == 1)) {
                m += 1;
                if (m == 0x400) {
                    m = 0;
                    e += 1;
                }
            }
        }
    }

    return @intCast((sign << 15) | (e << 10) | m);
}

/// 将 f32 数组量化为 Q8_0 格式
pub fn quantizeQ8_0(allocator: std.mem.Allocator, data: []const f32) ![]BlockQ8_0 {
    const n_elements = data.len;
    const n_blocks = (n_elements + 31) / 32;
    const blocks = try allocator.alloc(BlockQ8_0, n_blocks);
    errdefer allocator.free(blocks);

    for (0..n_blocks) |i| {
        var max_abs: f32 = 0;
        for (0..32) |j| {
            const idx = i * 32 + j;
            if (idx < n_elements) {
                const v = @abs(data[idx]);
                if (v > max_abs) max_abs = v;
            }
        }
        const d = if (max_abs > 0) max_abs / 127.0 else 0.0;
        blocks[i].d = f32ToF16(d);
        for (0..32) |j| {
            const idx = i * 32 + j;
            if (idx < n_elements) {
                if (d > 0) {
                    const q = @round(data[idx] / d);
                    blocks[i].qs[j] = @intCast(std.math.clamp(q, -127, 127));
                } else {
                    blocks[i].qs[j] = 0;
                }
            } else {
                blocks[i].qs[j] = 0;
            }
        }
    }
    return blocks;
}

/// 将 Q8_0 块数组反量化为 f32（n_elements 可以小于 blocks.len * 32）
pub fn dequantizeQ8_0Slice(allocator: std.mem.Allocator, blocks: []const BlockQ8_0, n_elements: usize) ![]f32 {
    const out = try allocator.alloc(f32, n_elements);
    for (0..n_elements) |idx| {
        const block_idx = idx / 32;
        const q_idx = idx % 32;
        const d = f16ToF32(blocks[block_idx].d);
        out[idx] = d * @as(f32, @floatFromInt(blocks[block_idx].qs[q_idx]));
    }
    return out;
}

// IQ2_XXS 查找表 (256 × 8 packed bytes)
// 每项是一个 uint64_t，包含 8 个字节，每个字节代表一个量化网格值
// 字节值只有三种：0x08=8, 0x19=25, 0x2b=43
// 来源：llama.cpp ggml/src/ggml-common.h 中的 iq2xxs_grid
pub const iq2xxs_grid = [256]u64{
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x08080808082b0808,
    0x08080808082b082b, 0x08080808082b2b08, 0x08080808082b2b2b, 0x0808080819080819,
    0x0808080819081908, 0x0808080819190808, 0x0808080819192b08, 0x08080808192b0819,
    0x08080808192b1908, 0x080808082b080808, 0x080808082b08082b, 0x080808082b082b2b,
    0x080808082b2b082b, 0x0808081908080819, 0x0808081908081908, 0x0808081908190808,
    0x0808081908191919, 0x0808081919080808, 0x080808192b081908, 0x080808192b192b08,
    0x0808082b08080808, 0x0808082b0808082b, 0x0808082b082b082b, 0x0808082b2b08082b,
    0x0808190808080819, 0x0808190808081908, 0x0808190808190808, 0x08081908082b0819,
    0x08081908082b1908, 0x0808190819080808, 0x080819081908082b, 0x0808190819082b08,
    0x08081908192b0808, 0x080819082b080819, 0x080819082b081908, 0x080819082b190808,
    0x080819082b2b1908, 0x0808191908080808, 0x080819190808082b, 0x0808191908082b08,
    0x08081919082b0808, 0x080819191908192b, 0x08081919192b2b19, 0x080819192b080808,
    0x080819192b190819, 0x0808192b08082b19, 0x0808192b08190808, 0x0808192b19080808,
    0x0808192b2b081908, 0x0808192b2b2b1908, 0x08082b0808080808, 0x08082b0808081919,
    0x08082b0808082b08, 0x08082b0808191908, 0x08082b08082b2b08, 0x08082b0819080819,
    0x08082b0819081908, 0x08082b0819190808, 0x08082b081919082b, 0x08082b082b082b08,
    0x08082b1908081908, 0x08082b1919080808, 0x08082b2b0808082b, 0x08082b2b08191908,
    0x0819080808080819, 0x0819080808081908, 0x0819080808190808, 0x08190808082b0819,
    0x0819080819080808, 0x08190808192b0808, 0x081908082b081908, 0x081908082b190808,
    0x081908082b191919, 0x0819081908080808, 0x0819081908082b08, 0x08190819082b0808,
    0x0819081919190808, 0x0819081919192b2b, 0x081908192b080808, 0x0819082b082b1908,
    0x0819082b19081919, 0x0819190808080808, 0x0819190808082b08, 0x08191908082b0808,
    0x08191908082b1919, 0x0819190819082b19, 0x081919082b080808, 0x0819191908192b08,
    0x08191919192b082b, 0x0819192b08080808, 0x0819192b0819192b, 0x08192b0808080819,
    0x08192b0808081908, 0x08192b0808190808, 0x08192b0819080808, 0x08192b082b080819,
    0x08192b1908080808, 0x08192b1908081919, 0x08192b192b2b0808, 0x08192b2b19190819,
    0x082b080808080808, 0x082b08080808082b, 0x082b080808082b2b, 0x082b080819081908,
    0x082b0808192b0819, 0x082b08082b080808, 0x082b08082b08082b, 0x082b0819082b2b19,
    0x082b081919082b08, 0x082b082b08080808, 0x082b082b0808082b, 0x082b190808080819,
    0x082b190808081908, 0x082b190808190808, 0x082b190819080808, 0x082b19081919192b,
    0x082b191908080808, 0x082b191919080819, 0x082b1919192b1908, 0x082b192b2b190808,
    0x082b2b0808082b08, 0x082b2b08082b0808, 0x082b2b082b191908, 0x082b2b2b19081908,
    0x1908080808080819, 0x1908080808081908, 0x1908080808190808, 0x1908080808192b08,
    0x19080808082b0819, 0x19080808082b1908, 0x1908080819080808, 0x1908080819082b08,
    0x190808081919192b, 0x19080808192b0808, 0x190808082b080819, 0x190808082b081908,
    0x190808082b190808, 0x1908081908080808, 0x19080819082b0808, 0x19080819192b0819,
    0x190808192b080808, 0x190808192b081919, 0x1908082b08080819, 0x1908082b08190808,
    0x1908082b19082b08, 0x1908082b1919192b, 0x1908082b192b2b08, 0x1908190808080808,
    0x1908190808082b08, 0x19081908082b0808, 0x190819082b080808, 0x190819082b192b19,
    0x190819190819082b, 0x19081919082b1908, 0x1908192b08080808, 0x19082b0808080819,
    0x19082b0808081908, 0x19082b0808190808, 0x19082b0819080808, 0x19082b0819081919,
    0x19082b1908080808, 0x19082b1919192b08, 0x19082b19192b0819, 0x19082b192b08082b,
    0x19082b2b19081919, 0x19082b2b2b190808, 0x1919080808080808, 0x1919080808082b08,
    0x1919080808190819, 0x1919080808192b19, 0x19190808082b0808, 0x191908082b080808,
    0x191908082b082b08, 0x1919081908081908, 0x191908191908082b, 0x191908192b2b1908,
    0x1919082b2b190819, 0x191919082b190808, 0x191919082b19082b, 0x1919191908082b2b,
    0x1919192b08080819, 0x1919192b19191908, 0x19192b0808080808, 0x19192b0808190819,
    0x19192b0808192b19, 0x19192b08192b1908, 0x19192b1919080808, 0x19192b2b08082b08,
    0x192b080808081908, 0x192b080808190808, 0x192b080819080808, 0x192b0808192b2b08,
    0x192b081908080808, 0x192b081919191919, 0x192b082b08192b08, 0x192b082b192b0808,
    0x192b190808080808, 0x192b190808081919, 0x192b191908190808, 0x192b19190819082b,
    0x192b19192b081908, 0x192b2b081908082b, 0x2b08080808080808, 0x2b0808080808082b,
    0x2b08080808082b2b, 0x2b08080819080819, 0x2b0808082b08082b, 0x2b08081908081908,
    0x2b08081908192b08, 0x2b08081919080808, 0x2b08082b08190819, 0x2b08190808080819,
    0x2b08190808081908, 0x2b08190808190808, 0x2b08190808191919, 0x2b08190819080808,
    0x2b081908192b0808, 0x2b08191908080808, 0x2b0819191908192b, 0x2b0819192b191908,
    0x2b08192b08082b19, 0x2b08192b19080808, 0x2b08192b192b0808, 0x2b082b080808082b,
    0x2b082b1908081908, 0x2b082b2b08190819, 0x2b19080808081908, 0x2b19080808190808,
    0x2b190808082b1908, 0x2b19080819080808, 0x2b1908082b2b0819, 0x2b1908190819192b,
    0x2b1908192b080808, 0x2b19082b19081919, 0x2b19190808080808, 0x2b191908082b082b,
    0x2b19190819081908, 0x2b19191919190819, 0x2b192b082b080819, 0x2b192b19082b0808,
    0x2b2b08080808082b, 0x2b2b080819190808, 0x2b2b08082b081919, 0x2b2b081908082b19,
    0x2b2b082b08080808, 0x2b2b190808192b08, 0x2b2b2b0819190808, 0x2b2b2b1908081908,
};

// 位掩码查找表：用于 IQ2_XS 系列解码
pub const kmask_iq2xs = [8]u8{ 1, 2, 4, 8, 16, 32, 64, 128 };

// IQ4_NL / IQ4_XS 查找表：4-bit 索引 → 非线性 int8 值
pub const kvalues_iq4nl = [16]i8{
    -127, -104, -83, -65, -49, -35, -22, -10,
    1,    13,   25,  38,  53,  69,  89,  113,
};

// IQ3_S 查找表：512 项，每项打包 4 个 int8 值
pub const iq3s_grid = [512]u32{
    0x01010101, 0x01010103, 0x01010105, 0x0101010b, 0x0101010f, 0x01010301, 0x01010303, 0x01010305,
    0x01010309, 0x0101030d, 0x01010501, 0x01010503, 0x0101050b, 0x01010707, 0x01010901, 0x01010905,
    0x0101090b, 0x0101090f, 0x01010b03, 0x01010b07, 0x01010d01, 0x01010d05, 0x01010f03, 0x01010f09,
    0x01010f0f, 0x03010101, 0x03010103, 0x03010105, 0x03010109, 0x03010301, 0x03010303, 0x0301030b,
    0x03010501, 0x03010507, 0x0301050f, 0x03010703, 0x0301070b, 0x03010909, 0x03010d03, 0x03010d0b,
    0x03010f05, 0x05010101, 0x05010103, 0x0501010b, 0x0501010f, 0x05010301, 0x05010307, 0x0501030d,
    0x05050503, 0x0505050b, 0x05050701, 0x05050709, 0x05050905, 0x0505090b, 0x0505090f, 0x05050b03,
    0x05050b07, 0x05050f01, 0x05050f07, 0x07010107, 0x07010303, 0x0701030b, 0x07010501, 0x07010505,
    0x07070703, 0x07070707, 0x0707070d, 0x07070909, 0x07070b01, 0x07070b05, 0x07070d0f, 0x07070f03,
    0x07070f0b, 0x09010101, 0x09010307, 0x0901030f, 0x09010503, 0x09010509, 0x09010705, 0x09090901,
    0x09090907, 0x09090b03, 0x09090f01, 0x0b010105, 0x0b010109, 0x0b050501, 0x0b050505, 0x0b05050d,
    0x0b070707, 0x0b090903, 0x0b09090b, 0x0b09090f, 0x0b0d0d0d, 0x0b0f0707, 0x0d01010d, 0x0d030303,
    0x0d030307, 0x0d070703, 0x0d0b0b05, 0x0d0f0303, 0x0f010101, 0x0f010105, 0x0f010109, 0x0f050501,
    0x0f050505, 0x0f05050d, 0x0f070707, 0x0f0b0b01, 0x0f0b0b09, 0x03010101, 0x03010103, 0x03010105,
    0x03010109, 0x03010301, 0x03010303, 0x03010307, 0x0301030b, 0x0301030f, 0x03010501, 0x03010505,
    0x03010703, 0x03010709, 0x0301070d, 0x03010b09, 0x03010b0d, 0x03010d03, 0x03010f05, 0x03030101,
    0x03030103, 0x03030107, 0x0303010d, 0x03030301, 0x03030309, 0x03030503, 0x03030701, 0x03030707,
    0x03030903, 0x03030b01, 0x03030b05, 0x03030f01, 0x03030f0d, 0x03050101, 0x03050305, 0x0305030b,
    0x0305030f, 0x03050501, 0x03050509, 0x03050705, 0x03050901, 0x03050907, 0x03050b0b, 0x03050d01,
    0x03050f05, 0x03070103, 0x03070109, 0x0307010f, 0x03070301, 0x03070307, 0x03070503, 0x0307050f,
    0x03070701, 0x03070709, 0x03070903, 0x03070d05, 0x03070f01, 0x03090107, 0x0309010b, 0x03090305,
    0x03090309, 0x03090703, 0x03090707, 0x03090905, 0x0309090d, 0x03090b01, 0x03090b09, 0x030b0103,
    0x030b0301, 0x030b0307, 0x030b0503, 0x030b0701, 0x030b0705, 0x030b0b03, 0x030d0501, 0x030d0509,
    0x030d050f, 0x030d0909, 0x030d090d, 0x030f0103, 0x030f0107, 0x030f0301, 0x030f0305, 0x030f0503,
    0x030f070b, 0x030f0903, 0x030f0d05, 0x030f0f01, 0x05010101, 0x05010103, 0x05010107, 0x0501010b,
    0x0501010f, 0x05010301, 0x05010305, 0x05010309, 0x0501030d, 0x05010503, 0x05010507, 0x0501050f,
    0x05010701, 0x05010705, 0x05010903, 0x05010907, 0x0501090b, 0x05010b01, 0x05010b05, 0x05010d0f,
    0x05010f01, 0x05010f07, 0x05010f0b, 0x05030101, 0x05030105, 0x05030301, 0x05030307, 0x0503030f,
    0x05030505, 0x0503050b, 0x05030703, 0x05030709, 0x05030905, 0x05030b03, 0x05050103, 0x05050109,
    0x0505010f, 0x05050503, 0x05050507, 0x05050701, 0x0505070f, 0x05050903, 0x05050b07, 0x05050b0f,
    0x05050f03, 0x05050f09, 0x05070101, 0x05070105, 0x0507010b, 0x05070303, 0x05070505, 0x05070509,
    0x05070703, 0x05070707, 0x05070905, 0x05070b01, 0x05070d0d, 0x05090103, 0x0509010f, 0x05090501,
    0x05090507, 0x05090705, 0x0509070b, 0x05090903, 0x05090f05, 0x05090f0b, 0x050b0109, 0x050b0303,
    0x050b0505, 0x050b070f, 0x050b0901, 0x050b0b07, 0x050b0f01, 0x050d0101, 0x050d0105, 0x050d010f,
    0x050d0503, 0x050d0b0b, 0x050d0d03, 0x050f010b, 0x050f0303, 0x050f050d, 0x050f0701, 0x050f0907,
    0x050f0b01, 0x07010105, 0x07010303, 0x07010307, 0x0701030b, 0x0701030f, 0x07010505, 0x07010703,
    0x07010707, 0x0701070b, 0x07010905, 0x07010909, 0x0701090f, 0x07010b03, 0x07010d07, 0x07010f03,
    0x07030103, 0x07030107, 0x0703010b, 0x07030309, 0x07030503, 0x07030507, 0x07030901, 0x07030d01,
    0x07030f05, 0x07030f0d, 0x07050101, 0x07050305, 0x07050501, 0x07050705, 0x07050709, 0x07050b01,
    0x07070103, 0x07070301, 0x07070309, 0x07070503, 0x07070507, 0x0707050f, 0x07070701, 0x07070903,
    0x07070907, 0x0707090f, 0x07070b0b, 0x07070f07, 0x07090107, 0x07090303, 0x0709030d, 0x07090505,
    0x07090703, 0x07090b05, 0x07090d01, 0x07090d09, 0x070b0103, 0x070b0301, 0x070b0305, 0x070b050b,
    0x070b0705, 0x070b0909, 0x070b0b0d, 0x070b0f07, 0x070d030d, 0x070d0903, 0x070f0103, 0x070f0107,
    0x070f0501, 0x070f0505, 0x070f070b, 0x09010101, 0x09010109, 0x09010305, 0x09010501, 0x09010509,
    0x0901050f, 0x09010705, 0x09010903, 0x09010b01, 0x09010f01, 0x09030105, 0x0903010f, 0x09030303,
    0x09030307, 0x09030505, 0x09030701, 0x0903070b, 0x09030907, 0x09030b03, 0x09030b0b, 0x09050103,
    0x09050107, 0x09050301, 0x0905030b, 0x09050503, 0x09050707, 0x09050901, 0x09050b0f, 0x09050d05,
    0x09050f01, 0x09070109, 0x09070303, 0x09070307, 0x09070501, 0x09070505, 0x09070703, 0x0907070b,
    0x09090101, 0x09090105, 0x09090509, 0x0909070f, 0x09090901, 0x09090f03, 0x090b010b, 0x090b010f,
    0x090b0503, 0x090b0d05, 0x090d0307, 0x090d0709, 0x090d0d01, 0x090f0301, 0x090f030b, 0x090f0701,
    0x090f0907, 0x090f0b03, 0x0b010105, 0x0b010301, 0x0b010309, 0x0b010505, 0x0b010901, 0x0b010909,
    0x0b01090f, 0x0b010b05, 0x0b010d0d, 0x0b010f09, 0x0b030103, 0x0b030107, 0x0b03010b, 0x0b030305,
    0x0b030503, 0x0b030705, 0x0b030f05, 0x0b050101, 0x0b050303, 0x0b050507, 0x0b050701, 0x0b05070d,
    0x0b050b07, 0x0b070105, 0x0b07010f, 0x0b070301, 0x0b07050f, 0x0b070909, 0x0b070b03, 0x0b070d0b,
    0x0b070f07, 0x0b090103, 0x0b090109, 0x0b090501, 0x0b090705, 0x0b09090d, 0x0b0b0305, 0x0b0b050d,
    0x0b0b0b03, 0x0b0b0b07, 0x0b0d0905, 0x0b0f0105, 0x0b0f0109, 0x0b0f0505, 0x0d010303, 0x0d010307,
    0x0d01030b, 0x0d010703, 0x0d010707, 0x0d010d01, 0x0d030101, 0x0d030501, 0x0d03050f, 0x0d030d09,
    0x0d050305, 0x0d050709, 0x0d050905, 0x0d050b0b, 0x0d050d05, 0x0d050f01, 0x0d070101, 0x0d070309,
    0x0d070503, 0x0d070901, 0x0d09050b, 0x0d090907, 0x0d090d05, 0x0d0b0101, 0x0d0b0107, 0x0d0b0709,
    0x0d0b0d01, 0x0d0d010b, 0x0d0d0901, 0x0d0f0303, 0x0d0f0307, 0x0f010101, 0x0f010109, 0x0f01010f,
    0x0f010501, 0x0f010505, 0x0f01070d, 0x0f010901, 0x0f010b09, 0x0f010d05, 0x0f030105, 0x0f030303,
    0x0f030509, 0x0f030907, 0x0f03090b, 0x0f050103, 0x0f050109, 0x0f050301, 0x0f05030d, 0x0f050503,
    0x0f050701, 0x0f050b03, 0x0f070105, 0x0f070705, 0x0f07070b, 0x0f070b07, 0x0f090103, 0x0f09010b,
    0x0f090307, 0x0f090501, 0x0f090b01, 0x0f0b0505, 0x0f0b0905, 0x0f0d0105, 0x0f0d0703, 0x0f0f0101,
};

// 符号查找表：从 uint16_t 索引提取 8 个符号位
pub const ksigns_iq2xs = [128]u8{
    0,   129, 130, 3,   132, 5,   6,   135, 136, 9,   10,  139, 12,  141, 142, 15,
    144, 17,  18,  147, 20,  149, 150, 23,  24,  153, 154, 27,  156, 29,  30,  159,
    160, 33,  34,  163, 36,  165, 166, 39,  40,  169, 170, 43,  172, 45,  46,  175,
    48,  177, 178, 51,  180, 53,  54,  183, 184, 57,  58,  187, 60,  189, 190, 63,
    192, 65,  66,  195, 68,  197, 198, 71,  72,  201, 202, 75,  204, 77,  78,  207,
    80,  209, 210, 83,  212, 85,  86,  215, 216, 89,  90,  219, 92,  221, 222, 95,
    96,  225, 226, 99,  228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
};

/// IQ2_XXS 块结构
pub const BlockIQ2XXS = extern struct {
    d: u16, // FP16 缩放因子
    qs: [32]u16, // 量化索引

    pub const BLOCK_SIZE = 256;
    pub const BLOCK_BYTES = 66;
};

/// 反量化一个 IQ2_XXS 块到 f32
/// out: 至少 256 个元素的输出缓冲区
/// 参考实现：llama.cpp ggml-quants.c dequantize_row_iq2_xxs
pub fn dequantizeBlock(block: *const BlockIQ2XXS, out: []f32) void {
    std.debug.assert(out.len >= 256);

    const d = f16ToF32(block.d);

    // 256 元素分为 8 个子组（ib32），每个子组 32 元素
    // 每个子组由 4 个 uint16_t（8 字节）编码
    for (0..8) |ib32| {
        // 将 4 个 uint16_t 拷贝到两个 uint32_t
        // qs[4*ib32..4*ib32+3] → aux32[0..1]
        const qs_base = ib32 * 4;
        var aux32: [2]u32 = undefined;
        aux32[0] = @as(u32, block.qs[qs_base]) | (@as(u32, block.qs[qs_base + 1]) << 16);
        aux32[1] = @as(u32, block.qs[qs_base + 2]) | (@as(u32, block.qs[qs_base + 3]) << 16);

        // 以字节方式访问同一内存
        const aux8: [*]const u8 = @ptrCast(&aux32);

        // Level shift: aux32[1] 的高 4 位
        const ls: f32 = @floatFromInt(aux32[1] >> 28);
        const db = d * (0.5 + ls) * 0.25;

        // 4 批，每批 8 个值
        for (0..4) |l| {
            const grid_idx = aux8[l];
            const grid_packed = iq2xxs_grid[grid_idx];
            // 将 u64 解释为 8 个 u8
            const grid_bytes: [*]const u8 = @ptrCast(&grid_packed);

            const signs = ksigns_iq2xs[(aux32[1] >> (7 * @as(u5, @intCast(l)))) & 0x7F];

            const base = ib32 * 32 + l * 8;
            for (0..8) |j| {
                const sign: f32 = if ((signs >> @as(u3, @intCast(j))) & 1 != 0) -1.0 else 1.0;
                out[base + j] = db * @as(f32, @floatFromInt(grid_bytes[j])) * sign;
            }
        }
    }
}

/// 反量化整个 IQ2_XXS 张量
/// data: 量化数据指针，n_elements: 元素总数
pub fn dequantizeIQ2XXS(allocator: std.mem.Allocator, data: [*]const u8, n_elements: u64) ![]f32 {
    const n_blocks = (n_elements + 255) / 256;
    const out = try allocator.alloc(f32, n_blocks * 256);

    const blocks: [*]const BlockIQ2XXS = @ptrCast(@alignCast(data));

    for (0..n_blocks) |i| {
        dequantizeBlock(&blocks[i], out[i * 256 ..]);
    }

    return out[0..n_elements];
}

// ========== Q8_0 ==========
pub const BlockQ8_0 = extern struct {
    d: u16, // FP16 scale
    qs: [32]i8,
    pub const BLOCK_SIZE = 32;
    pub const BLOCK_BYTES = 34;
};

pub fn dequantizeQ8_0(allocator: std.mem.Allocator, data: [*]const u8, n_elements: u64) ![]f32 {
    const n_blocks = (n_elements + 31) / 32;
    const out = try allocator.alloc(f32, n_blocks * 32);
    const blocks: [*]const BlockQ8_0 = @ptrCast(@alignCast(data));
    for (0..n_blocks) |i| {
        const d = f16ToF32(blocks[i].d);
        for (0..32) |j| {
            out[i * 32 + j] = d * @as(f32, @floatFromInt(blocks[i].qs[j]));
        }
    }
    return out[0..n_elements];
}

// ========== Q5_K ==========
pub const BlockQ5_K = extern struct {
    d: u16, // FP16 super-block scale for quantized scales
    dmin: u16, // FP16 super-block scale for quantized mins
    scales: [12]u8, // scales and mins, quantized with 6 bits
    qh: [32]u8, // quants, high bit
    qs: [128]u8, // quants, low 4 bits
    pub const BLOCK_SIZE = 256;
    pub const BLOCK_BYTES = 176;
};

pub fn dequantizeQ5_K(allocator: std.mem.Allocator, data: [*]const u8, n_elements: u64) ![]f32 {
    const n_blocks = (n_elements + 255) / 256;
    const out = try allocator.alloc(f32, n_blocks * 256);
    const blocks: [*]const BlockQ5_K = @ptrCast(@alignCast(data));
    for (0..n_blocks) |i| {
        const d = f16ToF32(blocks[i].d);
        const dmin = f16ToF32(blocks[i].dmin);

        // Unpack scales and mins from 12 bytes
        // sc[0..3] = s[0..3] & 0x3F
        // sc[4..7] = (s[8..11] & 0x0F) | ((s[0..3] >> 2) & 0x30)
        // min[0..3] = s[4..7] & 0x3F
        // min[4..7] = (s[8..11] >> 4) | ((s[4..7] >> 2) & 0x30)
        var sc: [8]u8 = undefined;
        var min: [8]u8 = undefined;
        const s = blocks[i].scales;
        for (0..4) |k| {
            sc[k] = s[k] & 0x3F;
            min[k] = s[4 + k] & 0x3F;
            sc[4 + k] = (s[8 + k] & 0x0F) | ((s[k] >> 2) & 0x30);
            min[4 + k] = (s[8 + k] >> 4) | ((s[4 + k] >> 2) & 0x30);
        }

        for (0..256) |j| {
            const sub_block = j / 32;
            const offset = j % 32;
            const low = (blocks[i].qs[j / 2] >> @as(u3, @intCast(4 * (j % 2)))) & 0xF;
            const high = (blocks[i].qh[offset] >> @as(u3, @intCast(sub_block))) & 1;
            const q = low | (high << 4);
            out[i * 256 + j] = d * @as(f32, @floatFromInt(sc[sub_block])) * @as(f32, @floatFromInt(q)) -
                dmin * @as(f32, @floatFromInt(min[sub_block]));
        }
    }
    return out[0..n_elements];
}

// ========== Q6_K ==========
pub const BlockQ6_K = extern struct {
    ql: [128]u8, // lower 4 bits
    qh: [64]u8, // upper 2 bits
    scales: [16]i8,
    d: u16, // FP16 super-block scale
    pub const BLOCK_SIZE = 256;
    pub const BLOCK_BYTES = 210;
};

pub fn dequantizeQ6_K(allocator: std.mem.Allocator, data: [*]const u8, n_elements: u64) ![]f32 {
    const n_blocks = (n_elements + 255) / 256;
    const out = try allocator.alloc(f32, n_blocks * 256);
    const blocks: [*]const BlockQ6_K = @ptrCast(@alignCast(data));
    for (0..n_blocks) |i| {
        const d = f16ToF32(blocks[i].d);
        var y_off: usize = i * 256;
        var ql_off: usize = 0;
        var qh_off: usize = 0;
        var sc_off: usize = 0;
        for (0..2) |_| {
            for (0..32) |l| {
                const is = l / 16;
                const q1 = @as(i8, @intCast((blocks[i].ql[ql_off + l] & 0xF) | (((blocks[i].qh[qh_off + l] >> 0) & 3) << 4))) - 32;
                const q2 = @as(i8, @intCast((blocks[i].ql[ql_off + 32 + l] & 0xF) | (((blocks[i].qh[qh_off + l] >> 2) & 3) << 4))) - 32;
                const q3 = @as(i8, @intCast((blocks[i].ql[ql_off + l] >> 4) | (((blocks[i].qh[qh_off + l] >> 4) & 3) << 4))) - 32;
                const q4 = @as(i8, @intCast((blocks[i].ql[ql_off + 32 + l] >> 4) | (((blocks[i].qh[qh_off + l] >> 6) & 3) << 4))) - 32;
                out[y_off + l] = d * @as(f32, @floatFromInt(blocks[i].scales[sc_off + is])) * @as(f32, @floatFromInt(q1));
                out[y_off + 32 + l] = d * @as(f32, @floatFromInt(blocks[i].scales[sc_off + is + 2])) * @as(f32, @floatFromInt(q2));
                out[y_off + 64 + l] = d * @as(f32, @floatFromInt(blocks[i].scales[sc_off + is + 4])) * @as(f32, @floatFromInt(q3));
                out[y_off + 96 + l] = d * @as(f32, @floatFromInt(blocks[i].scales[sc_off + is + 6])) * @as(f32, @floatFromInt(q4));
            }
            y_off += 128;
            ql_off += 64;
            qh_off += 32;
            sc_off += 8;
        }
    }
    return out[0..n_elements];
}

// ========== IQ4_NL ==========
pub const BlockIQ4_NL = extern struct {
    d: u16, // FP16 scale
    qs: [16]u8, // 32 × 4-bit packed
    pub const BLOCK_SIZE = 32;
    pub const BLOCK_BYTES = 18;
};

pub fn dequantizeIQ4_NL(allocator: std.mem.Allocator, data: [*]const u8, n_elements: u64) ![]f32 {
    const n_blocks = (n_elements + 31) / 32;
    const out = try allocator.alloc(f32, n_blocks * 32);
    const blocks: [*]const BlockIQ4_NL = @ptrCast(@alignCast(data));
    for (0..n_blocks) |i| {
        const d = f16ToF32(blocks[i].d);
        for (0..16) |j| {
            const q = blocks[i].qs[j];
            out[i * 32 + j] = d * @as(f32, @floatFromInt(kvalues_iq4nl[q & 0xF]));
            out[i * 32 + j + 16] = d * @as(f32, @floatFromInt(kvalues_iq4nl[q >> 4]));
        }
    }
    return out[0..n_elements];
}

// ========== IQ4_XS ==========
pub const BlockIQ4_XS = extern struct {
    d: u16, // FP16 scale
    scales_h: u16,
    scales_l: [4]u8,
    qs: [128]u8,
    pub const BLOCK_SIZE = 256;
    pub const BLOCK_BYTES = 136;
};

pub fn dequantizeIQ4_XS(allocator: std.mem.Allocator, data: [*]const u8, n_elements: u64) ![]f32 {
    const n_blocks = (n_elements + 255) / 256;
    const out = try allocator.alloc(f32, n_blocks * 256);
    const blocks: [*]const BlockIQ4_XS = @ptrCast(@alignCast(data));
    for (0..n_blocks) |i| {
        const d = f16ToF32(blocks[i].d);
        var y_off: usize = i * 256;
        var qs_off: usize = 0;
        for (0..8) |ib| {
            const ls = ((blocks[i].scales_l[ib / 2] >> (4 * @as(u3, @intCast(ib % 2)))) & 0xF) |
                (((blocks[i].scales_h >> (2 * @as(u4, @intCast(ib)))) & 3) << 4);
            const dl = d * (@as(f32, @floatFromInt(ls)) - 32.0);
            for (0..16) |j| {
                const q = blocks[i].qs[qs_off + j];
                out[y_off + j] = dl * @as(f32, @floatFromInt(kvalues_iq4nl[q & 0xF]));
                out[y_off + j + 16] = dl * @as(f32, @floatFromInt(kvalues_iq4nl[q >> 4]));
            }
            y_off += 32;
            qs_off += 16;
        }
    }
    return out[0..n_elements];
}

// ========== Q4_K ==========
pub const BlockQ4_K = extern struct {
    d: u16,
    dmin: u16,
    scales: [12]u8,
    qs: [128]u8,
    pub const BLOCK_SIZE = 256;
    pub const BLOCK_BYTES = 144;
};

inline fn getScaleMinK4(j: usize, q: [*]const u8, d: *u8, m: *u8) void {
    if (j < 4) {
        d.* = q[j] & 63;
        m.* = q[j + 4] & 63;
    } else {
        d.* = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m.* = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

pub fn dequantizeQ4_K(allocator: std.mem.Allocator, data: [*]const u8, n_elements: u64) ![]f32 {
    const n_blocks = (n_elements + 255) / 256;
    const out = try allocator.alloc(f32, n_blocks * 256);
    const blocks: [*]const BlockQ4_K = @ptrCast(@alignCast(data));
    for (0..n_blocks) |i| {
        const d = f16ToF32(blocks[i].d);
        const min = f16ToF32(blocks[i].dmin);
        var y_off: usize = i * 256;
        var qs_off: usize = 0;
        var is: usize = 0;
        for (0..4) |_| {
            var sc: u8 = 0;
            var m0: u8 = 0;
            getScaleMinK4(is, &blocks[i].scales, &sc, &m0);
            const d1 = d * @as(f32, @floatFromInt(sc));
            const m1 = min * @as(f32, @floatFromInt(m0));
            getScaleMinK4(is + 1, &blocks[i].scales, &sc, &m0);
            const d2 = d * @as(f32, @floatFromInt(sc));
            const m2 = min * @as(f32, @floatFromInt(m0));
            for (0..32) |l| {
                out[y_off + l] = d1 * @as(f32, @floatFromInt(blocks[i].qs[qs_off + l] & 0xF)) - m1;
            }
            y_off += 32;
            for (0..32) |l| {
                out[y_off + l] = d2 * @as(f32, @floatFromInt(blocks[i].qs[qs_off + l] >> 4)) - m2;
            }
            y_off += 32;
            qs_off += 32;
            is += 2;
        }
    }
    return out[0..n_elements];
}

// ========== IQ3_S ==========
pub const BlockIQ3_S = extern struct {
    d: u16, // FP16 scale
    qs: [64]u8, // low 8 bits of 9-bit indices
    qh: [8]u8, // high 1 bit of 9-bit indices
    signs: [32]u8,
    scales: [4]u8,
    pub const BLOCK_SIZE = 256;
    pub const BLOCK_BYTES = 110;
};

pub fn dequantizeIQ3_S(allocator: std.mem.Allocator, data: [*]const u8, n_elements: u64) ![]f32 {
    const n_blocks = (n_elements + 255) / 256;
    const out = try allocator.alloc(f32, n_blocks * 256);
    const blocks: [*]const BlockIQ3_S = @ptrCast(@alignCast(data));
    for (0..n_blocks) |i| {
        const d = f16ToF32(blocks[i].d);
        var y_off: usize = i * 256;
        for (0..8) |ib32| {
            const scale_byte = blocks[i].scales[ib32 / 2];
            const db1 = d * (1.0 + 2.0 * @as(f32, @floatFromInt(scale_byte & 0xF)));
            const db2 = d * (1.0 + 2.0 * @as(f32, @floatFromInt(scale_byte >> 4)));
            const db = if (ib32 % 2 == 0) db1 else db2;
            for (0..4) |l| {
                const qs_idx = ib32 * 8 + l;
                const qh_idx = ib32;
                const grid_packed = iq3s_grid[blocks[i].qs[qs_idx] | (@as(u16, (blocks[i].qh[qh_idx] >> @as(u3, @intCast(l))) & 1) << 8)];
                const grid_bytes: [*]const u8 = @ptrCast(&grid_packed);
                const sign_byte = blocks[i].signs[ib32 * 4 + l];
                for (0..4) |j| {
                    const sign: f32 = if ((sign_byte >> @as(u3, @intCast(j))) & 1 != 0) -1.0 else 1.0;
                    out[y_off + l * 4 + j] = db * @as(f32, @floatFromInt(grid_bytes[j])) * sign;
                }
            }
            y_off += 32;
        }
    }
    return out[0..n_elements];
}

/// 通用反量化：根据 GgmlType 反量化张量到 f32
pub fn dequantize(allocator: std.mem.Allocator, data: [*]const u8, ggml_type: @import("gguf.zig").GgmlType, n_elements: u64) ![]f32 {
    return switch (ggml_type) {
        .f32 => {
            const src: [*]const f32 = @ptrCast(@alignCast(data));
            return allocator.dupe(f32, src[0..n_elements]);
        },
        .f16 => {
            const src: [*]const u16 = @ptrCast(@alignCast(data));
            const out = try allocator.alloc(f32, n_elements);
            for (out, 0..) |*o, i| {
                o.* = f16ToF32(src[i]);
            }
            return out;
        },
        .q8_0 => dequantizeQ8_0(allocator, data, n_elements),
        .q6_k => dequantizeQ6_K(allocator, data, n_elements),
        .q5_k => dequantizeQ5_K(allocator, data, n_elements),
        .q4_k => dequantizeQ4_K(allocator, data, n_elements),
        .iq4_nl => dequantizeIQ4_NL(allocator, data, n_elements),
        .iq4_xs => dequantizeIQ4_XS(allocator, data, n_elements),
        .iq3_s => dequantizeIQ3_S(allocator, data, n_elements),
        .iq2_xxs => dequantizeIQ2XXS(allocator, data, n_elements),
        else => {
            std.log.err("未实现的反量化类型: {s}", .{@tagName(ggml_type)});
            return error.UnsupportedQuantizationType;
        },
    };
}

// ========== 测试 ==========

test "f16ToF32 subnormal precision" {
    // 0x0001 = smallest subnormal = 2^(-24) ≈ 5.96e-8
    const smallest = f16ToF32(0x0001);
    try std.testing.expectApproxEqAbs(@as(f32, 5.9604645e-8), smallest, 1e-14);

    // 0x03FF = largest subnormal = 2^(-14) * (1 - 2^(-10)) ≈ 6.097e-5
    const largest_sub = f16ToF32(0x03FF);
    try std.testing.expectApproxEqAbs(@as(f32, 6.097555e-5), largest_sub, 1e-8);

    // 0x0080 = 2^(-17) ≈ 7.629e-6
    const mid_sub = f16ToF32(0x0080);
    try std.testing.expectApproxEqAbs(@as(f32, 7.6293945e-6), mid_sub, 1e-12);
}

test "f16ToF32 comprehensive" {
    // Zero
    try std.testing.expectEqual(@as(f32, 0.0), f16ToF32(0x0000));
    try std.testing.expectEqual(@as(f32, -0.0), f16ToF32(0x8000));
    // 1.0
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), f16ToF32(0x3C00), 1e-4);
    // 2.0
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), f16ToF32(0x4000), 1e-4);
    // 0.5
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), f16ToF32(0x3800), 1e-4);
    // -1.0
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), f16ToF32(0xBC00), 1e-4);
    // Subnormal: 0x03FF = 2^-14 * (1 - 2^-10) ≈ 6.097e-5
    const subnormal = f16ToF32(0x03FF);
    try std.testing.expect(subnormal > 0.0);
    try std.testing.expect(subnormal < 0.001);
    // Infinity
    try std.testing.expect(std.math.isInf(f16ToF32(0x7C00)));
    try std.testing.expect(std.math.isInf(f16ToF32(0xFC00)));
    // NaN
    try std.testing.expect(std.math.isNan(f16ToF32(0x7C01)));
}

test "f16 to f32 conversion" {
    // 0x3C00 = 1.0 in FP16
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), f16ToF32(0x3C00), 1e-4);
    // 0x4000 = 2.0 in FP16
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), f16ToF32(0x4000), 1e-4);
    // 0x0000 = 0.0
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), f16ToF32(0x0000), 1e-4);
}

test "IQ2_XXS block size" {
    try std.testing.expectEqual(@as(usize, 66), @sizeOf(BlockIQ2XXS));
    try std.testing.expectEqual(@as(u64, 256), BlockIQ2XXS.BLOCK_SIZE);
}
