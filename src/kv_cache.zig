// KV 缓存管理 — 借鉴 vLLM PagedAttention 的 Block-based 设计
//
// 核心改进：
//   - 物理 KV cache 划分为固定大小的 block（BLOCK_SIZE=16 tokens/block）
//   - Block Table 将逻辑 token 位置映射到物理 block ID
//   - 支持 Prefix Caching：通过链式 xxHash64 复用公共前缀 block
//   - 减少内存碎片，支持动态序列增长

const std = @import("std");
const HyperParams = @import("model.zig").HyperParams;
const dequant = @import("dequant.zig");

/// 每个 block 固定容纳的 token 数
pub const BLOCK_SIZE = 16;

/// 物理块状态
const Block = struct {
    ref_count: u32,
    /// xxHash64（带前缀链），maxInt(u64) 表示未计算/未满
    hash: u64,
};

/// MLA 风格 KV 缓存：Block-based 管理
pub const KVCache = struct {
    allocator: std.mem.Allocator,
    hp: HyperParams,

    /// 总物理块数
    num_blocks: u32,
    /// 块大小（tokens per block）
    block_size: u32,

    // 物理 KV cache 存储
    /// [n_layers, num_blocks, block_size, kv_lora_rank]
    kv_compressed: []f32,
    /// [n_layers, num_blocks, block_size, qk_rope_head_dim]
    k_pe: []f32,

    // 块管理
    blocks: []Block,
    free_list: std.ArrayList(u32),

    // Prefix Caching: hash -> block_id
    hash_map: std.AutoHashMap(u64, u32),

    // 当前序列的 Block Table（逻辑 block -> 物理 block ID）
    block_table: std.ArrayList(u32),
    /// 当前序列长度
    len: u32,

    // ===== KV Cache 量化（长期：Q8_0）=====
    /// 是否启用 Q8_0 量化
    quantized: bool,
    /// [n_layers, num_blocks, block_size, kv_q8_per_token]
    kv_q8: ?[]dequant.BlockQ8_0,
    /// [n_layers, num_blocks, block_size, pe_q8_per_token]
    k_pe_q8: ?[]dequant.BlockQ8_0,
    /// 反量化临时缓冲区
    kv_scratch: []f32,
    pe_scratch: []f32,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, hp: HyperParams, max_seq_len: u32) !Self {
        const n_layers = hp.num_hidden_layers;
        const kv_rank = hp.kv_lora_rank;
        const rope_dim = hp.qk_rope_head_dim;

        const num_blocks = (max_seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
        std.debug.assert(num_blocks > 0);

        const kv_compressed = try allocator.alloc(f32, n_layers * num_blocks * BLOCK_SIZE * kv_rank);
        errdefer allocator.free(kv_compressed);
        @memset(kv_compressed, 0);

        const k_pe = try allocator.alloc(f32, n_layers * num_blocks * BLOCK_SIZE * rope_dim);
        errdefer allocator.free(k_pe);
        @memset(k_pe, 0);

        const blocks = try allocator.alloc(Block, num_blocks);
        errdefer allocator.free(blocks);
        for (blocks) |*b| {
            b.* = .{ .ref_count = 0, .hash = std.math.maxInt(u64) };
        }

        var free_list: std.ArrayList(u32) = .empty;
        errdefer free_list.deinit(allocator);
        try free_list.ensureTotalCapacity(allocator, num_blocks);
        for (0..num_blocks) |i| {
            free_list.appendAssumeCapacity(@intCast(num_blocks - 1 - i));
        }

        var hash_map = std.AutoHashMap(u64, u32).init(allocator);
        errdefer hash_map.deinit();

        var block_table: std.ArrayList(u32) = .empty;
        errdefer block_table.deinit(allocator);

        const kv_scratch = try allocator.alloc(f32, kv_rank);
        errdefer allocator.free(kv_scratch);
        const pe_scratch = try allocator.alloc(f32, rope_dim);
        errdefer allocator.free(pe_scratch);

        return .{
            .allocator = allocator,
            .hp = hp,
            .num_blocks = @intCast(num_blocks),
            .block_size = BLOCK_SIZE,
            .kv_compressed = kv_compressed,
            .k_pe = k_pe,
            .blocks = blocks,
            .free_list = free_list,
            .hash_map = hash_map,
            .block_table = block_table,
            .len = 0,
            .quantized = false,
            .kv_q8 = null,
            .k_pe_q8 = null,
            .kv_scratch = kv_scratch,
            .pe_scratch = pe_scratch,
        };
    }

    pub fn deinit(self: *Self) void {
        // 释放当前序列占用的所有 block
        for (self.block_table.items) |bid| {
            self.releaseBlock(bid);
        }
        self.block_table.deinit(self.allocator);
        self.allocator.free(self.kv_compressed);
        self.allocator.free(self.k_pe);
        self.allocator.free(self.blocks);
        self.free_list.deinit(self.allocator);
        self.hash_map.deinit();
        if (self.kv_q8) |buf| self.allocator.free(buf);
        if (self.k_pe_q8) |buf| self.allocator.free(buf);
        self.allocator.free(self.kv_scratch);
        self.allocator.free(self.pe_scratch);
    }

    /// 启用 Q8_0 KV Cache 量化（lazy，可随时调用）
    pub fn enableQuantization(self: *Self) !void {
        if (self.quantized) return;
        const n_layers = self.hp.num_hidden_layers;
        const kv_rank = self.hp.kv_lora_rank;
        const rope_dim = self.hp.qk_rope_head_dim;
        const kv_q8_per_token = (kv_rank + 31) / 32;
        const pe_q8_per_token = (rope_dim + 31) / 32;
        const total_tokens = n_layers * self.num_blocks * self.block_size;

        self.kv_q8 = try self.allocator.alloc(dequant.BlockQ8_0, total_tokens * kv_q8_per_token);
        self.k_pe_q8 = try self.allocator.alloc(dequant.BlockQ8_0, total_tokens * pe_q8_per_token);
        self.quantized = true;
    }

    // ===== Block 管理 =====

    fn allocateBlock(self: *Self) !u32 {
        if (self.free_list.items.len == 0) return error.OutOfKVCacheBlocks;
        const block_id = self.free_list.pop().?;
        self.blocks[block_id].ref_count = 1;
        self.blocks[block_id].hash = std.math.maxInt(u64);
        return block_id;
    }

    fn releaseBlock(self: *Self, block_id: u32) void {
        std.debug.assert(block_id < self.num_blocks);
        const block = &self.blocks[block_id];
        std.debug.assert(block.ref_count > 0);
        block.ref_count -= 1;
        if (block.ref_count == 0) {
            if (block.hash != std.math.maxInt(u64)) {
                _ = self.hash_map.remove(block.hash);
            }
            block.hash = std.math.maxInt(u64);
            self.free_list.append(self.allocator, block_id) catch {};
        }
    }

    fn retainBlock(self: *Self, block_id: u32) void {
        std.debug.assert(block_id < self.num_blocks);
        self.blocks[block_id].ref_count += 1;
    }

    // ===== 序列 Block Table 管理 =====

    /// 确保 block_table 能容纳至少 n_tokens 个位置
    pub fn ensureCapacity(self: *Self, n_tokens: u32) !void {
        const needed_blocks = (n_tokens + self.block_size - 1) / self.block_size;
        while (self.block_table.items.len < needed_blocks) {
            const bid = try self.allocateBlock();
            try self.block_table.append(self.allocator, bid);
        }
    }

    /// 截断到指定长度，释放多余 block
    pub fn truncate(self: *Self, n_tokens: u32) void {
        const needed_blocks = (n_tokens + self.block_size - 1) / self.block_size;
        while (self.block_table.items.len > needed_blocks) {
            const bid = self.block_table.pop();
            self.releaseBlock(bid);
        }
        if (n_tokens < self.len) {
            self.len = n_tokens;
        }
    }

    /// 重置序列：释放所有 block，清空 block_table
    pub fn reset(self: *Self) void {
        for (self.block_table.items) |bid| {
            self.releaseBlock(bid);
        }
        self.block_table.clearRetainingCapacity();
        self.len = 0;
    }

    // ===== KV 读写 =====

    fn resolveBlock(self: *const Self, pos: u32) !u32 {
        const block_idx = pos / self.block_size;
        if (block_idx >= self.block_table.items.len) return error.PositionOutOfBounds;
        return self.block_table.items[block_idx];
    }

    pub fn getKVCompressed(self: *Self, layer: u32, pos: u32) ![]const f32 {
        if (layer >= self.hp.num_hidden_layers) return error.LayerOutOfBounds;
        if (pos >= self.len) return error.PositionNotWritten;
        const block_id = try self.resolveBlock(pos);
        const offset = pos % self.block_size;
        const kv_rank = self.hp.kv_lora_rank;

        if (self.quantized) {
            const kv_q8_per_token = (kv_rank + 31) / 32;
            const base_kv_q8 = ((layer * self.num_blocks + block_id) * self.block_size + offset) * kv_q8_per_token;
            const blocks = self.kv_q8.?[base_kv_q8..][0..kv_q8_per_token];
            for (0..kv_q8_per_token) |bi| {
                const start = bi * 32;
                const end = @min(start + 32, kv_rank);
                const d = dequant.f16ToF32(blocks[bi].d);
                for (start..end) |vi| {
                    const qi = vi - start;
                    self.kv_scratch[vi] = d * @as(f32, @floatFromInt(blocks[bi].qs[qi]));
                }
            }
            return self.kv_scratch[0..kv_rank];
        }

        const base = ((layer * self.num_blocks + block_id) * self.block_size + offset) * kv_rank;
        return self.kv_compressed[base..][0..kv_rank];
    }

    pub fn getKPe(self: *Self, layer: u32, pos: u32) ![]const f32 {
        if (layer >= self.hp.num_hidden_layers) return error.LayerOutOfBounds;
        if (pos >= self.len) return error.PositionNotWritten;
        const block_id = try self.resolveBlock(pos);
        const offset = pos % self.block_size;
        const rope_dim = self.hp.qk_rope_head_dim;

        if (self.quantized) {
            const pe_q8_per_token = (rope_dim + 31) / 32;
            const base_pe_q8 = ((layer * self.num_blocks + block_id) * self.block_size + offset) * pe_q8_per_token;
            const blocks = self.k_pe_q8.?[base_pe_q8..][0..pe_q8_per_token];
            for (0..pe_q8_per_token) |bi| {
                const start = bi * 32;
                const end = @min(start + 32, rope_dim);
                const d = dequant.f16ToF32(blocks[bi].d);
                for (start..end) |vi| {
                    const qi = vi - start;
                    self.pe_scratch[vi] = d * @as(f32, @floatFromInt(blocks[bi].qs[qi]));
                }
            }
            return self.pe_scratch[0..rope_dim];
        }

        const base = ((layer * self.num_blocks + block_id) * self.block_size + offset) * rope_dim;
        return self.k_pe[base..][0..rope_dim];
    }

    pub fn put(self: *Self, layer: u32, pos: u32, kv_comp: []const f32, k_pe_val: []const f32) !void {
        std.debug.assert(kv_comp.len == self.hp.kv_lora_rank);
        std.debug.assert(k_pe_val.len == self.hp.qk_rope_head_dim);
        if (layer >= self.hp.num_hidden_layers) return error.LayerOutOfBounds;

        const block_id = try self.resolveBlock(pos);
        const offset = pos % self.block_size;
        const kv_rank = self.hp.kv_lora_rank;
        const rope_dim = self.hp.qk_rope_head_dim;

        if (self.quantized) {
            // 量化写入 Q8_0
            const kv_q8_per_token = (kv_rank + 31) / 32;
            const base_kv_q8 = ((layer * self.num_blocks + block_id) * self.block_size + offset) * kv_q8_per_token;
            var kv_blocks = self.kv_q8.?[base_kv_q8..][0..kv_q8_per_token];
            for (0..kv_q8_per_token) |bi| {
                const start = bi * 32;
                const end = @min(start + 32, kv_rank);
                var max_abs: f32 = 0;
                for (kv_comp[start..end]) |v| {
                    const av = @abs(v);
                    if (av > max_abs) max_abs = av;
                }
                const d = if (max_abs > 0) max_abs / 127.0 else 0.0;
                kv_blocks[bi].d = dequant.f32ToF16(d);
                for (start..end, 0..) |vi, qi| {
                    var q_i = @as(i32, @intFromFloat(@round(kv_comp[vi] / d)));
                    if (q_i > 127) q_i = 127;
                    if (q_i < -127) q_i = -127;
                    kv_blocks[bi].qs[qi] = @intCast(q_i);
                }
                for (end - start..32) |qi| {
                    kv_blocks[bi].qs[qi] = 0;
                }
            }

            const pe_q8_per_token = (rope_dim + 31) / 32;
            const base_pe_q8 = ((layer * self.num_blocks + block_id) * self.block_size + offset) * pe_q8_per_token;
            var pe_blocks = self.k_pe_q8.?[base_pe_q8..][0..pe_q8_per_token];
            for (0..pe_q8_per_token) |bi| {
                const start = bi * 32;
                const end = @min(start + 32, rope_dim);
                var max_abs: f32 = 0;
                for (k_pe_val[start..end]) |v| {
                    const av = @abs(v);
                    if (av > max_abs) max_abs = av;
                }
                const d = if (max_abs > 0) max_abs / 127.0 else 0.0;
                pe_blocks[bi].d = dequant.f32ToF16(d);
                for (start..end, 0..) |vi, qi| {
                    var q_i = @as(i32, @intFromFloat(@round(k_pe_val[vi] / d)));
                    if (q_i > 127) q_i = 127;
                    if (q_i < -127) q_i = -127;
                    pe_blocks[bi].qs[qi] = @intCast(q_i);
                }
                for (end - start..32) |qi| {
                    pe_blocks[bi].qs[qi] = 0;
                }
            }
        } else {
            const base_kv = ((layer * self.num_blocks + block_id) * self.block_size + offset) * kv_rank;
            @memcpy(self.kv_compressed[base_kv..][0..kv_rank], kv_comp);

            const base_pe = ((layer * self.num_blocks + block_id) * self.block_size + offset) * rope_dim;
            @memcpy(self.k_pe[base_pe..][0..rope_dim], k_pe_val);
        }

        if (pos >= self.len) {
            self.len = pos + 1;
        }
    }

    // ===== Prefix Caching =====

    /// 尝试对当前序列的已填满 block 进行 Prefix Caching 注册
    /// 应在处理完一个完整的 block 后调用
    pub fn cacheBlock(self: *Self, block_idx: u32, token_ids: []const u32, prefix_hash: u64) void {
        std.debug.assert(block_idx < self.block_table.items.len);
        if (token_ids.len != self.block_size) return; // 只缓存满 block

        const block_id = self.block_table.items[block_idx];
        const h = computeHash(token_ids, prefix_hash);

        // 碰撞检查：如果 hash 已存在但 block 内容不同，不注册
        if (self.hash_map.get(h)) |existing_id| {
            if (existing_id != block_id) return; // hash 碰撞，跳过
        }

        self.blocks[block_id].hash = h;
        self.hash_map.put(h, block_id) catch {};
    }

    /// 通过 token 序列查找可复用的前缀 block
    /// 返回可复用的 block 数（从序列开头连续的已缓存 block）
    pub fn findPrefixBlocks(self: *Self, tokens: []const u32) u32 {
        var prefix_hash: u64 = std.math.maxInt(u64);
        var reusable: u32 = 0;

        const num_blocks = (tokens.len + self.block_size - 1) / self.block_size;
        for (0..num_blocks) |bi| {
            const start = bi * self.block_size;
            const end = @min(start + self.block_size, tokens.len);
            if (end - start != self.block_size) break; // 不满的 block 不参与缓存

            const h = computeHash(tokens[start..end], prefix_hash);
            if (self.hash_map.get(h)) |_| {
                // 命中：可以复用该 block
                prefix_hash = h;
                reusable += 1;
            } else {
                break;
            }
        }
        return reusable;
    }

    /// 复用缓存的 prefix block：将 block_table 的前 N 个 block 替换为缓存的物理块
    /// 调用前需确保 tokens 的前 N*block_size 个 token 与缓存内容一致
    pub fn reusePrefixBlocks(self: *Self, tokens: []const u32, n_blocks: u32) !void {
        if (n_blocks == 0) return;

        var prefix_hash: u64 = std.math.maxInt(u64);
        for (0..n_blocks) |bi| {
            const start = bi * self.block_size;
            const h = computeHash(tokens[start..][0..self.block_size], prefix_hash);
            const cached_id = self.hash_map.get(h) orelse return error.PrefixCacheMiss;

            if (bi < self.block_table.items.len) {
                // 释放旧 block
                self.releaseBlock(self.block_table.items[bi]);
            }

            // 复用缓存 block
            self.retainBlock(cached_id);

            if (bi < self.block_table.items.len) {
                self.block_table.items[bi] = cached_id;
            } else {
                try self.block_table.append(self.allocator, cached_id);
            }

            prefix_hash = h;
        }

        self.len = @intCast(n_blocks * self.block_size);
    }

    /// 计算内存占用（字节）
    pub fn memoryUsage(self: *const Self) u64 {
        const kv_bytes = self.kv_compressed.len * @sizeOf(f32);
        const pe_bytes = self.k_pe.len * @sizeOf(f32);
        return kv_bytes + pe_bytes;
    }

    /// 剩余空闲块数
    pub fn numFreeBlocks(self: *const Self) u32 {
        return @intCast(self.free_list.items.len);
    }
};

/// 计算 block 的链式 xxHash64（模块级辅助函数）
/// token_ids: 该 block 中的 token ID
/// prefix_hash: 前一个 block 的 hash，首块传 std.math.maxInt(u64)
pub fn computeHash(token_ids: []const u32, prefix_hash: u64) u64 {
    var hasher = std.hash.XxHash64.init(0);
    if (prefix_hash != std.math.maxInt(u64)) {
        var prefix_bytes: [8]u8 = undefined;
        std.mem.writeInt(u64, &prefix_bytes, prefix_hash, .little);
        hasher.update(&prefix_bytes);
    }
    for (token_ids) |tid| {
        var bytes: [4]u8 = undefined;
        std.mem.writeInt(u32, &bytes, tid, .little);
        hasher.update(&bytes);
    }
    return hasher.final();
}
