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
            const bid = self.block_table.pop().?;
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
                if (d > 0) {
                    for (start..end, 0..) |vi, qi| {
                        var q_i = @as(i32, @intFromFloat(@round(kv_comp[vi] / d)));
                        if (q_i > 127) q_i = 127;
                        if (q_i < -127) q_i = -127;
                        kv_blocks[bi].qs[qi] = @intCast(q_i);
                    }
                } else {
                    for (start..end, 0..) |_, qi| {
                        kv_blocks[bi].qs[qi] = 0;
                    }
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
                if (d > 0) {
                    for (start..end, 0..) |vi, qi| {
                        var q_i = @as(i32, @intFromFloat(@round(k_pe_val[vi] / d)));
                        if (q_i > 127) q_i = 127;
                        if (q_i < -127) q_i = -127;
                        pe_blocks[bi].qs[qi] = @intCast(q_i);
                    }
                } else {
                    for (start..end, 0..) |_, qi| {
                        pe_blocks[bi].qs[qi] = 0;
                    }
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

    // ========== 磁盘持久化 ==========

    /// 将当前 KV Cache 状态保存到文件（简单格式：header + block_table + 完整物理块池）
    pub fn saveToFile(self: *const Self, path: []const u8) !void {
        const fd = try std.posix.openat(
            std.posix.AT.FDCWD,
            path,
            .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true, .CLOEXEC = true },
            0o644,
        );
        errdefer _ = std.posix.system.close(fd);
        defer _ = std.posix.system.close(fd);

        const header = CheckpointHeader{
            .magic = .{ 'K', 'V', 'C', 'K' },
            .version = 1,
            .len = self.len,
            .quantized = if (self.quantized) 1 else 0,
            .block_table_len = @intCast(self.block_table.items.len),
            .num_layers = self.hp.num_hidden_layers,
            .num_blocks = self.num_blocks,
            .block_size = self.block_size,
            .kv_rank = self.hp.kv_lora_rank,
            .rope_dim = self.hp.qk_rope_head_dim,
        };
        try writeAll(fd, std.mem.asBytes(&header));

        // block_table
        for (self.block_table.items) |bid| {
            var buf: [4]u8 = undefined;
            std.mem.writeInt(u32, &buf, bid, .little);
            try writeAll(fd, &buf);
        }

        if (self.quantized) {
            const kv_q8_per_token = (self.hp.kv_lora_rank + 31) / 32;
            const pe_q8_per_token = (self.hp.qk_rope_head_dim + 31) / 32;
            const total_tokens = self.hp.num_hidden_layers * self.num_blocks * self.block_size;
            try writeAll(fd, std.mem.sliceAsBytes(self.kv_q8.?[0 .. total_tokens * kv_q8_per_token]));
            try writeAll(fd, std.mem.sliceAsBytes(self.k_pe_q8.?[0 .. total_tokens * pe_q8_per_token]));
        } else {
            try writeAll(fd, std.mem.sliceAsBytes(self.kv_compressed));
            try writeAll(fd, std.mem.sliceAsBytes(self.k_pe));
        }
    }

    /// 从文件加载 KV Cache 状态（覆盖当前状态）
    pub fn loadFromFile(self: *Self, path: []const u8) !void {
        const fd = try std.posix.openat(
            std.posix.AT.FDCWD,
            path,
            .{ .ACCMODE = .RDONLY, .CLOEXEC = true },
            0,
        );
        errdefer _ = std.posix.system.close(fd);
        defer _ = std.posix.system.close(fd);

        var header: CheckpointHeader = undefined;
        try readNoEof(fd, std.mem.asBytes(&header));
        if (!std.mem.eql(u8, &header.magic, "KVCK")) return error.InvalidCheckpointMagic;
        if (header.version != 1) return error.UnsupportedCheckpointVersion;
        if (header.num_layers != self.hp.num_hidden_layers or
            header.num_blocks != self.num_blocks or
            header.block_size != self.block_size or
            header.kv_rank != self.hp.kv_lora_rank or
            header.rope_dim != self.hp.qk_rope_head_dim)
        {
            return error.CheckpointShapeMismatch;
        }

        self.len = header.len;

        // 释放旧 block_table，重建
        for (self.block_table.items) |bid| {
            self.releaseBlock(bid);
        }
        self.block_table.clearRetainingCapacity();
        for (0..header.block_table_len) |_| {
            var buf: [4]u8 = undefined;
            try readNoEof(fd, &buf);
            const bid = std.mem.readInt(u32, &buf, .little);
            self.retainBlock(bid);
            try self.block_table.append(self.allocator, bid);
        }

        if (header.quantized == 1) {
            if (!self.quantized) try self.enableQuantization();
            const kv_q8_per_token = (self.hp.kv_lora_rank + 31) / 32;
            const pe_q8_per_token = (self.hp.qk_rope_head_dim + 31) / 32;
            const total_tokens = self.hp.num_hidden_layers * self.num_blocks * self.block_size;
            try readNoEof(fd, std.mem.sliceAsBytes(self.kv_q8.?[0 .. total_tokens * kv_q8_per_token]));
            try readNoEof(fd, std.mem.sliceAsBytes(self.k_pe_q8.?[0 .. total_tokens * pe_q8_per_token]));
        } else {
            try readNoEof(fd, std.mem.sliceAsBytes(self.kv_compressed));
            try readNoEof(fd, std.mem.sliceAsBytes(self.k_pe));
        }
    }
};

const CheckpointHeader = extern struct {
    magic: [4]u8,
    version: u32,
    len: u32,
    quantized: u8,
    block_table_len: u32,
    num_layers: u32,
    num_blocks: u32,
    block_size: u32,
    kv_rank: u32,
    rope_dim: u32,
};

fn writeAll(fd: std.posix.fd_t, buf: []const u8) !void {
    var offset: usize = 0;
    while (offset < buf.len) {
        const rc = std.posix.system.write(fd, buf[offset..].ptr, buf.len - offset);
        if (rc < 0) {
            switch (std.posix.errno(rc)) {
                .INTR => continue,
                .AGAIN => return error.WouldBlock,
                .BADF => return error.Unexpected,
                .IO => return error.InputOutput,
                .NOBUFS => return error.SystemResources,
                .NOMEM => return error.SystemResources,
                .CONNRESET => return error.ConnectionResetByPeer,
                .PIPE => return error.BrokenPipe,
                else => return error.Unexpected,
            }
        }
        if (rc == 0) return error.WriteFailed;
        offset += @intCast(rc);
    }
}

fn readNoEof(fd: std.posix.fd_t, buf: []u8) !void {
    var offset: usize = 0;
    while (offset < buf.len) {
        const n = try std.posix.read(fd, buf[offset..]);
        if (n == 0) return error.EndOfStream;
        offset += n;
    }
}

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

// ========== TDD 测试 ==========

const testing = std.testing;

fn testHyperParams() HyperParams {
    return .{
        .hidden_size = 64,
        .intermediate_size = 128,
        .moe_intermediate_size = 128,
        .num_hidden_layers = 2,
        .num_attention_heads = 4,
        .num_key_value_heads = 1,
        .vocab_size = 1000,
        .max_position_embeddings = 512,
        .kv_lora_rank = 16,
        .q_lora_rank = 16,
        .qk_rope_head_dim = 8,
        .qk_nope_head_dim = 8,
        .v_head_dim = 16,
        .n_routed_experts = 4,
        .n_shared_experts = 1,
        .num_experts_per_tok = 2,
        .routed_scaling_factor = 1.0,
        .norm_topk_prob = true,
        .leading_dense_block_count = 1,
        .rms_norm_eps = 1e-5,
        .rope_freq_base = 1e6,
        .rope_dimension_count = 8,
        .head_dim = 16,
    };
}

test "KVCache init and basic operations" {
    const hp = testHyperParams();
    var kv = try KVCache.init(testing.allocator, hp, 64);
    defer kv.deinit();

    try testing.expectEqual(@as(u32, 4), kv.num_blocks); // (64 + 15) / 16 = 4
    try testing.expectEqual(@as(u32, 4), kv.numFreeBlocks());
    try testing.expectEqual(@as(u32, 0), kv.len);

    try kv.ensureCapacity(20);
    try testing.expectEqual(@as(u32, 2), kv.block_table.items.len); // ceil(20/16)=2
    try testing.expectEqual(@as(u32, 2), kv.numFreeBlocks());
}

test "KVCache put and get" {
    const hp = testHyperParams();
    var kv = try KVCache.init(testing.allocator, hp, 64);
    defer kv.deinit();

    try kv.ensureCapacity(4);

    var kv_comp = try testing.allocator.alloc(f32, hp.kv_lora_rank);
    defer testing.allocator.free(kv_comp);
    for (0..hp.kv_lora_rank) |i| kv_comp[i] = @floatFromInt(i);

    var k_pe_val = try testing.allocator.alloc(f32, hp.qk_rope_head_dim);
    defer testing.allocator.free(k_pe_val);
    for (0..hp.qk_rope_head_dim) |i| k_pe_val[i] = @floatFromInt(i + 100);

    try kv.put(0, 0, kv_comp, k_pe_val);
    try testing.expectEqual(@as(u32, 1), kv.len);

    const got_kv = try kv.getKVCompressed(0, 0);
    for (0..hp.kv_lora_rank) |i| {
        try testing.expectEqual(@as(f32, @floatFromInt(i)), got_kv[i]);
    }

    const got_pe = try kv.getKPe(0, 0);
    for (0..hp.qk_rope_head_dim) |i| {
        try testing.expectEqual(@as(f32, @floatFromInt(i + 100)), got_pe[i]);
    }
}

test "KVCache quantization roundtrip" {
    const hp = testHyperParams();
    var kv = try KVCache.init(testing.allocator, hp, 64);
    defer kv.deinit();
    try kv.enableQuantization();
    try testing.expect(kv.quantized);

    try kv.ensureCapacity(4);

    var kv_comp = try testing.allocator.alloc(f32, hp.kv_lora_rank);
    defer testing.allocator.free(kv_comp);
    for (0..hp.kv_lora_rank) |i| kv_comp[i] = 0.5 * @as(f32, @floatFromInt(i));

    var k_pe_val = try testing.allocator.alloc(f32, hp.qk_rope_head_dim);
    defer testing.allocator.free(k_pe_val);
    for (0..hp.qk_rope_head_dim) |i| k_pe_val[i] = 0.25 * @as(f32, @floatFromInt(i));

    try kv.put(0, 0, kv_comp, k_pe_val);

    const got_kv = try kv.getKVCompressed(0, 0);
    const got_pe = try kv.getKPe(0, 0);

    // Q8_0 量化误差应小于 5%（小值相对误差较大，用绝对阈值兜底）
    for (0..hp.kv_lora_rank) |i| {
        const expected: f32 = 0.5 * @as(f32, @floatFromInt(i));
        const diff = @abs(got_kv[i] - expected);
        try testing.expect(diff < 0.05 * @abs(expected) + 1e-2);
    }
    for (0..hp.qk_rope_head_dim) |i| {
        const expected: f32 = 0.25 * @as(f32, @floatFromInt(i));
        const diff = @abs(got_pe[i] - expected);
        try testing.expect(diff < 0.05 * @abs(expected) + 1e-2);
    }
}

test "KVCache save/load non-quantized" {
    const hp = testHyperParams();
    var kv = try KVCache.init(testing.allocator, hp, 64);
    defer kv.deinit();

    try kv.ensureCapacity(4);

    var kv_comp = try testing.allocator.alloc(f32, hp.kv_lora_rank);
    defer testing.allocator.free(kv_comp);
    for (0..hp.kv_lora_rank) |i| kv_comp[i] = @floatFromInt(i);

    var k_pe_val = try testing.allocator.alloc(f32, hp.qk_rope_head_dim);
    defer testing.allocator.free(k_pe_val);
    for (0..hp.qk_rope_head_dim) |i| k_pe_val[i] = @floatFromInt(i + 100);

    try kv.put(0, 0, kv_comp, k_pe_val);
    try kv.put(1, 1, kv_comp, k_pe_val);

    const test_path = "test_kv_nonq.bin";
    try kv.saveToFile(test_path);
    defer _ = std.c.unlink(test_path);

    // 加载到新实例
    var kv2 = try KVCache.init(testing.allocator, hp, 64);
    defer kv2.deinit();
    try kv2.loadFromFile(test_path);

    try testing.expectEqual(kv.len, kv2.len);
    try testing.expectEqualSlices(u32, kv.block_table.items, kv2.block_table.items);

    // 验证数据
    const got_kv = try kv2.getKVCompressed(0, 0);
    const got_pe = try kv2.getKPe(0, 0);
    for (0..hp.kv_lora_rank) |i| {
        try testing.expectEqual(kv_comp[i], got_kv[i]);
    }
    for (0..hp.qk_rope_head_dim) |i| {
        try testing.expectEqual(k_pe_val[i], got_pe[i]);
    }
}

test "KVCache save/load quantized" {
    const hp = testHyperParams();
    var kv = try KVCache.init(testing.allocator, hp, 64);
    defer kv.deinit();
    try kv.enableQuantization();

    try kv.ensureCapacity(4);

    var kv_comp = try testing.allocator.alloc(f32, hp.kv_lora_rank);
    defer testing.allocator.free(kv_comp);
    for (0..hp.kv_lora_rank) |i| kv_comp[i] = 0.5 * @as(f32, @floatFromInt(i));

    var k_pe_val = try testing.allocator.alloc(f32, hp.qk_rope_head_dim);
    defer testing.allocator.free(k_pe_val);
    for (0..hp.qk_rope_head_dim) |i| k_pe_val[i] = 0.25 * @as(f32, @floatFromInt(i));

    try kv.put(0, 0, kv_comp, k_pe_val);
    try kv.put(1, 1, kv_comp, k_pe_val);

    const test_path = "test_kv_q.bin";
    try kv.saveToFile(test_path);
    defer _ = std.c.unlink(test_path);

    var kv2 = try KVCache.init(testing.allocator, hp, 64);
    defer kv2.deinit();
    try kv2.loadFromFile(test_path);

    try testing.expect(kv2.quantized);
    try testing.expectEqual(kv.len, kv2.len);
    try testing.expectEqualSlices(u32, kv.block_table.items, kv2.block_table.items);
}

test "KVCache checkpoint magic/version mismatch" {
    const hp = testHyperParams();
    var kv = try KVCache.init(testing.allocator, hp, 64);
    defer kv.deinit();

    const test_path = "test_kv_bad.bin";
    const fd = try std.posix.openat(
        std.posix.AT.FDCWD,
        test_path,
        .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true, .CLOEXEC = true },
        0o644,
    );
    defer _ = std.posix.system.close(fd);

    // 写入一个完整但 magic 错误的 header
    const bad_header = CheckpointHeader{
        .magic = .{ 'B', 'A', 'D', '!' },
        .version = 1,
        .len = 0,
        .quantized = 0,
        .block_table_len = 0,
        .num_layers = hp.num_hidden_layers,
        .num_blocks = 4,
        .block_size = BLOCK_SIZE,
        .kv_rank = hp.kv_lora_rank,
        .rope_dim = hp.qk_rope_head_dim,
    };
    try writeAll(fd, std.mem.asBytes(&bad_header));
    defer _ = std.c.unlink(test_path);

    try testing.expectError(error.InvalidCheckpointMagic, kv.loadFromFile(test_path));
}

test "KVCache truncate and reset" {
    const hp = testHyperParams();
    var kv = try KVCache.init(testing.allocator, hp, 64);
    defer kv.deinit();

    try kv.ensureCapacity(35); // 3 blocks
    try testing.expectEqual(@as(u32, 3), kv.block_table.items.len);

    kv.truncate(16); // 1 block needed
    try testing.expectEqual(@as(u32, 1), kv.block_table.items.len);

    kv.reset();
    try testing.expectEqual(@as(u32, 0), kv.block_table.items.len);
    try testing.expectEqual(@as(u32, 0), kv.len);
    try testing.expectEqual(@as(u32, 4), kv.numFreeBlocks());
}
