// Continuous Batching 骨架：多请求序列调度
//
// 当前为 Phase 1 骨架：管理多个 Sequence 的生命周期，
// 提供 batch 合并接口，但调度策略为简单轮询。
// 未来扩展：动态抢占、KV Cache 页回收、优先级调度。

const std = @import("std");
const kv_cache = @import("kv_cache.zig");
const model = @import("model.zig");
const c = std.c;

/// Zig 0.16 兼容：获取单调毫秒时间戳
fn getMonotonicMs() i64 {
    var ts: c.timespec = undefined;
    _ = c.clock_gettime(c.CLOCK.MONOTONIC, &ts);
    return @as(i64, ts.sec) * 1000 + @divTrunc(@as(i64, ts.nsec), 1_000_000);
}

pub const SequenceStatus = enum {
    waiting, // 等待调度
    running, // 正在生成
    paused, // 因资源限制暂停
    completed, // 已结束（EOS 或达到 max_tokens）
};

pub const SequencePriority = enum {
    low,
    normal,
    high,
    critical,
};

pub const Sequence = struct {
    id: u64,
    status: SequenceStatus,
    priority: SequencePriority,
    tokens: std.ArrayList(u32),
    max_tokens: u32,
    kv: kv_cache.KVCache,
    /// 创建时间戳（用于相同优先级下的 FIFO）
    created_at: u64,

    pub fn init(allocator: std.mem.Allocator, seq_id: u64, hp: model.HyperParams, max_seq_len: u32, max_tokens: u32) !Sequence {
        return .{
            .id = seq_id,
            .status = .waiting,
            .priority = .normal,
            .tokens = .empty,
            .max_tokens = max_tokens,
            .kv = try kv_cache.KVCache.init(allocator, hp, max_seq_len),
            .created_at = @intCast(getMonotonicMs()),
        };
    }

    pub fn initWithPriority(allocator: std.mem.Allocator, seq_id: u64, hp: model.HyperParams, max_seq_len: u32, max_tokens: u32, priority: SequencePriority) !Sequence {
        var seq = try init(allocator, seq_id, hp, max_seq_len, max_tokens);
        seq.priority = priority;
        return seq;
    }

    pub fn deinit(self: *Sequence, allocator: std.mem.Allocator) void {
        self.tokens.deinit(allocator);
        self.kv.deinit();
    }

    pub fn isFinished(self: *const Sequence, eos_token: u32) bool {
        if (self.status == .completed) return true;
        if (self.tokens.items.len >= self.max_tokens) return true;
        if (self.tokens.items.len > 0 and self.tokens.items[self.tokens.items.len - 1] == eos_token) return true;
        return false;
    }
};

/// 批量调度器：管理多个 Sequence，将 waiting 序列合并为 batch
pub const BatchScheduler = struct {
    allocator: std.mem.Allocator,
    sequences: std.ArrayList(Sequence),
    next_id: u64,
    max_batch_size: usize,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, max_batch_size: usize) Self {
        return .{
            .allocator = allocator,
            .sequences = .empty,
            .next_id = 1,
            .max_batch_size = max_batch_size,
        };
    }

    pub fn deinit(self: *Self) void {
        for (self.sequences.items) |*seq| {
            seq.deinit(self.allocator);
        }
        self.sequences.deinit(self.allocator);
    }

    /// 添加新请求到 waiting 队列（默认 normal 优先级）
    pub fn addRequest(self: *Self, hp: model.HyperParams, max_seq_len: u32, max_tokens: u32) !u64 {
        return self.addRequestWithPriority(hp, max_seq_len, max_tokens, .normal);
    }

    /// 添加指定优先级的新请求
    pub fn addRequestWithPriority(self: *Self, hp: model.HyperParams, max_seq_len: u32, max_tokens: u32, priority: SequencePriority) !u64 {
        const id = self.next_id;
        self.next_id += 1;
        const seq = try Sequence.initWithPriority(self.allocator, id, hp, max_seq_len, max_tokens, priority);
        try self.sequences.append(self.allocator, seq);
        return id;
    }

    /// 尝试回收 paused/completed 序列的 KV Cache block，以释放资源给高优先级请求
    /// 返回成功回收的 block 数
    pub fn reclaimBlocks(self: *Self) u32 {
        var reclaimed: u32 = 0;
        var i: usize = 0;
        while (i < self.sequences.items.len) {
            const seq = &self.sequences.items[i];
            if (seq.status == .paused or seq.status == .completed) {
                // 释放该序列占用的所有 block
                const before = seq.kv.numFreeBlocks();
                seq.kv.reset();
                const after = seq.kv.numFreeBlocks();
                reclaimed += after - before;
            }
            i += 1;
        }
        return reclaimed;
    }

    /// 获取当前可合并为 batch 的序列索引（running + waiting，不超过 max_batch_size）
    /// 优先级排序：critical > high > normal > low，同优先级按 created_at FIFO
    pub fn collectBatchIndices(self: *Self, eos_token: u32) ![]usize {
        var indices = try self.allocator.alloc(usize, self.max_batch_size);
        errdefer self.allocator.free(indices);

        // 先收集所有候选序列
        var candidates = try self.allocator.alloc(usize, self.sequences.items.len);
        defer self.allocator.free(candidates);

        var count: usize = 0;
        for (self.sequences.items, 0..) |*seq, i| {
            if (seq.status == .running or seq.status == .waiting) {
                if (!seq.isFinished(eos_token)) {
                    candidates[count] = i;
                    count += 1;
                }
            }
        }

        // 按优先级排序（简单选择排序，候选数通常很小）
        if (count > 1) {
            for (0..count - 1) |i| {
                var max_idx = i;
                for (i + 1..count) |j| {
                    const seq_a = &self.sequences.items[candidates[max_idx]];
                    const seq_b = &self.sequences.items[candidates[j]];
                    if (priorityValue(seq_b.priority) > priorityValue(seq_a.priority)) {
                        max_idx = j;
                    } else if (priorityValue(seq_b.priority) == priorityValue(seq_a.priority) and
                        seq_b.created_at < seq_a.created_at)
                    {
                        max_idx = j;
                    }
                }
                if (max_idx != i) {
                    const tmp = candidates[i];
                    candidates[i] = candidates[max_idx];
                    candidates[max_idx] = tmp;
                }
            }
        }

        // 取前 max_batch_size 个
        const result_count = @min(count, self.max_batch_size);
        for (0..result_count) |i| {
            indices[i] = candidates[i];
        }

        return try self.allocator.realloc(indices, result_count);
    }

    fn priorityValue(p: SequencePriority) u8 {
        return switch (p) {
            .low => 0,
            .normal => 1,
            .high => 2,
            .critical => 3,
        };
    }

    /// 标记指定序列为 running（调度前调用）
    pub fn markRunning(self: *Self, indices: []const usize) void {
        for (indices) |i| {
            if (self.sequences.items[i].status == .waiting) {
                self.sequences.items[i].status = .running;
            }
        }
    }

    /// 标记指定序列为 completed（收到 EOS 或 max_tokens）
    pub fn markCompleted(self: *Self, seq_id: u64) void {
        for (self.sequences.items) |*seq| {
            if (seq.id == seq_id) {
                seq.status = .completed;
                break;
            }
        }
    }

    /// 清理已完成的序列（释放 KV Cache）
    pub fn reapCompleted(self: *Self) void {
        var i: usize = 0;
        while (i < self.sequences.items.len) {
            if (self.sequences.items[i].status == .completed) {
                var seq = self.sequences.orderedRemove(i);
                seq.deinit(self.allocator);
            } else {
                i += 1;
            }
        }
    }

    pub fn runningCount(self: *const Self) usize {
        var count: usize = 0;
        for (self.sequences.items) |seq| {
            if (seq.status == .running) count += 1;
        }
        return count;
    }
};

test "BatchScheduler lifecycle" {
    const testing = std.testing;
    var sched = BatchScheduler.init(testing.allocator, 4);
    defer sched.deinit();

    const hp = model.HyperParams{
        .hidden_size = 128,
        .intermediate_size = 64,
        .moe_intermediate_size = 64,
        .num_attention_heads = 4,
        .num_key_value_heads = 4,
        .head_dim = 32,
        .qk_rope_head_dim = 8,
        .qk_nope_head_dim = 24,
        .v_head_dim = 32,
        .num_hidden_layers = 2,
        .vocab_size = 100,
        .max_position_embeddings = 128,
        .kv_lora_rank = 16,
        .q_lora_rank = 16,
        .n_routed_experts = 4,
        .n_shared_experts = 1,
        .num_experts_per_tok = 2,
        .routed_scaling_factor = 1.0,
        .norm_topk_prob = false,
        .leading_dense_block_count = 1,
        .rms_norm_eps = 1e-5,
        .rope_freq_base = 10000.0,
        .rope_dimension_count = 8,
    };

    const id1 = try sched.addRequest(hp, 64, 32);
    const id2 = try sched.addRequest(hp, 64, 32);
    try testing.expectEqual(@as(u64, 1), id1);
    try testing.expectEqual(@as(u64, 2), id2);

    const indices = try sched.collectBatchIndices(0);
    defer testing.allocator.free(indices);
    try testing.expectEqual(@as(usize, 2), indices.len);

    sched.markRunning(indices);
    try testing.expectEqual(@as(usize, 2), sched.runningCount());

    sched.markCompleted(id1);
    sched.reapCompleted();
    try testing.expectEqual(@as(usize, 1), sched.sequences.items.len);
}

test "BatchScheduler priority ordering" {
    const testing = std.testing;
    var sched = BatchScheduler.init(testing.allocator, 2);
    defer sched.deinit();

    const hp = model.HyperParams{
        .hidden_size = 128,
        .intermediate_size = 64,
        .moe_intermediate_size = 64,
        .num_attention_heads = 4,
        .num_key_value_heads = 4,
        .head_dim = 32,
        .qk_rope_head_dim = 8,
        .qk_nope_head_dim = 24,
        .v_head_dim = 32,
        .num_hidden_layers = 2,
        .vocab_size = 100,
        .max_position_embeddings = 128,
        .kv_lora_rank = 16,
        .q_lora_rank = 16,
        .n_routed_experts = 4,
        .n_shared_experts = 1,
        .num_experts_per_tok = 2,
        .routed_scaling_factor = 1.0,
        .norm_topk_prob = false,
        .leading_dense_block_count = 1,
        .rms_norm_eps = 1e-5,
        .rope_freq_base = 10000.0,
        .rope_dimension_count = 8,
    };

    // 添加 3 个请求：low, critical, normal
    _ = try sched.addRequestWithPriority(hp, 64, 32, .low);
    const id_critical = try sched.addRequestWithPriority(hp, 64, 32, .critical);
    const id_normal = try sched.addRequestWithPriority(hp, 64, 32, .normal);

    // max_batch_size = 2，应选出 critical 和 normal
    const indices = try sched.collectBatchIndices(0);
    defer testing.allocator.free(indices);
    try testing.expectEqual(@as(usize, 2), indices.len);

    // 验证第一个被选中的序列是 critical 优先级
    const first_seq = sched.sequences.items[indices[0]];
    try testing.expectEqual(SequencePriority.critical, first_seq.priority);
    try testing.expectEqual(id_critical, first_seq.id);

    // 验证第二个是 normal（优先级高于 low）
    const second_seq = sched.sequences.items[indices[1]];
    try testing.expectEqual(SequencePriority.normal, second_seq.priority);
    try testing.expectEqual(id_normal, second_seq.id);
}

test "BatchScheduler KV block reclaim" {
    const testing = std.testing;
    var sched = BatchScheduler.init(testing.allocator, 4);
    defer sched.deinit();

    const hp = model.HyperParams{
        .hidden_size = 128,
        .intermediate_size = 64,
        .moe_intermediate_size = 64,
        .num_attention_heads = 4,
        .num_key_value_heads = 4,
        .head_dim = 32,
        .qk_rope_head_dim = 8,
        .qk_nope_head_dim = 24,
        .v_head_dim = 32,
        .num_hidden_layers = 2,
        .vocab_size = 100,
        .max_position_embeddings = 128,
        .kv_lora_rank = 16,
        .q_lora_rank = 16,
        .n_routed_experts = 4,
        .n_shared_experts = 1,
        .num_experts_per_tok = 2,
        .routed_scaling_factor = 1.0,
        .norm_topk_prob = false,
        .leading_dense_block_count = 1,
        .rms_norm_eps = 1e-5,
        .rope_freq_base = 10000.0,
        .rope_dimension_count = 8,
    };

    const id1 = try sched.addRequest(hp, 64, 32);
    _ = try sched.addRequest(hp, 64, 32);

    // 预先为序列分配 KV cache block（模拟推理过程）
    try sched.sequences.items[0].kv.ensureCapacity(32);
    try sched.sequences.items[1].kv.ensureCapacity(32);

    // 模拟两个请求都 running，然后一个 completed
    const indices = try sched.collectBatchIndices(0);
    defer testing.allocator.free(indices);
    sched.markRunning(indices);

    sched.markCompleted(id1);

    // 回收前：completed 序列（items[0]）占用 block
    const before_free = sched.sequences.items[0].kv.numFreeBlocks();

    // 回收 paused/completed 序列的 block
    const reclaimed = sched.reclaimBlocks();
    try testing.expect(reclaimed > 0);

    // 回收后：completed 序列的 block 被释放回其 free_list
    const after_free = sched.sequences.items[0].kv.numFreeBlocks();
    try testing.expect(after_free > before_free);
}
