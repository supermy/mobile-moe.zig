// MoE 调度器：异构专家路由
//
// 冷热专家定义：按推理过程中每层专家的使用频率动态划分
//   Top 20% 高频专家 = 热专家 → GPU 执行
//   其余 80% = 冷专家 → CPU 小核执行
//   热/冷边界随运行时统计滑动更新
//
// 受 KTransformers 延迟专家启发：若热/延迟预算超支，
// 跳过延迟专家并回退到共享"平均专家"偏置

const std = @import("std");
const math = @import("math.zig");
const ThrottleState = @import("thermal.zig").ThrottleState;
const ThrottleLevel = @import("thermal.zig").ThrottleLevel;
const op_perf_db = @import("op_perf_db.zig");
const sync = @import("sync.zig");

/// 专家使用频率统计
pub const ExpertStats = struct {
    /// 每层每个专家的激活计数
    counts: []u64,
    n_experts: u32,
    n_layers: u32,
    total_tokens: u64,

    pub fn init(allocator: std.mem.Allocator, n_layers: u32, n_experts: u32) !ExpertStats {
        const counts = try allocator.alloc(u64, n_layers * n_experts);
        @memset(counts, 0);
        return .{
            .counts = counts,
            .n_experts = n_experts,
            .n_layers = n_layers,
            .total_tokens = 0,
        };
    }

    pub fn deinit(self: *ExpertStats, allocator: std.mem.Allocator) void {
        allocator.free(self.counts);
    }

    /// 记录一次专家激活
    pub fn record(self: *ExpertStats, layer: u32, expert_id: u32) void {
        self.counts[layer * self.n_experts + expert_id] += 1;
    }

    /// 判断某专家是否为热专家（Top 20%）
    pub fn isHot(self: *const ExpertStats, layer: u32, expert_id: u32) bool {
        const threshold = self.hotThreshold(layer);
        return self.counts[layer * self.n_experts + expert_id] >= threshold;
    }

    /// 获取热专家阈值（Top 20% 的最低激活次数）
    fn hotThreshold(self: *const ExpertStats, layer: u32) u64 {
        if (self.total_tokens == 0) return 0;
        const layer_counts = self.counts[layer * self.n_experts ..][0..self.n_experts];
        // 使用线性选择找到 Top-20% 阈值，避免动态分配
        const k = @max(1, self.n_experts / 5);
        return quickSelectThreshold(layer_counts, k);
    }

    /// 线性时间选择：找到降序第 k 大的值（k >= 1）
    fn quickSelectThreshold(layer_counts: []const u64, k: usize) u64 {
        if (layer_counts.len == 0 or k == 0) return 0;

        // 小数组优化：直接复制到栈上排序
        if (layer_counts.len <= 256) {
            var buf: [256]u64 = undefined;
            @memcpy(buf[0..layer_counts.len], layer_counts);
            std.sort.block(u64, buf[0..layer_counts.len], {}, struct {
                pub fn lessThan(_: void, a: u64, b: u64) bool {
                    return a > b;
                }
            }.lessThan);
            return buf[k - 1];
        }
        // 大数组：使用 partition 算法（Hoare）
        var scratch = std.heap.stackFallback(256, std.heap.page_allocator);
        const allocator = scratch.get();
        const buf = allocator.alloc(u64, layer_counts.len) catch return 0;
        defer allocator.free(buf);
        @memcpy(buf, layer_counts);
        return hoareSelect(u64, buf, k - 1, struct {
            pub fn lessThan(_: void, a: u64, b: u64) bool {
                return a > b;
            }
        }.lessThan);
    }

    fn hoareSelect(comptime T: type, items: []T, k: usize, lessThan: fn (void, T, T) bool) T {
        var left: usize = 0;
        var right: usize = items.len - 1;
        while (true) {
            if (left == right) return items[left];
            const pivot_idx = partition(T, items, left, right, lessThan);
            if (k == pivot_idx) {
                return items[k];
            } else if (k < pivot_idx) {
                right = pivot_idx - 1;
            } else {
                left = pivot_idx + 1;
            }
        }
    }

    fn partition(comptime T: type, items: []T, left: usize, right: usize, lessThan: fn (void, T, T) bool) usize {
        const pivot = items[right];
        var i: usize = left;
        for (left..right) |j| {
            if (lessThan({}, items[j], pivot)) {
                std.mem.swap(T, &items[i], &items[j]);
                i += 1;
            }
        }
        std.mem.swap(T, &items[i], &items[right]);
        return i;
    }
};

/// MoE 路由结果
pub const RouteResult = struct {
    /// 选中的专家 ID
    expert_ids: []u32,
    /// 对应的路由权重
    weights: []f32,
    /// 每个专家的执行目标
    targets: []ExecTarget,
    /// 是否有专家被延迟/跳过
    deferred: bool,

    pub const ExecTarget = enum {
        gpu, // 热专家 → GPU
        cpu, // 冷专家 → CPU 小核
        skipped, // 延迟/跳过，回退到共享偏置
    };
};

/// MoE 调度器
pub const Scheduler = struct {
    allocator: std.mem.Allocator,
    stats: ExpertStats,
    throttle: *ThrottleState,
    /// 模型定义的路由缩放因子
    routed_scaling_factor: f32,
    /// 延迟专家回退偏置（预校准）
    deferral_bias: ?[]f32 = null,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, n_layers: u32, n_experts: u32, throttle_state: *ThrottleState, routed_scaling_factor: f32, hidden_size: u32) !Self {
        const deferral_bias = try allocator.alloc(f32, hidden_size);
        errdefer allocator.free(deferral_bias);
        @memset(deferral_bias, 0.0);
        return .{
            .allocator = allocator,
            .stats = try ExpertStats.init(allocator, n_layers, n_experts),
            .throttle = throttle_state,
            .routed_scaling_factor = routed_scaling_factor,
            .deferral_bias = deferral_bias,
        };
    }

    pub fn deinit(self: *Self) void {
        self.stats.deinit(self.allocator);
        if (self.deferral_bias) |b| self.allocator.free(b);
    }

    /// 执行 MoE 路由
    /// gate_scores: [n_experts] 门控网络输出（softmax 前）
    pub fn route(self: *Self, layer: u32, gate_scores: []const f32) !RouteResult {
        const k = @min(self.throttle.active_expert_limit, @as(u32, 4));

        // Top-K 选择
        const top_k = try math.topK(self.allocator, gate_scores, k);
        defer self.allocator.free(top_k);

        const expert_ids = try self.allocator.alloc(u32, k);
        errdefer self.allocator.free(expert_ids);
        const weights = try self.allocator.alloc(f32, k);
        errdefer self.allocator.free(weights);
        const targets = try self.allocator.alloc(RouteResult.ExecTarget, k);
        errdefer self.allocator.free(targets);

        // Softmax 归一化 Top-K 权重（数值稳定：减去最大值）
        var max_score: f32 = top_k[0].score;
        for (top_k[1..]) |entry| {
            if (entry.score > max_score) max_score = entry.score;
        }
        var weight_sum: f32 = 0;
        for (top_k, 0..) |entry, i| {
            expert_ids[i] = entry.index;
            weights[i] = @exp(entry.score - max_score);
            weight_sum += weights[i];
        }
        for (weights) |*w| {
            w.* /= weight_sum;
        }

        // 应用 routed_scaling_factor
        for (weights) |*w| {
            w.* *= self.getRoutedScalingFactor();
        }

        // 决定每个专家的执行目标
        // Bug 8 修复：先收集所有 isHot 结果，再记录激活，避免记录影响同一轮的 isHot 判断
        var deferred = false;
        var is_hot_results = try self.allocator.alloc(bool, k);
        defer self.allocator.free(is_hot_results);

        for (expert_ids, 0..) |eid, i| {
            is_hot_results[i] = self.stats.isHot(layer, eid);
            targets[i] = if (is_hot_results[i]) .gpu else .cpu;

            // 热节流时热专家迁移到 CPU 冷池
            if (is_hot_results[i] and self.throttle.hot_experts_on_cpu) {
                targets[i] = .cpu;
            }
        }

        // 在所有 isHot 判断完成后，再记录专家激活
        for (expert_ids) |eid| {
            self.stats.record(layer, eid);
        }

        // 检查延迟预算：若热节流级别高，跳过部分冷专家
        if (self.throttle.level == .heavy and k > 1) {
            // 最大节流时只保留权重最高的专家
            for (1..k) |i| {
                targets[i] = .skipped;
                deferred = true;
            }
            // 重新归一化剩余权重，避免 FFN 输出幅值偏离
            var remaining_sum: f32 = 0;
            for (0..k) |i| {
                if (targets[i] != .skipped) remaining_sum += weights[i];
            }
            if (remaining_sum > 0) {
                const scale = self.getRoutedScalingFactor() / remaining_sum;
                for (0..k) |i| {
                    if (targets[i] != .skipped) weights[i] *= scale;
                }
            }
        }

        self.stats.total_tokens += 1;

        return .{
            .expert_ids = expert_ids,
            .weights = weights,
            .targets = targets,
            .deferred = deferred,
        };
    }

    fn getRoutedScalingFactor(self: *const Self) f32 {
        // 热节流时降低路由缩放因子以减少路由专家的影响
        return switch (self.throttle.level) {
            .normal => self.routed_scaling_factor,
            .light => self.routed_scaling_factor * 0.83,
            .medium => self.routed_scaling_factor * 0.67,
            .heavy => self.routed_scaling_factor * 0.5,
        };
    }
};

// ========== 算子级异构调度框架（Operator-level Heterogeneous Scheduling） ==========

/// 执行后端类型
pub const Backend = enum(u8) {
    cpu,
    gpu,
    npu,
};

/// 单个算子的调度计划
pub const OpSchedule = struct {
    /// 算子类型
    op: op_perf_db.OpType,
    /// 执行后端
    backend: Backend,
    /// 输入形状
    shape: op_perf_db.Shape,
    /// 精度
    precision: op_perf_db.Precision,
    /// 预估延迟（微秒），0 表示未估算
    estimated_us: u64,
    /// 执行后是否需要同步（多设备并行时，跨设备依赖需同步）
    sync_after: bool,
};

/// 一层内的算子调度计划
pub const LayerPlan = struct {
    ops: std.ArrayList(OpSchedule),
    allocator: std.mem.Allocator,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator) LayerPlan {
        return .{
            .ops = .empty,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        self.ops.deinit(self.allocator);
    }

    pub fn append(self: *Self, schedule: OpSchedule) !void {
        try self.ops.append(self.allocator, schedule);
    }

    /// 计算该层在关键路径上的总预估延迟（按并行后端分组取 max）
    pub fn estimatedCriticalPathUs(self: *const Self) u64 {
        var total: u64 = 0;
        var group_us: u64 = 0;
        var current_backend: ?Backend = null;

        for (self.ops.items) |op| {
            if (current_backend == null or op.sync_after) {
                // 同步点：累加之前组的延迟，开启新组
                total += group_us;
                group_us = op.estimated_us;
                current_backend = op.backend;
            } else if (op.backend == current_backend) {
                group_us += op.estimated_us;
            } else {
                // 不同后端并行：取 max
                group_us = @max(group_us, op.estimated_us);
            }
        }
        total += group_us;
        return total;
    }

    /// 获取该层涉及的设备掩码（用于 sync）
    pub fn deviceMask(self: *const Self) u8 {
        var mask: u8 = 0;
        for (self.ops.items) |op| {
            switch (op.backend) {
                .cpu => mask |= 0b001,
                .gpu => mask |= 0b010,
                .npu => mask |= 0b100,
            }
        }
        return mask;
    }
};

/// 算子级调度器
/// 基于性能数据库和设备可用性，为每层生成算子级执行计划
pub const OpScheduler = struct {
    allocator: std.mem.Allocator,
    /// 性能数据库引用
    perf_db: ?*const op_perf_db.OpPerfDb,
    /// 热节流状态
    throttle: *const ThrottleState,
    /// 设备可用性掩码（bit 0=cpu, 1=gpu, 2=npu）
    device_mask: u8,
    /// 强制回退到 CPU 的算子类型（兼容性/稳定性黑名单）
    cpu_blacklist: std.EnumSet(op_perf_db.OpType),

    const Self = @This();

    pub fn init(
        allocator: std.mem.Allocator,
        perf_db: ?*const op_perf_db.OpPerfDb,
        throttle: *const ThrottleState,
        has_gpu: bool,
        has_npu: bool,
    ) !OpScheduler {
        var cpu_blacklist = std.EnumSet(op_perf_db.OpType).initEmpty();
        // 默认：attention 和 rope 暂放 CPU（需后端验证后再开放）
        cpu_blacklist.insert(.attention);
        cpu_blacklist.insert(.rope);

        return .{
            .allocator = allocator,
            .perf_db = perf_db,
            .throttle = throttle,
            .device_mask = 0b001 |
                (if (has_gpu) @as(u8, 0b010) else 0) |
                (if (has_npu) @as(u8, 0b100) else 0),
            .cpu_blacklist = cpu_blacklist,
        };
    }

    pub fn deinit(self: *Self) void {
        _ = self;
    }

    /// 为单层生成算子调度计划
    /// 假设模型结构：RMSNorm → Attention → RMSNorm → MoE FFN
    pub fn planLayer(
        self: *const Self,
        layer: u32,
        hp: PlanHyperParams,
    ) !LayerPlan {
        var plan = LayerPlan.init(self.allocator);
        errdefer plan.deinit();

        const is_dense = layer < hp.leading_dense_layers;

        // 1. RMSNorm (attn_norm) → CPU（低延迟，迁移开销不划算）
        try plan.append(.{
            .op = .rmsNorm,
            .backend = .cpu,
            .shape = .{ .dim = hp.hidden_size },
            .precision = .f32,
            .estimated_us = self.queryEstUs(.rmsNorm, .{ .dim = hp.hidden_size }, .f32),
            .sync_after = false,
        });

        // 2. MLA Attention → NPU（若可用），否则 CPU
        const attn_backend = if (self.deviceAvailable(.npu) and !self.cpu_blacklist.contains(.attention))
            Backend.npu
        else
            Backend.cpu;
        try plan.append(.{
            .op = .attention,
            .backend = attn_backend,
            .shape = .{ .m = hp.num_heads, .n = hp.head_dim, .k = hp.seq_len_hint },
            .precision = .f32,
            .estimated_us = self.queryEstUs(.attention, .{ .m = hp.num_heads, .n = hp.head_dim, .k = hp.seq_len_hint }, .f32),
            .sync_after = attn_backend != .cpu, // NPU 执行后需同步才能继续 CPU 残差连接
        });

        // 3. RMSNorm (ffn_norm) → CPU
        try plan.append(.{
            .op = .rmsNorm,
            .backend = .cpu,
            .shape = .{ .dim = hp.hidden_size },
            .precision = .f32,
            .estimated_us = self.queryEstUs(.rmsNorm, .{ .dim = hp.hidden_size }, .f32),
            .sync_after = false,
        });

        // 4. FFN
        if (is_dense or hp.n_routed_experts <= 1) {
            // 稠密 FFN：gate/up → NPU（行>列，INT8友好），down → GPU（行<列，FP16友好）
            const gate_backend: Backend = if (self.deviceAvailable(.npu) and !self.isThrottledHeavy()) .npu else .cpu;
            try plan.append(.{
                .op = .ffn_gate_up,
                .backend = gate_backend,
                .shape = .{ .m = hp.intermediate_size, .k = hp.hidden_size },
                .precision = .q8_0,
                .estimated_us = self.queryEstUs(.matVecMulQ8_0, .{ .m = hp.intermediate_size, .k = hp.hidden_size }, .q8_0),
                .sync_after = gate_backend != .cpu,
            });

            // SiLU * up 元素乘法 → 跟随 gate/up 后端或 CPU
            try plan.append(.{
                .op = .siluMul,
                .backend = if (gate_backend == .npu) .cpu else gate_backend, // NPU 不擅长 element-wise，放 CPU
                .shape = .{ .dim = hp.intermediate_size },
                .precision = .f32,
                .estimated_us = self.queryEstUs(.siluMul, .{ .dim = hp.intermediate_size }, .f32),
                .sync_after = false,
            });

            const down_backend: Backend = if (self.deviceAvailable(.gpu) and !self.isThrottledHeavy()) .gpu else .cpu;
            try plan.append(.{
                .op = .ffn_down,
                .backend = down_backend,
                .shape = .{ .m = hp.hidden_size, .k = hp.intermediate_size },
                .precision = .q8_0,
                .estimated_us = self.queryEstUs(.matVecMulQ8_0, .{ .m = hp.hidden_size, .k = hp.intermediate_size }, .q8_0),
                .sync_after = true, // FFN 结束需同步才能残差连接
            });
        } else {
            // MoE FFN：路由在 CPU，各专家按已有 Scheduler.route 的目标执行
            // 这里简化：假设 MoE 调度由上层 Scheduler.route 处理，
            // OpScheduler 仅规划共享专家的 gate/up/down
            const shared_backend: Backend = if (self.deviceAvailable(.npu) and !self.isThrottledHeavy()) .npu else .cpu;
            try plan.append(.{
                .op = .ffn_gate_up,
                .backend = shared_backend,
                .shape = .{ .m = hp.intermediate_size, .k = hp.hidden_size },
                .precision = .q8_0,
                .estimated_us = self.queryEstUs(.matVecMulQ8_0, .{ .m = hp.intermediate_size, .k = hp.hidden_size }, .q8_0),
                .sync_after = shared_backend != .cpu,
            });

            try plan.append(.{
                .op = .siluMul,
                .backend = .cpu,
                .shape = .{ .dim = hp.intermediate_size },
                .precision = .f32,
                .estimated_us = self.queryEstUs(.siluMul, .{ .dim = hp.intermediate_size }, .f32),
                .sync_after = false,
            });

            try plan.append(.{
                .op = .ffn_down,
                .backend = if (self.deviceAvailable(.gpu) and !self.isThrottledHeavy()) .gpu else .cpu,
                .shape = .{ .m = hp.hidden_size, .k = hp.intermediate_size },
                .precision = .q8_0,
                .estimated_us = self.queryEstUs(.matVecMulQ8_0, .{ .m = hp.hidden_size, .k = hp.intermediate_size }, .q8_0),
                .sync_after = true,
            });
        }

        return plan;
    }

    fn queryEstUs(self: *const Self, op: op_perf_db.OpType, shape: op_perf_db.Shape, precision: op_perf_db.Precision) u64 {
        if (self.perf_db) |db| {
            const key = op_perf_db.OpKey{ .op = op, .shape = shape, .precision = precision };
            const entry = db.query(key);
            if (entry) |e| return e.cpu_us;
        }
        // 无数据库时返回典型经验值（微秒）
        return fallbackEstimate(op, shape);
    }

    fn deviceAvailable(self: *const Self, backend: Backend) bool {
        return switch (backend) {
            .cpu => self.device_mask & 0b001 != 0,
            .gpu => self.device_mask & 0b010 != 0,
            .npu => self.device_mask & 0b100 != 0,
        };
    }

    fn isThrottledHeavy(self: *const Self) bool {
        return self.throttle.level == .heavy;
    }

    /// 无性能数据库时的经验估值（用于调度器初始化前的快速路径）
    fn fallbackEstimate(op: op_perf_db.OpType, shape: op_perf_db.Shape) u64 {
        const dim = @max(shape.m, shape.n);
        return switch (op) {
            .rmsNorm => dim / 40, // ~50us for 2048
            .matVecMulQ8_0 => (shape.m * shape.k) / 40000, // ~150us for [1536,2048]
            .matMulQ8_0 => (shape.m * shape.n * shape.k) / 8000000, // batch 分摊
            .siluMul => dim / 80,
            .attention => (shape.m * shape.n * shape.k) / 2000000,
            .ffn_gate_up => (shape.m * shape.k) / 40000,
            .ffn_down => (shape.m * shape.k) / 35000,
            else => 100,
        };
    }
};

/// 生成 LayerPlan 所需的超参数子集（避免依赖完整 ModelWeights）
pub const PlanHyperParams = struct {
    hidden_size: u32,
    intermediate_size: u32,
    num_heads: u32,
    head_dim: u32,
    seq_len_hint: u32 = 128,
    leading_dense_layers: u32,
    n_routed_experts: u32,
};

// ========== 测试 ==========

test "OpSchedule basic" {
    const testing = std.testing;
    const schedule = OpSchedule{
        .op = .rmsNorm,
        .backend = .cpu,
        .shape = .{ .dim = 2048 },
        .precision = .f32,
        .estimated_us = 50,
        .sync_after = false,
    };
    try testing.expectEqual(op_perf_db.OpType.rmsNorm, schedule.op);
    try testing.expectEqual(Backend.cpu, schedule.backend);
}

test "LayerPlan build and estimate" {
    const testing = std.testing;
    var plan = LayerPlan.init(testing.allocator);
    defer plan.deinit();

    try plan.append(.{
        .op = .rmsNorm,
        .backend = .cpu,
        .shape = .{ .dim = 2048 },
        .precision = .f32,
        .estimated_us = 50,
        .sync_after = false,
    });
    try plan.append(.{
        .op = .matVecMulQ8_0,
        .backend = .npu,
        .shape = .{ .m = 1536, .k = 2048 },
        .precision = .q8_0,
        .estimated_us = 150,
        .sync_after = true,
    });
    try plan.append(.{
        .op = .siluMul,
        .backend = .cpu,
        .shape = .{ .dim = 1536 },
        .precision = .f32,
        .estimated_us = 20,
        .sync_after = false,
    });

    try testing.expectEqual(@as(usize, 3), plan.ops.items.len);
    const est = plan.estimatedCriticalPathUs();
    try testing.expect(est > 0);
}

test "LayerPlan deviceMask" {
    const testing = std.testing;
    var plan = LayerPlan.init(testing.allocator);
    defer plan.deinit();

    try plan.append(.{ .op = .rmsNorm, .backend = .cpu, .shape = .{}, .precision = .f32, .estimated_us = 10, .sync_after = false });
    try plan.append(.{ .op = .ffn_gate_up, .backend = .npu, .shape = .{}, .precision = .q8_0, .estimated_us = 20, .sync_after = false });
    try plan.append(.{ .op = .ffn_down, .backend = .gpu, .shape = .{}, .precision = .q8_0, .estimated_us = 30, .sync_after = false });

    const mask = plan.deviceMask();
    try testing.expectEqual(@as(u8, 0b111), mask);
}

test "OpScheduler planLayer all CPU when no accel" {
    const testing = std.testing;
    var throttle = ThrottleState.init();

    var sched = try OpScheduler.init(testing.allocator, null, &throttle, false, false);
    defer sched.deinit();

    const hp = PlanHyperParams{
        .hidden_size = 2048,
        .intermediate_size = 1536,
        .num_heads = 16,
        .head_dim = 128,
        .leading_dense_layers = 1,
        .n_routed_experts = 64,
    };

    var plan = try sched.planLayer(0, hp);
    defer plan.deinit();

    try testing.expect(plan.ops.items.len > 0);
    for (plan.ops.items) |op| {
        try testing.expectEqual(Backend.cpu, op.backend);
    }
}

test "OpScheduler planLayer uses NPU and GPU when available" {
    const testing = std.testing;
    var throttle = ThrottleState.init();

    var sched = try OpScheduler.init(testing.allocator, null, &throttle, true, true);
    defer sched.deinit();

    const hp = PlanHyperParams{
        .hidden_size = 2048,
        .intermediate_size = 1536,
        .num_heads = 16,
        .head_dim = 128,
        .leading_dense_layers = 1,
        .n_routed_experts = 64,
    };

    var plan = try sched.planLayer(0, hp);
    defer plan.deinit();

    var has_npu = false;
    var has_gpu = false;
    for (plan.ops.items) |op| {
        if (op.backend == .npu) has_npu = true;
        if (op.backend == .gpu) has_gpu = true;
    }
    // 稠密层应有 NPU (gate_up) 和 GPU (down)
    try testing.expect(has_npu);
    try testing.expect(has_gpu);
}

test "OpScheduler heavy throttle falls back to CPU" {
    const testing = std.testing;
    var throttle = ThrottleState.init();
    throttle.throttle(.heavy);

    var sched = try OpScheduler.init(testing.allocator, null, &throttle, true, true);
    defer sched.deinit();

    const hp = PlanHyperParams{
        .hidden_size = 2048,
        .intermediate_size = 1536,
        .num_heads = 16,
        .head_dim = 128,
        .leading_dense_layers = 1,
        .n_routed_experts = 64,
    };

    var plan = try sched.planLayer(0, hp);
    defer plan.deinit();

    for (plan.ops.items) |op| {
        // attention 在黑名单中，本来就是 CPU；其余 heavy 时应回退 CPU
        if (op.op != .attention) {
            try testing.expectEqual(Backend.cpu, op.backend);
        }
    }
}

test "OpScheduler with perf db uses recorded values" {
    const testing = std.testing;
    var db = op_perf_db.OpPerfDb.init(testing.allocator);
    defer db.deinit();

    const key = op_perf_db.OpKey{
        .op = .rmsNorm,
        .shape = .{ .dim = 2048 },
        .precision = .f32,
    };
    try db.record(key, .{ .cpu_us = 42, .iterations = 10 });

    var throttle = ThrottleState.init();
    var sched = try OpScheduler.init(testing.allocator, &db, &throttle, false, false);
    defer sched.deinit();

    const hp = PlanHyperParams{
        .hidden_size = 2048,
        .intermediate_size = 1536,
        .num_heads = 16,
        .head_dim = 128,
        .leading_dense_layers = 1,
        .n_routed_experts = 64,
    };

    var plan = try sched.planLayer(0, hp);
    defer plan.deinit();

    // 查找 rmsNorm 条目并验证使用了数据库中的 42us
    for (plan.ops.items) |op| {
        if (op.op == .rmsNorm) {
            try testing.expectEqual(@as(u64, 42), op.estimated_us);
        }
    }
}
