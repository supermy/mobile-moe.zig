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

    pub fn init(allocator: std.mem.Allocator, n_layers: u32, n_experts: u32, throttle_state: *ThrottleState, routed_scaling_factor: f32) !Self {
        return .{
            .allocator = allocator,
            .stats = try ExpertStats.init(allocator, n_layers, n_experts),
            .throttle = throttle_state,
            .routed_scaling_factor = routed_scaling_factor,
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
        const weights = try self.allocator.alloc(f32, k);
        const targets = try self.allocator.alloc(RouteResult.ExecTarget, k);

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
