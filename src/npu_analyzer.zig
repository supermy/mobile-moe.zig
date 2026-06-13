// NPU 算子友好性分析框架
//
// 参考 HeteroLLM 洞察：NPU（Hexagon HTP）对 MatMul 形状极度敏感：
//   - 行 > 列（如 [1, 2048] x [2048, 1536]）→ INT8 友好，延迟低
//   - 行 < 列（如 [1, 1536] x [1536, 2048]）→ NPU 不友好，应走 GPU
//
// 本模块提供离线/在线分析能力，为异构调度器提供决策输入。

const std = @import("std");
const op_perf_db = @import("op_perf_db.zig");

/// NPU 友好性评分（0.0 = 极不友好，1.0 = 极友好）
pub const FriendlinessScore = f32;

/// 形状敏感性规则（基于 HeteroLLM 在骁龙 8 Gen 3 上的实测数据）
pub const ShapeRule = struct {
    /// 最小推荐批大小（NPU 批处理优势阈值）
    min_batch_npu: u32 = 4,
    /// 行/列比例阈值：当 M/N < 此值时，NPU 效率显著下降
    row_col_ratio_threshold: f32 = 0.8,
    /// 最大推荐 K 维度（过大的 K 导致 NPU 寄存器压力）
    max_k_dim: u32 = 4096,
};

/// 算子-NPU 兼容性分析器
pub const NpuAnalyzer = struct {
    rules: ShapeRule,
    /// 可选：接入实际性能数据库做数据驱动决策
    perf_db: ?*op_perf_db.OpPerfDb,

    const Self = @This();

    pub fn init(rules: ShapeRule, perf_db: ?*op_perf_db.OpPerfDb) Self {
        return .{
            .rules = rules,
            .perf_db = perf_db,
        };
    }

    /// 分析 MatMul 形状在 NPU 上的友好性
    /// m: 输出行数（通常 = batch=1 或 seq_len）
    /// n: 输出列数（通常 = hidden / intermediate）
    /// k: 收缩维度
    pub fn analyzeMatMul(self: *const Self, m: u32, n: u32, k: u32, precision: op_perf_db.Precision) FriendlinessScore {
        var score: f32 = 1.0;

        // 1. 批大小惩罚：单 token（m=1）NPU 利用率低
        if (m < self.rules.min_batch_npu) {
            const penalty = 1.0 - (@as(f32, @floatFromInt(m)) / @as(f32, @floatFromInt(self.rules.min_batch_npu)));
            score *= (1.0 - penalty * 0.4); // 最大 40% 惩罚
        }

        // 2. 行/列比例惩罚：M < N 时 NPU 不友好
        if (n > 0) {
            const ratio = @as(f32, @floatFromInt(m)) / @as(f32, @floatFromInt(n));
            if (ratio < self.rules.row_col_ratio_threshold) {
                const penalty = (self.rules.row_col_ratio_threshold - ratio) / self.rules.row_col_ratio_threshold;
                score *= (1.0 - @min(penalty, 1.0) * 0.5); // 最大 50% 惩罚
            }
        }

        // 3. K 维度惩罚：过大的 K 导致 HTP 寄存器 spill
        if (k > self.rules.max_k_dim) {
            const penalty = @as(f32, @floatFromInt(k - self.rules.max_k_dim)) / @as(f32, @floatFromInt(self.rules.max_k_dim));
            score *= (1.0 - @min(penalty, 1.0) * 0.3); // 最大 30% 惩罚
        }

        // 4. 精度奖励：NPU 对 INT8/Q8_0 最友好
        score *= switch (precision) {
            .q8_0 => 1.0,
            .q4_k => 0.95,
            .f16 => 0.85,
            .f32 => 0.6,
        };

        // 5. 如果有实测数据，加权融合
        if (self.perf_db) |db| {
            const key = op_perf_db.OpKey{
                .op = .matMulQ8_0,
                .shape = .{ .m = m, .n = n, .k = k },
                .precision = precision,
            };
            if (db.query(key)) |entry| {
                // 如果 NPU 延迟显著高于 GPU，降低分数
                if (entry.gpu_us > 0 and entry.npu_us > @as(u64, @intFromFloat(@as(f32, @floatFromInt(entry.gpu_us)) * 1.2))) {
                    const ratio = @as(f32, @floatFromInt(entry.npu_us)) / @as(f32, @floatFromInt(entry.gpu_us));
                    score *= (1.0 / ratio);
                }
            }
        }

        return @max(0.0, @min(1.0, score));
    }

    /// 分析 FFN down_proj 是否应走 GPU（HeteroLLM 核心洞察）
    pub fn shouldDownProjUseGpu(self: *const Self, hidden_size: u32, intermediate_size: u32) bool {
        // down_proj: [1, intermediate] x [intermediate, hidden]
        // 对 GLM-4.7-Flash：intermediate=1536, hidden=2048
        // M=1, N=2048, K=1536 → M/N = 0.0005 << 0.8，强烈建议 GPU
        const score = self.analyzeMatMul(1, hidden_size, intermediate_size, .q8_0);
        return score < 0.5;
    }

    /// 分析 Attention QK^T 在 NPU 上的友好性
    pub fn analyzeAttentionQK(self: *const Self, seq_len: u32, head_dim: u32) FriendlinessScore {
        // QK^T: [seq_len, head_dim] x [head_dim, seq_len]
        // 当 seq_len 较大时，NPU 批处理优势显现
        return self.analyzeMatMul(seq_len, seq_len, head_dim, .q8_0);
    }

    /// 生成完整层分析报告
    pub fn analyzeLayer(
        self: *const Self,
        allocator: std.mem.Allocator,
        hidden_size: u32,
        intermediate_size: u32,
        seq_len: u32,
        head_dim: u32,
    ) !LayerAnalysis {
        var report = LayerAnalysis{
            .gate_up_score = self.analyzeMatMul(seq_len, intermediate_size, hidden_size, .q8_0),
            .down_score = self.analyzeMatMul(seq_len, hidden_size, intermediate_size, .q8_0),
            .attn_qk_score = self.analyzeAttentionQK(seq_len, head_dim),
            .recommendations = .empty,
            .allocator = allocator,
        };
        errdefer report.recommendations.deinit(allocator);

        var buf: [256]u8 = undefined;
        if (report.down_score < 0.5) {
            const msg = try std.fmt.bufPrint(&buf, "down_proj NPU 友好度 {d:.2}，建议走 GPU\n", .{report.down_score});
            try report.recommendations.appendSlice(allocator, msg);
        }
        if (report.gate_up_score > 0.8) {
            const msg = try std.fmt.bufPrint(&buf, "gate/up_proj NPU 友好度 {d:.2}，建议走 NPU\n", .{report.gate_up_score});
            try report.recommendations.appendSlice(allocator, msg);
        }
        if (report.attn_qk_score > 0.7) {
            const msg = try std.fmt.bufPrint(&buf, "Attention QK^T NPU 友好度 {d:.2}，建议走 NPU\n", .{report.attn_qk_score});
            try report.recommendations.appendSlice(allocator, msg);
        }

        return report;
    }
};

/// 单层分析结果
pub const LayerAnalysis = struct {
    gate_up_score: FriendlinessScore,
    down_score: FriendlinessScore,
    attn_qk_score: FriendlinessScore,
    recommendations: std.ArrayList(u8),
    allocator: std.mem.Allocator,

    pub fn deinit(self: *LayerAnalysis) void {
        self.recommendations.deinit(self.allocator);
    }
};

// ========== TDD ==========

test "NpuAnalyzer MatMul 形状评分" {
    const analyzer = NpuAnalyzer.init(.{}, null);

    // gate_proj: [1, 2048] x [2048, 1536] → M=1, N=1536, K=2048
    // 行>列（1/1536 很小），应较低分
    const gate_score = analyzer.analyzeMatMul(1, 1536, 2048, .q8_0);
    try std.testing.expect(gate_score < 0.7);

    // batch=64 时同一形状，应更高分
    const batch_score = analyzer.analyzeMatMul(64, 1536, 2048, .q8_0);
    try std.testing.expect(batch_score > gate_score);

    // down_proj: [1, 1536] x [1536, 2048] → M=1, N=2048, K=1536
    // 行<列，应更低分
    const down_score = analyzer.analyzeMatMul(1, 2048, 1536, .q8_0);
    try std.testing.expect(down_score < gate_score);

    // f32 精度惩罚
    const f32_score = analyzer.analyzeMatMul(1, 1536, 2048, .f32);
    try std.testing.expect(f32_score < gate_score);
}

test "NpuAnalyzer shouldDownProjUseGpu" {
    const analyzer = NpuAnalyzer.init(.{}, null);

    // GLM-4.7-Flash 配置：hidden=2048, intermediate=1536
    // down_proj: [1, 1536] x [1536, 2048] → 行<列，应建议 GPU
    const use_gpu = analyzer.shouldDownProjUseGpu(2048, 1536);
    try std.testing.expect(use_gpu);

    // 反向配置：hidden=512, intermediate=2048
    // down_proj: [1, 2048] x [2048, 512] → 行<<列，NPU 仍不友好，建议 GPU
    const use_gpu2 = analyzer.shouldDownProjUseGpu(512, 2048);
    try std.testing.expect(use_gpu2);
}

test "NpuAnalyzer analyzeLayer 报告生成" {
    const allocator = std.testing.allocator;
    var analyzer = NpuAnalyzer.init(.{}, null);

    var report = try analyzer.analyzeLayer(allocator, 2048, 1536, 1, 64);
    defer report.deinit();

    try std.testing.expect(report.recommendations.items.len > 0);
    std.debug.print("NPU 分析建议:\n{s}\n", .{report.recommendations.items});
}
