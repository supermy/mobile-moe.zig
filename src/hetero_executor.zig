// 异构执行器 Phase 3：CPU / GPU / NPU 算子级多设备并行
//
// 目标：同一层内实现真正的流水线并行：
//   - Attention → NPU
//   - gate/up → NPU / GPU
//   - down → GPU
//   - 路由 / RMSNorm / SiLU → CPU
//
// 使用 sync.zig 的 SyncBarrier + PredictiveSync 实现低延迟同步。

const std = @import("std");
const scheduler = @import("scheduler.zig");
const sync = @import("sync.zig");
const op_perf_db = @import("op_perf_db.zig");
const math = @import("math.zig");

/// 设备可用性掩码
pub const DeviceMask = struct {
    cpu: bool = true,
    gpu: bool = false,
    npu: bool = false,
};

/// 设备后端接口（VTable 模式，允许运行时切换 Metal/OpenCL/CPU）
pub const DeviceBackend = union(enum) {
    cpu,
    metal: *anyopaque, // MetalBackend*
    opencl: *anyopaque, // OpenCLBackend*
    npu_dummy, // NPU 占位，待 QNN SDK 接入
};

/// 异构执行器：管理一层内的多设备并行执行
pub const HeteroExecutor = struct {
    allocator: std.mem.Allocator,
    /// 当前可用设备掩码
    device_mask: DeviceMask,
    /// 同步屏障（每层一个）
    barrier: sync.SyncBarrier,
    /// 预测性同步器
    predictive: sync.PredictiveSync,
    /// 性能数据库引用
    perf_db: ?*op_perf_db.OpPerfDb,
    /// 后端实例
    backends: Backends,

    const Self = @This();

    pub const Backends = struct {
        cpu: bool = true,
        metal: ?*anyopaque = null,
        opencl: ?*anyopaque = null,
        npu: ?*anyopaque = null,
    };

    pub fn init(
        allocator: std.mem.Allocator,
        device_mask: DeviceMask,
        perf_db: ?*op_perf_db.OpPerfDb,
        backends: Backends,
    ) !Self {
        return .{
            .allocator = allocator,
            .device_mask = device_mask,
            .barrier = sync.SyncBarrier.init(),
            .predictive = sync.PredictiveSync.init(perf_db),
            .perf_db = perf_db,
            .backends = backends,
        };
    }

    pub fn deinit(self: *Self) void {
        // SyncBarrier 无堆分配，无需显式释放
        _ = self;
    }

    /// 执行单层的前向传播（算子级异构并行）
    ///
    /// 执行流程：
    ///   1. 用 OpScheduler 生成 LayerPlan
    ///   2. 按 sync_after 分组，并行组内算子同时下发到不同设备
    ///   3. 每组结束后 SyncBarrier.waitAll
    ///   4. 预测性同步减少空等时间
    pub fn executeLayer(
        self: *Self,
        plan: scheduler.LayerPlan,
        hidden: []f32,
        layer: u32,
        pos: u32,
    ) !void {
        _ = layer;
        _ = pos;

        var group_start: usize = 0;
        while (group_start < plan.ops.items.len) {
            // 找到下一个 sync_after=true 的位置
            var group_end = group_start;
            while (group_end < plan.ops.items.len and !plan.ops.items[group_end].sync_after) {
                group_end += 1;
            }
            if (group_end < plan.ops.items.len) group_end += 1; // 包含 sync 点

            // 重置屏障（本组涉及的设备数）
            self.barrier.reset();

            // 并行下发本组所有算子
            for (plan.ops.items[group_start..group_end]) |op| {
                switch (op.backend) {
                    .cpu => try self.dispatchCpu(op, hidden),
                    .gpu => try self.dispatchGpu(op, hidden),
                    .npu => try self.dispatchNpu(op, hidden),
                }
            }

            // 精确屏障等待
            const devices_mask = self.computeDevicesMask(plan.ops.items[group_start..group_end]);
            self.barrier.waitAll(devices_mask, sync.DEFAULT_TIMEOUT_MS) catch |e| {
                std.log.err("HeteroExecutor layer sync failed: {any}", .{e});
                return error.SyncFailed;
            };

            group_start = group_end;
        }
    }

    fn computeDevicesMask(_: *Self, ops: []const scheduler.OpSchedule) u8 {
        var mask: u8 = 0;
        for (ops) |op| {
            switch (op.backend) {
                .cpu => mask |= 0b001,
                .gpu => mask |= 0b010,
                .npu => mask |= 0b100,
            }
        }
        return mask;
    }

    fn dispatchCpu(_: *Self, op: scheduler.OpSchedule, hidden: []f32) !void {
        switch (op.op) {
            .rmsNorm => {
                // CPU RMSNorm（已有 SIMD 优化）
                _ = hidden;
                // math.rmsNorm(hidden, hidden, weight, eps);
            },
            .siluMul => {
                // CPU SiLU 元素乘
            },
            .matVecMulF32 => {
                // CPU TopK 路由 / matVecMul
            },
            else => {},
        }
    }

    fn dispatchGpu(_: *Self, op: scheduler.OpSchedule, hidden: []f32) !void {
        _ = op;
        _ = hidden;
        // TODO: 通过 backends.metal / backends.opencl 调用实际 GPU 算子
        // 当前为骨架，待 Metal/OpenCL 后端完全接入 inference.zig 后填充
    }

    fn dispatchNpu(_: *Self, op: scheduler.OpSchedule, hidden: []f32) !void {
        _ = op;
        _ = hidden;
        // TODO: 通过 QNN SDK 或 NNAPI 调用 NPU 算子
        // 当前为骨架，待 NPU 驱动接入后填充
    }
};

/// 动态负载均衡器：基于实时温度和 perf_db 调整设备分配比例
pub const LoadBalancer = struct {
    allocator: std.mem.Allocator,
    perf_db: *op_perf_db.OpPerfDb,
    /// 温度状态引用
    throttle: *@import("thermal.zig").ThrottleState,
    /// 当前负载比例 (npu_ratio, gpu_ratio, cpu_ratio)，总和为 1.0
    ratios: [3]f32,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, perf_db: *op_perf_db.OpPerfDb, throttle: *@import("thermal.zig").ThrottleState) Self {
        return .{
            .allocator = allocator,
            .perf_db = perf_db,
            .throttle = throttle,
            .ratios = .{ 0.4, 0.4, 0.2 }, // 默认 NPU 40%, GPU 40%, CPU 20%
        };
    }

    /// 根据温度和实际执行时间重新平衡负载
    pub fn rebalance(self: *Self) void {
        const level = self.throttle.level;

        // 热节流时：降低 GPU 比例（发热大户），提升 NPU/CPU
        const target: [3]f32 = switch (level) {
            .normal => .{ 0.45, 0.40, 0.15 },
            .light => .{ 0.50, 0.30, 0.20 },
            .medium => .{ 0.55, 0.20, 0.25 },
            .heavy => .{ 0.30, 0.10, 0.60 }, // heavy 时大量回退 CPU
        };

        // 平滑过渡（EMA）
        const alpha: f32 = 0.3;
        for (&self.ratios, target) |*r, t| {
            r.* = r.* * (1.0 - alpha) + t * alpha;
        }

        // 归一化
        const sum = self.ratios[0] + self.ratios[1] + self.ratios[2];
        for (&self.ratios) |*r| r.* /= sum;
    }

    /// 查询当前某类算子应使用的后端
    pub fn selectBackend(self: *const Self, op: op_perf_db.OpType, shape: op_perf_db.Shape) scheduler.Backend {
        // 如果 perf_db 有实测数据，按延迟最小选择
        if (self.perf_db.query(.{ .op = op, .shape = shape, .precision = .q8_0 })) |entry| {
            const cpu_us = @max(1, entry.cpu_us);
            const gpu_us = if (entry.gpu_us > 0) entry.gpu_us else std.math.maxInt(u64);
            const npu_us = if (entry.npu_us > 0) entry.npu_us else std.math.maxInt(u64);

            // 加权实际延迟 * 负载比例倒数（避免过载）
            const cpu_score = @as(f32, @floatFromInt(cpu_us)) / self.ratios[2];
            const gpu_score = @as(f32, @floatFromInt(gpu_us)) / self.ratios[1];
            const npu_score = @as(f32, @floatFromInt(npu_us)) / self.ratios[0];

            if (npu_score <= gpu_score and npu_score <= cpu_score) return .npu;
            if (gpu_score <= cpu_score) return .gpu;
            return .cpu;
        }

        // 无数据时回退到默认策略
        return switch (op) {
            .attention => .npu,
            .ffn_gate_up => .npu,
            .ffn_down => .gpu,
            .rmsNorm, .siluMul, .matVecMulF32, .matVecMulQ8_0 => .cpu,
            else => .cpu,
        };
    }
};

// ========== TDD ==========

test "LoadBalancer rebalance 比例归一化" {
    const allocator = std.testing.allocator;
    var perf_db = op_perf_db.OpPerfDb.init(allocator);
    defer perf_db.deinit();
    var throttle = @import("thermal.zig").ThrottleState.init();

    var lb = LoadBalancer.init(allocator, &perf_db, &throttle);

    // 默认比例和应为 1.0
    const sum0 = lb.ratios[0] + lb.ratios[1] + lb.ratios[2];
    try std.testing.expectApproxEqAbs(sum0, 1.0, 1e-4);

    // 热节流后比例仍应为 1.0
    throttle.throttle(.medium);
    lb.rebalance();
    const sum1 = lb.ratios[0] + lb.ratios[1] + lb.ratios[2];
    try std.testing.expectApproxEqAbs(sum1, 1.0, 1e-4);
}

test "LoadBalancer selectBackend 无 perf_db 回退" {
    const allocator = std.testing.allocator;
    var perf_db = op_perf_db.OpPerfDb.init(allocator);
    defer perf_db.deinit();
    var throttle = @import("thermal.zig").ThrottleState.init();

    const lb = LoadBalancer.init(allocator, &perf_db, &throttle);

    // attention 默认应选 NPU
    const be_attn = lb.selectBackend(.attention, .{ .m = 1, .n = 256, .k = 256 });
    try std.testing.expectEqual(scheduler.Backend.npu, be_attn);

    // ffn_down 默认应选 GPU
    const be_down = lb.selectBackend(.ffn_down, .{ .m = 1, .n = 2048, .k = 1536 });
    try std.testing.expectEqual(scheduler.Backend.gpu, be_down);

    // rmsNorm 默认应选 CPU
    const be_norm = lb.selectBackend(.rmsNorm, .{ .dim = 2048 });
    try std.testing.expectEqual(scheduler.Backend.cpu, be_norm);
}

test "HeteroExecutor init/deinit" {
    const allocator = std.testing.allocator;
    var perf_db = op_perf_db.OpPerfDb.init(allocator);
    defer perf_db.deinit();

    var executor = try HeteroExecutor.init(
        allocator,
        .{ .cpu = true, .gpu = true, .npu = true },
        &perf_db,
        .{},
    );
    defer executor.deinit();
}
