// 快速同步机制（FastSync）
//
// GPU-NPU 并行的关键瓶颈是同步开销（通常 400μs+）。
// 采用 HeteroLLM 风格的预测性同步策略：
//   1. 统一内存池：通过 ION / DMA-BUF 分配 CPU/GPU/NPU 共享物理内存（OS 层实现，Zig 层预留接口）
//   2. 历史执行时间预测：维护算子-设备性能数据库，预估 GPU/NPU 执行时间
//   3. 预测性睡眠：睡眠至预计完成前 PREDICTIVE_MARGIN_US，减少空轮询时间
//   4. 精确轮询：最后 PREDICTIVE_MARGIN_US 空轮询标志位，同步精度达 ~20μs
//
// 当前实现为 CPU 侧同步骨架， Metal / OpenCL / NPU 后端通过标志位通知完成。

const std = @import("std");
const c = std.c;
const op_perf_db = @import("op_perf_db.zig");

/// 精确轮询前的安全边际（微秒）
pub const PREDICTIVE_MARGIN_US: u64 = 100;

/// 获取单调时钟时间（微秒），替代 Zig 0.16 移除的 std.time.microTimestamp
fn getMonotonicUs() u64 {
    var ts: c.timespec = undefined;
    const rc = c.clock_gettime(c.clockid_t.MONOTONIC, &ts);
    if (rc != 0) unreachable;
    return @as(u64, @intCast(ts.sec)) * 1_000_000 + @as(u64, @intCast(ts.nsec)) / 1000;
}

/// 睡眠指定微秒，替代 Zig 0.16 移除的 std.time.sleep
fn sleepUs(us: u64) void {
    const ns = us * 1000;
    const req = c.timespec{
        .sec = @intCast(ns / 1_000_000_000),
        .nsec = @intCast(ns % 1_000_000_000),
    };
    _ = c.nanosleep(&req, null);
}

/// 默认同步超时（毫秒）
pub const DEFAULT_TIMEOUT_MS: u64 = 5000;

/// 设备类型
pub const Device = enum(u8) {
    cpu,
    gpu,
    npu,
};

/// 原子同步标志（基于 std.atomic，可映射到共享内存）
pub const SyncFlag = struct {
    state: std.atomic.Value(u32),

    pub const PENDING: u32 = 0;
    pub const DONE: u32 = 1;
    pub const ERROR: u32 = 2;

    pub fn init() SyncFlag {
        return .{ .state = std.atomic.Value(u32).init(PENDING) };
    }

    pub fn setDone(self: *SyncFlag) void {
        self.state.store(DONE, .release);
    }

    pub fn setError(self: *SyncFlag) void {
        self.state.store(ERROR, .release);
    }

    pub fn isDone(self: *const SyncFlag) bool {
        return self.state.load(.acquire) == DONE;
    }

    pub fn isError(self: *const SyncFlag) bool {
        return self.state.load(.acquire) == ERROR;
    }

    pub fn reset(self: *SyncFlag) void {
        self.state.store(PENDING, .monotonic);
    }
};

/// 多设备同步屏障（一层内的 CPU/GPU/NPU 同步点）
pub const SyncBarrier = struct {
    cpu_flag: SyncFlag,
    gpu_flag: SyncFlag,
    npu_flag: SyncFlag,

    pub fn init() SyncBarrier {
        return .{
            .cpu_flag = SyncFlag.init(),
            .gpu_flag = SyncFlag.init(),
            .npu_flag = SyncFlag.init(),
        };
    }

    pub fn reset(self: *SyncBarrier) void {
        self.cpu_flag.reset();
        self.gpu_flag.reset();
        self.npu_flag.reset();
    }

    pub fn signal(self: *SyncBarrier, device: Device) void {
        switch (device) {
            .cpu => self.cpu_flag.setDone(),
            .gpu => self.gpu_flag.setDone(),
            .npu => self.npu_flag.setDone(),
        }
    }

    /// 等待所有指定设备完成
    /// devices: 需要等待的设备位掩码（bit 0=cpu, 1=gpu, 2=npu）
    /// timeout_ms: 超时时间，0 表示无限等待
    pub fn waitAll(self: *const SyncBarrier, devices_mask: u8, timeout_ms: u64) error{ Timeout, DeviceError }!void {
        const wait_us = timeout_ms * 1000;
        const start = getMonotonicUs();

        while (true) {
            if (devices_mask & 0b001 != 0 and self.cpu_flag.isError()) return error.DeviceError;
            if (devices_mask & 0b010 != 0 and self.gpu_flag.isError()) return error.DeviceError;
            if (devices_mask & 0b100 != 0 and self.npu_flag.isError()) return error.DeviceError;

            const all_done =
                (devices_mask & 0b001 == 0 or self.cpu_flag.isDone()) and
                (devices_mask & 0b010 == 0 or self.gpu_flag.isDone()) and
                (devices_mask & 0b100 == 0 or self.npu_flag.isDone());

            if (all_done) return;

            if (timeout_ms > 0) {
                const elapsed = @as(u64, @intCast(getMonotonicUs() - start));
                if (elapsed >= wait_us) return error.Timeout;
            }

            // 自旋 + 轻量 pause，降低功耗同时保持低延迟
            std.atomic.spinLoopHint();
        }
    }

    /// 检查是否全部完成（非阻塞）
    pub fn pollAll(self: *const SyncBarrier, devices_mask: u8) bool {
        return (devices_mask & 0b001 == 0 or self.cpu_flag.isDone()) and
            (devices_mask & 0b010 == 0 or self.gpu_flag.isDone()) and
            (devices_mask & 0b100 == 0 or self.npu_flag.isDone());
    }
};

/// 预测性同步器
/// 基于性能数据库预估执行时间，先睡眠再精确轮询
pub const PredictiveSync = struct {
    /// 性能数据库引用（可选，无数据库时退化到纯轮询）
    perf_db: ?*const op_perf_db.OpPerfDb,

    pub fn init(perf_db: ?*const op_perf_db.OpPerfDb) PredictiveSync {
        return .{ .perf_db = perf_db };
    }

    /// 执行预测性同步：
    /// 1. 查询性能数据库获取预估 GPU/NPU 执行时间
    /// 2. 睡眠至预计完成前 PREDICTIVE_MARGIN_US
    /// 3. 精确轮询 barrier 直到完成或超时
    pub fn sync(
        self: *const PredictiveSync,
        barrier: *const SyncBarrier,
        key_gpu: ?op_perf_db.OpKey,
        key_npu: ?op_perf_db.OpKey,
        devices_mask: u8,
        timeout_ms: u64,
    ) error{ Timeout, DeviceError }!void {
        var gpu_est_us: u64 = 0;
        var npu_est_us: u64 = 0;

        if (self.perf_db) |db| {
            if (key_gpu) |k| gpu_est_us = db.queryCpuUs(k); // 先用 CPU 数据近似 GPU
            if (key_npu) |k| npu_est_us = db.queryCpuUs(k); // 先用 CPU 数据近似 NPU
        }

        const max_est = @max(gpu_est_us, npu_est_us);

        // 预测性睡眠：若预估时间足够长，先睡一会儿
        if (max_est > PREDICTIVE_MARGIN_US * 2) {
            const sleep_us = max_est - PREDICTIVE_MARGIN_US;
            sleepUs(sleep_us);
        }

        // 精确轮询阶段
        return barrier.waitAll(devices_mask, timeout_ms);
    }

    /// 轻量同步：不查询数据库，直接轮询（用于已知延迟极短的场景）
    pub fn syncSpin(
        _: *const PredictiveSync,
        barrier: *const SyncBarrier,
        devices_mask: u8,
        timeout_ms: u64,
    ) error{ Timeout, DeviceError }!void {
        return barrier.waitAll(devices_mask, timeout_ms);
    }
};

/// 层同步管理器（每层的同步状态）
pub const LayerSync = struct {
    barriers: []SyncBarrier,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, n_layers: u32) !LayerSync {
        const barriers = try allocator.alloc(SyncBarrier, n_layers);
        for (barriers) |*b| b.* = SyncBarrier.init();
        return .{
            .barriers = barriers,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *LayerSync) void {
        self.allocator.free(self.barriers);
    }

    pub fn resetLayer(self: *LayerSync, layer: u32) void {
        if (layer < self.barriers.len) {
            self.barriers[layer].reset();
        }
    }

    pub fn signal(self: *LayerSync, layer: u32, device: Device) void {
        if (layer < self.barriers.len) {
            self.barriers[layer].signal(device);
        }
    }

    pub fn waitLayer(
        self: *const LayerSync,
        layer: u32,
        devices_mask: u8,
        timeout_ms: u64,
    ) error{ Timeout, DeviceError }!void {
        if (layer >= self.barriers.len) return;
        return self.barriers[layer].waitAll(devices_mask, timeout_ms);
    }
};

// ========== 测试 ==========

test "SyncFlag set/clear" {
    const testing = std.testing;
    var flag = SyncFlag.init();
    try testing.expect(!flag.isDone());
    try testing.expect(!flag.isError());

    flag.setDone();
    try testing.expect(flag.isDone());
    try testing.expect(!flag.isError());

    flag.reset();
    try testing.expect(!flag.isDone());
}

test "SyncFlag error state" {
    const testing = std.testing;
    var flag = SyncFlag.init();
    flag.setError();
    try testing.expect(flag.isError());
    try testing.expect(!flag.isDone());
}

test "SyncBarrier wait all done" {
    var barrier = SyncBarrier.init();

    // CPU + GPU 完成
    barrier.signal(.cpu);
    barrier.signal(.gpu);

    try barrier.waitAll(0b011, 1000);
}

test "SyncBarrier wait timeout" {
    const testing = std.testing;
    var barrier = SyncBarrier.init();

    // 只 signal CPU，GPU 未完成 → 超时
    barrier.signal(.cpu);
    const result = barrier.waitAll(0b011, 10); // 10ms 超时
    try testing.expectError(error.Timeout, result);
}

test "SyncBarrier device error" {
    const testing = std.testing;
    var barrier = SyncBarrier.init();

    barrier.cpu_flag.setDone();
    barrier.gpu_flag.setError();

    const result = barrier.waitAll(0b011, 1000);
    try testing.expectError(error.DeviceError, result);
}

test "SyncBarrier poll non-blocking" {
    const testing = std.testing;
    var barrier = SyncBarrier.init();

    try testing.expect(!barrier.pollAll(0b011));
    barrier.signal(.cpu);
    try testing.expect(!barrier.pollAll(0b011));
    barrier.signal(.gpu);
    try testing.expect(barrier.pollAll(0b011));
}

test "LayerSync init/deinit" {
    const testing = std.testing;
    var ls = try LayerSync.init(testing.allocator, 4);
    defer ls.deinit();
    try testing.expectEqual(@as(usize, 4), ls.barriers.len);
}

test "LayerSync signal and wait" {
    const testing = std.testing;
    var ls = try LayerSync.init(testing.allocator, 2);
    defer ls.deinit();

    ls.signal(0, .cpu);
    ls.signal(0, .gpu);
    try ls.waitLayer(0, 0b011, 100);
}

test "PredictiveSync with empty db" {
    var syncer = PredictiveSync.init(null);
    var barrier = SyncBarrier.init();

    barrier.signal(.cpu);
    try syncer.syncSpin(&barrier, 0b001, 100);
}

test "PredictiveSync with perf db" {
    const testing = std.testing;
    var db = op_perf_db.OpPerfDb.init(testing.allocator);
    defer db.deinit();

    const key = op_perf_db.OpKey{
        .op = .matVecMulQ8_0,
        .shape = .{ .m = 1536, .k = 2048 },
        .precision = .q8_0,
    };
    try db.record(key, .{ .cpu_us = 50, .iterations = 10 });

    var syncer = PredictiveSync.init(&db);
    var barrier = SyncBarrier.init();

    barrier.signal(.cpu);
    // GPU 使用 db 中的 key，但当前测试中只有 CPU 完成
    try syncer.sync(&barrier, key, null, 0b001, 100);
}
