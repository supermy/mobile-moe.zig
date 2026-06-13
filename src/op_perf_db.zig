// 算子性能数据库（Operator Performance Database）
//
// 目标：建立关键算子的 CPU 基准延迟库，记录 (op_type, shape, precision) → cpu_us
// 为后续 GPU/NPU 对比和异构调度决策提供数据依据。
//
// 设计原则：
//   1. 键空间紧凑：用 64-bit hash 作为查询键，避免复杂结构体比较。
//   2. 线程安全：当前仅单线程写入（启动时基准测试），查询阶段只读。
//   3. 可序列化：支持保存/加载性能数据库，避免每次启动重新测试。

const std = @import("std");
const c = std.c;
const dequant = @import("dequant.zig");
const qmm = @import("qmm.zig");
const math = @import("math.zig");

/// 获取单调时钟时间（微秒），替代 Zig 0.16 移除的 std.time.microTimestamp
fn getMonotonicUs() u64 {
    var ts: c.timespec = undefined;
    const rc = c.clock_gettime(c.clockid_t.MONOTONIC, &ts);
    if (rc != 0) unreachable;
    return @as(u64, @intCast(ts.sec)) * 1_000_000 + @as(u64, @intCast(ts.nsec)) / 1000;
}

/// 算子类型枚举（覆盖当前 CPU 路径的所有关键算子）
pub const OpType = enum(u8) {
    matMulQ8_0, // Q8_0 批量矩阵乘法
    matMulQ4_K, // Q4_K 批量矩阵乘法（预留）
    matMulF32, // f32 批量矩阵乘法（预留）
    matVecMulQ8_0, // Q8_0 矩阵-向量乘法
    matVecMulF32, // f32 矩阵-向量乘法
    rmsNorm, // RMSNorm 归一化
    siluMul, // SiLU(gate) * up（SwiGLU 融合算子）
    rope, // RoPE 位置编码应用
    softmax, // Softmax
    dot, // 向量点积
    attention, // 完整 MLA Attention（高层聚合）
    ffn_gate_up, // FFN gate + up 投影（聚合，用于异构决策）
    ffn_down, // FFN down 投影（聚合，用于异构决策）
};

/// 精度类型
pub const Precision = enum(u8) {
    q8_0,
    q4_k,
    f32,
    f16,
};

/// 张量形状描述（用于性能数据库键）
pub const Shape = struct {
    m: u32 = 0,
    n: u32 = 0,
    k: u32 = 0,
    dim: u32 = 0, // 用于 rmsNorm / siluMul 等一维算子

    /// 生成紧凑的 64-bit 哈希键（非密码学安全，仅用于 HashMap）
    pub fn hash(self: Shape) u64 {
        var h: u64 = self.m;
        h = h *% 31 +% self.n;
        h = h *% 31 +% self.k;
        h = h *% 31 +% self.dim;
        return h;
    }
};

/// 性能数据库键（算子 + 形状 + 精度）
pub const OpKey = struct {
    op: OpType,
    shape: Shape,
    precision: Precision,

    pub fn hash(self: OpKey) u64 {
        var h: u64 = @intFromEnum(self.op);
        h = h *% 17 +% self.shape.hash();
        h = h *% 17 +% @intFromEnum(self.precision);
        return h;
    }

    pub fn eql(a: OpKey, b: OpKey) bool {
        return a.op == b.op and a.precision == b.precision and
            a.shape.m == b.shape.m and a.shape.n == b.shape.n and
            a.shape.k == b.shape.k and a.shape.dim == b.shape.dim;
    }
};

/// 性能条目：单条记录
pub const PerfEntry = struct {
    /// CPU 参考延迟（微秒）
    cpu_us: u64,
    /// GPU 延迟（微秒），0 表示未测量
    gpu_us: u64 = 0,
    /// NPU 延迟（微秒），0 表示未测量
    npu_us: u64 = 0,
    /// 测量时的运行次数（用于置信度评估）
    iterations: u32,
};

/// 算子性能数据库
pub const OpPerfDb = struct {
    allocator: std.mem.Allocator,
    map: std.HashMap(OpKey, PerfEntry, OpKeyContext, std.hash_map.default_max_load_percentage),

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator) Self {
        return .{
            .allocator = allocator,
            .map = std.HashMap(OpKey, PerfEntry, OpKeyContext, std.hash_map.default_max_load_percentage).init(allocator),
        };
    }

    pub fn deinit(self: *Self) void {
        self.map.deinit();
    }

    /// 记录或更新性能数据
    pub fn record(self: *Self, key: OpKey, entry: PerfEntry) !void {
        const result = try self.map.getOrPut(key);
        if (result.found_existing) {
            // 简单平均更新（可扩展为指数移动平均）
            const old = result.value_ptr.*;
            const total_iter = old.iterations + entry.iterations;
            result.value_ptr.cpu_us = (old.cpu_us * old.iterations + entry.cpu_us * entry.iterations) / total_iter;
            if (entry.gpu_us > 0) {
                result.value_ptr.gpu_us = if (old.gpu_us > 0)
                    (old.gpu_us * old.iterations + entry.gpu_us * entry.iterations) / total_iter
                else
                    entry.gpu_us;
            }
            if (entry.npu_us > 0) {
                result.value_ptr.npu_us = if (old.npu_us > 0)
                    (old.npu_us * old.iterations + entry.npu_us * entry.iterations) / total_iter
                else
                    entry.npu_us;
            }
            result.value_ptr.iterations = total_iter;
        } else {
            result.key_ptr.* = key;
            result.value_ptr.* = entry;
        }
    }

    /// 查询性能数据，返回 null 表示未记录
    pub fn query(self: *const Self, key: OpKey) ?PerfEntry {
        return self.map.get(key);
    }

    /// 查询 CPU 延迟，未记录返回 0
    pub fn queryCpuUs(self: *const Self, key: OpKey) u64 {
        const entry = self.query(key) orelse return 0;
        return entry.cpu_us;
    }

    /// 判断是否有某算子的记录
    pub fn has(self: *const Self, key: OpKey) bool {
        return self.map.contains(key);
    }

    /// 条目数
    pub fn count(self: *const Self) usize {
        return self.map.count();
    }

    /// 清空所有记录
    pub fn clear(self: *Self) void {
        self.map.clearRetainingCapacity();
    }

    /// 遍历所有条目
    pub const EntryIter = std.HashMap(OpKey, PerfEntry, OpKeyContext, std.hash_map.default_max_load_percentage).Iterator;
    pub fn iterator(self: *Self) EntryIter {
        return self.map.iterator();
    }
};

/// HashMap 上下文
pub const OpKeyContext = struct {
    pub fn hash(_: @This(), key: OpKey) u64 {
        return key.hash();
    }
    pub fn eql(_: @This(), a: OpKey, b: OpKey) bool {
        return OpKey.eql(a, b);
    }
};

/// 运行基准测试，返回中位数延迟（微秒）
/// 使用 clock_gettime 高精度计时（Zig 0.16 兼容）
pub fn benchmarkCpu(comptime op_fn: anytype, args: anytype, iterations: u32) u64 {
    // 预热：先执行一次，避免冷缓存影响
    @call(.auto, op_fn, args);

    const start_us = getMonotonicUs();

    const n = @max(1, iterations);
    for (0..n) |_| {
        @call(.auto, op_fn, args);
    }

    const elapsed_us = getMonotonicUs() - start_us;
    return elapsed_us / n; // 转换为单次微秒
}

/// 对 matMulQ8_0 进行基准测试并记录到数据库
pub fn benchRecordMatMulQ8_0(db: *OpPerfDb, m: u32, n: u32, k: u32, iterations: u32) !void {
    const blocks_per_row = (k + 31) / 32;
    const n_blocks = m * blocks_per_row;

    var blocks = try db.allocator.alloc(dequant.BlockQ8_0, n_blocks);
    defer db.allocator.free(blocks);
    for (0..n_blocks) |i| {
        blocks[i].d = dequant.f32ToF16(1.0 / 127.0);
        for (0..32) |j| {
            blocks[i].qs[j] = 1;
        }
    }

    const b = try db.allocator.alloc(f32, n * k);
    defer db.allocator.free(b);
    @memset(b, 0.5);

    const out = try db.allocator.alloc(f32, m * n);
    defer db.allocator.free(out);

    const us = benchmarkCpu(qmm.matMulQ8_0, .{ out, blocks, b, m, n, k }, iterations);

    try db.record(.{
        .op = .matMulQ8_0,
        .shape = .{ .m = m, .n = n, .k = k },
        .precision = .q8_0,
    }, .{ .cpu_us = us, .iterations = iterations });
}

/// 对 rmsNorm 进行基准测试并记录到数据库
pub fn benchRecordRmsNorm(db: *OpPerfDb, dim: u32, iterations: u32) !void {
    const x = try db.allocator.alloc(f32, dim);
    defer db.allocator.free(x);
    const w = try db.allocator.alloc(f32, dim);
    defer db.allocator.free(w);
    const out = try db.allocator.alloc(f32, dim);
    defer db.allocator.free(out);

    @memset(x, 0.5);
    @memset(w, 1.0);

    const us = benchmarkCpu(math.rmsNorm, .{ out, x, w, 1e-5 }, iterations);

    try db.record(.{
        .op = .rmsNorm,
        .shape = .{ .dim = dim },
        .precision = .f32,
    }, .{ .cpu_us = us, .iterations = iterations });
}

/// 对 matVecMulQ8_0 进行基准测试并记录到数据库
pub fn benchRecordMatVecMulQ8_0(db: *OpPerfDb, out_dim: u32, in_dim: u32, iterations: u32) !void {
    const blocks_per_row = (in_dim + 31) / 32;
    const n_blocks = out_dim * blocks_per_row;

    var blocks = try db.allocator.alloc(dequant.BlockQ8_0, n_blocks);
    defer db.allocator.free(blocks);
    for (0..n_blocks) |i| {
        blocks[i].d = dequant.f32ToF16(1.0 / 127.0);
        for (0..32) |j| {
            blocks[i].qs[j] = 1;
        }
    }

    const x = try db.allocator.alloc(f32, in_dim);
    defer db.allocator.free(x);
    @memset(x, 0.5);

    const out = try db.allocator.alloc(f32, out_dim);
    defer db.allocator.free(out);

    const us = benchmarkCpu(qmm.matVecMulQ8_0, .{ out, blocks, x, out_dim, in_dim }, iterations);

    try db.record(.{
        .op = .matVecMulQ8_0,
        .shape = .{ .m = out_dim, .k = in_dim },
        .precision = .q8_0,
    }, .{ .cpu_us = us, .iterations = iterations });
}

/// 对 siluMul 进行基准测试并记录到数据库
pub fn benchRecordSiluMul(db: *OpPerfDb, dim: u32, iterations: u32) !void {
    const gate = try db.allocator.alloc(f32, dim);
    defer db.allocator.free(gate);
    const up = try db.allocator.alloc(f32, dim);
    defer db.allocator.free(up);

    @memset(gate, 0.5);
    @memset(up, 0.5);

    const us = benchmarkCpu(math.siluInplace, .{gate}, iterations) +
        benchmarkCpu(math.mul, .{ up, gate, up }, iterations);

    try db.record(.{
        .op = .siluMul,
        .shape = .{ .dim = dim },
        .precision = .f32,
    }, .{ .cpu_us = us, .iterations = iterations });
}

/// 运行标准基准测试套件，覆盖模型典型形状
/// 返回成功记录的条目数
pub fn runStandardBenchmarks(db: *OpPerfDb) !usize {
    const start_count = db.count();

    // GLM-4.7-Flash 典型形状
    const hidden_size: u32 = 2048;
    const intermediate_size: u32 = 1536;
    const num_heads: u32 = 16;
    const head_dim: u32 = 128;

    // 矩阵-向量乘法（token generation 阶段）
    try benchRecordMatVecMulQ8_0(db, intermediate_size, hidden_size, 100);
    try benchRecordMatVecMulQ8_0(db, hidden_size, intermediate_size, 100);
    try benchRecordMatVecMulQ8_0(db, hidden_size, hidden_size, 100);

    // 批量矩阵乘法（prefill 阶段，batch=64）
    try benchRecordMatMulQ8_0(db, 64, intermediate_size, hidden_size, 20);
    try benchRecordMatMulQ8_0(db, 64, hidden_size, intermediate_size, 20);

    // RMSNorm
    try benchRecordRmsNorm(db, hidden_size, 1000);
    try benchRecordRmsNorm(db, num_heads * head_dim, 1000);

    // SwiGLU
    try benchRecordSiluMul(db, intermediate_size, 1000);

    // 小形状快速路径（热节流时使用）
    try benchRecordMatVecMulQ8_0(db, intermediate_size / 2, hidden_size, 50);
    try benchRecordRmsNorm(db, hidden_size / 2, 500);

    return db.count() - start_count;
}

// ========== 测试 ==========

test "OpPerfDb init/deinit" {
    const testing = std.testing;
    var db = OpPerfDb.init(testing.allocator);
    defer db.deinit();
    try testing.expectEqual(@as(usize, 0), db.count());
}

test "record and query" {
    const testing = std.testing;
    var db = OpPerfDb.init(testing.allocator);
    defer db.deinit();

    const key = OpKey{
        .op = .rmsNorm,
        .shape = .{ .dim = 2048 },
        .precision = .f32,
    };

    try db.record(key, .{ .cpu_us = 150, .iterations = 10 });
    try testing.expect(db.has(key));

    const entry = db.query(key).?;
    try testing.expectEqual(@as(u64, 150), entry.cpu_us);
    try testing.expectEqual(@as(u64, 0), entry.gpu_us);
}

test "update entry averages" {
    const testing = std.testing;
    var db = OpPerfDb.init(testing.allocator);
    defer db.deinit();

    const key = OpKey{
        .op = .matVecMulQ8_0,
        .shape = .{ .m = 1536, .k = 2048 },
        .precision = .q8_0,
    };

    try db.record(key, .{ .cpu_us = 100, .iterations = 10 });
    try db.record(key, .{ .cpu_us = 200, .iterations = 10 });

    const entry = db.query(key).?;
    try testing.expectEqual(@as(u64, 150), entry.cpu_us); // (100*10 + 200*10) / 20 = 150
    try testing.expectEqual(@as(u32, 20), entry.iterations);
}

test "benchmark matMulQ8_0" {
    const testing = std.testing;
    var db = OpPerfDb.init(testing.allocator);
    defer db.deinit();

    try benchRecordMatMulQ8_0(&db, 4, 8, 64, 5);

    const key = OpKey{
        .op = .matMulQ8_0,
        .shape = .{ .m = 4, .n = 8, .k = 64 },
        .precision = .q8_0,
    };
    try testing.expect(db.has(key));
    const entry = db.query(key).?;
    try testing.expect(entry.cpu_us > 0);
}

test "benchmark rmsNorm" {
    const testing = std.testing;
    var db = OpPerfDb.init(testing.allocator);
    defer db.deinit();

    try benchRecordRmsNorm(&db, 2048, 100);

    const key = OpKey{
        .op = .rmsNorm,
        .shape = .{ .dim = 2048 },
        .precision = .f32,
    };
    try testing.expect(db.has(key));
    const entry = db.query(key).?;
    try testing.expect(entry.cpu_us > 0);
}

test "runStandardBenchmarks" {
    const testing = std.testing;
    var db = OpPerfDb.init(testing.allocator);
    defer db.deinit();

    const recorded = try runStandardBenchmarks(&db);
    try testing.expect(recorded > 0);
    try testing.expect(db.count() >= recorded);
}

test "query missing returns null" {
    const testing = std.testing;
    var db = OpPerfDb.init(testing.allocator);
    defer db.deinit();

    const key = OpKey{
        .op = .attention,
        .shape = .{ .m = 16, .n = 128, .k = 128 },
        .precision = .f32,
    };
    try testing.expect(!db.has(key));
    try testing.expectEqual(@as(?PerfEntry, null), db.query(key));
    try testing.expectEqual(@as(u64, 0), db.queryCpuUs(key));
}
