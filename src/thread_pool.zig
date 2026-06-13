// 自定义线程池：替代每次 spawn/join，减少线程创建开销
//
// 设计：固定 N 个工作线程，使用 pthread Mutex + Condition 等待任务队列。
// 任务提交到 FIFO 队列，工作线程竞争取出执行。
// Zig 0.16 兼容：std.Thread.Mutex/Condition 已移除，改用 pthread 原语。

const std = @import("std");
const c = std.c;

pub const Task = struct {
    run_fn: *const fn (ctx: *anyopaque) void,
    ctx: *anyopaque,
};

pub const ThreadPool = struct {
    allocator: std.mem.Allocator,
    threads: []std.Thread,
    queue: std.ArrayList(Task),
    mutex: c.pthread_mutex_t,
    cond: c.pthread_cond_t,
    shutdown: bool,
    running: usize,

    const Self = @This();

    /// 初始化线程池（调用者提供存储，避免栈地址逃逸问题）
    pub fn init(self: *Self, allocator: std.mem.Allocator, n_threads: usize) !void {
        self.allocator = allocator;
        self.threads = try allocator.alloc(std.Thread, n_threads);
        errdefer allocator.free(self.threads);

        self.queue = .empty;
        self.mutex = c.PTHREAD_MUTEX_INITIALIZER;
        self.cond = c.PTHREAD_COND_INITIALIZER;
        self.shutdown = false;
        self.running = 0;

        for (0..n_threads) |i| {
            self.threads[i] = try std.Thread.spawn(.{}, workerLoop, .{self});
        }
    }

    pub fn deinit(self: *Self) void {
        _ = c.pthread_mutex_lock(&self.mutex);
        self.shutdown = true;
        _ = c.pthread_cond_broadcast(&self.cond);
        _ = c.pthread_mutex_unlock(&self.mutex);

        for (self.threads) |t| {
            t.join();
        }

        self.allocator.free(self.threads);
        self.queue.deinit(self.allocator);
        _ = c.pthread_mutex_destroy(&self.mutex);
        _ = c.pthread_cond_destroy(&self.cond);
    }

    /// 提交任务（非阻塞，复制到队列）
    pub fn submit(self: *Self, task: Task) !void {
        _ = c.pthread_mutex_lock(&self.mutex);
        defer _ = c.pthread_mutex_unlock(&self.mutex);
        if (self.shutdown) return error.PoolShutdown;
        try self.queue.append(self.allocator, task);
        _ = c.pthread_cond_signal(&self.cond);
    }

    /// 等待当前队列中所有任务完成（包括正在执行的）
    pub fn waitAll(self: *Self) void {
        _ = c.pthread_mutex_lock(&self.mutex);
        while (self.queue.items.len > 0 or self.running > 0) {
            _ = c.pthread_cond_wait(&self.cond, &self.mutex);
        }
        _ = c.pthread_mutex_unlock(&self.mutex);
    }

    fn workerLoop(pool: *Self) void {
        while (true) {
            _ = c.pthread_mutex_lock(&pool.mutex);
            while (pool.queue.items.len == 0 and !pool.shutdown) {
                _ = c.pthread_cond_wait(&pool.cond, &pool.mutex);
            }
            if (pool.shutdown and pool.queue.items.len == 0) {
                _ = c.pthread_mutex_unlock(&pool.mutex);
                break;
            }
            const task = pool.queue.orderedRemove(0);
            pool.running += 1;
            _ = c.pthread_mutex_unlock(&pool.mutex);

            task.run_fn(task.ctx);

            _ = c.pthread_mutex_lock(&pool.mutex);
            pool.running -= 1;
            _ = c.pthread_cond_signal(&pool.cond);
            _ = c.pthread_mutex_unlock(&pool.mutex);
        }
    }
};

test "thread pool basic submit and wait" {
    const testing = std.testing;
    var pool: ThreadPool = undefined;
    try ThreadPool.init(&pool, testing.allocator, 2);
    defer pool.deinit();

    var counter: usize = 0;
    const Ctx = struct {
        counter: *usize,
        fn run(ctx: *anyopaque) void {
            const self = @as(*@This(), @ptrCast(@alignCast(ctx)));
            self.counter.* += 1;
        }
    };

    var ctx = Ctx{ .counter = &counter };
    for (0..10) |_| {
        try pool.submit(.{ .run_fn = Ctx.run, .ctx = &ctx });
    }

    // 给线程足够时间处理（Zig 0.16 兼容：使用 c.nanosleep）
    const req = c.timespec{ .sec = 0, .nsec = 100 * 1_000_000 };
    _ = c.nanosleep(&req, null);
    pool.waitAll();

    try testing.expectEqual(@as(usize, 10), counter);
}

test "thread pool shutdown safely" {
    const testing = std.testing;
    var pool: ThreadPool = undefined;
    try ThreadPool.init(&pool, testing.allocator, 4);
    defer pool.deinit();

    // 空池直接 shutdown 不应 panic
    try testing.expect(true);
}
