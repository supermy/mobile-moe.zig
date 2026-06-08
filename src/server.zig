// mobile-moe HTTP 服务端
//
// 目标：基于 std.Io (Zig 0.16.0) 实现高性能 OpenAI 兼容 API
// 当前：占位框架，展示 Continuous Batching 接口设计
//
// TODO (中期):
//   - 使用 std.Io.Threaded 或 std.Io.Evented 实现异步 HTTP 服务
//   - 支持 /v1/chat/completions (流式 + 非流式)
//   - 支持 /v1/completions/batch (请求队列 + 动态批处理)
//   - 支持 /health 健康检查
//
// TODO (长期):
//   - 接入 nng 或 libh2o 实现高性能网络层
//   - SSE (Server-Sent Events) 流式输出
//   - Token 级流控与背压

const std = @import("std");
const mobile_moe = @import("mobile_moe");

/// 请求队列：用于 Continuous Batching 的输入缓冲
/// 实际实现中应由独立工作线程消费
const RequestQueue = struct {
    allocator: std.mem.Allocator,
    items: std.ArrayList(CompletionRequest),

    const CompletionRequest = struct {
        prompt: []const u8,
        max_tokens: u32,
        temperature: f32,
    };

    pub fn init(allocator: std.mem.Allocator) RequestQueue {
        return .{
            .allocator = allocator,
            .items = std.ArrayList(CompletionRequest).empty,
        };
    }

    pub fn deinit(self: *RequestQueue) void {
        for (self.items.items) |*req| {
            self.allocator.free(req.prompt);
        }
        self.items.deinit(self.allocator);
    }

    pub fn push(self: *RequestQueue, prompt: []const u8, max_tokens: u32, temperature: f32) !void {
        const prompt_copy = try self.allocator.dupe(u8, prompt);
        errdefer self.allocator.free(prompt_copy);
        try self.items.append(self.allocator, .{
            .prompt = prompt_copy,
            .max_tokens = max_tokens,
            .temperature = temperature,
        });
    }
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    var args_iter = std.process.Args.Iterator.init(init.minimal.args);

    _ = args_iter.next(); // 跳过程序名

    const model_path = args_iter.next() orelse {
        std.debug.print("用法: mobile-moe-server <model.gguf> [--port <N>]\n", .{});
        return;
    };

    var port: u16 = 8080;
    while (args_iter.next()) |arg| {
        if (std.mem.eql(u8, arg, "--port")) {
            if (args_iter.next()) |p| {
                port = std.fmt.parseInt(u16, p, 10) catch 8080;
            }
        }
    }

    std.debug.print("加载模型: {s}\n", .{model_path});

    var engine = try mobile_moe.Engine.load(allocator, model_path, 4096, false);
    defer engine.deinit();

    var queue = RequestQueue.init(allocator);
    defer queue.deinit();

    std.debug.print("HTTP 服务端待实现 (port={d}, queue_size={d})\n", .{ port, queue.items.items.len });
}
