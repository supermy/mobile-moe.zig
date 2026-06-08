// mobile-moe HTTP 服务端（占位）
//
// 目标：nng 实现高性能 OpenAI 兼容 API
// 当前：基础 HTTP 服务框架

const std = @import("std");
const mobile_moe = @import("mobile_moe");

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    var args_iter = std.process.Args.Iterator.init(init.minimal.args);

    _ = args_iter.next(); // 跳过程序名

    const model_path = args_iter.next() orelse {
        std.debug.print("用法: mobile-moe-server <model.gguf> [--port <N>]\n", .{});
        return;
    };

    std.debug.print("加载模型: {s}\n", .{model_path});

    var engine = try mobile_moe.Engine.load(allocator, model_path, 4096, false);
    defer engine.deinit();

    std.debug.print("HTTP 服务端待实现 (nng)\n", .{});
}
