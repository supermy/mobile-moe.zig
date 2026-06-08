// mobile-moe HTTP 服务端
//
// 基于 Zig 0.16.0 std.Io 实现同步 HTTP 服务
// 支持 /v1/chat/completions 和 /health

const std = @import("std");
const mobile_moe = @import("mobile_moe");

fn handleRequest(
    allocator: std.mem.Allocator,
    engine: *mobile_moe.Engine,
    request: []const u8,
    writer: *std.Io.Writer,
) !void {
    const body_start = std.mem.indexOf(u8, request, "\r\n\r\n") orelse return error.BadRequest;
    const body = request[body_start + 4 ..];

    // 提取路径（极简解析）
    const first_space = std.mem.indexOfScalar(u8, request, ' ') orelse return error.BadRequest;
    const second_space = std.mem.indexOfScalar(u8, request[first_space + 1 ..], ' ') orelse return error.BadRequest;
    const path = request[first_space + 1 ..][0..second_space];

    if (std.mem.eql(u8, path, "/v1/chat/completions")) {
        const prompt = extractJsonString(body, "content") orelse "hello";
        const max_tokens = extractJsonInt(body, "max_tokens") orelse 32;

        const generated = try engine.generateText(prompt, max_tokens);
        defer allocator.free(generated);

        var response_buf: [4096]u8 = undefined;
        const response = try std.fmt.bufPrint(&response_buf, "{{\"choices\":[{{\"message\":{{\"content\":\"{s}\"}}}}]}}\n", .{generated});

        var header_buf: [256]u8 = undefined;
        const header = try std.fmt.bufPrint(&header_buf, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nConnection: close\r\n\r\n", .{response.len});

        try writer.writeAll(header);
        try writer.writeAll(response);
    } else if (std.mem.eql(u8, path, "/health")) {
        try writer.writeAll("HTTP/1.1 200 OK\r\nContent-Length: 15\r\nConnection: close\r\n\r\n{\"status\":\"ok\"}");
    } else {
        try writer.writeAll("HTTP/1.1 404 Not Found\r\nContent-Length: 9\r\nConnection: close\r\n\r\nNot Found");
    }
}

fn extractJsonString(body: []const u8, key: []const u8) ?[]const u8 {
    const pattern = std.fmt.allocPrint(std.heap.page_allocator, "\"{s}\":\"", .{key}) catch return null;
    defer std.heap.page_allocator.free(pattern);
    const start = std.mem.indexOf(u8, body, pattern) orelse return null;
    const val_start = start + pattern.len;
    const val_end = std.mem.indexOfScalar(u8, body[val_start..], '"') orelse return null;
    return body[val_start..][0..val_end];
}

fn extractJsonInt(body: []const u8, key: []const u8) ?u32 {
    const pattern = std.fmt.allocPrint(std.heap.page_allocator, "\"{s}\":", .{key}) catch return null;
    defer std.heap.page_allocator.free(pattern);
    const start = std.mem.indexOf(u8, body, pattern) orelse return null;
    const val_start = start + pattern.len;
    var end = val_start;
    while (end < body.len and std.ascii.isDigit(body[end])) : (end += 1) {}
    return std.fmt.parseInt(u32, body[val_start..end], 10) catch null;
}

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

    var threaded = std.Io.Threaded.init(allocator, .{});
    defer threaded.deinit();
    const io = threaded.io();

    const address = try std.Io.net.IpAddress.parse("0.0.0.0", port);
    var server = try address.listen(io, .{ .reuse_address = true });
    defer server.deinit(io);

    std.debug.print("Server listening on 0.0.0.0:{d}\n", .{port});

    while (true) {
        var stream = server.accept(io) catch |err| {
            std.debug.print("Accept error: {s}\n", .{@errorName(err)});
            continue;
        };

        var read_buf: [4096]u8 = undefined;
        var write_buf: [4096]u8 = undefined;
        var reader = stream.reader(io, &read_buf);
        var writer = stream.writer(io, &write_buf);

        var req_buf: [4096]u8 = undefined;
        const n = reader.interface.readSliceShort(&req_buf) catch |err| {
            std.debug.print("Read error: {s}\n", .{@errorName(err)});
            stream.close(io);
            continue;
        };
        const request = req_buf[0..n];

        handleRequest(allocator, &engine, request, &writer.interface) catch |err| {
            std.debug.print("Handle error: {s}\n", .{@errorName(err)});
        };

        stream.close(io);
    }
}
