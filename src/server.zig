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
        const response = buildJsonResponse(&response_buf, generated) catch |err| switch (err) {
            error.NoSpaceLeft => {
                try writer.writeAll("HTTP/1.1 413 Payload Too Large\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
                return;
            },
        };

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

/// 安全构建 JSON 响应，转义 content 中的特殊字符
fn buildJsonResponse(buf: []u8, content: []const u8) error{NoSpaceLeft}![]u8 {
    var off: usize = 0;
    const prefix = "{\"choices\":[{\"message\":{\"content\":\"";
    if (off + prefix.len > buf.len) return error.NoSpaceLeft;
    @memcpy(buf[off..][0..prefix.len], prefix);
    off += prefix.len;

    for (content) |c| {
        const esc: ?[]const u8 = switch (c) {
            '"' => "\\\"",
            '\\' => "\\\\",
            '\n' => "\\n",
            '\r' => "\\r",
            '\t' => "\\t",
            else => null,
        };
        if (esc) |e| {
            if (off + e.len > buf.len) return error.NoSpaceLeft;
            @memcpy(buf[off..][0..e.len], e);
            off += e.len;
        } else if (c < 0x20) {
            var hex_buf: [6]u8 = undefined;
            const hex = std.fmt.bufPrint(&hex_buf, "\\u{x:0>4}", .{c}) catch return error.NoSpaceLeft;
            if (off + hex.len > buf.len) return error.NoSpaceLeft;
            @memcpy(buf[off..][0..hex.len], hex);
            off += hex.len;
        } else {
            if (off + 1 > buf.len) return error.NoSpaceLeft;
            buf[off] = c;
            off += 1;
        }
    }

    const suffix = "\"}}]}\n";
    if (off + suffix.len > buf.len) return error.NoSpaceLeft;
    @memcpy(buf[off..][0..suffix.len], suffix);
    off += suffix.len;
    return buf[0..off];
}

fn extractJsonString(body: []const u8, key: []const u8) ?[]const u8 {
    // 尝试匹配 "key":" 和 "key": "
    var search_buf1: [256]u8 = undefined;
    var search_buf2: [256]u8 = undefined;
    const pattern_no_space = std.fmt.bufPrint(&search_buf1, "\"{s}\":\"", .{key}) catch return null;
    const pattern_space = std.fmt.bufPrint(&search_buf2, "\"{s}\": \"", .{key}) catch return null;
    const val_start = blk: {
        if (std.mem.indexOf(u8, body, pattern_no_space)) |s| break :blk s + pattern_no_space.len;
        if (std.mem.indexOf(u8, body, pattern_space)) |s| break :blk s + pattern_space.len;
        return null;
    };

    // 解析值，处理转义引号 \"
    var i = val_start;
    while (i < body.len) {
        if (body[i] == '\\' and i + 1 < body.len) {
            i += 2; // 跳过转义字符
            continue;
        }
        if (body[i] == '"') return body[val_start..i];
        i += 1;
    }
    return null;
}

fn extractJsonInt(body: []const u8, key: []const u8) ?u32 {
    var search_buf: [256]u8 = undefined;
    const pattern = std.fmt.bufPrint(&search_buf, "\"{s}\":", .{key}) catch return null;
    const start = std.mem.indexOf(u8, body, pattern) orelse return null;
    var i = start + pattern.len;
    // 跳过冒号后的空白
    while (i < body.len and std.ascii.isWhitespace(body[i])) : (i += 1) {}
    var end = i;
    while (end < body.len and std.ascii.isDigit(body[end])) : (end += 1) {}
    return std.fmt.parseInt(u32, body[i..end], 10) catch null;
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
