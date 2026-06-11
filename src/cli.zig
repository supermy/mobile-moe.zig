// mobile-moe CLI：命令行推理界面
//
// 支持 REPL 交互、基准测试模式

const std = @import("std");
const mobile_moe = @import("mobile_moe");
const detect = @import("detect.zig");

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    var args_iter = std.process.Args.Iterator.init(init.minimal.args);

    // 跳过程序名
    _ = args_iter.next();

    const first_arg = args_iter.next() orelse {
        std.debug.print(
            \\用法: mobile-moe <model.gguf> [选项]
            \\       mobile-moe --detect
            \\
            \\选项:
            \\  --detect            检测 CPU/GPU/NPU 硬件信息
            \\  --max-tokens <N>    最大生成 token 数 (默认 512)
            \\  --max-seq-len <N>   最大序列长度 (默认 4096)
            \\  --benchmark         基准测试模式
            \\  --thermal-ok        允许设备上完整模型运行
            \\
        , .{});
        return;
    };

    // 检测模式：无需模型路径
    if (std.mem.eql(u8, first_arg, "--detect")) {
        const info = try detect.detect(allocator);
        defer detect.deinit(info, allocator);
        var buf: [4096]u8 = undefined;
        const written = try std.fmt.bufPrint(&buf,
            "=== CPU 检测 ===\n" ++
            "  架构:       {s}\n" ++
            "  品牌:       {s}\n" ++
            "  逻辑核心:   {d}\n" ++
            "  物理核心:   {d}\n" ++
            "  设备型号:   {s}\n" ++
            "\n=== GPU 检测 ===\n" ++
            "  名称:       {s}\n" ++
            "  Metal 支持: {s}\n" ++
            "\n=== NPU 检测 ===\n" ++
            "  可用:       {s}\n" ++
            "  名称:       {s}\n",
            .{
                info.cpu_arch,
                info.cpu_brand,
                info.cpu_cores,
                info.cpu_physical_cores,
                info.hw_model,
                info.gpu_name,
                if (info.gpu_metal_support) "是" else "否",
                if (info.npu_available) "是" else "否",
                info.npu_name,
            },
        );
        std.debug.print("{s}\n", .{written});
        return;
    }

    const model_path = first_arg;
    var max_tokens: u32 = 512;
    var max_seq_len: u32 = 4096;
    var benchmark = false;

    while (args_iter.next()) |arg| {
        if (std.mem.eql(u8, arg, "--max-tokens")) {
            const val = args_iter.next() orelse return;
            max_tokens = try std.fmt.parseInt(u32, val, 10);
        } else if (std.mem.eql(u8, arg, "--max-seq-len")) {
            const val = args_iter.next() orelse return;
            max_seq_len = try std.fmt.parseInt(u32, val, 10);
        } else if (std.mem.eql(u8, arg, "--benchmark")) {
            benchmark = true;
        }
    }

    std.debug.print("加载模型: {s}\n", .{model_path});
    var engine = try mobile_moe.Engine.load(allocator, model_path, max_seq_len, false);
    defer engine.deinit();

    const hp = engine.engine.weights.hp;
    std.debug.print("模型: GLM-4.7-Flash (DeepSeek2 架构)\n", .{});
    std.debug.print("  hidden_size={d} layers={d} heads={d} experts={d}+{d}\n", .{
        hp.hidden_size,
        hp.num_hidden_layers,
        hp.num_attention_heads,
        hp.n_routed_experts,
        hp.n_shared_experts,
    });
    std.debug.print("  vocab_size={d} kv_lora_rank={d} q_lora_rank={d}\n", .{
        hp.vocab_size,
        hp.kv_lora_rank,
        hp.q_lora_rank,
    });

    if (benchmark) {
        try runBenchmark(&engine, max_tokens, io);
        return;
    }

    // REPL 模式
    std.debug.print("\n输入提示词开始对话 (Ctrl+C 退出)\n\n", .{});

    var input_buf: [4096]u8 = undefined;
    while (true) {
        std.debug.print("> ", .{});

        // 使用 POSIX read 从 stdin 读取
        const n = std.posix.read(std.posix.STDIN_FILENO, &input_buf) catch break;
        if (n == 0) break;
        const line = std.mem.trimEnd(u8, input_buf[0..n], "\n\r ");
        if (line.len == 0) continue;

        engine.session(.{});

        // 调试：打印 token 数量
        var token_ids: std.ArrayList(u32) = .empty;
        defer token_ids.deinit(engine.allocator);
        try engine.engine.tok.encode(line, &token_ids, engine.allocator);
        std.debug.print("prompt token 数: {d}\n", .{token_ids.items.len});

        const gen_n = try engine.prompt(line, max_tokens);
        std.debug.print("生成 {d} tokens\n", .{gen_n});
    }
}

fn runBenchmark(engine: *mobile_moe.Engine, max_tokens: u32, io: std.Io) !void {
    std.debug.print("基准测试: 生成 {d} tokens...\n", .{max_tokens});

    engine.session(.{});

    const start = std.Io.Clock.now(.awake, io);
    const n = try engine.prompt("你好", max_tokens);
    const end = std.Io.Clock.now(.awake, io);

    const duration = start.durationTo(end);
    const elapsed_ns: f64 = @floatFromInt(duration.toNanoseconds());
    const elapsed_s = elapsed_ns / 1_000_000_000.0;
    const tokens_per_sec = @as(f64, @floatFromInt(n)) / elapsed_s;

    std.debug.print("结果: {d} tokens / {d:.2}s = {d:.1} tokens/s\n", .{ n, elapsed_s, tokens_per_sec });
}
