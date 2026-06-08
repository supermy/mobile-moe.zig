// 集成测试

const std = @import("std");
const mobile_moe = @import("mobile_moe");

test "GGUF 文件解析 - 真实模型" {
    const allocator = std.testing.allocator;

    const model_path = "/Users/moyong/project/ai/models/GLM-4.7-Flash-UD-IQ2_XXS.gguf";

    // 检查文件是否存在
    const fd = std.posix.openat(
        std.posix.AT.FDCWD,
        model_path,
        .{ .ACCMODE = .RDONLY, .CLOEXEC = true },
        0,
    ) catch {
        std.debug.print("跳过：模型文件不存在\n", .{});
        return;
    };
    _ = std.posix.system.close(fd);

    var gguf = try mobile_moe.gguf.loadFromFile(allocator, model_path, false);
    defer gguf.deinit();

    // 验证基本元数据
    const arch = gguf.metaString("general.architecture");
    try std.testing.expect(arch != null);
    try std.testing.expectEqualStrings("deepseek2", arch.?);

    const name = gguf.metaString("general.name");
    try std.testing.expect(name != null);

    // 验证张量数量
    try std.testing.expect(gguf.tensor_count > 0);

    std.debug.print("GGUF 解析成功: arch={s}, tensors={d}, metadata={d}\n", .{
        arch.?,
        gguf.tensor_count,
        gguf.metadata.count(),
    });
}

test "超参数提取 - 仅解析元数据" {
    const allocator = std.testing.allocator;

    const model_path = "/Users/moyong/project/ai/models/GLM-4.7-Flash-UD-IQ2_XXS.gguf";

    const fd = std.posix.openat(
        std.posix.AT.FDCWD,
        model_path,
        .{ .ACCMODE = .RDONLY, .CLOEXEC = true },
        0,
    ) catch {
        std.debug.print("跳过：模型文件不存在\n", .{});
        return;
    };
    _ = std.posix.system.close(fd);

    var gguf = try mobile_moe.gguf.loadFromFile(allocator, model_path, false);
    defer gguf.deinit();

    // 提取超参数
    const hp = try mobile_moe.model.HyperParams.fromGguf(&gguf);

    // 验证关键超参数
    try std.testing.expectEqual(@as(u32, 2048), hp.hidden_size);
    try std.testing.expectEqual(@as(u32, 47), hp.num_hidden_layers);
    try std.testing.expectEqual(@as(u32, 20), hp.num_attention_heads);
    try std.testing.expectEqual(@as(u32, 64), hp.n_routed_experts);
    try std.testing.expectEqual(@as(u32, 1), hp.n_shared_experts);
    try std.testing.expectEqual(@as(u32, 4), hp.num_experts_per_tok);
    try std.testing.expectEqual(@as(u32, 512), hp.kv_lora_rank);
    try std.testing.expectEqual(@as(u32, 768), hp.q_lora_rank);

    std.debug.print("超参数验证通过: hidden={d} layers={d} experts={d}+{d} kv_rank={d} q_rank={d}\n", .{
        hp.hidden_size,
        hp.num_hidden_layers,
        hp.n_routed_experts,
        hp.n_shared_experts,
        hp.kv_lora_rank,
        hp.q_lora_rank,
    });
}
