const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // 核心库模块
    const lib_mod = b.addModule("mobile_moe", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });

    // CLI 可执行文件
    const cli_mod = b.createModule(.{
        .root_source_file = b.path("src/cli.zig"),
        .target = target,
        .optimize = optimize,
    });
    cli_mod.addImport("mobile_moe", lib_mod);
    const cli_exe = b.addExecutable(.{
        .name = "mobile-moe",
        .root_module = cli_mod,
    });
    b.installArtifact(cli_exe);

    // 服务端可执行文件
    const server_mod = b.createModule(.{
        .root_source_file = b.path("src/server.zig"),
        .target = target,
        .optimize = optimize,
    });
    server_mod.addImport("mobile_moe", lib_mod);
    const server_exe = b.addExecutable(.{
        .name = "mobile-moe-server",
        .root_module = server_mod,
    });
    b.installArtifact(server_exe);

    // 单元测试
    const test_mod = b.createModule(.{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });
    const lib_tests = b.addTest(.{
        .root_module = test_mod,
    });
    const run_lib_tests = b.addRunArtifact(lib_tests);

    const test_step = b.step("test", "运行单元测试");
    test_step.dependOn(&run_lib_tests.step);

    // 集成测试
    const integration_mod = b.createModule(.{
        .root_source_file = b.path("tests/integration.zig"),
        .target = target,
        .optimize = optimize,
    });
    integration_mod.addImport("mobile_moe", lib_mod);
    const integration_tests = b.addTest(.{
        .root_module = integration_mod,
    });
    const run_integration_tests = b.addRunArtifact(integration_tests);

    const integration_step = b.step("test-integration", "运行集成测试");
    integration_step.dependOn(&run_integration_tests.step);
}
