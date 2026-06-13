const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const is_darwin = target.result.os.tag.isDarwin();
    const is_android = target.result.os.tag == .linux and target.result.cpu.arch == .aarch64;

    // 核心库模块
    const lib_mod = b.addModule("mobile_moe", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Metal C 桥接源文件（macOS/iOS 专用）
    const metal_c_source = b.path("src/metal_bridge.c");

    // CLI 可执行文件
    const cli_mod = b.createModule(.{
        .root_source_file = b.path("src/cli.zig"),
        .target = target,
        .optimize = optimize,
    });
    cli_mod.addImport("mobile_moe", lib_mod);
    if (is_darwin) {
        cli_mod.addCSourceFile(.{ .file = metal_c_source, .flags = &.{} });
        cli_mod.link_libc = true;
        cli_mod.linkFramework("Metal", .{});
        cli_mod.linkFramework("Foundation", .{});
    }
    if (is_android) {
        cli_mod.link_libc = true;
        // Android 动态加载 OpenCL 需要 libdl
        cli_mod.linkSystemLibrary("dl", .{});
    }
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
    if (is_darwin) {
        server_mod.addCSourceFile(.{ .file = metal_c_source, .flags = &.{} });
        server_mod.link_libc = true;
        server_mod.linkFramework("Metal", .{});
        server_mod.linkFramework("Foundation", .{});
    }
    if (is_android) {
        server_mod.link_libc = true;
        server_mod.linkSystemLibrary("dl", .{});
    }
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
    if (is_darwin) {
        test_mod.addCSourceFile(.{ .file = metal_c_source, .flags = &.{} });
        test_mod.link_libc = true;
        test_mod.linkFramework("Metal", .{});
        test_mod.linkFramework("Foundation", .{});
    }
    if (is_android) {
        test_mod.link_libc = true;
        test_mod.linkSystemLibrary("dl", .{});
    }
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
    if (is_darwin) {
        integration_mod.addCSourceFile(.{ .file = metal_c_source, .flags = &.{} });
        integration_mod.link_libc = true;
        integration_mod.linkFramework("Metal", .{});
        integration_mod.linkFramework("Foundation", .{});
    }
    if (is_android) {
        integration_mod.link_libc = true;
        integration_mod.linkSystemLibrary("dl", .{});
    }
    const integration_tests = b.addTest(.{
        .root_module = integration_mod,
    });
    const run_integration_tests = b.addRunArtifact(integration_tests);

    const integration_step = b.step("test-integration", "运行集成测试");
    integration_step.dependOn(&run_integration_tests.step);

    // Android 交叉编译步骤说明
    if (is_android) {
        const android_step = b.step("android", "构建 Android 可执行文件 (aarch64-linux-android)");
        android_step.dependOn(b.getInstallStep());
        std.log.info("Android 交叉编译配置：", .{});
        std.log.info("  使用 Zig 内置交叉编译支持，无需外部 NDK 工具链", .{});
        std.log.info("  运行方式：zig build -Dtarget=aarch64-linux-android", .{});
    }
}
