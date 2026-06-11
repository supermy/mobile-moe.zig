const std = @import("std");

pub const DeviceInfo = struct {
    cpu_cores: u32,
    cpu_physical_cores: u32,
    cpu_brand: []const u8,
    cpu_arch: []const u8,
    hw_model: []const u8,
    gpu_name: []const u8,
    gpu_metal_support: bool,
    npu_available: bool,
    npu_name: []const u8,

    pub fn print(self: DeviceInfo, writer: anytype) !void {
        try writer.print("=== CPU 检测 ===\n", .{});
        try writer.print("  架构:       {s}\n", .{self.cpu_arch});
        try writer.print("  品牌:       {s}\n", .{self.cpu_brand});
        try writer.print("  逻辑核心:   {d}\n", .{self.cpu_cores});
        try writer.print("  物理核心:   {d}\n", .{self.cpu_physical_cores});
        try writer.print("  设备型号:   {s}\n", .{self.hw_model});
        try writer.print("\n=== GPU 检测 ===\n", .{});
        try writer.print("  名称:       {s}\n", .{self.gpu_name});
        try writer.print("  Metal 支持: {s}\n", .{if (self.gpu_metal_support) "是" else "否"});
        try writer.print("\n=== NPU 检测 ===\n", .{});
        try writer.print("  可用:       {s}\n", .{if (self.npu_available) "是" else "否"});
        try writer.print("  名称:       {s}\n", .{self.npu_name});
    }
};

fn sysctlString(allocator: std.mem.Allocator, name: [:0]const u8) ![]const u8 {
    var len: usize = 0;
    const ret1 = std.c.sysctlbyname(name, null, &len, null, 0);
    if (ret1 != 0) return error.SysctlFailed;

    const buf = try allocator.alloc(u8, len);
    const ret2 = std.c.sysctlbyname(name, buf.ptr, &len, null, 0);
    if (ret2 != 0) {
        allocator.free(buf);
        return error.SysctlFailed;
    }
    // 去掉末尾的 \0
    if (len > 0 and buf[len - 1] == 0) {
        return allocator.realloc(buf, len - 1);
    }
    return buf;
}

fn sysctlInt(name: [:0]const u8) u32 {
    var value: u32 = 0;
    var len: usize = @sizeOf(u32);
    _ = std.c.sysctlbyname(name, &value, &len, null, 0);
    return value;
}

fn detectAppleSilicon(model: []const u8) bool {
    // Apple Silicon 型号以 Mac, iPhone, iPad, Apple 等开头
    const prefixes = &[_][]const u8{
        "Mac", "iPhone", "iPad", "Apple", "Watch", "Vision",
        "Mac15", "Mac14", "Mac13", "Mac12", "MacBook",
    };
    for (prefixes) |prefix| {
        if (std.mem.startsWith(u8, model, prefix)) return true;
    }
    return false;
}

fn detectNPUFromModel(model: []const u8, cpu_brand: []const u8) bool {
    // 只有 Apple Silicon 设备才有 Neural Engine
    // 通过 CPU 品牌字符串判断：包含 "Apple" 即为 Apple Silicon
    if (std.mem.indexOf(u8, cpu_brand, "Apple") != null) return true;
    // iOS 设备 (iPhone, iPad, Watch, Vision)
    if (std.mem.startsWith(u8, model, "iPhone")) return true;
    if (std.mem.startsWith(u8, model, "iPad")) return true;
    if (std.mem.startsWith(u8, model, "Watch")) return true;
    if (std.mem.startsWith(u8, model, "Vision")) return true;
    return false;
}

fn getGpuName(allocator: std.mem.Allocator) ![]const u8 {
    // macOS 上优先通过 sysctl 获取 GPU 相关信息
    const model = sysctlString(allocator, "hw.model") catch return try allocator.dupe(u8, "未知");
    defer allocator.free(model);

    // Apple Silicon: GPU 集成在芯片中，根据型号推断
    if (detectAppleSilicon(model)) {
        const brand = sysctlString(allocator, "machdep.cpu.brand_string") catch return try allocator.dupe(u8, "Apple 集成 GPU");
        defer allocator.free(brand);
        // 品牌字符串如 "Apple M1 Max" -> GPU 为 "Apple M1 Max GPU"
        var buf: [256]u8 = undefined;
        const name = std.fmt.bufPrint(&buf, "{s} GPU", .{brand}) catch return try allocator.dupe(u8, "Apple 集成 GPU");
        return try allocator.dupe(u8, name);
    }

    // Intel Mac: 尝试读取显卡信息（简化处理）
    return try allocator.dupe(u8, "Intel/AMD 独立/集成 GPU");
}

pub fn detect(allocator: std.mem.Allocator) !DeviceInfo {
    const cpu_cores = sysctlInt("hw.ncpu");
    const cpu_physical_cores = sysctlInt("hw.physicalcpu");
    const cpu_brand = sysctlString(allocator, "machdep.cpu.brand_string") catch try allocator.dupe(u8, "未知");
    const hw_model = sysctlString(allocator, "hw.model") catch try allocator.dupe(u8, "未知");
    const cpu_arch = @import("builtin").cpu.arch;

    const gpu_name = try getGpuName(allocator);
    const metal_support = std.mem.indexOf(u8, gpu_name, "Apple") != null or
        detectAppleSilicon(hw_model);

    const npu_available = detectNPUFromModel(hw_model, cpu_brand);
    const npu_name = if (npu_available) "Apple Neural Engine" else "无";

    return DeviceInfo{
        .cpu_cores = cpu_cores,
        .cpu_physical_cores = cpu_physical_cores,
        .cpu_brand = cpu_brand,
        .cpu_arch = @tagName(cpu_arch),
        .hw_model = hw_model,
        .gpu_name = gpu_name,
        .gpu_metal_support = metal_support,
        .npu_available = npu_available,
        .npu_name = npu_name,
    };
}

pub fn deinit(self: DeviceInfo, allocator: std.mem.Allocator) void {
    allocator.free(self.cpu_brand);
    allocator.free(self.hw_model);
    allocator.free(self.gpu_name);
}
