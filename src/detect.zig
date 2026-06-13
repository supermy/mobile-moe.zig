const std = @import("std");
const builtin = @import("builtin");

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
    opencl_available: bool,

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
        try writer.print("  OpenCL 可用: {s}\n", .{if (self.opencl_available) "是" else "否"});
        try writer.print("\n=== NPU 检测 ===\n", .{});
        try writer.print("  可用:       {s}\n", .{if (self.npu_available) "是" else "否"});
        try writer.print("  名称:       {s}\n", .{self.npu_name});
    }
};

// ---------- macOS 路径 ----------

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
    const prefixes = &[_][]const u8{
        "Mac",   "iPhone", "iPad",  "Apple", "Watch",   "Vision",
        "Mac15", "Mac14",  "Mac13", "Mac12", "MacBook",
    };
    for (prefixes) |prefix| {
        if (std.mem.startsWith(u8, model, prefix)) return true;
    }
    return false;
}

fn detectNPUFromModel(model: []const u8, cpu_brand: []const u8) bool {
    if (std.mem.indexOf(u8, cpu_brand, "Apple") != null) return true;
    if (std.mem.startsWith(u8, model, "iPhone")) return true;
    if (std.mem.startsWith(u8, model, "iPad")) return true;
    if (std.mem.startsWith(u8, model, "Watch")) return true;
    if (std.mem.startsWith(u8, model, "Vision")) return true;
    return false;
}

fn getGpuNameMac(allocator: std.mem.Allocator) ![]const u8 {
    const model = sysctlString(allocator, "hw.model") catch return try allocator.dupe(u8, "未知");
    defer allocator.free(model);

    if (detectAppleSilicon(model)) {
        const brand = sysctlString(allocator, "machdep.cpu.brand_string") catch return try allocator.dupe(u8, "Apple 集成 GPU");
        defer allocator.free(brand);
        var buf: [256]u8 = undefined;
        const name = std.fmt.bufPrint(&buf, "{s} GPU", .{brand}) catch return try allocator.dupe(u8, "Apple 集成 GPU");
        return try allocator.dupe(u8, name);
    }
    return try allocator.dupe(u8, "Intel/AMD 独立/集成 GPU");
}

fn detectMac(allocator: std.mem.Allocator) !DeviceInfo {
    const cpu_cores = sysctlInt("hw.ncpu");
    const cpu_physical_cores = sysctlInt("hw.physicalcpu");
    const cpu_brand = sysctlString(allocator, "machdep.cpu.brand_string") catch try allocator.dupe(u8, "未知");
    const hw_model = sysctlString(allocator, "hw.model") catch try allocator.dupe(u8, "未知");
    const cpu_arch = builtin.cpu.arch;

    const gpu_name = try getGpuNameMac(allocator);
    const metal_support = std.mem.indexOf(u8, gpu_name, "Apple") != null or detectAppleSilicon(hw_model);
    const npu_available = detectNPUFromModel(hw_model, cpu_brand);
    const npu_name = if (npu_available) try allocator.dupe(u8, "Apple Neural Engine") else try allocator.dupe(u8, "无");

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
        .opencl_available = false, // macOS 优先 Metal
    };
}

// ---------- Linux / Android 路径 ----------

fn readFileFirstLine(allocator: std.mem.Allocator, path: []const u8) ![]const u8 {
    const fd = try std.posix.openat(std.posix.AT.FDCWD, path, .{ .ACCMODE = .RDONLY, .CLOEXEC = true }, 0);
    defer _ = std.posix.system.close(fd);

    var buf: [4096]u8 = undefined;
    const n = try std.posix.read(fd, &buf);
    const content = std.mem.trim(u8, buf[0..n], " \t\r\n");
    return try allocator.dupe(u8, content);
}

fn parseCpuinfoCores(content: []const u8) u32 {
    var count: u32 = 0;
    var it = std.mem.split(u8, content, "\n");
    while (it.next()) |line| {
        if (std.mem.startsWith(u8, line, "processor\t:")) count += 1;
    }
    return if (count > 0) count else 1;
}

fn parseCpuinfoBrand(content: []const u8) []const u8 {
    var it = std.mem.split(u8, content, "\n");
    while (it.next()) |line| {
        if (std.mem.startsWith(u8, line, "Hardware\t:")) {
            const val = std.mem.trim(u8, line["Hardware\t:".len..], " \t");
            return val;
        }
        if (std.mem.startsWith(u8, line, "model name\t:")) {
            return std.mem.trim(u8, line["model name\t:".len..], " \t");
        }
    }
    return "未知";
}

fn accessOk(path: []const u8) bool {
    const rc = std.c.access(path, std.posix.X_OK);
    return rc == 0;
}

fn getGpuNameAndroid(allocator: std.mem.Allocator) ![]const u8 {
    // 优先读取 Adreno GPU 型号
    if (readFileFirstLine(allocator, "/sys/class/kgsl/kgsl-3d0/gpu_model")) |model| {
        return model;
    } else |_| {}

    // 次选：通过 OpenCL 库推断
    if (accessOk("/vendor/lib64/libOpenCL_adreno.so")) {
        return try allocator.dupe(u8, "Adreno GPU (OpenCL)");
    }
    if (accessOk("/vendor/lib64/libOpenCL.so")) {
        return try allocator.dupe(u8, "OpenCL GPU");
    }

    return try allocator.dupe(u8, "未知");
}

fn checkOpenCLAvailable() bool {
    return accessOk("/vendor/lib64/libOpenCL_adreno.so") or
        accessOk("/vendor/lib64/libOpenCL.so") or
        accessOk("/usr/lib/libOpenCL.so");
}

fn checkHexagonNPU() bool {
    return accessOk("/dev/hexagon") or accessOk("/dev/subsys_hvx");
}

fn readFileAllocPosix(allocator: std.mem.Allocator, path: []const u8, max_size: usize) ![]u8 {
    const fd = try std.posix.openat(std.posix.AT.FDCWD, path, .{ .ACCMODE = .RDONLY, .CLOEXEC = true }, 0);
    defer _ = std.posix.system.close(fd);

    var buf = try allocator.alloc(u8, max_size);
    errdefer allocator.free(buf);
    var total: usize = 0;
    while (total < max_size) {
        const n = try std.posix.read(fd, buf[total..]);
        if (n == 0) break;
        total += n;
    }
    return allocator.realloc(buf, total);
}

fn detectAndroid(allocator: std.mem.Allocator) !DeviceInfo {
    const cpu_arch = builtin.cpu.arch;

    // 读取 /proc/cpuinfo
    const cpuinfo = readFileAllocPosix(allocator, "/proc/cpuinfo", 8192) catch try allocator.dupe(u8, "");
    defer allocator.free(cpuinfo);

    const cpu_cores = parseCpuinfoCores(cpuinfo);
    const cpu_brand_raw = parseCpuinfoBrand(cpuinfo);
    const cpu_brand = try allocator.dupe(u8, cpu_brand_raw);

    // 物理核心：近似为逻辑核心 / 2（SMT 场景）
    const cpu_physical_cores = if (cpu_cores > 1) cpu_cores / 2 else cpu_cores;

    // 设备型号：尝试读取 build.prop
    const hw_model = readFileFirstLine(allocator, "/system/build.prop") catch blk: {
        // 简化：从 cpuinfo 的 Hardware 字段推断
        break :blk try allocator.dupe(u8, cpu_brand_raw);
    };

    const gpu_name = try getGpuNameAndroid(allocator);
    const opencl_avail = checkOpenCLAvailable();
    const npu_avail = checkHexagonNPU();

    return DeviceInfo{
        .cpu_cores = cpu_cores,
        .cpu_physical_cores = cpu_physical_cores,
        .cpu_brand = cpu_brand,
        .cpu_arch = @tagName(cpu_arch),
        .hw_model = hw_model,
        .gpu_name = gpu_name,
        .gpu_metal_support = false,
        .npu_available = npu_avail,
        .npu_name = if (npu_avail) try allocator.dupe(u8, "Qualcomm Hexagon") else try allocator.dupe(u8, "无"),
        .opencl_available = opencl_avail,
    };
}

// ---------- 通用入口 ----------

pub fn detect(allocator: std.mem.Allocator) !DeviceInfo {
    return switch (builtin.os.tag) {
        .macos => detectMac(allocator),
        .linux => detectAndroid(allocator),
        else => {
            // 通用回退
            const cpu_arch = builtin.cpu.arch;
            return DeviceInfo{
                .cpu_cores = 1,
                .cpu_physical_cores = 1,
                .cpu_brand = try allocator.dupe(u8, "未知"),
                .cpu_arch = @tagName(cpu_arch),
                .hw_model = try allocator.dupe(u8, "未知"),
                .gpu_name = try allocator.dupe(u8, "未知"),
                .gpu_metal_support = false,
                .npu_available = false,
                .npu_name = try allocator.dupe(u8, "无"),
                .opencl_available = false,
            };
        },
    };
}

pub fn deinit(self: DeviceInfo, allocator: std.mem.Allocator) void {
    allocator.free(self.cpu_brand);
    allocator.free(self.hw_model);
    allocator.free(self.gpu_name);
    allocator.free(self.npu_name);
}
