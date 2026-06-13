// 热节流 API：throttle(level)
//
// level 0: 正常运行
// level 1: 减少活跃专家数（Top-K 从 4 降至 3）
// level 2: 提高延迟率，热专家迁移至 CPU 冷池
// level 3: 最大延迟，最少 GPU 专家，暂停 NPU 注意力图

const std = @import("std");

pub const ThrottleLevel = enum(u2) {
    normal = 0,
    light = 1,
    medium = 2,
    heavy = 3,
};

pub const ThrottleState = struct {
    level: ThrottleLevel,
    /// 当前活跃专家数上限
    active_expert_limit: u32,
    /// 是否暂停 NPU 注意力
    npu_attention_paused: bool,
    /// 热专家是否迁移到 CPU 冷池
    hot_experts_on_cpu: bool,
    /// token 间延迟（微秒）
    token_delay_us: u32,
    /// 温度读数（毫摄氏度）
    temperature_mc: i32,

    const Self = @This();

    pub fn init() Self {
        return .{
            .level = .normal,
            .active_expert_limit = 4, // 默认 Top-4
            .npu_attention_paused = false,
            .hot_experts_on_cpu = false,
            .token_delay_us = 0,
            .temperature_mc = 0,
        };
    }

    /// 设置节流级别，自动调整所有参数
    pub fn throttle(self: *Self, level: ThrottleLevel) void {
        self.level = level;
        switch (level) {
            .normal => {
                self.active_expert_limit = 4;
                self.npu_attention_paused = false;
                self.hot_experts_on_cpu = false;
                self.token_delay_us = 0;
            },
            .light => {
                self.active_expert_limit = 3;
                self.npu_attention_paused = false;
                self.hot_experts_on_cpu = false;
                self.token_delay_us = 1000; // 1ms
            },
            .medium => {
                self.active_expert_limit = 2;
                self.npu_attention_paused = false;
                self.hot_experts_on_cpu = true;
                self.token_delay_us = 5000; // 5ms
            },
            .heavy => {
                self.active_expert_limit = 1;
                self.npu_attention_paused = true;
                self.hot_experts_on_cpu = true;
                self.token_delay_us = 20000; // 20ms
            },
        }
    }

    /// 根据温度自动选择节流级别
    /// temp_mc: 温度（毫摄氏度），如 85000 = 85°C
    pub fn updateFromTemperature(self: *Self, temp_mc: i32) void {
        self.temperature_mc = temp_mc;
        if (temp_mc > 85000) {
            self.throttle(.heavy);
        } else if (temp_mc > 80000) {
            // 80-85°C：渐进节流
            self.throttle(.medium);
        } else if (temp_mc > 75000) {
            self.throttle(.light);
        } else {
            self.throttle(.normal);
        }
    }

    /// 读取热区温度（跨平台）
    /// Linux: 遍历 /sys/class/thermal/thermal_zone*/temp
    /// macOS: 返回 0（TODO: 通过 IOKit 读取）
    pub fn readThermalZone(_: std.mem.Allocator) !i32 {
        return switch (comptime @import("builtin").os.tag) {
            .linux => readThermalZoneLinux(),
            .macos => 0,
            else => 0,
        };
    }

    fn readThermalZoneLinux() !i32 {
        var max_temp: i32 = 0;

        const dir_fd = std.posix.openat(
            std.posix.AT.FDCWD,
            "/sys/class/thermal",
            .{ .ACCMODE = .RDONLY, .DIRECTORY = true, .CLOEXEC = true },
            0,
        ) catch return max_temp;
        defer std.posix.close(dir_fd);

        var dir_buf: [2048]u8 align(@alignOf(std.posix.dirent)) = undefined;
        var offset: usize = 0;
        var consumed: usize = 0;

        while (true) {
            if (consumed >= offset) {
                const n = std.posix.getdents64(dir_fd, &dir_buf) catch return max_temp;
                if (n == 0) break;
                offset = @intCast(n);
                consumed = 0;
            }
            const entry: *std.posix.dirent = @ptrCast(@alignCast(&dir_buf[consumed]));
            consumed += entry.reclen;
            const name = std.mem.sliceTo(@as([*:0]u8, @ptrCast(&entry.name)), 0);
            if (!std.mem.startsWith(u8, name, "thermal_zone")) continue;

            var path_buf: [256]u8 = undefined;
            const path = std.fmt.bufPrint(&path_buf, "/sys/class/thermal/{s}/temp", .{name}) catch continue;

            const fd = std.posix.openat(
                std.posix.AT.FDCWD,
                path,
                .{ .ACCMODE = .RDONLY, .CLOEXEC = true },
                0,
            ) catch continue;
            defer std.posix.close(fd);

            var buf: [32]u8 = undefined;
            var total: usize = 0;
            while (total < buf.len) {
                const n2 = std.posix.read(fd, buf[total..]) catch break;
                if (n2 == 0) break;
                total += n2;
            }
            const temp_str = std.mem.trimRight(u8, buf[0..total], "\n\r ");
            const temp = std.fmt.parseInt(i32, temp_str, 10) catch continue;
            if (temp > max_temp) max_temp = temp;
        }

        return max_temp;
    }
};

// ========== 测试 ==========

test "ThrottleState init and throttle" {
    var state = ThrottleState.init();
    try std.testing.expectEqual(ThrottleLevel.normal, state.level);
    try std.testing.expectEqual(@as(u32, 4), state.active_expert_limit);

    state.throttle(.heavy);
    try std.testing.expectEqual(@as(u32, 1), state.active_expert_limit);
    try std.testing.expect(state.npu_attention_paused);
    try std.testing.expect(state.hot_experts_on_cpu);
}

test "ThrottleState updateFromTemperature" {
    var state = ThrottleState.init();

    state.updateFromTemperature(70000); // 70°C
    try std.testing.expectEqual(ThrottleLevel.normal, state.level);

    state.updateFromTemperature(76000); // 76°C
    try std.testing.expectEqual(ThrottleLevel.light, state.level);

    state.updateFromTemperature(82000); // 82°C
    try std.testing.expectEqual(ThrottleLevel.medium, state.level);

    state.updateFromTemperature(86000); // 86°C
    try std.testing.expectEqual(ThrottleLevel.heavy, state.level);
}
