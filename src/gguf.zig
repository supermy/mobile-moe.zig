// GGUF 二进制格式解析器
// 规范参考：https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
//
// GGUF v3 格式布局：
//   [header: magic + version + tensor_count + metadata_kv_count]
//   [metadata_kv × N]
//   [tensor_info × tensor_count]
//   [padding to ALIGNMENT]
//   [tensor_data]

const std = @import("std");
const mem = std.mem;
const io = std.io;
const Allocator = mem.Allocator;

pub const GGUF_MAGIC = 0x46554747; // "GGUF" 小端序
pub const GGUF_VERSION: u32 = 3;
pub const GGUF_DEFAULT_ALIGNMENT: u64 = 32;

// GGML 张量类型枚举
pub const GgmlType = enum(u32) {
    f32 = 0,
    f16 = 1,
    q4_0 = 2,
    q4_1 = 3,
    q5_0 = 6,
    q5_1 = 7,
    q8_0 = 8,
    q8_1 = 9,
    q2_k = 10,
    q3_k = 11,
    q4_k = 12,
    q5_k = 13,
    q6_k = 14,
    q8_k = 15,
    iq2_xxs = 16,
    iq2_xs = 17,
    iq3_xxs = 18,
    iq1_s = 19,
    iq4_nl = 20,
    iq3_s = 21,
    iq2_s = 22,
    iq4_xs = 23,
    i8 = 24,
    i16 = 25,
    i32 = 26,
    i64 = 27,
    f64 = 28,
    iq1_m = 29,
    bf16 = 30,
    tq1_0 = 34,
    tq2_0 = 35,
    mxfp4 = 39,
    nvfp4 = 40,
    q1_0 = 41,
    _,

    /// 每种量化类型每个块包含的元素数
    pub fn blockSize(self: GgmlType) u64 {
        return switch (self) {
            .f32, .f16, .bf16, .i8, .i16, .i32, .i64, .f64 => 1,
            .q4_0, .q4_1 => 32,
            .q5_0, .q5_1 => 32,
            .q8_0, .q8_1 => 32,
            .q2_k, .q3_k, .q4_k, .q5_k, .q6_k, .q8_k => 256,
            .iq2_xxs, .iq2_xs, .iq3_xxs, .iq1_s, .iq4_nl, .iq3_s, .iq2_s, .iq4_xs, .iq1_m => 256,
            .tq1_0, .tq2_0 => 256,
            .mxfp4, .nvfp4 => 32,
            .q1_0 => 32,
            else => 1,
        };
    }

    /// 每种量化类型每个块占用的字节数
    pub fn blockBytes(self: GgmlType) u64 {
        return switch (self) {
            .f32 => 4,
            .f16 => 2,
            .bf16 => 2,
            .i8 => 1,
            .i16 => 2,
            .i32 => 4,
            .i64 => 8,
            .f64 => 8,
            .q4_0 => 18, // 2 (scale) + 16 (4bit × 32 / 8)
            .q4_1 => 20, // 2 (scale) + 2 (min) + 16
            .q5_0 => 22,
            .q5_1 => 24,
            .q8_0 => 34, // 2 (scale) + 32 (8bit × 32 / 8)
            .q8_1 => 36,
            .q2_k => 256 / 2 + 2 + 2 + 16, // 84
            .q3_k => 110,
            .q4_k => 144,
            .q5_k => 176,
            .q6_k => 210,
            .q8_k => 292,
            .iq2_xxs => 66, // 2 (d) + 64 (qs[32] × 2)
            .iq2_xs => 74,
            .iq3_xxs => 98,
            .iq1_s => 33,
            .iq4_nl => 18,
            .iq3_s => 110,
            .iq2_s => 82,
            .iq4_xs => 136,
            .iq1_m => 56,
            .tq1_0 => 33,
            .tq2_0 => 66,
            else => 0,
        };
    }

    /// 计算给定元素数的张量所需字节数
    pub fn tensorBytes(self: GgmlType, n_elements: u64) u64 {
        const bs = self.blockSize();
        const bb = self.blockBytes();
        return ((n_elements + bs - 1) / bs) * bb;
    }
};

// 元数据值类型枚举
pub const MetadataValueType = enum(u32) {
    uint8 = 0,
    int8 = 1,
    uint16 = 2,
    int16 = 3,
    uint32 = 4,
    int32 = 5,
    float32 = 6,
    bool = 7,
    string = 8,
    array = 9,
    uint64 = 10,
    int64 = 11,
    float64 = 12,
    _,
};

// 元数据值
pub const MetadataValue = union(MetadataValueType) {
    uint8: u8,
    int8: i8,
    uint16: u16,
    int16: i16,
    uint32: u32,
    int32: i32,
    float32: f32,
    bool: bool,
    string: []const u8,
    array: MetadataArray,
    uint64: u64,
    int64: i64,
    float64: f64,

    pub const MetadataArray = struct {
        elem_type: MetadataValueType,
        len: u64,
        data: []const u8, // 原始字节数组，按 elem_type 解析
    };
};

// 张量描述符
pub const TensorInfo = struct {
    name: []const u8,
    n_dimensions: u32,
    dimensions: [4]u64,
    ggml_type: GgmlType,
    offset: u64, // 相对于 tensor_data 区的偏移

    pub fn nElements(self: *const TensorInfo) u64 {
        var n: u64 = 1;
        for (self.dimensions[0..self.n_dimensions]) |d| {
            n *= d;
        }
        return n;
    }
};

// GGUF 文件解析结果
pub const GgufFile = struct {
    allocator: Allocator,
    version: u32,
    tensor_count: u64,
    metadata: std.StringArrayHashMapUnmanaged(MetadataValue),
    tensor_infos: std.StringArrayHashMapUnmanaged(TensorInfo),
    alignment: u64,
    /// tensor_data 区在文件中的起始偏移
    data_offset: u64,
    /// mmap 映射的文件内容
    mapped_data: ?[]align(std.heap.page_size_min) const u8 = null,

    pub fn deinit(self: *GgufFile) void {
        var mit = self.metadata.iterator();
        while (mit.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            switch (entry.value_ptr.*) {
                .string => |s| self.allocator.free(s),
                .array => |a| self.allocator.free(a.data),
                else => {},
            }
        }
        self.metadata.deinit(self.allocator);

        var tit = self.tensor_infos.iterator();
        while (tit.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.tensor_infos.deinit(self.allocator);

        if (self.mapped_data) |md| {
            const ptr = @constCast(md.ptr);
            std.posix.munmap(ptr[0..md.len]);
        }

        self.* = undefined;
    }

    /// 获取 tensor_data 区的指针
    /// 仅在通过 loadFromFile（mmap）加载后可用
    pub fn tensorDataPtr(self: *const GgufFile) ?[*]const u8 {
        const base = if (self.mapped_data) |md| md.ptr else return null;
        return base + self.data_offset;
    }

    /// 获取指定张量的数据指针
    pub fn tensorPtr(self: *const GgufFile, name: []const u8) ?[*]const u8 {
        const ti = self.tensor_infos.get(name) orelse return null;
        const base = self.tensorDataPtr() orelse return null;
        return base + ti.offset;
    }

    /// 获取元数据字符串
    pub fn metaString(self: *const GgufFile, key: []const u8) ?[]const u8 {
        const v = self.metadata.get(key) orelse return null;
        if (v == .string) return v.string;
        return null;
    }

    /// 获取元数据整数
    pub fn metaInt(self: *const GgufFile, key: []const u8) ?i64 {
        const v = self.metadata.get(key) orelse return null;
        return switch (v) {
            .uint8 => |x| @as(i64, x),
            .int8 => |x| @as(i64, x),
            .uint16 => |x| @as(i64, x),
            .int16 => |x| @as(i64, x),
            .uint32 => |x| @as(i64, x),
            .int32 => |x| @as(i64, x),
            .uint64 => |x| @as(i64, @intCast(x)),
            .int64 => |x| x,
            else => null,
        };
    }

    /// 获取元数据浮点数
    pub fn metaFloat(self: *const GgufFile, key: []const u8) ?f64 {
        const v = self.metadata.get(key) orelse return null;
        return switch (v) {
            .float32 => |x| @as(f64, x),
            .float64 => |x| x,
            else => null,
        };
    }
};

// 从 Reader 解析 GGUF 文件
pub fn parse(allocator: Allocator, reader: anytype) !GgufFile {
    // 1. 读取文件头
    const magic = try reader.readInt(u32, .little);
    if (magic != GGUF_MAGIC) {
        return error.InvalidMagic;
    }

    const version = try reader.readInt(u32, .little);
    if (version != GGUF_VERSION) {
        return error.UnsupportedVersion;
    }

    const tensor_count = try reader.readInt(u64, .little);
    const metadata_kv_count = try reader.readInt(u64, .little);

    // 2. 解析元数据
    var metadata = std.StringArrayHashMapUnmanaged(MetadataValue){};
    errdefer {
        var it = metadata.iterator();
        while (it.next()) |entry| {
            allocator.free(entry.key_ptr.*);
            switch (entry.value_ptr.*) {
                .string => |s| allocator.free(s),
                .array => |a| allocator.free(a.data),
                else => {},
            }
        }
        metadata.deinit(allocator);
    }

    for (0..metadata_kv_count) |_| {
        const key = try readString(allocator, reader);
        errdefer allocator.free(key);

        const vtype: MetadataValueType = @enumFromInt(try reader.readInt(u32, .little));
        const value = try readMetadataValue(allocator, reader, vtype);

        try metadata.put(allocator, key, value);
    }

    // 3. 解析张量描述符
    var tensor_infos = std.StringArrayHashMapUnmanaged(TensorInfo){};
    errdefer {
        var it = tensor_infos.iterator();
        while (it.next()) |entry| {
            allocator.free(entry.key_ptr.*);
        }
        tensor_infos.deinit(allocator);
    }

    for (0..tensor_count) |_| {
        const name = try readString(allocator, reader);
        errdefer allocator.free(name);

        const n_dimensions = try reader.readInt(u32, .little);
        if (n_dimensions > 4) return error.TooManyDimensions;

        var dimensions: [4]u64 = .{ 1, 1, 1, 1 };
        for (0..n_dimensions) |i| {
            dimensions[i] = try reader.readInt(u64, .little);
        }

        const ggml_type: GgmlType = @enumFromInt(try reader.readInt(u32, .little));
        const offset = try reader.readInt(u64, .little);

        try tensor_infos.put(allocator, name, .{
            .name = name,
            .n_dimensions = n_dimensions,
            .dimensions = dimensions,
            .ggml_type = ggml_type,
            .offset = offset,
        });
    }

    // 4. 计算 tensor_data 区偏移
    const alignment: u64 = blk: {
        if (metadata.get("general.alignment")) |v| {
            if (v == .uint32) break :blk v.uint32;
            if (v == .uint64) break :blk v.uint64;
        }
        break :blk GGUF_DEFAULT_ALIGNMENT;
    };

    // 当前位置即为 tensor_info 之后的偏移
    // 需要对齐到 alignment 边界
    const current_offset = try reader.getPos();
    const data_offset = current_offset + (alignment - (current_offset % alignment)) % alignment;

    return .{
        .allocator = allocator,
        .version = version,
        .tensor_count = tensor_count,
        .metadata = metadata,
        .tensor_infos = tensor_infos,
        .alignment = alignment,
        .data_offset = data_offset,
    };
}

/// 从文件路径加载 GGUF，使用 mmap 映射权重数据
/// use_metal: 当 true 时使用 MAP_SHARED（Metal GPU 需要零拷贝 MTLBuffer）
///           当 false 时使用 MAP_PRIVATE（CPU 路径，Darwin 上避免 VM panic）
pub fn loadFromFile(allocator: Allocator, path: []const u8, use_metal: bool) !GgufFile {
    const fd = try std.posix.openat(
        std.posix.AT.FDCWD,
        path,
        .{ .ACCMODE = .RDONLY, .CLOEXEC = true },
        0,
    );
    errdefer {
        _ = std.posix.system.close(fd);
    }

    var reader = PosixFileReader.init(fd);

    var result = try parse(allocator, &reader);
    errdefer result.deinit();

    // 获取实际文件大小（而非已读字节数）
    // reader.bytes_read 只包含 header+metadata+tensor_info，不含张量数据
    var stat: std.posix.Stat = undefined;
    const rc = std.posix.system.fstat(fd, &stat);
    if (rc != 0) return error.StatFailed;
    const file_size: usize = @intCast(stat.size);

    // CPU 路径使用 MAP_PRIVATE，Metal 路径使用 MAP_SHARED
    // 在 Darwin 上，CPU 路径必须使用 MAP_PRIVATE 以避免内核 VM panic
    const map_flags = if (use_metal)
        std.posix.MAP{ .TYPE = .SHARED }
    else
        std.posix.MAP{ .TYPE = .PRIVATE };
    const mapped = try std.posix.mmap(
        null,
        file_size,
        .{ .READ = true },
        map_flags,
        fd,
        0,
    );
    result.mapped_data = mapped;

    _ = std.posix.system.close(fd);

    // 预取页面，避免解码时延迟缺页中断
    std.posix.madvise(@constCast(mapped.ptr), mapped.len, std.posix.MADV.WILLNEED) catch {};

    return result;
}

/// 简单的 POSIX 文件读取器，提供 parse 所需的接口
const PosixFileReader = struct {
    fd: std.posix.fd_t,
    bytes_read: usize,

    fn init(fd: std.posix.fd_t) PosixFileReader {
        return .{ .fd = fd, .bytes_read = 0 };
    }

    fn readInt(self: *PosixFileReader, comptime T: type, endian: std.builtin.Endian) !T {
        var buf: [@sizeOf(T)]u8 = undefined;
        try self.readNoEof(&buf);
        return std.mem.readInt(T, &buf, endian);
    }

    fn readNoEof(self: *PosixFileReader, buf: []u8) !void {
        var offset: usize = 0;
        while (offset < buf.len) {
            const n = try std.posix.read(self.fd, buf[offset..]);
            if (n == 0) return error.EndOfStream;
            offset += n;
        }
        self.bytes_read += buf.len;
    }

    fn getPos(self: *PosixFileReader) !u64 {
        return @intCast(self.bytes_read);
    }
};

fn readString(allocator: Allocator, reader: anytype) ![]const u8 {
    const len = try reader.readInt(u64, .little);
    if (len > 65536) return error.StringTooLong;
    const buf = try allocator.alloc(u8, len);
    errdefer allocator.free(buf);
    try reader.readNoEof(buf);
    return buf;
}

fn readMetadataValue(allocator: Allocator, reader: anytype, vtype: MetadataValueType) !MetadataValue {
    return switch (vtype) {
        .uint8 => .{ .uint8 = try reader.readInt(u8, .little) },
        .int8 => .{ .int8 = try reader.readInt(i8, .little) },
        .uint16 => .{ .uint16 = try reader.readInt(u16, .little) },
        .int16 => .{ .int16 = try reader.readInt(i16, .little) },
        .uint32 => .{ .uint32 = try reader.readInt(u32, .little) },
        .int32 => .{ .int32 = try reader.readInt(i32, .little) },
        .float32 => .{ .float32 = @bitCast(try reader.readInt(u32, .little)) },
        .bool => .{ .bool = (try reader.readInt(u8, .little)) != 0 },
        .string => .{ .string = try readString(allocator, reader) },
        .uint64 => .{ .uint64 = try reader.readInt(u64, .little) },
        .int64 => .{ .int64 = try reader.readInt(i64, .little) },
        .float64 => .{ .float64 = @bitCast(try reader.readInt(u64, .little)) },
        .array => {
            const elem_type: MetadataValueType = @enumFromInt(try reader.readInt(u32, .little));
            const len = try reader.readInt(u64, .little);
            if (len > 10_000_000) return error.ArrayTooLong;

            // string 数组需要逐个读取字符串
            if (elem_type == .string) {
                // 预分配字符串偏移表：存储每个字符串的 (offset, len) 对
                // 然后连续存储所有字符串数据
                var strings = try allocator.alloc([]const u8, len);
                errdefer {
                    for (strings) |s| allocator.free(s);
                    allocator.free(strings);
                }

                // 先收集所有字符串数据到连续缓冲区
                // 使用 ArrayList 累积所有字符串字节
                var all_data: std.ArrayList(u8) = .empty;
                defer all_data.deinit(allocator);

                var offsets = try allocator.alloc(u64, len);
                defer allocator.free(offsets);
                var lengths = try allocator.alloc(u64, len);
                defer allocator.free(lengths);

                for (0..len) |i| {
                    const str_len = try reader.readInt(u64, .little);
                    if (str_len > 65536) return error.StringTooLong;
                    offsets[i] = all_data.items.len;
                    lengths[i] = str_len;
                    if (str_len > 0) {
                        const old_len = all_data.items.len;
                        try all_data.resize(allocator, old_len + str_len);
                        try reader.readNoEof(all_data.items[old_len..][0..str_len]);
                    }
                }

                // 从连续缓冲区切片创建字符串
                const base_data = try allocator.dupe(u8, all_data.items);
                errdefer allocator.free(base_data);

                for (0..len) |i| {
                    strings[i] = base_data[offsets[i]..][0..lengths[i]];
                }

                // 将字符串数组编码为原始字节存储
                // 格式：[n_strings(u64)] [str_len(u64) + str_data × n]
                // 这样 tokenizer 可以从 data 中解析
                var encoded: std.ArrayList(u8) = .empty;
                defer encoded.deinit(allocator);

                for (0..len) |i| {
                    const s = strings[i];
                    var len_buf: [8]u8 = undefined;
                    std.mem.writeInt(u64, &len_buf, s.len, .little);
                    try encoded.appendSlice(allocator, &len_buf);
                    try encoded.appendSlice(allocator, s);
                }

                allocator.free(strings);
                allocator.free(base_data);

                return .{ .array = .{
                    .elem_type = .string,
                    .len = len,
                    .data = try allocator.dupe(u8, encoded.items),
                } };
            }

            // 数值类型数组：直接读取原始字节
            const elem_size: usize = switch (elem_type) {
                .uint8, .int8, .bool => 1,
                .uint16, .int16 => 2,
                .uint32, .int32, .float32 => 4,
                .uint64, .int64, .float64 => 8,
                else => return error.UnsupportedArrayType,
            };
            const total_bytes = len * elem_size;
            const data = try allocator.alloc(u8, total_bytes);
            errdefer allocator.free(data);
            try reader.readNoEof(data);

            return .{ .array = .{
                .elem_type = elem_type,
                .len = len,
                .data = data,
            } };
        },
        else => return error.UnsupportedMetadataType,
    };
}

// ========== 测试 ==========

test "GGUF magic and version constants" {
    try std.testing.expectEqual(@as(u32, 0x46554747), GGUF_MAGIC);
    try std.testing.expectEqual(@as(u32, 3), GGUF_VERSION);
}

test "GgmlType blockSize and blockBytes" {
    // IQ2_XXS: 256 元素/块, 66 字节/块
    try std.testing.expectEqual(@as(u64, 256), GgmlType.iq2_xxs.blockSize());
    try std.testing.expectEqual(@as(u64, 66), GgmlType.iq2_xxs.blockBytes());

    // F32: 1 元素/块, 4 字节/块
    try std.testing.expectEqual(@as(u64, 1), GgmlType.f32.blockSize());
    try std.testing.expectEqual(@as(u64, 4), GgmlType.f32.blockBytes());

    // Q8_0: 32 元素/块, 34 字节/块
    try std.testing.expectEqual(@as(u64, 32), GgmlType.q8_0.blockSize());
    try std.testing.expectEqual(@as(u64, 34), GgmlType.q8_0.blockBytes());
}

test "GgmlType tensorBytes" {
    // IQ2_XXS: 256 元素 = 1 块 = 66 字节
    try std.testing.expectEqual(@as(u64, 66), GgmlType.iq2_xxs.tensorBytes(256));
    // F32: 4 元素 = 16 字节
    try std.testing.expectEqual(@as(u64, 16), GgmlType.f32.tensorBytes(4));
}
