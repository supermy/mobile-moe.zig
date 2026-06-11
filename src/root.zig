// mobile-moe 公共 API
//
// 狭窄的公共接口：load, session, prompt, next, save_kv, load_kv, throttle

pub const gguf = @import("gguf.zig");
pub const model = @import("model.zig");
pub const dequant = @import("dequant.zig");
pub const tokenizer = @import("tokenizer.zig");
pub const math = @import("math.zig");
pub const inference = @import("inference.zig");
pub const scheduler = @import("scheduler.zig");
pub const thermal = @import("thermal.zig");
pub const kv_cache = @import("kv_cache.zig");

const std = @import("std");
const GgufFile = gguf.GgufFile;
const ModelWeights = model.ModelWeights;
const Tokenizer = tokenizer.Tokenizer;
const InferenceEngine = inference.InferenceEngine;
const ThrottleLevel = thermal.ThrottleLevel;
const TokenId = tokenizer.TokenId;

/// 引擎实例：整个推理系统的入口
pub const Engine = struct {
    allocator: std.mem.Allocator,
    engine: InferenceEngine,
    active_session: Session,

    const Self = @This();

    /// 加载模型
    /// path: GGUF 文件路径
    /// max_seq_len: 最大序列长度（KV 缓存容量）
    /// use_metal: 是否使用 Metal GPU（影响 mmap 策略）
    pub fn load(allocator: std.mem.Allocator, path: []const u8, max_seq_len: u32, use_metal: bool) !Self {
        var gguf_file = try gguf.loadFromFile(allocator, path, use_metal);
        var gguf_owned = true;
        errdefer if (gguf_owned) gguf_file.deinit();

        const weights = try ModelWeights.init(allocator, gguf_file);
        gguf_owned = false; // weights 现在拥有 gguf_file 的所有权
        errdefer @constCast(&weights).deinit(allocator);

        var tok = try Tokenizer.init(allocator, &gguf_file);
        errdefer tok.deinit();

        var engine = try InferenceEngine.init(allocator, weights, tok, max_seq_len);
        errdefer engine.deinit();

        const active_session = try Session.init(allocator, &engine);

        return .{
            .allocator = allocator,
            .engine = engine,
            .active_session = active_session,
        };
    }

    pub fn deinit(self: *Self) void {
        self.active_session.deinit();
        self.engine.deinit();
        // GGUF 数据由 ModelWeights.deinit → GgufFile.deinit 统一释放
    }

    /// 创建新会话
    pub fn session(self: *Self, config: SessionConfig) void {
        self.active_session.reset();
        _ = config;
    }

    /// 编码 prompt 并处理所有 token
    /// 返回生成的 token 数
    pub fn prompt(self: *Self, text: []const u8, max_tokens: u32) !u32 {
        var token_ids: std.ArrayList(TokenId) = .empty;
        defer token_ids.deinit(self.allocator);

        try self.engine.tok.encode(text, &token_ids, self.allocator);

        try self.active_session.sync(token_ids.items);
        return self.active_session.generate(max_tokens);
    }

    /// 生成文本并返回解码后的字符串（调用者负责释放返回的内存）
    pub fn generateText(self: *Self, text: []const u8, max_tokens: u32) ![]u8 {
        var token_ids: std.ArrayList(TokenId) = .empty;
        defer token_ids.deinit(self.allocator);
        try self.engine.tok.encode(text, &token_ids, self.allocator);
        try self.active_session.sync(token_ids.items);
        _ = try self.active_session.generate(max_tokens);

        var result = std.ArrayList(u8).empty;
        errdefer result.deinit(self.allocator);
        try self.engine.tok.decode(self.active_session.tokens.items, &result, self.allocator);
        return result.toOwnedSlice(self.allocator);
    }

    /// 生成下一个 token（通过 Session 维护状态）
    pub fn next(self: *Self, token_id: TokenId, pos: u32) !TokenId {
        return self.active_session.step(token_id, pos);
    }

    /// 热节流
    pub fn throttle(self: *Self, level: ThrottleLevel) void {
        self.engine.throttle_state.throttle(level);
    }

    /// 保存 KV 缓存到磁盘
    /// TODO: 当前 block-based KV cache 的序列化尚未实现，需要保存 block_table、
    /// blocks 元数据和 hash_map 才能正确恢复 Prefix Caching 状态。
    pub fn saveKv(_: *Self, _: []const u8) !void {
        return error.NotImplemented;
    }

    /// 从磁盘加载 KV 缓存
    /// TODO: 与 saveKv 配对实现，当前直接返回未实现错误以避免状态不一致。
    pub fn loadKv(_: *Self, _: []const u8) !void {
        return error.NotImplemented;
    }
};

pub const SessionConfig = struct {
    temperature: f32 = 1.0,
    top_p: f32 = 0.9,
    max_tokens: u32 = 512,
};

/// 会话：支持 Block-based KV 缓存和 Prefix Caching
/// 借鉴 vLLM PagedAttention 设计，通过 block_table 管理逻辑到物理 block 映射
pub const Session = struct {
    allocator: std.mem.Allocator,
    engine: *InferenceEngine,
    /// 当前完整 token 序列（prompt + generated）
    tokens: std.ArrayList(TokenId),
    /// 已写入 KV cache 的 token 数
    processed_len: u32,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, engine: *InferenceEngine) !Self {
        return .{
            .allocator = allocator,
            .engine = engine,
            .tokens = .empty,
            .processed_len = 0,
        };
    }

    pub fn deinit(self: *Self) void {
        self.tokens.deinit(self.allocator);
    }

    pub fn reset(self: *Self) void {
        self.tokens.clearRetainingCapacity();
        self.processed_len = 0;
        self.engine.kv.reset();
    }

    /// 同步 token 序列，自动 Prefix Caching
    pub fn sync(self: *Self, new_tokens: []const TokenId) !void {
        // 1. 尝试 Prefix Caching
        const reusable_blocks = self.engine.kv.findPrefixBlocks(new_tokens);
        if (reusable_blocks > 0) {
            try self.engine.kv.reusePrefixBlocks(new_tokens, reusable_blocks);
            self.processed_len = @intCast(reusable_blocks * kv_cache.BLOCK_SIZE);
        } else {
            self.engine.kv.reset();
            self.processed_len = 0;
        }

        // 2. 确保 block_table 容量足够（reset 后需要重新分配）
        try self.engine.kv.ensureCapacity(@intCast(new_tokens.len));

        // 3. 处理未缓存的 token（使用批量 forward）
        const start = self.processed_len;
        if (start < new_tokens.len) {
            const batch = new_tokens[start..];
            _ = try self.engine.forwardBatch(batch, @intCast(start));
        }

        // 4. 如果全部命中缓存但没有 forward 任何 token，需要 forward 最后一个 token 获得 logits
        if (start == new_tokens.len and new_tokens.len > 0) {
            const last_pos: u32 = @intCast(new_tokens.len - 1);
            _ = try self.engine.forward(new_tokens[last_pos], last_pos);
        }

        self.processed_len = @intCast(new_tokens.len);

        // 5. 更新 token 列表
        self.tokens.clearRetainingCapacity();
        try self.tokens.appendSlice(self.allocator, new_tokens);

        // 6. 注册新满 block 的 hash
        const total_blocks = new_tokens.len / kv_cache.BLOCK_SIZE;
        var prefix_hash: u64 = std.math.maxInt(u64);
        for (0..total_blocks) |bi| {
            const start_tok = bi * kv_cache.BLOCK_SIZE;
            if (reusable_blocks > 0 and bi < reusable_blocks) {
                prefix_hash = kv_cache.computeHash(new_tokens[start_tok..][0..kv_cache.BLOCK_SIZE], prefix_hash);
                continue;
            }
            self.engine.kv.cacheBlock(@intCast(bi), new_tokens[start_tok..][0..kv_cache.BLOCK_SIZE], prefix_hash);
            prefix_hash = kv_cache.computeHash(new_tokens[start_tok..][0..kv_cache.BLOCK_SIZE], prefix_hash);
        }
    }

    /// 单步生成：forward 一个 token 并返回下一个 token
    pub fn step(self: *Self, token_id: TokenId, pos: u32) !TokenId {
        const logits = try self.engine.forward(token_id, pos);
        const next_token = self.engine.sampleGreedy(logits);

        try self.tokens.append(self.allocator, token_id);
        self.processed_len += 1;

        // 注册新满 block 的 hash
        const tokens_len = self.tokens.items.len;
        if (tokens_len % kv_cache.BLOCK_SIZE == 0) {
            const block_idx = (tokens_len / kv_cache.BLOCK_SIZE) - 1;
            var prefix_hash: u64 = std.math.maxInt(u64);
            for (0..block_idx) |bi| {
                const start = bi * kv_cache.BLOCK_SIZE;
                prefix_hash = kv_cache.computeHash(self.tokens.items[start..][0..kv_cache.BLOCK_SIZE], prefix_hash);
            }
            self.engine.kv.cacheBlock(@intCast(block_idx), self.tokens.items[block_idx * kv_cache.BLOCK_SIZE..][0..kv_cache.BLOCK_SIZE], prefix_hash);
        }

        return next_token;
    }

    /// 自回归生成 max_tokens 个 token
    pub fn generate(self: *Self, max_tokens: u32) !u32 {
        var generated: u32 = 0;
        var last_token: TokenId = if (self.tokens.items.len > 0)
            self.engine.sampleGreedy(self.engine.logits)
        else
            self.engine.tok.bos_token;

        while (generated < max_tokens) : (generated += 1) {
            const pos = self.processed_len;
            const logits = try self.engine.forward(last_token, pos);
            const next_token = self.engine.sampleGreedy(logits);

            if (next_token == self.engine.tok.eos_token) break;

            try self.tokens.append(self.allocator, last_token);
            self.processed_len += 1;
            last_token = next_token;

            // 注册新满 block 的 hash
            const tokens_len = self.tokens.items.len;
            if (tokens_len % kv_cache.BLOCK_SIZE == 0) {
                const block_idx = (tokens_len / kv_cache.BLOCK_SIZE) - 1;
                var prefix_hash: u64 = std.math.maxInt(u64);
                for (0..block_idx) |bi| {
                    const start = bi * kv_cache.BLOCK_SIZE;
                    prefix_hash = kv_cache.computeHash(self.tokens.items[start..][0..kv_cache.BLOCK_SIZE], prefix_hash);
                }
                self.engine.kv.cacheBlock(@intCast(block_idx), self.tokens.items[block_idx * kv_cache.BLOCK_SIZE..][0..kv_cache.BLOCK_SIZE], prefix_hash);
            }
        }
        return generated;
    }
};

fn writeAll(fd: std.posix.fd_t, data: []const u8) !void {
    var offset: usize = 0;
    while (offset < data.len) {
        const n = try std.posix.write(fd, data[offset..]);
        if (n == 0) return error.UnexpectedEof;
        offset += n;
    }
}

fn readAll(fd: std.posix.fd_t, buf: []u8) !void {
    var offset: usize = 0;
    while (offset < buf.len) {
        const n = try std.posix.read(fd, buf[offset..]);
        if (n == 0) return error.EndOfStream;
        offset += n;
    }
}

test "Session block hash chain" {
    // 验证 computeHash 的链式计算正确性
    const testing = std.testing;

    const tokens_a = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
    const tokens_b = [_]u32{ 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32 };

    const h1 = kv_cache.computeHash(&tokens_a, std.math.maxInt(u64));
    const h2 = kv_cache.computeHash(&tokens_b, h1);
    const h2_direct = kv_cache.computeHash(&tokens_b, h1);

    // 相同输入应产生相同 hash
    try testing.expectEqual(h2, h2_direct);

    // 不同前缀应产生不同 hash
    const h2_other = kv_cache.computeHash(&tokens_b, 0);
    try testing.expect(h2 != h2_other);
}
