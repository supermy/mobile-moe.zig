// BPE 分词器（GPT-2 风格，GLM-4 预处理方案）
//
// 从 GGUF 元数据中加载 tokenizer.ggml.tokens、tokenizer.ggml.scores、tokenizer.ggml.token_type
// 实现 BPE 编码与解码

const std = @import("std");
const GgufFile = @import("gguf.zig").GgufFile;
const MetadataValue = @import("gguf.zig").MetadataValue;

pub const TokenId = u32;
pub const SPECIAL_TOKEN = 0xFFFF_FFFF;

pub const Tokenizer = struct {
    allocator: std.mem.Allocator,
    tokens: []const []const u8,
    scores: []const f32,
    token_types: []const u32,
    token_to_id: std.StringHashMapUnmanaged(TokenId),
    bos_token: TokenId,
    eos_token: TokenId,
    pad_token: TokenId,

    // BPE 合并规则（从 token 排序推导）
    // GPT-2 BPE 不使用显式 merge 表，而是通过 score 排序

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, gguf: *const GgufFile) !Self {
        // 读取 token 列表
        const tokens_meta = gguf.metadata.get("tokenizer.ggml.tokens") orelse return error.MissingTokens;
        if (tokens_meta != .array) return error.InvalidTokensFormat;

        const n_tokens: u32 = @intCast(tokens_meta.array.len);
        const tokens = try allocator.alloc([]const u8, n_tokens);
        errdefer allocator.free(tokens);

        // 解析 token 字符串数组
        // GGUF 中 token 数组存储为连续的 string 元数据
        // 需要从原始字节中解析
        var pos: usize = 0;
        const data = tokens_meta.array.data;

        for (0..n_tokens) |i| {
            if (pos + 8 > data.len) return error.InvalidTokenData;
            const len = std.mem.readInt(u64, data[pos..][0..8], .little);
            pos += 8;
            if (len > 1024) return error.TokenTooLong;
            if (pos + len > data.len) return error.InvalidTokenData;
            tokens[i] = data[pos..][0..@intCast(len)];
            pos += len;
        }

        // 读取 scores
        const scores_meta = gguf.metadata.get("tokenizer.ggml.scores") orelse return error.MissingScores;
        const scores = try allocator.alloc(f32, n_tokens);
        errdefer allocator.free(scores);

        if (scores_meta == .array and scores_meta.array.elem_type == .float32) {
            const score_bytes = scores_meta.array.data;
            for (0..n_tokens) |i| {
                const offset = i * 4;
                if (offset + 4 > score_bytes.len) return error.InvalidScoreData;
                scores[i] = @bitCast(std.mem.readInt(u32, score_bytes[offset..][0..4], .little));
            }
        }

        // 读取 token_type
        const types_meta = gguf.metadata.get("tokenizer.ggml.token_type") orelse return error.MissingTokenTypes;
        const token_types = try allocator.alloc(u32, n_tokens);
        errdefer allocator.free(token_types);

        if (types_meta == .array and types_meta.array.elem_type == .int32) {
            const type_bytes = types_meta.array.data;
            for (0..n_tokens) |i| {
                const offset = i * 4;
                if (offset + 4 > type_bytes.len) return error.InvalidTypeData;
                token_types[i] = std.mem.readInt(u32, type_bytes[offset..][0..4], .little);
            }
        }

        // 构建 token → id 映射
        var token_to_id = std.StringHashMapUnmanaged(TokenId){};
        errdefer token_to_id.deinit(allocator);
        try token_to_id.ensureTotalCapacity(allocator, n_tokens);

        for (0..n_tokens) |i| {
            token_to_id.putAssumeCapacity(tokens[i], @intCast(i));
        }

        // 特殊 token
        const bos_token: TokenId = @intCast(gguf.metaInt("tokenizer.ggml.bos_token_id") orelse 0);
        const eos_token: TokenId = @intCast(gguf.metaInt("tokenizer.ggml.eos_token_id") orelse 2);
        const pad_token: TokenId = @intCast(gguf.metaInt("tokenizer.ggml.padding_token_id") orelse eos_token);

        return .{
            .allocator = allocator,
            .tokens = tokens,
            .scores = scores,
            .token_types = token_types,
            .token_to_id = token_to_id,
            .bos_token = bos_token,
            .eos_token = eos_token,
            .pad_token = pad_token,
        };
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.tokens);
        self.allocator.free(self.scores);
        self.allocator.free(self.token_types);
        self.token_to_id.deinit(self.allocator);
    }

    /// 查找 token ID
    pub fn tokenToId(self: *const Self, text: []const u8) ?TokenId {
        return self.token_to_id.get(text);
    }

    /// 获取 token 文本
    pub fn idToToken(self: *const Self, id: TokenId) ?[]const u8 {
        if (id >= self.tokens.len) return null;
        return self.tokens[id];
    }

    /// 简单编码：逐字节查找 + 特殊 token 匹配
    /// 完整 BPE 编码需要实现 GPT-2 merge 规则，此处先实现基础版本
    pub fn encode(self: *const Self, text: []const u8, out: *std.ArrayList(TokenId), allocator: std.mem.Allocator) !void {
        // GLM-4 使用 glm4 预处理方案
        // 简化实现：逐字符编码 + UTF-8 字节回退
        var i: usize = 0;
        while (i < text.len) {
            // 尝试最长匹配
            var matched = false;
            var end = text.len;
            while (end > i) : (end -= 1) {
                const slice = text[i..end];
                if (self.tokenToId(slice)) |id| {
                    try out.append(allocator, id);
                    i = end;
                    matched = true;
                    break;
                }
            }

            if (!matched) {
                // UTF-8 字节回退：将单个字节编码为 <0xHH> 格式
                const byte = text[i];
                var buf: [6]u8 = undefined;
                const hex = std.fmt.bufPrint(&buf, "<0x{X:0>2}>", .{byte}) catch unreachable;
                if (self.tokenToId(hex)) |id| {
                    try out.append(allocator, id);
                } else {
                    // 最终回退：跳过无法编码的字节
                    // 不使用 bos_token，因为 BOS 是特殊标记不应出现在编码文本中间
                }
                i += 1;
            }
        }
    }

    /// 解码 token 序列为文本
    pub fn decode(self: *const Self, tokens: []const TokenId, out: *std.ArrayList(u8), allocator: std.mem.Allocator) !void {
        for (tokens) |id| {
            const text = self.idToToken(id) orelse continue;
            // 处理字节回退 token（格式：<0xHH>）
            if (text.len == 6 and text[0] == '<' and text[1] == '0' and text[2] == 'x' and text[5] == '>') {
                const hex = std.fmt.parseInt(u8, text[3..5], 16) catch {
                    try out.appendSlice(allocator, text);
                    continue;
                };
                try out.append(allocator, hex);
                continue;
            }
            // 跳过控制 token（如 <s>, </s>, <pad> 等）
            if (text.len > 2 and text[0] == '<' and text[text.len - 1] == '>') continue;
            try out.appendSlice(allocator, text);
        }
    }
};
