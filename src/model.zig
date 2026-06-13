// GLM-4.7-Flash (DeepSeek2 架构) 模型定义
//
// 架构特征：
//   - MLA (Multi-Latent Attention)：KV 通过 LoRA 压缩到低秩潜在空间
//   - MoE：64 路由专家 + 1 共享专家，Top-4 路由
//   - 1 个稠密层 + 46 个 MoE 层（共 47 层）
//   - IQ2_XXS 量化 (~2.06 bpw)

const std = @import("std");
const GgufFile = @import("gguf.zig").GgufFile;
const GgmlType = @import("gguf.zig").GgmlType;

/// 模型超参数，从 GGUF 元数据提取
pub const HyperParams = struct {
    // 基础维度
    hidden_size: u32,
    intermediate_size: u32, // 共享专家/稠密层 FFN 中间维度
    moe_intermediate_size: u32, // 路由专家 FFN 中间维度
    num_hidden_layers: u32,
    num_attention_heads: u32,
    num_key_value_heads: u32,
    vocab_size: u32,
    max_position_embeddings: u32,

    // MLA 注意力参数
    kv_lora_rank: u32, // KV 压缩 LoRA 秩
    q_lora_rank: u32, // Q 压缩 LoRA 秩
    qk_rope_head_dim: u32, // 应用 RoPE 的 Q/K 头维度 (= rope_dimension_count)
    qk_nope_head_dim: u32, // 不应用 RoPE 的 Q/K 头维度 (= key_length - key_length_mla)
    v_head_dim: u32, // Value 头维度 (= value_length_mla)

    // MoE 参数
    n_routed_experts: u32,
    n_shared_experts: u32,
    num_experts_per_tok: u32, // Top-K
    routed_scaling_factor: f32,
    norm_topk_prob: bool,
    leading_dense_block_count: u32, // 前导稠密层数量

    // 归一化
    rms_norm_eps: f32,

    // RoPE
    rope_freq_base: f32,
    rope_dimension_count: u32,

    // 派生参数
    head_dim: u32, // qk_rope_head_dim + qk_nope_head_dim

    pub fn fromGguf(gguf: *const GgufFile) !HyperParams {
        // 确认架构
        const arch = gguf.metaString("general.architecture") orelse return error.MissingArchitecture;
        if (!std.mem.eql(u8, arch, "deepseek2")) {
            return error.UnsupportedArchitecture;
        }

        const hidden_size: u32 = @intCast(gguf.metaInt("deepseek2.embedding_length") orelse return error.MissingParam);
        const intermediate_size: u32 = @intCast(gguf.metaInt("deepseek2.feed_forward_length") orelse return error.MissingParam);
        const moe_intermediate_size: u32 = @intCast(gguf.metaInt("deepseek2.expert_feed_forward_length") orelse intermediate_size);
        const num_hidden_layers: u32 = @intCast(gguf.metaInt("deepseek2.block_count") orelse return error.MissingParam);
        const num_attention_heads: u32 = @intCast(gguf.metaInt("deepseek2.attention.head_count") orelse return error.MissingParam);
        const num_key_value_heads: u32 = @intCast(gguf.metaInt("deepseek2.attention.head_count_kv") orelse num_attention_heads);
        const max_position_embeddings: u32 = @intCast(gguf.metaInt("deepseek2.context_length") orelse 2048);

        const kv_lora_rank: u32 = @intCast(gguf.metaInt("deepseek2.attention.kv_lora_rank") orelse return error.MissingParam);
        const q_lora_rank: u32 = @intCast(gguf.metaInt("deepseek2.attention.q_lora_rank") orelse return error.MissingParam);

        // ===== MLA 注意力维度（基于真实 GGUF 元数据语义） =====
        // key_length       = kv_lora_rank + qk_rope_head_dim  (kv_a_proj K 部分输出维)
        // value_length     = kv_lora_rank                     (kv_a_proj V 部分输出维)
        // key_length_mla   = head_dim = qk_nope + qk_rope     (每头 Q/K 总维)
        // value_length_mla = v_head_dim                       (每头 V 维)
        const key_length: u32 = @intCast(gguf.metaInt("deepseek2.attention.key_length") orelse 0);
        const value_length: u32 = @intCast(gguf.metaInt("deepseek2.attention.value_length") orelse 0);
        const key_length_mla: u32 = @intCast(gguf.metaInt("deepseek2.attention.key_length_mla") orelse 0);
        const value_length_mla: u32 = @intCast(gguf.metaInt("deepseek2.attention.value_length_mla") orelse 0);

        const rope_dimension_count: u32 = @intCast(gguf.metaInt("deepseek2.rope.dimension_count") orelse 64);

        // qk_rope_head_dim 直接从 rope.dimension_count 获取
        const qk_rope_head_dim = rope_dimension_count;

        // head_dim = key_length_mla (每头 Q/K 总维)
        const head_dim: u32 = if (key_length_mla > 0) key_length_mla else (qk_rope_head_dim + 128);

        // qk_nope_head_dim = head_dim - qk_rope_head_dim
        const qk_nope_head_dim = head_dim - qk_rope_head_dim;

        // v_head_dim = value_length_mla
        const v_head_dim: u32 = if (value_length_mla > 0) value_length_mla else 128;

        // kv_lora_rank 优先从元数据，回退到 key_length - qk_rope
        const final_kv_lora_rank: u32 = if (kv_lora_rank > 0)
            kv_lora_rank
        else if (key_length > qk_rope_head_dim)
            key_length - qk_rope_head_dim
        else
            value_length;

        std.log.info("MLA dims: head_dim={d}, qk_rope={d}, qk_nope={d}, v_head={d}, q_rank={d}, kv_rank={d}", .{
            head_dim, qk_rope_head_dim, qk_nope_head_dim, v_head_dim, q_lora_rank, final_kv_lora_rank,
        });

        const n_routed_experts: u32 = @intCast(gguf.metaInt("deepseek2.expert_count") orelse 1);
        const n_shared_experts: u32 = @intCast(gguf.metaInt("deepseek2.expert_shared_count") orelse 0);
        const num_experts_per_tok: u32 = @intCast(gguf.metaInt("deepseek2.expert_used_count") orelse 1);
        const routed_scaling_factor: f32 = @floatCast(gguf.metaFloat("deepseek2.expert_weights_scale") orelse 1.0);
        const norm_topk_prob: bool = true; // GLM-4.7-Flash 默认归一化
        const leading_dense_block_count: u32 = @intCast(gguf.metaInt("deepseek2.leading_dense_block_count") orelse 1);

        const rms_norm_eps: f32 = @floatCast(gguf.metaFloat("deepseek2.attention.layer_norm_rms_epsilon") orelse 1e-5);
        const rope_freq_base: f32 = @floatCast(gguf.metaFloat("deepseek2.rope.freq_base") orelse 1e6);

        // vocab_size 从 deepseek2.vocab_size 获取
        const vocab_size: u32 = @intCast(gguf.metaInt("deepseek2.vocab_size") orelse 154880);

        return .{
            .hidden_size = hidden_size,
            .intermediate_size = intermediate_size,
            .moe_intermediate_size = moe_intermediate_size,
            .num_hidden_layers = num_hidden_layers,
            .num_attention_heads = num_attention_heads,
            .num_key_value_heads = num_key_value_heads,
            .vocab_size = vocab_size,
            .max_position_embeddings = max_position_embeddings,
            .kv_lora_rank = final_kv_lora_rank,
            .q_lora_rank = q_lora_rank,
            .qk_rope_head_dim = qk_rope_head_dim,
            .qk_nope_head_dim = qk_nope_head_dim,
            .v_head_dim = v_head_dim,
            .n_routed_experts = n_routed_experts,
            .n_shared_experts = n_shared_experts,
            .num_experts_per_tok = num_experts_per_tok,
            .routed_scaling_factor = routed_scaling_factor,
            .norm_topk_prob = norm_topk_prob,
            .leading_dense_block_count = leading_dense_block_count,
            .rms_norm_eps = rms_norm_eps,
            .rope_freq_base = rope_freq_base,
            .rope_dimension_count = rope_dimension_count,
            .head_dim = head_dim,
        };
    }
};

/// 单层权重引用（指向 mmap 区域，零拷贝）
pub const LayerWeights = struct {
    // 注意力权重
    attn_norm: ?[*]const u8 = null,
    attn_norm_type: GgmlType = .f32,
    attn_norm_len: u64 = 0,

    // MLA 压缩投影
    kv_a_proj: ?[*]const u8 = null,
    kv_a_proj_type: GgmlType = .f32,
    kv_a_proj_len: u64 = 0,

    kv_a_norm: ?[*]const u8 = null,
    kv_a_norm_type: GgmlType = .f32,
    kv_a_norm_len: u64 = 0,

    k_b: ?[*]const u8 = null,
    k_b_type: GgmlType = .f32,
    k_b_len: u64 = 0,

    v_b: ?[*]const u8 = null,
    v_b_type: GgmlType = .f32,
    v_b_len: u64 = 0,

    q_a_proj: ?[*]const u8 = null,
    q_a_proj_type: GgmlType = .f32,
    q_a_proj_len: u64 = 0,

    q_a_norm: ?[*]const u8 = null,
    q_a_norm_type: GgmlType = .f32,
    q_a_norm_len: u64 = 0,

    q_b_proj: ?[*]const u8 = null,
    q_b_proj_type: GgmlType = .f32,
    q_b_proj_len: u64 = 0,

    // 输出投影
    o_proj: ?[*]const u8 = null,
    o_proj_type: GgmlType = .f32,
    o_proj_len: u64 = 0,

    // FFN / MoE
    ffn_norm: ?[*]const u8 = null,
    ffn_norm_type: GgmlType = .f32,
    ffn_norm_len: u64 = 0,

    // 稠密 FFN (前导层)
    dense_gate: ?[*]const u8 = null,
    dense_up: ?[*]const u8 = null,
    dense_down: ?[*]const u8 = null,
    dense_type: GgmlType = .f32,

    // 共享专家
    shared_gate: ?[*]const u8 = null,
    shared_up: ?[*]const u8 = null,
    shared_down: ?[*]const u8 = null,
    shared_type: GgmlType = .f32,

    // 路由专家 [n_routed_experts]
    expert_gate: ?[*]const u8 = null,
    expert_up: ?[*]const u8 = null,
    expert_down: ?[*]const u8 = null,
    expert_type: GgmlType = .f32,

    // MoE 门控
    moe_gate: ?[*]const u8 = null,
    moe_gate_type: GgmlType = .f32,
};

/// 模型权重集合
pub const ModelWeights = struct {
    hp: HyperParams,
    gguf: GgufFile,

    // 全局权重
    token_embd: ?[*]const u8 = null,
    token_embd_type: GgmlType = .f32,
    token_embd_len: u64 = 0,

    output_norm: ?[*]const u8 = null,
    output_norm_type: GgmlType = .f32,
    output_norm_len: u64 = 0,

    output_proj: ?[*]const u8 = null,
    output_proj_type: GgmlType = .f32,
    output_proj_len: u64 = 0,

    layers: []LayerWeights,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, gguf: GgufFile) !Self {
        const hp = try HyperParams.fromGguf(&gguf);

        var self = Self{
            .hp = hp,
            .gguf = gguf,
            .layers = try allocator.alloc(LayerWeights, hp.num_hidden_layers),
        };
        @memset(self.layers, LayerWeights{});

        // 绑定全局权重
        self.bindGlobal();
        // 绑定每层权重
        for (0..hp.num_hidden_layers) |i| {
            self.bindLayer(@intCast(i));
        }

        return self;
    }

    pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
        allocator.free(self.layers);
        self.gguf.deinit();
    }

    fn bindTensor(self: *Self, name: []const u8) ?struct { ptr: [*]const u8, ggml_type: GgmlType, len: u64 } {
        const ti = self.gguf.tensor_infos.get(name) orelse return null;
        const base = self.gguf.tensorDataPtr() orelse return null;
        return .{
            .ptr = base + ti.offset,
            .ggml_type = ti.ggml_type,
            .len = ti.nElements(),
        };
    }

    fn bindGlobal(self: *Self) void {
        if (self.bindTensor("token_embd.weight")) |t| {
            self.token_embd = t.ptr;
            self.token_embd_type = t.ggml_type;
            self.token_embd_len = t.len;
        }
        if (self.bindTensor("output_norm.weight")) |t| {
            self.output_norm = t.ptr;
            self.output_norm_type = t.ggml_type;
            self.output_norm_len = t.len;
        }
        if (self.bindTensor("output.weight")) |t| {
            self.output_proj = t.ptr;
            self.output_proj_type = t.ggml_type;
            self.output_proj_len = t.len;
        }
    }

    fn bindLayer(self: *Self, layer: u32) void {
        var buf: [128]u8 = undefined;
        const L = struct {
            fn fmt(out_buf: []u8, layer_idx: u32, suffix: []const u8) []const u8 {
                return std.fmt.bufPrint(out_buf, "blk.{d}.{s}", .{ layer_idx, suffix }) catch suffix;
            }
        };

        const lw = &self.layers[layer];

        if (self.bindTensor(L.fmt(&buf, layer, "attn_norm.weight"))) |t| {
            lw.attn_norm = t.ptr;
            lw.attn_norm_type = t.ggml_type;
            lw.attn_norm_len = t.len;
        }
        if (self.bindTensor(L.fmt(&buf, layer, "attn_kv_a_mqa.weight"))) |t| {
            lw.kv_a_proj = t.ptr;
            lw.kv_a_proj_type = t.ggml_type;
            lw.kv_a_proj_len = t.len;
        }
        if (self.bindTensor(L.fmt(&buf, layer, "attn_kv_a_norm.weight"))) |t| {
            lw.kv_a_norm = t.ptr;
            lw.kv_a_norm_type = t.ggml_type;
            lw.kv_a_norm_len = t.len;
        }
        if (self.bindTensor(L.fmt(&buf, layer, "attn_k_b.weight"))) |t| {
            lw.k_b = t.ptr;
            lw.k_b_type = t.ggml_type;
            lw.k_b_len = t.len;
        }
        if (self.bindTensor(L.fmt(&buf, layer, "attn_v_b.weight"))) |t| {
            lw.v_b = t.ptr;
            lw.v_b_type = t.ggml_type;
            lw.v_b_len = t.len;
        }
        if (self.bindTensor(L.fmt(&buf, layer, "attn_q_a.weight"))) |t| {
            lw.q_a_proj = t.ptr;
            lw.q_a_proj_type = t.ggml_type;
            lw.q_a_proj_len = t.len;
        }
        if (self.bindTensor(L.fmt(&buf, layer, "attn_q_a_norm.weight"))) |t| {
            lw.q_a_norm = t.ptr;
            lw.q_a_norm_type = t.ggml_type;
            lw.q_a_norm_len = t.len;
        }
        if (self.bindTensor(L.fmt(&buf, layer, "attn_q_b.weight"))) |t| {
            lw.q_b_proj = t.ptr;
            lw.q_b_proj_type = t.ggml_type;
            lw.q_b_proj_len = t.len;
        }
        if (self.bindTensor(L.fmt(&buf, layer, "attn_output.weight"))) |t| {
            lw.o_proj = t.ptr;
            lw.o_proj_type = t.ggml_type;
            lw.o_proj_len = t.len;
        }
        if (self.bindTensor(L.fmt(&buf, layer, "ffn_norm.weight"))) |t| {
            lw.ffn_norm = t.ptr;
            lw.ffn_norm_type = t.ggml_type;
            lw.ffn_norm_len = t.len;
        }

        // 稠密 FFN (前导层使用 ffn_gate/up/down)
        if (self.bindTensor(L.fmt(&buf, layer, "ffn_gate.weight"))) |t| {
            lw.dense_gate = t.ptr;
            lw.dense_type = t.ggml_type;
        }
        if (self.bindTensor(L.fmt(&buf, layer, "ffn_up.weight"))) |t| {
            lw.dense_up = t.ptr;
            lw.dense_type = t.ggml_type;
        }
        if (self.bindTensor(L.fmt(&buf, layer, "ffn_down.weight"))) |t| {
            lw.dense_down = t.ptr;
            lw.dense_type = t.ggml_type;
        }

        // 共享专家
        if (self.bindTensor(L.fmt(&buf, layer, "ffn_gate_shexp.weight"))) |t| {
            lw.shared_gate = t.ptr;
            lw.shared_type = t.ggml_type;
        }
        if (self.bindTensor(L.fmt(&buf, layer, "ffn_up_shexp.weight"))) |t| {
            lw.shared_up = t.ptr;
            lw.shared_type = t.ggml_type;
        }
        if (self.bindTensor(L.fmt(&buf, layer, "ffn_down_shexp.weight"))) |t| {
            lw.shared_down = t.ptr;
            lw.shared_type = t.ggml_type;
        }

        // 路由专家（连续存储）
        if (self.bindTensor(L.fmt(&buf, layer, "ffn_gate_exps.weight"))) |t| {
            lw.expert_gate = t.ptr;
            lw.expert_type = t.ggml_type;
        }
        if (self.bindTensor(L.fmt(&buf, layer, "ffn_up_exps.weight"))) |t| {
            lw.expert_up = t.ptr;
        }
        if (self.bindTensor(L.fmt(&buf, layer, "ffn_down_exps.weight"))) |t| {
            lw.expert_down = t.ptr;
        }

        // MoE 门控（路由器）
        if (self.bindTensor(L.fmt(&buf, layer, "ffn_gate_inp.weight"))) |t| {
            lw.moe_gate = t.ptr;
            lw.moe_gate_type = t.ggml_type;
        }
    }
};
