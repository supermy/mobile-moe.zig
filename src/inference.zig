// CPU 参考推理核心
//
// 实现 GLM-4.7-Flash (DeepSeek2 架构) 的完整前向传播：
//   1. Embedding 查找
//   2. 对每层：
//      a. RMSNorm (attn_norm)
//      b. MLA 注意力 (Q/K/V 压缩投影 + RoPE + FlashAttention + 输出投影)
//      c. 残差连接
//      d. RMSNorm (ffn_norm)
//      e. MoE FFN (门控路由 + 共享专家 + 路由专家)
//      f. 残差连接
//   3. RMSNorm (output_norm)
//   4. 输出投影 (lm_head)
//
// CPU 参考路径：正确性优先于速度
// GPU/NPU 路径必须与此路径的输出对齐验证

const std = @import("std");
const math = @import("math.zig");
const dequant = @import("dequant.zig");
const qmm = @import("qmm.zig");
const model = @import("model.zig");
const GgmlType = @import("gguf.zig").GgmlType;
const tokenizer = @import("tokenizer.zig");
const kv_cache = @import("kv_cache.zig");
const scheduler = @import("scheduler.zig");
const thermal = @import("thermal.zig");
const thread_pool = @import("thread_pool.zig");
const opencl_backend = @import("opencl_backend.zig");

pub const InferenceEngine = struct {
    allocator: std.mem.Allocator,
    weights: model.ModelWeights,
    tok: tokenizer.Tokenizer,
    kv: kv_cache.KVCache,
    sched: scheduler.Scheduler,
    throttle_state: thermal.ThrottleState,
    rope_cache: math.RoPECache,

    // 工作缓冲区
    hidden: []f32, // [hidden_size]
    q_comp: []f32, // [q_lora_rank] Q 压缩
    q_out: []f32, // [num_heads * head_dim] Q 展开后
    kv_comp: []f32, // [kv_lora_rank] KV 压缩
    kv_a_out: []f32, // [kv_lora_rank + qk_rope_head_dim] kv_a_proj 完整输出
    k_nope_out: []f32, // [num_heads * qk_nope_head_dim]
    v_out: []f32, // [num_heads * v_head_dim]
    attn_out: []f32, // [num_heads * v_head_dim] 注意力输出（拼接后）
    ffn_out: []f32, // [hidden_size] FFN 输出
    logits: []f32, // [vocab_size]

    // 批量推理缓冲区（Prompt Prefill 优化）
    max_batch_size: u32,
    batch_hidden: []f32, // [max_batch_size, hidden_size]
    batch_attn_out: []f32, // [max_batch_size, hidden_size]
    batch_ffn_out: []f32, // [max_batch_size, hidden_size]

    // 层权重缓存（Prompt Prefill 层优先优化）
    cached_layer: u32,
    cached_k_b_w: ?[]f32,
    cached_v_b_w: ?[]f32,

    // 残差缓冲区（避免每层堆分配）
    residual_buf: []f32,
    residual2_buf: []f32,

    // MLA 注意力预分配缓冲区（避免每次 attention 堆分配）
    cur_k_nope_buf: []f32, // [num_heads * qk_nope_head_dim]
    cur_v_buf: []f32, // [num_heads * v_head_dim]
    attn_o_vec_buf: []f32, // [v_head_dim]

    // KV 展开缓存（避免 attention 循环内重复 matVecMul）
    kv_nope_cache: []f32, // [max_seq_len, num_heads * qk_nope_head_dim]
    v_cache: []f32, // [max_seq_len, num_heads * v_head_dim]
    kv_cache_valid: []bool, // [max_seq_len]
    cached_layer_kv: u32,

    // 批量推理额外缓冲区（matMulQ8_0 批量 attention）
    batch_residual_buf: []f32, // [max_batch_size, hidden_size]
    batch_residual2_buf: []f32, // [max_batch_size, hidden_size]
    batch_q_comp: []f32, // [max_batch_size, q_lora_rank]
    batch_q_out: []f32, // [max_batch_size, num_heads * head_dim]
    batch_kv_a_out: []f32, // [max_batch_size, kv_lora_rank + qk_rope_head_dim]
    batch_attn_vec: []f32, // [max_batch_size, num_heads * v_head_dim]
    batch_proj_tmp: []f32, // [max_proj_dim, max_batch_size]

    // 线程池（gate/up 并行 + 专家并行）
    pool: thread_pool.ThreadPool,

    // OpenCL GPU 后端（Android/Adreno MVP）
    opencl: ?opencl_backend.OpenCLBackend,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, weights: model.ModelWeights, tok: tokenizer.Tokenizer, max_seq_len: u32) !Self {
        const hp = weights.hp;

        const kv = try kv_cache.KVCache.init(allocator, hp, max_seq_len);
        const rope_cache = try math.RoPECache.init(allocator, max_seq_len, hp.qk_rope_head_dim, hp.rope_freq_base);

        var throttle_state = thermal.ThrottleState.init();
        const sched = try scheduler.Scheduler.init(allocator, hp.num_hidden_layers, hp.n_routed_experts, &throttle_state, hp.routed_scaling_factor, hp.hidden_size);

        const hidden = try allocator.alloc(f32, hp.hidden_size);
        const q_comp = try allocator.alloc(f32, hp.q_lora_rank);
        const q_out = try allocator.alloc(f32, hp.num_attention_heads * hp.head_dim);
        const kv_comp_buf = try allocator.alloc(f32, hp.kv_lora_rank);
        const kv_a_out = try allocator.alloc(f32, hp.kv_lora_rank + hp.qk_rope_head_dim);
        const k_nope_out = try allocator.alloc(f32, hp.num_attention_heads * hp.qk_nope_head_dim);
        const v_out = try allocator.alloc(f32, hp.num_attention_heads * hp.v_head_dim);
        const attn_out = try allocator.alloc(f32, hp.num_attention_heads * hp.v_head_dim);
        const ffn_out = try allocator.alloc(f32, hp.hidden_size);
        const logits = try allocator.alloc(f32, hp.vocab_size);

        const max_batch_size: u32 = 256; // Prompt Prefill 最大 batch
        const batch_hidden = try allocator.alloc(f32, max_batch_size * hp.hidden_size);
        const batch_attn_out = try allocator.alloc(f32, max_batch_size * hp.hidden_size);
        const batch_ffn_out = try allocator.alloc(f32, max_batch_size * hp.hidden_size);

        const batch_residual_buf = try allocator.alloc(f32, max_batch_size * hp.hidden_size);
        const batch_residual2_buf = try allocator.alloc(f32, max_batch_size * hp.hidden_size);
        const batch_q_comp = try allocator.alloc(f32, max_batch_size * hp.q_lora_rank);
        const batch_q_out = try allocator.alloc(f32, max_batch_size * hp.num_attention_heads * hp.head_dim);
        const batch_kv_a_out = try allocator.alloc(f32, max_batch_size * (hp.kv_lora_rank + hp.qk_rope_head_dim));
        const batch_attn_vec = try allocator.alloc(f32, max_batch_size * hp.num_attention_heads * hp.v_head_dim);
        var max_proj_dim = hp.hidden_size;
        max_proj_dim = @max(max_proj_dim, hp.num_attention_heads * hp.head_dim);
        max_proj_dim = @max(max_proj_dim, hp.q_lora_rank);
        max_proj_dim = @max(max_proj_dim, hp.kv_lora_rank + hp.qk_rope_head_dim);
        max_proj_dim = @max(max_proj_dim, hp.num_attention_heads * hp.v_head_dim);
        max_proj_dim = @max(max_proj_dim, hp.n_routed_experts);
        max_proj_dim = @max(max_proj_dim, hp.intermediate_size);
        max_proj_dim = @max(max_proj_dim, hp.moe_intermediate_size);
        const batch_proj_tmp = try allocator.alloc(f32, max_batch_size * max_proj_dim);

        const residual_buf = try allocator.alloc(f32, hp.hidden_size);
        const residual2_buf = try allocator.alloc(f32, hp.hidden_size);

        const cur_k_nope_buf = try allocator.alloc(f32, hp.num_attention_heads * hp.qk_nope_head_dim);
        const cur_v_buf = try allocator.alloc(f32, hp.num_attention_heads * hp.v_head_dim);
        const attn_o_vec_buf = try allocator.alloc(f32, hp.v_head_dim);

        const kv_nope_cache = try allocator.alloc(f32, max_seq_len * hp.num_attention_heads * hp.qk_nope_head_dim);
        const v_cache = try allocator.alloc(f32, max_seq_len * hp.num_attention_heads * hp.v_head_dim);
        const kv_cache_valid = try allocator.alloc(bool, max_seq_len);
        @memset(kv_cache_valid, false);

        var pool: thread_pool.ThreadPool = undefined;
        try thread_pool.ThreadPool.init(&pool, allocator, 4);
        errdefer pool.deinit();

        // 尝试初始化 OpenCL 后端（Adreno GPU），失败则优雅回退
        var opencl: ?opencl_backend.OpenCLBackend = null;
        if (opencl_backend.OpenCLBackend.isAvailable()) {
            opencl = opencl_backend.OpenCLBackend.init(allocator) catch null;
        }
        errdefer if (opencl) |*cl| cl.deinit();

        return .{
            .allocator = allocator,
            .weights = weights,
            .tok = tok,
            .kv = kv,
            .sched = sched,
            .throttle_state = throttle_state,
            .rope_cache = rope_cache,
            .hidden = hidden,
            .q_comp = q_comp,
            .q_out = q_out,
            .kv_comp = kv_comp_buf,
            .kv_a_out = kv_a_out,
            .k_nope_out = k_nope_out,
            .v_out = v_out,
            .attn_out = attn_out,
            .ffn_out = ffn_out,
            .logits = logits,
            .max_batch_size = max_batch_size,
            .batch_hidden = batch_hidden,
            .batch_attn_out = batch_attn_out,
            .batch_ffn_out = batch_ffn_out,
            .batch_residual_buf = batch_residual_buf,
            .batch_residual2_buf = batch_residual2_buf,
            .batch_q_comp = batch_q_comp,
            .batch_q_out = batch_q_out,
            .batch_kv_a_out = batch_kv_a_out,
            .batch_attn_vec = batch_attn_vec,
            .batch_proj_tmp = batch_proj_tmp,
            .cached_layer = std.math.maxInt(u32),
            .cached_k_b_w = null,
            .cached_v_b_w = null,
            .residual_buf = residual_buf,
            .residual2_buf = residual2_buf,
            .cur_k_nope_buf = cur_k_nope_buf,
            .cur_v_buf = cur_v_buf,
            .attn_o_vec_buf = attn_o_vec_buf,
            .kv_nope_cache = kv_nope_cache,
            .v_cache = v_cache,
            .kv_cache_valid = kv_cache_valid,
            .cached_layer_kv = std.math.maxInt(u32),
            .pool = pool,
            .opencl = opencl,
        };
    }

    pub fn deinit(self: *Self) void {
        const allocator = self.allocator;
        allocator.free(self.hidden);
        allocator.free(self.q_comp);
        allocator.free(self.q_out);
        allocator.free(self.kv_comp);
        allocator.free(self.kv_a_out);
        allocator.free(self.k_nope_out);
        allocator.free(self.v_out);
        allocator.free(self.attn_out);
        allocator.free(self.ffn_out);
        allocator.free(self.logits);
        allocator.free(self.batch_hidden);
        allocator.free(self.batch_attn_out);
        allocator.free(self.batch_ffn_out);
        allocator.free(self.batch_residual_buf);
        allocator.free(self.batch_residual2_buf);
        allocator.free(self.batch_q_comp);
        allocator.free(self.batch_q_out);
        allocator.free(self.batch_kv_a_out);
        allocator.free(self.batch_attn_vec);
        allocator.free(self.batch_proj_tmp);
        allocator.free(self.residual_buf);
        allocator.free(self.residual2_buf);
        allocator.free(self.cur_k_nope_buf);
        allocator.free(self.cur_v_buf);
        allocator.free(self.attn_o_vec_buf);
        allocator.free(self.kv_nope_cache);
        allocator.free(self.v_cache);
        allocator.free(self.kv_cache_valid);
        if (self.cached_k_b_w) |w| allocator.free(w);
        if (self.cached_v_b_w) |w| allocator.free(w);
        self.pool.deinit();
        if (self.opencl) |*cl| cl.deinit();
        self.rope_cache.deinit(allocator);
        self.kv.deinit();
        self.sched.deinit();
        self.tok.deinit();
        self.weights.deinit(allocator);
    }

    /// 前向传播：处理一个 token，返回 logits
    pub fn forward(self: *Self, token_id: u32, pos: u32) ![]f32 {
        const hp = self.weights.hp;

        // 重置层缓存，防止 forwardBatch 遗留状态污染单 token 推理
        self.cached_layer = std.math.maxInt(u32);

        // 热节流检查（Android 上读取温度）
        self.checkThermal();

        // 1. Embedding 查找
        try self.embedToken(token_id);

        // 2. 逐层前向
        for (0..hp.num_hidden_layers) |layer| {
            try self.forwardLayer(@intCast(layer), pos);
        }

        // 3. 输出 RMSNorm
        if (self.weights.output_norm) |norm_ptr| {
            const norm = try dequant.dequantize(self.allocator, norm_ptr, self.weights.output_norm_type, hp.hidden_size);
            defer self.allocator.free(norm);
            math.rmsNorm(self.hidden, self.hidden, norm, hp.rms_norm_eps);
        }

        // 4. 输出投影 (lm_head)
        try self.computeLogits();

        return self.logits;
    }

    /// 批量前向传播：处理多个 token（Prompt Prefill 层优先优化）
    /// 层优先：逐层处理所有 token，每层权重（k_b/v_b）只反量化一次
    pub fn forwardBatch(self: *Self, token_ids: []const u32, start_pos: u32) ![]f32 {
        if (token_ids.len == 0) return error.EmptyBatch;
        if (token_ids.len > self.max_batch_size) return error.BatchTooLarge;
        // 边界检查：确保 start_pos + batch_size 不超过 KV cache 容量
        const kv_capacity = @as(u32, @intCast(self.kv_cache_valid.len));
        if (start_pos + token_ids.len > kv_capacity) return error.SeqLenExceeded;
        if (token_ids.len == 1) {
            return self.forward(token_ids[0], start_pos);
        }

        // 重置层缓存，确保 Prompt Prefill 层优先优化正确工作
        self.cached_layer = std.math.maxInt(u32);

        const hp = self.weights.hp;

        // 1. Embedding 查找所有 token 到 batch_hidden
        for (token_ids, 0..) |tid, i| {
            try self.embedToken(tid);
            @memcpy(self.batch_hidden[i * hp.hidden_size ..][0..hp.hidden_size], self.hidden);
        }

        // 2. 层优先：外层循环 layer，内层循环 token
        for (0..hp.num_hidden_layers) |layer| {
            const lw = self.weights.layers[layer];

            // 2a. 保存残差 + RMSNorm (attn_norm)
            if (lw.attn_norm) |norm_ptr| {
                const norm = try dequant.dequantize(self.allocator, norm_ptr, lw.attn_norm_type, hp.hidden_size);
                defer self.allocator.free(norm);
                for (0..token_ids.len) |bi| {
                    const hidden = self.batch_hidden[bi * hp.hidden_size ..][0..hp.hidden_size];
                    @memcpy(self.batch_residual_buf[bi * hp.hidden_size ..][0..hp.hidden_size], hidden);
                    math.rmsNorm(hidden, hidden, norm, hp.rms_norm_eps);
                }
            } else {
                for (0..token_ids.len) |bi| {
                    const hidden = self.batch_hidden[bi * hp.hidden_size ..][0..hp.hidden_size];
                    @memcpy(self.batch_residual_buf[bi * hp.hidden_size ..][0..hp.hidden_size], hidden);
                }
            }

            // 2b. 批量 MLA attention（使用 matMulQ8_0 做批量投影）
            try self.mlaAttentionBatch(@intCast(layer), start_pos, @intCast(token_ids.len));

            // 2c. 残差连接
            for (0..token_ids.len) |bi| {
                const hidden = self.batch_hidden[bi * hp.hidden_size ..][0..hp.hidden_size];
                const residual = self.batch_residual_buf[bi * hp.hidden_size ..][0..hp.hidden_size];
                for (hidden, residual) |*h, r| h.* += r;
            }

            // 2d. 保存残差2 + RMSNorm (ffn_norm)
            if (lw.ffn_norm) |norm_ptr| {
                const norm = try dequant.dequantize(self.allocator, norm_ptr, lw.ffn_norm_type, hp.hidden_size);
                defer self.allocator.free(norm);
                for (0..token_ids.len) |bi| {
                    const hidden = self.batch_hidden[bi * hp.hidden_size ..][0..hp.hidden_size];
                    @memcpy(self.batch_residual2_buf[bi * hp.hidden_size ..][0..hp.hidden_size], hidden);
                    math.rmsNorm(hidden, hidden, norm, hp.rms_norm_eps);
                }
            } else {
                for (0..token_ids.len) |bi| {
                    const hidden = self.batch_hidden[bi * hp.hidden_size ..][0..hp.hidden_size];
                    @memcpy(self.batch_residual2_buf[bi * hp.hidden_size ..][0..hp.hidden_size], hidden);
                }
            }

            // 2e. 逐 token MoE FFN（路由不同，难以 batch）
            for (0..token_ids.len) |bi| {
                @memcpy(self.hidden, self.batch_hidden[bi * hp.hidden_size ..][0..hp.hidden_size]);
                try self.moeFFN(@intCast(layer));
                @memcpy(self.batch_hidden[bi * hp.hidden_size ..][0..hp.hidden_size], self.hidden);
            }

            // 2f. 残差连接
            for (0..token_ids.len) |bi| {
                const hidden = self.batch_hidden[bi * hp.hidden_size ..][0..hp.hidden_size];
                const residual2 = self.batch_residual2_buf[bi * hp.hidden_size ..][0..hp.hidden_size];
                for (hidden, residual2) |*h, r| h.* += r;
            }
        }

        // 3. 输出 RMSNorm（仅最后一个 token）
        @memcpy(self.hidden, self.batch_hidden[(token_ids.len - 1) * hp.hidden_size ..][0..hp.hidden_size]);
        if (self.weights.output_norm) |norm_ptr| {
            const norm = try dequant.dequantize(self.allocator, norm_ptr, self.weights.output_norm_type, hp.hidden_size);
            defer self.allocator.free(norm);
            math.rmsNorm(self.hidden, self.hidden, norm, hp.rms_norm_eps);
        }

        // 4. 输出投影 (lm_head)
        try self.computeLogits();

        return self.logits;
    }

    /// 采样下一个 token（贪心）
    pub fn sampleGreedy(_: *Self, logits: []const f32) u32 {
        var best_id: u32 = 0;
        var best_score: f32 = logits[0];
        for (logits[1..], 1..) |s, i| {
            if (s > best_score) {
                best_score = s;
                best_id = @intCast(i);
            }
        }
        return best_id;
    }

    fn embedToken(self: *Self, token_id: u32) !void {
        const hp = self.weights.hp;
        if (token_id >= hp.vocab_size) return error.TokenIdOutOfBounds;
        if (self.weights.token_embd) |embd_ptr| {
            // 从 embedding 矩阵中取出对应行
            const row_bytes = self.weights.token_embd_type.tensorBytes(hp.hidden_size);
            const embd = try dequant.dequantize(
                self.allocator,
                embd_ptr + token_id * row_bytes,
                self.weights.token_embd_type,
                hp.hidden_size,
            );
            defer self.allocator.free(embd);
            std.debug.assert(embd.len == hp.hidden_size);
            @memcpy(self.hidden, embd);
        }
    }

    fn forwardLayer(self: *Self, layer: u32, pos: u32) !void {
        const hp = self.weights.hp;
        const lw = self.weights.layers[layer];

        // 保存残差到预分配缓冲区
        @memcpy(self.residual_buf, self.hidden);

        // a. RMSNorm (attn_norm)
        if (lw.attn_norm) |norm_ptr| {
            const norm = try dequant.dequantize(self.allocator, norm_ptr, lw.attn_norm_type, hp.hidden_size);
            defer self.allocator.free(norm);
            math.rmsNorm(self.hidden, self.hidden, norm, hp.rms_norm_eps);
        }

        // b. MLA 注意力
        try self.mlaAttention(layer, pos);

        // c. 残差连接
        for (self.hidden, self.residual_buf) |*h, r| {
            h.* += r;
        }

        // 保存注意力后的残差到预分配缓冲区
        @memcpy(self.residual2_buf, self.hidden);

        // d. RMSNorm (ffn_norm)
        if (lw.ffn_norm) |norm_ptr| {
            const norm = try dequant.dequantize(self.allocator, norm_ptr, lw.ffn_norm_type, hp.hidden_size);
            defer self.allocator.free(norm);
            math.rmsNorm(self.hidden, self.hidden, norm, hp.rms_norm_eps);
        }

        // e. MoE FFN
        try self.moeFFN(layer);

        // f. 残差连接
        for (self.hidden, self.residual2_buf) |*h, r| {
            h.* += r;
        }
    }

    /// MLA 注意力前向
    /// DeepSeek2 MLA 流程：
    ///   1. Q: hidden → q_a_proj → q_comp → RMSNorm(q_a_norm) → q_b_proj → q_out
    ///   2. KV: hidden → kv_a_proj → kv_comp → RMSNorm(kv_a_norm) → 存入缓存
    ///      k_pe 从 kv_a_proj 输出的最后 qk_rope_head_dim 维提取
    ///   3. RoPE: 应用于 Q 每头的最后 qk_rope_head_dim 维
    ///   4. 注意力: 对每个头，kv_b_proj 展开 kv_comp 得到 K_nope 和 V，
    ///      K_rope 从缓存取 k_pe，然后标准缩放点积注意力
    ///   5. 输出投影: attn_out → o_proj → hidden
    fn mlaAttention(self: *Self, layer: u32, pos: u32) !void {
        const hp = self.weights.hp;
        const lw = self.weights.layers[layer];
        const n_heads = hp.num_attention_heads;
        const head_dim = hp.head_dim;
        const qk_nope = hp.qk_nope_head_dim;
        const qk_rope = hp.qk_rope_head_dim;
        const v_dim = hp.v_head_dim;
        const kv_rank = hp.kv_lora_rank;

        // ===== 1. Q 压缩投影: hidden[hidden_size] → q_comp[q_lora_rank] =====
        if (lw.q_a_proj) |w_ptr| {
            try self.matVecMulUnified(self.q_comp, w_ptr, lw.q_a_proj_type, self.hidden, hp.q_lora_rank, hp.hidden_size);
        }

        // Q RMSNorm
        if (lw.q_a_norm) |norm_ptr| {
            const norm = try dequant.dequantize(self.allocator, norm_ptr, lw.q_a_norm_type, hp.q_lora_rank);
            defer self.allocator.free(norm);
            math.rmsNorm(self.q_comp, self.q_comp, norm, hp.rms_norm_eps);
        }

        // Q 展开投影: q_comp[q_lora_rank] → q_out[num_heads * head_dim]
        if (lw.q_b_proj) |w_ptr| {
            try self.matVecMulUnified(self.q_out, w_ptr, lw.q_b_proj_type, self.q_comp, n_heads * head_dim, hp.q_lora_rank);
        }

        // ===== 2. KV 压缩投影: hidden[hidden_size] → kv_a_out[kv_lora_rank + qk_rope_head_dim] =====
        // DeepSeek2 中 kv_a_proj 输出 kv_lora_rank + qk_rope_head_dim 维
        // 前 kv_lora_rank 维为压缩 KV，后 qk_rope_head_dim 维为 k_pe
        const kv_a_dim = kv_rank + qk_rope;
        if (lw.kv_a_proj) |w_ptr| {
            try self.matVecMulUnified(self.kv_a_out, w_ptr, lw.kv_a_proj_type, self.hidden, kv_a_dim, hp.hidden_size);
        }

        // 分离 kv_comp 和 k_pe
        @memcpy(self.kv_comp, self.kv_a_out[0..kv_rank]);
        const k_pe = self.kv_a_out[kv_rank..][0..qk_rope];

        // KV RMSNorm（仅对 kv_comp 部分，k_pe 不做归一化）
        if (lw.kv_a_norm) |norm_ptr| {
            const norm = try dequant.dequantize(self.allocator, norm_ptr, lw.kv_a_norm_type, kv_rank);
            defer self.allocator.free(norm);
            math.rmsNorm(self.kv_comp, self.kv_comp, norm, hp.rms_norm_eps);
        }

        // ===== 3. 对 Q 和 k_pe 应用 RoPE =====

        // 对 Q 的每头应用 RoPE（最后 qk_rope_head_dim 维）
        for (0..n_heads) |h| {
            const q_head = self.q_out[h * head_dim ..][0..head_dim];
            // RoPE 应用于每头的最后 qk_rope_head_dim 维
            const q_rope = q_head[qk_nope..][0..qk_rope];
            self.rope_cache.apply(q_rope, pos);
        }

        // 对 k_pe 应用 RoPE
        self.rope_cache.apply(k_pe, pos);

        // ===== 4. 写入 KV 缓存 =====
        try self.kv.put(layer, pos, self.kv_comp, k_pe);

        // ===== 5. 注意力计算（FlashAttention 分块）=====
        @memset(self.attn_out, 0);

        const k_b_dim = n_heads * qk_nope;
        const v_b_dim = n_heads * v_dim;
        // 层权重缓存：k_b / v_b 每层只反量化一次
        if (self.cached_layer != layer or self.cached_k_b_w == null) {
            if (self.cached_k_b_w) |w| self.allocator.free(w);
            if (self.cached_v_b_w) |w| self.allocator.free(w);
            self.cached_k_b_w = null;
            self.cached_v_b_w = null;

            if (lw.k_b) |w_ptr| {
                self.cached_k_b_w = try dequant.dequantize(self.allocator, w_ptr, lw.k_b_type, k_b_dim * kv_rank);
            }
            if (lw.v_b) |w_ptr| {
                self.cached_v_b_w = try dequant.dequantize(self.allocator, w_ptr, lw.v_b_type, v_b_dim * kv_rank);
            }
            self.cached_layer = layer;
        }

        if (self.cached_k_b_w) |w| {
            math.matVecMul(self.k_nope_out, w, self.kv_comp, k_b_dim, kv_rank);
        }
        if (self.cached_v_b_w) |w| {
            math.matVecMul(self.v_out, w, self.kv_comp, v_b_dim, kv_rank);
        }

        // 使用预分配缓冲区，避免每次 attention 堆分配
        const cur_k_nope_all = self.cur_k_nope_buf;
        const cur_v_all = self.cur_v_buf;
        @memcpy(cur_k_nope_all, self.k_nope_out);
        @memcpy(cur_v_all, self.v_out);

        // ===== 5.1 KV 展开缓存：预展开所有缺失的历史位置 =====
        if (self.cached_layer_kv != layer) {
            @memset(self.kv_cache_valid, false);
            self.cached_layer_kv = layer;
        }
        for (0..pos) |p| {
            if (!self.kv_cache_valid[p]) {
                const kv_cached = try self.kv.getKVCompressed(layer, @intCast(p));
                if (self.cached_k_b_w) |w| {
                    math.matVecMul(self.kv_nope_cache[p * k_b_dim ..], w, kv_cached, k_b_dim, kv_rank);
                }
                if (self.cached_v_b_w) |w| {
                    math.matVecMul(self.v_cache[p * v_b_dim ..], w, kv_cached, v_b_dim, kv_rank);
                }
                self.kv_cache_valid[p] = true;
            }
        }

        // ===== 5.2 注意力计算（FlashAttention 分块）=====
        // 注：self.attn_out 已在 5. 开头清零，此处无需重复
        const BLOCK_SIZE: usize = 64;
        const o_vec = self.attn_o_vec_buf;

        for (0..n_heads) |h| {
            const q_head = self.q_out[h * head_dim ..][0..head_dim];
            const q_nope = q_head[0..qk_nope];
            const q_rope = q_head[qk_nope..][0..qk_rope];
            const cur_k_nope = cur_k_nope_all[h * qk_nope ..][0..qk_nope];
            const cur_v = cur_v_all[h * v_dim ..][0..v_dim];
            const head_out: []f32 = self.attn_out[h * v_dim ..][0..v_dim];

            var m_vec: f32 = -std.math.inf(f32);
            var l_vec: f32 = 0;
            @memset(o_vec, 0);

            const num_blocks = (pos + 1 + BLOCK_SIZE - 1) / BLOCK_SIZE;
            for (0..num_blocks) |bj| {
                const b_start = bj * BLOCK_SIZE;
                const b_end = @min(b_start + BLOCK_SIZE, pos + 1);
                const b_len = b_end - b_start;

                var scores: [BLOCK_SIZE]f32 = @splat(0);
                for (b_start..b_end) |p| {
                    var score: f32 = 0;

                    if (p == pos) {
                        for (q_nope, cur_k_nope) |qv, kv| {
                            score += qv * kv;
                        }
                    } else {
                        const hist_k_nope = self.kv_nope_cache[p * k_b_dim ..][h * qk_nope ..][0..qk_nope];
                        for (q_nope, hist_k_nope) |qv, kv| {
                            score += qv * kv;
                        }
                    }

                    const k_pe_cached = try self.kv.getKPe(layer, @intCast(p));
                    for (q_rope, k_pe_cached) |qv, kv| {
                        score += qv * kv;
                    }

                    scores[p - b_start] = score / @sqrt(@as(f32, @floatFromInt(head_dim)));
                }

                var m_j: f32 = -std.math.inf(f32);
                for (0..b_len) |i| {
                    if (scores[i] > m_j) m_j = scores[i];
                }
                var l_j: f32 = 0;
                for (0..b_len) |i| {
                    scores[i] = @exp(scores[i] - m_j);
                    l_j += scores[i];
                }

                const m_new = @max(m_vec, m_j);
                const scale_o = @exp(m_vec - m_new);
                const scale_s = @exp(m_j - m_new);

                for (0..v_dim) |i| {
                    o_vec[i] = o_vec[i] * scale_o;
                }

                for (b_start..b_end) |p| {
                    const weight = scores[p - b_start] * scale_s;

                    if (p == pos) {
                        for (0..v_dim) |i| {
                            o_vec[i] += weight * cur_v[i];
                        }
                    } else {
                        const hist_v = self.v_cache[p * v_b_dim ..][h * v_dim ..][0..v_dim];
                        for (0..v_dim) |i| {
                            o_vec[i] += weight * hist_v[i];
                        }
                    }
                }

                m_vec = m_new;
                l_vec = l_vec * scale_o + l_j * scale_s;
            }

            for (0..v_dim) |i| {
                if (l_vec > 0) {
                    head_out[i] = o_vec[i] / l_vec;
                } else {
                    head_out[i] = 0;
                }
            }
        }

        // ===== 6. 输出投影: attn_out[num_heads * v_head_dim] → hidden[hidden_size] =====
        if (lw.o_proj) |w_ptr| {
            try self.matVecMulUnified(self.hidden, w_ptr, lw.o_proj_type, self.attn_out, hp.hidden_size, n_heads * v_dim);
        }
    }

    /// 批量 MLA 注意力前向
    /// 使用 matMulQ8_0 对 Q/KV/O 投影做批量矩阵乘法
    fn mlaAttentionBatch(self: *Self, layer: u32, start_pos: u32, batch_size: usize) !void {
        const hp = self.weights.hp;
        const lw = self.weights.layers[layer];
        const n_heads = hp.num_attention_heads;
        const head_dim = hp.head_dim;
        const qk_nope = hp.qk_nope_head_dim;
        const qk_rope = hp.qk_rope_head_dim;
        const v_dim = hp.v_head_dim;
        const kv_rank = hp.kv_lora_rank;
        const k_b_dim = n_heads * qk_nope;
        const v_b_dim = n_heads * v_dim;
        const kv_a_dim = kv_rank + qk_rope;

        // ===== 1. Batch Q 压缩投影 =====
        if (lw.q_a_proj) |w_ptr| {
            try self.batchMatMulUnified(self.batch_q_comp, w_ptr, lw.q_a_proj_type, self.batch_hidden, hp.q_lora_rank, batch_size, hp.hidden_size);
        }

        // ===== 2. Q RMSNorm（每层反量化一次） =====
        if (lw.q_a_norm) |norm_ptr| {
            const norm = try dequant.dequantize(self.allocator, norm_ptr, lw.q_a_norm_type, hp.q_lora_rank);
            defer self.allocator.free(norm);
            for (0..batch_size) |bi| {
                const q_comp = self.batch_q_comp[bi * hp.q_lora_rank ..][0..hp.q_lora_rank];
                math.rmsNorm(q_comp, q_comp, norm, hp.rms_norm_eps);
            }
        }

        // ===== 3. Batch Q 展开投影 =====
        if (lw.q_b_proj) |w_ptr| {
            try self.batchMatMulUnified(self.batch_q_out, w_ptr, lw.q_b_proj_type, self.batch_q_comp, n_heads * head_dim, batch_size, hp.q_lora_rank);
        }

        // ===== 4. Batch KV 压缩投影 =====
        if (lw.kv_a_proj) |w_ptr| {
            try self.batchMatMulUnified(self.batch_kv_a_out, w_ptr, lw.kv_a_proj_type, self.batch_hidden, kv_a_dim, batch_size, hp.hidden_size);
        }

        // ===== 5. 逐 token: split, norm, RoPE, write cache =====
        for (0..batch_size) |bi| {
            const pos = start_pos + @as(u32, @intCast(bi));
            const kv_a_out = self.batch_kv_a_out[bi * kv_a_dim ..][0..kv_a_dim];
            const kv_comp = kv_a_out[0..kv_rank];
            const k_pe = kv_a_out[kv_rank..][0..qk_rope];

            if (lw.kv_a_norm) |norm_ptr| {
                const norm = try dequant.dequantize(self.allocator, norm_ptr, lw.kv_a_norm_type, kv_rank);
                defer self.allocator.free(norm);
                math.rmsNorm(kv_comp, kv_comp, norm, hp.rms_norm_eps);
            }

            // RoPE on Q
            const q_out = self.batch_q_out[bi * n_heads * head_dim ..][0 .. n_heads * head_dim];
            for (0..n_heads) |h| {
                const q_head = q_out[h * head_dim ..][0..head_dim];
                const q_rope = q_head[qk_nope..][0..qk_rope];
                self.rope_cache.apply(q_rope, pos);
            }

            // RoPE on k_pe
            self.rope_cache.apply(k_pe, pos);

            // Write KV cache
            try self.kv.put(layer, pos, kv_comp, k_pe);
        }

        // ===== 6. KV 层权重缓存 =====
        if (self.cached_layer != layer or self.cached_k_b_w == null) {
            if (self.cached_k_b_w) |w| self.allocator.free(w);
            if (self.cached_v_b_w) |w| self.allocator.free(w);
            self.cached_k_b_w = null;
            self.cached_v_b_w = null;

            if (lw.k_b) |w_ptr| {
                self.cached_k_b_w = try dequant.dequantize(self.allocator, w_ptr, lw.k_b_type, k_b_dim * kv_rank);
            }
            if (lw.v_b) |w_ptr| {
                self.cached_v_b_w = try dequant.dequantize(self.allocator, w_ptr, lw.v_b_type, v_b_dim * kv_rank);
            }
            self.cached_layer = layer;
        }

        // ===== 7. 预展开 KV cache 到 max_pos =====
        const max_pos = start_pos + @as(u32, @intCast(batch_size)) - 1;
        if (self.cached_layer_kv != layer) {
            @memset(self.kv_cache_valid, false);
            self.cached_layer_kv = layer;
        }
        for (0..max_pos + 1) |p| {
            if (!self.kv_cache_valid[p]) {
                const kv_cached = try self.kv.getKVCompressed(layer, @intCast(p));
                if (self.cached_k_b_w) |w| {
                    math.matVecMul(self.kv_nope_cache[p * k_b_dim ..], w, kv_cached, k_b_dim, kv_rank);
                }
                if (self.cached_v_b_w) |w| {
                    math.matVecMul(self.v_cache[p * v_b_dim ..], w, kv_cached, v_b_dim, kv_rank);
                }
                self.kv_cache_valid[p] = true;
            }
        }

        // ===== 8. 逐 token 注意力计算（FlashAttention 分块）=====
        const BLOCK_SIZE: usize = 64;
        const o_vec = self.attn_o_vec_buf;

        for (0..batch_size) |bi| {
            const pos = start_pos + @as(u32, @intCast(bi));
            const q_out = self.batch_q_out[bi * n_heads * head_dim ..][0 .. n_heads * head_dim];
            const attn_vec = self.batch_attn_vec[bi * n_heads * v_dim ..][0 .. n_heads * v_dim];
            @memset(attn_vec, 0);

            // 展开当前 token 的 KV
            const kv_a_out = self.batch_kv_a_out[bi * kv_a_dim ..][0..kv_a_dim];
            const kv_comp = kv_a_out[0..kv_rank];
            if (self.cached_k_b_w) |w| {
                math.matVecMul(self.cur_k_nope_buf, w, kv_comp, k_b_dim, kv_rank);
            }
            if (self.cached_v_b_w) |w| {
                math.matVecMul(self.cur_v_buf, w, kv_comp, v_b_dim, kv_rank);
            }

            for (0..n_heads) |h| {
                const q_head = q_out[h * head_dim ..][0..head_dim];
                const q_nope = q_head[0..qk_nope];
                const q_rope = q_head[qk_nope..][0..qk_rope];
                const cur_k_nope = self.cur_k_nope_buf[h * qk_nope ..][0..qk_nope];
                const cur_v = self.cur_v_buf[h * v_dim ..][0..v_dim];
                const head_out = attn_vec[h * v_dim ..][0..v_dim];

                var m_vec: f32 = -std.math.inf(f32);
                var l_vec: f32 = 0;
                @memset(o_vec, 0);

                const num_blocks = (pos + 1 + BLOCK_SIZE - 1) / BLOCK_SIZE;
                for (0..num_blocks) |bj| {
                    const b_start = bj * BLOCK_SIZE;
                    const b_end = @min(b_start + BLOCK_SIZE, pos + 1);
                    const b_len = b_end - b_start;

                    var scores: [BLOCK_SIZE]f32 = @splat(0);
                    for (b_start..b_end) |p| {
                        var score: f32 = 0;

                        if (p == pos) {
                            for (q_nope, cur_k_nope) |qv, kv| {
                                score += qv * kv;
                            }
                        } else {
                            const hist_k_nope = self.kv_nope_cache[p * k_b_dim ..][h * qk_nope ..][0..qk_nope];
                            for (q_nope, hist_k_nope) |qv, kv| {
                                score += qv * kv;
                            }
                        }

                        const k_pe_cached = try self.kv.getKPe(layer, @intCast(p));
                        for (q_rope, k_pe_cached) |qv, kv| {
                            score += qv * kv;
                        }

                        scores[p - b_start] = score / @sqrt(@as(f32, @floatFromInt(head_dim)));
                    }

                    var m_j: f32 = -std.math.inf(f32);
                    for (0..b_len) |i| {
                        if (scores[i] > m_j) m_j = scores[i];
                    }
                    var l_j: f32 = 0;
                    for (0..b_len) |i| {
                        scores[i] = @exp(scores[i] - m_j);
                        l_j += scores[i];
                    }

                    const m_new = @max(m_vec, m_j);
                    const scale_o = @exp(m_vec - m_new);
                    const scale_s = @exp(m_j - m_new);

                    for (0..v_dim) |i| {
                        o_vec[i] = o_vec[i] * scale_o;
                    }

                    for (b_start..b_end) |p| {
                        const weight = scores[p - b_start] * scale_s;

                        if (p == pos) {
                            for (0..v_dim) |i| {
                                o_vec[i] += weight * cur_v[i];
                            }
                        } else {
                            const hist_v = self.v_cache[p * v_b_dim ..][h * v_dim ..][0..v_dim];
                            for (0..v_dim) |i| {
                                o_vec[i] += weight * hist_v[i];
                            }
                        }
                    }

                    m_vec = m_new;
                    l_vec = l_vec * scale_o + l_j * scale_s;
                }

                for (0..v_dim) |i| {
                    if (l_vec > 0) {
                        head_out[i] = o_vec[i] / l_vec;
                    } else {
                        head_out[i] = 0;
                    }
                }
            }
        }

        // ===== 9. Batch O 投影 =====
        if (lw.o_proj) |w_ptr| {
            try self.batchMatMulUnified(self.batch_hidden, w_ptr, lw.o_proj_type, self.batch_attn_vec, hp.hidden_size, batch_size, n_heads * v_dim);
        }
    }

    /// MoE FFN 前向
    fn moeFFN(self: *Self, layer: u32) !void {
        const hp = self.weights.hp;
        const lw = self.weights.layers[layer];

        // 前 leading_dense_block_count 层为稠密 FFN，其余为 MoE
        if (layer < hp.leading_dense_block_count or hp.n_routed_experts <= 1) {
            try self.denseFFN(layer);
            return;
        }

        // MoE 门控
        if (lw.moe_gate == null) {
            return error.MissingMoEGateWeights;
        }
        const gate_scores = try self.allocator.alloc(f32, hp.n_routed_experts);
        defer self.allocator.free(gate_scores);

        try self.matVecMulUnified(gate_scores, lw.moe_gate.?, lw.moe_gate_type, self.hidden, hp.n_routed_experts, hp.hidden_size);

        // 路由调度
        const route = try self.sched.route(layer, gate_scores);
        defer {
            self.allocator.free(route.expert_ids);
            self.allocator.free(route.weights);
            self.allocator.free(route.targets);
        }

        // 初始化 FFN 输出
        @memset(self.ffn_out, 0);

        // 执行路由专家（并行）
        var active_count: usize = 0;
        for (route.targets) |t| {
            if (t != .skipped) active_count += 1;
        }

        if (active_count > 0) {
            const expert_outputs = try self.allocator.alloc([]f32, active_count);
            defer {
                for (expert_outputs) |out| self.allocator.free(out);
                self.allocator.free(expert_outputs);
            }

            for (0..active_count) |i| {
                expert_outputs[i] = try self.allocator.alloc(f32, hp.hidden_size);
                @memset(expert_outputs[i], 0);
            }

            const ExpertTask = struct {
                engine: *Self,
                layer: u32,
                eid: u32,
                out: []f32,
                err: ?anyerror = null,

                fn run(ctx: *anyopaque) void {
                    const task = @as(*@This(), @ptrCast(@alignCast(ctx)));
                    const result = task.engine.expertFFN(task.layer, task.eid) catch |err| {
                        task.err = err;
                        return;
                    };
                    defer task.engine.allocator.free(result);
                    @memcpy(task.out, result);
                }
            };

            // 防御性断言：Top-K 最多 4 个专家
            std.debug.assert(active_count <= 4);

            var tasks: [4]ExpertTask = undefined;
            var out_idx: usize = 0;

            for (route.expert_ids, route.targets) |eid, target| {
                if (target == .skipped) continue;

                tasks[out_idx] = ExpertTask{
                    .engine = self,
                    .layer = layer,
                    .eid = eid,
                    .out = expert_outputs[out_idx],
                };
                try self.pool.submit(.{ .run_fn = ExpertTask.run, .ctx = &tasks[out_idx] });
                out_idx += 1;
            }

            self.pool.waitAll();

            for (0..out_idx) |i| {
                if (tasks[i].err) |err| return err;
            }

            // 加权累加
            out_idx = 0;
            for (route.expert_ids, route.targets, 0..) |_, target, i| {
                if (target == .skipped) continue;
                const w = route.weights[i];
                for (self.ffn_out, expert_outputs[out_idx]) |*f, e| {
                    f.* += w * e;
                }
                out_idx += 1;
            }
        }

        // 共享专家
        if (lw.shared_gate) |_| {
            const shared_out = try self.sharedExpertFFN(layer);
            defer self.allocator.free(shared_out);

            for (self.ffn_out, shared_out) |*f, s| {
                f.* += s;
            }
        }

        // 延迟专家回退偏置
        if (route.deferred) {
            if (self.sched.deferral_bias) |bias| {
                for (self.ffn_out, bias) |*f, b| {
                    f.* += b;
                }
            }
        }

        // 写回 hidden
        @memcpy(self.hidden, self.ffn_out);
    }

    /// matVecMul 并行任务上下文（用于 ThreadPool 提交）
    const MatVecTask = struct {
        engine: *Self,
        out: []f32,
        ptr: [*]const u8,
        type_: GgmlType,
        in: []const f32,
        out_dim: usize,
        in_dim: usize,
        err: ?anyerror = null,

        fn run(ctx: *anyopaque) void {
            const self = @as(*@This(), @ptrCast(@alignCast(ctx)));
            self.engine.matVecMulUnified(self.out, self.ptr, self.type_, self.in, self.out_dim, self.in_dim) catch |err| {
                self.err = err;
            };
        }
    };

    /// 稠密 FFN（第 0 层）
    /// Gate/Up 投影并行执行（ThreadPool），减少约 30-40% FFN 延迟
    fn denseFFN(self: *Self, layer: u32) !void {
        const hp = self.weights.hp;
        const lw = self.weights.layers[layer];

        // 使用稠密层权重（dense_gate/up/down）作为稠密 FFN
        if (lw.dense_gate == null or lw.dense_up == null or lw.dense_down == null) {
            return error.MissingDenseWeights;
        }

        const intermediate = hp.intermediate_size;
        const hidden = hp.hidden_size;

        // Gate 投影（QMM 直通）
        const gate_out = try self.allocator.alloc(f32, intermediate);
        defer self.allocator.free(gate_out);

        // Up 投影（QMM 直通）
        const up_out = try self.allocator.alloc(f32, intermediate);
        defer self.allocator.free(up_out);

        // Gate / Up 并行提交到线程池
        var gate_task = MatVecTask{
            .engine = self,
            .out = gate_out,
            .ptr = lw.dense_gate.?,
            .type_ = lw.dense_type,
            .in = self.hidden,
            .out_dim = intermediate,
            .in_dim = hidden,
        };
        var up_task = MatVecTask{
            .engine = self,
            .out = up_out,
            .ptr = lw.dense_up.?,
            .type_ = lw.dense_type,
            .in = self.hidden,
            .out_dim = intermediate,
            .in_dim = hidden,
        };

        try self.pool.submit(.{ .run_fn = MatVecTask.run, .ctx = &gate_task });
        try self.pool.submit(.{ .run_fn = MatVecTask.run, .ctx = &up_task });
        self.pool.waitAll();

        if (gate_task.err) |err| return err;
        if (up_task.err) |err| return err;

        // SiLU(gate) * up
        math.siluInplace(gate_out);
        math.mul(up_out, gate_out, up_out);

        // Down 投影（QMM 直通）
        const result = try self.allocator.alloc(f32, hidden);
        defer self.allocator.free(result);
        try self.matVecMulUnified(result, lw.dense_down.?, lw.dense_type, up_out, hidden, intermediate);

        @memcpy(self.hidden, result);
    }

    /// 共享专家 FFN: SiLU(gate) * up → down
    /// Gate/Up 投影并行执行（ThreadPool）
    fn sharedExpertFFN(self: *Self, layer: u32) ![]f32 {
        const hp = self.weights.hp;
        const lw = self.weights.layers[layer];

        const intermediate = hp.intermediate_size;
        const hidden = hp.hidden_size;

        // Gate 投影（QMM 直通）
        const gate_out = try self.allocator.alloc(f32, intermediate);
        defer self.allocator.free(gate_out);

        // Up 投影（QMM 直通）
        const up_out = try self.allocator.alloc(f32, intermediate);
        defer self.allocator.free(up_out);

        // Gate / Up 并行提交到线程池
        var gate_task = MatVecTask{
            .engine = self,
            .out = gate_out,
            .ptr = lw.shared_gate.?,
            .type_ = lw.shared_type,
            .in = self.hidden,
            .out_dim = intermediate,
            .in_dim = hidden,
        };
        var up_task = MatVecTask{
            .engine = self,
            .out = up_out,
            .ptr = lw.shared_up.?,
            .type_ = lw.shared_type,
            .in = self.hidden,
            .out_dim = intermediate,
            .in_dim = hidden,
        };

        try self.pool.submit(.{ .run_fn = MatVecTask.run, .ctx = &gate_task });
        try self.pool.submit(.{ .run_fn = MatVecTask.run, .ctx = &up_task });
        self.pool.waitAll();

        if (gate_task.err) |err| return err;
        if (up_task.err) |err| return err;

        // SiLU(gate) * up
        math.siluInplace(gate_out);
        math.mul(up_out, gate_out, up_out);

        // Down 投影（QMM 直通）
        const result = try self.allocator.alloc(f32, hidden);
        errdefer self.allocator.free(result);
        try self.matVecMulUnified(result, lw.shared_down.?, lw.shared_type, up_out, hidden, intermediate);

        return result;
    }

    /// 路由专家 FFN: SiLU(gate) * up → down
    /// 专家权重在 GGUF 中以 3D 张量连续存储：[n_experts, out_dim, in_dim]
    /// 每个专家的权重块偏移 = expert_id * (out_dim * row_bytes_for_type)
    /// 其中 row_bytes_for_type 取决于量化类型
    fn expertFFN(self: *Self, layer: u32, expert_id: u32) ![]f32 {
        const hp = self.weights.hp;
        const lw = self.weights.layers[layer];

        const intermediate = hp.moe_intermediate_size;
        const hidden = hp.hidden_size;
        const ggml_type = lw.expert_type;

        // 计算每个专家的权重块字节大小
        const gate_up_bytes = ggml_type.tensorBytes(intermediate * hidden);
        const down_bytes = ggml_type.tensorBytes(hidden * intermediate);

        const gate_offset = expert_id * gate_up_bytes;
        const up_offset = expert_id * gate_up_bytes;
        const down_offset = expert_id * down_bytes;

        // Gate 投影（QMM 直通）
        const gate_out = try self.allocator.alloc(f32, intermediate);
        defer self.allocator.free(gate_out);
        try self.matVecMulUnified(gate_out, lw.expert_gate.? + gate_offset, ggml_type, self.hidden, intermediate, hidden);

        // Up 投影（QMM 直通）
        const up_out = try self.allocator.alloc(f32, intermediate);
        defer self.allocator.free(up_out);
        try self.matVecMulUnified(up_out, lw.expert_up.? + up_offset, ggml_type, self.hidden, intermediate, hidden);

        // SiLU(gate) * up
        math.siluInplace(gate_out);
        math.mul(up_out, gate_out, up_out);

        // Down 投影（QMM 直通）
        const result = try self.allocator.alloc(f32, hidden);
        errdefer self.allocator.free(result);
        try self.matVecMulUnified(result, lw.expert_down.? + down_offset, ggml_type, up_out, hidden, intermediate);

        return result;
    }

    /// 计算 logits: hidden → vocab_size
    fn computeLogits(self: *Self) !void {
        const hp = self.weights.hp;
        if (self.weights.output_proj == null) {
            return error.MissingOutputProjection;
        }
        try self.matVecMulUnified(self.logits, self.weights.output_proj.?, self.weights.output_proj_type, self.hidden, hp.vocab_size, hp.hidden_size);
    }

    /// 统一矩阵-向量乘法：根据权重类型自动选择 QMM / OpenCL / f32 GEMM
    fn matVecMulUnified(self: *Self, out: []f32, ptr: [*]const u8, type_: GgmlType, x: []const f32, out_dim: usize, in_dim: usize) !void {
        switch (type_) {
            .q8_0 => {
                // MVP：Q8_0 优先走 OpenCL，失败则回退 CPU
                if (self.opencl) |*cl| {
                    const block_size: usize = 32;
                    const blocks_per_row = (in_dim + block_size - 1) / block_size;
                    const n_blocks = out_dim * blocks_per_row;
                    const bytes = ptr[0 .. n_blocks * @sizeOf(dequant.BlockQ8_0)];
                    if (cl.matMulQ8_0(out, bytes, x, @intCast(out_dim), @intCast(in_dim))) {
                        return;
                    } else |_| {
                        // OpenCL 失败，回退 CPU 路径
                    }
                }
                const block_size: usize = 32;
                const blocks_per_row = (in_dim + block_size - 1) / block_size;
                const n_blocks = out_dim * blocks_per_row;
                const bytes = ptr[0 .. n_blocks * @sizeOf(dequant.BlockQ8_0)];
                const blocks: []const dequant.BlockQ8_0 = @ptrCast(@alignCast(bytes));
                qmm.matVecMulQ8_0(out, blocks, x, out_dim, in_dim);
            },
            else => {
                const w = try dequant.dequantize(self.allocator, ptr, type_, out_dim * in_dim);
                defer self.allocator.free(w);
                math.matVecMul(out, w, x, out_dim, in_dim);
            },
        }
    }

    /// 批量统一矩阵乘法：Q8_0 使用 matMulQ8_0，其他类型回退到逐向量 f32
    fn batchMatMulUnified(
        self: *Self,
        out: []f32, // [batch, out_dim]
        ptr: [*]const u8,
        type_: GgmlType,
        in: []const f32, // [batch, in_dim]
        out_dim: usize,
        batch: usize,
        in_dim: usize,
    ) !void {
        switch (type_) {
            .q8_0 => {
                const block_size: usize = 32;
                const blocks_per_row = (in_dim + block_size - 1) / block_size;
                const n_blocks = out_dim * blocks_per_row;
                const bytes = ptr[0 .. n_blocks * @sizeOf(dequant.BlockQ8_0)];
                const blocks: []const dequant.BlockQ8_0 = @ptrCast(@alignCast(bytes));
                qmm.matMulQ8_0(self.batch_proj_tmp, blocks, in, out_dim, batch, in_dim);
                // Transpose: [out_dim, batch] → [batch, out_dim]
                for (0..batch) |bi| {
                    for (0..out_dim) |oi| {
                        out[bi * out_dim + oi] = self.batch_proj_tmp[oi * batch + bi];
                    }
                }
            },
            else => {
                const w = try dequant.dequantize(self.allocator, ptr, type_, out_dim * in_dim);
                defer self.allocator.free(w);
                for (0..batch) |bi| {
                    math.matVecMul(out[bi * out_dim ..], w, in[bi * in_dim ..], out_dim, in_dim);
                }
            },
        }
    }

    /// 热节流检查
    fn checkThermal(self: *Self) void {
        // 仅在 Android 上读取热区温度
        if (comptime @import("builtin").os.tag == .linux) {
            if (thermal.ThrottleState.readThermalZone(self.allocator)) |temp| {
                self.throttle_state.updateFromTemperature(temp);
            } else |_| {}
        }
    }
};
