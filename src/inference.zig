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

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, weights: model.ModelWeights, tok: tokenizer.Tokenizer, max_seq_len: u32) !Self {
        const hp = weights.hp;

        const kv = try kv_cache.KVCache.init(allocator, hp, max_seq_len);
        const rope_cache = try math.RoPECache.init(allocator, max_seq_len, hp.qk_rope_head_dim, hp.rope_freq_base);

        var throttle_state = thermal.ThrottleState.init();
        const sched = try scheduler.Scheduler.init(allocator, hp.num_hidden_layers, hp.n_routed_experts, &throttle_state, hp.routed_scaling_factor);

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
            .cached_layer = std.math.maxInt(u32),
            .cached_k_b_w = null,
            .cached_v_b_w = null,
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
        if (self.cached_k_b_w) |w| allocator.free(w);
        if (self.cached_v_b_w) |w| allocator.free(w);
        self.rope_cache.deinit(allocator);
        self.kv.deinit();
        self.sched.deinit();
        self.tok.deinit();
        self.weights.deinit(allocator);
    }

    /// 前向传播：处理一个 token，返回 logits
    pub fn forward(self: *Self, token_id: u32, pos: u32) ![]f32 {
        const hp = self.weights.hp;

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
        if (token_ids.len == 1) {
            return self.forward(token_ids[0], start_pos);
        }

        const hp = self.weights.hp;

        // 1. Embedding 查找所有 token 到 batch_hidden
        for (token_ids, 0..) |tid, i| {
            try self.embedToken(tid);
            @memcpy(self.batch_hidden[i * hp.hidden_size ..][0..hp.hidden_size], self.hidden);
        }

        // 2. 层优先：外层循环 layer，内层循环 token
        // 这样 cached_layer 在同一层处理多个 token 期间保持不变，
        // k_b / v_b 权重每层只反量化一次
        for (0..hp.num_hidden_layers) |layer| {
            for (0..token_ids.len) |bi| {
                const pos = start_pos + @as(u32, @intCast(bi));
                @memcpy(self.hidden, self.batch_hidden[bi * hp.hidden_size ..][0..hp.hidden_size]);
                try self.forwardLayer(@intCast(layer), pos);
                @memcpy(self.batch_hidden[bi * hp.hidden_size ..][0..hp.hidden_size], self.hidden);
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
            // Bug 5 修复：使用 tensorBytes 计算行字节偏移，而非假设 F32 布局
            const row_bytes = self.weights.token_embd_type.tensorBytes(hp.hidden_size);
            const embd = try dequant.dequantize(
                self.allocator,
                embd_ptr + token_id * row_bytes,
                self.weights.token_embd_type,
                hp.hidden_size,
            );
            defer self.allocator.free(embd);
            @memcpy(self.hidden, embd);
        }
    }

    fn forwardLayer(self: *Self, layer: u32, pos: u32) !void {
        const hp = self.weights.hp;
        const lw = self.weights.layers[layer];

        // 保存残差
        const residual = try self.allocator.dupe(f32, self.hidden);
        defer self.allocator.free(residual);

        // a. RMSNorm (attn_norm)
        if (lw.attn_norm) |norm_ptr| {
            const norm = try dequant.dequantize(self.allocator, norm_ptr, lw.attn_norm_type, hp.hidden_size);
            defer self.allocator.free(norm);
            math.rmsNorm(self.hidden, self.hidden, norm, hp.rms_norm_eps);
        }

        // b. MLA 注意力
        try self.mlaAttention(layer, pos);

        // c. 残差连接
        for (self.hidden, residual) |*h, r| {
            h.* += r;
        }

        // 保存注意力后的残差
        const residual2 = try self.allocator.dupe(f32, self.hidden);
        defer self.allocator.free(residual2);

        // d. RMSNorm (ffn_norm)
        if (lw.ffn_norm) |norm_ptr| {
            const norm = try dequant.dequantize(self.allocator, norm_ptr, lw.ffn_norm_type, hp.hidden_size);
            defer self.allocator.free(norm);
            math.rmsNorm(self.hidden, self.hidden, norm, hp.rms_norm_eps);
        }

        // e. MoE FFN
        try self.moeFFN(layer);

        // f. 残差连接
        for (self.hidden, residual2) |*h, r| {
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

        const cur_k_nope_all = try self.allocator.dupe(f32, self.k_nope_out);
        defer self.allocator.free(cur_k_nope_all);
        const cur_v_all = try self.allocator.dupe(f32, self.v_out);
        defer self.allocator.free(cur_v_all);

        const BLOCK_SIZE: usize = 64;
        var o_vec = try self.allocator.alloc(f32, v_dim);
        defer self.allocator.free(o_vec);

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
                        const kv_cached = try self.kv.getKVCompressed(layer, @intCast(p));
                        if (self.cached_k_b_w) |w| {
                            math.matVecMul(self.k_nope_out, w, kv_cached, k_b_dim, kv_rank);
                        }
                        const hist_k_nope = self.k_nope_out[h * qk_nope ..][0..qk_nope];
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
                        const kv_cached = try self.kv.getKVCompressed(layer, @intCast(p));
                        if (self.cached_v_b_w) |w| {
                            math.matVecMul(self.v_out, w, kv_cached, v_b_dim, kv_rank);
                        }
                        const hist_v = self.v_out[h * v_dim ..][0..v_dim];
                        for (0..v_dim) |i| {
                            o_vec[i] += weight * hist_v[i];
                        }
                    }
                }

                m_vec = m_new;
                l_vec = l_vec * scale_o + l_j * scale_s;
            }

            for (0..v_dim) |i| {
                head_out[i] = o_vec[i] / l_vec;
            }
        }

        // ===== 6. 输出投影: attn_out[num_heads * v_head_dim] → hidden[hidden_size] =====
        if (lw.o_proj) |w_ptr| {
            try self.matVecMulUnified(self.hidden, w_ptr, lw.o_proj_type, self.attn_out, hp.hidden_size, n_heads * v_dim);
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

            const ExpertCtx = struct {
                engine: *Self,
                layer: u32,
                eid: u32,
                out: []f32,
            };

            const worker = struct {
                fn run(ctx: ExpertCtx) void {
                    const result = ctx.engine.expertFFN(ctx.layer, ctx.eid) catch {
                        @memset(ctx.out, 0);
                        return;
                    };
                    defer ctx.engine.allocator.free(result);
                    @memcpy(ctx.out, result);
                }
            }.run;

            // 防御性断言：Top-K 最多 4 个专家，固定线程数组足够
            std.debug.assert(active_count <= 4);

            var threads: [4]std.Thread = undefined;
            var thread_idx: usize = 0;
            var out_idx: usize = 0;

            errdefer {
                // spawn 失败时，join 已启动的线程避免泄漏
                for (0..thread_idx) |ti| {
                    threads[ti].join();
                }
            }

            for (route.expert_ids, route.targets) |eid, target| {
                if (target == .skipped) continue;

                const ctx = ExpertCtx{
                    .engine = self,
                    .layer = layer,
                    .eid = eid,
                    .out = expert_outputs[out_idx],
                };
                out_idx += 1;

                threads[thread_idx] = try std.Thread.spawn(.{}, worker, .{ctx});
                thread_idx += 1;
            }

            for (0..thread_idx) |ti| {
                threads[ti].join();
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

    /// 稠密 FFN（第 0 层）
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
        try self.matVecMulUnified(gate_out, lw.dense_gate.?, lw.dense_type, self.hidden, intermediate, hidden);

        // Up 投影（QMM 直通）
        const up_out = try self.allocator.alloc(f32, intermediate);
        defer self.allocator.free(up_out);
        try self.matVecMulUnified(up_out, lw.dense_up.?, lw.dense_type, self.hidden, intermediate, hidden);

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
    fn sharedExpertFFN(self: *Self, layer: u32) ![]f32 {
        const hp = self.weights.hp;
        const lw = self.weights.layers[layer];

        const intermediate = hp.intermediate_size;
        const hidden = hp.hidden_size;

        // Gate 投影（QMM 直通）
        const gate_out = try self.allocator.alloc(f32, intermediate);
        defer self.allocator.free(gate_out);
        try self.matVecMulUnified(gate_out, lw.shared_gate.?, lw.shared_type, self.hidden, intermediate, hidden);

        // Up 投影（QMM 直通）
        const up_out = try self.allocator.alloc(f32, intermediate);
        defer self.allocator.free(up_out);
        try self.matVecMulUnified(up_out, lw.shared_up.?, lw.shared_type, self.hidden, intermediate, hidden);

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

    /// 统一矩阵-向量乘法：根据权重类型自动选择 QMM 或 f32 GEMM
    fn matVecMulUnified(self: *Self, out: []f32, ptr: [*]const u8, type_: GgmlType, x: []const f32, out_dim: usize, in_dim: usize) !void {
        switch (type_) {
            .q8_0 => {
                const block_size: usize = 32;
                const blocks_per_row = (in_dim + block_size - 1) / block_size;
                const n_blocks = out_dim * blocks_per_row;
                const bytes = ptr[0..n_blocks * @sizeOf(dequant.BlockQ8_0)];
                const blocks: []const dequant.BlockQ8_0 = @alignCast(@ptrCast(bytes));
                qmm.matVecMulQ8_0(out, blocks, x, out_dim, in_dim);
            },
            else => {
                const w = try dequant.dequantize(self.allocator, ptr, type_, out_dim * in_dim);
                defer self.allocator.free(w);
                math.matVecMul(out, w, x, out_dim, in_dim);
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


