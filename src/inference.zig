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

    /// 批量前向传播：处理多个 token（Prompt Prefill 优化入口）
    /// 当前实现退化为逐 token forward，但接口已建立，后续可接入 FlashAttention 批量计算
    pub fn forwardBatch(self: *Self, token_ids: []const u32, start_pos: u32) ![]f32 {
        if (token_ids.len == 0) return error.EmptyBatch;
        if (token_ids.len == 1) {
            return self.forward(token_ids[0], start_pos);
        }

        // TODO: FlashAttention 批量 Prompt Prefill
        // 当前逐 token 处理，后续优化为层优先批量计算
        for (token_ids, 0..) |tid, i| {
            _ = try self.forward(tid, @intCast(start_pos + i));
        }
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
            const w = try dequant.dequantize(self.allocator, w_ptr, lw.q_a_proj_type, hp.q_lora_rank * hp.hidden_size);
            defer self.allocator.free(w);
            math.matVecMul(self.q_comp, w, self.hidden, hp.q_lora_rank, hp.hidden_size);
        }

        // Q RMSNorm
        if (lw.q_a_norm) |norm_ptr| {
            const norm = try dequant.dequantize(self.allocator, norm_ptr, lw.q_a_norm_type, hp.q_lora_rank);
            defer self.allocator.free(norm);
            math.rmsNorm(self.q_comp, self.q_comp, norm, hp.rms_norm_eps);
        }

        // Q 展开投影: q_comp[q_lora_rank] → q_out[num_heads * head_dim]
        if (lw.q_b_proj) |w_ptr| {
            const w = try dequant.dequantize(self.allocator, w_ptr, lw.q_b_proj_type, n_heads * head_dim * hp.q_lora_rank);
            defer self.allocator.free(w);
            math.matVecMul(self.q_out, w, self.q_comp, n_heads * head_dim, hp.q_lora_rank);
        }

        // ===== 2. KV 压缩投影: hidden[hidden_size] → kv_a_out[kv_lora_rank + qk_rope_head_dim] =====
        // DeepSeek2 中 kv_a_proj 输出 kv_lora_rank + qk_rope_head_dim 维
        // 前 kv_lora_rank 维为压缩 KV，后 qk_rope_head_dim 维为 k_pe
        const kv_a_dim = kv_rank + qk_rope;
        if (lw.kv_a_proj) |w_ptr| {
            const w = try dequant.dequantize(self.allocator, w_ptr, lw.kv_a_proj_type, kv_a_dim * hp.hidden_size);
            defer self.allocator.free(w);
            math.matVecMul(self.kv_a_out, w, self.hidden, kv_a_dim, hp.hidden_size);
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
        var k_b_w: ?[]const f32 = null;
        var v_b_w: ?[]const f32 = null;
        if (lw.k_b) |w_ptr| {
            k_b_w = try dequant.dequantize(self.allocator, w_ptr, lw.k_b_type, k_b_dim * kv_rank);
            math.matVecMul(self.k_nope_out, k_b_w.?, self.kv_comp, k_b_dim, kv_rank);
        }
        if (lw.v_b) |w_ptr| {
            v_b_w = try dequant.dequantize(self.allocator, w_ptr, lw.v_b_type, v_b_dim * kv_rank);
            math.matVecMul(self.v_out, v_b_w.?, self.kv_comp, v_b_dim, kv_rank);
        }
        defer {
            if (k_b_w) |w| self.allocator.free(w);
            if (v_b_w) |w| self.allocator.free(w);
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
                        if (k_b_w) |w| {
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
                        if (v_b_w) |w| {
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
            const w = try dequant.dequantize(self.allocator, w_ptr, lw.o_proj_type, hp.hidden_size * n_heads * v_dim);
            defer self.allocator.free(w);
            math.matVecMul(self.hidden, w, self.attn_out, hp.hidden_size, n_heads * v_dim);
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
        const gate_ptr = lw.moe_gate.?;
        const gate_w = try dequant.dequantize(self.allocator, gate_ptr, lw.moe_gate_type, hp.n_routed_experts * hp.hidden_size);
        defer self.allocator.free(gate_w);

        const gate_scores = try self.allocator.alloc(f32, hp.n_routed_experts);
        defer self.allocator.free(gate_scores);

        math.matVecMul(gate_scores, gate_w, self.hidden, hp.n_routed_experts, hp.hidden_size);

        // 路由调度
        const route = try self.sched.route(layer, gate_scores);
        defer {
            self.allocator.free(route.expert_ids);
            self.allocator.free(route.weights);
            self.allocator.free(route.targets);
        }

        // 初始化 FFN 输出
        @memset(self.ffn_out, 0);

        // 执行路由专家
        for (route.expert_ids, 0..) |eid, i| {
            switch (route.targets[i]) {
                .skipped => continue,
                .gpu, .cpu => {
                    const expert_out = try self.expertFFN(layer, eid);
                    defer self.allocator.free(expert_out);

                    // 加权累加
                    const w = route.weights[i];
                    for (self.ffn_out, expert_out) |*f, e| {
                        f.* += w * e;
                    }
                },
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

        // Gate 投影
        const gate_w = try dequant.dequantize(self.allocator, lw.dense_gate.?, lw.dense_type, intermediate * hidden);
        defer self.allocator.free(gate_w);
        const gate_out = try self.allocator.alloc(f32, intermediate);
        math.matVecMul(gate_out, gate_w, self.hidden, intermediate, hidden);

        // Up 投影
        const up_w = try dequant.dequantize(self.allocator, lw.dense_up.?, lw.dense_type, intermediate * hidden);
        defer self.allocator.free(up_w);
        const up_out = try self.allocator.alloc(f32, intermediate);
        math.matVecMul(up_out, up_w, self.hidden, intermediate, hidden);

        // SiLU(gate) * up
        math.siluInplace(gate_out);
        math.mul(up_out, gate_out, up_out);
        self.allocator.free(gate_out);

        // Down 投影
        const down_w = try dequant.dequantize(self.allocator, lw.dense_down.?, lw.dense_type, hidden * intermediate);
        defer self.allocator.free(down_w);
        const result = try self.allocator.alloc(f32, hidden);
        math.matVecMul(result, down_w, up_out, hidden, intermediate);
        self.allocator.free(up_out);

        @memcpy(self.hidden, result);
        self.allocator.free(result);
    }

    /// 共享专家 FFN: SiLU(gate) * up → down
    fn sharedExpertFFN(self: *Self, layer: u32) ![]f32 {
        const hp = self.weights.hp;
        const lw = self.weights.layers[layer];

        const intermediate = hp.intermediate_size;
        const hidden = hp.hidden_size;

        // Gate 投影
        const gate_w = try dequant.dequantize(self.allocator, lw.shared_gate.?, lw.shared_type, intermediate * hidden);
        defer self.allocator.free(gate_w);
        const gate_out = try self.allocator.alloc(f32, intermediate);
        math.matVecMul(gate_out, gate_w, self.hidden, intermediate, hidden);

        // Up 投影
        const up_w = try dequant.dequantize(self.allocator, lw.shared_up.?, lw.shared_type, intermediate * hidden);
        defer self.allocator.free(up_w);
        const up_out = try self.allocator.alloc(f32, intermediate);
        math.matVecMul(up_out, up_w, self.hidden, intermediate, hidden);

        // SiLU(gate) * up
        math.siluInplace(gate_out);
        math.mul(up_out, gate_out, up_out);
        self.allocator.free(gate_out);

        // Down 投影
        const down_w = try dequant.dequantize(self.allocator, lw.shared_down.?, lw.shared_type, hidden * intermediate);
        defer self.allocator.free(down_w);
        const result = try self.allocator.alloc(f32, hidden);
        math.matVecMul(result, down_w, up_out, hidden, intermediate);
        self.allocator.free(up_out);

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
        // gate/up: [intermediate, hidden] per expert
        // down: [hidden, intermediate] per expert
        const gate_up_bytes = ggml_type.tensorBytes(intermediate * hidden);
        const down_bytes = ggml_type.tensorBytes(hidden * intermediate);

        const gate_offset = expert_id * gate_up_bytes;
        const up_offset = expert_id * gate_up_bytes; // up 张量独立存储，每专家偏移相同
        const down_offset = expert_id * down_bytes;

        // Gate 投影
        const gate_w = try dequant.dequantize(self.allocator, lw.expert_gate.? + gate_offset, ggml_type, intermediate * hidden);
        defer self.allocator.free(gate_w);
        const gate_out = try self.allocator.alloc(f32, intermediate);
        math.matVecMul(gate_out, gate_w, self.hidden, intermediate, hidden);

        // Up 投影
        const up_w = try dequant.dequantize(self.allocator, lw.expert_up.? + up_offset, ggml_type, intermediate * hidden);
        defer self.allocator.free(up_w);
        const up_out = try self.allocator.alloc(f32, intermediate);
        math.matVecMul(up_out, up_w, self.hidden, intermediate, hidden);

        // SiLU(gate) * up
        math.siluInplace(gate_out);
        math.mul(up_out, gate_out, up_out);
        self.allocator.free(gate_out);

        // Down 投影
        const down_w = try dequant.dequantize(self.allocator, lw.expert_down.? + down_offset, ggml_type, hidden * intermediate);
        defer self.allocator.free(down_w);
        const result = try self.allocator.alloc(f32, hidden);
        math.matVecMul(result, down_w, up_out, hidden, intermediate);
        self.allocator.free(up_out);

        return result;
    }

    /// 计算 logits: hidden → vocab_size
    fn computeLogits(self: *Self) !void {
        const hp = self.weights.hp;
        if (self.weights.output_proj == null) {
            return error.MissingOutputProjection;
        }
        const w = try dequant.dequantize(self.allocator, self.weights.output_proj.?, self.weights.output_proj_type, hp.vocab_size * hp.hidden_size);
        defer self.allocator.free(w);
        math.matVecMul(self.logits, w, self.hidden, hp.vocab_size, hp.hidden_size);
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


