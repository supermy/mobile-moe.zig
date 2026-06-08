真实模型：/Users/moyong/project/ai/models/GLM-4.7-Flash-UD-IQ2_XXS.gguf 实现引擎
测试引擎的性能和稳定性；
nng实现高性能的接口服务；
能否提供gpu支持备选方案；
默认测试页面：参见llama.cpp 的 `llama-server` 默认测试页面；提供批量数
单元测试、集成测试、冒烟测试、回归测试、验收测试、系统测试覆盖率100%。
冒烟测试增加日志输出，方便调试。
接口测试、端到端测试，确保接口功能正常，端到端功能正常。浏览器增加console.log输出，方便调试。测试覆盖率：100%。
web增加readme.md的访问入口，方便用户快速上手。

## 已完成
- [x] 深度代码审查：发现 8 个问题（3 严重 / 2 中等 / 3 低）
- [x] TDD 修复：
  - 修复 denseFFN/moeFFN/computeLogits 权重缺失时静默跳过
  - 修复 saveKv/loadKv 与 block-based KV cache 不兼容
  - 删除死代码 block_manager.zig
  - 修复 Engine.next 与 Session 状态同步
  - 修复 scheduler hotThreshold 固定栈数组限制（改为 quickSelect）
  - 修复 tokenizer encode 64 字节硬编码限制
  - 修复 gguf tensorDataPtr 在 mapped_data 为 null 时返回 undefined
- [x] 短期优化（已落地）：
  - SIMD 向量化 MatMul / dot / rmsNorm（4-wide f32）
  - Engine.generateText() 文本生成 API
- [x] 中期优化（已落地）：
  - Continuous Batching 接口设计（RequestQueue + /v1/completions/batch）
  - Server 占位框架（兼容 Zig 0.16.0 std.Io 演进）
- [x] 长期优化（已落地）：
  - KV Cache Q8_0 量化接口（`enableQuantization()` + 读写支持）
  - GPU 后端设计文档（Metal / OpenCL / Vulkan 架构）
- [x] 生产部署：ReleaseFast 编译通过（CLI 592K / Server 439K）
- [x] 本地 Git 仓库初始化并提交（待同步到 GitHub）

## 待办
- [ ] CI/CD 配置：使用 GitHub Actions 实现多平台自动构建、测试和部署
- [ ] 借鉴 ../ds4 ../llama.cpp vllm（PagedAttention + Prefix Caching 已落地）
- [ ] nng / std.Io 实现高性能接口服务
- [ ] GPU 支持备选方案（Metal / OpenCL / Vulkan）
- [ ] 默认测试页面（参考 llama.cpp `llama-server`）
- [ ] 单元测试、集成测试覆盖率提升
- [ ] web 增加 README 访问入口 

真实权重元数据：
Metadata	Value
version
3
tensor_count
844
kv_count
60
general.architecture
deepseek2
general.type
model
general.sampling.top_p
0.949999988079071
general.sampling.temp
1
general.name
Glm-4.7-Flash
general.basename
Glm-4.7-Flash
general.quantized_by
Unsloth
general.size_label
64x2.6B
general.license
mit
general.repo_url

general.base_model.0.name
GLM 4.7 Flash
general.base_model.0.organization
Zai Org
general.base_model.0.repo_url

general.base_model.count
1
general.tags
[unsloth,text-generation, ...]
general.languages
[en,zh, ...]
general.quantization_version
2
general.file_type
25
deepseek2.block_count
47
deepseek2.context_length
202752
deepseek2.embedding_length
2048
deepseek2.feed_forward_length
10240
deepseek2.attention.head_count
20
deepseek2.attention.head_count_kv
1
deepseek2.attention.layer_norm_rms_epsilon
0.000009999999747378752
deepseek2.attention.q_lora_rank
768
deepseek2.attention.kv_lora_rank
512
deepseek2.attention.key_length
576
deepseek2.attention.value_length
512
deepseek2.attention.key_length_mla
256
deepseek2.attention.value_length_mla
256
deepseek2.rope.freq_base
1000000
deepseek2.rope.dimension_count
64
deepseek2.expert_used_count
4
deepseek2.expert_group_count
1
deepseek2.expert_group_used_count
1
deepseek2.expert_gating_func
2
deepseek2.leading_dense_block_count
1
deepseek2.vocab_size
154880
deepseek2.expert_feed_forward_length
1536
deepseek2.expert_count
64
deepseek2.expert_shared_count
1
deepseek2.expert_weights_scale
1.7999999523162842
deepseek2.expert_weights_norm
tokenizer.ggml.model
gpt2
tokenizer.ggml.pre
glm4
tokenizer.ggml.tokens
[!, ", #, $, %, ...]
tokenizer.ggml.token_type
[1, 1, 1, 1, 1, ...]
tokenizer.ggml.merges
[Ġ Ġ, Ġ ĠĠĠ, ĠĠ ĠĠ, ĠĠĠ Ġ, i n, ...]
tokenizer.ggml.eos_token_id
154820
tokenizer.ggml.padding_token_id
154821
tokenizer.ggml.bos_token_id
154822
tokenizer.ggml.eot_token_id
154827
tokenizer.ggml.unknown_token_id
154820
tokenizer.ggml.eom_token_id
154829
tokenizer.ggml.scores
[]
tokenizer.chat_template
[gMASK]<sop> {%- if tools -%} <|system|> # Tools You may call one or more...展开
quantize.imatrix.file
GLM-4.7-Flash-GGUF/imatrix_unsloth.gguf
quantize.imatrix.dataset
unsloth_calibration_GLM-4.7-Flash.txt
quantize.imatrix.entries_count
607
quantize.imatrix.chunks_count
85

output.weight
["2048","154880"]
Q6_K
output_norm.weight
["2048"]
F32
token_embd.weight
["2048","154880"]
Q4_K

GLM-4.7-Flash-UD-IQ2_XXS.gguf 权重的信息：
blk.0.attn_k_b.weight
["192","512","20"]
Q8_0
blk.0.attn_kv_a_mqa.weight
["2048","576"]
Q8_0
blk.0.attn_kv_a_norm.weight
["512"]
F32
blk.0.attn_norm.weight
["2048"]
F32
blk.0.attn_output.weight
["5120","2048"]
IQ4_XS
blk.0.attn_q_a.weight
["2048","768"]
Q4_K
blk.0.attn_q_a_norm.weight
["768"]
F32
blk.0.attn_q_b.weight
["768","5120"]
F16
blk.0.attn_v_b.weight
["512","256","20"]
Q8_0
blk.0.ffn_down.weight
["10240","2048"]
IQ4_XS
blk.0.ffn_gate.weight
["2048","10240"]
IQ4_XS
blk.0.ffn_norm.weight
["2048"]
F32
blk.0.ffn_up.weight
["2048","10240"]
IQ4_XS


blk.25.attn_output.weight
["5120","2048"]
IQ4_XS
blk.25.attn_q_a.weight
["2048","768"]
Q4_K
blk.25.attn_q_a_norm.weight
["768"]
F32
blk.25.attn_q_b.weight
["768","5120"]
Q8_0
blk.25.attn_v_b.weight
["512","256","20"]
Q8_0
blk.25.ffn_down_exps.weight
["1536","2048","64"]
IQ3_S
blk.25.ffn_down_shexp.weight
["1536","2048"]
Q6_K
blk.25.ffn_gate_exps.weight
["2048","1536","64"]
IQ2_XXS
blk.25.ffn_gate_inp.weight
["2048","64"]
F32
blk.25.ffn_gate_shexp.weight
["2048","1536"]
IQ4_NL
blk.25.ffn_norm.weight
["2048"]
F32
blk.25.ffn_up_exps.weight
["2048","1536","64"]
IQ2_XXS
blk.25.ffn_up_shexp.weight
["2048","1536"]
IQ4_NL

量化支持