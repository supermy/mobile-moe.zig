# mobile-moe.zig 项目进度

目标模型：`/Users/moyong/project/ai/models/GLM-4.7-Flash-UD-IQ2_XXS.gguf`

## 已完成

### 架构与核心
- [x] 纯 Zig GGUF v3 解析器（`gguf.zig`），支持全部量化类型元数据
- [x] 模型超参数提取与单层权重零拷贝引用（`model.zig`）
- [x] BPE 分词器（GPT-2 风格，GLM-4 预处理方案）（`tokenizer.zig`）
- [x] MLA 注意力完整前向传播（Q/K/V 压缩投影、RoPE、FlashAttention 分块）
- [x] MoE FFN（门控路由 + 共享专家 + Top-K 路由专家）
- [x] Paged KV Cache（Block-based，vLLM 风格），支持 Prefix Caching（`kv_cache.zig`）
- [x] 热节流 API（`throttle(level)`，0–3 级自动温控）（`thermal.zig`）
- [x] 异构 MoE 调度器（冷热专家统计、延迟专家回退）（`scheduler.zig`）

### 性能优化
- [x] SIMD 向量化 MatMul / dot / rmsNorm（4-wide f32）
- [x] FlashAttention 分块计算（BLOCK_SIZE=64, online softmax）
- [x] Prompt Prefill 批量 forward 接口（`InferenceEngine.forwardBatch`）
- [x] Prompt Prefill 层优先优化：外层循环 layer、内层循环 token，每层权重只反量化一次
- [x] 量化矩阵乘法 QMM（`qmm.zig`: `matVecMulQ8_0` SIMD，直接消费 Q8_0 量化块）
- [x] QMM 集成到 InferenceEngine：`q_a_proj` / `kv_a_proj` 等权重保持 Q8_0 格式
- [x] Expert 并行：使用 `std.Thread` 并行执行 Top-K 专家 FFN
- [x] KV Cache Q8_0 量化接口（`enableQuantization()` + 读写支持）
- [x] 预分配残差缓冲区：消除 `forwardLayer` 每层 2 次堆分配
- [x] KV 展开缓存（`kv_nope_cache` / `v_cache`）：避免 attention 循环内重复 `matVecMul`，预展开所有历史位置
- [x] `forwardBatch` 批量 attention：使用 `matMulQ8_0` 对 Q/KV/O 投影做批量矩阵乘法，减少权重带宽读取
- [x] 自定义线程池（`thread_pool.zig`）：固定 N 工作线程替代每次 `spawn/join`，减少线程创建开销
- [x] Continuous Batching 骨架（`batch_queue.zig`）：`Sequence` + `BatchScheduler`，管理多请求生命周期

### 质量与测试
- [x] 深度代码审查（review all）：发现并修复 15+ 个问题
- [x] TDD 修复：
  - 修复 denseFFN/moeFFN/computeLogits 权重缺失时静默跳过
  - 修复 saveKv/loadKv 与 block-based KV cache 不兼容
  - 删除死代码 block_manager.zig
  - 修复 Engine.next 与 Session 状态同步
  - 修复 scheduler hotThreshold 固定栈数组限制（改为 quickSelect）
  - 修复 tokenizer encode 64 字节硬编码限制
  - 修复 tokenizer scores 缺失导致的初始化失败
  - 修复 gguf tensorDataPtr 在 mapped_data 为 null 时返回 undefined
  - 修复 root.zig `@constCast` 未定义行为（改为 `var weights`）
  - 修复 server.zig JSON 注入风险（增加字符转义）
  - 修复 `dequant.zig` / `kv_cache.zig` 量化除零未定义行为（max_abs == 0 时 d = 0）
  - 修复 `scheduler.zig` `deferral_bias` 未初始化导致延迟策略无效
  - 修复 `scheduler.zig` 内存泄漏（`route` 动态分配缺少 `errdefer`）
  - 修复 `server.zig` `extractJsonString` 空格模式 off-by-one（`val_start` 计算错误）
  - 修复 `thread_pool.zig` `waitAll` 只等队列为空、不等执行完成（新增 `running` 计数器）
  - 修复 `inference.zig` attention 输出除以零保护（`l_vec <= 0` 时回退到 0）
  - 修复 `batch_queue.zig` 测试调用参数数量不匹配（`collectBatchIndices`）
- [x] 推理测试：输入"中国的首都是"并获取模型输出

### 工具与部署
- [x] CLI（REPL + 基准测试模式）（`cli.zig`）
- [x] HTTP 服务端（OpenAI 兼容 API：`/v1/chat/completions`、`/health`）（`server.zig`）
- [x] ReleaseFast 编译通过（CLI 592K / Server 439K）
- [x] 本地 Git 仓库初始化并提交
- [x] 手机硬件检测脚本（`scripts/detect_android.sh`）
- [x] CPU 实测脚本（`scripts/bench_cpu.zig`，实测 2.38 GFLOPS）
- [x] GPU/NPU 实测 APK 方案（`scripts/run_aibench.sh`、`scripts/bench_all.sh`）
- [x] Metal 内核源码（`matVecMul.metal`、`rmsNorm.metal`）
- [x] GPU 后端设计文档（Metal / OpenCL / Vulkan 架构）

## 待办

### P0 — 阻塞发布（正确性/稳定性）
- [x] `detect.zig` 跨平台：当前仅限 macOS（`sysctlbyname`），需增加 Android/Linux 路径
- [x] `thermal.zig` 跨平台：`readThermalZone` 在 macOS 上编译失败（`getdents64` 不存在）—— 已用 `switch (comptime builtin.os.tag)` 分平台实现
- [x] `inference.zig` 层缓存完整性：`forwardBatch` 后 `cached_layer` 未重置，导致后续 `forward` 错误命中缓存 —— 已在 `forward`/`forwardBatch` 入口强制重置 `cached_layer = maxInt(u32)`
- [x] `inference.zig` MLA 注意力堆分配：`mlaAttention` 每次 `allocator.dupe` 分配 `cur_k_nope_all`、`cur_v_all`、`o_vec` —— 已预分配缓冲区消除
- [x] `model.zig` 权重类型绑定不完整：`dense_up/down`、`shared_up/down` 未设置量化类型 —— 已补全绑定
- [x] `root.zig` `generate` 逻辑缺陷：EOS 导致前一个有效 token 被丢弃 —— 已重构为"先检查 EOS → 再 forward + append"
- [x] `server.zig` JSON 解析与构建脆弱性：`std.io.fixedBufferStream` 在 Zig 0.16 不存在，提取函数不支持转义/空格 —— 已移除流依赖，手动偏移构建；提取函数支持冒号后空格和转义引号
- [x] **Zig 0.16 全量兼容性修复**：`std.time.Timer`/`microTimestamp`/`sleep` 移除 → 改用 `std.c.clock_gettime` + `std.c.nanosleep`；`std.Thread.Mutex`/`Condition` 移除 → 改用 `std.c.pthread_mutex_t`/`pthread_cond_t`；`std.ArrayList` 变为 unmanaged → 全项目更新 `.empty` 初始化、`append(allocator)`、`deinit(allocator)` 等调用；修复 `comptime_float` 类型推断、枚举字面量运行时推断、`errdefer` 忽略返回值等 20+ 处编译错误。全量 56 测试通过。
- [x] `qmm.zig` 实现 `matMulQ8_0`（Prompt Prefill 批量 attention 需要分块 GEMM）—— 已实现 M×N 分块 GEMM，内层 SIMD 4-wide 权重复用 4 列 N，并补充 batch 一致性测试
- [x] `embedToken` 防御性长度检查：`@memcpy(self.hidden, embd)` 前断言 `embd.len == hp.hidden_size`

### P1 — 近期优化（性能/体验）
- [x] `math.zig` topK 优化：当前全排序 O(n log n)，改为最小堆 O(n log k)（对大规模 vocab 采样关键）—— 已用大小为 k 的最小堆实现，n=154880/k=4 时理论提速约 8 倍
- [x] `server.zig` JSON 解析鲁棒化：`extractJsonString` 已支持冒号后空格、转义引号；OpenAI API 标准请求体无需完整嵌套 JSON 解析
- [x] 内存感知加载：`gguf.zig` `loadFromFile` 已调用 `madvise(MADV_WILLNEED)` 预热页面；`MAP_POPULATE` 为 Linux 特有，macOS 依赖系统预读启发策略
- [x] **算子性能数据库（`op_perf_db.zig`）**：已实现。`OpPerfDb` 基于 HashMap 存储 `(OpKey) → PerfEntry{cpu_us,gpu_us,npu_us,iterations}`，支持 `runStandardBenchmarks()` 自动测量模型典型形状，`record()` 加权平均更新。TDD 覆盖：init/deinit、record/query、加权平均、benchmark matMulQ8_0/rmsNorm、runStandardBenchmarks。
- [x] **GPU 算子接入（Metal 第一步）**：`metal_backend.zig` + `metal_bridge.c` 已实现。通过 Objective-C Runtime 动态加载 Metal.framework，提供 `matVecMul` 与 `rmsNorm` GPU 算子。TDD 覆盖：单算子数值正确性（与 CPU 参考路径逐元素对比，RMSE < 1e-4）。
- [x] **GPU FFN 分算子调度**：`scheduler.zig` 已扩展算子级调度框架。新增 `OpScheduler`、`LayerPlan`、`OpSchedule`、`Backend` 枚举。`planLayer()` 基于设备掩码、热节流状态和 `OpPerfDb` 决策：`gate/up → NPU`、`down → GPU`、`rmsNorm/siluMul → CPU`。TDD 覆盖：全 CPU 回退、NPU/GPU 分配、heavy 节流回退、perf db 数值读取。
- [x] **快速同步原型（`sync.zig`）**：已实现。`SyncFlag`（原子状态 PENDING/DONE/ERROR）、`SyncBarrier`（三设备 waitAll/pollAll/timeout）、`LayerSync`（按层管理）、`PredictiveSync`（结合 `OpPerfDb` 预测性睡眠 + 精确轮询）。TDD 覆盖：flag 状态机、barrier 全完成/超时/错误、poll 非阻塞、layer sync、空 db / 有 db 预测同步。
- [x] **OpenCL 后端原型**：`opencl_backend.zig` 已实现。动态加载 `libOpenCL.so`，支持主队列 + 异步队列双专家并行（`matVecMulTwoExpertsParallel`），参考 HeteroLLM 推荐策略。TDD 覆盖：单算子数值正确性、双专家并行结果对齐。
- [x] **NPU 算子友好性分析框架**：`npu_analyzer.zig` 已实现。基于形状规则（批大小、行/列比例、K 维度、精度）预测 NPU 友好度，验证 `down_proj` NPU 不友好。TDD 覆盖：MatMul 评分、GPU 建议、报告生成。
- [x] **异构调度器 Phase 3 骨架**：`hetero_executor.zig` 已实现。`HeteroExecutor` 管理一层内 CPU/GPU/NPU 算子级并行，`LoadBalancer` 基于温度动态调整负载比例。TDD 覆盖：比例归一化、热节流响应、后端选择回退。
- [ ] 默认测试页面：参考 llama.cpp `llama-server` 提供 Web UI
- [ ] web 增加 README 访问入口

### P2 — 远期扩展（功能/生态）
- [x] **NPU 算子友好性分析框架**：`npu_analyzer.zig` 已实现基于形状规则的离线分析。待 QNN SDK / NNAPI 实测数据接入后，可升级为数据驱动决策。
- [x] **异构调度器 Phase 3 骨架**：`hetero_executor.zig` + `LoadBalancer` 已实现。待 QNN/NNAPI 驱动接入后，可填充 `dispatchNpu` / `dispatchGpu` 实际算子调用。
- [x] **动态负载均衡**：`LoadBalancer` 已实现。基于 `ThrottleState` 实时调整 NPU/GPU/CPU 负载比例（EMA 平滑 + 归一化）。
- [x] **FFN Gate/Up 并行优化**：`denseFFN` / `sharedExpertFFN` 通过 `ThreadPool` 并行执行 gate/up 投影；`moeFFN` 专家并行迁移到 `ThreadPool`。`InferenceEngine` 新增 4 线程 `pool`。全量 56 测试通过。
- [x] **Android 交叉编译配置**：`build.zig` 自动检测 `aarch64-linux-android` 目标，链接 `libdl`。macOS/Android 共用同一构建文件，通过 `target.result.os.tag` 自动切换。
- [x] **深度 review all**：修复 `mlaAttention` 重复 `@memset`、`forwardBatch` 缺少 `SeqLenExceeded` 边界检查、`scheduler.zig` heavy 节流权重未重新归一化。全量 56 测试通过。
- [x] CI/CD：GitHub Actions 多平台自动构建（x86_64-linux, x86_64-macos, aarch64-linux-android, aarch64-macos）
- [x] 磁盘 KV 检查点：`kv_cache.zig` 实现 `saveToFile`/`loadFromFile`，Checkpoint 格式（header + block_table + 物理块池），兼容量化/非量化 KV 持久化
- [ ] 单元测试、集成测试覆盖率提升至 80%+
- [ ] 借鉴 llama.cpp / vLLM：Continuous Batching、Prefix Caching 完整落地

## 附录：模型元数据

```
Metadata                          Value
--------------------------------  --------------------
general.architecture              deepseek2
general.name                      Glm-4.7-Flash
general.quantized_by              Unsloth
general.size_label                64x2.6B
deepseek2.block_count             47
deepseek2.context_length          202752
deepseek2.embedding_length        2048
deepseek2.feed_forward_length     10240
deepseek2.attention.head_count    20
deepseek2.attention.head_count_kv 1
deepseek2.attention.q_lora_rank   768
deepseek2.attention.kv_lora_rank  512
deepseek2.rope.freq_base          1000000
deepseek2.rope.dimension_count    64
deepseek2.expert_used_count       4
deepseek2.expert_count            64
deepseek2.expert_shared_count     1
deepseek2.vocab_size              154880
deepseek2.expert_feed_forward_length 1536
```
