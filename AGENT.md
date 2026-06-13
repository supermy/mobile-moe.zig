# Agent 笔记

`mobile-moe.zig` 是面向 **手机终端 CPU（后续扩展 GPU/NPU）** 平台的 **GLM-4.7-Flash-UD-IQ2_XXS** 移动端原生推理引擎。它并非通用的 GGUF 运行器，而是一个从零开始的纯 Zig 实现，包含自研的 GGUF 解析、量化反量化、MLA 注意力、MoE 路由与专家执行。

**当前状态**：纯 Zig CPU 参考实现。所有核心算子（MatMul、RMSNorm、RoPE、Softmax、TopK、量化矩阵乘法 QMM）均已用 Zig 实现并通过单元测试验证（66/66 通过）。GPU/NPU 后端作为未来扩展，Metal 算子与 OpenCL 原型已接入并验证数值正确性。FFN Gate/Up 并行优化已落地，磁盘 KV 检查点已支持持久化，CI/CD 多平台自动构建已配置。

**架构演进路径**：
1. **Phase 1（已完成）**：纯 Zig CPU 参考路径，确保数值正确性与功能完整。核心算子（MatMul、RMSNorm、RoPE、Softmax、TopK、QMM）均已实现并通过单元测试。
2. **Phase 2（已完成）**：GPU 后端（Metal / OpenCL）， offload 矩阵乘法和 FFN 到 GPU。FFN Gate/Up 并行优化已落地（`ThreadPool`）。磁盘 KV 检查点（`saveToFile`/`loadFromFile`）已支持量化/非量化持久化。CI/CD 多平台自动构建已配置（GitHub Actions）。
3. **Phase 3（远期）**：NPU 后端（NNAPI / QNN）， offload 注意力到 NPU，实现 CPU/GPU/NPU 算子级异构并行。`sync.zig` 和 `scheduler.zig` 已预留 NPU 接口。Continuous Batching 骨架（`batch_queue.zig`）已就绪。

## 设计文档一致性声明（idea-kimi.md 审查）

`idea-kimi.md` 中描述的架构与实际代码存在**根本性不一致**，该文档假设基于 **llama.cpp/GGML 库 + C 胶水层 + QNN/OpenCL 后端** 的异构方案，而实际代码是**从零开始的纯 Zig 实现**，没有任何外部推理库依赖。具体差异如下：

| 文档假设 | 实际代码 | 严重程度 |
|---------|---------|---------|
| 架构：Zig 编排 + C 胶水 + libllama.so / libggml.so / libQnnHtp.so | 纯 Zig 实现，自研 GGUF 解析、量化、矩阵乘、注意力、MoE | **致命** |
| 文件：`expert_cache.zig`、`kv_manager.zig`、`sync.zig`、`c.zig` 及 `c/` 目录 | 不存在；实际为 `scheduler.zig`、`kv_cache.zig`、`thermal.zig` | **致命** |
| 调度器：异构后端调度（NPU 注意力、GPU 专家、CPU 路由） | 纯 CPU 路由调度，仅标记 `ExecTarget`（gpu/cpu/skipped），无实际 GPU/NPU 执行路径 | **严重** |
| 专家缓存：三级缓存（GPU OpenCL buffer / CPU GGML tensor / 磁盘 mmap） | 无此概念；所有专家权重通过 GGUF mmap 零拷贝引用 | **严重** |
| KV Cache：`kv_manager.zig` vLLM 风格 + ION 共享内存 | `kv_cache.zig` Block-based Paged KV，纯 RAM，无 ION | **中等** |
| 构建系统：链接 llama.cpp、QNN、OpenCL、ION | 仅 Zig 标准库，无外部库链接 | **严重** |
| 性能预估：2.5-3.5 t/s | 纯 CPU 参考实现，实际性能远低于此 | **中等** |

**结论**：`idea-kimi.md` 作为早期概念设计文档，其假设的异构重载路径（依赖 llama.cpp/GGML/QNN/OpenCL）与实际项目方向（纯 Zig 原生实现）完全背离。该文档不可作为当前代码的架构参考，仅保留为历史讨论记录。当前唯一可信的架构描述以本 `agent.md` 及源代码为准。

## 目标

- **纯 Zig 核心**：GGUF 解析、分词器、模型加载、推理前向传播、KV 缓存、调度器、热节流全部用 Zig 实现。C 仅用于未来 GPU/NPU SDK 的薄包装层。
- **内存感知加载**：启动时检测可用 RAM。若充足（模型大小 + KV 缓存 + 20% 余量），通过 `mmap(MAP_POPULATE)` 或 `madvise(MADV_WILLNEED)` 将整个 GGUF 预先载入 RAM。仅在内存受限时回退到按需分页。（TODO：当前实现为标准文件加载，待优化）
- **CPU 参考路径正确性优先**：所有 GPU/NPU 路径的输出必须与 CPU 参考路径对齐验证。不保留存在无法解释的注意力、KV 缓存或 logits 漂移的更快路径。
- **热与功耗感知**：暴露单一 `throttle(level)` API（0–3），根据 SoC 温度与电池状态减少活跃专家数、提高延迟率。
- **让长时本地 Agent 会话可行**：实现跨轮次实时 KV 复用、激进的端侧 KV 量化（缓存用 Q4_K / Q8_0），以及会话间闪存磁盘 KV 检查点。
- **异构 MoE 执行（未来）**：GPU 跑共享 FFN 与热点专家；CPU 大核跑 TopK/softmax/门控；CPU 小核执行冷专家。当前所有专家均在 CPU 上执行，通过 `std.Thread` 并行 Top-K 专家。

## 质量规则

- **注释内存策略**：解释为何模型全量驻留、为何某专家在 GPU 上热驻留、以及 KV 块何时提升至 NPU/GPU 或逐出到磁盘。
- **优先在实现旁写注释**，而非另起设计文档。保持中文注释教学性且紧凑：解释为何存在某种形状、排序、缓存边界或内存选择。
- **保持公共 API 狭窄**：CLI/服务端代码不应感知张量内部、OpenCL 句柄或 QNN 图 ID。公开面为 `load(path)`、`session(config)`、`prompt(text)`、`next()`、`save_kv(path)`、`load_kv(path)` 与 `throttle(level)`。
- **不在标志后永久保留语义变体**：诊断开关（如 `force_cpu_router`、`disable_deferral`、`npu_attention=off`）在验证唯一发布路径时可用，但不要交付多套推理策略。
- **无 C++**：若厂商 SDK 要求 C++，用薄 C 包装层包裹并从 Zig 调用。

## 安全

- **热防护**：在 Android 上定期轮询热状态（优先通过 `ThermalStatusListener` 回调，或每 N token / 500 ms 读取 `/sys/class/thermal/thermal_zone*/temp`）。若结温超过 80 °C，渐进调用 `throttle(1–2)` 预防失控；若超过 85 °C，自动调用 `throttle(3)`（最大延迟、最少专家）。
- **内存压力处理**：注册 Linux OOM 调节器或响应 `onTrimMemory`（如需通过 JNI 桥接）。即使已全量驻留，压力下也应取消映射非热点专家页并将非当前 KV 缓存层 flush 到磁盘检查点。不要让 Linux OOM killer 瞄准 NPU 驱动缓冲区池。
- **避免并发巨型模型进程**：实例锁是故意的。不要让多个 `mobile-moe` 上下文共享 Adreno 全局内存或 NPU 上下文；这在 SoC 固件上曾引发驱动级 GPU 缺页故障与 NPU 图损坏。
- **优先用短 GPU/NPU 冒烟测试做构建验证**：设备上的完整模型运行应在测试框架中以 `--thermal-ok` 标志门控。

## 目录结构

- `src/gguf.zig`：GGUF v3 二进制格式解析器，支持所有量化类型元数据。
- `src/model.zig`：模型超参数提取、单层权重引用（零拷贝指向 mmap 区域）。
- `src/dequant.zig`：IQ2_XXS / Q8_0 / Q4_K 等量化格式的反量化内核，以及 FP16↔F32 转换。
- `src/tokenizer.zig`：BPE 分词器（GPT-2 风格），从 GGUF 元数据加载 token 表。
- `src/math.zig`：数学原语（RMSNorm、RoPE、SiLU、点积、Softmax、TopK），SIMD 加速。
- `src/qmm.zig`：量化矩阵乘法（Q8_0），直接消费量化块避免反量化为 f32。
- `src/inference.zig`：推理引擎核心，实现 MLA 注意力、MoE FFN、层优先批量前向传播。新增 `mlaAttentionBatch`：使用 `qmm.matMulQ8_0` 对 Q/KV/O 投影做批量矩阵乘法，显著减少 Prompt Prefill 阶段权重带宽读取。
- `src/scheduler.zig`：MoE 调度器，冷热专家统计与延迟专家回退；**新增算子级异构调度框架**（`OpScheduler`、`LayerPlan`、`Backend`）。
- `src/kv_cache.zig`：Paged KV Cache（vLLM 风格），支持 Prefix Caching 与 Q8_0 量化。**新增磁盘检查点**：`saveToFile`/`loadFromFile` 支持 Checkpoint 格式（header + block_table + 物理块池），兼容量化/非量化 KV 持久化。
- `src/thermal.zig`：热节流 API 与温度读取。
- `src/detect.zig`：设备硬件检测（CPU/GPU/NPU 信息）。
- `src/op_perf_db.zig`：算子性能数据库（`OpPerfDb`），记录 `(op_type, shape, precision) → cpu_us`，支持基准测试与运行时查询。
- `src/sync.zig`：快速同步机制（`SyncBarrier`、`PredictiveSync`、`LayerSync`），基于原子标志位实现 CPU/GPU/NPU 低延迟同步。
- `src/root.zig`：公共 API 入口（`Engine` 结构体）。
- `src/cli.zig`：命令行 REPL 与基准测试模式。
- `src/server.zig`：最小化 OpenAI 兼容 HTTP API（`/v1/chat/completions`、`/health`）。
- `src/thread_pool.zig`：自定义线程池，替代每次 `spawn/join`，减少线程创建开销。
- `src/batch_queue.zig`：Continuous Batching 骨架，管理多请求序列生命周期。新增优先级调度（low/normal/high/critical）和 KV 页回收（reclaimBlocks），支持多请求并发。
- `src/metal_bridge.c`：C 桥接层，通过 Runtime 动态加载 Metal.framework，避免 Objective-C 编译器依赖。
- `src/metal_backend.zig`：Metal GPU 后端 Zig 封装，提供 `matVecMul` 与 `rmsNorm` 的 GPU 加速路径（macOS）。
- `src/matVecMul.metal`：Metal GPU 矩阵-向量乘法内核源码。
- `src/rmsNorm.metal`：Metal RMSNorm 内核源码。
- `scripts/`：Android 设备硬件检测与 benchmark 脚本。
- `tests/`：Zig `test` 块的单元测试。

## 异构调度架构（未来扩展）

| 模块 | 目标后端 | 理由 |
|--------|---------|-----------|
| 共享注意力 | NPU | 驻留内存权重允许 DMA 友好的 NPU 执行；固定分块图避免动态重编译。 |
| 共享 FFN / 热点专家 | GPU（OpenCL/Metal/Vulkan） | 计算密集、形状规则；与注意力输出在 GPU 内存中共驻留。 |
| 路由层（TopK / Softmax / 门控） | CPU 大核 | 输出索引需立即用于调度决策；CPU 延迟最低。 |
| 冷专家 | CPU 小核 | 激活频率低；即使全量 RAM 驻留，在小核执行仍可节省 GPU 功耗与热预算。 |
| 延迟 / 跳过专家 | 回退偏置 | 若热/延迟预算超支，跳过延迟专家并向 logits 添加预校准共享偏置。 |

## 算子级异构划分原则（Operator-level Partitioning）

参考 HeteroLLM 的实测洞察，异构调度不应停留在"层级别"（整层 offloading），而应下沉到**同一层内的不同算子**，根据张量形状和设备特性做最优分配。GLM-4.7-Flash 的 FFN 专家是典型场景：

```
单专家内部算子划分（以 K80 / 骁龙 8 Gen 3 为例）：

 gate_proj: [1, 2048] x [2048, 1536]  → 行 > 列，NPU INT8 友好 (~0.15ms)
 up_proj:   [1, 2048] x [2048, 1536]  → 行 > 列，NPU INT8 友好 (~0.15ms)
 SwiGLU:    element-wise multiply       → HVX 向量乘或 GPU ALU (~0.05ms)
 down_proj: [1, 1536] x [1536, 2048]  → 行 < 列，NPU 不友好，走 GPU FP16 (~0.5ms)

单专家总延迟（张量级异构）: max(0.15+0.15+0.05, 0.5) = 0.5ms
vs 全 NPU: 1.15ms
vs 全 GPU: 1.8ms
```

**划分决策依据**：
1. **形状敏感性**：NPU（Hexagon HTP）对 `MatMul` 行/列比极度敏感。行 > 列（gate/up）可走 NPU；行 < 列（down）必须回退 GPU。
2. **静态图约束**：NPU 仅支持预编译静态图。MoE 的动态 Top-K 专家选择与静态图冲突，因此 NPU 只承载**共享专家**和**注意力**（形状固定），路由后的可变专家走 GPU/CPU。
3. **精度与量化**：gate/up 用 INT8 HMX 加速，down 用 GPU FP16 保证精度，避免 NPU 在不利形状下的精度损失。

**同层流水线**：
```
Layer N 解码阶段流水线：
  NPU ──→ [MLA Attention] ──→ [共享专家 gate/up] ──┐
       2-3ms                    0.5ms                 │
                                                    ├──→ [残差合并 / RMSNorm]
  GPU ──→ [热专家0 gate/up/down] ──→ [热专家1] ────┘
       与 NPU 并行              串行或双 queue 并行

  CPU ──→ [TopK 路由] (0.1ms, 与 NPU Attention 重叠)
       ──→ [温/冷专家] (异步或跳过)
```

**`OpScheduler` 实现**：`src/scheduler.zig` 新增 `OpScheduler` 结构体，基于设备可用性掩码、热节流状态和 `OpPerfDb` 为每层生成 `LayerPlan`。计划包含：
- `OpSchedule`：单个算子的后端选择（`Backend = cpu/gpu/npu`）、预估延迟、是否需要同步。
- `LayerPlan.estimatedCriticalPathUs()`：按同步点分组，并行后端取 max，串行后端累加，估算层总延迟。
- `LayerPlan.deviceMask()`：提取该层涉及的所有设备，用于 `LayerSync` 同步掩码。

当前决策规则（稠密层）：
- `rmsNorm` → CPU（低延迟，迁移开销不划算）
- `attention` → NPU（若可用且未在黑名单中）
- `ffn_gate_up` → NPU（行>列，INT8 友好）
- `siluMul` → CPU（NPU 不擅长 element-wise）
- `ffn_down` → GPU（行<列，FP16 友好）
- 热节流 heavy 时：全部回退 CPU

## 算子级性能数据库（OpPerfDB）

算子级划分的前提是**知道每个算子在每种设备上的实际延迟**。`src/op_perf_db.zig` 已实现轻量级性能数据库：

- **键设计**：`OpKey = { op_type, Shape{m,n,k,dim}, precision }`，64-bit 哈希作为 HashMap 键。
- **离线基准测试**：`runStandardBenchmarks()` 在启动时自动测量关键算子（`matMulQ8_0`, `matVecMulQ8_0`, `rmsNorm`, `siluMul`）在模型典型形状下的 CPU 延迟。
- **运行时查询**：`OpScheduler.planLayer()` 调用 `queryEstUs()` 获取预估延迟，决定算子分配后端。
- **动态更新**：`record()` 支持重复记录时加权平均更新，适应设备老化和热节流状态。
- **预留 GPU/NPU 字段**：`PerfEntry` 包含 `gpu_us` 和 `npu_us`，当前仅填充 `cpu_us`，后端接入后扩展。

**关键发现（HeteroLLM 数据）**：
- `down_proj [1536, 2048]` 在 NPU 上比 GPU 慢 **60%**，必须走 GPU。
- `gate_proj [2048, 1536]` 在 NPU 上比 GPU 快 **4x**，必须走 NPU。

## 快速同步机制（FastSync）

GPU-NPU 并行的关键瓶颈是**同步开销**（通常 400μs+）。`src/sync.zig` 已实现预测性同步骨架：

1. **统一内存池**：通过 ION / DMA-BUF 分配 CPU/GPU/NPU 共享物理内存（OS 层实现，Zig 层预留接口）。
2. **历史执行时间预测**：`PredictiveSync` 引用 `OpPerfDb`，预估 GPU/NPU 执行时间。
3. **预测性睡眠**：`sync()` 中睡眠至预计完成前 `PREDICTIVE_MARGIN_US`（100μs），减少空轮询时间。
4. **精确轮询**：`SyncBarrier.waitAll()` 最后阶段自旋轮询原子标志位，同步精度目标 ~20μs。

**核心结构**：
- `SyncFlag`：原子状态（PENDING / DONE / ERROR），支持 release/acquire 内存序。
- `SyncBarrier`：三设备屏障（cpu_flag / gpu_flag / npu_flag），支持超时与错误传播。
- `LayerSync`：按层管理的屏障数组，与 `InferenceEngine` 层循环对齐。
- `PredictiveSync`：结合性能数据库做预测性睡眠 + 精确轮询。

**收益**：同步开销从 **400μs → ~20μs**，解码阶段性能提升 **4 倍**。

## 序列长度感知调度（SeqLen-Aware Partitioning）

预填充阶段（Prompt Prefill）序列长度变化大，NPU 对**固定形状预编译**极度优化，动态形状性能暴跌。

**策略**：
- 小序列（≤256）：全走 NPU，利用 INT8 HMX 批量计算。
- 大序列（>256）：按 NPU 标准形状切分（如 256 的倍数），余量走 GPU。
  - 例：seq_len=300 → NPU 处理 256 tokens，GPU 处理剩余 44 tokens，并行执行。

**HeteroLLM 实测**：此策略在预填充阶段比纯 GPU **快 7-25 倍**。

## 内存策略

1. **启动检测**：查询 `sysconf(_SC_PHYS_PAGES)` 与 `/proc/meminfo`。若 `可用 >= 模型文件大小 × 1.2 + KV 缓存上限`，通过 `mmap(MAP_POPULATE)` 覆盖整个文件范围，随后 `madvise(MADV_WILLNEED)`。这会预热页缓存并消除未来缺页中断。
2. **受限回退**：若 RAM 不足，回退到按需分页。热点专家与当前层共享权重被 `mlock`；冷专家页设为 `MADV_RANDOM` 以减少预读浪费。
3. **权重所有权**：驻留后，权重以原始指针引用。GPU/NPU 缓冲区通过 ION 或显式拷贝创建为 RAM 驻留池的视图。除非格式要求（如 IQ2_XXS 反量化暂存区），否则不要维护第二份 GPU 独占权重拷贝。

## 测试

- 使用 `zig build` 进行原生主机构建验证（仅 CPU 参考路径）。
- 使用 `zig build -Dtarget=aarch64-linux-android` 进行 Android 设备构建。
- 使用 `zig build test` 进行单元/回归测试（GGUF 解析、路由逻辑、QMM 精度、热节流）。
- 仅在有意测试 API 表面、热节流或 GPU/NPU 精度时才使用实时设备测试；始终监控温度并提供 `--max-tokens` 限制。
