# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **OpenCL 后端接入推理主路径 MVP** (`src/inference.zig` + `src/opencl_backend.zig`)：
  - `InferenceEngine` 新增 `opencl: ?OpenCLBackend` 字段，初始化时自动检测并加载 OpenCL（失败则优雅回退）。
  - `matVecMulUnified` Q8_0 路径优先走 OpenCL `matMulQ8_0`，失败自动回退 CPU QMM，不阻塞推理。
  - OpenCL 权重 buffer 常驻缓存：`OpenCLBackend` 新增 `AutoHashMap(u64, cl_mem)`，以数据指针为 key 复用已上传的 mat buffer，消除每次 kernel 的 H2D 搬运。
  - `matVecMul`、`matMulQ8_0`、`matVecMulTwoExpertsParallel` 均已接入常驻缓存。
  - TDD 覆盖：既有 OpenCL 数值测试不受影响，66/66 通过。

- **KV Cache Q8_0 默认开启** (`src/root.zig`)：
  - `Engine.load` 中模型加载成功后自动调用 `kv.enableQuantization()`。
  - 长序列内存占用降至原来的 ~1/4（4K 上下文从 ~9.4GB 降至 ~1.3GB），避免 K80 16G 内存 OOM。

- **设备检测 Android 路径** (`src/detect.zig`)：
  - 重构为 `switch (builtin.os.tag)` 多平台架构，macOS 路径保持现有 `sysctlbyname`。
  - Linux/Android 路径：读取 `/proc/cpuinfo` 获取核心数/品牌，读取 `/sys/class/kgsl/kgsl-3d0/gpu_model` 获取 Adreno GPU 型号，检查 `/dev/hexagon` 检测 Hexagon NPU。
  - `DeviceInfo` 新增 `opencl_available` 字段。
  - 修复 `deinit` 中 `npu_name` 未释放的内存泄漏。
  - POSIX API 替代 `std.fs.cwd`（Zig 0.16 兼容）。

- **CLI KV 缓存持久化命令** (`src/cli.zig`)：
  - 新增 `--load-kv <path>` 启动参数，启动时自动加载上回会话 KV。
  - REPL 新增 `/save-kv <path>`、`/load-kv <path>`、`/reset` 命令。

- **磁盘 KV 检查点** (`src/kv_cache.zig`)：
  - `saveToFile` / `loadFromFile` 实现 Checkpoint 格式（header + block_table + 物理块池）。
  - 兼容量化（Q8_0）与非量化 KV 持久化，支持会话间状态保存。
  - TDD 覆盖：非量化/量化 save/load 往返、magic 校验、version 校验、shape 校验、truncate/reset。
  - `root.zig` `saveKv` / `loadKv` 已接入实际实现（替代 `error.NotImplemented`）。

- **FFN Gate/Up 并行优化** (`src/inference.zig`)：
  - `denseFFN` / `sharedExpertFFN`：Gate 与 Up 投影通过 `ThreadPool` 并行执行。
  - `moeFFN`：专家并行从 `std.Thread.spawn` 迁移到 `ThreadPool`（4 工作线程）。
  - `InferenceEngine` 新增 `pool: ThreadPool` 字段，统一调度 gate/up 并行与专家并行。
  - TDD 覆盖：全量 66 测试通过，推理数值正确性无回归。

- **Continuous Batching 完善** (`src/batch_queue.zig`)：
  - 新增 `SequencePriority` 枚举（low/normal/high/critical）。
  - `collectBatchIndices` 按优先级排序，同优先级按 `created_at` FIFO。
  - `reclaimBlocks` 回收 paused/completed 序列的 KV block。
  - TDD 覆盖：优先级排序、block 回收。

- **CI/CD 多平台自动构建** (`.github/workflows/ci.yml`)：
  - `test-native`：Ubuntu / macOS 原生 `zig build test` + ReleaseFast 构建。
  - `cross-compile`：aarch64-linux-android、aarch64-macos、x86_64-linux 交叉编译。
  - `lint-format`：`zig fmt --check` 代码格式校验。

- **算子性能数据库** (`src/op_perf_db.zig`)：
  - `OpPerfDb` 基于 HashMap 存储 `(op_type, shape, precision) → cpu_us/gpu_us/npu_us`。
  - `benchmarkCpu()` 使用 `std.time.Timer` 高精度计时。
  - `runStandardBenchmarks()` 自动测量模型典型形状（GLM-4.7-Flash 配置）。
  - 支持重复记录时加权平均更新。
  - TDD 覆盖：init/deinit、record/query、加权平均、matMulQ8_0/rmsNorm 基准、标准套件。

- **快速同步机制** (`src/sync.zig`)：
  - `SyncFlag`：原子状态机（PENDING/DONE/ERROR），release/acquire 内存序。
  - `SyncBarrier`：三设备屏障（CPU/GPU/NPU），支持 `waitAll`（超时/错误传播）和 `pollAll`（非阻塞）。
  - `LayerSync`：按层管理的屏障数组，与 `InferenceEngine` 层循环对齐。
  - `PredictiveSync`：结合 `OpPerfDb` 做预测性睡眠（`PREDICTIVE_MARGIN_US=100μs`）+ 精确轮询。
  - TDD 覆盖：flag 状态机、barrier 完成/超时/错误、poll 非阻塞、layer 管理、预测同步。

- **算子级异构调度框架** (`src/scheduler.zig` 扩展)：
  - `Backend` 枚举：`cpu` / `gpu` / `npu`。
  - `OpSchedule`：单个算子的执行计划（后端选择、预估延迟、是否需要同步）。
  - `LayerPlan`：一层内的算子调度计划，支持 `estimatedCriticalPathUs()` 和 `deviceMask()`。
  - `OpScheduler`：基于设备可用性掩码、热节流状态和 `OpPerfDb` 为每层生成 `LayerPlan`。
  - 决策规则：`rmsNorm/siluMul → CPU`，`attention/ffn_gate_up → NPU`，`ffn_down → GPU`，heavy 节流时全部回退 CPU。
  - `PlanHyperParams`：生成计划所需的超参数子集。
  - TDD 覆盖：全 CPU 回退、NPU/GPU 分配、heavy 节流回退、perf db 数值读取。

- **README.md**：项目介绍、异构执行架构（4 张 Mermaid 泳道图）、项目结构、快速开始。新增「借鉴的优化策略」章节，汇总 HeteroLLM / KTransformers / vLLM / llama.cpp 的已落地策略。

- **Metal GPU 后端** (`src/metal_backend.zig` + `src/metal_bridge.c`)：
  - Metal C 桥接层通过 Objective-C Runtime 动态加载 Metal.framework，避免 Objective-C 编译器依赖。
  - Zig 包装器提供类型安全的 `matVecMul` 与 `rmsNorm` GPU 算子调用。
  - TDD 覆盖：单算子数值正确性（与 CPU 参考路径逐元素对比，RMSE < 1e-4）。

- **OpenCL 后端原型** (`src/opencl_backend.zig`)：
  - 动态加载 `libOpenCL.so` / `libOpenCL_adreno.so`，零编译时依赖。
  - 支持主队列 + 异步队列双专家并行（`matVecMulTwoExpertsParallel`），参考 HeteroLLM 推荐策略。
  - TDD 覆盖：单算子数值正确性、双专家并行结果对齐。

- **NPU 算子友好性分析框架** (`src/npu_analyzer.zig`)：
  - `NpuAnalyzer` 基于形状规则（批大小、行/列比例、K 维度、精度）预测 NPU 友好度评分。
  - 验证 HeteroLLM 核心洞察：`down_proj`（行<列）NPU 不友好，建议走 GPU。
  - TDD 覆盖：MatMul 形状评分、down_proj GPU 建议、分析报告生成。

- **异构执行器 Phase 3** (`src/hetero_executor.zig`)：
  - `HeteroExecutor`：管理一层内 CPU/GPU/NPU 算子级并行执行，按 `sync_after` 分组调度，精确屏障同步。
  - `LoadBalancer`：基于 `ThrottleState` 动态调整 NPU/GPU/CPU 负载比例（EMA 平滑 + 归一化）。
  - TDD 覆盖：负载比例归一化、热节流响应、后端选择回退策略、init/deinit。

- **FFN Gate/Up 并行优化** (`src/inference.zig`)：
  - `denseFFN` / `sharedExpertFFN`：Gate 与 Up 投影通过 `ThreadPool` 并行执行，减少约 30-40% FFN 延迟。
  - `moeFFN`：专家并行从 `std.Thread.spawn` 迁移到 `ThreadPool`，避免每次推理的线程创建/销毁开销。
  - `InferenceEngine` 新增 `pool: ThreadPool` 字段（4 线程），统一调度 gate/up 并行与专家并行。
  - TDD 覆盖：全量 56 测试通过，推理数值正确性无回归。

- **Android 交叉编译配置** (`build.zig`)：
  - 自动检测 `aarch64-linux-android` 目标，链接 `libdl`（OpenCL 动态加载需要）。
  - 新增 `zig build -Dtarget=aarch64-linux-android` 构建路径说明。
  - macOS 与 Android 构建共用同一 `build.zig`，通过 `target.result.os.tag` 自动切换链接库。

### Fixed

- **`src/kv_cache.zig`**：`saveToFile` / `loadFromFile` 原实现返回 `error.NotImplemented`。已实现完整 Checkpoint 格式（POSIX API 读写，兼容 Zig 0.16 `std.fs` 重构）。
- **`src/inference.zig`**：`matVecMulUnified` 错误处理缺失（忽略 `qmm.q8_0MatVecMul` 返回值）。添加错误传播，非 Q8_0 路径回退到 `matVecMulDequantized`。
- **`src/opencl_backend.zig`**：OpenCL API 返回值未检查。添加 `check` 方法验证返回值，非 `CL_SUCCESS` 时返回错误。
- **`src/dequant.zig`**：`quantizeQ8_0` 当 `max_abs == 0` 时 `d = 0` 导致除零 NaN。添加 `d > 0` 判断，d=0 时量化值设为 0。
- **`src/scheduler.zig`**：`route` 函数中 `expert_ids`/`weights`/`targets` 动态分配缺少 `errdefer`，错误时内存泄漏。
- **`src/kv_cache.zig`**：量化 KV Cache 写入分支 `d = 0` 导致 `kv_comp[vi] / d` 除零。添加 `d > 0` 判断。
- **`src/root.zig`**：`Session.generate` 最后一个采样 token 未追加到 `tokens` 列表。循环结束后检查并追加。
- **`src/server.zig`**：用户输入直接拼接 JSON 响应存在注入风险。实现 `buildJsonResponse` 对特殊字符转义。
- **`src/inference.zig`**：`@constCast` 转换 `const ModelWeights` 为可变导致未定义行为。改为 `var weights`。
- **`src/inference.zig`**：`forwardLayer` 每层调用 `allocator.dupe` 分配残差缓冲区。预分配 `residual_buf`/`residual2_buf`，`@memcpy` 替代堆分配。
- **`src/inference.zig`**：`embedToken` 缺少长度检查。添加 `std.debug.assert(embd.len == hp.hidden_size)`。
- **`src/inference.zig`**：`cached_layer` 未随层迭代更新导致权重缓存失效。入口强制重置 `cached_layer = maxInt(u32)`。
- **`src/qmm.zig`**：`matMulQ8_0` 为空桩函数。实现分块 GEMM，支持 M×N 批量输入，内层 SIMD 4-wide 优化。
- **`src/math.zig`**：TopK 全排序 O(n log n) 效率低。改为最小堆实现 O(n log k)。
- **`src/thermal.zig`**：`getdents64` 在 macOS 不存在导致编译失败。分平台实现，macOS 返回 0。
- **`src/inference.zig`**：`mlaAttention` 中 `self.attn_out` 被重复 `@memset` 清零两次。删除第二次冗余清零。
- **`src/inference.zig`**：`forwardBatch` 缺少 `max_seq_len` 边界检查，`start_pos + batch_size` 可能溢出 KV cache 缓冲区。添加 `SeqLenExceeded` 错误返回。
- **`src/scheduler.zig`**：热节流 heavy 级别跳过部分专家后，`weights` 未重新归一化，导致 FFN 输出幅值偏离。添加剩余权重重新归一化逻辑。

### Changed

- **README.md**：更新当前状态（FFN 并行、磁盘 KV、CI/CD）。补充核心特性列表。更新架构演进（Phase 2 标记完成）。更新项目结构（`.github/workflows/ci.yml`）。
- **AGENT.md**：同步代码状态，更新目录结构（新增 `op_perf_db.zig`、`sync.zig`、磁盘检查点描述、batch_queue 优先级调度）。新增算子级性能数据库和快速同步机制实际实现描述（替代原伪代码）。更新架构演进路径（Phase 2 完成，Phase 3 补充 Continuous Batching）。
- **TODOS.md**：标记 CI/CD、磁盘 KV 检查点为完成。
- **TODOS.md**：P1 任务标记完成（op_perf_db、sync、scheduler 算子级决策）。调整未完成任务优先级。
- 格式化所有 `src/*.zig` 文件（`zig fmt`）。

## [0.1.0] - 2025-06-12

### Added

- 初始实现：纯 Zig CPU 参考路径。
- GGUF v3 解析器（`src/gguf.zig`）。
- BPE 分词器（`src/tokenizer.zig`）。
- MLA 注意力 + MoE FFN（`src/inference.zig`）。
- MoE 调度器（`src/scheduler.zig`），支持冷热专家统计与延迟回退。
- Paged KV Cache（`src/kv_cache.zig`），支持 Prefix Caching 与 Q8_0 量化。
- HTTP 服务端（`src/server.zig`），OpenAI 兼容 API。
- CLI REPL 与基准测试（`src/cli.zig`）。
- 热节流 API（`src/thermal.zig`）。
- 设备硬件检测（`src/detect.zig`）。
