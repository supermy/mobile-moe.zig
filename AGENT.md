# Agent 笔记

`mobile-moe.zig` 是面向 **手机终端 cpu+gpu+npu** 平台的 **GLM-4.7-Flash-UD-IQ2_XXS** 移动端原生推理引擎。它并非通用的 GGUF 运行器。目标是以 Zig 编写小巧、可读、高性能的代码库，仅在 GPU/NPU SDK 要求时使用 C，计算内核置于 `kernels/` 目录下。

该引擎实现了受 KTransformers *延迟专家（Expert Deferral）* 启发的**异构 MoE 调度器**：路由层跑在 CPU 大核上；注意力卸载到 NPU；共享 FFN 与热点专家跑在 GPU；冷专家在 CPU 小核执行。当系统内存充足时，整个 IQ2_XXS GGUF 预先载入 RAM，消除 `mmap` 缺页延迟并最大化 NPU/GPU DMA 吞吐。Zig 作为宿主语言；C 仅用于厂商 SDK 交互与内核源码。

## 目标

- **内存感知加载**：启动时检测可用 RAM。若充足（模型大小 + KV 缓存 + 20% 余量），则通过 `mmap(MAP_POPULATE)` 或 `madvise(MADV_WILLNEED)` 将整个 GGUF 预先载入 RAM。仅在内存受限时回退到按需分页。
- **注意力上 NPU**：将完整注意力栈（Q/K/V 投影、RoPE、类 FlashAttention 分块、输出投影）卸载到 NPU。使用固定序列分块图（如 128 token 瓦片）避免动态图重编译；CPU 路由层负责最后一块的填充与掩码。
- **异构 MoE 执行**：GPU 跑共享 FFN 与热点专家；CPU 大核跑 TopK/softmax/门控，调度前零跨设备同步；CPU 小核从同一块驻留内存的权重池中执行冷专家。
  - *冷热专家定义*：按推理过程中每层专家的使用频率动态划分，Top 20% 高频专家为**热专家**（Hot Experts），其余 80% 为**冷专家**（Cold Experts）。热/冷边界随运行时统计滑动更新，非固定硬件绑定。
- **端侧延迟专家**：CPU 路由层选出 TopK 专家后，将冷专家推入小核异步工作队列。若热或延迟预算超支，跳过延迟专家并回退到共享"平均专家"偏置——通过固定补偿向量保持 logits 稳定。
- **正确性优先于速度**：不保留存在无法解释的注意力、KV 缓存或 logits 漂移的更快路径。NPU 注意力图必须在每次模型发布时以 CPU 参考路径验证。延迟专家回退偏置必须离线校准并固化在模型头中。
- **热与功耗感知**：暴露单一 `throttle(level)` API（0–3），根据 SoC 温度与电池状态减少活跃专家数、提高延迟率、并将热点专家迁移至 CPU 冷池。
- **让长时本地 Agent 会话可行**：实现跨轮次实时 KV 复用、激进的端侧 KV 量化（缓存用 Q4_K / Q8_0），以及会话间闪存磁盘 KV 检查点。
- **Zig + C，无 C++**：宿主编排、GGUF 解析、会话管理、异构调度器均为 Zig。C 仅用于 OpenCL/Vulkan/HVX SDK 样板代码与内核入口点。

## 质量规则

- **注释调度边界**：每次跨越 GPU/CPU/NPU 或大/小核的调用都必须解释为何如此拆分、目标延迟预算是多少、以及数据如何同步（OpenCL 事件、CPU 栅栏、QNN 图完成句柄或 `std.Thread.Futex`）。
- **注释内存策略**：解释为何模型全量驻留、为何某专家在 GPU 上热驻留、以及 KV 块何时提升至 NPU/GPU 或逐出到磁盘。
- **优先在实现旁写注释**，而非另起设计文档。保持注释教学性且紧凑：解释为何存在某种形状、排序、缓存边界或内存选择。
- **保持公共 API 狭窄**：CLI/服务端代码不应感知张量内部、OpenCL 句柄或 QNN 图 ID。公开面为 `load(path)`、`session(config)`、`prompt(text)`、`next()`、`save_kv(path)`、`load_kv(path)` 与 `throttle(level)`。
- **不在标志后永久保留语义变体**：诊断开关（如 `force_cpu_router`、`disable_deferral`、`npu_attention=off`）在验证唯一发布路径时可用，但不要交付多套推理策略。
- **无 C++**：若厂商 SDK 要求 C++，用薄 C 包装层包裹并从 Zig 调用。

## 安全

- **热防护**：在 Android 上定期轮询热状态（优先通过 `ThermalStatusListener` 回调，或每 N token / 500 ms 读取 `/sys/class/thermal/thermal_zone*/temp`）。若结温超过 80 °C，渐进调用 `throttle(1–2)` 预防失控；若超过 85 °C，自动调用 `throttle(3)`（最大延迟、最少 GPU 专家、暂停 NPU 注意力图）。
- **内存压力处理**：注册 Linux OOM 调节器或响应 `onTrimMemory`（如需通过 JNI 桥接）。即使已全量驻留，压力下也应取消映射非热点专家页并将非当前 KV 缓存层 flush 到磁盘检查点。不要让 Linux OOM killer 瞄准 NPU 驱动缓冲区池。
- **避免并发巨型模型进程**：实例锁是故意的。不要让多个 `mobile-moe` 上下文共享 Adreno 全局内存或 NPU 上下文；这在 SoC 固件上曾引发驱动级 GPU 缺页故障与 NPU 图损坏。
- **优先用短 NPU + OpenCL 冒烟测试做构建验证**：设备上的完整模型运行应在测试框架中以 `--thermal-ok` 标志门控。
- **Adreno + Hexagon 共存**：当 GPU 与 NPU 同时活跃时，确保权重缓冲区不在两个子系统中双重锁定。尽可能使用统一 ION/DMA-BUF 分配器；否则以缓存维护显式管理所有权转移。

## 目录结构

- `mobile-moe.zig`：模型加载（GGUF/IQ2_XXS、积极 RAM 驻留检测）、分词器、路由与冷专家的 CPU 参考代码、OpenCL 图调度、NPU 注意力图编排、异构专家延迟队列、会话生命周期、磁盘 KV 负载序列化。
- `mobile-moe-cli.zig`：命令行、linenoise 风格 REPL（通过 Termux 或 adb shell）、交互式会话处理、端侧基准测试模式。
- `mobile-moe-server.zig`：最小化 OpenAI 兼容 HTTP API（可选，用于本地 Agent 应用）、工作队列、流式传输、工具调用映射、磁盘 KV 缓存策略。
- `mobile-moe-gpu.zig`：OpenCL/Vulkan 上下文创建、Adreno 内存缓冲区管理、内核入队与事件同步的 Zig 包装层。
- `mobile-moe-npu.zig`：Hexagon NN / QNN SDK 的 Zig 绑定；管理 NPU 注意力图编译（固定分块大小）、静态子图卸载（embedding、固定投影）与 DMA 缓冲区编组。
- `kernels/opencl/*.cl`：FFN、MoE 投影、IQ2_XXS 反量化的 OpenCL 计算内核。**此处无注意力内核**——生产环境注意力仅由 NPU 处理。
- `kernels/vulkan/*.comp`：可选 Vulkan 计算着色器，用于未来 Adreno 驱动兼容性。
- `kernels/hexagon/*.c`：NPU 注意力分块、embedding 查找、固定形状投影的图定义。
- `tests/`：Zig `test` 块的单元测试，覆盖 GGUF 解析、路由逻辑、延迟队列、热节流与 NPU 注意力参考匹配。实时集成测试需要 K80 设备与小型 GGUF 夹具。
- `misc/`：忽略的笔记、量化校准数据、热分析日志与 NPU 图转储产物。

## 异构调度架构

| 模块 | 后端 | 理由 |
|--------|---------|-----------|
| 共享注意力 | NPU | 驻留内存权重允许 DMA 友好的 NPU 执行；固定 128-token 分块图避免动态重编译；NPU 擅长 FlashAttention 规则 GEMM + softmax 瓦片模式。 |
| 共享 FFN / 热点专家 | GPU（OpenCL） | 计算密集、形状规则；与注意力输出在 GPU 内存中共驻留；OpenCL 调度对每 token 专家数变化灵活。 |
| 路由层（TopK / Softmax / 门控） | CPU 大核（CPU 大核） | 输出索引需立即用于调度决策；CPU 延迟最低，避免专家分发前的 GPU↔NPU 同步停顿。 |
| 冷专家 | CPU 小核（A720 / A520） | 激活频率低；即使全量 RAM 驻留，在小核执行仍可节省 GPU 功耗与热预算。 |
| 延迟 / 跳过专家 | 回退偏置 | 若热/延迟预算超支，跳过延迟专家并向 logits 添加预校准共享偏置。 |
| Embedding / 输入输出投影 | NPU | 静态形状，完美适配 QNN 图编译；可能与第一块注意力融合。 |

## 内存策略

1. **启动检测**：查询 `sysconf(_SC_PHYS_PAGES)` 与 `/proc/meminfo`。若 `可用 >= 模型文件大小 × 1.2 + KV 缓存上限`，通过 `mmap(MAP_POPULATE)` 覆盖整个文件范围，随后 `madvise(MADV_WILLNEED)`。这会预热页缓存并消除未来缺页中断。
2. **受限回退**：若 RAM 不足，回退到按需分页。热点专家与当前层共享权重被 `mlock`；冷专家页设为 `MADV_RANDOM` 以减少预读浪费。
3. **权重所有权**：驻留后，权重以原始指针引用。GPU/NPU 缓冲区通过 ION 或显式拷贝创建为 RAM 驻留池的视图。除非格式要求（如 IQ2_XXS 反量化暂存区），否则不要维护第二份 GPU 独占权重拷贝。

## 测试

- 使用 `zig build` 进行原生主机构建验证（仅 CPU 参考路径，NPU 打桩）。
- 使用 `zig build -Dtarget=aarch64-linux-android -Dgpu=opencl -Dnpu=qnn` 进行设备构建。
- 当模型桩、OpenCL 与 QNN 桩可用时，使用 `zig build test` 进行单元/回归测试。
- 仅在有意测试 API 表面、热节流或 NPU 注意力精度时才使用实时设备测试；始终监控温度并提供 `--max-tokens` 限制。
