// Metal 内核：RMSNorm
//
// Phase 1：每个线程处理一个元素
// x_norm = x / sqrt(mean(x^2) + eps) * weight

#include <metal_stdlib>
using namespace metal;

kernel void rmsNorm(
    device float* out        [[buffer(0)]],
    device const float* x    [[buffer(1)]],
    device const float* weight [[buffer(2)]],
    constant uint& n         [[buffer(3)]],
    constant float& eps      [[buffer(4)]],
    uint gid                 [[thread_position_in_grid]]
) {
    if (gid >= n) return;

    // 每个线程单独计算 sum(x^2) —— 对于小维度 hidden_size 足够
    // 后续优化：使用 threadgroup 并行归约
    float ss = 0.0f;
    for (uint i = 0; i < n; i++) {
        ss += x[i] * x[i];
    }
    ss = ss / float(n) + eps;
    float inv = 1.0f / sqrt(ss);

    out[gid] = x[gid] * inv * weight[gid];
}
