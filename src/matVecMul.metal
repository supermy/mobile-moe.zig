// Metal 内核：矩阵-向量乘法
//
// Phase 1：基础实现，每个线程计算输出向量的一个元素
// 后续优化：使用 tile-based GEMM 和 shared memory

#include <metal_stdlib>
using namespace metal;

kernel void matVecMul(
    device float* out        [[buffer(0)]],
    device const float* mat  [[buffer(1)]],
    device const float* vec  [[buffer(2)]],
    constant uint& out_dim   [[buffer(3)]],
    constant uint& in_dim    [[buffer(4)]],
    uint gid                 [[thread_position_in_grid]]
) {
    if (gid >= out_dim) return;

    float sum = 0.0f;
    for (uint i = 0; i < in_dim; i++) {
        sum += mat[gid * in_dim + i] * vec[i];
    }
    out[gid] = sum;
}
