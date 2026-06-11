#!/usr/bin/env bash
# Android 全平台算力实测总控脚本
# 依次执行: CPU benchmark -> GPU/NPU 实测(APK方案) -> 硬件信息检测
#
# 用法: ./scripts/bench_all.sh [device_serial]

set -euo pipefail

DEVICE="${1:-}"
ADB="adb ${DEVICE:+-s $DEVICE}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULT_DIR="$SCRIPT_DIR/../results"

mkdir -p "$RESULT_DIR"

echo "========================================"
echo "  Android 全平台算力实测"
echo "========================================"
echo ""

# 检查 adb
if ! command -v adb &> /dev/null; then
    echo "错误: 未找到 adb 命令"
    exit 1
fi

# 检查设备
if ! $ADB devices 2>/dev/null | grep -v "List" | grep -q "device$"; then
    echo "错误: 未检测到已连接的 adb 设备"
    exit 1
fi

DEVICE_SERIAL=$($ADB devices 2>/dev/null | grep -v "List" | grep "device$" | awk '{print $1}' | head -n 1)
echo "目标设备: $DEVICE_SERIAL"
echo ""

# -------------------------------
# 1. CPU 算力实测
# -------------------------------
echo "[1/3] CPU 算力实测..."
CPU_BIN="/tmp/bench_cpu_android"

if command -v zig &> /dev/null; then
    echo "  编译 CPU benchmark..."
    zig build-exe "$SCRIPT_DIR/bench_cpu.zig" \
        -target aarch64-linux-android \
        -O ReleaseFast \
        -femit-bin="$CPU_BIN" 2>/dev/null || {
        echo "  编译失败，跳过 CPU 实测"
        CPU_BIN=""
    }
else
    echo "  未找到 zig，跳过 CPU 编译"
    CPU_BIN=""
fi

if [ -n "$CPU_BIN" ] && [ -f "$CPU_BIN" ]; then
    echo "  推送并运行 CPU benchmark..."
    $ADB push "$CPU_BIN" /data/local/tmp/bench_cpu >/dev/null 2>&1
    $ADB shell "chmod +x /data/local/tmp/bench_cpu && /data/local/tmp/bench_cpu" || {
        echo "  CPU benchmark 运行失败"
    }
fi
echo ""

# -------------------------------
# 2. GPU/NPU 实测
# -------------------------------
echo "[2/3] GPU/NPU 算力实测..."

# 2a. 尝试直接代码测试（OpenCL）
GPU_DIRECT_OK=false
GPU_BIN="/tmp/bench_gpu_android"

if command -v zig &> /dev/null; then
    echo "  尝试编译 GPU 直接测试 (OpenCL)..."
    if zig build-exe "$SCRIPT_DIR/bench_gpu.zig" \
        -target aarch64-linux-android \
        -O ReleaseFast \
        -femit-bin="$GPU_BIN" 2>/dev/null; then
        $ADB push "$GPU_BIN" /data/local/tmp/bench_gpu >/dev/null 2>&1
        echo "  运行 GPU 直接测试..."
        if $ADB shell "chmod +x /data/local/tmp/bench_gpu && /data/local/tmp/bench_gpu" >/dev/null 2>&1; then
            GPU_DIRECT_OK=true
        else
            echo "  GPU 直接测试失败 (OpenCL 驱动兼容性/权限问题)"
        fi
    else
        echo "  GPU 直接测试编译失败"
    fi
fi

# 2b. 如果直接测试失败，切换到 APK 方案
if [ "$GPU_DIRECT_OK" = false ]; then
    echo ""
    echo "  >>> 切换到 APK 方案进行 GPU/NPU 实测 <<<"
    echo ""
    bash "$SCRIPT_DIR/run_aibench.sh" "$DEVICE"
else
    echo "  GPU 直接测试成功，NPU 仍需 APK 方案补充"
    bash "$SCRIPT_DIR/run_aibench.sh" "$DEVICE"
fi
echo ""

# -------------------------------
# 3. 硬件信息汇总
# -------------------------------
echo "[3/3] 硬件信息检测..."
bash "$SCRIPT_DIR/detect_android.sh" "$DEVICE"
echo ""

echo "========================================"
echo "  全部测试流程已完成"
echo "========================================"
echo ""
echo "结果汇总:"
echo "  - CPU 实测:     见上方 benchmark 输出"
echo "  - GPU/NPU 实测: 见 AI Benchmark 应用结果或截图"
echo "  - 硬件信息:     见上方 detect_android 输出"
echo ""
echo "如需重新运行某项测试，可单独执行:"
echo "  ./scripts/bench_cpu.zig       (需 zig 编译)"
echo "  ./scripts/run_aibench.sh      (APK 方案)"
echo "  ./scripts/detect_android.sh   (硬件检测)"
