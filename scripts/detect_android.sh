#!/usr/bin/env bash
# Android 设备 CPU/GPU/NPU 硬件检测脚本
# 用法: ./detect_android.sh [device_serial]
# 复用: adb 连接状态下直接运行，输出格式化硬件信息与理论算力估算

set -euo pipefail

DEVICE="${1:-}"
ADB="adb ${DEVICE:+-s $DEVICE}"

echo "============================================"
echo "   Android 设备硬件检测脚本"
echo "============================================"
echo ""

# 1. 检查设备连接
echo "[1/5] 检查 adb 设备连接..."
DEVICE_LIST=$($ADB devices 2>/dev/null | grep -v "List" | grep "device$" | awk '{print $1}')
if [ -z "$DEVICE_LIST" ]; then
    echo "错误: 未检测到已连接的 adb 设备"
    exit 1
fi

if [ -z "$DEVICE" ]; then
    DEVICE=$(echo "$DEVICE_LIST" | head -n 1)
    ADB="adb -s $DEVICE"
fi

echo "  设备序列号: $DEVICE"
echo ""

# 2. 基础属性
echo "[2/5] 读取设备基础属性..."
HARDWARE=$($ADB shell getprop ro.hardware 2>/dev/null | tr -d '\r')
PLATFORM=$($ADB shell getprop ro.board.platform 2>/dev/null | tr -d '\r')
BOARD=$($ADB shell getprop ro.product.board 2>/dev/null | tr -d '\r')
MODEL=$($ADB shell getprop ro.product.model 2>/dev/null | tr -d '\r')
MANUFACTURER=$($ADB shell getprop ro.product.manufacturer 2>/dev/null | tr -d '\r')

echo "  厂商:       $MANUFACTURER"
echo "  型号:       $MODEL"
echo "  硬件:       $HARDWARE"
echo "  平台:       $PLATFORM"
echo "  主板:       $BOARD"
echo ""

# 3. CPU 检测
echo "[3/5] CPU 检测..."
CPU_COUNT=$($ADB shell nproc 2>/dev/null | tr -d '\r')
CPU_ARCH=$($ADB shell "cat /proc/cpuinfo | grep -m1 'CPU architecture' | awk '{print \$3}'" 2>/dev/null | tr -d '\r')

# 读取所有核心频率
FREQS=$($ADB shell "cat /sys/devices/system/cpu/cpu*/cpufreq/cpuinfo_max_freq 2>/dev/null" 2>/dev/null | tr -d '\r' | sort -n | uniq -c | sort -rn | sed 's/^[ ]*//' | awk '{printf "%sMHz(x%s) ", $2/1000, $1}')

# 尝试识别核心类型 (通过 CPU part)
CPU_PARTS=$($ADB shell "cat /proc/cpuinfo | grep 'CPU part' | awk '{print \$3}' | sort | uniq -c | sort -rn" 2>/dev/null | tr -d '\r' | sed 's/^[ ]*//')

echo "  架构:       ARMv$CPU_ARCH"
echo "  核心数:     $CPU_COUNT"
echo "  频率分布:   $FREQS"

# 核心类型映射 (ARM CPU part ID)
CORE_TYPES=""
while IFS= read -r line; do
    count=$(echo "$line" | awk '{print $1}')
    part=$(echo "$line" | awk '{print $2}')
    case "$part" in
        0xd01|0xd02|0xd03|0xd04|0xd05) name="Cortex-A5x" ;;
        0xd06|0xd07|0xd08|0xd09) name="Cortex-A7x" ;;
        0xd0a) name="Cortex-A78" ;;
        0xd0b) name="Cortex-A710" ;;
        0xd0c) name="Cortex-A715" ;;
        0xd0d) name="Cortex-A720" ;;
        0xd0e) name="Cortex-A725" ;;
        0xd13) name="Cortex-A53" ;;
        0xd14) name="Cortex-A55" ;;
        0xd15) name="Cortex-A520" ;;
        0xd16) name="Cortex-A510" ;;
        0xd40) name="Cortex-A76" ;;
        0xd41) name="Cortex-A77" ;;
        0xd44) name="Cortex-X1" ;;
        0xd45) name="Cortex-X2" ;;
        0xd46) name="Cortex-X3" ;;
        0xd47) name="Cortex-X4" ;;
        0xd48) name="Cortex-X925" ;;
        0xd4f) name="Cortex-A510(v2)" ;;
        *) name="Unknown($part)" ;;
    esac
    CORE_TYPES="$CORE_TYPES$name(x$count) "
done <<< "$CPU_PARTS"

if [ -n "$CORE_TYPES" ] && [ "$CORE_TYPES" != "Unknown(:)(x8) " ]; then
    echo "  核心类型:   $CORE_TYPES"
fi

# CPU 温度
CPU_TEMP=$($ADB shell "cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null | head -n 1" 2>/dev/null | tr -d '\r')
if [ -n "$CPU_TEMP" ] && [ "$CPU_TEMP" != "" ]; then
    # 有些设备温度单位是千分之一摄氏度，有些是摄氏度
    if [ "$CPU_TEMP" -gt 10000 ] 2>/dev/null; then
        CPU_TEMP_C=$(echo "scale=1; $CPU_TEMP / 1000" | bc 2>/dev/null || echo "$((CPU_TEMP / 1000))")
    else
        CPU_TEMP_C=$CPU_TEMP
    fi
    echo "  温度:       ${CPU_TEMP_C}°C"
fi
echo ""

# 4. GPU 检测
echo "[4/5] GPU 检测..."
GLES_INFO=$($ADB shell "dumpsys SurfaceFlinger 2>/dev/null | grep GLES" 2>/dev/null | tr -d '\r')
EGL_DRIVER=$($ADB shell getprop ro.hardware.egl 2>/dev/null | tr -d '\r')

if [ -n "$GLES_INFO" ]; then
    # 提取 GPU 型号
    GPU_NAME=$(echo "$GLES_INFO" | sed -n 's/.*GLES: .*Adreno (TM) \([0-9]*\).*/Adreno \1/p')
    if [ -z "$GPU_NAME" ]; then
        GPU_NAME=$(echo "$GLES_INFO" | sed -n 's/.*GLES: .*Mali-\([^,]*\).*/Mali-\1/p')
    fi
    if [ -z "$GPU_NAME" ]; then
        GPU_NAME=$(echo "$GLES_INFO" | sed -n 's/.*GLES: \([^,]*\).*/\1/p')
    fi
    echo "  型号:       $GPU_NAME"
    echo "  GLES 信息:  $GLES_INFO"
else
    echo "  型号:       $EGL_DRIVER (通过 egl driver)"
fi

# GPU 温度
GPU_TEMP_ZONES=$($ADB shell "cat /sys/class/thermal/thermal_zone*/type 2>/dev/null | grep -n gpu | head -n 1 | cut -d: -f1" 2>/dev/null | tr -d '\r')
if [ -n "$GPU_TEMP_ZONES" ]; then
    GPU_TEMP=$($ADB shell "cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null | sed -n '${GPU_TEMP_ZONES}p'" 2>/dev/null | tr -d '\r')
    if [ -n "$GPU_TEMP" ] && [ "$GPU_TEMP" != "" ]; then
        if [ "$GPU_TEMP" -gt 10000 ] 2>/dev/null; then
            GPU_TEMP_C=$(echo "scale=1; $GPU_TEMP / 1000" | bc 2>/dev/null || echo "$((GPU_TEMP / 1000))")
        else
            GPU_TEMP_C=$GPU_TEMP
        fi
        echo "  温度:       ${GPU_TEMP_C}°C"
    fi
fi
echo ""

# 5. NPU 检测
echo "[5/5] NPU 检测..."
NPU_ENABLED=$($ADB shell getprop ro.boot.vendor.qspa.npu 2>/dev/null | tr -d '\r')
NPU_PROP=$($ADB shell "getprop | grep -i npu 2>/dev/null | grep -v 'stagefright\|input\|output'" 2>/dev/null | tr -d '\r')

case "$PLATFORM" in
    pineapple|taro|kalama|lahaina|shima|yupik)
        CHIP_VENDOR="Qualcomm"
        NPU_NAME="Hexagon DSP/NPU"
        ;;
    mt*|k69*)
        CHIP_VENDOR="MediaTek"
        NPU_NAME="APU (AI Processing Unit)"
        ;;
    kirin*|kirin*)
        CHIP_VENDOR="Huawei"
        NPU_NAME="DaVinci NPU"
        ;;
    exynos*)
        CHIP_VENDOR="Samsung"
        NPU_NAME="NPU (Samsung)"
        ;;
    *)
        CHIP_VENDOR="Unknown"
        NPU_NAME="Unknown NPU"
        ;;
esac

if [ "$NPU_ENABLED" = "enabled" ] || [ -n "$NPU_PROP" ]; then
    echo "  状态:       可用"
    echo "  厂商:       $CHIP_VENDOR"
    echo "  型号:       $NPU_NAME"
    if [ -n "$NPU_PROP" ]; then
        echo "  NPU 属性:"
        echo "$NPU_PROP" | sed 's/^/    /'
    fi
else
    echo "  状态:       未检测到或不可用"
    echo "  厂商:       $CHIP_VENDOR"
fi

# 检查 NPU/DSP 设备节点
DSP_DEVICES=$($ADB shell "ls /dev 2>/dev/null | grep -E 'adsprpc|cdsp|remoteproc'" 2>/dev/null | tr -d '\r')
if [ -n "$DSP_DEVICES" ]; then
    echo "  DSP/NPU 节点:"
    echo "$DSP_DEVICES" | sed 's#^#    /dev/#'
fi
echo ""

# 6. 理论算力估算
echo "============================================"
echo "   理论 AI 算力估算"
echo "============================================"
echo ""

case "$PLATFORM" in
    pineapple)
        # Snapdragon 8 Gen 3
        CPU_TOPS="~50"
        GPU_TOPS="~45"
        NPU_TOPS="~45"
        ;;
    taro)
        # Snapdragon 8 Gen 2
        CPU_TOPS="~40"
        GPU_TOPS="~35"
        NPU_TOPS="~35"
        ;;
    kalama)
        # Snapdragon 8 Gen 1 / 8+ Gen 1
        CPU_TOPS="~30"
        GPU_TOPS="~25"
        NPU_TOPS="~25"
        ;;
    lahaina)
        # Snapdragon 888
        CPU_TOPS="~26"
        GPU_TOPS="~20"
        NPU_TOPS="~20"
        ;;
    mt689*|mt698*|mt699*)
        # MediaTek Dimensity 9000/9200/9300 系列
        CPU_TOPS="~30-50"
        GPU_TOPS="~25-40"
        NPU_TOPS="~25-45"
        ;;
    *)
        CPU_TOPS="未知"
        GPU_TOPS="未知"
        NPU_TOPS="未知"
        ;;
esac

echo "  CPU INT8:   $CPU_TOPS TOPS"
echo "  GPU INT8:   $GPU_TOPS TOPS"
echo "  NPU INT8:   $NPU_TOPS TOPS"
if [ "$CPU_TOPS" != "未知" ]; then
    echo ""
    echo "  合计峰值:   $(echo "$CPU_TOPS + $GPU_TOPS + $NPU_TOPS" | sed 's/~//g' | bc 2>/dev/null || echo "N/A") INT8 TOPS"
    echo "  (实际 AI 推理通常优先使用 NPU/GPU)"
fi

echo ""
echo "============================================"
echo "   检测完成"
echo "============================================"
