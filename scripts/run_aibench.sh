#!/usr/bin/env bash
# GPU/NPU 实测 - AI Benchmark APK 方案
# 当直接 OpenCL/Vulkan 代码测试因驱动兼容性失败时，使用此脚本通过 APK 实测 GPU/NPU 算力
#
# 用法: ./scripts/run_aibench.sh [device_serial]

set -euo pipefail

DEVICE="${1:-}"
ADB="adb ${DEVICE:+-s $DEVICE}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
APK_URL="https://download.ai-benchmark.com/s/ZknmZnGxfxJmfJE/download/AI_Benchmark_V6.0.6.apk"
APK_FILE="$SCRIPT_DIR/AI_Benchmark.apk"
PACKAGE="org.benchmark.demo"
RESULT_DIR="$SCRIPT_DIR/../results"

echo "========================================"
echo "  GPU/NPU 实测 - AI Benchmark APK 方案"
echo "========================================"
echo ""

# 检查 adb
if ! command -v adb &> /dev/null; then
    echo "错误: 未找到 adb 命令，请确保 Android SDK 已安装并加入 PATH"
    exit 1
fi

# 检查设备连接
echo "[1/4] 检查 adb 设备连接..."
if ! $ADB devices 2>/dev/null | grep -v "List" | grep -q "device$"; then
    echo "错误: 未检测到已连接的 adb 设备"
    exit 1
fi
DEVICE_SERIAL=$($ADB devices 2>/dev/null | grep -v "List" | grep "device$" | awk '{print $1}' | head -n 1)
echo "  设备序列号: $DEVICE_SERIAL"
echo ""

# 下载 APK
if [ ! -f "$APK_FILE" ]; then
    echo "[2/4] 下载 AI Benchmark V6 APK..."
    mkdir -p "$(dirname "$APK_FILE")"
    if command -v curl &> /dev/null; then
        if curl -L --progress-bar -o "$APK_FILE" "$APK_URL"; then
            echo "  下载成功: $APK_FILE"
        else
            echo "  下载失败，请手动从 https://ai-benchmark.com/download 下载 APK"
            echo "  并保存到: $APK_FILE"
            exit 1
        fi
    else
        echo "  未找到 curl，请手动下载 AI Benchmark APK 并保存到: $APK_FILE"
        exit 1
    fi
else
    echo "[2/4] 使用本地 APK: $APK_FILE"
fi
echo ""

# 安装 APK
echo "[3/4] 安装 AI Benchmark..."
if $ADB shell pm list packages 2>/dev/null | grep -q "$PACKAGE"; then
    echo "  AI Benchmark 已安装，执行覆盖安装..."
fi
if ! $ADB install -r -d "$APK_FILE" 2>/dev/null; then
    # 检测 MIUI/定制系统，提供针对性指导
    MANUFACTURER=$($ADB shell getprop ro.product.manufacturer 2>/dev/null | tr -d '\r')
    MIUI_VER=$($ADB shell getprop ro.miui.ui.version.name 2>/dev/null | tr -d '\r' || true)

    echo "  adb 直接安装失败 (INSTALL_FAILED_USER_RESTRICTED)"
    echo ""

    if [ "$MANUFACTURER" = "Xiaomi" ] || [ -n "$MIUI_VER" ]; then
        echo "  检测到小米/Redmi (MIUI $MIUI_VER) 设备，需要额外授权："
        echo ""
        echo "    1. 打开手机【设置】->【更多设置】->【开发者选项】"
        echo "    2. 找到【USB调试（安全设置）】并开启"
        echo "       （允许通过USB安装应用、模拟点击）"
        echo "    3. 如果仍失败，尝试关闭【启用MIUI优化】"
        echo "       （设置 -> 更多设置 -> 开发者选项 -> 启用MIUI优化）"
        echo ""
    else
        echo "  请检查以下授权："
        echo "    - 开发者选项 -> 允许通过USB安装应用"
        echo "    - 设置 -> 应用 -> 特殊权限 -> 安装未知应用 -> 允许"
        echo ""
    fi

    # 备用方案：推送 APK 到 Download，提示用户手动安装
    echo "  >>> 正在推送 APK 到设备 Download 目录..."
    $ADB shell mkdir -p /sdcard/Download 2>/dev/null || true
    $ADB push "$APK_FILE" /sdcard/Download/AI_Benchmark.apk >/dev/null 2>&1 && {
        echo "  APK 已推送到: 内部存储/Download/AI_Benchmark.apk"
        echo ""
        echo "  请手动安装："
        echo "    1. 打开手机【文件管理】->【Download】"
        echo "    2. 点击 AI_Benchmark.apk 安装"
        echo "    3. 安装完成后返回终端按回车继续..."
        echo ""
        read -r
    } || {
        echo "  推送失败，请手动复制 APK 到手机并安装"
        exit 1
    }
fi
echo "  安装成功"
echo ""

# 授予权限（用于保存结果和截图）
$ADB shell "pm grant $PACKAGE android.permission.READ_EXTERNAL_STORAGE 2>/dev/null || true"
$ADB shell "pm grant $PACKAGE android.permission.WRITE_EXTERNAL_STORAGE 2>/dev/null || true"

# 启动 AI Benchmark
echo "[4/4] 启动 AI Benchmark..."
$ADB shell "am start -n ${PACKAGE}/.MainActivity" || {
    echo "  启动失败"
    exit 1
}
echo "  启动成功"
echo ""

echo "========================================"
echo "AI Benchmark 已在设备上启动。"
echo ""
echo "请按以下步骤操作："
echo ""
echo "  1. 在设备上点击 'RUN BENCHMARK' 或 'START'"
echo "  2. 等待测试完成（约 2-5 分钟，期间设备会发热）"
echo "  3. 测试完成后，屏幕会显示各后端分数："
echo ""
echo "     CPU  INT8 / FP16  分数"
echo "     GPU  INT8 / FP16  分数"
echo "     NPU  INT8 / FP16  分数"
echo ""
echo "  4. 请手动记录或使用设备截图保存结果"
echo ""
echo "按回车键将自动截取当前屏幕..."
echo "========================================"
read -r

# 截屏
mkdir -p "$RESULT_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SCREENSHOT="$RESULT_DIR/aibench_result_${TIMESTAMP}.png"
$ADB shell screencap -p /sdcard/aibench_screenshot.png 2>/dev/null || true
if $ADB pull /sdcard/aibench_screenshot.png "$SCREENSHOT" >/dev/null 2>&1; then
    echo "截图已保存: $SCREENSHOT"
else
    echo "截图拉取失败，请手动在设备上截图"
fi

echo ""
echo "GPU/NPU APK 实测流程已完成。"
