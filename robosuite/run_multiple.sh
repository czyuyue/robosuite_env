#!/bin/bash

# 脚本：运行50次run.py
# 用法：bash run_multiple.sh

# 设置运行次数
TOTAL_RUNS=50

# 设置工作目录
WORK_DIR="/home/yunzhe/zzzzzworkspaceyy/robosuite_data/robosuite"

# 创建日志目录
LOG_DIR="$WORK_DIR/logs"
mkdir -p "$LOG_DIR"

# 创建结果统计文件
STATS_FILE="$LOG_DIR/run_stats.txt"
echo "Run Statistics - $(date)" > "$STATS_FILE"
echo "===========================================" >> "$STATS_FILE"

# 计数器
SUCCESS_COUNT=0
FAILED_COUNT=0

echo "开始运行 $TOTAL_RUNS 次 run.py..."
echo "工作目录: $WORK_DIR"
echo "日志目录: $LOG_DIR"
echo ""

# 进入工作目录
cd "$WORK_DIR"

# 循环运行
for i in $(seq 1 $TOTAL_RUNS); do
    echo "=========================================="
    echo "运行第 $i/$TOTAL_RUNS 次 ($(date))"
    echo "=========================================="
    
    # 创建当前运行的日志文件
    LOG_FILE="$LOG_DIR/run_${i}.log"
    
    # 运行python脚本并记录日志
    if python run.py > "$LOG_FILE" 2>&1; then
        echo "✅ 第 $i 次运行成功"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        echo "Run $i: SUCCESS" >> "$STATS_FILE"
    else
        echo "❌ 第 $i 次运行失败，查看日志: $LOG_FILE"
        FAILED_COUNT=$((FAILED_COUNT + 1))
        echo "Run $i: FAILED" >> "$STATS_FILE"
        
        # 显示错误信息的最后几行
        echo "错误信息："
        tail -10 "$LOG_FILE"
    fi
    
    # 显示进度
    PROGRESS=$((i * 100 / TOTAL_RUNS))
    echo "进度: $PROGRESS% ($i/$TOTAL_RUNS)"
    echo ""
    
    # 短暂暂停，避免系统过载
    sleep 1
done

# 最终统计
echo "=========================================="
echo "运行完成！最终统计："
echo "总运行次数: $TOTAL_RUNS"
echo "成功次数: $SUCCESS_COUNT"
echo "失败次数: $FAILED_COUNT"
echo "成功率: $(echo "scale=2; $SUCCESS_COUNT * 100 / $TOTAL_RUNS" | bc)%"
echo "=========================================="

# 写入统计文件
echo "" >> "$STATS_FILE"
echo "Final Statistics:" >> "$STATS_FILE"
echo "Total runs: $TOTAL_RUNS" >> "$STATS_FILE"
echo "Successful: $SUCCESS_COUNT" >> "$STATS_FILE"
echo "Failed: $FAILED_COUNT" >> "$STATS_FILE"
echo "Success rate: $(echo "scale=2; $SUCCESS_COUNT * 100 / $TOTAL_RUNS" | bc)%" >> "$STATS_FILE"

echo "详细日志保存在: $LOG_DIR"
echo "统计信息保存在: $STATS_FILE" 