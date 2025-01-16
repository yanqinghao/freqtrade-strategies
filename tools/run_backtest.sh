#!/bin/bash

# 设置工作目录
SCRIPT_DIR="/home/yanqinghao/ft_userdata/freqtrade-strategies"
cd $SCRIPT_DIR

# 激活虚拟环境
source ../.venv/bin/activate

# 设置日志目录
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p $LOG_DIR

# 获取当前时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/backtest_$TIMESTAMP.log"

# 运行Python脚本并记录日志
echo "Starting backtesting task at $(date)" > $LOG_FILE
python tools/scheduler_backtesting.py >> $LOG_FILE 2>&1

# 记录完成时间
echo "Task completed at $(date)" >> $LOG_FILE

# 只保留最近7天的日志
find $LOG_DIR -name "backtest_*.log" -mtime +7 -delete