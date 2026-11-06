#!/bin/bash
GPU_ID=$1
save_path=$2
CSV_FILE=$save_path

# 创建 CSV 文件，并写入表头
echo "Time,Memory_Used_MB,utilization,power" > $CSV_FILE

# 每隔2秒采集一次显存使用情况，直到train.sh结束
while true; do
    # 获取当前系统时间
    current_time=$(date "+%Y-%m-%d/%H:%M:%S")
    
    # 获取当前 GPU 显存使用情况（以 MB 为单位）
    memory_used=$(nvidia-smi --id=$GPU_ID --query-gpu=memory.used --format=csv,noheader,nounits)
    utilization=$(nvidia-smi --id=$GPU_ID --query-gpu=utilization.gpu --format=csv,noheader,nounits)
    power=$(nvidia-smi --id=$GPU_ID --query-gpu=power.draw --format=csv,noheader,nounits)
    # 将当前时间和显存使用情况写入 CSV 文件
    echo "$current_time,$memory_used,$utilization,$power" >> $CSV_FILE

    # 每隔2秒记录一次
    sleep 2
done