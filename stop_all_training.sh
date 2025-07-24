#!/bin/bash
echo "停止所有训练进程..."

declare -A COT_TYPES=(
    ["tiny"]="1"
    ["short"]="2" 
    ["medium"]="3"
    ["long"]="4"
    ["very_long"]="5"
    ["ultra_long"]="6"
)

for cot_type in "${!COT_TYPES[@]}"; do
    if [ -f "logs/train_${cot_type}.pid" ]; then
        pid=$(cat "logs/train_${cot_type}.pid")
        if ps -p $pid > /dev/null; then
            echo "停止 ${cot_type} 训练 (PID: $pid)..."
            kill $pid
        fi
        rm -f "logs/train_${cot_type}.pid"
    fi
done

echo "所有训练进程已停止。"
