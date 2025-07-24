#!/bin/bash
echo "=================================================================================="
echo "训练进度监控"
echo "=================================================================================="

declare -A COT_TYPES=(
    ["tiny"]="1"
    ["short"]="2" 
    ["medium"]="3"
    ["long"]="4"
    ["very_long"]="5"
    ["ultra_long"]="6"
)

while true; do
    clear
    echo "🔍 训练状态 $(date)"
    echo "--------------------------------------------------------------------------------"
    
    for cot_type in "${!COT_TYPES[@]}"; do
        gpu_id=${COT_TYPES[$cot_type]}
        if [ -f "logs/train_${cot_type}.pid" ]; then
            pid=$(cat "logs/train_${cot_type}.pid")
            if ps -p $pid > /dev/null; then
                echo "✅ ${cot_type} (GPU ${gpu_id}): 运行中 (PID: $pid)"
                if [ -f "logs/train_${cot_type}.log" ]; then
                    last_line=$(tail -1 "logs/train_${cot_type}.log" 2>/dev/null)
                    echo "   最新: $last_line"
                fi
            else
                echo "❌ ${cot_type} (GPU ${gpu_id}): 已停止 (PID: $pid)"
            fi
        else
            echo "⚪ ${cot_type} (GPU ${gpu_id}): 未启动"
        fi
        echo ""
    done
    
    echo "--------------------------------------------------------------------------------"
    echo "🖥️  GPU使用情况:"
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | while read line; do
        echo "   $line"
    done
    
    echo ""
    echo "按 Ctrl+C 退出监控"
    sleep 10
done
