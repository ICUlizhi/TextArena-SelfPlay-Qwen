#!/bin/bash
# GPU资源监控脚本 - 实时显示8GPU使用情况

echo "🎮 8GPU资源监控器"
echo "按 Ctrl+C 停止监控"
echo ""

while true; do
    clear
    echo "🎮 GPU使用状况 - $(date '+%H:%M:%S')"
    echo "=" * 60
    
    # 显示GPU使用情况
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits | \
        while IFS=',' read -r index name util mem_used mem_total temp; do
            # 计算内存使用百分比
            mem_percent=$((mem_used * 100 / mem_total))
            
            # 根据使用率设置状态
            if [ "$util" -gt 80 ]; then
                status="🔥 忙碌"
            elif [ "$util" -gt 40 ]; then
                status="⚡ 工作"
            elif [ "$util" -gt 10 ]; then
                status="💤 轻载"
            else
                status="😴 空闲"
            fi
            
            printf "GPU%s: %s | %s%% GPU | %s%% 内存 (%sMB/%sMB) | %s°C | %s\n" \
                "$index" "$status" "$util" "$mem_percent" "$mem_used" "$mem_total" "$temp" "${name:0:20}"
        done
    else
        echo "❌ nvidia-smi 未找到"
    fi
    
    echo ""
    echo "💡 提示："
    echo "  - 🔥 忙碌: GPU使用率 > 80%"
    echo "  - ⚡ 工作: GPU使用率 40-80%"
    echo "  - 💤 轻载: GPU使用率 10-40%"
    echo "  - 😴 空闲: GPU使用率 < 10%"
    
    sleep 2
done
