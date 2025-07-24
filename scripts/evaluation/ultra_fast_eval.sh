#!/bin/bash
# Ultra Fast 全模型评估 - 8GPU极速版本
# 目标：最快完成所有7个模型的100个测试样例

echo "🚀 Ultra Fast 8GPU 全模型评估"
echo "🎯 目标：100个测试样例 × 7个模型 = 700个评估任务"
echo "⚡ 策略：8GPU并行，最大化吞吐量"
echo "⏱️  预计时间：~10-15分钟"
echo ""

# 记录开始时间
start_time=$(date +%s)

# 进入评估目录
cd "$(dirname "$0")"

echo "🚀 启动Ultra Turbo评估器..."
python ultra_turbo_evaluator.py

# 计算总耗时
end_time=$(date +%s)
duration=$((end_time - start_time))
minutes=$((duration / 60))
seconds=$((duration % 60))

echo ""
echo "🎉 Ultra Fast评估完成！"
echo "⏱️  总耗时: ${minutes}分${seconds}秒"
echo "🚀 完成了700个模型-案例评估组合"
echo ""

# 显示最新结果
echo "📁 最新结果文件:"
ls -la ../evaluation_results/ultra_turbo_evaluation_*.json | tail -1

echo ""
echo "📊 结果概览:"
python -c "
import json
import glob
import os

# 找到最新的结果文件
files = glob.glob('../evaluation_results/ultra_turbo_evaluation_*.json')
if files:
    latest_file = max(files, key=os.path.getctime)
    with open(latest_file, 'r') as f:
        data = json.load(f)
    
    print(f'📈 整体速度: {data[\"evaluation_info\"][\"overall_speed\"]:.1f} 案例/秒')
    print(f'✅ 成功评估: {data[\"evaluation_info\"][\"successful_evaluations\"]}/{data[\"evaluation_info\"][\"total_models\"]} 模型')
    
    # 显示性能排名
    successful = [r for r in data['results'] if r.get('success', False)]
    if successful:
        sorted_results = sorted(successful, key=lambda x: x.get('accuracy', 0), reverse=True)
        print(f'🏆 最佳模型: {sorted_results[0].get(\"model_name\", \"Unknown\")} ({sorted_results[0].get(\"accuracy\", 0):.1f}%)')
        if len(sorted_results) > 1:
            print(f'📉 最差模型: {sorted_results[-1].get(\"model_name\", \"Unknown\")} ({sorted_results[-1].get(\"accuracy\", 0):.1f}%)')
"
