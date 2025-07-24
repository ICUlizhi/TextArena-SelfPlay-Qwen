#!/bin/bash

cd scripts/data_generation


echo "使用串行模式生成6种不同长度的CoT数据..."
python generate_all_cot_lengths_serial.py
fi生成到评估的完整流程

echo "🎯 TextArena SelfPlay Qwen - 快速入门"
echo "完整的井字棋CoT训练和评估流程"
echo ""

# 记录开始时间
start_time=$(date +%s)

cd scripts/data_generation

echo "✅ CoT数据生成完成"
echo ""

# 步骤2: 生成测试集
echo "📋 步骤2: 生成测试集"
cd ../../evaluation
python generate_test_set.py --num_cases 100 --output_file ../data/processed/tictactoe_test_set_100.json
echo "✅ 测试集生成完成"
echo ""

# 步骤3: 训练模型 (可选，耗时较长)
echo "🏋️ 步骤3: 训练模型"
echo "⚠️  注意: 训练需要大量时间和GPU资源"

echo "🚀 启动多GPU并行训练..."
cd ../scripts/training
bash train_all_models_parallel.sh
echo "✅ 训练已启动，请使用 'bash monitor_training.sh' 监控进度"

#等待15分钟
echo "⏳ 等待15分钟以确保训练完成..."
sleep 900

# 步骤4: 模型评估
echo "🚀 步骤4: 模型评估"
echo "启动Ultra Fast 8GPU并行评估..."
cd ../evaluation
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python run_full_evaluation.py --num-cases 
100 --parallel

echo "✅ 评估完成"
echo ""

# 计算总耗时
end_time=$(date +%s)
duration=$((end_time - start_time))
minutes=$((duration / 60))
seconds=$((duration % 60))

echo "🎉 快速入门流程完成!"
echo "⏱️  总耗时: ${minutes}分${seconds}秒"
echo ""
echo "📁 生成的文件:"
echo "  - CoT训练数据: data/processed/tictactoe_*_cot_data.json"
echo "  - 测试集: data/processed/tictactoe_test_set_100.json"
echo "  - 评估结果: evaluation_results/"
echo ""
echo "📖 更多信息请查看:"
echo "  - README.md - 项目文档"
echo "  - docs/SCRIPTS_GUIDE.md - 脚本使用指南"
