#!/bin/bash

# 设置环境变量
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export CUDA_VISIBLE_DEVICES=7

# 创建输出目录
mkdir -p models/qwen_sft_lora

echo "开始训练 Qwen 模型..."
echo "数据集: TicTacToe Long CoT (100 samples)"
echo "模型: Qwen-2.5-0.5B"
echo "方法: LoRA SFT"
echo "训练配置: llama_factory_configs/qwen_sft_config.yaml"

# 启动训练
llamafactory-cli train llama_factory_configs/qwen_sft_config.yaml

echo "训练完成！"
echo "模型保存在: models/qwen_sft_lora/"
