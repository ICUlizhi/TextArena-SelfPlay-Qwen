#!/bin/bash

# 简化版批量训练脚本
# 训练六种CoT长度的模型

set -e

echo "=========================================="
echo "CoT模型批量训练脚本"
echo "=========================================="

# 设置环境变量
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export CUDA_VISIBLE_DEVICES=7

# 检查基础模型
BASE_MODEL="/mnt/cvda/cvda_avatar/1/textarena-selfplay-qwen/qwen"
if [ ! -d "$BASE_MODEL" ]; then
    echo "❌ 基础模型不存在: $BASE_MODEL"
    exit 1
fi

# 检查数据文件
DATA_DIR="data/processed"
echo "🔍 检查训练数据文件..."

COT_TYPES=("tiny" "short" "medium" "long" "very_long" "ultra_long")
AVAILABLE_DATA=()

for cot_type in "${COT_TYPES[@]}"; do
    # 找到最新的数据文件
    latest_file=$(ls -t ${DATA_DIR}/long_cot_sft_data_${cot_type}_*.json 2>/dev/null | head -n1)
    if [ -n "$latest_file" ]; then
        echo "✅ $cot_type: $latest_file"
        AVAILABLE_DATA+=("$cot_type:$latest_file")
    else
        echo "❌ $cot_type: 未找到数据文件"
    fi
done

if [ ${#AVAILABLE_DATA[@]} -eq 0 ]; then
    echo "❌ 未找到任何训练数据文件!"
    echo "请先运行: python generate_all_cot_lengths.py"
    exit 1
fi

echo "📊 找到 ${#AVAILABLE_DATA[@]} 个数据文件，开始训练..."

# 创建必要目录
mkdir -p models
mkdir -p llama_factory_configs

# 训练每个模型
for data_entry in "${AVAILABLE_DATA[@]}"; do
    IFS=':' read -r cot_type data_file <<< "$data_entry"
    
    echo ""
    echo "=========================================="
    echo "训练 $cot_type CoT 模型"
    echo "=========================================="
    echo "数据文件: $data_file"
    
    # 创建训练配置
    output_dir="$(pwd)/models/qwen_${cot_type}_cot_lora"
    cutoff_len=1024
    if [[ "$cot_type" == "very_long" || "$cot_type" == "ultra_long" ]]; then
        cutoff_len=2048
    fi
    
    cat > "llama_factory_configs/qwen_${cot_type}_sft_config.yaml" << EOF
### model
model_name_or_path: $BASE_MODEL
model_revision: main

### method
stage: sft
do_train: true
finetuning_type: lora
lora_target: all
lora_rank: 8
lora_alpha: 16
lora_dropout: 0.1

### dataset
dataset: tictactoe_${cot_type}_cot
dataset_dir: $(pwd)/llama_factory_configs
template: qwen
cutoff_len: $cutoff_len
max_samples: 100
overwrite_cache: true
preprocessing_num_workers: 16

### output
output_dir: $output_dir
logging_steps: 5
save_steps: 50
plot_loss: true
overwrite_output_dir: true

### train
per_device_train_batch_size: 2
gradient_accumulation_steps: 4
learning_rate: 5.0e-4
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000

### eval
val_size: 0.1
per_device_eval_batch_size: 1
eval_strategy: steps
eval_steps: 50
EOF

    echo "📝 配置文件已创建"
    echo "🚀 开始训练 $cot_type 模型..."
    
    # 执行训练
    if llamafactory-cli train "llama_factory_configs/qwen_${cot_type}_sft_config.yaml"; then
        echo "✅ $cot_type 模型训练完成!"
        echo "   模型保存在: $output_dir"
    else
        echo "❌ $cot_type 模型训练失败!"
    fi
    
done

echo ""
echo "=========================================="
echo "批量训练完成!"
echo "=========================================="
echo "已训练的模型："
ls -la models/ | grep qwen_.*_cot_lora
echo ""
echo "下一步可以运行评估脚本:"
echo "python scripts/evaluate_all_cot_models.py"
