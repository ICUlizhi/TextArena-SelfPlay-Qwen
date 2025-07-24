#!/bin/bash

# 确保在项目根目录执行
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." &> /dev/null && pwd )"
cd "$PROJECT_ROOT"

echo "=================================================================================="
echo "多GPU并行训练6种CoT模型"
echo "当前工作目录: $(pwd)"
echo "=================================================================================="

# 检查llamafactory-cli是否可用
if ! command -v llamafactory-cli &> /dev/null; then
    echo "❌ llamafactory-cli 未找到，尝试使用conda激活环境..."
    source ~/anaconda3/etc/profile.d/conda.sh
    conda activate tictactoe
    if ! command -v llamafactory-cli &> /dev/null; then
        echo "❌ llamafactory-cli 仍未找到，请确保LLaMA Factory已正确安装"
        exit 1
    fi
fi

echo "✅ llamafactory-cli 可用: $(which llamafactory-cli)"

# 检查可用GPU
nvidia-smi --list-gpus
echo ""

# 检查训练数据是否存在
echo "🔍 检查训练数据文件..."
data_files_found=0
for cot_type in tiny short medium long very_long ultra_long; do
    pattern="data/processed/long_cot_sft_data_${cot_type}_*.json"
    files=($(ls $pattern 2>/dev/null))
    if [ ${#files[@]} -gt 0 ]; then
        latest_file="${files[-1]}"  # 获取最新文件
        echo "✅ 找到 ${cot_type} 数据: $latest_file"
        data_files_found=$((data_files_found + 1))
    else
        echo "❌ 未找到 ${cot_type} 数据文件: $pattern"
    fi
done

if [ $data_files_found -eq 0 ]; then
    echo "❌ 未找到任何训练数据文件，请先运行数据生成脚本"
    exit 1
fi

echo "✅ 找到 $data_files_found 个训练数据文件"
echo ""

# 定义CoT类型和对应的GPU
declare -A COT_TYPES=(
    ["tiny"]="1"
    ["short"]="2" 
    ["medium"]="3"
    ["long"]="4"
    ["very_long"]="5"
    ["ultra_long"]="6"
)

# 创建必要目录
mkdir -p models
mkdir -p llama_factory_configs
mkdir -p logs

echo "🚀 开始在6个GPU上并行训练..."

# 为每个CoT类型创建训练配置并启动训练
for cot_type in "${!COT_TYPES[@]}"; do
    gpu_id=${COT_TYPES[$cot_type]}
    
    # 查找对应的数据文件
    pattern="data/processed/long_cot_sft_data_${cot_type}_*.json"
    files=($(ls $pattern 2>/dev/null))
    if [ ${#files[@]} -eq 0 ]; then
        echo "❌ 跳过 ${cot_type}：未找到数据文件 $pattern"
        continue
    fi
    
    # 获取最新的数据文件
    latest_file="${files[-1]}"
    full_path="$(realpath "$latest_file")"
    
    echo "📝 为 ${cot_type} 创建配置，使用数据文件: $latest_file"
    
    # 设置cutoff_len
    cutoff_len=1024
    if [[ "$cot_type" == "very_long" || "$cot_type" == "ultra_long" ]]; then
        cutoff_len=2048
    fi
    
    # 更新dataset_info.json中的文件路径
    python3 -c "
import json
import os

dataset_info_path = 'llama_factory_configs/dataset_info.json'
if os.path.exists(dataset_info_path):
    with open(dataset_info_path, 'r') as f:
        dataset_info = json.load(f)
else:
    dataset_info = {}

dataset_info['tictactoe_${cot_type}_cot'] = {
    'file_name': '$full_path',
    'formatting': 'sharegpt',
    'columns': {
        'messages': 'conversations'
    },
    'tags': {
        'role_tag': 'from',
        'content_tag': 'value',
        'user_tag': 'human',
        'assistant_tag': 'gpt'
    }
}

with open(dataset_info_path, 'w') as f:
    json.dump(dataset_info, f, indent=2)
"
    
    # 创建训练配置文件
    cat > "llama_factory_configs/qwen_${cot_type}_sft_config.yaml" << EOF
### model
model_name_or_path: ./qwen
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
dataset_dir: ./llama_factory_configs
template: qwen
cutoff_len: $cutoff_len
max_samples: 1000
overwrite_cache: true
preprocessing_num_workers: 4

### output
output_dir: ./models/qwen_${cot_type}_cot_lora
logging_steps: 10
save_steps: 100
plot_loss: true
overwrite_output_dir: true

### train
per_device_train_batch_size: 4
gradient_accumulation_steps: 2
learning_rate: 5.0e-4
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000

### eval
val_size: 0.1
per_device_eval_batch_size: 2
eval_strategy: steps
eval_steps: 100
EOF

    echo "📝 创建 ${cot_type} 配置文件完成"
    
    # 在指定GPU上启动训练（后台运行）
    echo "🚀 在GPU ${gpu_id} 上启动 ${cot_type} 模型训练..."
    
    # 使用nohup在后台运行，并重定向输出到日志文件
    nohup bash -c "
        source ~/anaconda3/etc/profile.d/conda.sh
        conda activate tictactoe
        export CUDA_VISIBLE_DEVICES=${gpu_id}
        cd '$PROJECT_ROOT'
        echo \"开始训练 ${cot_type} 模型 (GPU ${gpu_id}) - \$(date)\" >> logs/train_${cot_type}.log
        echo \"工作目录: \$(pwd)\" >> logs/train_${cot_type}.log
        echo \"数据文件: $latest_file\" >> logs/train_${cot_type}.log
        echo \"配置文件: llama_factory_configs/qwen_${cot_type}_sft_config.yaml\" >> logs/train_${cot_type}.log
        echo \"---\" >> logs/train_${cot_type}.log
        llamafactory-cli train llama_factory_configs/qwen_${cot_type}_sft_config.yaml >> logs/train_${cot_type}.log 2>&1
        exit_code=\$?
        echo \"${cot_type} 模型训练完成，退出码: \$exit_code - \$(date)\" >> logs/train_${cot_type}.log
    " &
    
    # 保存进程ID
    echo $! > "logs/train_${cot_type}.pid"
    
    echo "   ✅ ${cot_type} 训练已启动，PID: $(cat logs/train_${cot_type}.pid)"
    echo "   📄 日志文件: logs/train_${cot_type}.log"
    echo ""
    
    # 短暂延迟避免同时启动导致资源冲突
    sleep 5
done

echo "=================================================================================="
echo "所有训练任务已启动！"
echo "=================================================================================="
echo ""
echo "📊 训练状态监控："
echo "   - 查看所有训练进程: ps aux | grep llamafactory-cli"
echo "   - 查看GPU使用情况: nvidia-smi"
echo "   - 查看训练日志: tail -f logs/train_<cot_type>.log"
echo ""

echo "🔍 当前训练进程："
for cot_type in "${!COT_TYPES[@]}"; do
    gpu_id=${COT_TYPES[$cot_type]}
    if [ -f "logs/train_${cot_type}.pid" ]; then
        pid=$(cat "logs/train_${cot_type}.pid")
        if ps -p $pid > /dev/null; then
            echo "   ✅ ${cot_type} (GPU ${gpu_id}): 运行中 (PID: $pid)"
        else
            echo "   ❌ ${cot_type} (GPU ${gpu_id}): 已停止 (PID: $pid)"
        fi
    fi
done

echo ""
echo "📝 监控脚本已创建："
echo "   - 运行 './monitor_training.sh' 来监控训练进度"
echo "   - 运行 './stop_all_training.sh' 来停止所有训练"

# 创建监控脚本
cat > monitor_training.sh << 'EOF'
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
EOF

# 创建停止脚本
cat > stop_all_training.sh << 'EOF'
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
EOF

chmod +x monitor_training.sh
chmod +x stop_all_training.sh

echo "📋 使用说明："
echo "   - 监控训练: ./monitor_training.sh"
echo "   - 停止训练: ./stop_all_training.sh"
echo "   - 查看日志: tail -f logs/train_<type>.log"
echo ""
echo "🎯 训练完成后，模型将保存在 models/ 目录下"
