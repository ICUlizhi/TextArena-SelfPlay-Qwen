# TextArena SelfPlay Qwen - 井字棋CoT训练项目

一个完整的基于Qwen模型的井字棋思维链(Chain-of-Thought)训练和评估系统。该项目实现了6种不同长度的CoT训练模式，支持多GPU并行训练和高速评估。

- 目前脚本bug还比较多, 复现不方便, readme是用ai写的

## 🎯 项目概述

本项目通过自对弈生成井字棋数据，训练6种不同CoT长度的模型：
- **Tiny CoT** (80-120字符): 超短推理
- **Short CoT** (150-250字符): 短推理  
- **Medium CoT** (300-500字符): 中等推理
- **Long CoT** (600-1000字符): 长推理
- **Very Long CoT** (1500-2500字符): 很长推理
- **Ultra Long CoT** (3000-5000字符): 超长推理

## 📋 项目结构

```
textarena-selfplay-qwen/
├── src/                          # 核心源码
│   ├── qwen_agent.py            # Qwen智能体实现
│   └── tictactoe_game.py        # 井字棋游戏逻辑
├── evaluation/                   # 评估模块
│   ├── model_evaluator.py       # 统一模型评估器
│   └── generate_test_set.py     # 测试集生成
├── scripts/                     # 脚本工具
│   ├── training/               # 训练脚本
│   ├── evaluation/             # 评估脚本  
│   ├── data_generation/        # 数据生成脚本
│   └── utils/                  # 工具脚本
├── configs/                     # 配置文件
├── llama_factory_configs/       # LlamaFactory训练配置
├── data/                        # 数据目录
├── models/                      # 训练后的模型
└── evaluation_results/          # 评估结果
```

## 🚀 快速开始

### 环境配置

```bash
# 克隆项目
git clone <repository-url>
cd textarena-selfplay-qwen

# 安装依赖
pip install -r requirements.txt

# 安装LlamaFactory (用于训练)
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e .
```

## 📊 使用指南

### 1. 测试集生成

生成用于模型评估的标准测试集：

```bash
# 生成100个测试案例的标准测试集
cd evaluation
python generate_test_set.py --num_cases 100 --output_file ../data/processed/tictactoe_test_set_100.json

# 生成不同难度的测试集
python generate_test_set.py --num_cases 200 --difficulty_distribution easy:50,medium:30,hard:20
```

**参数说明：**
- `--num_cases`: 生成的测试案例数量
- `--output_file`: 输出文件路径
- `--difficulty_distribution`: 难度分布 (easy/medium/hard)
- `--seed`: 随机种子，确保可重现

### 2. CoT数据生成

生成6种不同长度的CoT训练数据：

```bash
# 生成所有CoT长度的训练数据
cd scripts/data_generation
python generate_all_cot_lengths.py

# 自定义参数生成
python generate_all_cot_lengths.py --num_games 2000 --base_output_dir ../../data/processed
```

**生成的数据类型：**
- `tictactoe_tiny_cot_data.json` - 超短推理数据
- `tictactoe_short_cot_data.json` - 短推理数据  
- `tictactoe_medium_cot_data.json` - 中等推理数据
- `tictactoe_long_cot_data.json` - 长推理数据
- `tictactoe_very_long_cot_data.json` - 很长推理数据
- `tictactoe_ultra_long_cot_data.json` - 超长推理数据

**参数说明：**
- `--num_games`: 每种CoT类型生成的游戏数量
- `--base_output_dir`: 输出目录
- `--device`: 指定GPU设备

### 3. 模型训练

#### 单模型训练

使用LlamaFactory进行训练：

```bash
# 进入LlamaFactory目录
cd LLaMA-Factory

# 训练Tiny CoT模型
llamafactory-cli train ../llama_factory_configs/qwen_tiny_sft_config.yaml

# 训练其他模型
llamafactory-cli train ../llama_factory_configs/qwen_short_sft_config.yaml   # Short CoT
llamafactory-cli train ../llama_factory_configs/qwen_medium_sft_config.yaml  # Medium CoT
llamafactory-cli train ../llama_factory_configs/qwen_long_sft_config.yaml    # Long CoT
```

#### 多GPU并行训练

```bash
# 8GPU并行训练所有6个模型
cd scripts/training
bash train_all_models_parallel.sh

# 监控训练进度
bash monitor_training.sh

# 停止所有训练
bash stop_all_training.sh
```

**训练配置：**
- 使用LoRA微调减少显存占用
- 自动GPU分配: GPU1-6分别训练不同CoT模型
- 支持断点续训
- 实时日志监控

### 4. 模型评估

#### 快速评估 (推荐)

```bash
# Ultra Fast 8GPU并行评估所有模型
cd scripts/evaluation
bash ultra_fast_eval.sh

# 预计时间: 10-15分钟完成所有7个模型的100个测试样例
```

#### 单模型详细评估

```bash
# 评估单个模型
cd evaluation
python model_evaluator.py --model-path ../models/qwen_tiny_cot_lora --num-cases 100

# 评估基线模型
python model_evaluator.py --model-type base --num-cases 100

# 对比评估
python model_evaluator.py --model-type both --num-cases 50
```

#### 自定义评估

```bash
# 高并行度评估
python model_evaluator.py \
    --num-cases 100 \
    --max-workers 16 \
    --batch-size 32 \
    --model-path ../models/qwen_medium_cot_lora

# 多GPU评估
python scripts/evaluation/ultra_turbo_evaluator.py
```

**评估参数：**
- `--num-cases`: 测试案例数量
- `--max-workers`: 并行线程数
- `--batch-size`: 批处理大小
- `--device`: 指定GPU设备
- `--quiet`: 静默模式

## 🔧 高级功能

### GPU资源监控

```bash
# 实时监控8GPU使用情况
cd scripts/utils
bash gpu_monitor.sh
```

### 训练监控

```bash
# 监控所有模型训练状态
cd scripts/training
bash monitor_training.sh

# 查看特定模型训练日志
tail -f ../../logs/train_tiny.log
```

### 结果分析

```bash
# 查看评估结果
ls evaluation_results/

# 分析最新评估结果
python -c "
import json
import glob
import os
files = glob.glob('evaluation_results/ultra_turbo_evaluation_*.json')
if files:
    latest_file = max(files, key=os.path.getctime)
    with open(latest_file, 'r') as f:
        data = json.load(f)
    for result in data['results']:
        if result.get('success', False):
            print(f\"{result['model_name']}: {result.get('accuracy', 0):.1f}%\")
"
```

## 📈 性能基准

基于100个测试样例的评估结果：

| 模型 | 准确率 | CoT质量 | 推理速度 | 显存占用 |
|------|--------|---------|----------|----------|
| Qwen基线 | ~0% | 0/5 | 最快 | 最少 |
| Tiny CoT | ~15-20% | 2/5 | 快 | 少 |
| Short CoT | ~10-25% | 3/5 | 中等 | 中等 |
| Medium CoT | ~5-20% | 3/5 | 中等 | 中等 |
| Long CoT | ~0-15% | 4/5 | 慢 | 多 |
| Very Long CoT | ~0-10% | 4/5 | 更慢 | 更多 |
| Ultra Long CoT | ~0-5% | 5/5 | 最慢 | 最多 |

## 🛠️ 故障排除

### 常见问题

1. **CUDA内存不足**
   ```bash
   # 减少批大小
   --batch-size 8
   # 或使用更小的模型
   --max-workers 4
   ```

2. **训练中断**
   ```bash
   # 检查训练状态
   bash scripts/training/monitor_training.sh
   # 重启训练
   bash scripts/training/train_all_models_parallel.sh
   ```

3. **评估速度慢**
   ```bash
   # 使用更少测试案例
   --num-cases 20
   # 或使用Ultra Fast评估
   bash scripts/evaluation/ultra_fast_eval.sh
   ```

4. **模型路径错误**
   ```bash
   # 检查模型是否存在
   ls models/
   # 使用正确的模型路径
   --model-path ./models/qwen_tiny_cot_lora
   ```

### 日志文件

- 训练日志: `logs/train_*.log`
- 评估结果: `evaluation_results/`
- 错误日志: `nohup.out`

## 📝 脚本说明

### 训练脚本
- `scripts/training/train_all_models_parallel.sh` - 多GPU并行训练所有模型
- `scripts/training/monitor_training.sh` - 监控训练进度
- `scripts/training/stop_all_training.sh` - 停止所有训练进程

### 数据生成脚本  
- `scripts/data_generation/generate_all_cot_lengths.py` - 生成6种CoT数据
- `evaluation/generate_test_set.py` - 生成标准测试集

### 评估脚本
- `scripts/evaluation/ultra_fast_eval.sh` - 8GPU超高速评估
- `scripts/evaluation/ultra_turbo_evaluator.py` - 并行评估器
- `evaluation/model_evaluator.py` - 统一模型评估器

### 工具脚本
- `scripts/utils/gpu_monitor.sh` - GPU资源监控

## 🤝 贡献指南

1. Fork项目
2. 创建功能分支: `git checkout -b feature/new-feature`
3. 提交更改: `git commit -am 'Add new feature'`
4. 推送分支: `git push origin feature/new-feature`
5. 创建Pull Request

## 📄 许可证

本项目采用MIT许可证 - 详见 [LICENSE](LICENSE) 文件

## 🔗 相关链接

- [LlamaFactory](https://github.com/hiyouga/LLaMA-Factory) - 训练框架
- [Qwen模型](https://github.com/QwenLM/Qwen) - 基础模型
- [TextArena](https://github.com/textarena/textarena) - 游戏环境
