# 脚本使用指南

本文档详细说明项目中各个脚本的使用方法和参数配置。

## 📁 目录结构

```
scripts/
├── training/              # 训练相关脚本
│   ├── train_all_models_parallel.sh    # 多GPU并行训练
│   ├── train_all_models.sh             # 串行训练
│   ├── monitor_training.sh             # 训练监控
│   └── stop_all_training.sh            # 停止训练
├── evaluation/            # 评估相关脚本
│   ├── ultra_turbo_evaluator.py        # 超高速评估器
│   └── ultra_fast_eval.sh              # 一键快速评估
├── data_generation/       # 数据生成脚本
│   └── generate_all_cot_lengths.py     # CoT数据生成
└── utils/                 # 工具脚本
    └── gpu_monitor.sh                   # GPU监控
```

## 🏋️ 训练脚本

### train_all_models_parallel.sh
**功能**: 8GPU并行训练所有6个CoT模型

```bash
cd scripts/training
bash train_all_models_parallel.sh
```

**特点**:
- 自动分配GPU1-6给不同CoT模型
- 支持后台运行
- 生成详细训练日志
- 支持断点续训

**输出日志**:
- `logs/train_tiny.log` - Tiny CoT训练日志
- `logs/train_short.log` - Short CoT训练日志
- `logs/train_medium.log` - Medium CoT训练日志
- `logs/train_long.log` - Long CoT训练日志
- `logs/train_very_long.log` - Very Long CoT训练日志
- `logs/train_ultra_long.log` - Ultra Long CoT训练日志

### monitor_training.sh
**功能**: 实时监控所有模型的训练状态

```bash
cd scripts/training
bash monitor_training.sh
```

**显示信息**:
- 各模型训练进程状态
- GPU使用情况
- 训练进度
- 内存占用

### stop_all_training.sh
**功能**: 安全停止所有训练进程

```bash
cd scripts/training
bash stop_all_training.sh
```

**特点**:
- 发送SIGTERM信号优雅停止
- 保存当前训练状态
- 清理临时文件

## 📊 数据生成脚本

### generate_all_cot_lengths.py
**功能**: 生成6种不同长度的CoT训练数据

```bash
cd scripts/data_generation
python generate_all_cot_lengths.py [参数]
```

**参数说明**:
```bash
--num_games 1000              # 每种CoT类型生成的游戏数量 (默认: 1000)
--base_output_dir ../../data/processed  # 输出目录 (默认: ../../data/processed)
--device cuda:0               # 使用的GPU设备 (默认: 自动选择)
--seed 42                     # 随机种子 (默认: 42)
```

**使用示例**:
```bash
# 生成2000个游戏数据
python generate_all_cot_lengths.py --num_games 2000

# 指定输出目录和GPU
python generate_all_cot_lengths.py --num_games 1500 --base_output_dir /path/to/output --device cuda:1

# 设置随机种子确保可重现
python generate_all_cot_lengths.py --seed 123
```

**输出文件**:
- `tictactoe_tiny_cot_data.json` - 80-120字符推理
- `tictactoe_short_cot_data.json` - 150-250字符推理
- `tictactoe_medium_cot_data.json` - 300-500字符推理
- `tictactoe_long_cot_data.json` - 600-1000字符推理
- `tictactoe_very_long_cot_data.json` - 1500-2500字符推理
- `tictactoe_ultra_long_cot_data.json` - 3000-5000字符推理

## 🚀 评估脚本

### ultra_fast_eval.sh
**功能**: 一键启动8GPU超高速评估所有模型

```bash
cd scripts/evaluation
bash ultra_fast_eval.sh
```

**特点**:
- 8GPU并行评估7个模型 (6个CoT + 1个基线)
- 自动负载均衡
- 实时进度显示
- 自动生成评估报告

**预计时间**: 10-15分钟完成700个评估任务

### ultra_turbo_evaluator.py
**功能**: 可配置的高性能并行评估器

```bash
cd scripts/evaluation
python ultra_turbo_evaluator.py [参数]
```

**参数说明**:
```bash
--num-cases 100              # 每个模型的测试案例数 (默认: 100)
--num-gpus 8                 # 可用GPU数量 (默认: 8)
--models tiny short medium   # 指定要测试的模型 (默认: 所有模型)
--verbose                    # 详细输出模式
```

**使用示例**:
```bash
# 评估所有模型，每个100个案例
python ultra_turbo_evaluator.py --num-cases 100

# 仅评估特定模型
python ultra_turbo_evaluator.py --models tiny short --num-cases 50

# 使用4个GPU，详细输出
python ultra_turbo_evaluator.py --num-gpus 4 --verbose
```

**输出结果**:
- JSON格式评估报告
- 模型性能排名
- 详细错误分析
- 处理速度统计

## 🔧 工具脚本

### gpu_monitor.sh
**功能**: 实时监控8GPU使用情况

```bash
cd scripts/utils
bash gpu_monitor.sh
```

**显示信息**:
- GPU使用率
- 显存占用
- 温度状态
- 运行进程

**状态指示**:
- 🔥 忙碌: GPU使用率 > 80%
- ⚡ 工作: GPU使用率 40-80%
- 💤 轻载: GPU使用率 10-40%
- 😴 空闲: GPU使用率 < 10%

## 📝 最佳实践

### 训练阶段
1. **开始训练前**:
   ```bash
   # 检查GPU状态
   scripts/utils/gpu_monitor.sh
   
   # 确保数据已生成
   ls data/processed/tictactoe_*_cot_data.json
   ```

2. **启动训练**:
   ```bash
   # 使用screen/tmux保持会话
   screen -S training
   scripts/training/train_all_models_parallel.sh
   ```

3. **监控训练**:
   ```bash
   # 在另一个终端监控
   scripts/training/monitor_training.sh
   ```

### 评估阶段
1. **快速评估**:
   ```bash
   # 一键评估所有模型
   scripts/evaluation/ultra_fast_eval.sh
   ```

2. **自定义评估**:
   ```bash
   # 评估特定模型
   cd evaluation
   python model_evaluator.py --model-path ../models/qwen_tiny_cot_lora --num-cases 50
   ```

3. **结果分析**:
   ```bash
   # 查看最新结果
   ls -la evaluation_results/ | tail -5
   ```

## ⚠️ 注意事项

1. **资源要求**:
   - 训练: 至少6张GPU (每张4GB+ 显存)
   - 评估: 建议8张GPU (可用更少但速度较慢)

2. **路径设置**:
   - 确保所有路径相对于项目根目录正确
   - 检查LlamaFactory安装位置

3. **进程管理**:
   - 使用`stop_all_training.sh`优雅停止训练
   - 避免强制kill进程可能导致数据丢失

4. **磁盘空间**:
   - 每个模型约需要1-2GB存储空间
   - 确保足够磁盘空间存储日志和结果
