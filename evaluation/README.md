# 模型评估工具使用指南

## 概述

`model_evaluator.py` 是一个统一的模型评估工具，支持测试不同类型的模型、自定义测试案例数量和评估模式。

## 功能特性

- ✅ **多模型支持**: 可以测试微调模型、基础模型或两者对比
- ✅ **灵活配置**: 可指定测试案例数量、模型路径、设备等
- ✅ **详细分析**: 包含准确率、CoT推理质量、错误类型分析
- ✅ **结果保存**: 自动保存详细的评估结果到JSON文件
- ✅ **容错性强**: 支持模拟模式，即使没有安装PyTorch也能运行

## 使用方法

### 基本用法

```bash
# 快速测试微调模型 (20个案例)
python evaluation/model_evaluator.py

# 测试5个案例
python evaluation/model_evaluator.py --num-cases 5

# 静默模式运行
python evaluation/model_evaluator.py --num-cases 10 --quiet
```

### 指定GPU设备

```bash
# 使用GPU 6
CUDA_VISIBLE_DEVICES=6 python evaluation/model_evaluator.py --device cuda:6

# 使用GPU 7
CUDA_VISIBLE_DEVICES=7 python evaluation/model_evaluator.py --device cuda:7
```

### 对比不同模型

```bash
# 对比微调模型和基础模型
python evaluation/model_evaluator.py --model-type both --num-cases 20

# 只测试基础模型
python evaluation/model_evaluator.py --model-type base --num-cases 10
```

### 自定义模型路径

```bash
# 指定自定义的模型路径
python evaluation/model_evaluator.py \
  --model-path /path/to/your/finetuned/model \
  --base-model-path /path/to/your/base/model \
  --num-cases 15
```

### 保存结果到指定文件

```bash
# 保存结果到自定义文件
python evaluation/model_evaluator.py --output my_evaluation_results.json
```

## 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model-type` | str | `finetuned` | 模型类型: `finetuned`, `base`, `both` |
| `--num-cases` | int | `20` | 测试案例数量 |
| `--model-path` | str | 项目默认路径 | 微调模型路径 |
| `--base-model-path` | str | 项目默认路径 | 基础模型路径 |
| `--device` | str | 自动选择 | 指定设备 (如 `cuda:6`) |
| `--output` | str | 自动生成 | 输出文件名 |
| `--quiet` | flag | False | 静默模式，只显示最终结果 |

## 输出说明

### 控制台输出

```
🎯 统一模型评估工具
==================================================
模型类型: finetuned
测试案例数量: 20
设备: cuda:6

📋 加载了 20 个测试案例
🔄 加载模型 (设备: cuda:6)...
✅ 微调模型加载成功

🎯 评估模型: finetuned
==================================================

📋 测试案例 1/20
描述: 开局阶段，2步后的局面
难度: easy
  |   |  
---------
  | O |  
---------
  | X |  
当前玩家: X
🎯 预测: [6] | 正确: [6] | ✅
📝 响应: 思考：这是开局阶段...
📊 CoT质量: 4/5

...

📊 finetuned 模型结果:
  准确率: 15/20 = 75.0%
  平均CoT质量: 3.8/5
  格式提取失败: 1 个
  策略判断错误: 4 个

💾 评估结果已保存到: evaluation_results_20250724_120000.json

🎉 评估完成!
📊 finetuned 模型: 75.0% 准确率
```

### JSON结果文件

保存的JSON文件包含：

- **evaluation_info**: 评估元数据（时间戳、模型路径、设备等）
- **results**: 每个模型的详细结果
  - **summary**: 汇总统计（准确率、CoT质量等）
  - **detailed_results**: 每个测试案例的完整信息
  - **error_analysis**: 错误类型统计

## 使用示例

### 1. 快速准确率测试

```bash
# 测试微调模型在10个案例上的表现
CUDA_VISIBLE_DEVICES=6 python evaluation/model_evaluator.py --num-cases 10 --quiet
```

### 2. 详细分析

```bash
# 获取20个案例的详细分析，包括每个案例的完整生成内容
python evaluation/model_evaluator.py --num-cases 20
```

### 3. 模型对比

```bash
# 对比微调模型和基础模型的性能差异
python evaluation/model_evaluator.py --model-type both --num-cases 50
```

### 4. 大规模评估

```bash
# 在全部100个测试案例上进行评估
python evaluation/model_evaluator.py --num-cases 100 --output full_evaluation.json
```

## 故障排除

### 1. CUDA内存不足

```bash
# 使用不同的GPU
CUDA_VISIBLE_DEVICES=7 python evaluation/model_evaluator.py
```

### 2. 模型加载失败

程序会自动切换到模拟模式，提供基本功能演示。

### 3. 测试集找不到

确保测试集文件存在：`data/processed/tictactoe_test_set_100.json`

## 输出文件示例

查看保存的JSON文件来了解：
- 每个测试案例的完整prompt和响应
- 错误案例的具体分析
- CoT推理质量评分详情
- 可能的改进建议

这些详细信息有助于诊断模型性能问题和优化方向。
