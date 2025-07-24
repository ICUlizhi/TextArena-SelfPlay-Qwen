# 测试集规避系统使用指南

## 概述

测试集规避系统确保self-play生成的训练数据与测试集不重复，避免数据泄露，保证模型评估的可靠性。

## 系统组件

### 1. 测试集规避器 (`src/utils/test_set_avoider.py`)

**功能**：
- 加载并分析测试集数据
- 识别与测试集重复的训练样本
- 提供过滤接口

**核心方法**：
- `is_test_position(board_state, player)`: 检查特定位置是否在测试集中
- `filter_self_play_data(games_data)`: 过滤self-play游戏数据
- `filter_training_samples(samples)`: 过滤训练样本

### 2. 增强版Self-Play运行器 (`src/data_generation/selfplay_runner.py`)

**更新内容**：
- 集成测试集规避器
- 在保存数据时自动过滤重复内容
- 添加过滤统计信息

**新参数**：
- `enable_test_avoidance`: 是否启用测试集规避（默认True）

### 3. 增强版SFT数据生成脚本 (`scripts/generate_sft_data_with_avoidance.py`)

**功能**：
- 从self-play数据生成SFT训练样本
- 自动规避测试集重复
- 支持多种输出格式
- 提供详细的过滤统计

## 使用方法

### 1. 生成Self-Play数据（自动规避）

修改现有的self-play生成代码：

```python
# 原来的方式
runner = SelfPlayRunner(env, agents)

# 新的方式（自动启用测试集规避）
runner = SelfPlayRunner(env, agents, enable_test_avoidance=True)

# 禁用测试集规避（如果需要）
runner = SelfPlayRunner(env, agents, enable_test_avoidance=False)
```

### 2. 生成训练数据（规避测试集）

使用增强版脚本：

```bash
# 基本用法
python scripts/generate_sft_data_with_avoidance.py

# 自定义参数
python scripts/generate_sft_data_with_avoidance.py \
  --data-dir data/raw \
  --output-dir data/processed \
  --test-set-path data/processed/tictactoe_test_set_100.json \
  --format both \
  --max-samples 500
```

### 3. 独立使用测试集规避器

```python
from src.utils.test_set_avoider import TestSetAvoider

# 初始化
avoider = TestSetAvoider("path/to/test_set.json")

# 检查单个位置
is_duplicate = avoider.is_test_position(board_state, player)

# 过滤训练样本
filtered_samples = avoider.filter_training_samples(samples)

# 过滤self-play数据
filtered_games = avoider.filter_self_play_data(games_data)
```

## 命令行示例

### 1. 快速测试（10个样本）

```bash
python scripts/generate_sft_data_with_avoidance.py \
  --max-samples 10 \
  --format llama-factory
```

### 2. 生成完整训练集

```bash
python scripts/generate_sft_data_with_avoidance.py \
  --max-samples 1000 \
  --format both \
  --output-prefix final_training_data
```

### 3. 自定义路径

```bash
python scripts/generate_sft_data_with_avoidance.py \
  --data-dir /path/to/selfplay/data \
  --test-set-path /path/to/test_set.json \
  --output-dir /path/to/output
```

## 输出文件格式

### 标准格式
```json
{
  "generation_info": {
    "timestamp": "2025-07-24T16:05:30",
    "format": "standard",
    "total_samples": 100,
    "test_set_avoidance": {
      "total_test_positions": 89,
      "loaded_successfully": true
    },
    "quality_analysis": {
      "total_samples": 100,
      "average_cot_length": 191.1,
      "strategy_distribution": {...}
    }
  },
  "data": [...]
}
```

### LLaMA-Factory格式
```json
{
  "generation_info": {...},
  "data": [
    {
      "instruction": "井字棋游戏中，请作为X玩家...",
      "input": "当前游戏状态：...",
      "output": "思考过程：...答案: [4]"
    }
  ]
}
```

## 规避效果统计

运行后会显示详细统计：

```
📊 Self-play数据过滤结果:
  原始游戏: 128
  保留游戏: 128
  总计规避棋步: 128

📊 训练样本过滤结果:
  原始样本: 731
  规避样本: 65
  保留样本: 666
  规避率: 8.9%
```

## 配置选项

### TestSetAvoider参数
- `test_set_path`: 测试集文件路径
- 自动加载并解析测试集数据

### SelfPlayRunner参数
- `enable_test_avoidance`: 是否启用规避（默认True）

### 生成脚本参数
- `--test-set-path`: 测试集路径
- `--data-dir`: Self-play数据目录
- `--output-dir`: 输出目录
- `--format`: 输出格式 (standard/llama-factory/both)
- `--max-samples`: 最大样本数限制
- `--output-prefix`: 输出文件前缀

## 验证和调试

### 1. 测试规避器功能

```bash
python src/utils/test_set_avoider.py
```

### 2. 检查规避效果

查看生成的JSON文件中的`generation_info`部分，包含：
- 规避的样本数量
- 规避率统计
- 测试集加载状态

### 3. 手动验证

可以对比生成的训练样本和测试集，确保没有重复：

```python
# 加载测试集和训练数据进行对比
test_set = json.load(open("test_set.json"))
training_data = json.load(open("training_data.json"))

# 检查是否有重复的棋盘状态
```

## 注意事项

1. **路径配置**：确保测试集路径正确
2. **数据格式**：规避器适用于当前的数据格式
3. **性能影响**：规避过程会轻微增加处理时间
4. **规避率**：通常规避率在5-15%之间是正常的
5. **质量保证**：规避后仍保持训练数据的多样性和质量

## 故障排除

### 测试集加载失败
```
⚠️  测试集文件不存在: /path/to/test_set.json
```
**解决**：检查测试集文件路径是否正确

### 没有找到重复数据
如果规避率为0%，可能原因：
- 测试集和训练数据格式不匹配
- 数据来源不同，本身没有重复

### 规避率过高
如果规避率超过30%，可能需要：
- 检查数据生成策略
- 增加self-play数据的多样性
- 调整测试集大小

## 集成到现有流程

### 更新现有脚本

将原有的数据生成脚本替换为新的规避版本：

```bash
# 原来
python scripts/generate_sft_data.py

# 现在
python scripts/generate_sft_data_with_avoidance.py
```

### 更新训练配置

确保使用规避后的训练数据：

```yaml
# training_config.yaml
dataset_path: "data/processed/training_data_filtered_llama_factory_*.json"
```

这样就完成了测试集规避系统的部署和使用！
