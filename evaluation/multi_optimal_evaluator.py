#!/usr/bin/env python3
"""
多最优解模型评估器
专门为tictactoe_test_set_100_multi_optimal.json设计的简洁评估工具
"""

import json
import re
import os
import sys
import argparse
from typing import Optional, List, Dict

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("警告: PyTorch/Transformers不可用，将使用模拟模式")

class MultiOptimalEvaluator:
    """多最优解评估器"""
    
    def __init__(self, model_path: str, base_model_path: str = None, device: str = "auto"):
        self.model_path = model_path
        self.base_model_path = base_model_path
        self.device = device
        self.model = None
        self.tokenizer = None
        self.model_name = os.path.basename(model_path)  # 添加模型名称
        self.load_model()
    
    def load_model(self):
        """加载模型"""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch不可用，无法加载模型")
        
        # 设备选择
        if self.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        
        print(f"🔄 加载模型到设备: {self.device}")
        
        # 检查设备是否可用
        if self.device.startswith("cuda"):
            device_id = int(self.device.split(":")[1]) if ":" in self.device else 0
            if device_id >= torch.cuda.device_count():
                raise RuntimeError(f"GPU {device_id} 不存在，系统共有 {torch.cuda.device_count()} 个GPU")
            
            # 检查GPU内存
            torch.cuda.set_device(device_id)
            memory_info = torch.cuda.get_device_properties(device_id)
            free_memory = torch.cuda.get_device_properties(device_id).total_memory
            print(f"📊 GPU {device_id} 内存: {free_memory // (1024**3)}GB ({memory_info.name})")
        
        # 判断是微调模型还是基础模型
        if os.path.exists(os.path.join(self.model_path, "adapter_config.json")):
            # 微调模型 (LoRA)
            print("📦 加载微调模型...")
            if not self.base_model_path:
                self.base_model_path = "/mnt/cvda/cvda_avatar/1/textarena-selfplay-qwen/qwen"
            
            print(f"🔗 基础模型路径: {self.base_model_path}")
            print(f"🔗 LoRA适配器路径: {self.model_path}")
            
            # 检查路径是否存在
            if not os.path.exists(self.base_model_path):
                raise FileNotFoundError(f"基础模型路径不存在: {self.base_model_path}")
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"LoRA模型路径不存在: {self.model_path}")
            
            # 加载基础模型
            print("🔄 正在加载基础模型...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_path, trust_remote_code=True)
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_path,
                device_map=self.device,
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
            
            # 加载LoRA适配器
            print("🔄 正在加载LoRA适配器...")
            self.model = PeftModel.from_pretrained(base_model, self.model_path)
            print("✅ 微调模型加载成功")
            
        else:
            # 基础模型
            print("📦 加载基础模型...")
            print(f"🔗 模型路径: {self.model_path}")
            
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"基础模型路径不存在: {self.model_path}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map=self.device,
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
            print("✅ 基础模型加载成功")
    
    def generate_response(self, prompt: str) -> str:
        """生成模型响应"""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch不可用，无法进行模型推理")
        
        if self.model is None:
            raise RuntimeError("模型未加载，无法进行推理")
        
        try:
            # 实际模型推理
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # 移除原始prompt部分
            response = response[len(prompt):].strip()
            return response
            
        except Exception as e:
            raise RuntimeError(f"生成响应时出错: {e}")
    
    def extract_move(self, response: str) -> Optional[str]:
        """从响应中提取移动"""
        if not response:
            return None
        
        # 多种模式匹配
        patterns = [
            r'答案:\s*\[(\d)\]',           # 答案: [数字]
            r'选择:\s*\[(\d)\]',           # 选择: [数字]
            r'最终选择:\s*\[(\d)\]',       # 最终选择: [数字]
            r'我选择:\s*\[(\d)\]',         # 我选择: [数字]
            r'\[(\d)\]',                   # 任何 [数字]
            r'位置\s*(\d)',                # 位置数字
            r'选择位置\s*(\d)',            # 选择位置数字
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response)
            if matches:
                move_num = matches[-1]  # 取最后一个匹配
                return f"[{move_num}]"
        return None
    
    def is_move_optimal(self, predicted_move: str, case: Dict) -> bool:
        """检查预测的移动是否为最优解"""
        if predicted_move is None:
            return False
        
        # 如果测试用例有多个最优解列表，检查预测是否在其中
        if 'optimal_moves' in case:
            return predicted_move in case['optimal_moves']
        
        # 向后兼容：如果只有单个最优解，使用传统比较
        optimal_move = case.get('optimal_move')
        return predicted_move == optimal_move
    
    def create_prompt(self, case: Dict) -> str:
        """创建提示词"""
        return f"""你是一个井字棋专家，需要在当前局面下选择最优的落子位置。

【井字棋游戏规则】
1. 两名玩家轮流在3×3的棋盘上放置自己的标记（X或O）
2. 获胜条件：率先在横行、竖列或对角线上连成3个自己的标记
3. 平局条件：棋盘填满且无人获胜
4. 战略要点：
   - 优先级1：如果你能获胜，立即获胜
   - 优先级2：如果对手下一步能获胜，必须阻止
   - 优先级3：创造多重威胁（fork）

【当前棋盘状态】
{case['board_state']}

【位置编号对应关系】
0 | 1 | 2
---------
3 | 4 | 5
---------
6 | 7 | 8

【游戏信息】
- 你是玩家{case['player']}，现在轮到你落子
- 可选的位置有：{', '.join(case['available_moves'])}

【获胜线路说明】
- 横行：0-1-2, 3-4-5, 6-7-8
- 竖列：0-3-6, 1-4-7, 2-5-8  
- 对角线：0-4-8, 2-4-6

请进行详细的思考分析，然后给出你的最终选择。

【思考步骤】
1. 【局面分析】分析当前棋盘状态，识别已有的棋子分布
2. 【威胁检测】检查是否有立即获胜机会或需要阻止对手获胜
3. 【策略导向】确定当前应该采用的策略（进攻/防守/平衡）
4. 【候选评估】评估各个可选位置的价值和风险
5. 【最终决策】选择最优位置并说明理由

最后请用格式"答案: [数字]"给出你的选择。
"""
    
    def evaluate(self, test_set_path: str, num_cases: Optional[int] = None) -> tuple[float, List[Dict]]:
        """评估模型性能"""
        # 加载测试集
        with open(test_set_path, 'r', encoding='utf-8') as f:
            test_cases = json.load(f)
        
        if num_cases:
            test_cases = test_cases[:num_cases]
        
        print(f"📋 加载了 {len(test_cases)} 个测试案例")
        print(f"🤖 当前模型: {self.model_name}")
        print(f"📍 运行设备: {self.device}")
        print("-" * 80)
        
        correct_count = 0
        total_count = len(test_cases)
        detailed_results = []
        
        for i, case in enumerate(test_cases, 1):
            print(f"\n🔄 案例 {i}/{total_count} (ID: {case.get('id', i)})")
            print(f"   棋盘: {case.get('stage', 'unknown')} | 难度: {case.get('difficulty', 'unknown')} | 玩家: {case.get('player', 'unknown')}")
            print(f"   最优解: {case.get('optimal_moves', [])}")
            
            # 显示当前棋盘状态
            board_state = case.get('board_state', '')
            if board_state:
                print(f"   📋 当前棋盘:")
                # 将棋盘状态按行分割并缩进显示
                board_lines = board_state.split('\n')
                for line in board_lines:
                    print(f"      {line}")
            
            # 生成提示词
            prompt = self.create_prompt(case)
            
            # 获取模型响应
            print("   🔮 模型思考中...")
            response = self.generate_response(prompt)
            
            # 提取预测移动
            predicted_move = self.extract_move(response)
            
            # 检查是否正确
            is_correct = self.is_move_optimal(predicted_move, case)
            
            if is_correct:
                correct_count += 1
            
            # 实时显示结果
            status = "✅ 正确" if is_correct else "❌ 错误"
            current_accuracy = (correct_count / i) * 100
            print(f"   📤 模型选择: {predicted_move}")
            print(f"   📊 结果: {status} | 当前准确率: {current_accuracy:.1f}% ({correct_count}/{i})")
            
            # 显示模型输出的关键部分
            if len(response) > 150:
                output_preview = response[:150] + "..."
            else:
                output_preview = response
            print(f"   💭 模型回答: {output_preview}")
            
            # 记录详细结果
            result_item = {
                "case_id": case.get("id", i),
                "board_state": case.get("board_state", ""),
                "player": case.get("player", ""),
                "available_moves": case.get("available_moves", []),
                "optimal_moves": case.get("optimal_moves", [case.get("optimal_move", "")]),
                "model_output": response,
                "predicted_move": predicted_move,
                "is_correct": is_correct,
                "difficulty": case.get("difficulty", ""),
                "stage": case.get("stage", ""),
                "move_type": case.get("move_type", "")
            }
            detailed_results.append(result_item)
            
            # 每10个案例显示一次统计
            if i % 10 == 0 or i == total_count:
                print(f"\n📈 阶段性统计 ({i}/{total_count}):")
                print(f"   正确: {correct_count} | 错误: {i - correct_count} | 准确率: {current_accuracy:.1f}%")
                if i < total_count:
                    print("-" * 40)
        
        accuracy = (correct_count / total_count) * 100
        print(f"\n🎯 最终结果: {accuracy:.2f}% ({correct_count}/{total_count})")
        return accuracy, detailed_results

def load_test_cases(test_set_path: str, num_cases: Optional[int] = None) -> List[Dict]:
    """加载测试案例"""
    if not os.path.exists(test_set_path):
        print(f"❌ 测试集文件不存在: {test_set_path}")
        return []
    
    with open(test_set_path, 'r', encoding='utf-8') as f:
        test_cases = json.load(f)
    
    if num_cases:
        test_cases = test_cases[:num_cases]
    
    print(f"✅ 加载了 {len(test_cases)} 个测试案例")
    return test_cases

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="多最优解模型评估器")
    parser.add_argument("--model-path", type=str, required=True, help="模型路径")
    parser.add_argument("--base-model-path", type=str, 
                       default="/mnt/cvda/cvda_avatar/1/textarena-selfplay-qwen/qwen",
                       help="基础模型路径（微调模型需要）")
    parser.add_argument("--test-set", type=str, required=True, help="测试集路径")
    parser.add_argument("--num-cases", type=int, help="测试案例数量（可选）")
    parser.add_argument("--device", type=str, default="auto", help="设备（cuda:0, cpu等）")
    parser.add_argument("--output", type=str, help="输出JSON文件路径（可选）")
    
    args = parser.parse_args()
    
    print("🎯 多最优解模型评估器")
    print("=" * 50)
    print(f"模型路径: {args.model_path}")
    print(f"测试集: {args.test_set}")
    print(f"设备: {args.device}")
    if args.num_cases:
        print(f"测试案例数量: {args.num_cases}")
    print()
    
    # 初始化评估器
    evaluator = MultiOptimalEvaluator(
        model_path=args.model_path,
        base_model_path=args.base_model_path,
        device=args.device
    )
    
    # 执行评估
    accuracy, detailed_results = evaluator.evaluate(args.test_set, args.num_cases)
    
    # 准备完整的评估结果
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    evaluation_summary = {
        "evaluation_info": {
            "timestamp": datetime.now().isoformat(),
            "model_path": args.model_path,
            "test_set": args.test_set,
            "device": args.device,
            "num_cases": args.num_cases or len(detailed_results),
            "accuracy": accuracy
        },
        "summary": {
            "total_cases": len(detailed_results),
            "correct_cases": sum(1 for r in detailed_results if r["is_correct"]),
            "accuracy_percentage": accuracy,
            "difficulty_breakdown": {},
            "stage_breakdown": {},
            "move_type_breakdown": {}
        },
        "detailed_results": detailed_results
    }
    
    # 统计分析
    for result in detailed_results:
        # 难度统计
        difficulty = result["difficulty"]
        if difficulty not in evaluation_summary["summary"]["difficulty_breakdown"]:
            evaluation_summary["summary"]["difficulty_breakdown"][difficulty] = {"total": 0, "correct": 0}
        evaluation_summary["summary"]["difficulty_breakdown"][difficulty]["total"] += 1
        if result["is_correct"]:
            evaluation_summary["summary"]["difficulty_breakdown"][difficulty]["correct"] += 1
        
        # 阶段统计
        stage = result["stage"]
        if stage not in evaluation_summary["summary"]["stage_breakdown"]:
            evaluation_summary["summary"]["stage_breakdown"][stage] = {"total": 0, "correct": 0}
        evaluation_summary["summary"]["stage_breakdown"][stage]["total"] += 1
        if result["is_correct"]:
            evaluation_summary["summary"]["stage_breakdown"][stage]["correct"] += 1
        
        # 移动类型统计
        move_type = result["move_type"]
        if move_type not in evaluation_summary["summary"]["move_type_breakdown"]:
            evaluation_summary["summary"]["move_type_breakdown"][move_type] = {"total": 0, "correct": 0}
        evaluation_summary["summary"]["move_type_breakdown"][move_type]["total"] += 1
        if result["is_correct"]:
            evaluation_summary["summary"]["move_type_breakdown"][move_type]["correct"] += 1
    
    # 计算百分比
    for category in ["difficulty_breakdown", "stage_breakdown", "move_type_breakdown"]:
        for key in evaluation_summary["summary"][category]:
            stats = evaluation_summary["summary"][category][key]
            stats["accuracy"] = (stats["correct"] / stats["total"] * 100) if stats["total"] > 0 else 0
    
    print("=" * 50)
    print(f"🎉 评估完成！")
    print(f"📊 准确率: {accuracy:.2f}%")
    print()
    
    # 输出简要统计
    print("📈 详细统计:")
    print(f"总案例数: {evaluation_summary['summary']['total_cases']}")
    print(f"正确案例数: {evaluation_summary['summary']['correct_cases']}")
    print()
    
    if evaluation_summary["summary"]["difficulty_breakdown"]:
        print("按难度分类:")
        for difficulty, stats in evaluation_summary["summary"]["difficulty_breakdown"].items():
            print(f"  {difficulty}: {stats['correct']}/{stats['total']} ({stats['accuracy']:.1f}%)")
    
    if evaluation_summary["summary"]["stage_breakdown"]:
        print("按阶段分类:")
        for stage, stats in evaluation_summary["summary"]["stage_breakdown"].items():
            print(f"  {stage}: {stats['correct']}/{stats['total']} ({stats['accuracy']:.1f}%)")
    
    # 保存详细结果到JSON
    if args.output:
        output_file = args.output
    else:
        model_name = os.path.basename(args.model_path)
        output_file = f"multi_optimal_evaluation_{model_name}_{timestamp}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(evaluation_summary, f, ensure_ascii=False, indent=2)
    
    print(f"💾 详细评估结果已保存到: {output_file}")
    
    # 显示前几个案例的详细信息
    print("\n🔍 前3个案例的详细信息:")
    for i, result in enumerate(detailed_results[:3], 1):
        print(f"\n案例 {i} (ID: {result['case_id']}):")
        print(f"  棋盘状态:\n{result['board_state']}")
        print(f"  玩家: {result['player']}")
        print(f"  可选位置: {result['available_moves']}")
        print(f"  最优解: {result['optimal_moves']}")
        print(f"  模型选择: {result['predicted_move']}")
        print(f"  是否正确: {'✅' if result['is_correct'] else '❌'}")
        print(f"  模型输出: {result['model_output'][:100]}..." if len(result['model_output']) > 100 else f"  模型输出: {result['model_output']}")
    
    return evaluation_summary

if __name__ == "__main__":
    main()
