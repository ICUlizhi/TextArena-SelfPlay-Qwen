#!/usr/bin/env python3
"""
完整模型评估脚本
对所有7个模型进行100个样例的完整测试
"""

import os
import json
import subprocess
import time
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# 模型配置
MODELS = [
    {
        "id": "baseline",
        "name": "Qwen基线模型",
        "path": "/mnt/cvda/cvda_avatar/1/textarena-selfplay-qwen/qwen",
        "type": "base"
    },
    {
        "id": "tiny",
        "name": "Tiny CoT模型",
        "path": "/mnt/cvda/cvda_avatar/1/textarena-selfplay-qwen/models/qwen_tiny_cot_lora",
        "type": "lora"
    },
    {
        "id": "short",
        "name": "Short CoT模型", 
        "path": "/mnt/cvda/cvda_avatar/1/textarena-selfplay-qwen/models/qwen_short_cot_lora",
        "type": "lora"
    },
    {
        "id": "medium",
        "name": "Medium CoT模型",
        "path": "/mnt/cvda/cvda_avatar/1/textarena-selfplay-qwen/models/qwen_medium_cot_lora", 
        "type": "lora"
    },
    {
        "id": "long",
        "name": "Long CoT模型",
        "path": "/mnt/cvda/cvda_avatar/1/textarena-selfplay-qwen/models/qwen_long_cot_lora",
        "type": "lora"
    },
    {
        "id": "very_long",
        "name": "Very Long CoT模型",
        "path": "/mnt/cvda/cvda_avatar/1/textarena-selfplay-qwen/models/qwen_very_long_cot_lora",
        "type": "lora"
    },
    {
        "id": "ultra_long",
        "name": "Ultra Long CoT模型",
        "path": "/mnt/cvda/cvda_avatar/1/textarena-selfplay-qwen/models/qwen_ultra_long_cot_lora",
        "type": "lora"
    }
]

def get_available_gpus():
    """检测可用的GPU"""
    if not TORCH_AVAILABLE:
        print("⚠️  PyTorch不可用，将使用CPU模式")
        return []
    
    try:
        gpu_count = torch.cuda.device_count()
        if gpu_count == 0:
            print("⚠️  未检测到CUDA GPU，将使用CPU模式")
            return []
        
        available_gpus = []
        for i in range(gpu_count):
            try:
                # 检查GPU是否可用
                torch.cuda.set_device(i)
                memory_info = torch.cuda.get_device_properties(i)
                available_gpus.append({
                    "id": i,
                    "name": memory_info.name,
                    "memory": memory_info.total_memory // (1024**3)  # GB
                })
            except Exception as e:
                print(f"⚠️  GPU {i} 不可用: {e}")
        
        return available_gpus
    except Exception as e:
        print(f"⚠️  GPU检测失败: {e}")
        return []

def assign_models_to_gpus(models, available_gpus):
    """将模型分配到GPU"""
    if not available_gpus:
        # 没有GPU，使用CPU
        return [(model, "cpu") for model in models]
    
    # 循环分配模型到GPU
    assignments = []
    for i, model in enumerate(models):
        gpu_id = available_gpus[i % len(available_gpus)]["id"]
        device = f"cuda:{gpu_id}"
        assignments.append((model, device))
    
    return assignments

def run_single_evaluation(model_and_device, test_set_path, num_cases):
    """运行单个模型的评估"""
    model, device = model_and_device
    model_name = model['name']
    print(f"� 开始评估: {model_name} (设备: {device})")
    
    # 构建命令
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # 添加毫秒避免重名
    output_file = f"evaluation_{model['id']}_{device.replace(':', '_')}_{timestamp}.json"
    
    cmd = [
        "python", "multi_optimal_evaluator.py",
        "--model-path", model["path"],
        "--test-set", test_set_path,
        "--num-cases", str(num_cases),
        "--device", device,
        "--output", output_file
    ]
    
    # 如果是LoRA模型，添加基础模型路径
    if model["type"] == "lora":
        cmd.extend(["--base-model-path", "/mnt/cvda/cvda_avatar/1/textarena-selfplay-qwen/qwen"])
    
    start_time = time.time()
    
    try:
        # 设置环境变量，确保使用指定GPU
        env = os.environ.copy()
        if device.startswith("cuda:"):
            gpu_id = device.split(":")[1]
            env["CUDA_VISIBLE_DEVICES"] = gpu_id
        
        print(f"📋 {model_name}: 开始处理 {num_cases} 个测试案例...")
        
        # 运行评估
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd="/mnt/cvda/cvda_avatar/1/textarena-selfplay-qwen/evaluation",
            env=env
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            # 尝试从输出文件读取结果
            output_path = f"/mnt/cvda/cvda_avatar/1/textarena-selfplay-qwen/evaluation/{output_file}"
            if os.path.exists(output_path):
                with open(output_path, 'r', encoding='utf-8') as f:
                    eval_data = json.load(f)
                accuracy = eval_data["evaluation_info"]["accuracy"]
            else:
                # 从stdout解析准确率
                stdout_lines = result.stdout.split('\n')
                accuracy = None
                for line in stdout_lines:
                    if "准确率:" in line or "最终结果:" in line:
                        try:
                            # 匹配百分比
                            import re
                            match = re.search(r'(\d+\.?\d*)%', line)
                            if match:
                                accuracy = float(match.group(1))
                                break
                        except:
                            pass
                if accuracy is None:
                    accuracy = 0.0
            
            cases_per_sec = num_cases / duration if duration > 0 else 0
            print(f"✅ {model_name} 评估完成!")
            print(f"   📊 准确率: {accuracy:.1f}%")
            print(f"   ⏱️  耗时: {duration:.1f}秒 ({cases_per_sec:.1f} 案例/秒)")
            print(f"   💾 结果文件: {output_file}")
            
            return {
                "model_id": model["id"],
                "model_name": model["name"],
                "model_path": model["path"],
                "device": device,
                "success": True,
                "accuracy": accuracy,
                "duration": duration,
                "cases_per_second": cases_per_sec,
                "output_file": output_file,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
        else:
            error_msg = result.stderr[:200] + "..." if len(result.stderr) > 200 else result.stderr
            print(f"❌ {model_name} 评估失败 ({device})")
            print(f"   🔍 错误: {error_msg}")
            return {
                "model_id": model["id"], 
                "model_name": model["name"],
                "model_path": model["path"],
                "device": device,
                "success": False,
                "accuracy": 0.0,
                "duration": duration,
                "cases_per_second": 0.0,
                "error": result.stderr,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
            
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        print(f"❌ {model_name} 评估异常 ({device}): {str(e)}")
        return {
            "model_id": model["id"],
            "model_name": model["name"], 
            "model_path": model["path"],
            "device": device,
            "success": False,
            "accuracy": 0.0,
            "duration": duration,
            "cases_per_second": 0.0,
            "error": str(e)
        }

def main():
    parser = argparse.ArgumentParser(description="完整模型评估")
    parser.add_argument("--num-cases", type=int, default=100, help="测试案例数量")
    parser.add_argument("--test-set", type=str, 
                       default="/mnt/cvda/cvda_avatar/1/textarena-selfplay-qwen/data/processed/tictactoe_test_set_100_multi_optimal.json",
                       help="测试集路径")
    parser.add_argument("--parallel", action="store_true", help="并行执行（多GPU模式）")
    parser.add_argument("--max-workers", type=int, default=None, help="最大并行工作进程数")
    parser.add_argument("--force-cpu", action="store_true", help="强制使用CPU模式")
    
    args = parser.parse_args()
    
    # 检测可用GPU
    if args.force_cpu:
        available_gpus = []
        print("� 强制CPU模式")
    else:
        available_gpus = get_available_gpus()
    
    print("�🚀 完整模型评估开始")
    print("=" * 60)
    print(f"📋 测试集: {args.test_set}")
    print(f"🔢 测试案例数量: {args.num_cases}")
    print(f"🤖 模型数量: {len(MODELS)}")
    print(f"📊 总评估任务: {args.num_cases * len(MODELS)}")
    print(f"🔄 并行模式: {'启用' if args.parallel else '禁用'}")
    
    if available_gpus:
        print(f"🎮 检测到 {len(available_gpus)} 个GPU:")
        for gpu in available_gpus:
            print(f"  - GPU {gpu['id']}: {gpu['name']} ({gpu['memory']}GB)")
    else:
        print("💻 使用CPU模式")
    
    # 分配模型到设备
    model_assignments = assign_models_to_gpus(MODELS, available_gpus)
    
    print(f"\n📍 模型-设备分配:")
    for model, device in model_assignments:
        print(f"  - {model['name']}: {device}")
    print()
    
    start_time = time.time()
    results = []
    
    if args.parallel and len(available_gpus) > 0:
        # 多GPU并行执行
        max_workers = args.max_workers or min(len(available_gpus), len(MODELS))
        print(f"⚡ 多GPU并行执行模式 (最大工作进程: {max_workers})")
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_model = {
                executor.submit(run_single_evaluation, model_assignment, args.test_set, args.num_cases): model_assignment[0] 
                for model_assignment in model_assignments
            }
            
            for future in as_completed(future_to_model):
                result = future.result()
                results.append(result)
    else:
        # 串行执行（推荐用于调试）
        print("🔄 串行执行模式")
        for i, model_assignment in enumerate(model_assignments, 1):
            model, device = model_assignment
            print(f"\n📍 进度: {i}/{len(MODELS)} - {model['name']} ({device})")
            result = run_single_evaluation(model_assignment, args.test_set, args.num_cases)
            results.append(result)
    
    end_time = time.time()
    total_duration = end_time - start_time
    
    # 生成汇总报告
    successful_results = [r for r in results if r["success"]]
    failed_results = [r for r in results if not r["success"]]
    
    # 按准确率排序
    if successful_results:
        successful_results.sort(key=lambda x: x["accuracy"], reverse=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_report = {
        "evaluation_info": {
            "timestamp": datetime.now().isoformat(),
            "test_set": args.test_set,
            "num_cases": args.num_cases,
            "total_models": len(MODELS),
            "successful_evaluations": len(successful_results),
            "failed_evaluations": len(failed_results),
            "total_duration": total_duration,
            "parallel_mode": args.parallel,
            "available_gpus": len(available_gpus),
            "gpu_info": available_gpus,
            "total_cases_evaluated": args.num_cases * len(successful_results),
            "overall_speed": (args.num_cases * len(successful_results)) / total_duration if total_duration > 0 else 0
        },
        "device_assignments": [
            {
                "model_id": model["id"],
                "model_name": model["name"], 
                "device": device
            } for model, device in model_assignments
        ],
        "results": results,
        "summary": {
            "best_model": successful_results[0] if successful_results else None,
            "worst_model": successful_results[-1] if successful_results else None,
            "average_accuracy": sum(r["accuracy"] for r in successful_results) / len(successful_results) if successful_results else 0
        }
    }
    
    # 保存汇总报告
    summary_file = f"full_evaluation_summary_{timestamp}.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_report, f, ensure_ascii=False, indent=2)
    
    # 显示结果
    print("\n" + "=" * 60)
    print("🎉 完整评估完成！")
    print("=" * 60)
    print(f"⏱️  总耗时: {total_duration/60:.1f} 分钟")
    print(f"🚀 整体速度: {summary_report['evaluation_info']['overall_speed']:.1f} 案例/秒")
    print(f"✅ 成功: {len(successful_results)}/{len(MODELS)} 模型")
    if failed_results:
        print(f"❌ 失败: {len(failed_results)} 模型")
    print()
    
    if successful_results:
        print("🏆 模型性能排名:")
        for i, result in enumerate(successful_results, 1):
            duration = result["duration"]
            device = result.get("device", "unknown")
            cases_per_sec = args.num_cases / duration if duration > 0 else 0
            print(f"  {i}. {result['model_name']}: {result['accuracy']:.1f}% ({duration:.1f}秒, {cases_per_sec:.1f}案例/秒, {device})")
        
        print()
        print(f"🥇 最佳模型: {successful_results[0]['model_name']} ({successful_results[0]['accuracy']:.1f}%)")
        if len(successful_results) > 1:
            print(f"📉 最差模型: {successful_results[-1]['model_name']} ({successful_results[-1]['accuracy']:.1f}%)")
        print(f"📊 平均准确率: {summary_report['summary']['average_accuracy']:.1f}%")
    
    if failed_results:
        print("\n❌ 失败的模型:")
        for result in failed_results:
            device = result.get("device", "unknown")
            error_msg = result.get('error', '未知错误')
            print(f"  - {result['model_name']} ({device}): {error_msg[:100]}...")
    
    print(f"\n💾 汇总报告已保存: {summary_file}")
    
    return summary_report

if __name__ == "__main__":
    main()
