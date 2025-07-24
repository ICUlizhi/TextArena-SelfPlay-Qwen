#!/usr/bin/env python3
"""
评估结果统计器
统计所有evaluation_*.json文件中的模型性能
"""

import json
import glob
import os
from collections import defaultdict
import argparse

def parse_model_name_from_file(filename):
    """从文件名解析模型名称"""
    basename = os.path.basename(filename)
    # evaluation_medium_cuda_3_20250724_235359_806.json
    parts = basename.split('_')
    if len(parts) >= 2:
        return parts[1]  # medium, tiny, long等
    return "unknown"

def analyze_evaluation_file(filepath):
    """分析单个评估文件"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 获取基本信息
        eval_info = data.get('evaluation_info', {})
        summary = data.get('summary', {})
        detailed_results = data.get('detailed_results', [])
        
        model_path = eval_info.get('model_path', '')
        model_name = os.path.basename(model_path) if model_path else parse_model_name_from_file(filepath)
        
        # 计算统计信息
        total_cases = len(detailed_results)
        correct_cases = sum(1 for result in detailed_results if result.get('is_correct', False))
        accuracy = (correct_cases / total_cases * 100) if total_cases > 0 else 0
        
        # 按难度分类统计
        difficulty_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
        stage_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
        move_type_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
        
        for result in detailed_results:
            is_correct = result.get('is_correct', False)
            
            # 难度统计
            difficulty = result.get('difficulty', 'unknown')
            difficulty_stats[difficulty]['total'] += 1
            if is_correct:
                difficulty_stats[difficulty]['correct'] += 1
            
            # 阶段统计
            stage = result.get('stage', 'unknown')
            stage_stats[stage]['total'] += 1
            if is_correct:
                stage_stats[stage]['correct'] += 1
            
            # 移动类型统计
            move_type = result.get('move_type', 'unknown')
            move_type_stats[move_type]['total'] += 1
            if is_correct:
                move_type_stats[move_type]['correct'] += 1
        
        return {
            'filename': os.path.basename(filepath),
            'model_name': model_name,
            'model_path': model_path,
            'device': eval_info.get('device', 'unknown'),
            'timestamp': eval_info.get('timestamp', 'unknown'),
            'total_cases': total_cases,
            'correct_cases': correct_cases,
            'accuracy': accuracy,
            'difficulty_stats': dict(difficulty_stats),
            'stage_stats': dict(stage_stats),
            'move_type_stats': dict(move_type_stats)
        }
        
    except Exception as e:
        print(f"❌ 处理文件 {filepath} 时出错: {e}")
        return None

def calculate_category_accuracy(stats):
    """计算分类准确率"""
    for category in stats:
        if stats[category]['total'] > 0:
            stats[category]['accuracy'] = (stats[category]['correct'] / stats[category]['total'] * 100)
        else:
            stats[category]['accuracy'] = 0
    return stats

def main():
    parser = argparse.ArgumentParser(description="评估结果统计器")
    parser.add_argument("--pattern", type=str, default="evaluation_*.json", help="文件匹配模式")
    parser.add_argument("--output", type=str, help="输出统计结果到JSON文件")
    parser.add_argument("--sort-by", type=str, choices=['accuracy', 'name', 'timestamp'], default='accuracy', help="排序方式")
    
    args = parser.parse_args()
    
    # 查找匹配的文件
    pattern = args.pattern
    if not pattern.startswith('/'):
        pattern = os.path.join('.', pattern)
    
    files = glob.glob(pattern)
    if not files:
        print(f"❌ 未找到匹配模式 '{args.pattern}' 的文件")
        return
    
    print(f"📁 找到 {len(files)} 个评估文件")
    print("=" * 80)
    
    # 分析所有文件
    results = []
    for filepath in files:
        print(f"📄 分析文件: {os.path.basename(filepath)}")
        result = analyze_evaluation_file(filepath)
        if result:
            results.append(result)
    
    if not results:
        print("❌ 没有成功分析的文件")
        return
    
    # 排序结果
    if args.sort_by == 'accuracy':
        results.sort(key=lambda x: x['accuracy'], reverse=True)
    elif args.sort_by == 'name':
        results.sort(key=lambda x: x['model_name'])
    elif args.sort_by == 'timestamp':
        results.sort(key=lambda x: x['timestamp'])
    
    # 显示统计结果
    print("\n" + "=" * 80)
    print("📊 模型性能统计")
    print("=" * 80)
    
    print(f"{'模型名称':<25} {'准确率':<10} {'正确/总数':<12} {'设备':<10} {'文件名':<30}")
    print("-" * 80)
    
    for result in results:
        model_name = result['model_name'][:24]
        accuracy = f"{result['accuracy']:.1f}%"
        correct_total = f"{result['correct_cases']}/{result['total_cases']}"
        device = result['device']
        filename = result['filename'][:29]
        
        print(f"{model_name:<25} {accuracy:<10} {correct_total:<12} {device:<10} {filename}")
    
    # 汇总统计
    total_cases = sum(r['total_cases'] for r in results)
    total_correct = sum(r['correct_cases'] for r in results)
    avg_accuracy = (total_correct / total_cases * 100) if total_cases > 0 else 0
    
    print("-" * 80)
    print(f"{'总计':<25} {avg_accuracy:.1f}% {total_correct}/{total_cases}")
    
    # 详细分类统计
    print("\n📈 详细分类统计:")
    
    # 汇总所有模型的难度统计
    combined_difficulty = defaultdict(lambda: {'total': 0, 'correct': 0})
    combined_stage = defaultdict(lambda: {'total': 0, 'correct': 0})
    combined_move_type = defaultdict(lambda: {'total': 0, 'correct': 0})
    
    for result in results:
        for diff, stats in result['difficulty_stats'].items():
            combined_difficulty[diff]['total'] += stats['total']
            combined_difficulty[diff]['correct'] += stats['correct']
        
        for stage, stats in result['stage_stats'].items():
            combined_stage[stage]['total'] += stats['total']
            combined_stage[stage]['correct'] += stats['correct']
        
        for move_type, stats in result['move_type_stats'].items():
            combined_move_type[move_type]['total'] += stats['total']
            combined_move_type[move_type]['correct'] += stats['correct']
    
    # 显示汇总统计
    print("\n🎯 按难度汇总:")
    combined_difficulty = calculate_category_accuracy(combined_difficulty)
    for difficulty, stats in sorted(combined_difficulty.items()):
        print(f"  {difficulty}: {stats['correct']}/{stats['total']} ({stats['accuracy']:.1f}%)")
    
    print("\n🎯 按阶段汇总:")
    combined_stage = calculate_category_accuracy(combined_stage)
    for stage, stats in sorted(combined_stage.items()):
        print(f"  {stage}: {stats['correct']}/{stats['total']} ({stats['accuracy']:.1f}%)")
    
    print("\n🎯 按移动类型汇总:")
    combined_move_type = calculate_category_accuracy(combined_move_type)
    for move_type, stats in sorted(combined_move_type.items()):
        print(f"  {move_type}: {stats['correct']}/{stats['total']} ({stats['accuracy']:.1f}%)")
    
    # 保存详细统计结果
    if args.output:
        summary_data = {
            'analysis_info': {
                'total_files': len(results),
                'total_cases': total_cases,
                'total_correct': total_correct,
                'average_accuracy': avg_accuracy,
                'pattern': args.pattern,
                'sort_by': args.sort_by
            },
            'model_results': results,
            'combined_stats': {
                'difficulty': dict(combined_difficulty),
                'stage': dict(combined_stage),
                'move_type': dict(combined_move_type)
            }
        }
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 详细统计结果已保存到: {args.output}")
    
    # 显示最佳和最差模型
    if len(results) > 1:
        best_model = results[0]
        worst_model = results[-1]
        
        print(f"\n🏆 最佳模型: {best_model['model_name']} ({best_model['accuracy']:.1f}%)")
        print(f"📉 最差模型: {worst_model['model_name']} ({worst_model['accuracy']:.1f}%)")
        print(f"📊 性能差距: {best_model['accuracy'] - worst_model['accuracy']:.1f}%")

if __name__ == "__main__":
    main()
