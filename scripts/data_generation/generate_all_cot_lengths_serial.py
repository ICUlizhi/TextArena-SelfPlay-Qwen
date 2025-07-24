#!/usr/bin/env python3
"""
串行生成六种不同长度的CoT训练数据 - 安全版本
逐一生成，避免并发冲突，确保数据完整性
"""

import os
import sys
import subprocess
import json
import glob
import time
from datetime import datetime

# CoT配置
COT_CONFIGS = {
    'tiny': {'target': 100, 'range': (80, 120)},
    'short': {'target': 200, 'range': (150, 250)},
    'long': {'target': 800, 'range': (600, 1000)},
    'very_long': {'target': 2000, 'range': (1500, 2500)},
    'ultra_long': {'target': 4000, 'range': (3000, 5000)}
}

def generate_cot_data_safe(cot_type, games=100):
    """安全地生成单个CoT类型的数据"""
    try:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        print(f"[{cot_type}] 开始生成 ~{COT_CONFIGS[cot_type]['target']}字符 数据...")
        
        # 构建命令
        cmd = [
            sys.executable, 
            os.path.join(project_root, 'src', 'main.py'),
            '--num-games', str(games),
            '--cot-length', cot_type
        ]
        
        # 设置环境变量
        env = os.environ.copy()
        env['PYTHONPATH'] = project_root
        
        # 运行命令
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            cwd=project_root,
            env=env,
            timeout=300  # 5分钟超时
        )
        
        if result.returncode == 0:
            # 查找生成的文件
            data_dir = os.path.join(project_root, "data", "processed")
            pattern = f"long_cot_sft_data_{cot_type}_*.json"
            files = glob.glob(os.path.join(data_dir, pattern))
            
            if files:
                # 找到最新的文件
                latest_file = max(files, key=os.path.getctime)
                
                # 验证文件
                try:
                    with open(latest_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    sample_count = len(data)
                    if sample_count > 0:
                        # 计算CoT长度统计
                        lengths = []
                        for sample in data:
                            if 'conversations' in sample:
                                for conv in sample['conversations']:
                                    if conv.get('from') == 'gpt':
                                        length = len(conv.get('value', ''))
                                        lengths.append(length)
                        
                        if lengths:
                            avg_length = sum(lengths) / len(lengths)
                            min_length = min(lengths)
                            max_length = max(lengths)
                            
                            target_range = COT_CONFIGS[cot_type]['range']
                            status = "✓ 符合要求" if target_range[0] <= avg_length <= target_range[1] else "⚠ 长度需要调整"
                            
                            print(f"[{cot_type}] ✓ 生成成功")
                            print(f"  - 样本数量: {sample_count}")
                            print(f"  - 平均长度: {avg_length:.1f} 字符")
                            print(f"  - 长度范围: {min_length}-{max_length} 字符")
                            print(f"  - 期望范围: {target_range[0]}-{target_range[1]} 字符")
                            print(f"  - 状态: {status}")
                            
                            return {
                                'success': True,
                                'cot_type': cot_type,
                                'file_path': latest_file,
                                'sample_count': sample_count,
                                'avg_length': round(avg_length, 1),
                                'length_range': f"{min_length}-{max_length}",
                                'target_range': f"{target_range[0]}-{target_range[1]}",
                                'status': status
                            }
                        else:
                            return {'success': False, 'cot_type': cot_type, 'error': '无法提取CoT长度信息'}
                    else:
                        return {'success': False, 'cot_type': cot_type, 'error': '生成的数据为空'}
                
                except json.JSONDecodeError as e:
                    return {'success': False, 'cot_type': cot_type, 'error': f'JSON解析错误: {e}'}
            else:
                return {'success': False, 'cot_type': cot_type, 'error': '未找到生成的文件'}
        else:
            error_msg = result.stderr if result.stderr else "未知错误"
            print(f"[{cot_type}] ✗ 生成失败: {error_msg}")
            return {'success': False, 'cot_type': cot_type, 'error': error_msg, 'stdout': result.stdout}
            
    except subprocess.TimeoutExpired:
        return {'success': False, 'cot_type': cot_type, 'error': '生成超时（5分钟）'}
    except Exception as e:
        print(f"[{cot_type}] ✗ 生成失败: {str(e)}")
        return {'success': False, 'cot_type': cot_type, 'error': str(e)}

def main():
    print("=" * 80)
    print("五种CoT长度训练数据串行生成器 - 安全版本")
    print("=" * 80)
    start_time = datetime.now()
    print(f"开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 确保必要的目录存在
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.makedirs(os.path.join(project_root, "data", "processed"), exist_ok=True)
    
    # 配置任务
    games_per_type = 100  # 每种类型生成5个游戏
    cot_types = list(COT_CONFIGS.keys())
    
    print(f"将串行生成 {len(cot_types)} 种CoT长度类型，每种 {games_per_type} 个游戏")
    print()
    
    # 串行执行
    results = []
    for i, cot_type in enumerate(cot_types, 1):
        print(f"进度: {i}/{len(cot_types)} - 处理 {cot_type}")
        
        result = generate_cot_data_safe(cot_type, games_per_type)
        results.append(result)
        
        # 在每个类型之间添加短暂延迟，确保文件系统稳定
        if i < len(cot_types):
            print(f"等待2秒后继续...")
            time.sleep(2)
        
        print()
    
    # 生成报告
    print("=" * 80)
    print("生成结果汇总")
    print("=" * 80)
    
    successful = 0
    failed = 0
    
    for result in results:
        if result['success']:
            successful += 1
            cot_type = result['cot_type']
            print(f"✅ {cot_type:<12} - 成功: {result['sample_count']} 样本, 平均 {result['avg_length']} 字符, {result['status']}")
        else:
            failed += 1
            cot_type = result.get('cot_type', '未知')
            error = result['error'][:100] + "..." if len(result['error']) > 100 else result['error']
            print(f"❌ {cot_type:<12} - 失败: {error}")
    
    print(f"\n📊 统计: {successful}/{len(results)} 成功, {failed}/{len(results)} 失败")
    
    # 保存详细报告
    end_time = datetime.now()
    report = {
        'start_time': start_time.isoformat(),
        'end_time': end_time.isoformat(),
        'duration_seconds': (end_time - start_time).total_seconds(),
        'total_tasks': len(results),
        'successful': successful,
        'failed': failed,
        'results': results
    }
    
    report_file = os.path.join(project_root, "data", "processed", f"serial_cot_generation_report_{start_time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"📝 详细报告已保存: {report_file}")
    
    if successful == len(results):
        print("\n🎉 所有CoT数据生成成功!")
        return True
    else:
        print(f"\n❌ {failed} 个CoT数据生成失败!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
