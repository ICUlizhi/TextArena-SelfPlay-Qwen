#!/usr/bin/env python3
"""
Long CoT控制示例
演示如何使用不同的CoT长度生成训练数据

用法:
    python examples/long_cot_demo.py
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from agents.qwen_agent import QwenAgent
from utils.mock_env import MockTicTacToeEnv
from data_generation.selfplay_runner import SelfPlayRunner
import json
from datetime import datetime

def demo_cot_lengths():
    """演示不同CoT长度的效果"""
    print("="*60)
    print("Long CoT控制演示")
    print("="*60)
    
    # 测试观察状态
    observation = """[GAME] You are Player 0 in Tic Tac Toe.
Your goal is to win three in a row (horizontally, vertically, or diagonally) on the board.
On your turn, you should select the square number (0-8) you want to put your mark in next.

As Player 0, you will be 'X', while your opponent is 'O'.

[GAME] Current Board:

 0 | 1 | 2 
---+---+---
 3 | 4 | 5 
---+---+---
 6 | 7 | 8 

Available Moves: '[0]', '[1]', '[2]', '[3]', '[4]', '[5]', '[6]', '[7]', '[8]'"""
    
    # 测试不同CoT长度
    cot_lengths = ['short', 'medium', 'long', 'ultra_long']
    
    for cot_length in cot_lengths:
        print(f"\n--- {cot_length.upper()} CoT ---")
        agent = QwenAgent(strategy='aggressive', cot_length=cot_length)
        action = agent(observation)
        
        print(f"动作: {action}")
        print(f"推理长度: {len(agent.last_cot)} 字符")
        print(f"推理预览: {agent.last_cot[:150]}...")
        print("-" * 40)

def generate_long_cot_data():
    """生成包含Long CoT的训练数据"""
    print("\n" + "="*60)
    print("生成Long CoT训练数据")
    print("="*60)
    
    # 创建环境和agents
    env = MockTicTacToeEnv()
    agents = [
        QwenAgent(strategy='aggressive', cot_length='long'),
        QwenAgent(strategy='conservative', cot_length='ultra_long')
    ]
    
    runner = SelfPlayRunner(env, agents)
    
    # 生成数据
    print("生成3场游戏的Long CoT数据...")
    history = runner.run_self_play(num_games=3, cot_length_control=True)
    
    print(f"\n生成了 {len(history)} 场游戏")
    
    # 统计分析
    total_moves = 0
    total_cot_length = 0
    
    for game in history:
        game_moves = len(game['moves'])
        total_moves += game_moves
        
        for move in game['moves']:
            total_cot_length += len(move['cot'])
    
    avg_cot_length = total_cot_length / total_moves if total_moves > 0 else 0
    
    print(f"总计 {total_moves} 个动作")
    print(f"平均CoT长度: {avg_cot_length:.0f} 字符")
    print(f"数据保存位置: data/raw/self_play_data_*.json")

if __name__ == "__main__":
    demo_cot_lengths()
    generate_long_cot_data()
