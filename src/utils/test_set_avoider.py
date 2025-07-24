#!/usr/bin/env python3
"""
测试集规避器 - 确保训练数据与测试集不重复
用于self-play数据生成时过滤掉与测试集相同的局面
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Set, Tuple, Optional

class TestSetAvoider:
    """测试集规避器 - 防止训练数据与测试集重复"""
    
    def __init__(self, test_set_path: str = None):
        """
        初始化测试集规避器
        
        Args:
            test_set_path: 测试集文件路径
        """
        if test_set_path is None:
            # 默认路径 - 从项目根目录开始
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(current_dir))  # 上两级到项目根目录
            test_set_path = os.path.join(
                project_root, 'data', 'processed', 'tictactoe_test_set_100.json'
            )
        
        self.test_set_path = test_set_path
        self.test_positions = set()
        self.test_situations = set()
        self._load_test_set()
    
    def _load_test_set(self):
        """加载测试集并提取关键信息"""
        if not os.path.exists(self.test_set_path):
            print(f"⚠️  测试集文件不存在: {self.test_set_path}")
            return
        
        try:
            with open(self.test_set_path, 'r', encoding='utf-8') as f:
                test_data = json.load(f)
            
            print(f"📋 加载测试集: {len(test_data)} 个测试案例")
            
            for case in test_data:
                # 提取棋盘状态和玩家信息
                board_state = case['board_state']
                player = case['player']
                
                # 创建唯一标识符
                position_key = self._create_position_key(board_state, player)
                self.test_positions.add(position_key)
                
                # 也创建简化的局面标识（只考虑棋盘状态）
                board_key = self._normalize_board_state(board_state)
                self.test_situations.add(board_key)
            
            print(f"✅ 提取了 {len(self.test_positions)} 个测试位置")
            print(f"✅ 提取了 {len(self.test_situations)} 个独特局面")
            
        except Exception as e:
            print(f"❌ 加载测试集失败: {e}")
    
    def _create_position_key(self, board_state: str, player: str) -> str:
        """创建位置的唯一标识符"""
        # 标准化棋盘状态，然后加上当前玩家
        normalized_board = self._normalize_board_state(board_state)
        return f"{normalized_board}|{player}"
    
    def _normalize_board_state(self, board_state: str) -> str:
        """标准化棋盘状态，移除格式差异"""
        # 移除所有空白字符和分隔线，只保留X、O和空位
        lines = board_state.strip().split('\n')
        positions = []
        
        for line in lines:
            if '---' not in line:  # 跳过分隔线
                # 提取每行的位置
                chars = line.split('|')
                for char in chars:
                    char = char.strip()
                    if char == '':
                        positions.append(' ')
                    else:
                        positions.append(char)
        
        # 确保正好9个位置
        if len(positions) != 9:
            # 如果解析失败，使用简单方法
            clean_board = ''.join(c for c in board_state if c in 'XO ')
            if len(clean_board) == 9:
                positions = list(clean_board)
            else:
                # 最后的fallback，填充到9位
                positions = list(clean_board[:9].ljust(9))
        
        return ''.join(positions)
    
    def _extract_board_from_observation(self, observation: str) -> Tuple[str, str]:
        """从观察文本中提取棋盘状态和当前玩家"""
        lines = observation.strip().split('\n')
        
        # 寻找棋盘部分
        board_lines = []
        current_player = None
        
        for line in lines:
            # 检查是否包含棋盘格式
            if '|' in line and ('X' in line or 'O' in line or '  ' in line):
                board_lines.append(line)
            elif '---' in line:
                board_lines.append(line)
            elif '你是' in line and ('X' in line or 'O' in line):
                # 提取当前玩家
                if 'X' in line:
                    current_player = 'X'
                elif 'O' in line:
                    current_player = 'O'
        
        if board_lines:
            board_state = '\n'.join(board_lines)
            return board_state, current_player or 'X'
        
        return '', 'X'
    
    def is_test_position(self, board_state: str, player: str) -> bool:
        """检查给定位置是否在测试集中"""
        position_key = self._create_position_key(board_state, player)
        return position_key in self.test_positions
    
    def is_test_situation(self, board_state: str) -> bool:
        """检查给定局面是否在测试集中（不考虑玩家）"""
        board_key = self._normalize_board_state(board_state)
        return board_key in self.test_situations
    
    def should_avoid_move(self, move_data: Dict) -> bool:
        """判断一个棋步是否应该被规避（因为在测试集中）"""
        observation = move_data.get('observation', '')
        if not observation:
            return False
        
        # 从观察中提取棋盘状态和玩家
        board_state, player = self._extract_board_from_observation(observation)
        
        if not board_state:
            return False
        
        # 检查是否与测试集重复
        return self.is_test_position(board_state, player)
    
    def filter_game_moves(self, game_data: Dict) -> Dict:
        """过滤一个游戏中与测试集重复的棋步"""
        if 'moves' not in game_data:
            return game_data
        
        filtered_moves = []
        avoided_count = 0
        
        for move in game_data['moves']:
            if not self.should_avoid_move(move):
                filtered_moves.append(move)
            else:
                avoided_count += 1
        
        # 创建过滤后的游戏数据
        filtered_game = game_data.copy()
        filtered_game['moves'] = filtered_moves
        
        if avoided_count > 0:
            filtered_game['avoided_moves_count'] = avoided_count
            print(f"  规避了 {avoided_count} 个与测试集重复的棋步")
        
        return filtered_game
    
    def filter_training_samples(self, samples: List[Dict]) -> List[Dict]:
        """过滤训练样本，移除与测试集重复的样本"""
        filtered_samples = []
        avoided_count = 0
        
        for sample in samples:
            # 从input中提取棋盘信息
            input_text = sample.get('input', '')
            
            # 寻找棋盘状态
            board_state, player = self._extract_board_from_observation(input_text)
            
            if board_state and self.is_test_position(board_state, player):
                avoided_count += 1
                continue
            
            filtered_samples.append(sample)
        
        print(f"📊 训练样本过滤结果:")
        print(f"  原始样本: {len(samples)}")
        print(f"  规避样本: {avoided_count}")
        print(f"  保留样本: {len(filtered_samples)}")
        print(f"  规避率: {avoided_count/len(samples)*100:.1f}%")
        
        return filtered_samples
    
    def filter_self_play_data(self, games_data: List[Dict]) -> List[Dict]:
        """过滤self-play游戏数据，移除与测试集重复的内容"""
        filtered_games = []
        total_avoided = 0
        
        print(f"🔍 开始过滤 {len(games_data)} 个游戏的数据...")
        
        for i, game in enumerate(games_data):
            print(f"  处理游戏 {i+1}/{len(games_data)}")
            filtered_game = self.filter_game_moves(game)
            
            # 只保留有有效棋步的游戏
            if filtered_game.get('moves'):
                filtered_games.append(filtered_game)
                total_avoided += filtered_game.get('avoided_moves_count', 0)
            else:
                print(f"  游戏 {i+1} 所有棋步都被规避，跳过整个游戏")
        
        print(f"\n📊 Self-play数据过滤结果:")
        print(f"  原始游戏: {len(games_data)}")
        print(f"  保留游戏: {len(filtered_games)}")
        print(f"  总计规避棋步: {total_avoided}")
        
        return filtered_games
    
    def get_statistics(self) -> Dict:
        """获取测试集统计信息"""
        return {
            "test_set_path": self.test_set_path,
            "total_test_positions": len(self.test_positions),
            "total_test_situations": len(self.test_situations),
            "loaded_successfully": len(self.test_positions) > 0
        }
    
    def save_filtered_data(self, 
                          filtered_data: List[Dict], 
                          output_path: str,
                          original_count: int = None):
        """保存过滤后的数据并添加元数据"""
        # 添加过滤元数据
        metadata = {
            "filtering_info": {
                "test_set_path": self.test_set_path,
                "test_positions_count": len(self.test_positions),
                "original_data_count": original_count or len(filtered_data),
                "filtered_data_count": len(filtered_data),
                "avoided_count": (original_count or len(filtered_data)) - len(filtered_data),
                "filtering_timestamp": datetime.now().isoformat()
            },
            "data": filtered_data
        }
        
        # 保存到文件
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        print(f"💾 过滤后的数据已保存到: {output_path}")

def main():
    """测试函数"""
    print("🧪 测试集规避器测试")
    print("=" * 40)
    
    # 初始化规避器
    avoider = TestSetAvoider()
    
    # 显示统计信息
    stats = avoider.get_statistics()
    print(f"📊 测试集统计:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # 测试一些示例
    test_board = """  |   |  
---------
  | O |  
---------
  | X |  """
    
    print(f"\n🧪 测试棋盘:")
    print(test_board)
    print(f"是否在测试集中 (X): {avoider.is_test_position(test_board, 'X')}")
    print(f"是否在测试集中 (O): {avoider.is_test_position(test_board, 'O')}")

if __name__ == "__main__":
    main()
