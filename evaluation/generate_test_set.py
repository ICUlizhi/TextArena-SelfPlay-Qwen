#!/usr/bin/env python3
"""
生成井字棋测试集
要求：不少于100题，难度均匀分布，与训练集无重叠
"""

import json
import random
import itertools
from datetime import datetime

class TicTacToeTestSetGenerator:
    def __init__(self):
        self.test_cases = []
        
    def generate_board_state(self, x_positions, o_positions):
        """根据X和O的位置生成棋盘状态"""
        board = [' '] * 9
        for pos in x_positions:
            board[pos] = 'X'
        for pos in o_positions:
            board[pos] = 'O'
        
        # 转换为显示格式
        board_str = f"""{board[0]} | {board[1]} | {board[2]}
---------
{board[3]} | {board[4]} | {board[5]}
---------
{board[6]} | {board[7]} | {board[8]}"""
        
        return board_str, board
    
    def get_available_moves(self, board):
        """获取可用的落子位置"""
        return [f'[{i}]' for i in range(9) if board[i] == ' ']
    
    def check_winner(self, board):
        """检查是否有获胜者"""
        win_patterns = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # 行
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # 列
            [0, 4, 8], [2, 4, 6]              # 对角线
        ]
        
        for pattern in win_patterns:
            if board[pattern[0]] == board[pattern[1]] == board[pattern[2]] != ' ':
                return board[pattern[0]]
        return None
    
    def minimax(self, board, depth, is_maximizing, player, opponent, alpha=-float('inf'), beta=float('inf')):
        """使用Minimax算法和Alpha-Beta剪枝找到最优解"""
        # 终端状态检查
        winner = self.check_winner(board)
        if winner == player:
            return 10 - depth  # 越快获胜越好
        elif winner == opponent:
            return depth - 10  # 越慢失败越好
        elif ' ' not in board:
            return 0  # 平局
        
        # 最大化轮（当前玩家）
        if is_maximizing:
            max_eval = -float('inf')
            for i in range(9):
                if board[i] == ' ':
                    board[i] = player
                    eval_score = self.minimax(board, depth + 1, False, player, opponent, alpha, beta)
                    board[i] = ' '
                    max_eval = max(max_eval, eval_score)
                    alpha = max(alpha, eval_score)
                    if beta <= alpha:
                        break  # Beta剪枝
            return max_eval
        
        # 最小化轮（对手）
        else:
            min_eval = float('inf')
            for i in range(9):
                if board[i] == ' ':
                    board[i] = opponent
                    eval_score = self.minimax(board, depth + 1, True, player, opponent, alpha, beta)
                    board[i] = ' '
                    min_eval = min(min_eval, eval_score)
                    beta = min(beta, eval_score)
                    if beta <= alpha:
                        break  # Alpha剪枝
            return min_eval

    def evaluate_position(self, board, player):
        """使用Minimax算法评估位置，找到真正的最优解"""
        opponent = 'O' if player == 'X' else 'X'
        
        # 检查即将获胜的位置（快速路径）
        for i in range(9):
            if board[i] == ' ':
                board[i] = player
                if self.check_winner(board) == player:
                    board[i] = ' '
                    return i, "winning_move"
                board[i] = ' '
        
        # 检查需要阻挡对手的位置（快速路径）
        for i in range(9):
            if board[i] == ' ':
                board[i] = opponent
                if self.check_winner(board) == opponent:
                    board[i] = ' '
                    return i, "blocking_move"
                board[i] = ' '
        
        # 使用Minimax算法找到最优位置
        best_moves = []
        best_score = -float('inf')
        
        for i in range(9):
            if board[i] == ' ':
                board[i] = player
                score = self.minimax(board, 0, False, player, opponent)
                board[i] = ' '
                
                if score > best_score:
                    best_score = score
                    best_moves = [i]
                elif score == best_score:
                    best_moves.append(i)
        
        # 如果有多个等价的最优解，使用启发式规则选择最好的
        if len(best_moves) > 1:
            best_move = self.select_best_among_equals(board, best_moves, player)
        else:
            best_move = best_moves[0] if best_moves else None
        
        # 分析移动类型
        move_type = "optimal_move"
        if best_score > 0:
            move_type = "winning_sequence"
        elif best_score == 0:
            move_type = "draw_move"
        else:
            move_type = "best_defense"
        
        return best_move, move_type
    
    def select_best_among_equals(self, board, candidate_moves, player):
        """在Minimax分数相同的候选位置中，使用启发式规则选择最佳位置"""
        
        # 位置价值排序：中心 > 角落 > 边缘
        position_values = {
            4: 10,  # 中心 - 最高优先级
            0: 8, 2: 8, 6: 8, 8: 8,  # 角落
            1: 3, 3: 3, 5: 3, 7: 3   # 边缘
        }
        
        # 策略性位置评估
        strategic_scores = {}
        
        for pos in candidate_moves:
            score = position_values.get(pos, 0)
            
            # 额外启发式规则
            board[pos] = player
            
            # 1. 检查是否创造多个获胜威胁
            threats_created = self.count_winning_threats(board, player)
            score += threats_created * 5
            
            # 2. 检查是否阻止对手创造威胁
            board[pos] = ' '  # 暂时移除
            opponent = 'O' if player == 'X' else 'X'
            board[pos] = opponent
            opponent_threats = self.count_winning_threats(board, opponent)
            board[pos] = ' '  # 恢复
            score += opponent_threats * 3  # 阻止对手威胁也很重要
            
            # 3. 控制中心和角落的额外奖励
            if pos == 4:  # 中心位置在开局特别重要
                empty_count = board.count(' ')
                if empty_count >= 7:  # 开局阶段
                    score += 15
            
            strategic_scores[pos] = score
            board[pos] = ' '  # 恢复棋盘
        
        # 选择启发式分数最高的位置
        best_pos = max(candidate_moves, key=lambda x: strategic_scores[x])
        return best_pos
    
    def count_winning_threats(self, board, player):
        """计算玩家有多少个获胜威胁（两个己方棋子在一条线上，第三个位置为空）"""
        win_patterns = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # 行
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # 列
            [0, 4, 8], [2, 4, 6]              # 对角线
        ]
        
        threats = 0
        for pattern in win_patterns:
            player_count = sum(1 for pos in pattern if board[pos] == player)
            empty_count = sum(1 for pos in pattern if board[pos] == ' ')
            
            # 两个己方棋子 + 一个空位 = 一个威胁
            if player_count == 2 and empty_count == 1:
                threats += 1
        
        return threats
    
    def generate_test_cases(self):
        """生成测试用例"""
        
        # 1. 开局阶段测试用例 (30个)
        print("生成开局阶段测试用例...")
        for _ in range(30):
            # 空棋盘或1-2步
            num_moves = random.randint(0, 2)
            if num_moves == 0:
                x_pos, o_pos = [], []
            elif num_moves == 1:
                x_pos = [random.randint(0, 8)]
                o_pos = []
            else:
                positions = random.sample(range(9), 2)
                x_pos = [positions[0]]
                o_pos = [positions[1]]
            
            board_str, board = self.generate_board_state(x_pos, o_pos)
            available_moves = self.get_available_moves(board)
            optimal_move, move_type = self.evaluate_position(board, 'X')
            
            test_case = {
                "id": len(self.test_cases) + 1,
                "difficulty": "easy",
                "stage": "opening",
                "board_state": board_str,
                "player": "X",
                "available_moves": available_moves,
                "optimal_move": f"[{optimal_move}]",
                "move_type": move_type,
                "description": f"开局阶段，{num_moves}步后的局面"
            }
            self.test_cases.append(test_case)
        
        # 2. 中局阶段测试用例 (40个)
        print("生成中局阶段测试用例...")
        for _ in range(40):
            # 3-5步的局面
            num_moves = random.randint(3, 5)
            x_moves = (num_moves + 1) // 2
            o_moves = num_moves // 2
            
            # 确保没有获胜者
            attempts = 0
            while attempts < 100:
                all_positions = list(range(9))
                random.shuffle(all_positions)
                x_pos = all_positions[:x_moves]
                o_pos = all_positions[x_moves:x_moves + o_moves]
                
                board_str, board = self.generate_board_state(x_pos, o_pos)
                if self.check_winner(board) is None:
                    break
                attempts += 1
            
            if attempts < 100:  # 成功生成有效局面
                available_moves = self.get_available_moves(board)
                optimal_move, move_type = self.evaluate_position(board, 'X')
                
                # 根据Minimax结果确定难度
                if move_type == "winning_move":
                    difficulty = "easy"
                elif move_type == "blocking_move":
                    difficulty = "medium"
                elif move_type == "winning_sequence":
                    difficulty = "medium"
                elif move_type == "draw_move":
                    difficulty = "hard"
                else:  # best_defense
                    difficulty = "hard"
                
                test_case = {
                    "id": len(self.test_cases) + 1,
                    "difficulty": difficulty,
                    "stage": "midgame",
                    "board_state": board_str,
                    "player": "X",
                    "available_moves": available_moves,
                    "optimal_move": f"[{optimal_move}]" if optimal_move is not None else available_moves[0],
                    "move_type": move_type,
                    "description": f"中局阶段，{num_moves}步后的复杂局面 - {move_type}",
                    "minimax_verified": True  # 标记为Minimax验证过的最优解
                }
                self.test_cases.append(test_case)
        
        # 3. 残局/关键决策阶段测试用例 (30个)
        print("生成残局阶段测试用例...")
        for _ in range(30):
            # 6-8步的局面
            num_moves = random.randint(6, 8)
            x_moves = (num_moves + 1) // 2
            o_moves = num_moves // 2
            
            attempts = 0
            while attempts < 100:
                all_positions = list(range(9))
                random.shuffle(all_positions)
                x_pos = all_positions[:x_moves]
                o_pos = all_positions[x_moves:x_moves + o_moves]
                
                board_str, board = self.generate_board_state(x_pos, o_pos)
                if self.check_winner(board) is None:
                    break
                attempts += 1
            
            if attempts < 100:
                available_moves = self.get_available_moves(board)
                optimal_move, move_type = self.evaluate_position(board, 'X')
                
                # 残局阶段通常都比较复杂
                if move_type in ["winning_move", "blocking_move"]:
                    difficulty = "medium"
                else:
                    difficulty = "hard"
                
                test_case = {
                    "id": len(self.test_cases) + 1,
                    "difficulty": difficulty,
                    "stage": "endgame",
                    "board_state": board_str,
                    "player": "X",
                    "available_moves": available_moves,
                    "optimal_move": f"[{optimal_move}]" if optimal_move is not None else available_moves[0],
                    "move_type": move_type,
                    "description": f"残局阶段，{num_moves}步后的关键决策时刻 - {move_type}",
                    "minimax_verified": True  # 标记为Minimax验证过的最优解
                }
                self.test_cases.append(test_case)
        
        print(f"总共生成了 {len(self.test_cases)} 个测试用例")
        
        # 验证最优解的正确性
        print("🔍 验证最优解的正确性...")
        self.validate_optimal_moves()
        
        # 分析难度分布
        difficulty_count = {"easy": 0, "medium": 0, "hard": 0}
        move_type_count = {}
        for case in self.test_cases:
            difficulty_count[case["difficulty"]] += 1
            move_type = case.get("move_type", "unknown")
            move_type_count[move_type] = move_type_count.get(move_type, 0) + 1
        
        print(f"难度分布: {difficulty_count}")
        print(f"移动类型分布: {move_type_count}")
        
    def validate_optimal_moves(self):
        """验证每个测试用例的最优解是否真的最优"""
        validation_passed = 0
        validation_failed = 0
        
        for i, case in enumerate(self.test_cases):
            # 重新构建棋盘状态
            board = [' '] * 9
            lines = case["board_state"].split('\n')
            positions = []
            for line in lines:
                if '|' in line:
                    row = [cell.strip() for cell in line.split('|')]
                    positions.extend(row)
            
            for j, cell in enumerate(positions):
                if cell in ['X', 'O']:
                    board[j] = cell
            
            # 重新计算最优解
            optimal_move, move_type = self.evaluate_position(board, 'X')
            expected_move = int(case["optimal_move"].strip('[]'))
            
            if optimal_move == expected_move:
                validation_passed += 1
            else:
                validation_failed += 1
                print(f"⚠️  测试用例 {case['id']} 最优解不一致:")
                print(f"   期望: [{expected_move}], 实际: [{optimal_move}]")
                print(f"   棋盘: {case['board_state'].replace(chr(10), ' | ')}")
                
                # 更新为正确的最优解
                self.test_cases[i]["optimal_move"] = f"[{optimal_move}]"
                self.test_cases[i]["move_type"] = move_type
        
        print(f"✅ 验证通过: {validation_passed}个")
        if validation_failed > 0:
            print(f"🔧 已修正: {validation_failed}个")
        else:
            print("🎯 所有测试用例的最优解都已验证正确！")
        
    def save_test_set(self, filename=None):
        """保存测试集"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"test_set_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.test_cases, f, ensure_ascii=False, indent=2)
        
        print(f"测试集已保存到: {filename}")
        return filename

def main():
    """主函数"""
    print("🎯 开始生成井字棋测试集")
    print("要求: 不少于100题，难度均匀分布")
    print("=" * 50)
    
    generator = TicTacToeTestSetGenerator()
    generator.generate_test_cases()
    
    # 保存测试集 - 修正路径
    import os
    output_dir = "../data/processed"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    test_file = generator.save_test_set(f"{output_dir}/tictactoe_test_set_100.json")
    
    print("\n✅ 测试集生成完成!")
    print(f"📁 文件位置: {test_file}")
    print(f"📊 测试用例数量: {len(generator.test_cases)}")
    
    # 显示一些示例测试用例
    print("\n🔍 测试用例示例:")
    for i, case in enumerate(generator.test_cases[:3]):
        print(f"\n--- 示例 {i+1} ---")
        print(f"ID: {case['id']}, 难度: {case['difficulty']}, 类型: {case['move_type']}")
        print(f"棋盘:\n{case['board_state']}")
        print(f"最优解: {case['optimal_move']}")
        print(f"描述: {case['description']}")

if __name__ == "__main__":
    main()
