#!/usr/bin/env python3
"""
生成井字棋测试集 - 支持多个最优解版本
要求：不少于100题，难度均匀分布，与训练集无重叠，支持多个等价最优解
"""

import json
import random
import itertools
import argparse
from datetime import datetime

class TicTacToeMultiOptimalTestSetGenerator:
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

    def find_all_optimal_moves(self, board, player):
        """找到所有最优解（Minimax分数相同的所有位置）"""
        opponent = 'O' if player == 'X' else 'X'
        
        # 首先检查即将获胜的位置（最高优先级）
        winning_moves = []
        for i in range(9):
            if board[i] == ' ':
                board[i] = player
                if self.check_winner(board) == player:
                    winning_moves.append(i)
                board[i] = ' '
        
        if winning_moves:
            return winning_moves, "winning_move", 1000  # 获胜移动有最高分数
        
        # 然后检查需要阻挡对手的位置（第二优先级）
        blocking_moves = []
        for i in range(9):
            if board[i] == ' ':
                board[i] = opponent
                if self.check_winner(board) == opponent:
                    blocking_moves.append(i)
                board[i] = ' '
        
        if blocking_moves:
            return blocking_moves, "blocking_move", 500  # 阻挡移动有第二高分数
        
        # 使用Minimax算法找到所有最优位置
        move_scores = {}
        best_score = -float('inf')
        
        for i in range(9):
            if board[i] == ' ':
                board[i] = player
                score = self.minimax(board, 0, False, player, opponent)
                board[i] = ' '
                
                move_scores[i] = score
                if score > best_score:
                    best_score = score
        
        # 找到所有具有最佳分数的移动
        optimal_moves = [move for move, score in move_scores.items() if score == best_score]
        
        # 分析移动类型
        move_type = "optimal_move"
        if best_score > 0:
            move_type = "winning_sequence"
        elif best_score == 0:
            move_type = "draw_move"
        else:
            move_type = "best_defense"
        
        return optimal_moves, move_type, best_score
    
    def analyze_move_equivalence(self, board, optimal_moves, player):
        """分析最优解之间的等价性和策略差异"""
        move_analysis = {}
        
        # 位置价值评估
        position_values = {
            4: "center",      # 中心
            0: "corner", 2: "corner", 6: "corner", 8: "corner",  # 角落
            1: "edge", 3: "edge", 5: "edge", 7: "edge"           # 边缘
        }
        
        for move in optimal_moves:
            analysis = {
                "position_type": position_values.get(move, "unknown"),
                "threats_created": 0,
                "threats_blocked": 0,
                "strategic_value": 0
            }
            
            # 模拟移动并分析结果
            board[move] = player
            
            # 计算创造的威胁
            analysis["threats_created"] = self.count_winning_threats(board, player)
            
            # 计算阻挡的威胁
            board[move] = ' '  # 临时移除
            opponent = 'O' if player == 'X' else 'X'
            board[move] = opponent
            analysis["threats_blocked"] = self.count_winning_threats(board, opponent)
            
            # 恢复棋盘
            board[move] = ' '
            
            # 计算策略价值
            analysis["strategic_value"] = (
                analysis["threats_created"] * 10 + 
                analysis["threats_blocked"] * 5 +
                (15 if move == 4 else 8 if move in [0,2,6,8] else 3)
            )
            
            move_analysis[move] = analysis
        
        return move_analysis
    
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
    
    def generate_test_cases(self, multi_optimal=True):
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
            optimal_moves, move_type, score = self.find_all_optimal_moves(board, 'X')
            
            # 分析最优解
            move_analysis = self.analyze_move_equivalence(board, optimal_moves, 'X')
            
            test_case = {
                "id": len(self.test_cases) + 1,
                "difficulty": "easy",
                "stage": "opening",
                "board_state": board_str,
                "player": "X",
                "available_moves": available_moves,
                "move_type": move_type,
                "description": f"开局阶段，{num_moves}步后的局面",
                "minimax_score": score
            }
            
            if multi_optimal:
                test_case["optimal_moves"] = [f"[{move}]" for move in optimal_moves]
                test_case["move_analysis"] = {f"[{move}]": analysis for move, analysis in move_analysis.items()}
                test_case["primary_optimal"] = f"[{optimal_moves[0]}]"  # 保持兼容性
                test_case["optimal_move"] = f"[{optimal_moves[0]}]"  # 向后兼容
            else:
                # 单最优解模式，选择策略价值最高的
                best_move = max(optimal_moves, key=lambda x: move_analysis[x]["strategic_value"])
                test_case["optimal_move"] = f"[{best_move}]"
            
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
                optimal_moves, move_type, score = self.find_all_optimal_moves(board, 'X')
                
                # 根据结果确定难度
                if move_type == "winning_move":
                    difficulty = "easy"
                elif move_type == "blocking_move":
                    difficulty = "medium"
                elif move_type == "winning_sequence":
                    difficulty = "medium"
                elif move_type == "draw_move":
                    difficulty = "hard"
                else:
                    difficulty = "hard"
                
                # 分析最优解
                move_analysis = self.analyze_move_equivalence(board, optimal_moves, 'X')
                
                test_case = {
                    "id": len(self.test_cases) + 1,
                    "difficulty": difficulty,
                    "stage": "midgame",
                    "board_state": board_str,
                    "player": "X",
                    "available_moves": available_moves,
                    "move_type": move_type,
                    "description": f"中局阶段，{num_moves}步后的复杂局面 - {move_type}",
                    "minimax_verified": True,
                    "minimax_score": score
                }
                
                if multi_optimal:
                    test_case["optimal_moves"] = [f"[{move}]" for move in optimal_moves]
                    test_case["move_analysis"] = {f"[{move}]": analysis for move, analysis in move_analysis.items()}
                    test_case["primary_optimal"] = f"[{optimal_moves[0]}]"
                    test_case["optimal_move"] = f"[{optimal_moves[0]}]"  # 向后兼容
                    test_case["total_optimal_solutions"] = len(optimal_moves)
                else:
                    best_move = max(optimal_moves, key=lambda x: move_analysis[x]["strategic_value"])
                    test_case["optimal_move"] = f"[{best_move}]"
                
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
                optimal_moves, move_type, score = self.find_all_optimal_moves(board, 'X')
                
                # 残局阶段通常都比较复杂
                if move_type in ["winning_move", "blocking_move"]:
                    difficulty = "medium"
                else:
                    difficulty = "hard"
                
                # 分析最优解
                move_analysis = self.analyze_move_equivalence(board, optimal_moves, 'X')
                
                test_case = {
                    "id": len(self.test_cases) + 1,
                    "difficulty": difficulty,
                    "stage": "endgame",
                    "board_state": board_str,
                    "player": "X",
                    "available_moves": available_moves,
                    "move_type": move_type,
                    "description": f"残局阶段，{num_moves}步后的关键决策时刻 - {move_type}",
                    "minimax_verified": True,
                    "minimax_score": score
                }
                
                if multi_optimal:
                    test_case["optimal_moves"] = [f"[{move}]" for move in optimal_moves]
                    test_case["move_analysis"] = {f"[{move}]": analysis for move, analysis in move_analysis.items()}
                    test_case["primary_optimal"] = f"[{optimal_moves[0]}]"
                    test_case["optimal_move"] = f"[{optimal_moves[0]}]"  # 向后兼容
                    test_case["total_optimal_solutions"] = len(optimal_moves)
                else:
                    best_move = max(optimal_moves, key=lambda x: move_analysis[x]["strategic_value"])
                    test_case["optimal_move"] = f"[{best_move}]"
                
                self.test_cases.append(test_case)
        
        print(f"总共生成了 {len(self.test_cases)} 个测试用例")
        
        # 分析多最优解统计
        if multi_optimal:
            self.analyze_multi_optimal_statistics()
        
        # 分析难度分布
        difficulty_count = {"easy": 0, "medium": 0, "hard": 0}
        move_type_count = {}
        for case in self.test_cases:
            difficulty_count[case["difficulty"]] += 1
            move_type = case.get("move_type", "unknown")
            move_type_count[move_type] = move_type_count.get(move_type, 0) + 1
        
        print(f"难度分布: {difficulty_count}")
        print(f"移动类型分布: {move_type_count}")
    
    def analyze_multi_optimal_statistics(self):
        """分析多最优解的统计信息"""
        multi_optimal_count = 0
        total_optimal_solutions = 0
        max_solutions = 0
        
        for case in self.test_cases:
            if "optimal_moves" in case:
                num_solutions = len(case["optimal_moves"])
                total_optimal_solutions += num_solutions
                if num_solutions > 1:
                    multi_optimal_count += 1
                max_solutions = max(max_solutions, num_solutions)
        
        avg_solutions = total_optimal_solutions / len(self.test_cases) if self.test_cases else 0
        
        print(f"\n📊 多最优解统计:")
        print(f"  具有多个最优解的测试用例: {multi_optimal_count}/{len(self.test_cases)} ({multi_optimal_count/len(self.test_cases)*100:.1f}%)")
        print(f"  平均最优解数量: {avg_solutions:.2f}")
        print(f"  最多最优解数量: {max_solutions}")
        
        # 分析具有多最优解的案例
        position_type_multi = {"center": 0, "corner": 0, "edge": 0}
        for case in self.test_cases:
            if "optimal_moves" in case and len(case["optimal_moves"]) > 1:
                for move_str, analysis in case["move_analysis"].items():
                    pos_type = analysis["position_type"]
                    if pos_type in position_type_multi:
                        position_type_multi[pos_type] += 1
        
        print(f"  多最优解中位置类型分布: {position_type_multi}")
        
    def save_test_set(self, filename=None, format_type="multi_optimal"):
        """保存测试集"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if format_type == "multi_optimal":
                filename = f"multi_optimal_test_set_{timestamp}.json"
            else:
                filename = f"tictactoe_test_set_100.json"  # 兼容原格式
        
        # 如果是兼容模式，移除多最优解字段
        if format_type == "compatible":
            compatible_cases = []
            for case in self.test_cases:
                compatible_case = case.copy()
                # 移除多最优解专有字段
                fields_to_remove = ["optimal_moves", "move_analysis", "primary_optimal", "total_optimal_solutions", "minimax_score"]
                for field in fields_to_remove:
                    compatible_case.pop(field, None)
                compatible_cases.append(compatible_case)
            save_data = compatible_cases
        else:
            save_data = self.test_cases
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        
        print(f"测试集已保存到: {filename}")
        return filename

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='生成井字棋测试集 - 支持多个最优解')
    parser.add_argument('--multi-optimal', action='store_true', default=True,
                       help='生成包含多个最优解的测试集 (默认: True)')
    parser.add_argument('--compatible', action='store_true', default=False,
                       help='生成兼容原格式的测试集 (单最优解)')
    parser.add_argument('--output-file', type=str, default=None,
                       help='输出文件名 (默认: 自动生成)')
    parser.add_argument('--override-original', action='store_true', default=False,
                       help='覆盖原始测试集文件 (生成兼容格式并保存为原文件名)')
    
    args = parser.parse_args()
    
    print("🎯 开始生成井字棋测试集")
    if args.compatible or args.override_original:
        print("模式: 兼容原格式 (单最优解)")
        multi_optimal_mode = False
    else:
        print("模式: 多最优解支持")
        multi_optimal_mode = True
    
    print("要求: 不少于100题，难度均匀分布")
    print("=" * 50)
    
    generator = TicTacToeMultiOptimalTestSetGenerator()
    generator.generate_test_cases(multi_optimal=multi_optimal_mode)
    
    # 确定输出格式和文件名
    if args.override_original:
        # 覆盖原始文件模式
        output_file = "../data/processed/tictactoe_test_set_100.json"
        format_type = "compatible"
    elif args.compatible:
        # 兼容模式
        output_file = args.output_file or "../data/processed/tictactoe_test_set_100_compatible.json"
        format_type = "compatible"
    else:
        # 多最优解模式
        output_file = args.output_file or "../data/processed/tictactoe_test_set_100_multi_optimal.json"
        format_type = "multi_optimal"
    
    # 保存测试集
    import os
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    test_file = generator.save_test_set(output_file, format_type)
    
    print("\n✅ 测试集生成完成!")
    print(f"📁 文件位置: {test_file}")
    print(f"📊 测试用例数量: {len(generator.test_cases)}")
    
    # 显示一些示例测试用例
    print("\n🔍 测试用例示例:")
    for i, case in enumerate(generator.test_cases[:3]):
        print(f"\n--- 示例 {i+1} ---")
        print(f"ID: {case['id']}, 难度: {case['difficulty']}, 类型: {case['move_type']}")
        print(f"棋盘:\n{case['board_state']}")
        
        if multi_optimal_mode and "optimal_moves" in case:
            print(f"所有最优解: {case['optimal_moves']}")
            print(f"主要最优解: {case['primary_optimal']}")
            if len(case['optimal_moves']) > 1:
                print("解析:")
                for move, analysis in case['move_analysis'].items():
                    print(f"  {move}: {analysis['position_type']}, 威胁+{analysis['threats_created']}, 阻挡+{analysis['threats_blocked']}")
        else:
            print(f"最优解: {case['optimal_move']}")
        
        print(f"描述: {case['description']}")
    
    print(f"\n📖 使用说明:")
    print(f"  - 多最优解模式: python generate_multi_optimal_test_set.py")
    print(f"  - 兼容模式: python generate_multi_optimal_test_set.py --compatible")
    print(f"  - 覆盖原文件: python generate_multi_optimal_test_set.py --override-original")

if __name__ == "__main__":
    main()
