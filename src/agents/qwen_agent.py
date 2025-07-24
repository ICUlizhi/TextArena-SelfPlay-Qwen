import random
import re
import os
from typing import List, Dict, Tuple
from enum import Enum

try:
    from models.qwen_wrapper import QwenWrapper
    QWEN_AVAILABLE = True
except ImportError:
    print("Warning: QwenWrapper not available, using fallback strategy")
    QWEN_AVAILABLE = False

class StrategyType(Enum):
    CONSERVATIVE = "conservative"  # 保守型：优先防守，避免风险
    AGGRESSIVE = "aggressive"     # 激进型：优先进攻，寻求主动
    BALANCED = "balanced"         # 均衡型：攻守平衡
    OPPORTUNISTIC = "opportunistic"  # 机会主义：根据局面灵活调整

class CoTLengthType(Enum):
    TINY = "tiny"        # 极短推理 (~100 chars)
    SHORT = "short"      # 简短推理 (~200 chars)
    MEDIUM = "medium"    # 中等推理 (~500 chars) 
    LONG = "long"        # 详细推理 (~1000 chars)
    VERY_LONG = "very_long"  # 很长推理 (~2000 chars)
    ULTRA_LONG = "ultra_long"  # 超长推理 (~4000 chars)

class QwenAgent:
    def __init__(self, model_path=None, load_model=False, strategy=None, cot_length=None, use_lora=False):
        self.model_path = model_path
        self.model = None
        self.last_cot = ""
        self.player_mark = "X"  # Will be set dynamically
        self.use_lora = use_lora  # 是否使用LoRA模型
        
        # 策略多样性：随机选择或指定策略
        if isinstance(strategy, str):
            strategy_map = {
                'conservative': StrategyType.CONSERVATIVE,
                'aggressive': StrategyType.AGGRESSIVE, 
                'balanced': StrategyType.BALANCED,
                'opportunistic': StrategyType.OPPORTUNISTIC
            }
            self.strategy = strategy_map.get(strategy, StrategyType.BALANCED)
        else:
            self.strategy = strategy or random.choice(list(StrategyType))
            
        # CoT长度控制：新增功能
        if isinstance(cot_length, str):
            cot_map = {
                'tiny': CoTLengthType.TINY,
                'short': CoTLengthType.SHORT,
                'medium': CoTLengthType.MEDIUM,
                'long': CoTLengthType.LONG,
                'very_long': CoTLengthType.VERY_LONG,
                'ultra_long': CoTLengthType.ULTRA_LONG
            }
            self.cot_length = cot_map.get(cot_length, CoTLengthType.MEDIUM)
        else:
            self.cot_length = cot_length or random.choice(list(CoTLengthType))
            
        self.game_phase = "opening"  # opening, middle, endgame
        self.move_count = 0
        
        if load_model and QWEN_AVAILABLE:
            try:
                if use_lora:
                    # 加载LoRA微调模型
                    print(f"Loading LoRA model from: {model_path}")
                    # 假设基础模型在项目的qwen目录
                    base_model_path = model_path.replace("/models/", "/").replace("_cot_lora", "").split("/")[-1]
                    if "qwen" not in base_model_path:
                        base_model_path = os.path.join(os.path.dirname(model_path), "..", "..", "qwen")
                    self.model = QwenWrapper(model_path, use_lora=True, base_model_path=base_model_path)
                else:
                    # 加载基础模型
                    print(f"Loading base model from: {model_path}")
                    self.model = QwenWrapper(model_path)
                    
                self.model.load_model()
                print("Qwen model loaded successfully")
            except Exception as e:
                print(f"Failed to load Qwen model: {e}")
                print("Using fallback strategy")
                self.model = None

    def __call__(self, observation):
        """Main method called by the environment - compatible with SmartAgent interface"""
        return self.act(observation)

    def act(self, observation):
        """Generate action based on observation"""
        available_moves = self._parse_available_moves(observation)
        if not available_moves:
            return "[0]"  # Fallback to first position

        # 更新游戏阶段和步数统计
        self.move_count += 1
        self._update_game_phase(observation)

        # Determine player mark from observation
        if "Player X's turn" in observation:
            self.player_mark = "X"
        elif "Player O's turn" in observation:
            self.player_mark = "O"

        # Generate action using model or fallback strategy
        if self.model and hasattr(self.model, 'is_loaded') and self.model.is_loaded:
            # Use actual Qwen model
            cot, action = self.model.generate_move_with_cot(observation, self.player_mark)
            self.last_cot = cot
            
            # Validate action is in available moves
            try:
                action_int = int(action)
                if action_int in available_moves:
                    return f"[{action_int}]"
            except (ValueError, TypeError):
                pass
            
            # Fallback if invalid action
            return f"[{random.choice(available_moves)}]"
        else:
            # Use enhanced strategy-based reasoning
            cot, action = self.generate_strategic_cot(observation)
            self.last_cot = cot
            
            try:
                action_int = int(action)
                if action_int in available_moves:
                    return f"[{action_int}]"
            except (ValueError, TypeError):
                pass
            
            return f"[{random.choice(available_moves)}]"
    
    def _update_game_phase(self, observation):
        """Update game phase based on board state"""
        board = self._parse_board_state(observation)
        occupied_count = sum(1 for cell in board if cell != ' ')
        
        if occupied_count <= 2:
            self.game_phase = "opening"
        elif occupied_count <= 6:
            self.game_phase = "middle"
        else:
            self.game_phase = "endgame"

    def _parse_available_moves(self, observation) -> List[int]:
        """Parse available moves from observation string"""
        if "Available Moves" in observation:
            # Find the LAST occurrence of "Available Moves:" to get the current state
            available_moves_parts = observation.split("Available Moves:")
            if len(available_moves_parts) > 1:
                last_moves_part = available_moves_parts[-1]
                # Extract numbers from '[0]', '[1]', etc.
                import re
                matches = re.findall(r'\[(\d)\]', last_moves_part)
                available = [int(match) for match in matches]
                return available

        # Fallback: parse from board representation
        if "Current Board:" in observation:
            # Extract board and find empty positions
            lines = observation.split('\n')
            board_found = False
            available = []
            
            for i, line in enumerate(lines):
                if "Current Board:" in line:
                    # Get board representation - next few lines
                    try:
                        # Skip empty line, get board lines
                        board_lines = lines[i+2:i+7]  # Expect 5 lines total
                        if len(board_lines) >= 5:
                            # Parse each row
                            for row_idx, board_line_idx in enumerate([0, 2, 4]):  # Skip separator lines
                                if board_line_idx < len(board_lines):
                                    row_line = board_lines[board_line_idx]
                                    # Parse positions in format " X | 1 | 2 "
                                    parts = row_line.split(' | ')
                                    for col_idx, part in enumerate(parts):
                                        part = part.strip()
                                        position_number = row_idx * 3 + col_idx
                                        # If it's a digit, the position is available
                                        if part.isdigit():
                                            available.append(position_number)
                    except:
                        pass
            
            if available:
                return available

        # Final fallback - all positions
        return list(range(9))

    def _extract_current_state(self, game_history: str) -> dict:
        """Extract current game state information from game history"""
        # Determine player symbol
        if "Player 0" in game_history:
            if "you will be 'O'" in game_history.lower():
                player_symbol = 'O'
            else:
                player_symbol = 'X'
        elif "Player 1" in game_history:
            if "you will be 'X'" in game_history.lower():
                player_symbol = 'X'
            else:
                player_symbol = 'O'
        else:
            player_symbol = 'X'  # Default
        
        # Extract available moves
        available_moves = self._parse_available_moves(game_history)
        
        # Select move (this will be overridden by the strategic analysis)
        selected_move = available_moves[0] if available_moves else 4
        
        return {
            'player_symbol': player_symbol,
            'available_moves': available_moves,
            'selected_move': selected_move
        }

    def generate_strategic_cot(self, game_history: str) -> Tuple[str, int]:
        """Generate strategy-aware chain-of-thought reasoning"""
        # Extract current game state
        current_state = self._extract_current_state(game_history)
        board_state = self._parse_board_state(game_history)
        my_symbol = current_state['player_symbol']
        opponent_symbol = 'O' if my_symbol == 'X' else 'X'
        available_moves = current_state['available_moves']
        
        # 深层分析
        analysis = self._deep_board_analysis(board_state, my_symbol, opponent_symbol, available_moves)
        
        # 根据策略类型选择决策
        selected_move = self._strategic_decision(analysis, available_moves)
        
        # 生成详细的推理过程
        cot = self._generate_detailed_reasoning(analysis, selected_move, my_symbol, opponent_symbol)
        
        return cot, selected_move
    
    def _deep_board_analysis(self, board: list, my_symbol: str, opponent_symbol: str, available_moves: list) -> dict:
        """深层棋盘分析，包含多维度评估"""
        analysis = {
            'board_state': board.copy(),
            'my_symbol': my_symbol,
            'opponent_symbol': opponent_symbol,
            'available_moves': available_moves.copy(),
            'occupied_positions': {},
            'threats': {
                'my_winning_moves': [],
                'opponent_winning_moves': [],
                'my_fork_opportunities': [],
                'opponent_fork_opportunities': []
            },
            'strategic_positions': {
                'center': 4 in available_moves,
                'corners': [pos for pos in [0, 2, 6, 8] if pos in available_moves],
                'edges': [pos for pos in [1, 3, 5, 7] if pos in available_moves]
            },
            'game_phase': self.game_phase,
            'move_count': self.move_count
        }
        
        # 记录已占位置
        for i, cell in enumerate(board):
            if cell != ' ':
                analysis['occupied_positions'][i] = cell
        
        # 威胁分析
        analysis['threats']['my_winning_moves'] = self._find_all_winning_moves(board, my_symbol, available_moves)
        analysis['threats']['opponent_winning_moves'] = self._find_all_winning_moves(board, opponent_symbol, available_moves)
        analysis['threats']['my_fork_opportunities'] = self._find_all_fork_opportunities(board, my_symbol, available_moves)
        analysis['threats']['opponent_fork_opportunities'] = self._find_all_fork_opportunities(board, opponent_symbol, available_moves)
        
        # 每个可选位置的评估
        analysis['move_evaluations'] = {}
        for move in available_moves:
            analysis['move_evaluations'][move] = self._evaluate_move(board, move, my_symbol, opponent_symbol)
        
        return analysis
    
    def _strategic_decision(self, analysis: dict, available_moves: list) -> int:
        """根据策略类型和分析结果做出决策"""
        threats = analysis['threats']
        strategy = self.strategy
        
        # 1. 绝对优先：自己能获胜
        if threats['my_winning_moves']:
            return random.choice(threats['my_winning_moves'])
        
        # 2. 绝对优先：阻止对手获胜
        if threats['opponent_winning_moves']:
            if len(threats['opponent_winning_moves']) == 1:
                return threats['opponent_winning_moves'][0]
            else:
                # 多个威胁时，根据策略选择
                if strategy == StrategyType.CONSERVATIVE:
                    # 保守策略：阻止最危险的威胁
                    return self._select_most_critical_defense(analysis, threats['opponent_winning_moves'])
                else:
                    return random.choice(threats['opponent_winning_moves'])
        
        # 3. 策略导向的决策
        if strategy == StrategyType.AGGRESSIVE:
            return self._aggressive_decision(analysis, available_moves)
        elif strategy == StrategyType.CONSERVATIVE:
            return self._conservative_decision(analysis, available_moves)
        elif strategy == StrategyType.OPPORTUNISTIC:
            return self._opportunistic_decision(analysis, available_moves)
        else:  # BALANCED
            return self._balanced_decision(analysis, available_moves)
    
    def _aggressive_decision(self, analysis: dict, available_moves: list) -> int:
        """激进策略：优先进攻和创造威胁"""
        # 1. 寻找fork机会
        if analysis['threats']['my_fork_opportunities']:
            return random.choice(analysis['threats']['my_fork_opportunities'])
        
        # 2. 占据中心
        if analysis['strategic_positions']['center']:
            return 4
        
        # 3. 占据角落，增加获胜可能
        if analysis['strategic_positions']['corners']:
            return random.choice(analysis['strategic_positions']['corners'])
        
        # 4. 随机选择剩余位置
        return random.choice(available_moves)
    
    def _conservative_decision(self, analysis: dict, available_moves: list) -> int:
        """保守策略：优先防守和稳定发展"""
        # 1. 阻止对手fork
        if analysis['threats']['opponent_fork_opportunities']:
            return random.choice(analysis['threats']['opponent_fork_opportunities'])
        
        # 2. 稳步占据好位置
        if analysis['strategic_positions']['center']:
            return 4
        
        # 3. 优先占角落而非边缘
        if analysis['strategic_positions']['corners']:
            return random.choice(analysis['strategic_positions']['corners'])
        
        # 4. 最后选择边缘
        if analysis['strategic_positions']['edges']:
            return random.choice(analysis['strategic_positions']['edges'])
            
        return random.choice(available_moves)
    
    def _opportunistic_decision(self, analysis: dict, available_moves: list) -> int:
        """机会主义策略：根据局面灵活调整"""
        # 根据游戏阶段调整策略
        if analysis['game_phase'] == 'opening':
            # 开局阶段，优先占据关键位置
            if analysis['strategic_positions']['center']:
                return 4
            if analysis['strategic_positions']['corners']:
                return random.choice(analysis['strategic_positions']['corners'])
        
        elif analysis['game_phase'] == 'middle':
            # 中局阶段，平衡攻守
            if analysis['threats']['my_fork_opportunities']:
                return random.choice(analysis['threats']['my_fork_opportunities'])
            if analysis['threats']['opponent_fork_opportunities']:
                return random.choice(analysis['threats']['opponent_fork_opportunities'])
        
        # 默认选择评分最高的位置
        best_moves = self._get_best_evaluated_moves(analysis)
        return random.choice(best_moves)
    
    def _balanced_decision(self, analysis: dict, available_moves: list) -> int:
        """均衡策略：攻守兼备"""
        # 综合考虑所有因素，选择评分最高的位置
        best_moves = self._get_best_evaluated_moves(analysis)
        
        # 在最佳选择中引入少量随机性
        if len(best_moves) > 1:
            return random.choice(best_moves)
        
        return best_moves[0] if best_moves else random.choice(available_moves)
        """Generate detailed chain-of-thought reasoning for the current move"""
        # Extract current board state and available moves
        current_state = self._extract_current_state(game_history)
        
        # Parse the board for strategic analysis
        board_state = self._parse_board_state(game_history)
        my_symbol = current_state['player_symbol']
        opponent_symbol = 'O' if my_symbol == 'X' else 'X'
        available_moves = current_state['available_moves']
        
        # Strategic analysis
        winning_move = self._find_winning_move(board_state, my_symbol, available_moves)
        blocking_move = self._find_winning_move(board_state, opponent_symbol, available_moves)  # Block opponent's win
        fork_opportunity = self._find_fork_opportunity(board_state, my_symbol, available_moves)
        
        # Generate reasoning based on actual board analysis
        analysis_parts = []
        
        # Current board analysis
        analysis_parts.append(f"当前棋盘分析: 我是{my_symbol}，对手是{opponent_symbol}")
        analysis_parts.append(f"可选位置: {available_moves}")
        
        # Board state description
        occupied_positions = []
        for i in range(9):
            if board_state[i] != ' ':
                occupied_positions.append(f"位置{i}被{board_state[i]}占据")
        if occupied_positions:
            analysis_parts.append(f"已占位置: {', '.join(occupied_positions)}")
        
        # Strategic reasoning
        if winning_move is not None:
            analysis_parts.append(f"发现获胜机会: 在位置{winning_move}可以获胜")
            selected_move = winning_move
        elif blocking_move is not None:
            analysis_parts.append(f"需要阻止对手获胜: 对手在位置{blocking_move}可获胜，必须阻挡")
            selected_move = blocking_move
        elif fork_opportunity is not None:
            analysis_parts.append(f"发现fork机会: 位置{fork_opportunity}可创造多个获胜路径")
            selected_move = fork_opportunity
        else:
            # Strategic position selection
            center = 4
            corners = [0, 2, 6, 8]
            edges = [1, 3, 5, 7]
            
            if center in available_moves:
                analysis_parts.append("选择中心位置(4)：控制最多获胜路径")
                selected_move = center
            elif any(corner in available_moves for corner in corners):
                available_corners = [c for c in corners if c in available_moves]
                selected_move = available_corners[0]
                analysis_parts.append(f"选择角落位置({selected_move})：角落位置战略价值高")
            else:
                selected_move = available_moves[0]
                analysis_parts.append(f"选择边缘位置({selected_move})：当前最佳可选位置")
        
        # Format the complete reasoning
        cot = f"""
思考过程：
1. {analysis_parts[0]}
2. {analysis_parts[1]}
3. {analysis_parts[2] if len(analysis_parts) > 2 else "棋盘为空，开始游戏"}
4. 战略决策：{analysis_parts[3] if len(analysis_parts) > 3 else "选择最优位置"}

答案: [{selected_move}]"""
        
        return cot, selected_move
    
    def _parse_board_state(self, game_history: str) -> list:
        """Parse the current board state from game history"""
        # Initialize empty board
        board = [' '] * 9
        
        # Find the last board representation in the history
        lines = game_history.split('\n')
        board_found = False
        board_lines = []
        
        for i, line in enumerate(lines):
            if 'Current Board:' in line:
                # Get the next few lines containing the board
                board_lines = lines[i+2:i+7]  # Skip the empty line and get board lines
                board_found = True
                break
        
        if board_found and len(board_lines) >= 5:
            # Parse the board representation
            # Format is like: " X | 1 | 2 "
            try:
                row1 = board_lines[0].strip().split(' | ')
                row2 = board_lines[2].strip().split(' | ')  # Skip the separator line
                row3 = board_lines[4].strip().split(' | ')
                
                all_cells = row1 + row2 + row3
                for i, cell in enumerate(all_cells):
                    cell = cell.strip()
                    if cell in ['X', 'O']:
                        board[i] = cell
                    # else keep as ' ' (empty)
            except:
                pass  # Keep empty board if parsing fails
        
        return board
    
    def _find_winning_move(self, board: list, symbol: str, available_moves: list) -> int:
        """Find if there's a winning move for the given symbol"""
        # All possible winning combinations
        winning_combos = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
            [0, 4, 8], [2, 4, 6]              # Diagonals
        ]
        
        for combo in winning_combos:
            # Count symbols in this combo
            symbol_count = sum(1 for pos in combo if board[pos] == symbol)
            empty_positions = [pos for pos in combo if board[pos] == ' ']
            
            # If we have 2 symbols and 1 empty space, we can win
            if symbol_count == 2 and len(empty_positions) == 1:
                winning_pos = empty_positions[0]
                if winning_pos in available_moves:
                    return winning_pos
        
        return None
    
    def _find_fork_opportunity(self, board: list, symbol: str, available_moves: list) -> int:
        """Find fork opportunities (positions that create multiple winning threats)"""
        winning_combos = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
            [0, 4, 8], [2, 4, 6]              # Diagonals
        ]
        
        for pos in available_moves:
            if board[pos] == ' ':  # Should be empty
                # Try placing our symbol here
                test_board = board.copy()
                test_board[pos] = symbol
                
                # Count how many winning opportunities this creates
                winning_opportunities = 0
                for combo in winning_combos:
                    symbol_count = sum(1 for p in combo if test_board[p] == symbol)
                    empty_count = sum(1 for p in combo if test_board[p] == ' ')
                    # A winning opportunity is when we have 2 symbols and 1 empty space
                    if symbol_count == 2 and empty_count == 1:
                        winning_opportunities += 1
                
                # If this position creates 2 or more winning opportunities, it's a fork
                if winning_opportunities >= 2:
                    return pos
        
        return None

    def _find_all_winning_moves(self, board: list, symbol: str, available_moves: list) -> list:
        """找到所有能获胜的位置"""
        winning_moves = []
        winning_combos = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
            [0, 4, 8], [2, 4, 6]              # Diagonals
        ]
        
        for combo in winning_combos:
            symbol_count = sum(1 for pos in combo if board[pos] == symbol)
            empty_positions = [pos for pos in combo if board[pos] == ' ']
            
            if symbol_count == 2 and len(empty_positions) == 1:
                winning_pos = empty_positions[0]
                if winning_pos in available_moves:
                    winning_moves.append(winning_pos)
        
        return winning_moves

    def _find_all_fork_opportunities(self, board: list, symbol: str, available_moves: list) -> list:
        """找到所有fork机会"""
        fork_moves = []
        winning_combos = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
            [0, 4, 8], [2, 4, 6]              # Diagonals
        ]
        
        for pos in available_moves:
            if board[pos] == ' ':
                test_board = board.copy()
                test_board[pos] = symbol
                
                winning_opportunities = 0
                for combo in winning_combos:
                    symbol_count = sum(1 for p in combo if test_board[p] == symbol)
                    empty_count = sum(1 for p in combo if test_board[p] == ' ')
                    if symbol_count == 2 and empty_count == 1:
                        winning_opportunities += 1
                
                if winning_opportunities >= 2:
                    fork_moves.append(pos)
        
        return fork_moves

    def _get_best_evaluated_moves(self, analysis: dict) -> list:
        """获取评分最高的移动"""
        evaluations = analysis['move_evaluations']
        if not evaluations:
            return analysis['available_moves']
        
        max_score = max(evaluations.values())
        return [move for move, score in evaluations.items() if score == max_score]
    
    def _select_most_critical_defense(self, analysis: dict, threat_moves: list) -> int:
        """选择最关键的防御位置"""
        # 简单实现：选择第一个威胁
        return threat_moves[0]
    
    def _evaluate_move(self, board: list, move: int, my_symbol: str, opponent_symbol: str) -> float:
        """评估单个移动的价值"""
        score = 0.0
        
        # 基础位置价值
        position_values = {
            4: 3.0,  # 中心
            0: 2.0, 2: 2.0, 6: 2.0, 8: 2.0,  # 角落
            1: 1.0, 3: 1.0, 5: 1.0, 7: 1.0   # 边缘
        }
        score += position_values.get(move, 0)
        
        # 创造威胁的价值
        test_board = board.copy()
        test_board[move] = my_symbol
        
        # 检查是否创造了获胜威胁
        winning_threats = len(self._find_all_winning_moves(test_board, my_symbol, list(range(9))))
        score += winning_threats * 2.0
        
        # 检查是否创造了fork
        fork_opportunities = len(self._find_all_fork_opportunities(test_board, my_symbol, list(range(9))))
        score += fork_opportunities * 1.5
        
        return score
    
    def _generate_detailed_reasoning(self, analysis: dict, selected_move: int, my_symbol: str, opponent_symbol: str) -> str:
        """生成详细的推理过程，支持不同CoT长度控制"""
        if self.cot_length == CoTLengthType.TINY:
            return self._generate_tiny_cot(analysis, selected_move, my_symbol, opponent_symbol)
        elif self.cot_length == CoTLengthType.SHORT:
            return self._generate_short_cot(analysis, selected_move, my_symbol, opponent_symbol)
        elif self.cot_length == CoTLengthType.MEDIUM:
            return self._generate_medium_cot(analysis, selected_move, my_symbol, opponent_symbol)
        elif self.cot_length == CoTLengthType.LONG:
            return self._generate_long_cot(analysis, selected_move, my_symbol, opponent_symbol)
        elif self.cot_length == CoTLengthType.VERY_LONG:
            return self._generate_very_long_cot_original(analysis, selected_move, my_symbol, opponent_symbol)
        else:  # ULTRA_LONG
            return self._generate_ultra_long_cot_original(analysis, selected_move, my_symbol, opponent_symbol)
    
    def _generate_tiny_cot(self, analysis: dict, selected_move: int, my_symbol: str, opponent_symbol: str) -> str:
        """生成极短推理过程 (约100字符)"""
        threats = analysis['threats']
        if threats['my_winning_moves']:
            return f"\n思考：发现获胜机会！位置{selected_move}可以立即获胜，必须选择。当前{analysis['game_phase']}阶段，这是最佳时机，毫不犹豫地选择这个位置。\n答案: [{selected_move}]"
        elif threats['opponent_winning_moves']:
            return f"\n思考：对手威胁！必须阻止位置{threats['opponent_winning_moves'][0]}的获胜，选择{selected_move}防守。{self.strategy.value}策略下的最优选择，保持游戏平衡。\n答案: [{selected_move}]"
        elif selected_move == 4:
            return f"\n思考：占据中心位置{selected_move}最有价值，控制4条获胜线路。在{analysis['game_phase']}阶段建立优势，符合{self.strategy.value}策略，为后续发展奠定基础。\n答案: [{selected_move}]"
        else:
            return f"\n思考：选择位置{selected_move}。{analysis['game_phase']}阶段有{len(analysis['available_moves'])}个选择，此位置符合{self.strategy.value}策略需求，是当前最优决策。\n答案: [{selected_move}]"
    
    def _generate_short_cot(self, analysis: dict, selected_move: int, my_symbol: str, opponent_symbol: str) -> str:
        """生成简短推理过程 (约100-150字符)"""
        reasoning_parts = []
        
        # 基本分析
        reasoning_parts.append(f"【局面分析】{analysis['game_phase']}阶段第{analysis['move_count']}步，我是{my_symbol}，对手是{opponent_symbol}")
        reasoning_parts.append(f"【可选位置】{len(analysis['available_moves'])}个：{analysis['available_moves']}")
        
        # 威胁评估
        threats = analysis['threats']
        if threats['my_winning_moves']:
            reasoning_parts.append(f"【获胜机会】发现{len(threats['my_winning_moves'])}个获胜位置：{threats['my_winning_moves']}")
        elif threats['opponent_winning_moves']:
            reasoning_parts.append(f"【防守需求】必须阻止对手位置：{threats['opponent_winning_moves']}")
        else:
            reasoning_parts.append(f"【策略选择】采用{self.strategy.value}策略")
        
        # 决策说明
        if selected_move == 4:
            reasoning_parts.append("【决策】选择中心位置，控制最多线路")
        elif selected_move in [0, 2, 6, 8]:
            reasoning_parts.append("【决策】选择角落位置，建立优势")
        else:
            reasoning_parts.append("【决策】选择边缘位置，防守补强")
        
        # 格式化
        cot = "\n思考过程：\n"
        for i, part in enumerate(reasoning_parts, 1):
            cot += f"{i}. {part}\n"
        cot += f"\n答案: [{selected_move}]"
        
        return cot
        
    def _generate_medium_cot(self, analysis: dict, selected_move: int, my_symbol: str, opponent_symbol: str) -> str:
        """生成中等长度推理过程 (约400字符)"""
        reasoning_parts = []
        
        # 详细局面分析
        occupied_count = len(analysis['occupied_positions'])
        reasoning_parts.append(f"【详细局面分析】当前是{analysis['game_phase']}阶段第{analysis['move_count']}步，我是{my_symbol}，对手是{opponent_symbol}")
        reasoning_parts.append(f"棋盘已占据{occupied_count}个位置，剩余{len(analysis['available_moves'])}个可选位置：{analysis['available_moves']}")
        
        # 详细棋盘状态
        if analysis['occupied_positions']:
            pos_names = ['左上角', '上边', '右上角', '左边', '中心', '右边', '左下角', '下边', '右下角']
            pos_details = []
            for pos, symbol in analysis['occupied_positions'].items():
                pos_details.append(f"{pos_names[pos]}({pos})被{symbol}占据")
            reasoning_parts.append(f"【棋盘状态】{', '.join(pos_details)}")
        
        # 全面威胁评估
        threats = analysis['threats']
        if threats['my_winning_moves']:
            reasoning_parts.append(f"【获胜机会评估】发现{len(threats['my_winning_moves'])}个立即获胜机会：{threats['my_winning_moves']}，应该优先考虑")
        if threats['opponent_winning_moves']:
            reasoning_parts.append(f"【防守需求评估】对手有{len(threats['opponent_winning_moves'])}个获胜威胁：{threats['opponent_winning_moves']}，必须立即阻止")
        if threats['my_fork_opportunities']:
            reasoning_parts.append(f"【进攻机会评估】发现{len(threats['my_fork_opportunities'])}个fork创造机会：{threats['my_fork_opportunities']}，可以建立双重威胁")
        if threats['opponent_fork_opportunities']:
            reasoning_parts.append(f"【防守警告评估】对手有{len(threats['opponent_fork_opportunities'])}个fork威胁：{threats['opponent_fork_opportunities']}，需要预防")
        
        # 战略位置分析
        strategic = analysis['strategic_positions']
        if strategic['center']:
            reasoning_parts.append("【位置价值分析】中心位置(4)仍可用，控制4条获胜线路，战略价值最高")
        if strategic['corners']:
            reasoning_parts.append(f"【位置价值分析】角落位置{strategic['corners']}可用，每个控制3条线路，适合建立长期优势")
        if strategic['edges']:
            reasoning_parts.append(f"【位置价值分析】边缘位置{strategic['edges']}可用，每个控制2条线路，主要用于防守")
        
        # 策略导向分析
        strategy_explanations = {
            StrategyType.AGGRESSIVE: "激进型策略，优先寻求进攻机会，主动创造威胁，迫使对手防守",
            StrategyType.CONSERVATIVE: "保守型策略，优先确保防守稳固，避免给对手机会，稳扎稳打", 
            StrategyType.BALANCED: "均衡型策略，攻守兼备，根据局面灵活调整重点，既重视进攻也重视防守",
            StrategyType.OPPORTUNISTIC: "机会主义策略，灵活应对局面变化，抓住一切有利时机，适应性强"
        }
        reasoning_parts.append(f"【策略导向分析】采用{strategy_explanations[self.strategy]}")
        
        # 候选动作评分
        move_evals = analysis['move_evaluations']
        if len(move_evals) > 1:
            top_moves = sorted(move_evals.items(), key=lambda x: x[1], reverse=True)[:3]
            reasoning_parts.append(f"【候选动作评分】前三选择：{[f'位置{pos}({score:.1f}分)' for pos, score in top_moves]}")
        
        # 最终决策说明
        decision_reason = self._explain_decision(analysis, selected_move, threats)
        reasoning_parts.append(f"【最终决策说明】选择位置{selected_move}：{decision_reason}")
        
        # 格式化中等推理
        cot = "\n思考过程：\n"
        for i, part in enumerate(reasoning_parts, 1):
            cot += f"{i}. {part}\n"
        cot += f"\n答案: [{selected_move}]"
        
        return cot
        
    def _generate_long_cot(self, analysis: dict, selected_move: int, my_symbol: str, opponent_symbol: str) -> str:
        """生成详细推理过程 (约800-1200字符)"""
        reasoning_parts = []
        
        # 1. 详细局面分析
        occupied_count = len(analysis['occupied_positions'])
        reasoning_parts.append(f"【详细局面分析】当前是{analysis['game_phase']}阶段第{analysis['move_count']}步，我是{my_symbol}，对手是{opponent_symbol}")
        reasoning_parts.append(f"棋盘状态：已占{occupied_count}个位置，剩余{len(analysis['available_moves'])}个可选位置：{analysis['available_moves']}")
        
        # 2. 威胁全面评估
        threats = analysis['threats']
        if threats['my_winning_moves']:
            reasoning_parts.append(f"【获胜机会分析】发现{len(threats['my_winning_moves'])}个立即获胜机会：{threats['my_winning_moves']}，这些位置可以直接获胜")
        if threats['opponent_winning_moves']:
            reasoning_parts.append(f"【防守需求分析】对手有{len(threats['opponent_winning_moves'])}个获胜威胁：{threats['opponent_winning_moves']}，必须优先阻止")
        if threats['my_fork_opportunities']:
            reasoning_parts.append(f"【进攻机会分析】发现{len(threats['my_fork_opportunities'])}个fork创造机会：{threats['my_fork_opportunities']}，可以建立多重威胁")
        if threats['opponent_fork_opportunities']:
            reasoning_parts.append(f"【防守警告分析】对手有{len(threats['opponent_fork_opportunities'])}个fork威胁：{threats['opponent_fork_opportunities']}，需要预防")
        
        # 3. 战略位置价值分析
        strategic = analysis['strategic_positions']
        reasoning_parts.append("【战略位置价值分析】")
        if strategic['center']:
            reasoning_parts.append("  - 中心位置(4)仍可用：控制4条获胜线路，战略价值极高，是开局和中局的关键位置")
        if strategic['corners']:
            reasoning_parts.append(f"  - 角落位置{strategic['corners']}可用：每个角落控制3条线路，适合建立长期控制优势")
        if strategic['edges']:
            reasoning_parts.append(f"  - 边缘位置{strategic['edges']}可用：每个边缘控制2条线路，主要用于防守和补强")
        
        # 4. 策略深度思考
        strategy_explanations = {
            StrategyType.AGGRESSIVE: "激进型策略：优先寻求进攻机会，主动创造威胁，迫使对手进入防守状态，通过压迫获得主动权",
            StrategyType.CONSERVATIVE: "保守型策略：优先确保防守稳固，避免给对手任何获胜机会，稳扎稳打，通过耐心等待对手失误", 
            StrategyType.BALANCED: "均衡型策略：攻守兼备，根据当前局面灵活调整重点，既不放过进攻机会也不忽视防守需求",
            StrategyType.OPPORTUNISTIC: "机会主义策略：灵活应对局面变化，优先抓住当前最有利的机会，适应性强"
        }
        reasoning_parts.append(f"【策略指导思想】采用{strategy_explanations[self.strategy]}")
        
        # 5. 候选动作详细评分分析
        move_evals = analysis['move_evaluations']
        if len(move_evals) > 1:
            top_moves = sorted(move_evals.items(), key=lambda x: x[1], reverse=True)
            reasoning_parts.append(f"【候选动作评分】完整排序：{[f'位置{pos}({score:.1f}分)' for pos, score in top_moves]}")
            
            # 分析前三个选择
            for i, (pos, score) in enumerate(top_moves[:3]):
                if i == 0:
                    reasoning_parts.append(f"  - 首选位置{pos}(得分{score:.1f})：{self._get_position_analysis(pos, analysis)}")
                elif i == 1:
                    reasoning_parts.append(f"  - 次选位置{pos}(得分{score:.1f})：{self._get_position_analysis(pos, analysis)}")
                else:
                    reasoning_parts.append(f"  - 第三选择{pos}(得分{score:.1f})：{self._get_position_analysis(pos, analysis)}")
        
        # 6. 反事实推理分析
        alternatives = [move for move in analysis['available_moves'] if move != selected_move]
        if alternatives:
            alt_move = alternatives[0]  # 分析最佳替代方案
            reasoning_parts.append(f"【反事实推理】如果选择位置{alt_move}而非{selected_move}：")
            reasoning_parts.append(f"  - 可能的后果分析：虽然{alt_move}也是可行选择，但{selected_move}在当前策略框架下更优")
            reasoning_parts.append(f"  - 风险评估：选择{alt_move}可能会减少我们的控制力或给对手更多机会")
        
        # 7. 最终决策推理
        decision_context = self._get_decision_context(analysis, selected_move, threats)
        reasoning_parts.append(f"【最终决策推理】选择位置{selected_move}的综合考虑：")
        reasoning_parts.append(f"  - 主要原因：{decision_context['primary_reason']}")
        reasoning_parts.append(f"  - 战术价值：{decision_context['tactical_value']}")
        reasoning_parts.append(f"  - 长期影响：{decision_context['long_term_impact']}")
        
        # 格式化完整推理
        cot = "\n详细思考过程：\n"
        for i, part in enumerate(reasoning_parts, 1):
            cot += f"{i}. {part}\n"
        cot += f"\n最终答案: [{selected_move}]"
        
        return cot
    
    def _get_position_analysis(self, position: int, analysis: dict) -> str:
        """获取位置的详细分析"""
        if position == 4:
            return "中心位置，控制4条线路，战略价值最高"
        elif position in [0, 2, 6, 8]:
            return "角落位置，控制3条线路，建立长期优势"
        elif position in [1, 3, 5, 7]:
            return "边缘位置，控制2条线路，主要用于防守"
        else:
            return "战略价值待评估"
    
    def _get_decision_context(self, analysis: dict, selected_move: int, threats: dict) -> dict:
        """获取决策的详细上下文"""
        if selected_move in threats['my_winning_moves']:
            return {
                'primary_reason': '立即获胜机会，必须抓住',
                'tactical_value': '获得游戏胜利，价值无限',
                'long_term_impact': '游戏结束，完美结局'
            }
        elif selected_move in threats['opponent_winning_moves']:
            return {
                'primary_reason': '阻止对手获胜，防守必要',
                'tactical_value': '避免immediate失败，保持游戏继续',
                'long_term_impact': '维持游戏平衡，争取后续机会'
            }
        elif selected_move == 4:
            return {
                'primary_reason': '占据中心控制核心，最大化影响力',
                'tactical_value': '控制四条获胜线路，战略价值极高',
                'long_term_impact': '为后续进攻或防守奠定最佳基础'
            }
        elif selected_move in [0, 2, 6, 8]:
            return {
                'primary_reason': '角落位置建立根据地，长期控制',
                'tactical_value': '控制三条线路，创造多种可能',
                'long_term_impact': '为中后期创造更多战术选择'
            }
        else:
            return {
                'primary_reason': '最优可用选择，符合当前策略',
                'tactical_value': '保持局面平衡，避免劣势',
                'long_term_impact': '为后续发展保留灵活性'
            }
    
    def _generate_counterfactual_analysis(self, analysis: dict, selected_move: int, my_symbol: str, opponent_symbol: str) -> str:
        """生成反事实分析"""
        alternatives = [move for move in analysis['available_moves'] if move != selected_move]
        if not alternatives:
            return "【反事实分析】无其他选择"
        
        # 选择一个替代方案进行分析
        alt_move = random.choice(alternatives[:2])  # 分析前2个替代方案之一
        
        # 模拟替代选择的结果
        test_board = analysis['board_state'].copy()
        test_board[alt_move] = my_symbol
        
        alt_winning = len(self._find_all_winning_moves(test_board, my_symbol, list(range(9))))
        current_board = analysis['board_state'].copy()
        current_board[selected_move] = my_symbol
        current_winning = len(self._find_all_winning_moves(current_board, my_symbol, list(range(9))))
        
        if current_winning > alt_winning:
            return f"【反事实分析】若选择位置{alt_move}，将减少{current_winning - alt_winning}个获胜机会"
        elif current_winning < alt_winning:
            return f"【反事实分析】位置{alt_move}虽能创造更多威胁，但当前选择更符合策略"
        else:
            return f"【反事实分析】位置{alt_move}与当前选择战略价值相当"
    
    def _explain_decision(self, analysis: dict, selected_move: int, threats: dict) -> str:
        """解释最终决策的原因"""
        if selected_move in threats['my_winning_moves']:
            return "立即获胜，最优选择"
        elif selected_move in threats['opponent_winning_moves']:
            return "阻止对手获胜，防守必要"
        elif selected_move in threats['my_fork_opportunities']:
            return "创造fork机会，建立多重威胁"
        elif selected_move in threats['opponent_fork_opportunities']:
            return "破坏对手fork，防守策略"
        elif selected_move == 4:
            return "占据中心位置，控制最多获胜路径"
        elif selected_move in [0, 2, 6, 8]:
            return "占据角落位置，建立长期优势"
        else:
            return "在可选位置中的最优战略选择"
            
    def _generate_ultra_long_cot_original(self, analysis: dict, selected_move: int, my_symbol: str, opponent_symbol: str) -> str:
        """生成超长推理过程 (2500-4000+ tokens)"""
        reasoning_parts = []
        
        # 1. 超详细局面分析
        occupied_count = len(analysis['occupied_positions'])
        total_moves = 9 - len(analysis['available_moves'])
        remaining_moves = len(analysis['available_moves'])
        
        reasoning_parts.append(f"【深度局面分析】")
        reasoning_parts.append(f"  - 游戏阶段：{analysis['game_phase']}阶段，当前第{analysis['move_count']}步")
        reasoning_parts.append(f"  - 角色定位：我是{my_symbol}，对手是{opponent_symbol}")
        reasoning_parts.append(f"  - 进度分析：棋盘{occupied_count}/9位置已占，剩余{remaining_moves}个空位")
        reasoning_parts.append(f"  - 可选位置：{analysis['available_moves']}")
        
        # 详细棋盘状态和模式识别
        if analysis['occupied_positions']:
            reasoning_parts.append(f"【棋盘状态详解】")
            pos_names = ['左上角', '上边', '右上角', '左边', '中心', '右边', '左下角', '下边', '右下角']
            for pos, symbol in analysis['occupied_positions'].items():
                pos_name = pos_names[pos]
                reasoning_parts.append(f"  - {pos_name}位置({pos})：被{symbol}占据，控制线路分析...")
        
        # 2. 全方位威胁与机会评估
        threats = analysis['threats']
        reasoning_parts.append(f"【威胁与机会全面评估】")
        
        if threats['my_winning_moves']:
            reasoning_parts.append(f"  - 【立即获胜】发现{len(threats['my_winning_moves'])}个立即获胜机会：{threats['my_winning_moves']}")
            for move in threats['my_winning_moves']:
                reasoning_parts.append(f"    * 位置{move}可立即获胜，完成某条线的三连")
        else:
            reasoning_parts.append(f"  - 【立即获胜】当前无立即获胜机会")
            
        if threats['opponent_winning_moves']:
            reasoning_parts.append(f"  - 【防守紧急】对手有{len(threats['opponent_winning_moves'])}个获胜威胁：{threats['opponent_winning_moves']}")
            for move in threats['opponent_winning_moves']:
                reasoning_parts.append(f"    * 必须阻止位置{move}，否则对手下回合获胜")
        else:
            reasoning_parts.append(f"  - 【防守状态】对手暂无立即获胜威胁")
            
        if threats['my_fork_opportunities']:
            reasoning_parts.append(f"  - 【Fork机会】发现{len(threats['my_fork_opportunities'])}个创造双重威胁的机会：{threats['my_fork_opportunities']}")
            for move in threats['my_fork_opportunities']:
                reasoning_parts.append(f"    * 位置{move}可创造fork，同时威胁多条线")
        
        if threats['opponent_fork_opportunities']:
            reasoning_parts.append(f"  - 【Fork防御】对手有{len(threats['opponent_fork_opportunities'])}个fork威胁：{threats['opponent_fork_opportunities']}")
            for move in threats['opponent_fork_opportunities']:
                reasoning_parts.append(f"    * 需要防范位置{move}的双重威胁创造")
        
        # 3. 战略位置深度分析
        strategic = analysis['strategic_positions']
        reasoning_parts.append(f"【战略位置价值分析】")
        
        if strategic['center']:
            reasoning_parts.append(f"  - 【中心控制】位置4(中心)仍可用：")
            reasoning_parts.append(f"    * 控制4条获胜线路（行、列、两对角）")
            reasoning_parts.append(f"    * 战略价值最高，是开局和中局的关键位置")
            reasoning_parts.append(f"    * 占据后可对所有区域施加影响")
        else:
            reasoning_parts.append(f"  - 【中心控制】位置4已被占据，需要调整战略重心")
            
        if strategic['corners']:
            reasoning_parts.append(f"  - 【角落控制】可用角落位置{strategic['corners']}：")
            for corner in strategic['corners']:
                corner_names = {0: '左上角', 2: '右上角', 6: '左下角', 8: '右下角'}
                reasoning_parts.append(f"    * {corner_names[corner]}({corner})：控制3条线路，建立长期优势的好选择")
        
        if strategic['edges']:
            reasoning_parts.append(f"  - 【边缘控制】可用边缘位置{strategic['edges']}：")
            for edge in strategic['edges']:
                edge_names = {1: '上边', 3: '左边', 5: '右边', 7: '下边'}
                reasoning_parts.append(f"    * {edge_names[edge]}({edge})：控制2条线路，主要用于防守和补强")
        
        # 4. 策略哲学与深度思考
        strategy_philosophy = {
            StrategyType.AGGRESSIVE: {
                "core": "激进型策略：主动进攻，创造压力",
                "details": [
                    "优先寻求进攻机会，主动创造威胁局面",
                    "迫使对手进入防守状态，掌握游戏节奏",
                    "愿意承担适度风险以获取更大优势",
                    "重视主动权，不给对手喘息机会"
                ]
            },
            StrategyType.CONSERVATIVE: {
                "core": "保守型策略：稳扎稳打，确保安全",
                "details": [
                    "优先确保防守稳固，避免给对手机会",
                    "重视风险控制，每步都考虑安全性",
                    "稳扎稳打，通过减少失误来获得优势",
                    "耐心等待对手犯错，然后抓住机会"
                ]
            },
            StrategyType.BALANCED: {
                "core": "均衡型策略：攻守兼备，灵活应变",
                "details": [
                    "攻守兼备，根据局面情况调整重点",
                    "保持战略平衡，避免过于激进或保守",
                    "综合考虑进攻和防守的优先级",
                    "适应性强，能够应对各种局面变化"
                ]
            },
            StrategyType.OPPORTUNISTIC: {
                "core": "机会主义策略：灵活机动，抓住时机",
                "details": [
                    "灵活应对局面变化，没有固定模式",
                    "善于发现和利用对手的弱点",
                    "抓住一切有利时机，快速转换策略",
                    "适应性极强，能在任何情况下找到最优解"
                ]
            }
        }
        
        strategy_info = strategy_philosophy[self.strategy]
        reasoning_parts.append(f"【战略哲学与指导思想】")
        reasoning_parts.append(f"  - 核心理念：{strategy_info['core']}")
        for detail in strategy_info['details']:
            reasoning_parts.append(f"    * {detail}")
        
        # 5. 全面候选动作评估与排序
        move_evals = analysis['move_evaluations']
        reasoning_parts.append(f"【候选动作全面评估】")
        
        if len(move_evals) >= 1:
            all_moves = sorted(move_evals.items(), key=lambda x: x[1], reverse=True)
            reasoning_parts.append(f"  - 完整评分排序：{[f'位置{pos}({score:.1f}分)' for pos, score in all_moves]}")
            
            # 详细分析每个可能的动作
            for i, (pos, score) in enumerate(all_moves):
                rank_desc = ['首选', '次选', '第三选择', '第四选择', '第五选择', '第六选择', '第七选择', '第八选择', '最后选择'][i]
                reasoning_parts.append(f"  - 【{rank_desc}】位置{pos}(得分{score:.1f})：")
                reasoning_parts.append(f"    * {self._explain_move_score(pos, analysis)}")
                reasoning_parts.append(f"    * 风险评估：{self._evaluate_move_risk(pos, analysis, my_symbol, opponent_symbol)}")
                reasoning_parts.append(f"    * 后续影响：{self._analyze_move_consequences(pos, analysis, my_symbol, opponent_symbol)}")
        
        # 6. 深度反事实与假设分析
        reasoning_parts.append(f"【深度反事实分析】")
        reasoning_parts.append(self._generate_comprehensive_counterfactual_analysis(analysis, selected_move, my_symbol, opponent_symbol))
        
        # 7. 游戏树与前瞻搜索
        reasoning_parts.append(f"【游戏树前瞻搜索】")
        reasoning_parts.append(self._generate_deep_game_tree_analysis(analysis, selected_move, my_symbol, opponent_symbol))
        
        # 8. 心理战与对手建模
        reasoning_parts.append(f"【对手行为模式分析】")
        reasoning_parts.append(self._analyze_opponent_behavior(analysis, my_symbol, opponent_symbol))
        
        # 9. 最终决策综合推理
        decision_reason = self._explain_comprehensive_decision(analysis, selected_move, threats)
        reasoning_parts.append(f"【最终决策综合推理】选择位置{selected_move}：")
        reasoning_parts.append(f"  - 主要理由：{decision_reason}")
        reasoning_parts.append(f"  - 风险收益比：{self._calculate_risk_reward_ratio(selected_move, analysis)}")
        reasoning_parts.append(f"  - 预期后续发展：{self._predict_game_development(selected_move, analysis, my_symbol, opponent_symbol)}")
        
        # 格式化超长推理
        cot = "\n【深度思考过程】：\n"
        for i, part in enumerate(reasoning_parts, 1):
            cot += f"{i}. {part}\n"
        cot += f"\n【最终答案】: [{selected_move}]"
        
        return cot
        
    # 辅助方法用于ultra_long_cot
    def _explain_move_score(self, pos: int, analysis: dict) -> str:
        """解释移动评分的原因"""
        strategic = analysis['strategic_positions']
        threats = analysis['threats']
        
        reasons = []
        if pos in threats['my_winning_moves']:
            reasons.append("立即获胜机会")
        if pos in threats['opponent_winning_moves']:
            reasons.append("必须防守位置")
        if pos in threats['my_fork_opportunities']:
            reasons.append("可创造fork机会")
        if pos == 4 and pos in analysis['available_moves']:
            reasons.append("中心位置，控制4条线")
        elif pos in [0, 2, 6, 8] and pos in strategic['corners']:
            reasons.append("角落位置，控制3条线")
        elif pos in [1, 3, 5, 7] and pos in strategic['edges']:
            reasons.append("边缘位置，控制2条线")
        
        return "，".join(reasons) if reasons else "常规战略位置"
    
    def _evaluate_move_risk(self, pos: int, analysis: dict, my_symbol: str, opponent_symbol: str) -> str:
        """评估移动的风险"""
        threats = analysis['threats']
        
        if pos in threats['opponent_winning_moves']:
            return "低风险，必要防守"
        elif pos in threats['my_winning_moves']:
            return "无风险，立即获胜"
        elif pos in threats['opponent_fork_opportunities']:
            return "中等风险，可能给对手创造机会"
        else:
            return "低风险，常规布局"
    
    def _analyze_move_consequences(self, pos: int, analysis: dict, my_symbol: str, opponent_symbol: str) -> str:
        """分析移动的后续影响"""
        strategic = analysis['strategic_positions']
        
        if pos == 4:
            return "控制棋盘中心，影响全局战略"
        elif pos in [0, 2, 6, 8]:
            return "建立角落优势，为后续布局打基础"
        elif pos in [1, 3, 5, 7]:
            return "加强边缘控制，辅助主要战略"
        else:
            return "维持当前局面，等待更好机会"
    
    def _generate_comprehensive_counterfactual_analysis(self, analysis: dict, selected_move: int, my_symbol: str, opponent_symbol: str) -> str:
        """生成全面的反事实分析"""
        alternatives = [move for move in analysis['available_moves'] if move != selected_move]
        if not alternatives:
            return "无其他可选方案进行对比分析"
        
        analysis_parts = []
        for alt_move in alternatives[:3]:  # 分析前3个替代方案
            risk = self._evaluate_move_risk(alt_move, analysis, my_symbol, opponent_symbol)
            consequence = self._analyze_move_consequences(alt_move, analysis, my_symbol, opponent_symbol)
            analysis_parts.append(f"  - 如选择位置{alt_move}：风险{risk}，后果{consequence}")
        
        return "\n".join(analysis_parts)
    
    def _generate_deep_game_tree_analysis(self, analysis: dict, selected_move: int, my_symbol: str, opponent_symbol: str) -> str:
        """生成深度游戏树分析"""
        remaining_moves = len(analysis['available_moves'])
        analysis_depth = min(3, remaining_moves - 1)  # 最多分析3步
        
        tree_analysis = []
        tree_analysis.append(f"  - 前瞻深度：{analysis_depth}步")
        tree_analysis.append(f"  - 当前选择位置{selected_move}后，对手可能的回应：")
        
        # 模拟对手可能的回应
        opponent_moves = [m for m in analysis['available_moves'] if m != selected_move][:3]
        for opp_move in opponent_moves:
            tree_analysis.append(f"    * 对手选择{opp_move}：局面评估为中性，需要进一步应对")
        
        return "\n".join(tree_analysis)
    
    def _analyze_opponent_behavior(self, analysis: dict, my_symbol: str, opponent_symbol: str) -> str:
        """分析对手行为模式"""
        behavior_analysis = []
        
        # 基于已有棋子分析对手风格
        opponent_positions = [pos for pos, symbol in analysis['occupied_positions'].items() if symbol == opponent_symbol]
        
        if 4 in opponent_positions:
            behavior_analysis.append("  - 对手采用中心控制策略，偏向进攻型")
        elif any(pos in [0, 2, 6, 8] for pos in opponent_positions):
            behavior_analysis.append("  - 对手采用角落布局，偏向稳健型")
        else:
            behavior_analysis.append("  - 对手策略暂不明确，需要观察更多回合")
        
        behavior_analysis.append("  - 预测对手下一步倾向：根据已有模式推断")
        
        return "\n".join(behavior_analysis)
    
    def _explain_comprehensive_decision(self, analysis: dict, selected_move: int, threats: dict) -> str:
        """全面解释决策理由"""
        reasons = []
        
        if selected_move in threats['my_winning_moves']:
            reasons.append("抓住立即获胜机会")
        elif selected_move in threats['opponent_winning_moves']:
            reasons.append("阻止对手获胜威胁")
        elif selected_move in threats['my_fork_opportunities']:
            reasons.append("创造双重威胁机会")
        elif selected_move == 4:
            reasons.append("占据战略要地中心位置")
        elif selected_move in [0, 2, 6, 8]:
            reasons.append("建立角落优势")
        else:
            reasons.append("基于当前局面的最优选择")
        
        return "，".join(reasons) if reasons else "战略考虑"
    
    def _calculate_risk_reward_ratio(self, selected_move: int, analysis: dict) -> str:
        """计算风险收益比"""
        threats = analysis['threats']
        
        if selected_move in threats['my_winning_moves']:
            return "极高收益，无风险"
        elif selected_move in threats['opponent_winning_moves']:
            return "高收益（防守成功），低风险"
        elif selected_move == 4:
            return "高收益，中等风险"
        else:
            return "中等收益，低风险"
    
    def _predict_game_development(self, selected_move: int, analysis: dict, my_symbol: str, opponent_symbol: str) -> str:
        """预测游戏发展"""
        remaining = len(analysis['available_moves']) - 1
        
        if remaining <= 2:
            return "游戏即将结束，关键在于执行力"
        elif remaining <= 4:
            return "进入中后期，每步都至关重要"
        else:
            return "仍在开局阶段，重在布局和控制"
    
    def _generate_extended_counterfactual_analysis(self, analysis: dict, selected_move: int, my_symbol: str, opponent_symbol: str) -> str:
        """生成扩展反事实分析"""
        alternatives = [move for move in analysis['available_moves'] if move != selected_move]
        if not alternatives:
            return "【反事实分析】无其他选择可供对比"
        
        # 选择1-2个替代方案进行详细分析
        alt_moves = alternatives[:2]
        analysis_parts = []
        
        for alt_move in alt_moves:
            score = analysis['move_evaluations'][alt_move]
            selected_score = analysis['move_evaluations'][selected_move]
            
            if score >= selected_score:
                comparison = "与当前选择价值相当"
            else:
                comparison = "价值略低于当前选择"
            
            reason = self._explain_move_score(alt_move, analysis)
            analysis_parts.append(f"如选择位置{alt_move}(得分{score:.1f})：{reason}，{comparison}")
        
        return "【反事实分析】" + "；".join(analysis_parts)
    
    def _generate_game_tree_analysis(self, analysis: dict, selected_move: int, my_symbol: str, opponent_symbol: str) -> str:
        """生成游戏树分析"""
        remaining_moves = len(analysis['available_moves']) - 1
        
        if remaining_moves <= 1:
            return "【游戏树分析】游戏即将结束，无需深度前瞻"
        
        tree_parts = []
        tree_parts.append("基于当前选择，分析可能的后续发展：")
        
        # 模拟对手最可能的几个回应
        opponent_likely_moves = [m for m in analysis['available_moves'] if m != selected_move][:2]
        
        for opp_move in opponent_likely_moves:
            tree_parts.append(f"对手选择{opp_move} → 局面转入下一阶段，需要灵活应对")
        
        return "【游戏树分析】" + "；".join(tree_parts)
    
    def _generate_very_long_cot_original(self, analysis: dict, selected_move: int, my_symbol: str, opponent_symbol: str) -> str:
        """生成很长推理过程 (约2000字符)"""
        reasoning_parts = []
        
        # 1. 超详细局面分析
        occupied_count = len(analysis['occupied_positions'])
        reasoning_parts.append(f"【超详细局面分析】当前处于{analysis['game_phase']}阶段第{analysis['move_count']}步，游戏进程{analysis['move_count']}/9步，完成度{analysis['move_count']/9*100:.1f}%")
        reasoning_parts.append(f"对战双方：我方执{my_symbol}，对手执{opponent_symbol}，棋盘已占据{occupied_count}个位置，剩余{len(analysis['available_moves'])}个可选位置：{analysis['available_moves']}")
        
        # 详细棋盘状态描述
        if analysis['occupied_positions']:
            pos_names = ['左上角', '上边', '右上角', '左边', '中心', '右边', '左下角', '下边', '右下角']
            pos_descriptions = []
            for pos, symbol in analysis['occupied_positions'].items():
                pos_descriptions.append(f"{pos_names[pos]}位置({pos})被{symbol}占据，影响{self._count_lines_for_position(pos)}条获胜线路")
            reasoning_parts.append(f"【详细棋盘状态】{'; '.join(pos_descriptions)}")
        else:
            reasoning_parts.append(f"【详细棋盘状态】棋盘完全空白，所有9个位置均可选择，这是游戏的初始状态")
        
        # 2. 全方位威胁态势评估
        threats = analysis['threats']
        reasoning_parts.append("【全方位威胁态势评估】")
        if threats['my_winning_moves']:
            reasoning_parts.append(f"  ◆ 我方获胜机会：发现{len(threats['my_winning_moves'])}个立即获胜位置{threats['my_winning_moves']}，这些位置将直接决定游戏胜负，必须优先考虑")
            for move in threats['my_winning_moves']:
                reasoning_parts.append(f"    - 位置{move}：可以完成获胜线路，立即结束游戏并获得胜利")
        if threats['opponent_winning_moves']:
            reasoning_parts.append(f"  ◆ 对手获胜威胁：检测到{len(threats['opponent_winning_moves'])}个对手获胜位置{threats['opponent_winning_moves']}，必须立即采取防御措施阻止")
            for move in threats['opponent_winning_moves']:
                reasoning_parts.append(f"    - 位置{move}：如不阻止，对手下一步可获胜，防守刻不容缓")
        if threats['my_fork_opportunities']:
            reasoning_parts.append(f"  ◆ 我方fork进攻：识别出{len(threats['my_fork_opportunities'])}个fork创造位置{threats['my_fork_opportunities']}，可建立多重同时威胁，迫使对手无法同时防守")
        if threats['opponent_fork_opportunities']:
            reasoning_parts.append(f"  ◆ 对手fork威胁：预警{len(threats['opponent_fork_opportunities'])}个对手fork位置{threats['opponent_fork_opportunities']}，需要提前预防和破坏对手的双重威胁计划")
        
        if not any([threats['my_winning_moves'], threats['opponent_winning_moves'], threats['my_fork_opportunities'], threats['opponent_fork_opportunities']]):
            reasoning_parts.append("  ◆ 威胁评估结果：当前无直接威胁，可以专注于战略布局和位置价值最大化，这是发展长期优势的好时机")
        
        # 3. 战略地形价值深度分析
        strategic = analysis['strategic_positions']
        reasoning_parts.append("【战略地形价值深度分析】")
        if strategic['center']:
            reasoning_parts.append("  ◆ 中心要塞价值分析：位置4(正中心)控制水平线(3-4-5)、垂直线(1-4-7)、主对角线(0-4-8)、副对角线(2-4-6)共4条获胜线路，是绝对的战略核心")
            reasoning_parts.append("    - 中心位置优势：最大化控制力，为后续发展提供最多选择，是开局的理想选择")
        if strategic['corners']:
            reasoning_parts.append(f"  ◆ 角落堡垒价值分析：{strategic['corners']}位置详细评估")
            reasoning_parts.append("    - 角落位置(0,2,6,8)特点：每个角落控制3条获胜线路，适合建立长期控制优势和防御根据地")
            reasoning_parts.append("    - 角落战略意义：虽然控制线路较少，但不易被攻击，适合稳健发展")
        if strategic['edges']:
            reasoning_parts.append(f"  ◆ 边缘连接价值分析：{strategic['edges']}位置战术评估")
            reasoning_parts.append("    - 边缘位置(1,3,5,7)特点：每个边缘控制2条获胜线路，主要用于防守连接和战术补强")
            reasoning_parts.append("    - 边缘战术作用：连接角落和中心，在特定情况下可发挥关键的桥梁作用")
        
        # 4. 策略哲学深度思考
        strategy_philosophies = {
            StrategyType.AGGRESSIVE: "激进型策略哲学深度解析：秉承进攻是最好的防守理念，优先寻求进攻机会，主动创造威胁，通过持续压迫迫使对手进入被动防守状态，从而获得主动权和心理优势。该策略适合快速决战，通过强势开局建立不可逆转的优势",
            StrategyType.CONSERVATIVE: "保守型策略哲学深度解析：以稳健为核心思想，优先确保防守的绝对稳固，避免给对手任何获胜机会，通过稳扎稳打和耐心等待对手的失误来寻求最终胜机。该策略强调风险控制，适合长期博弈和耐力战", 
            StrategyType.BALANCED: "均衡型策略哲学深度解析：攻守兼备的中庸之道，根据当前局面的具体情况灵活调整重点，既不放过任何进攻机会也不忽视防守需求，追求整体最优解。该策略适应性强，能够应对各种复杂局面",
            StrategyType.OPPORTUNISTIC: "机会主义策略哲学深度解析：高度灵活的适应性策略，根据局面变化随时调整战术，优先抓住当前最有利的机会，不拘泥于固定模式，善于变通。该策略强调时机把握和灵活应变"
        }
        reasoning_parts.append(f"【策略哲学深度思考】{strategy_philosophies[self.strategy]}")
        
        # 5. 候选动作全面评分分析
        move_evals = analysis['move_evaluations']
        if len(move_evals) > 1:
            all_moves = sorted(move_evals.items(), key=lambda x: x[1], reverse=True)
            reasoning_parts.append(f"【候选动作全面评分】完整排序结果：{[f'位置{pos}({score:.2f}分)' for pos, score in all_moves]}")
            
            # 详细分析前四个选择
            rank_names = ["首选方案", "次选方案", "第三方案", "第四方案"]
            for i, (pos, score) in enumerate(all_moves[:4]):
                if i < len(rank_names):
                    position_analysis = self._get_detailed_position_analysis_very_long(pos, analysis, threats)
                    reasoning_parts.append(f"  ◆ {rank_names[i]}位置{pos}(评分{score:.2f})：{position_analysis}")
                    reasoning_parts.append(f"    - 评分依据：基于威胁价值、战略位置、控制线路数量等多维度综合评估")
        
        # 6. 深度反事实推理分析
        alternatives = [move for move in analysis['available_moves'] if move != selected_move]
        if alternatives:
            reasoning_parts.append("【深度反事实推理分析】如果选择其他位置的后果预测和风险评估：")
            for i, alt_move in enumerate(alternatives[:3]):
                reasoning_parts.append(f"  ◆ 假设选择位置{alt_move}：")
                reasoning_parts.append(f"    - 即时影响：会产生不同的局面发展轨迹，改变棋盘控制格局")
                reasoning_parts.append(f"    - 中期后果：可能影响后续2-3步的战术选择和威胁创造能力")
                reasoning_parts.append(f"    - 风险评估：相比选择{selected_move}，可能减少我们的棋盘控制力或给对手创造更多反击机会")
        
        # 7. 最终决策综合论证
        decision_context = self._get_decision_context(analysis, selected_move, threats)
        reasoning_parts.append(f"【最终决策综合论证】选择位置{selected_move}的多维度深度分析：")
        reasoning_parts.append(f"  ◆ 核心驱动因素：{decision_context['primary_reason']}，这是决策的根本依据")
        reasoning_parts.append(f"  ◆ 战术层面价值：{decision_context['tactical_value']}，体现在即时的棋盘影响")
        reasoning_parts.append(f"  ◆ 战略层面意义：{decision_context['long_term_impact']}，关系到整个游戏的发展方向")
        reasoning_parts.append(f"  ◆ 风险收益分析：该选择在当前局面下风险最小，收益最大，符合理性决策原则")
        
        # 8. 后续发展预测
        reasoning_parts.append(f"【后续发展预测】选择位置{selected_move}后的局面演化分析：")
        if selected_move in threats['my_winning_moves']:
            reasoning_parts.append("  ◆ 游戏结果：立即获胜，游戏结束，完美结局")
        elif selected_move in threats['opponent_winning_moves']:
            reasoning_parts.append("  ◆ 防守效果：成功阻止对手获胜，游戏继续，需继续密切关注后续威胁发展")
        else:
            reasoning_parts.append("  ◆ 局面演化：棋盘将进入新的平衡状态，需根据对手响应调整后续策略")
            reasoning_parts.append("  ◆ 后续重点：继续监控威胁发展，寻找进攻机会，保持战略主动性")
        
        # 格式化超长推理
        cot = "\n超详细思考过程：\n"
        for i, part in enumerate(reasoning_parts, 1):
            cot += f"{i}. {part}\n"
        cot += f"\n经过全方位深度分析，最终答案: [{selected_move}]"
        
        return cot
    
    def _count_lines_for_position(self, pos: int) -> int:
        """计算位置影响的获胜线路数量"""
        if pos == 4:  # 中心
            return 4
        elif pos in [0, 2, 6, 8]:  # 角落
            return 3
        else:  # 边缘
            return 2
    
    def _get_detailed_position_analysis_very_long(self, position: int, analysis: dict, threats: dict) -> str:
        """获取位置的超详细分析"""
        if position in threats['my_winning_moves']:
            return "立即获胜位置，绝对优先选择，可直接结束游戏并获得胜利"
        elif position in threats['opponent_winning_moves']:
            return "对手获胜威胁位置，必须阻止，防守价值极高，关系游戏生死"
        elif position == 4:
            return "中心位置，控制4条线路，战略价值最高，适合开局或中局控制，为后续发展提供最大灵活性"
        elif position in [0, 2, 6, 8]:
            return "角落位置，控制3条线路，建立长期优势，有利于后期发展，防守相对稳固"
        elif position in [1, 3, 5, 7]:
            return "边缘位置，控制2条线路，主要用于防守连接和战术补强，在特定情况下可发挥桥梁作用"
        else:
            return "位置价值需要根据具体局面进一步评估，考虑与现有布局的协调性"
            for i, alt_move in enumerate(alternatives[:2]):
                reasoning_parts.append(f"  ◆ 假设选择位置{alt_move}：会产生不同的局面发展轨迹，虽然也是可行选择，但在当前策略框架和威胁评估下，位置{selected_move}具有更高的综合价值和长期优势")
        
        # 7. 终极决策综合框架
        decision_context = self._get_decision_context(analysis, selected_move, threats)
        reasoning_parts.append(f"【终极决策综合框架】选择位置{selected_move}的多维度综合考量：")
        reasoning_parts.append(f"  ◆ 核心驱动因素：{decision_context['primary_reason']}")
        reasoning_parts.append(f"  ◆ 战术层面价值：{decision_context['tactical_value']}")
        reasoning_parts.append(f"  ◆ 战略层面影响：{decision_context['long_term_impact']}")
        reasoning_parts.append(f"  ◆ 风险收益评估：这个选择在当前局面下能够最大化我方优势，同时最小化潜在风险")
        
        # 8. 预期发展轨迹
        reasoning_parts.append(f"【预期发展轨迹】选择位置{selected_move}后的局面预测：")
        if selected_move in threats['my_winning_moves']:
            reasoning_parts.append("  ◆ 游戏将立即结束，我方获得胜利，这是最理想的结果")
        elif selected_move in threats['opponent_winning_moves']:
            reasoning_parts.append("  ◆ 成功阻止对手获胜威胁，游戏继续进行，为后续发展争取时间和机会")
        else:
            reasoning_parts.append("  ◆ 局面将进入新的平衡状态，需要根据对手的响应来调整后续策略和战术重点")
        
        # 格式化超长推理
        cot = "\n超详细思考过程：\n"
        for i, part in enumerate(reasoning_parts, 1):
            cot += f"{i}. {part}\n"
        cot += f"\n经过深度分析和多维度评估，最终答案: [{selected_move}]"
        
        return cot
    
    def _get_decision_context(self, analysis: dict, selected_move: int, threats: dict) -> dict:
        """获取决策上下文信息"""
        if selected_move in threats['my_winning_moves']:
            return {
                'primary_reason': '直接获胜机会，这是最高优先级的决策依据',
                'tactical_value': '立即完成游戏获胜，无需考虑其他因素',
                'long_term_impact': '完美结局，展现优秀的战术执行力'
            }
        elif selected_move in threats['opponent_winning_moves']:
            return {
                'primary_reason': '阻止对手获胜，这是紧急防御需求',
                'tactical_value': '成功防守关键位置，避免即时失败',
                'long_term_impact': '为后续发展争取时间和机会'
            }
        elif selected_move == 4:  # 中心位置
            return {
                'primary_reason': '控制战略核心，获得最大棋盘影响力',
                'tactical_value': '控制4条获胜线路，为多种战术发展铺路',
                'long_term_impact': '建立持久的战略优势，增强胜率'
            }
        elif selected_move in [0, 2, 6, 8]:  # 角落
            return {
                'primary_reason': '占据角落堡垒，建立稳固根据地',
                'tactical_value': '控制3条获胜线路，增强防守能力',
                'long_term_impact': '为持久战略发展奠定基础'
            }
        else:  # 边缘位置
            return {
                'primary_reason': '选择边缘位置，进行战术补强',
                'tactical_value': '控制2条获胜线路，配合整体布局',
                'long_term_impact': '完善棋盘控制网络，增强整体协调性'
            }
    
    def _get_detailed_position_analysis_very_long(self, position: int, analysis: dict, threats: dict) -> str:
        """获取位置的超详细分析（用于very_long CoT）"""
        if position in threats['my_winning_moves']:
            return "立即获胜位置，绝对优先选择，将直接结束游戏并获得胜利"
        elif position in threats['opponent_winning_moves']:
            return "对手获胜威胁位置，必须立即阻止，否则下一步对手就会获胜"
        elif position == 4:
            return "中心核心位置，控制4条获胜线路，战略价值最高，适合开局占据或中局控制"
        elif position in [0, 2, 6, 8]:
            return "角落重要位置，控制3条获胜线路，有利于建立长期优势和防御根据地"
        elif position in [1, 3, 5, 7]:
            return "边缘辅助位置，控制2条获胜线路，主要用于防守连接和战术补强"
        else:
            return "特殊位置，需要根据具体局面进一步分析其战略价值和战术意义"
        threats = analysis['threats']
        reasoning_parts.append("【全方位威胁分析】")
        if threats['my_winning_moves']:
            reasoning_parts.append(f"  - 我方获胜机会：发现{len(threats['my_winning_moves'])}个立即获胜位置{threats['my_winning_moves']}，这些位置可以直接结束游戏并获得胜利")
        if threats['opponent_winning_moves']:
            reasoning_parts.append(f"  - 对手获胜威胁：对手有{len(threats['opponent_winning_moves'])}个获胜威胁位置{threats['opponent_winning_moves']}，如果不加阻止将立即失败")
        if threats['my_fork_opportunities']:
            reasoning_parts.append(f"  - 我方fork机会：发现{len(threats['my_fork_opportunities'])}个fork创造机会{threats['my_fork_opportunities']}，可以建立多重获胜威胁，迫使对手无法完全防守")
        if threats['opponent_fork_opportunities']:
            reasoning_parts.append(f"  - 对手fork威胁：对手有{len(threats['opponent_fork_opportunities'])}个潜在fork威胁{threats['opponent_fork_opportunities']}，需要提前预防和化解")
        
        # 3. 深度战略位置价值分析
        strategic = analysis['strategic_positions']
        reasoning_parts.append("【深度战略位置价值分析】")
        if strategic['center']:
            reasoning_parts.append("  - 中心位置(4)控制分析：中心位置控制水平、垂直、对角线共4条获胜线路，是整个棋盘的战略核心，占据此位置可以最大化后续选择的灵活性和威胁创造能力")
        if strategic['corners']:
            reasoning_parts.append(f"  - 角落位置{strategic['corners']}价值分析：每个角落位置控制3条获胜线路(一条水平、一条垂直、一条对角线)，适合建立长期控制优势和创造多线威胁")
        if strategic['edges']:
            reasoning_parts.append(f"  - 边缘位置{strategic['edges']}价值分析：每个边缘位置控制2条获胜线路，主要价值在于防守和补强，以及在特定情况下的战术配合")
        
        # 4. 深层策略哲学思考
        strategy_philosophies = {
            StrategyType.AGGRESSIVE: "激进型策略哲学：秉承进攻是最好的防守理念，优先寻求进攻机会，主动创造威胁，通过持续压迫迫使对手进入被动防守状态，从而获得主动权和心理优势",
            StrategyType.CONSERVATIVE: "保守型策略哲学：以稳健为核心，优先确保防守的绝对稳固，避免给对手任何获胜机会，通过稳扎稳打和耐心等待对手的失误来寻求胜机", 
            StrategyType.BALANCED: "均衡型策略哲学：攻守兼备的中庸之道，根据当前局面的具体情况灵活调整重点，既不放过任何进攻机会也不忽视防守需求，追求整体最优解",
            StrategyType.OPPORTUNISTIC: "机会主义策略哲学：高度灵活的适应性策略，根据局面变化随时调整战术，优先抓住当前最有利的机会，不拘泥于固定模式"
        }
        reasoning_parts.append(f"【深层策略哲学】{strategy_philosophies[self.strategy]}")
        
        # 5. 全候选动作深度评分分析
        move_evals = analysis['move_evaluations']
        if len(move_evals) > 1:
            all_moves = sorted(move_evals.items(), key=lambda x: x[1], reverse=True)
            reasoning_parts.append(f"【全候选动作深度评分】完整排序结果：{[f'位置{pos}({score:.2f}分)' for pos, score in all_moves]}")
            
            # 详细分析每个主要候选位置
            for i, (pos, score) in enumerate(all_moves):
                if i < 4:  # 分析前4个选择
                    rank_names = ["首选", "次选", "第三选择", "第四选择"]
                    detailed_analysis = self._get_detailed_position_analysis(pos, analysis, threats)
                    reasoning_parts.append(f"  - {rank_names[i]}位置{pos}(得分{score:.2f})：{detailed_analysis}")
        
        # 6. 多层次反事实推理
        alternatives = [move for move in analysis['available_moves'] if move != selected_move]
        if alternatives:
            reasoning_parts.append("【多层次反事实推理分析】")
            for i, alt_move in enumerate(alternatives[:3]):
                reasoning_parts.append(f"  - 假设选择位置{alt_move}：会产生不同的局面发展轨迹，虽然也是可行选择，但在当前策略框架和威胁评估下，位置{selected_move}具有更高的综合价值")
                reasoning_parts.append(f"    风险评估：选择{alt_move}可能减少我们的棋盘控制力，或给对手创造更多反击机会，不如当前选择稳妥")
        
        # 7. 终极决策推理框架
        decision_framework = self._get_ultimate_decision_framework(analysis, selected_move, threats)
        reasoning_parts.append(f"【终极决策推理框架】选择位置{selected_move}的多维度综合考量：")
        reasoning_parts.append(f"  - 核心驱动因素：{decision_framework['core_driver']}")
        reasoning_parts.append(f"  - 战术层面价值：{decision_framework['tactical_value']}")
        reasoning_parts.append(f"  - 战略层面影响：{decision_framework['strategic_impact']}")
        reasoning_parts.append(f"  - 心理层面考虑：{decision_framework['psychological_factor']}")
        reasoning_parts.append(f"  - 长远发展前景：{decision_framework['long_term_prospect']}")
        
        # 格式化完整推理
        cot = "\n超详细思考过程：\n"
        for i, part in enumerate(reasoning_parts, 1):
            cot += f"{i}. {part}\n"
        cot += f"\n最终答案: [{selected_move}]"
        
        return cot
    
    def _generate_ultra_long_cot(self, analysis: dict, selected_move: int, my_symbol: str, opponent_symbol: str) -> str:
        """生成超长推理过程 (约4000字符)"""
        reasoning_parts = []
        
        # 1. 全景式局面深度解析
        occupied_count = len(analysis['occupied_positions'])
        total_positions = 9
        game_progress_percent = (analysis['move_count'] / total_positions) * 100
        
        reasoning_parts.append(f"【全景式局面深度解析】")
        reasoning_parts.append(f"游戏进程：当前处于{analysis['game_phase']}阶段第{analysis['move_count']}步，游戏进度{game_progress_percent:.1f}%({analysis['move_count']}/{total_positions})")
        reasoning_parts.append(f"对战情况：我方执{my_symbol}，对手执{opponent_symbol}，当前棋盘占用率{(occupied_count/total_positions)*100:.1f}%")
        reasoning_parts.append(f"空间状态：棋盘已被占据{occupied_count}个位置，剩余{len(analysis['available_moves'])}个战略要地等待争夺：{analysis['available_moves']}")
        
        # 超详细棋盘地理分析
        if analysis['occupied_positions']:
            pos_names = ['左上角', '上边中央', '右上角', '左边中央', '正中心', '右边中央', '左下角', '下边中央', '右下角']
            line_controls = [
                [0, 1, 2], [3, 4, 5], [6, 7, 8],  # 水平线
                [0, 3, 6], [1, 4, 7], [2, 5, 8],  # 垂直线  
                [0, 4, 8], [2, 4, 6]              # 对角线
            ]
            
            reasoning_parts.append("【超详细棋盘地理分析】当前各位置占用情况：")
            for pos, symbol in analysis['occupied_positions'].items():
                controlled_lines = sum(1 for line in line_controls if pos in line)
                reasoning_parts.append(f"  - {pos_names[pos]}位置({pos})：被{symbol}方占据，该位置参与{controlled_lines}条获胜线路的构建")
        
        # 2. 全维度威胁态势评估
        threats = analysis['threats']
        reasoning_parts.append("【全维度威胁态势评估】")
        
        if threats['my_winning_moves']:
            reasoning_parts.append(f"  ★ 我方决定性优势：发现{len(threats['my_winning_moves'])}个立即获胜位置{threats['my_winning_moves']}")
            for win_pos in threats['my_winning_moves']:
                reasoning_parts.append(f"    → 位置{win_pos}：选择此位置将立即完成一条获胜线路，直接结束游戏并获得胜利，这是绝对优先级最高的选择")
        
        if threats['opponent_winning_moves']:
            reasoning_parts.append(f"  ⚠ 对手致命威胁：识别到{len(threats['opponent_winning_moves'])}个敌方获胜威胁位置{threats['opponent_winning_moves']}")
            for threat_pos in threats['opponent_winning_moves']:
                reasoning_parts.append(f"    → 位置{threat_pos}：如果对手下一步选择此位置将立即获胜，必须优先阻断，这是生死攸关的防守要点")
        
        if threats['my_fork_opportunities']:
            reasoning_parts.append(f"  ♦ 我方战术机会：发现{len(threats['my_fork_opportunities'])}个fork战术创造机会{threats['my_fork_opportunities']}")
            for fork_pos in threats['my_fork_opportunities']:
                reasoning_parts.append(f"    → 位置{fork_pos}：选择此位置可以同时创建多条获胜威胁线路，迫使对手无法同时防守所有威胁，这是高级战术的体现")
        
        if threats['opponent_fork_opportunities']:
            reasoning_parts.append(f"  ⚡ 对手战术威胁：监测到{len(threats['opponent_fork_opportunities'])}个敌方潜在fork威胁{threats['opponent_fork_opportunities']}")
            for opp_fork_pos in threats['opponent_fork_opportunities']:
                reasoning_parts.append(f"    → 位置{opp_fork_pos}：对手可能通过此位置建立多重威胁，需要提前预防和化解，避免陷入被动局面")
        
        # 3. 全方位战略地形价值分析
        strategic = analysis['strategic_positions']
        reasoning_parts.append("【全方位战略地形价值分析】")
        
        if strategic['center']:
            reasoning_parts.append("  ◆ 中心要塞价值分析：")
            reasoning_parts.append("    - 位置4(正中心)：这是整个3x3棋盘的绝对核心，控制水平线(3-4-5)、垂直线(1-4-7)、主对角线(0-4-8)、副对角线(2-4-6)共4条获胜线路")
            reasoning_parts.append("    - 战略意义：占据中心位置等于控制了棋盘50%的获胜可能性，无论从进攻还是防守角度都具有无可替代的价值")
            reasoning_parts.append("    - 控制力影响：中心位置的占有者在后续博弈中将拥有最大的选择灵活性和威胁创造能力")
        
        if strategic['corners']:
            reasoning_parts.append(f"  ◇ 角落据点价值分析：可用角落位置{strategic['corners']}")
            corner_details = {
                0: "左上角(0)：控制顶部水平线(0-1-2)、左侧垂直线(0-3-6)、主对角线(0-4-8)",
                2: "右上角(2)：控制顶部水平线(0-1-2)、右侧垂直线(2-5-8)、副对角线(2-4-6)", 
                6: "左下角(6)：控制底部水平线(6-7-8)、左侧垂直线(0-3-6)、副对角线(2-4-6)",
                8: "右下角(8)：控制底部水平线(6-7-8)、右侧垂直线(2-5-8)、主对角线(0-4-8)"
            }
            for corner in strategic['corners']:
                reasoning_parts.append(f"    - {corner_details[corner]}")
            reasoning_parts.append("    - 战略意义：角落位置适合建立长期控制优势，每个角落都参与3条线路，是构建复合威胁的理想基础")
        
        if strategic['edges']:
            reasoning_parts.append(f"  ◈ 边缘支点价值分析：可用边缘位置{strategic['edges']}")
            edge_details = {
                1: "上边中央(1)：控制顶部水平线(0-1-2)、中央垂直线(1-4-7)",
                3: "左边中央(3)：控制中部水平线(3-4-5)、左侧垂直线(0-3-6)",
                5: "右边中央(5)：控制中部水平线(3-4-5)、右侧垂直线(2-5-8)",
                7: "下边中央(7)：控制底部水平线(6-7-8)、中央垂直线(1-4-7)"
            }
            for edge in strategic['edges']:
                reasoning_parts.append(f"    - {edge_details[edge]}")
            reasoning_parts.append("    - 战略意义：边缘位置主要价值在于防守巩固和战术补强，在特定情况下也可以作为surprise attack的跳板")
        
        # 4. 超深层策略哲学体系
        reasoning_parts.append("【超深层策略哲学体系】")
        strategy_systems = {
            StrategyType.AGGRESSIVE: {
                "核心理念": "进攻至上主义：相信最好的防守就是进攻，通过持续的攻势压迫来掌控游戏节奏",
                "执行原则": "优先创造威胁→迫使对手防守→寻找破绽→一击致命",
                "心理战术": "通过不断的威胁创造给对手施加心理压力，增加其犯错概率",
                "适用场景": "开局抢占主动权，中局扩大优势，残局快速结束战斗"
            },
            StrategyType.CONSERVATIVE: {
                "核心理念": "稳健至上主义：稳扎稳打，确保每一步都不给对手任何机会",
                "执行原则": "先确保防守无懈可击→观察对手动向→等待对手失误→抓住机会反击",  
                "心理战术": "通过完美的防守给对手造成挫败感，诱导其采取冒险行动",
                "适用场景": "劣势局面稳住阵脚，均势局面等待机会，优势局面稳固胜果"
            },
            StrategyType.BALANCED: {
                "核心理念": "动态平衡主义：根据局面变化灵活调整攻守重点，追求整体最优",
                "执行原则": "评估当前局面→判断攻守优先级→选择最适合的行动→保持战略灵活性",
                "心理战术": "通过变化多端的策略让对手难以预测和针对",
                "适用场景": "复杂多变的中局，需要精细计算的关键时刻"
            },
            StrategyType.OPPORTUNISTIC: {
                "核心理念": "机会主义：高度适应性和灵活性，随时准备抓住一切有利机会",
                "执行原则": "持续监控局面→识别所有机会→选择当前最优→随时调整策略",
                "心理战术": "通过不可预测的行动干扰对手的计划和判断",
                "适用场景": "局面变化快速的时刻，需要临场应变的情况"
            }
        }
        current_strategy = strategy_systems[self.strategy]
        reasoning_parts.append(f"  当前采用：{self.strategy.value}策略体系")
        for aspect, description in current_strategy.items():
            reasoning_parts.append(f"    - {aspect}：{description}")
        
        # 5. 逐一候选动作深度剖析
        move_evals = analysis['move_evaluations']
        if len(move_evals) > 1:
            all_moves = sorted(move_evals.items(), key=lambda x: x[1], reverse=True)
            reasoning_parts.append(f"【逐一候选动作深度剖析】")
            reasoning_parts.append(f"完整评分体系结果：{[f'位置{pos}({score:.3f}分)' for pos, score in all_moves]}")
            
            # 超详细分析每个候选位置
            for i, (pos, score) in enumerate(all_moves):
                rank_names = ["绝对首选", "优先次选", "第三选择", "备用选择", "应急选择"]
                rank_name = rank_names[i] if i < len(rank_names) else f"第{i+1}选择"
                
                ultra_detailed_analysis = self._get_ultra_detailed_position_analysis(pos, analysis, threats, my_symbol, opponent_symbol)
                reasoning_parts.append(f"  {rank_name}位置{pos}(评分{score:.3f})：")
                reasoning_parts.append(f"    {ultra_detailed_analysis}")
        
        # 6. 多维度反事实推理矩阵
        alternatives = [move for move in analysis['available_moves'] if move != selected_move]
        if alternatives:
            reasoning_parts.append("【多维度反事实推理矩阵】")
            reasoning_parts.append("通过构建平行决策宇宙来验证当前选择的优越性：")
            
            for i, alt_move in enumerate(alternatives[:4]):
                scenario_analysis = self._generate_counterfactual_scenario(alt_move, selected_move, analysis, threats)
                reasoning_parts.append(f"  平行宇宙{i+1}：假设选择位置{alt_move}")
                reasoning_parts.append(f"    → 即时结果：{scenario_analysis['immediate_consequence']}")
                reasoning_parts.append(f"    → 中期影响：{scenario_analysis['medium_term_impact']}")  
                reasoning_parts.append(f"    → 长期后果：{scenario_analysis['long_term_consequence']}")
                reasoning_parts.append(f"    → 综合评判：{scenario_analysis['overall_assessment']}")
        
        # 7. 最终决策的哲学升华
        reasoning_parts.append("【最终决策的哲学升华】")
        ultimate_framework = self._get_philosophical_decision_framework(analysis, selected_move, threats, my_symbol, opponent_symbol)
        
        reasoning_parts.append(f"在经过多维度、多层次的深度分析后，选择位置{selected_move}体现了以下哲学思考：")
        reasoning_parts.append(f"  ◉ 存在论层面：{ultimate_framework['ontological_aspect']}")
        reasoning_parts.append(f"  ◉ 认识论层面：{ultimate_framework['epistemological_aspect']}")  
        reasoning_parts.append(f"  ◉ 方法论层面：{ultimate_framework['methodological_aspect']}")
        reasoning_parts.append(f"  ◉ 价值论层面：{ultimate_framework['axiological_aspect']}")
        reasoning_parts.append(f"  ◉ 实践论层面：{ultimate_framework['practical_aspect']}")
        
        # 8. 终极决策宣言
        reasoning_parts.append("【终极决策宣言】")
        reasoning_parts.append(f"综合所有分析维度，位置{selected_move}不仅是当前局面下的最优选择，更是体现了深层战略智慧、战术素养、心理洞察和哲学思辨的完美结合。")
        reasoning_parts.append(f"这个选择将推动游戏向着最有利的方向发展，无论从短期战术收益还是长期战略布局都具有无可替代的价值。")
        reasoning_parts.append(f"让我们以坚定的信念执行这个深思熟虑的决策！")
        
        # 格式化完整推理
        cot = "\n终极详细思考过程：\n"
        for i, part in enumerate(reasoning_parts, 1):
            cot += f"{i}. {part}\n"
        cot += f"\n最终答案: [{selected_move}]"
        
        return cot
    
    def _get_detailed_position_analysis(self, position: int, analysis: dict, threats: dict) -> str:
        """获取位置的详细分析"""
        if position in threats['my_winning_moves']:
            return "立即获胜位置，选择即可直接获得游戏胜利，绝对优先级"
        elif position in threats['opponent_winning_moves']:
            return "关键防守位置，必须阻止对手获胜，生死攸关的选择"
        elif position in threats['my_fork_opportunities']:
            return "fork战术位置，可以创造多重威胁，迫使对手无法完全防守"
        elif position == 4:
            return "战略核心位置，控制4条获胜线路，战略价值极高，长期收益最大"
        elif position in [0, 2, 6, 8]:
            return "角落控制位置，控制3条线路，适合建立稳固的战略基础和长期优势"
        elif position in [1, 3, 5, 7]:
            return "边缘支撑位置，控制2条线路，主要用于防守巩固和战术补强"
        else:
            return "战略价值有限，在当前局面下不是最优选择"
    
    def _get_ultimate_decision_framework(self, analysis: dict, selected_move: int, threats: dict) -> dict:
        """获取终极决策框架"""
        if selected_move in threats['my_winning_moves']:
            return {
                'core_driver': '绝对获胜机会，立即终结游戏',
                'tactical_value': '直接获得游戏胜利，战术价值无限大',
                'strategic_impact': '完美达成游戏目标，战略意义圆满',
                'psychological_factor': '展现决定性实力，给对手以震撼',
                'long_term_prospect': '游戏即将结束，开启新的胜利篇章'
            }
        elif selected_move in threats['opponent_winning_moves']:
            return {
                'core_driver': '生死存亡的关键防守，阻止对手获胜',
                'tactical_value': '避免immediate失败，延续游戏生命',
                'strategic_impact': '保持游戏平衡，维护竞争机会',
                'psychological_factor': '展现顽强意志，不轻易认输',
                'long_term_prospect': '为后续发展争取宝贵时间和空间'
            }
        elif selected_move == 4:
            return {
                'core_driver': '占据战略制高点，掌控游戏核心',
                'tactical_value': '控制最多获胜线路，最大化影响力',
                'strategic_impact': '建立长期控制优势，奠定胜利基础',
                'psychological_factor': '展现战略眼光，彰显大局观',
                'long_term_prospect': '为所有后续动作创造最佳条件'
            }
        else:
            return {
                'core_driver': '综合考虑下的最优平衡选择',
                'tactical_value': '在当前条件下实现最大价值',
                'strategic_impact': '符合整体策略方向，保持发展势头',
                'psychological_factor': '展现稳健风格，积累微小优势',
                'long_term_prospect': '为未来发展保留最大灵活性'
            }
    
    def _get_ultra_detailed_position_analysis(self, position: int, analysis: dict, threats: dict, my_symbol: str, opponent_symbol: str) -> str:
        """获取超详细位置分析"""
        analysis_parts = []
        
        # 威胁状态分析
        if position in threats['my_winning_moves']:
            analysis_parts.append("【威胁状态】立即获胜位置，选择可直接终结游戏")
        elif position in threats['opponent_winning_moves']:
            analysis_parts.append("【威胁状态】对手威胁位置，必须优先防守")
        elif position in threats['my_fork_opportunities']:
            analysis_parts.append("【威胁状态】我方fork机会，可创建多重威胁")
        elif position in threats['opponent_fork_opportunities']:
            analysis_parts.append("【威胁状态】对手潜在fork位置，需要预防")
        else:
            analysis_parts.append("【威胁状态】中性位置，无直接威胁关系")
        
        # 地理价值分析
        if position == 4:
            analysis_parts.append("【地理价值】中心要塞，控制4条线路，战略价值最高")
        elif position in [0, 2, 6, 8]:
            analysis_parts.append(f"【地理价值】角落据点，控制3条线路，稳固基础")
        else:
            analysis_parts.append(f"【地理价值】边缘支点，控制2条线路，辅助作用")
        
        # 控制线路分析
        line_controls = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # 水平线
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # 垂直线  
            [0, 4, 8], [2, 4, 6]              # 对角线
        ]
        controlled_lines = [i for i, line in enumerate(line_controls) if position in line]
        line_names = ["顶横线", "中横线", "底横线", "左竖线", "中竖线", "右竖线", "主对角", "副对角"]
        analysis_parts.append(f"【控制线路】参与{len(controlled_lines)}条线路：{[line_names[i] for i in controlled_lines]}")
        
        return " | ".join(analysis_parts)
    
    def _generate_counterfactual_scenario(self, alt_move: int, selected_move: int, analysis: dict, threats: dict) -> dict:
        """生成反事实场景分析"""
        if alt_move in threats['my_winning_moves']:
            return {
                'immediate_consequence': '立即获得游戏胜利，最优结果',
                'medium_term_impact': '游戏结束，无中期发展',
                'long_term_consequence': '完美胜局，但失去了展现其他战术的机会',
                'overall_assessment': f'虽然可以立即获胜，但选择{selected_move}可能有其他考虑'
            }
        elif alt_move in threats['opponent_winning_moves']:
            return {
                'immediate_consequence': '成功阻止对手获胜，化解危机',
                'medium_term_impact': '保持游戏继续，争取发展机会',
                'long_term_consequence': '避免失败，但可能仍处被动状态',
                'overall_assessment': f'虽然防守有效，但{selected_move}可能有更积极的价值'
            }
        else:
            return {
                'immediate_consequence': '获得一定控制力，但无决定性优势',
                'medium_term_impact': '推进游戏进程，保持竞争态势',
                'long_term_consequence': '为后续发展奠定基础，但效果有限',
                'overall_assessment': f'相比{selected_move}，综合价值略显不足'
            }
    
    def _get_philosophical_decision_framework(self, analysis: dict, selected_move: int, threats: dict, my_symbol: str, opponent_symbol: str) -> dict:
        """获取哲学决策框架"""
        return {
            'ontological_aspect': f'位置{selected_move}的存在意义体现在它作为棋盘空间中具有独特价值属性的点，其存在为游戏状态转换提供了可能性',
            'epistemological_aspect': f'我们通过深度分析得知位置{selected_move}的价值，这种认知基于逻辑推理、威胁评估和战略思维的综合运用',
            'methodological_aspect': f'选择位置{selected_move}的方法论基础是系统性分析、多维度评估和最优化决策理论的实际应用',
            'axiological_aspect': f'位置{selected_move}的价值体现在它对实现游戏目标的贡献度，以及在当前策略框架下的重要性',
            'practical_aspect': f'从实践角度看，选择位置{selected_move}是理论分析与实际行动的完美结合，体现了知行合一的智慧'
        }
    
    def _generate_very_long_cot(self, analysis: dict, selected_move: int, my_symbol: str, opponent_symbol: str) -> str:
        """生成超长推理过程 (约2000字符)"""
        reasoning_parts = []
        
        # 1. 全面局面分析
        occupied_count = len(analysis['occupied_positions'])
        reasoning_parts.append(f"【超详细局面分析】当前处于{analysis['game_phase']}阶段第{analysis['move_count']}步，我方执{my_symbol}，对方执{opponent_symbol}")
        reasoning_parts.append(f"棋盘占用情况：已占据{occupied_count}个位置，剩余{len(analysis['available_moves'])}个可选位置：{analysis['available_moves']}")
        
        # 2. 威胁分析
        threats = analysis['threats']
        reasoning_parts.append("【威胁分析】")
        if threats['my_winning_moves']:
            reasoning_parts.append(f"  - 我方获胜机会：发现{len(threats['my_winning_moves'])}个立即获胜位置{threats['my_winning_moves']}")
        if threats['opponent_winning_moves']:
            reasoning_parts.append(f"  - 对手获胜威胁：检测到{len(threats['opponent_winning_moves'])}个对手获胜位置{threats['opponent_winning_moves']}")
        
        # 3. 战略位置分析
        strategic = analysis['strategic_positions']
        reasoning_parts.append("【战略位置分析】")
        if strategic['center']:
            reasoning_parts.append("  - 中心位置(4)可用：控制4条获胜线路，战略价值最高")
        if strategic['corners']:
            reasoning_parts.append(f"  - 角落位置{strategic['corners']}可用：每个角落控制3条线路")
        if strategic['edges']:
            reasoning_parts.append(f"  - 边缘位置{strategic['edges']}可用：每个边缘控制2条线路")
        
        # 4. 策略框架
        strategy_explanations = {
            StrategyType.AGGRESSIVE: "激进型策略：优先创造威胁，迫使对手防守",
            StrategyType.CONSERVATIVE: "保守型策略：重点确保防守稳固，等待对手失误",
            StrategyType.BALANCED: "均衡型策略：攻守兼备，灵活调整重点",
            StrategyType.OPPORTUNISTIC: "机会主义策略：敏锐捕捉有利时机"
        }
        reasoning_parts.append(f"【策略框架】{strategy_explanations[self.strategy]}")
        
        # 5. 候选动作评估
        move_evals = analysis['move_evaluations']
        if len(move_evals) > 1:
            top_moves = sorted(move_evals.items(), key=lambda x: x[1], reverse=True)
            reasoning_parts.append(f"【候选动作评估】完整排序：{[f'位置{pos}({score:.1f}分)' for pos, score in top_moves]}")
        
        # 6. 最终决策论证
        decision_context = self._get_decision_context(analysis, selected_move, threats)
        reasoning_parts.append(f"【最终决策论证】选择位置{selected_move}的理由：")
        reasoning_parts.append(f"  - 主要原因：{decision_context['primary_reason']}")
        reasoning_parts.append(f"  - 战术价值：{decision_context['tactical_value']}")
        reasoning_parts.append(f"  - 长期影响：{decision_context['long_term_impact']}")
        
        # 格式化
        cot = "\n超详细思考过程：\n"
        for i, part in enumerate(reasoning_parts, 1):
            cot += f"{i}. {part}\n"
        cot += f"\n经过深度分析，最终答案: [{selected_move}]"
        
        return cot
