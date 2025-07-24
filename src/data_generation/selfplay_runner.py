import json
import random
from datetime import datetime
import os
import sys

# 添加utils路径以便导入TestSetAvoider
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from utils.test_set_avoider import TestSetAvoider
    TEST_SET_AVOIDANCE_AVAILABLE = True
except ImportError:
    TEST_SET_AVOIDANCE_AVAILABLE = False
    print("⚠️  测试集规避功能不可用，将生成所有数据")

class SelfPlayRunner:
    def __init__(self, env, agents, enable_test_avoidance=True):
        self.env = env
        self.agents = agents
        self.enable_test_avoidance = enable_test_avoidance
        
        # 初始化测试集规避器
        self.test_avoider = None
        if self.enable_test_avoidance and TEST_SET_AVOIDANCE_AVAILABLE:
            try:
                self.test_avoider = TestSetAvoider()
                print(f"✅ 测试集规避功能已启用")
            except Exception as e:
                print(f"⚠️  测试集规避初始化失败: {e}")
                self.test_avoider = None
        elif self.enable_test_avoidance:
            print("⚠️  测试集规避功能不可用")
        else:
            print("📋 测试集规避功能已禁用")

    def run_self_play(self, num_games, cot_length_control=True, fixed_cot_length=None):
        """Run self-play for specified number of games with diverse strategies and CoT length control"""
        history = []
        strategies = ['aggressive', 'conservative', 'balanced', 'opportunistic']
        
        # 如果指定了固定的CoT长度，使用固定值；否则使用多样化的CoT长度
        if fixed_cot_length:
            cot_lengths = [fixed_cot_length]
            print(f"使用固定CoT长度: {fixed_cot_length}")
        elif cot_length_control:
            cot_lengths = ['short', 'medium', 'long', 'ultra_long']
        else:
            cot_lengths = ['medium']
        
        for game_id in range(num_games):
            print(f"Starting game {game_id + 1}/{num_games}")
            
            # 为每个游戏随机分配不同的策略组合
            strategy_combo = self._get_strategy_combination(strategies, game_id)
            
            # 为每个游戏分配CoT长度组合
            if fixed_cot_length:
                # 使用固定的CoT长度
                cot_combo = [fixed_cot_length, fixed_cot_length]
            elif cot_length_control:
                # 使用多样化的CoT长度
                cot_combo = self._get_cot_length_combination(cot_lengths, game_id)
            else:
                # 使用默认medium长度
                cot_combo = ['medium', 'medium']
            
            print(f"Strategy combination: Player 0 = {strategy_combo[0]}, Player 1 = {strategy_combo[1]}")
            print(f"CoT length combination: Player 0 = {cot_combo[0]}, Player 1 = {cot_combo[1]}")
            
            # 重新初始化agents with new strategies and CoT lengths
            self._update_agent_configuration(strategy_combo, cot_combo)
            
            game_history = self._run_single_game()
            game_history['strategies'] = strategy_combo  # 记录策略组合
            game_history['cot_lengths'] = cot_combo  # 记录CoT长度组合
            history.append(game_history)
        
        # Save the data
        self._save_self_play_data(history)
        return history
    
    def _get_strategy_combination(self, strategies, game_id):
        """生成多样化的策略组合"""
        # 确保策略多样性的几种模式
        patterns = [
            # 相同策略对战
            lambda: [random.choice(strategies)] * 2,
            # 完全不同策略对战  
            lambda: random.sample(strategies, 2),
            # 激进vs保守对战
            lambda: ['aggressive', 'conservative'] if random.random() > 0.5 else ['conservative', 'aggressive'],
            # 均衡与其他策略对战
            lambda: ['balanced', random.choice([s for s in strategies if s != 'balanced'])],
        ]
        
        # 根据游戏ID选择不同的模式，确保多样性
        pattern = patterns[game_id % len(patterns)]
        return pattern()
    
    def _get_cot_length_combination(self, cot_lengths, game_id):
        """生成多样化的CoT长度组合"""
        # CoT长度组合模式
        cot_patterns = [
            # 相同长度对战
            lambda: [random.choice(cot_lengths)] * 2,
            # 不同长度对战
            lambda: random.sample(cot_lengths, 2) if len(cot_lengths) >= 2 else [cot_lengths[0]] * 2,
            # 短vs长对战
            lambda: ['short', 'long'] if 'short' in cot_lengths and 'long' in cot_lengths else [random.choice(cot_lengths)] * 2,
            # 中等与其他长度对战
            lambda: ['medium', random.choice([c for c in cot_lengths if c != 'medium'])] if 'medium' in cot_lengths else [random.choice(cot_lengths)] * 2,
        ]
        
        pattern = cot_patterns[game_id % len(cot_patterns)]
        return pattern()
    
    def _update_agent_configuration(self, strategies, cot_lengths=None):
        """更新agents的策略和CoT长度配置"""
        if hasattr(self.agents[0], 'strategy'):
            # 重新设置策略
            from agents.qwen_agent import StrategyType, CoTLengthType
            strategy_map = {
                'conservative': StrategyType.CONSERVATIVE,
                'aggressive': StrategyType.AGGRESSIVE, 
                'balanced': StrategyType.BALANCED,
                'opportunistic': StrategyType.OPPORTUNISTIC
            }
            
            cot_map = {
                'tiny': CoTLengthType.TINY,
                'short': CoTLengthType.SHORT,
                'medium': CoTLengthType.MEDIUM,
                'long': CoTLengthType.LONG,
                'very_long': CoTLengthType.VERY_LONG,
                'ultra_long': CoTLengthType.ULTRA_LONG
            }
            
            # 更新策略
            self.agents[0].strategy = strategy_map[strategies[0]]
            self.agents[1].strategy = strategy_map[strategies[1]]
            
            # 只有在提供CoT长度时才更新CoT长度
            if cot_lengths:
                self.agents[0].cot_length = cot_map[cot_lengths[0]]
                self.agents[1].cot_length = cot_map[cot_lengths[1]]
            
            # 重置游戏状态
            self.agents[0].game_phase = "opening"
            self.agents[0].move_count = 0
            self.agents[1].game_phase = "opening" 
            self.agents[1].move_count = 0

    def _run_single_game(self):
        self.env.reset(num_players=2)
        game_history = []
        
        for turn in range(9):  # Maximum number of turns
            player_id, observation = self.env.get_observation()
            action = self.agents[player_id](observation)
            
            print(f"\nPlayer {player_id} action: {action}")
            
            done, info = self.env.step(action)
            
            # 保存更完整的信息，包括观察数据和 CoT
            move_data = {
                "player": player_id,
                "observation": observation,  # 保存观察数据
                "action": action,
                "turn": turn + 1,
                "info": info.copy()
            }
            
            # 如果智能体有 CoT 推理过程，也保存下来
            if hasattr(self.agents[player_id], 'last_cot') and self.agents[player_id].last_cot:
                move_data["cot"] = self.agents[player_id].last_cot
            
            game_history.append(move_data)
            
            if done:
                break
        
        rewards, game_info = self.env.close()
        print("\n=== Game Result ===")
        print(game_info)
        
        # 返回完整的游戏数据
        return {
            "moves": game_history,
            "result": game_info,
            "rewards": rewards
        }
    
    def _save_self_play_data(self, history):
        """Save self-play data to JSON file with optional test set avoidance"""
        import os
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 应用测试集规避过滤
        if self.test_avoider:
            print("🛡️  应用测试集规避过滤...")
            original_count = len(history)
            history = self.test_avoider.filter_self_play_data(history)
            filtered_count = len(history)
            print(f"📊 过滤结果: {original_count} -> {filtered_count} 个游戏")
        
        # Use absolute path to ensure file is saved correctly
        try:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            data_dir = os.path.join(project_root, "data", "raw")
            os.makedirs(data_dir, exist_ok=True)
            filename = os.path.join(data_dir, f"self_play_data_{timestamp}.json")
        except Exception as e:
            # Fallback to current directory
            print(f"Could not create data directory: {e}")
            filename = f"self_play_data_{timestamp}.json"
        
        # 添加元数据
        data_with_metadata = {
            "generation_info": {
                "timestamp": timestamp,
                "total_games": len(history),
                "test_set_avoidance_enabled": self.test_avoider is not None,
                "avoider_stats": self.test_avoider.get_statistics() if self.test_avoider else None
            },
            "games": history
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data_with_metadata, f, indent=2, ensure_ascii=False)
        
        print(f"Self-play data saved to {filename}")
        if self.test_avoider:
            print(f"✅ 数据已经过测试集规避过滤")