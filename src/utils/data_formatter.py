"""
Data formatter for converting self-play game data into training format
"""

import json
import os
from typing import List, Dict, Any
from datetime import datetime


class SelfPlayDataFormatter:
    """Convert self-play game data into format suitable for SFT training"""
    
    def __init__(self):
        self.training_data = []
    
    def _get_winner_from_result(self, result: Dict) -> int:
        """从result字典中判断获胜者"""
        if not result:
            return None
            
        # 检查每个玩家的结果信息
        for player_id, player_result in result.items():
            if isinstance(player_result, dict):
                reason = player_result.get('reason', '')
                if 'has won!' in reason:
                    # 从reason中提取获胜的玩家号
                    if 'Player 0 has won!' in reason:
                        return 0
                    elif 'Player 1 has won!' in reason:
                        return 1
        
        return None
    
    def load_self_play_data(self, data_file: str) -> List[Dict]:
        """Load self-play data from JSON file"""
        with open(data_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def format_for_sft(self, games_data: List[Dict], filter_winners_only: bool = True) -> List[Dict]:
        """
        Convert game data into SFT training format
        
        Args:
            games_data: List of game dictionaries from self-play
            filter_winners_only: If True, only use moves from winning games
            
        Returns:
            List of training samples in format:
            {
                "instruction": "井字棋游戏中，请分析当前棋盘状态并选择最优落子位置。",
                "input": "当前棋盘状态：...\n你是X，请详细思考后给出答案。",
                "output": "思考过程：...\n答案: [位置]"
            }
        """
        
        training_samples = []
        
        for game in games_data:
            # 从result中判断获胜者
            winner = self._get_winner_from_result(game.get('result', {}))
            
            # Skip draws if filtering winners only
            if filter_winners_only and winner is None:
                continue
            
            moves = game.get('moves', [])
            
            # Extract moves from the winning player
            if winner is not None:
                for move in moves:
                    if move['player'] == winner:
                        # Create training sample
                        sample = self._create_training_sample(move, game)
                        if sample:
                            training_samples.append(sample)
        
        return training_samples
    
    def _create_training_sample(self, move_data: Dict, game_data: Dict) -> Dict:
        """Create a single training sample from move data"""
        
        # 提取数据
        observation = move_data.get('observation', '')
        action = move_data.get('action', '')
        cot = move_data.get('cot', '')
        player = move_data['player']
        player_mark = 'X' if player == 0 else 'O'
        
        # 清理动作格式（去除方括号）
        if isinstance(action, str):
            action_clean = action.strip('[]')
        else:
            action_clean = str(action)
        
        # 如果没有观察数据，创建一个简化版本
        if not observation:
            observation = f"Player {player_mark}'s turn. Choose your move."
        
        # 构建输出（包含 CoT 或生成默认推理）
        if cot:
            # 使用实际的 CoT 推理
            output = f"{cot}\n\n答案: [{action_clean}]"
        else:
            # 生成默认的推理过程
            output = f"""思考过程：
1. 当前是{player_mark}的回合，需要选择最优落子位置
2. 分析棋盘状态，寻找获胜机会或防守需求
3. 优先考虑中心位置、角落位置，然后是边缘位置
4. 根据当前局面选择最佳策略

答案: [{action_clean}]"""
        
        # 创建训练样本
        sample = {
            "instruction": "井字棋游戏中，请分析当前棋盘状态并选择最优落子位置。",
            "input": f"当前游戏状态：\n{observation}\n\n你是{player_mark}，请详细分析棋盘局面，思考最优策略，然后给出你的落子选择。",
            "output": output
        }
        
        return sample
    
    def save_training_data(self, training_samples: List[Dict], output_file: str = None):
        """Save training data to JSON file"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"training_data_{timestamp}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(training_samples, f, indent=2, ensure_ascii=False)
        
        print(f"Training data saved to {output_file}")
        print(f"Generated {len(training_samples)} training samples")
        
        return output_file
    
    def create_llama_factory_format(self, training_samples: List[Dict]) -> List[Dict]:
        """Convert to LLaMA-Factory compatible format"""
        
        llama_factory_samples = []
        
        for sample in training_samples:
            # LLaMA-Factory format
            formatted_sample = {
                "conversations": [
                    {
                        "from": "human",
                        "value": f"{sample['instruction']}\n\n{sample['input']}"
                    },
                    {
                        "from": "gpt", 
                        "value": sample['output']
                    }
                ]
            }
            llama_factory_samples.append(formatted_sample)
        
        return llama_factory_samples
    
    def format_for_llama_factory(self, games_data: List[Dict], filter_winners_only: bool = True) -> List[Dict]:
        """
        Convert game data directly into LLaMA-Factory training format
        
        Args:
            games_data: List of game dictionaries from self-play
            filter_winners_only: If True, only use moves from winning games
            
        Returns:
            List of training samples in LLaMA-Factory conversation format
        """
        # First convert to standard format
        standard_samples = self.format_for_sft(games_data, filter_winners_only)
        
        # Then convert to LLaMA-Factory format
        return self.create_llama_factory_format(standard_samples)
    
    def process_self_play_directory(self, data_dir: str, output_dir: str = None):
        """Process all self-play data files in a directory"""
        
        if output_dir is None:
            output_dir = data_dir
        
        all_training_samples = []
        
        # Find all self-play data files
        for filename in os.listdir(data_dir):
            if filename.startswith('self_play_data_') and filename.endswith('.json'):
                filepath = os.path.join(data_dir, filename)
                print(f"Processing {filename}...")
                
                # Load and format data
                games_data = self.load_self_play_data(filepath)
                training_samples = self.format_for_sft(games_data)
                all_training_samples.extend(training_samples)
        
        if all_training_samples:
            # Save combined training data
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Standard format
            output_file = os.path.join(output_dir, f"sft_training_data_{timestamp}.json")
            self.save_training_data(all_training_samples, output_file)
            
            # LLaMA-Factory format
            llama_factory_data = self.create_llama_factory_format(all_training_samples)
            llama_factory_file = os.path.join(output_dir, f"sft_training_data_llama_factory_{timestamp}.json")
            
            with open(llama_factory_file, 'w', encoding='utf-8') as f:
                json.dump(llama_factory_data, f, indent=2, ensure_ascii=False)
            
            print(f"LLaMA-Factory format saved to {llama_factory_file}")
            
            return output_file, llama_factory_file
        else:
            print("No training samples generated")
            return None, None


# Legacy functions for compatibility
def format_self_play_data(history):
    """Legacy function for compatibility"""
    formatted_data = []
    
    for entry in history:
        formatted_entry = {
            "player": entry["player"],
            "action": entry["action"], 
            "info": entry["info"]
        }
        formatted_data.append(formatted_entry)
    
    return formatted_data


def save_to_json(data, file_path):
    import json
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)


def load_from_json(file_path):
    import json
    with open(file_path, 'r') as json_file:
        return json.load(json_file)


def main():
    """Example usage of the data formatter"""
    formatter = SelfPlayDataFormatter()
    
    # Process all self-play data in the data/raw directory
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    raw_data_dir = os.path.join(project_root, "data", "raw")
    processed_data_dir = os.path.join(project_root, "data", "processed")
    
    # Ensure processed directory exists
    os.makedirs(processed_data_dir, exist_ok=True)
    
    # Process the data
    if os.path.exists(raw_data_dir):
        formatter.process_self_play_directory(raw_data_dir, processed_data_dir)
    else:
        print(f"Raw data directory not found: {raw_data_dir}")


if __name__ == "__main__":
    main()
