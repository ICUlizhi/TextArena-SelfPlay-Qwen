import sys
import traceback
import argparse
import os
import glob
import json
from datetime import datetime

try:
    import textarena as ta
    TEXTARENA_AVAILABLE = True
except ImportError:
    print("Warning: TextArena not available, will use mock environment")
    TEXTARENA_AVAILABLE = False

from agents.qwen_agent import QwenAgent
from agents.smart_agent import SmartAgent
from data_generation.selfplay_runner import SelfPlayRunner

def main():
    parser = argparse.ArgumentParser(description='TicTacToe Self-Play Data Generation')
    parser.add_argument('--num-games', type=int, default=10, help='Number of games to play')
    parser.add_argument('--load-qwen', action='store_true', help='Load actual Qwen model (requires GPU)')
    parser.add_argument('--model-path', type=str, default=None, help='Path to Qwen model')
    parser.add_argument('--cot-length', type=str, default='medium', 
                       choices=['tiny', 'short', 'medium', 'long', 'very_long', 'ultra_long'],
                       help='CoT length type for data generation')
    parser.add_argument('--process-id', type=str, default=None, help='Process ID for parallel execution')
    parser.add_argument('--output-suffix', type=str, default=None, help='Output file suffix for independent files')
    
    args = parser.parse_args()
    
    try:
        if TEXTARENA_AVAILABLE:
            print("Using TextArena environment...")
            # Initialize the TextArena environment
            env = ta.make("TicTacToe-v0")
        else:
            print("Using mock environment...")
            # Use mock environment as fallback
            from utils.mock_env import MockTicTacToeEnv
            env = MockTicTacToeEnv()
        
        # Initialize agents - use the same agent type for true self-play
        print("Initializing agents...")
        if args.load_qwen:
            print("Loading Qwen models (this may take a while)...")
            agent_x = QwenAgent(model_path=args.model_path, load_model=True, cot_length=args.cot_length)  # Player 0 (X)
            agent_o = QwenAgent(model_path=args.model_path, load_model=True, cot_length=args.cot_length)  # Player 1 (O)
        else:
            print("Using rule-based strategy...")
            agent_x = QwenAgent(cot_length=args.cot_length)  # Player 0 (X)
            agent_o = QwenAgent(cot_length=args.cot_length)  # Player 1 (O)
        
        agents = {
            0: agent_x,
            1: agent_o,
        }
        
        # Set up self-play runner
        print("Setting up self-play runner...")
        self_play_runner = SelfPlayRunner(env, agents)
        
        # Run self-play data generation
        print(f"Starting self-play data generation for {args.num_games} games...")
        print(f"Using CoT length: {args.cot_length}")
        if args.process_id:
            print(f"Process ID: {args.process_id}")
        
        # 使用固定的CoT长度参数
        if args.cot_length:
            self_play_runner.run_self_play(num_games=args.num_games, cot_length_control=False, fixed_cot_length=args.cot_length)
        else:
            self_play_runner.run_self_play(num_games=args.num_games)
        print("Self-play completed successfully!")
        
        # 格式化数据为SFT训练格式
        print("🔄 格式化数据为SFT训练格式...")
        from utils.data_formatter import SelfPlayDataFormatter
        
        # 找到最新生成的数据文件
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(project_root, "data", "raw")
        pattern = f"{data_dir}/self_play_data_*.json"
        files = glob.glob(pattern)
        
        if files:
            latest_file = max(files, key=os.path.getctime)
            print(f"📂 处理文件: {latest_file}")
            
            # 使用SelfPlayDataFormatter来格式化数据
            formatter = SelfPlayDataFormatter()
            games_data = formatter.load_self_play_data(latest_file)
            
            # 提取games数组（如果存在metadata）
            if isinstance(games_data, dict) and 'games' in games_data:
                games_data = games_data['games']
            
            # 格式化为LLaMA Factory格式
            llama_factory_data = formatter.format_for_llama_factory(games_data)
            
            # 保存格式化后的数据
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            if args.output_suffix:
                # 使用指定的后缀创建独立文件名
                output_file = os.path.join(project_root, "data", "processed", f"long_cot_sft_data_{args.output_suffix}.json")
            else:
                output_file = os.path.join(project_root, "data", "processed", f"long_cot_sft_data_{args.cot_length}_{timestamp}.json")
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(llama_factory_data, f, indent=2, ensure_ascii=False)
            
            print(f"✅ SFT数据已保存到: {output_file}")
        else:
            print("❌ 未找到生成的数据文件")
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"Error during execution: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()