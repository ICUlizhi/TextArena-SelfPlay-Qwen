#!/usr/bin/env python3
"""
Quick Start Example for TicTacToe Self-Play

This example demonstrates how to:
1. Run a few self-play games
2. Generate SFT training data
3. Check data quality
"""

import os
import sys
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from agents.qwen_agent import QwenAgent
from agents.smart_agent import SmartAgent
from utils.mock_env import MockTicTacToeEnv
from data_generation.selfplay_runner import SelfPlayRunner

def main():
    print("🎮 TicTacToe Self-Play Quick Start Example")
    print("=" * 50)
    
    # Step 1: Setup environment and agents
    print("1. Setting up environment and agents...")
    env = MockTicTacToeEnv()
    
    # Create agents (using fallback strategy since we don't have Qwen loaded)
    agent_0 = QwenAgent(player_id=0, model_path=None)  # Will use fallback
    agent_1 = SmartAgent(player_id=1)
    agents = [agent_0, agent_1]
    
    print("   ✓ Environment and agents ready")
    
    # Step 2: Run a few self-play games
    print("\n2. Running self-play games...")
    runner = SelfPlayRunner(env, agents)
    
    # Run 3 games as demonstration
    history = runner.run_self_play(num_games=3)
    print(f"   ✓ Completed {len(history)} games")
    
    # Step 3: Analyze the results
    print("\n3. Analyzing results...")
    total_moves = sum(len(game['moves']) for game in history)
    cot_moves = sum(1 for game in history for move in game['moves'] if move.get('cot'))
    
    print(f"   • Total moves: {total_moves}")
    print(f"   • Moves with CoT: {cot_moves}")
    print(f"   • CoT coverage: {cot_moves/total_moves:.1%}")
    
    # Step 4: Show sample data
    print("\n4. Sample training data preview:")
    if history and history[0]['moves']:
        sample_move = history[0]['moves'][0]
        print("   Sample observation:")
        print(f"   {sample_move['observation'][:100]}...")
        if sample_move.get('cot'):
            print("   Sample CoT:")
            print(f"   {sample_move['cot'][:100]}...")
    
    print(f"\n🎉 Quick start complete!")
    print(f"Generated data is saved automatically.")
    print(f"Next steps:")
    print(f"  • Run more games: python run.py --num-games 50")
    print(f"  • Generate training data: python scripts/generate_sft_data.py")
    print(f"  • Train your model with the generated data")

if __name__ == "__main__":
    main()
