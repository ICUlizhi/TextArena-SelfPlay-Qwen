#!/usr/bin/env python3

"""
Simple test runner for the TicTacToe self-play system
"""

import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.qwen_agent import QwenAgent
from utils.mock_env import MockTicTacToeEnv
from data_generation.selfplay_runner import SelfPlayRunner

def test_simple_game():
    """Test a single game to ensure everything works"""
    print("=== Testing Simple Game ===")
    
    # Create environment
    env = MockTicTacToeEnv()
    
    # Create agents
    agent_x = QwenAgent()
    agent_o = QwenAgent() 
    
    agents = {0: agent_x, 1: agent_o}
    
    # Run one game
    env.reset()
    game_history = []
    
    print("Starting game...")
    for turn in range(9):  # Max 9 turns
        player_id, observation = env.get_observation()
        print(f"\nTurn {turn + 1}, Player {player_id}:")
        print("Observation:", observation[:100] + "..." if len(observation) > 100 else observation)
        
        action = agents[player_id](observation)
        print(f"Action: {action}")
        
        done, info = env.step(action)
        game_history.append({
            "player": player_id,
            "action": action,
            "turn": turn + 1,
            "info": info.copy()
        })
        
        if done:
            print(f"Game ended: {info}")
            break
    
    rewards, final_info = env.close()
    print(f"\nFinal result: {final_info}")
    print(f"Game history: {len(game_history)} moves")
    
    return True

def test_self_play_runner():
    """Test the self-play runner with 2 games"""
    print("\n=== Testing Self-Play Runner ===")
    
    # Create environment and agents
    env = MockTicTacToeEnv()
    agent_x = QwenAgent()
    agent_o = QwenAgent()
    agents = {0: agent_x, 1: agent_o}
    
    # Create runner
    runner = SelfPlayRunner(env, agents)
    
    # Run 2 games
    print("Running 2 games...")
    try:
        history = runner.run_self_play(num_games=2)
        print(f"✓ Successfully completed {len(history)} games")
        return True
    except Exception as e:
        print(f"✗ Error in self-play runner: {e}")
        return False

def test_data_formatting():
    """Test data formatting functionality"""
    print("\n=== Testing Data Formatting ===")
    
    try:
        from utils.data_formatter import SelfPlayDataFormatter
        
        formatter = SelfPlayDataFormatter()
        print("✓ Data formatter created successfully")
        
        # Create some mock data
        mock_data = [{
            "game_id": 1,
            "moves": [
                {"player": 0, "action": "[4]", "turn": 1},
                {"player": 1, "action": "[0]", "turn": 2},
                {"player": 0, "action": "[8]", "turn": 3}
            ],
            "result": {"winner": 0, "game_over": True}
        }]
        
        training_samples = formatter.format_for_sft(mock_data)
        print(f"✓ Generated {len(training_samples)} training samples")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in data formatting: {e}")
        return False

def main():
    """Run all tests"""
    print("Starting TicTacToe Self-Play System Tests")
    print("=" * 50)
    
    tests = [
        ("Simple Game", test_simple_game),
        ("Self-Play Runner", test_self_play_runner), 
        ("Data Formatting", test_data_formatting)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            if result:
                print(f"✓ {test_name}: PASSED")
                passed += 1
            else:
                print(f"✗ {test_name}: FAILED")
        except Exception as e:
            print(f"✗ {test_name}: ERROR - {e}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The system is ready for use.")
        print("\nTo generate self-play data, run:")
        print("python main.py --num-games 50")
        print("python main.py --num-games 50 --load-qwen  # To use actual Qwen model")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
