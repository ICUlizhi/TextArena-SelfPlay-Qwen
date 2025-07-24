"""
Mock TicTacToe environment for testing self-play when TextArena is not available.
"""

import random
import re
from typing import List, Tuple, Dict, Any


class MockTicTacToeEnv:
    """Mock implementation of TicTacToe environment compatible with TextArena API"""
    
    def __init__(self):
        self.board = [' '] * 9
        self.current_player = 0
        self.game_over = False
        self.winner = None
        self.turn_count = 0
    
    def reset(self, num_players=2):
        """Reset the environment to initial state"""
        self.board = [' '] * 9
        self.current_player = 0
        self.game_over = False
        self.winner = None
        self.turn_count = 0
        return self.get_observation()
    
    def get_observation(self) -> Tuple[int, str]:
        """Get current observation for the active player"""
        board_str = self._board_to_string()
        available_moves = [i for i, cell in enumerate(self.board) if cell == ' ']
        
        # Format observation similar to TextArena
        obs = f"""Game Board:
{board_str}

Player {'X' if self.current_player == 0 else 'O'}'s turn.
Available Moves: {['[{}]'.format(i) for i in available_moves]}"""
        
        return self.current_player, obs
    
    def _board_to_string(self) -> str:
        """Convert board to string representation"""
        rows = []
        for i in range(0, 9, 3):
            row = " | ".join(self.board[i:i+3])
            rows.append(row)
        return "\n---------\n".join(rows)
    
    def step(self, action) -> Tuple[bool, Dict[str, Any]]:
        """Execute an action and return (done, info)"""
        # Parse action from string format like "[4]" -> 4
        if isinstance(action, str):
            try:
                action = int(action.strip('[]'))
            except ValueError:
                return False, {"error": f"Invalid action format: {action}"}
        
        # Validate action
        if not (0 <= action <= 8):
            return False, {"error": f"Action out of range: {action}"}
        
        if self.board[action] != ' ':
            return False, {"error": f"Position {action} is already occupied"}
        
        # Make move
        self.board[action] = 'X' if self.current_player == 0 else 'O'
        self.turn_count += 1
        
        # Check for winner
        if self._check_winner():
            self.game_over = True
            self.winner = self.current_player
        elif ' ' not in self.board:
            self.game_over = True
            self.winner = None  # Draw
        
        # Switch players for next turn
        if not self.game_over:
            self.current_player = 1 - self.current_player
        
        info = {
            "winner": self.winner,
            "turn": self.turn_count,
            "board": self.board.copy(),
            "game_over": self.game_over
        }
        
        return self.game_over, info
    
    def _check_winner(self) -> bool:
        """Check if current player has won"""
        # Define winning lines: rows, columns, diagonals
        lines = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # columns
            [0, 4, 8], [2, 4, 6]              # diagonals
        ]
        
        current_mark = 'X' if self.current_player == 0 else 'O'
        
        for line in lines:
            if all(self.board[i] == current_mark for i in line):
                return True
        
        return False
    
    def close(self) -> Tuple[List[float], Dict[str, Any]]:
        """Close the environment and return final results"""
        # Calculate rewards: +1 for winner, -1 for loser, 0 for draw
        if self.winner is None:
            rewards = [0.0, 0.0]  # Draw
        else:
            rewards = [0.0, 0.0]
            rewards[self.winner] = 1.0
            rewards[1 - self.winner] = -1.0
        
        game_info = {
            "winner": self.winner,
            "final_board": self._board_to_string(),
            "game_over": self.game_over,
            "total_turns": self.turn_count,
            "outcome": "draw" if self.winner is None else f"player_{self.winner}_wins"
        }
        
        return rewards, game_info


def make(game_name: str):
    """Factory function to create environment (TextArena-like interface)"""
    if game_name == "TicTacToe-v0":
        return MockTicTacToeEnv()
    else:
        raise ValueError(f"Unknown game: {game_name}")
