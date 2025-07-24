import random
import re
from typing import List

class SmartAgent:
    def __init__(self, player_mark):
        self.player_mark = player_mark
        self.last_valid_moves = []

    def __call__(self, observation):
        self.last_valid_moves = self._parse_available_moves(observation)
        
        if not self.last_valid_moves:
            return "[0]"
        
        priority_moves = [4, 0, 2, 6, 8]
        for move in priority_moves:
            if move in self.last_valid_moves:
                return f"[{move}]"
        
        return f"[{random.choice(self.last_valid_moves)}]"

    def _parse_available_moves(self, observation) -> List[int]:
        if "Available Moves" in observation:
            return list(map(int, re.findall(r"'\[(\d+)'", observation.split("Available Moves:")[1])))
        
        board_str = observation.split("Game Board")[1].split("\n")[1:6:2]
        available = []
        for r, row in enumerate(board_str):
            for c in range(3):
                cell = row[1+4*c]
                if cell.strip() == "":
                    available.append(r*3 + c)
        return available