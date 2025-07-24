def parse_available_moves(observation):
    """Parse the available moves from the game observation."""
    if "Available Moves" in observation:
        return list(map(int, re.findall(r"'\[(\d+)'", observation.split("Available Moves:")[1])))
    
    board_str = observation.split("Game Board")[1].split("\n")[1:6:2]
    available = []
    for r, row in enumerate(board_str):
        for c in range(3):
            cell = row[1 + 4 * c]
            if cell.strip() == "":
                available.append(r * 3 + c)
    return available

def parse_game_state(observation):
    """Parse the game state from the observation."""
    # Extract relevant game state information
    game_state = {}
    game_state['board'] = observation.split("Game Board")[1].split("\n")[1:6:2]
    game_state['current_player'] = observation.split("Current Player: ")[1].split("\n")[0].strip()
    game_state['available_moves'] = parse_available_moves(observation)
    return game_state