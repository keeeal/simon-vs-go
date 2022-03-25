from dlgo.goboard_fast import GameState, Move
from dlgo.utils import point_from_coords

class HumanPlayer:
    def __init__(self, board_size: int) -> None:
        pass

    def select_move(self, game: GameState) -> Move:
        while True:
            i = input("-- ").upper().strip()
            if not i:
                continue

            p = point_from_coords (i)
            move = Move.play(p)

            if game.board.is_on_grid(p) and game.is_valid_move(move):
                return move

            print(f"Invalid move: {i}")
