from random import randint

from dlgo.goboard_fast import GameState, Move
from dlgo.gotypes import Point


class RandomPlayer:
    def __init__(self, board_size: int) -> None:
        pass
    
    def select_move(self, game: GameState) -> Move:
        while True:
            point = Point(
                row=randint(1, game.board.num_rows),
                col=randint(1, game.board.num_cols),
            )

            move = Move.play(point)

            if game.is_valid_move(move):
                return move
