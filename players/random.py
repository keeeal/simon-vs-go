from itertools import product
from random import choice

from dlgo.goboard_fast import GameState, Move
from dlgo.gotypes import Point


class RandomPlayer:
    def __init__(self, board_size: int) -> None:
        self.moves = [
            Move.play(Point(row + 1, col + 1))
            for row, col in product(range(board_size), repeat=2)
        ]
        self.moves.append(Move.pass_turn())

    def select_move(self, game: GameState) -> Move:
        valid_moves = list(map(game.is_valid_move, self.moves))
        return choice(valid_moves)
