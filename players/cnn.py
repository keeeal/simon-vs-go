
from pathlib import Path

import torch

from dlgo.goboard_fast import GameState, Move
from dlgo.gotypes import Player, Point
from dlgo.utils import print_board

from model.unet import UNet2d
from utils.func import flatten


class CNNPlayer:
    def __init__(self) -> None:
        self.model = UNet2d(in_channels=1, out_channels=1, width=1, n_conv=1, n_pool=2)

    def select_move(self, game: GameState) -> Move:
        num_rows, num_cols = game.board.num_rows, game.board.num_cols
        state = torch.zeros([1, 1, num_rows, num_cols])

        for point, go_string in game.board._grid.items():
            if go_string:
                state[0, 0, point.row - 1, point.col - 1] = (
                    -1 if go_string.color == Player.black else 1
                )

        with torch.no_grad():
            value = self.model(state).squeeze()

        while True:
            row, col = (value == torch.max(value)).nonzero()[0]
            point = Point(row.item() + 1, col.item() + 1)
            move = Move.play(point)

            if game.is_valid_move(move):
                return move
            
            value[point.row - 1, point.col - 1] = -torch.inf
            
            if torch.all(value == -torch.inf):
                return Move.resign()

    def get_parameters(self) -> list[float]:
        return flatten([p.tolist() for p in self.model.parameters()])

    def set_parameters(self, value: list[float]) -> None:
        state = self.model.state_dict()
        a = 0

        for k, p in state.items():
            b = p.numel()
            state[k] = torch.tensor(value[a:a + b]).reshape(p.shape)
            a += b
        
        self.model.load_state_dict(state)

    def save_parameters(self, path: Path):
        torch.save(self.model.state_dict(), path)
