
from csv import reader, writer
from math import ceil
from pathlib import Path

import torch
from torch.nn import ReflectionPad2d

from dlgo.goboard_fast import GameState, Move
from dlgo.gotypes import Player, Point

from model.unet import UNet2d
from utils.func import flatten


BOARD_SIZE = 9


class CNNPlayer:
    def __init__(self) -> None:
        noise_size = ceil(BOARD_SIZE / 2)
        pad_size = BOARD_SIZE - noise_size

        self.pad = ReflectionPad2d((0, pad_size, 0, pad_size))
        self.noise = torch.empty([1] + 2 * [noise_size]).normal_(std=1)
        self.padded_noise = self.pad(self.noise)

        self.model = UNet2d(in_channels=2, out_channels=1, width=1, n_conv=1, n_pool=2)

    def select_move(self, game: GameState) -> Move:
        zeros = torch.zeros([1] + 2 * [BOARD_SIZE])
        state = torch.stack((zeros, self.padded_noise), dim=1)

        for point, go_string in game.board._grid.items():
            if go_string:
                state[0, 0, point.row - 1, point.col - 1] = (
                    -1 if go_string.color == Player.black else 1
                )

        with torch.no_grad():
            value = self.model(state).squeeze()

        while True:
            best_value = value.max()

            if best_value < 0:
                return Move.pass_turn()

            row, col = (value == best_value).nonzero()[0]
            point = Point(row.item() + 1, col.item() + 1)
            move = Move.play(point)

            if game.is_valid_move(move):
                return move
            
            value[point.row - 1, point.col - 1] = -torch.inf
            
    def get_parameters(self) -> list[float]:
        params = [p.tolist() for p in self.model.parameters()]
        noise = self.noise.tolist()
        return flatten([params, noise])

    def set_parameters(self, value: list[float]) -> None:
        state = self.model.state_dict()
        a = 0

        for k, p in state.items():
            b = p.numel()
            state[k] = torch.tensor(value[a:a + b]).reshape(p.shape)
            a += b
        
        self.model.load_state_dict(state)
        self.noise = torch.tensor(value[a:]).reshape(self.noise.shape)
        self.padded_noise = self.pad(self.noise)

    def save_parameters(self, file: Path):
        params = self.get_parameters()

        with open(file, "w") as f:
            writer(f).writerow(params)
    
    def load_parameters(self, file: Path):
        with open(file) as f:
            params = list(map(float, next(reader(f))))
        
        self.set_parameters(params)
        

