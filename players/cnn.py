
from csv import reader, writer
from math import ceil
from pathlib import Path
from random import gauss

import torch
from torch.nn import ReflectionPad2d

from dlgo.goboard_fast import GameState, Move
from dlgo.gotypes import Player, Point

from model.unet import UNet2d
from utils.func import flatten


class CNNPlayer:
    def __init__(self, board_size: int) -> None:
        self.board_size = board_size
        self.noise_size = ceil(board_size / 2)
        self.pad = ReflectionPad2d(2 * (0, self.board_size - self.noise_size))
        self.noise = torch.empty([1] + 2 * [self.noise_size])
        self.model = UNet2d(in_channels=2, out_channels=1, width=1, n_conv=1, n_pool=2)

        n_params = len(self.get_parameters())
        self.set_parameters([gauss(mu=0, sigma=1) for _ in range(n_params)])

    def select_move(self, game: GameState) -> Move:
        zeros = torch.zeros([1] + 2 * [self.board_size])
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
    
    def set_noise(self, value: list[float]):
        self.noise = torch.tensor(value).reshape(self.noise.shape)
        self.padded_noise = self.pad(self.noise)
    
    def get_parameters(self) -> list[float]:
        params = [p.tolist() for p in self.model.parameters()]
        noise = self.noise.tolist()
        return flatten([params, noise])

    def set_parameters(self, value: list[float]):
        state = self.model.state_dict()
        a = 0

        for k, p in state.items():
            b = p.numel()
            state[k] = torch.tensor(value[a:a + b]).reshape(p.shape)
            a += b
        
        self.model.load_state_dict(state)
        self.set_noise(value[a:])

    def save_parameters(self, file: Path):
        params = self.get_parameters()

        with open(file, "w") as f:
            writer(f).writerow(params)
    
    def load_parameters(self, file: Path):
        with open(file) as f:
            params = list(map(float, next(reader(f))))
        
        self.set_parameters(params)
        

