from pathlib import Path
from time import time

import numpy as np

import torch
from torch import Tensor, dtype, nn

from dlgo.goboard_fast import GameState, Move
from dlgo.encoders import SimpleEncoder

from torchvision.models import squeezenet1_1


class CNNPlayer:
    def __init__(self, dtype: dtype = torch.float32, device: str = "cpu"):
        super().__init__()
        self.dtype = dtype
        self.device = device
        self.fitness = 0
        self.encoder = SimpleEncoder((19, 19))
        self.model = nn.Sequential(
            nn.Conv2d(self.encoder.num_planes, 3, 1),
            squeezenet1_1(num_classes=1),
        ).to(dtype=dtype, device=device)
        self.model.eval()

    def select_move(self, game: GameState) -> Move:
        moves = game.legal_moves()
        batch = map(game.apply_move, moves)
        batch = map(self.encoder.encode, batch)
        batch = torch.tensor(np.array(list(batch))).to(dtype=self.dtype, device=self.device)
        value = self.model(batch)
        return moves[value.argmax().item()]

    def parameters(self) -> list[Tensor]:
        return list(self.model.parameters())

    def save(self, file: Path):
        torch.save(self.model.state_dict(), file)

    def load(self, file: Path):
        self.model.load_state_dict(torch.load(file))
