from argparse import ArgumentParser
from copy import deepcopy
from datetime import datetime
from itertools import count
from pathlib import Path
from random import gauss, random, randint, shuffle

import torch
from torch import Tensor
from tqdm import tqdm

from deap.tools.selection import selTournament

from dlgo.goboard_fast import GameState
from dlgo.gotypes import Player
from dlgo.scoring import evaluate_territory
from dlgo.utils import print_board

from players.cnn import CNNPlayer


def play_game(
    black_player: CNNPlayer,
    white_player: CNNPlayer,
    board_size: int,
    komi: float = 6.5,
) -> float:
    game = GameState.new_game(board_size)

    while not game.is_over():
        if game.next_player == Player.black:
            move = black_player.select_move(game)
        else:
            move = white_player.select_move(game)

        game = game.apply_move(move)

    territory = evaluate_territory(game.board)
    result = territory.num_black_territory - territory.num_white_territory - komi

    print_board(game.board)
    print(result)

    return result


def evaluate(population: list[CNNPlayer], board_size: int):
    shuffle(population)

    for i, j in zip(tqdm(population[0::2]), population[1::2]):
        f_i = play_game(i, j, board_size)
        f_j = play_game(j, i, board_size)
        i.fitness = f_i - f_j
        j.fitness = f_j - f_i


def select(population: list[CNNPlayer]) -> list[CNNPlayer]:
    population = selTournament(population, k=len(population), tournsize=7)
    return list(map(deepcopy, population))


def cross_tensors(t_i: Tensor, t_j: Tensor):
    x = randint(0, min(t_i.numel(), t_j.numel()))
    t_i, t_j = t_i.flatten(), t_j.flatten()
    t_i[x:], t_j[x:] = t_j[x:], torch.clone(t_i[x:])


def crossover(population: list[CNNPlayer], p: float) -> list[CNNPlayer]:
    shuffle(population)
    n = 0

    for i, j in zip(tqdm(population[0::2]), population[1::2]):
        if random() < p:
            for t_i, t_j in zip(i.parameters(), j.parameters()):
                cross_tensors(t_i, t_j)

            n += 2

    print(f"Crossed {n} individuals.")

    return population


def mutate_tensor(t: Tensor, p: float = 0.05):
    for n in range(t.numel()):
        if random() < p:
            t.flatten()[n] *= gauss(1, 1)


def mutate(population: list[CNNPlayer], p: float):
    n = 0

    for i in tqdm(population):
        if random() < p:
            for t in i.parameters():
                mutate_tensor(t)

            n += 1

    print(f"Mutated {n} individuals.")

    return population


def save(population: list[CNNPlayer], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    for n, i in enumerate(tqdm(population)):
        i.save((output_dir / str(n)).with_suffix(".params"))


def train(board_size: int, pop_size: int, p_crossover: float, p_mutate: float):
    output_dir = Path("output") / str(datetime.now())

    print("\nGenerating population...")
    population = [CNNPlayer() for _ in tqdm(range(pop_size))]

    for generation in count():
        heading = f"\n{generation = }".upper()
        print(f"{heading}\n{'=' * len(heading)}")

        print("\nSaving...")
        save(population, output_dir)  # / str(generation)

        print("\nEvaluating...")
        evaluate(population, board_size)

        print("\nSelecting...")
        population = select(population)

        print("\nCrossing...")
        population = crossover(population, p=p_crossover)

        print("\nMutating...")
        population = mutate(population, p=p_mutate)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-b", "--board-size", type=int, default=19)
    parser.add_argument("-pop", "--pop-size", type=int, default=500)
    parser.add_argument("-px", "--p-crossover", type=float, default=0.5)
    parser.add_argument("-pm", "--p-mutate", type=float, default=0.2)

    with torch.no_grad():
        train(**vars(parser.parse_args()))
