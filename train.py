from argparse import ArgumentParser
from copy import deepcopy
from datetime import datetime
from itertools import count
from pathlib import Path
from random import random, shuffle
from time import time

from deap.tools.crossover import cxTwoPoint
from deap.tools.selection import selTournament

from dlgo.goboard_fast import GameState
from dlgo.gotypes import Player
from dlgo.scoring import compute_game_result
from dlgo.utils import print_board

from players.cnn import CNNPlayer


def play_game(
    black_player: CNNPlayer, white_player: CNNPlayer, board_size: int
) -> CNNPlayer:
    game = GameState.new_game(board_size)

    while not game.is_over():
        if game.next_player == Player.black:
            move = black_player.select_move(game)
        else:
            move = white_player.select_move(game)

        game = game.apply_move(move)

    result = compute_game_result(game)

    print_board(game.board)
    print(result)

    return result.b - result.w - result.komi


def evaluate(population: list[CNNPlayer], board_size: int):
    for black, white in zip(population[0::2], population[1::2]):
        black.fitness = play_game(black, white, board_size)
        white.fitness = -black.fitness


def select(population: list[CNNPlayer]) -> list[CNNPlayer]:
    population = selTournament(population, k=len(population), tournsize=3)
    return list(map(deepcopy, population))


def crossover(population: list[CNNPlayer], p: float) -> list[CNNPlayer]:
    shuffle(population)

    for i, j in zip(population[0::2], population[1::2]):
        if random() < p:
            p_i, p_j = cxTwoPoint(i.get_parameters(), j.get_parameters())
            i.set_parameters(p_i)
            j.set_parameters(p_j)

    return population


def mutate(population: list[CNNPlayer], p: float, board_size: int):
    for i in population:
        if random() < p:
            params = i.get_parameters()
            delta = CNNPlayer(board_size).get_parameters()
            p_i = [x + dx * (random() < 0.05) for x, dx in zip(params, delta)]
            i.set_parameters(p_i)

    return population


def save(population: list[CNNPlayer], output_dir: Path):
    for n, i in enumerate(population):
        i.save_parameters((output_dir / str(n)).with_suffix(".params"))


def train(board_size: int, pop_size: int, p_crossover: float, p_mutate: float):
    output_dir = Path("output") / str(datetime.now())
    population = [CNNPlayer(board_size) for _ in range(pop_size)]

    for generation in count():
        print(f"\n{generation = }")
        start_time = time()

        generation_dir = output_dir / str(generation)
        generation_dir.mkdir(parents=True)
        save(population, generation_dir)

        evaluate(population, board_size)
        population = select(population)
        population = crossover(population, p_crossover)
        population = mutate(population, p_mutate, board_size)

        print(f"{time() - start_time:.2f} s")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-b", "--board-size", type=int, default=9)
    parser.add_argument("-pop", "--pop-size", type=int, default=300)
    parser.add_argument("-pcx", "--p-crossover", type=float, default=0.7)
    parser.add_argument("-pmut", "--p-mutate", type=float, default=0.2)
    train(**vars(parser.parse_args()))
