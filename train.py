from datetime import datetime
from itertools import count
from pathlib import Path
import random
from time import time

from deap.tools.crossover import cxTwoPoint
from deap.tools.mutation import mutGaussian

from dlgo.goboard_fast import GameState
from dlgo.gotypes import Player
from dlgo.scoring import compute_game_result

from players.cnn import CNNPlayer


BOARD_SIZE = 9


def play_game(black_player: CNNPlayer, white_player: CNNPlayer) -> CNNPlayer:
    game = GameState.new_game(BOARD_SIZE)

    while not game.is_over():
        if game.next_player == Player.black:
            move = black_player.select_move(game)
        else:
            move = white_player.select_move(game)

        game = game.apply_move(move)

    result = compute_game_result(game)
    return black_player if result.winner == Player.black else white_player


def select(population: list[CNNPlayer], shuffle: bool = True) -> list[CNNPlayer]:
    winners = []

    for _ in range(2):
        if shuffle:
            random.shuffle(population)

        for i, j in zip(population[0::2], population[1::2]):
            winners.append(play_game(i, j))

    return winners


def crossover(
    population: list[CNNPlayer], p: float, shuffle: bool = True
) -> list[CNNPlayer]:
    if shuffle:
        random.shuffle(population)

    for i, j in zip(population[0::2], population[1::2]):
        if random.random() < p:
            p_i, p_j = cxTwoPoint(i.get_parameters(), j.get_parameters())
            i.set_parameters(p_i)
            j.set_parameters(p_j)

    return population


def mutate(population: list[CNNPlayer], p: float):
    for i in population:
        if random.random() < p:
            p_i = mutGaussian(i.get_parameters(), mu=0, sigma=0.2, indpb=0.05)[0]
            i.set_parameters(p_i)

    return population


def save(population: list[CNNPlayer], output_dir: Path):
    for n, i in enumerate(population):
        i.save_parameters((output_dir / str(n)).with_suffix(".params"))


def train(pop_size: int = 100):
    output_dir = Path("output") / str(datetime.now())
    population = [CNNPlayer() for _ in range(pop_size)]

    for generation in count():
        print(f"\n{generation = }")
        start_time = time()

        generation_dir = output_dir / str(generation)
        generation_dir.mkdir(parents=True)
        save(population, generation_dir)

        population = select(population)
        population = crossover(population, p=0.7)
        population = mutate(population, p=0.2)

        print(f"{time() - start_time:.2f} s")


if __name__ == "__main__":
    train()
