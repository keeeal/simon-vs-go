import random

from deap.tools.crossover import cxTwoPoint
from deap.tools.mutation import mutGaussian

from dlgo.goboard_fast import GameState
from dlgo.gotypes import Player
from dlgo.scoring import compute_game_result
from dlgo.utils import print_board

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

        for black_player, white_player in zip(population[0::2], population[1::2]):
            winners.append(play_game(black_player, white_player))

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
            p_i = mutGaussian(i.get_parameters(), 0, 0.1, 0.1)[0]
            i.set_parameters(p_i)

    return population


def train(pop_size: int = 100):
    population = [CNNPlayer() for _ in range(pop_size)]

    for generation in range(1000):
        print(generation)
        population = select(population)
        population = crossover(population, 0.7)
        population = mutate(population, 0.1)


if __name__ == "__main__":
    train()
