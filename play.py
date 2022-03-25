from argparse import ArgumentParser
from dlgo.goboard_fast import GameState
from dlgo.gotypes import Player
from dlgo.scoring import compute_game_result
from dlgo.utils import print_board

from players.cnn import CNNPlayer
from players.human import HumanPlayer
from players.random import RandomPlayer


BOARD_SIZE = 9


def play(black: str, white: str, verbose: bool):
    player_classes = {
        "human": HumanPlayer,
        "random": RandomPlayer,
    }

    black_player = player_classes.get(black, CNNPlayer)()
    white_player = player_classes.get(white, CNNPlayer)()

    if isinstance(black_player, CNNPlayer):
        black_player.load_parameters(black)
    
    if isinstance(white_player, CNNPlayer):
        white_player.load_parameters(white)

    game = GameState.new_game(BOARD_SIZE)

    while not game.is_over():
        if verbose:
            print_board(game.board)

        if game.next_player == Player.black:
            move = black_player.select_move(game)
        else:
            move = white_player.select_move(game)

        game = game.apply_move(move)
    
    if verbose:
        print_board(game.board)

    result = compute_game_result(game)
    print(result.winner)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-b', '--black')
    parser.add_argument('-w', '--white')
    parser.add_argument('-v', '--verbose', action="store_true")
    play(**vars(parser.parse_args()))
