from dlgo import goboard_fast as goboard
from dlgo import gotypes
from dlgo.utils import print_board, print_move, point_from_coords

from players.cnn import CNNPlayer as Player


def main():
    board_size = 9
    game = goboard.GameState.new_game(board_size)
    bot = Player()

    while not game.is_over():
        print_board(game.board)

        if game.next_player == gotypes.Player.black:
            while True:
                i = input("-- ").upper().strip()
                if not i:
                    continue

                p = point_from_coords(i)
                move = goboard.Move.play(p)

                if game.board.is_on_grid(p) and game.is_valid_move(move):
                    break

                print(f"Invalid move: {i}")
        else:
            move = bot.select_move(game)

        print_move(game.next_player, move)
        game = game.apply_move(move)


if __name__ == "__main__":
    main()
