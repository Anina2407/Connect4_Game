import numpy as np
from typing import Callable, Any
from metrics.metrics import GameMetrics
from game_utils import BoardPiece, PlayerAction, SavedState, MoveStatus, check_move_status


def query_user(prompt_function: Callable) -> Any:
    """
    Prompts the user for input using the provided prompt function.

    Args:
        prompt_function (Callable): A function that takes a prompt string and returns user input.

    Returns:
        Any: The input received from the user via the prompt function.
    """
    usr_input = prompt_function("Column? ")
    return usr_input


def user_move(board: np.ndarray,_player: BoardPiece,
              saved_state: SavedState | None,
              player_name: str,
              metrics: GameMetrics | None = None) -> tuple[PlayerAction, SavedState | None]:
    """
    Prompts the user to input a move for the Connect4 game, validates the move, and returns the chosen action along with the saved state.
    Args:
        board (np.ndarray): The current game board as a NumPy array.
        _player (BoardPiece): The piece representing the current player.
        saved_state (SavedState | None): The current saved state, or None if not applicable.
        player_name (str): The name of the player making the move.
        metrics (GameMetrics | None, optional): An optional object for recording move metrics. Defaults to None.
    Returns:
        tuple[PlayerAction, SavedState | None]: A tuple containing the player's chosen action and the (possibly updated) saved state.
    Notes:
        - The function repeatedly prompts the user until a valid move is entered.
        - If an invalid move is attempted, the function records the attempt (if metrics is provided) and prompts the user again.
    """
    move_status = None
    while move_status != MoveStatus.IS_VALID:
        input_move_string = query_user(input)
        input_move = convert_str_to_action(input_move_string, _player,player_name, metrics)
        if input_move is None:
            continue
        move_status = check_move_status(board, input_move)
        if move_status != MoveStatus.IS_VALID:
            metrics.record_move(
                player_name, 0.0, is_legal=False
            )
            # print('metrics', metrics)
            print(f'Move is invalid: {move_status.value}')
            print('Try again.')

    return input_move, saved_state

def convert_str_to_action(input_move_string: str, _player: BoardPiece, player_name: str, metrics: GameMetrics | None = None) -> PlayerAction | None:
    """
    Converts a string input representing a move into a PlayerAction object.

    Attempts to parse the input string as a PlayerAction. If the input is invalid,
    records the move as illegal in the provided metrics (if any), prints an error message,
    and returns None.

    Args:
        input_move_string (str): The string input representing the player's move.
        _player (BoardPiece): The player making the move (unused in this function).
        player_name (str): The name of the player making the move, used for metrics recording.
        metrics (GameMetrics | None, optional): An optional metrics object for recording move legality.

    Returns:
        PlayerAction | None: The corresponding PlayerAction if the input is valid, otherwise None.
    """
    try:
        input_move = PlayerAction(input_move_string)
    except ValueError:
        input_move = None
        metrics.record_move(
                player_name, 0.0, is_legal=False
            )
        print('Invalid move: Input must be an integer.')
        print('Try again.')
    return input_move