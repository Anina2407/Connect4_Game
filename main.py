"""
main.py - Connect Four Agent Evaluation and Game Runner

This script serves as the entry point for evaluating and testing various Connect Four agents,
including human, random, MCTS, hierarchical MCTS, and AlphaZero-based agents.
It provides CLI or hardcoded options for running different matchups between agents
and collects performance metrics such as win rates, move durations, and legality statistics.

Features:
- Support for Human vs AI, AI vs AI games
- Evaluation of AlphaZero agent checkpoints
- Automatic GPU/MPS/CPU device selection
- Plots and saves performance metrics for analysis
"""

import os
import time
import torch
from typing import Callable

from game_utils import (
    PLAYER1, PLAYER2, PLAYER1_PRINT, PLAYER2_PRINT,
    GameState, MoveStatus, GenMove,
    initialize_game_state, pretty_print_board,
    apply_player_action, check_end_state, check_move_status,
)

from agents.agent_human_user import user_move
from agents.agent_random import generate_move as random_move
from agents.agent_MCTS.mcts import MCTSAgent
from agents.agent_MCTS.hierarchical_mcts import HierarchicalMCTSAgent
from agents.agent_MCTS.alphazero_mcts import AlphazeroMCTSAgent
from agents.alphazero.network import Connect4Net
from agents.alphazero.inference import policy_value

from metrics.metrics import GameMetrics

# Device selection
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA GPU")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple MPS backend")
else:
    device = torch.device("cpu")
    print("Using CPU")


def agent_vs_agent(
    generate_move_1: GenMove,
    generate_move_2: GenMove = user_move,
    player_1: str = "Player 1",
    player_2: str = "Player 2",
    args_1: tuple = (),
    args_2: tuple = (),
    init_1: Callable = lambda board, player: None,
    init_2: Callable = lambda board, player: None,
    metrics: GameMetrics = None,
    verbose: bool = True
) -> tuple:
    """
    Run a game between two agents (or humans) with optional metric tracking.

    Args:
        generate_move_1 (GenMove): Move function for player 1.
        generate_move_2 (GenMove): Move function for player 2.
        player_1 (str): Name of player 1.
        player_2 (str): Name of player 2.
        args_1 (tuple): Args for player 1.
        args_2 (tuple): Args for player 2.
        init_1 (Callable): Optional init for player 1.
        init_2 (Callable): Optional init for player 2.
        metrics (GameMetrics): Metrics object to record data.
        verbose (bool): If True, print detailed game info.

    Returns:
        tuple: (results list, metrics object)
    """
    if metrics is None:
        metrics = GameMetrics()

    players = (PLAYER1, PLAYER2)
    results = []
    for play_first in (1, -1):
        if play_first == 1:
            inits = (init_1, init_2)
            gen_moves = (generate_move_1, generate_move_2)
            player_names = (player_1, player_2)
            gen_args = (args_1, args_2)
        else:
            inits = (init_2, init_1)
            gen_moves = (generate_move_2, generate_move_1)
            player_names = (player_2, player_1)
            gen_args = (args_2, args_1)

        for init, player in zip(inits, players):
            init(initialize_game_state(), player)

        saved_state = {PLAYER1: None, PLAYER2: None}
        board = initialize_game_state()
        player_name_map = {players[0]: player_names[0], players[1]: player_names[1]}

        playing = True
        while playing:
            for player, player_name, gen_move, args in zip(players, player_names, gen_moves, gen_args):
                t0 = time.time()
                if verbose:
                    print(pretty_print_board(board))
                    print(f'{player_name} you are playing with {PLAYER1_PRINT if player == PLAYER1 else PLAYER2_PRINT}')

                action, saved_state[player] = gen_move(
                    board.copy(), player, saved_state[player], player_name, metrics, *args
                )
                elapsed = time.time() - t0
                if verbose:
                    print(f'Move time: {elapsed:.3f}s')

                move_status = check_move_status(board, action)
                is_legal = move_status == MoveStatus.IS_VALID
                metrics.record_move(player_name, elapsed, is_legal)

                if move_status != MoveStatus.IS_VALID:
                    print(f'Move {action} is invalid: {move_status.value}')
                    print(f'{player_name} lost by making an illegal move.')
                    metrics.record_result(player_name, 'loss')
                    opponent = PLAYER2 if player == PLAYER1 else PLAYER1
                    metrics.record_result(player_name_map[opponent], 'win')
                    playing = False
                    results.append('Error')
                    break

                apply_player_action(board, action, player)
                end_state = check_end_state(board, player)

                if end_state != GameState.STILL_PLAYING:
                    if verbose:
                        print(pretty_print_board(board))
                    if end_state == GameState.IS_DRAW:
                        print('Game ended in draw')
                        for name in player_name_map.values():
                            metrics.record_result(name, 'draw')
                        results.append('Draw')
                    else:
                        winner_name = player_name
                        loser = PLAYER2 if player == PLAYER1 else PLAYER1
                        loser_name = player_name_map[loser]
                        print(f'{winner_name} won playing {PLAYER1_PRINT if player == PLAYER1 else PLAYER2_PRINT}')
                        metrics.record_result(winner_name, 'win')
                        metrics.record_result(loser_name, 'loss')
                        results.append(PLAYER1_PRINT if player == PLAYER1 else PLAYER2_PRINT)
                    playing = False
                    break
    return results, metrics


def run_alphazero_vs_mcts(num_games: int, alpha_iterations=100):
    """
    Evaluate AlphaZero agent against MCTS baseline.

    Args:
        num_games (int): Number of games to play.
        alpha_iterations (int): MCTS sims for AlphaZero agent.

    Returns:
        GameMetrics: Collected metrics from all games.
    """
    total_metrics = GameMetrics()
    model = Connect4Net()
    checkpoint = torch.load(f"checkpoints/iteration_{itterarionNumber}.pt", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    alpha_agent = AlphazeroMCTSAgent(
        policy_value=lambda state: policy_value(state, model, device),
        iterationnumber=alpha_iterations
    )

    alpha_wins_started = 0
    alpha_wins_not_started = 0
    mcts_wins_started = 0
    mcts_wins_not_started = 0
    draws = 0
    errors = 0

    for game_idx in range(num_games):
        print(f"\nGame {game_idx+1}/{num_games}")
        verbose = (game_idx == 0)

        if game_idx % 2 == 0:
            agent1, agent2 = alpha_agent, MCTSAgent(100)
            player1, player2 = "AlphaZero Agent", "MCTS Agent"
        else:
            agent1, agent2 = MCTSAgent(100), alpha_agent
            player1, player2 = "MCTS Agent", "AlphaZero Agent"

        results, _ = agent_vs_agent(
            agent1, agent2, player_1=player1, player_2=player2, metrics=total_metrics, verbose=verbose
        )

        if game_idx % 2 == 0:
            if results[0] == PLAYER1_PRINT:
                alpha_wins_started += 1
            elif results[0] == PLAYER2_PRINT:
                mcts_wins_not_started += 1
        else:
            if results[0] == PLAYER1_PRINT:
                mcts_wins_started += 1
            elif results[0] == PLAYER2_PRINT:
                alpha_wins_not_started += 1

        if results[0] == 'Draw':
            draws += 1
        elif results[0] == 'Error':
            errors += 1

    print(f"\nResults after {num_games} games:")
    print(f"AlphaZero wins when starting: {alpha_wins_started}")
    print(f"AlphaZero wins when not starting: {alpha_wins_not_started}")
    print(f"MCTS wins when starting: {mcts_wins_started}")
    print(f"MCTS wins when not starting: {mcts_wins_not_started}")
    print(f"Draws: {draws}")
    print(f"Errors: {errors}")
    return total_metrics


def run_alphazero_vs_random(num_games: int, alpha_iterations=100):
    """
    Evaluate AlphaZero agent against a Random agent.

    Args:
        num_games (int): Number of games to play.
        alpha_iterations (int): MCTS sims for AlphaZero agent.

    Returns:
        GameMetrics: Collected metrics from all games.
    """
    total_metrics = GameMetrics()
    model = Connect4Net()
    checkpoint = torch.load(f"checkpoints/deep/iteration_{itterarionNumber}.pt", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    alpha_agent = AlphazeroMCTSAgent(
        policy_value=lambda state: policy_value(state, model, device),
        iterationnumber=alpha_iterations
    )

    alpha_wins_started = 0
    alpha_wins_not_started = 0
    random_wins_started = 0
    random_wins_not_started = 0
    draws = 0
    errors = 0

    for game_idx in range(num_games):
        print(f"\nGame {game_idx+1}/{num_games}")
        verbose = (game_idx == 0)

        if game_idx % 2 == 0:
            agent1, agent2 = alpha_agent, random_move
            player1, player2 = "AlphaZero Agent", "Random Agent"
        else:
            agent1, agent2 = random_move, alpha_agent
            player1, player2 = "Random Agent", "AlphaZero Agent"

        results, _ = agent_vs_agent(
            agent1, agent2, player_1=player1, player_2=player2, metrics=total_metrics, verbose=verbose
        )

        if game_idx % 2 == 0:
            if results[0] == PLAYER1_PRINT:
                alpha_wins_started += 1
            elif results[0] == PLAYER2_PRINT:
                random_wins_not_started += 1
        else:
            if results[0] == PLAYER1_PRINT:
                random_wins_started += 1
            elif results[0] == PLAYER2_PRINT:
                alpha_wins_not_started += 1

        if results[0] == 'Draw':
            draws += 1
        elif results[0] == 'Error':
            errors += 1

    print(f"\nResults after {num_games} games:")
    print(f"AlphaZero wins when starting: {alpha_wins_started}")
    print(f"AlphaZero wins when not starting: {alpha_wins_not_started}")
    print(f"Random wins when starting: {random_wins_started}")
    print(f"Random wins when not starting: {random_wins_not_started}")
    print(f"Draws: {draws}")
    print(f"Errors: {errors}")
    return total_metrics


if __name__ == "__main__":
    print("Connect Four Game")
    print("Choose game mode:")
    print("1: User vs Random Agent")
    print("2: User vs MCTS Agent")
    print("3: MCTS Agent vs Random Agent (baseline test)")
    print("4: Human vs Human (2 players)")
    print("5: MCTS Agent vs Hierarchical MCTS Agent")
    print("6: Hierarchical MCTS Agent vs Random Agent (baseline test)")
    print("7: AlphaZero Agent vs Random Agent (baseline test)")
    print("8: AlphaZero Agent vs MCTS Agent (performance test)")

    mode = "8"
    metrics = GameMetrics()
    itterarionNumber = 0

    for it in [1, 17, 18, 19]:
        itterarionNumber = it

        if mode == "7":
            num_games = 20
            metrics = run_alphazero_vs_random(num_games)
        elif mode == "8":
            num_games = 20
            metrics = run_alphazero_vs_mcts(num_games)

        print("\nFinal Performance Metrics:")
        print(metrics)

        metrics.plot_results(save_path=f"plots/{itterarionNumber}/{'MCTS' if mode == '8' else 'random'}/results.png")
        for agent in metrics.agents:
            metrics.plot_move_duration_distribution(agent, save_path=f"plots/{itterarionNumber}/{'MCTS' if mode == '8' else 'random'}/move_duration_{agent}.png")
        metrics.plot_performance_radar(save_path=f"plots/{itterarionNumber}/{'MCTS' if mode == '8' else 'random'}/radar.png")
