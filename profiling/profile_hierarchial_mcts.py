import cProfile
import pstats
import sys
from agents.agent_MCTS.hierarchical_mcts import HierachicalMCTSAgent
from game_utils import initialize_game_state, PLAYER1
from memory_profiler import profile
from line_profiler import LineProfiler
from agents.agent_MCTS.hierarchical_mcts import HierarchicalMCTSAgent
from game_utils import initialize_game_state, PLAYER1

# Redirect memory profiling output to a file
memory_log = open('memory_profile_hierarchical_mcts.log', 'w')
sys.stdout = memory_log  # Redirect sys.stdout directly to the file

@profile
def memory_profiled_hierarchical_mcts_move():
    """Memory profile the Hierarchical MCTS move function."""
    board = initialize_game_state()
    player = PLAYER1
    saved_state = None
    agent = HierarchicalMCTSAgent(iterationnumber=100, max_depth_for_minmax=10, max_simulation_depth=40)
    action, saved_state = agent.mcts_move(board.copy(), player, saved_state, player_name="Player1")

def line_profiled_hierarchical_mcts_move():
    """Line profile the Hierarchical MCTS move function."""
    profiler = LineProfiler()
    board = initialize_game_state()
    player = PLAYER1
    saved_state = None
    agent = HierarchicalMCTSAgent(iterationnumber=50, max_depth_for_minmax=10, max_simulation_depth=40)
    profiler.add_function(agent.mcts_move)
    profiler.enable()
    action, saved_state = agent.mcts_move(board.copy(), player, saved_state, player_name="Player1")
    profiler.disable()
    # Save line profiling results to a file
    with open('line_profile_hierarchical_mcts.log', 'w') as line_log:
        profiler.print_stats(stream=line_log)

def cpu_profiled_hierarchical_mcts_move():
    """CPU profile the Hierarchical MCTS move function."""
    board = initialize_game_state()
    player = PLAYER1
    saved_state = None
    agent = HierarchicalMCTSAgent(iterationnumber=50, max_depth_for_minmax=10, max_simulation_depth=40)

    profiler = cProfile.Profile()
    profiler.enable()
    action, saved_state = agent.mcts_move(board.copy(), player, saved_state, player_name="Player1")
    profiler.disable()

    # Save CPU profiling results to a file
    with open('cpu_profile_hierarchical_mcts.log', 'w') as cpu_log:
        stats = pstats.Stats(profiler, stream=cpu_log)
        stats.strip_dirs().sort_stats('tottime').print_stats(20)

if __name__ == "__main__":
    # Run memory profiling
    memory_profiled_hierarchical_mcts_move()
    memory_log.close()

    # Restore stdout for CPU profiling
    sys.stdout = sys.__stdout__

    # Run line profiling
    line_profiled_hierarchical_mcts_move()
    cpu_profiled_hierarchical_mcts_move()