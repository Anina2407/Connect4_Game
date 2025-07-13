# Connect Four Game Simulation

This project implements intelligent software agents that play the game **Connect Four**.  
The focus is on applying principles of **software design**, **modular architecture**, and **clean code**.

> 💡 See our planning board: [Miro Link](https://miro.com/app/board/uXjVIsyA0Qk=/)



## Agents

- **MCTS Agent:** Classic Monte Carlo Tree Search agent.
- **Hierarchical MCTS Agent:** MCTS with heuristics (immediate win/lose detection, two/three-in-a-row preference).
- **AlphaZero MCTS Agent:** MCTS guided by a neural network.
   - different implementation regarding training and network architecture 
- **Random Agent:** Selects random valid moves.
- **Human Agent:** Allows a human to play via console input.

## Getting Started

### Requirements

- Python 3.8+
- See `requirements.txt` for all dependencies.

### Installation

1. Clone this repository.
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

### Running Simulations

To run a series of games between the MCTS agent and the Random agent:

```bash
python main.py
```

You will be prompted to select the game mode (e.g., Human vs Random Agent, Human vs MCTS Agent, ...).

--- 

### Running Tests

To run all tests:

```bash
pytest tests/
```

## 🗂️ Project Structure 🗂️

```
Game/
├── agents/
│   ├── agent_MCTS/
│   │   ├── mcts.py
│   │   ├── improved_mcts.py
│   │   ├── hierarchical_mcts.py
│   │   ├── alphazero_mcts.py
│   │   ├── node.py
│   │   └── __init__.py
│   ├── agent_random/
│   │   └── __init__.py
│   └── agent_human_user/
│       ├── human_user.py
│       └── __init__.py
├── alphazero/
│   ├── network.py
|   ├── network_CNN.py
│   ├── inference.py 
│   ├── model.pt
│   └── train_dummy_data.py
├── profiling/
│   ├── mcts_profile.stats 
│   ├── profile_alphanet.py
│   ├── profile_gpu.py
│   ├── profile_hirachical_mcts.py
│   ├── profile_mcts.py
│   └── ProfilingReport.py
├── metrics/
│   ├── metrics.py
│   └── __init__.py
├── tests/
│   ├── test_game_utils.py
│   ├── test_node.py
│   ├── test_mcts.py
│   ├── test_hierarchical_mcts.py
│   ├── test_metrics.py
│   └── test_training.py
├── game_utils.py
├── main.py
├── requirements.txt
├── train_alphazero_CNN.py
└── train_alphazero.py
```
## Training Data & Pretrained Data
Pre-trained AlphaZero models (with residual layers) are available here:
[ Google Drive Link](https://drive.google.com/drive/folders/1S6eljs_s0Wlq_DL237q-xXZV4ZdG_5Ou)
Default used: Iteration 17
To switch, update the checkpoint folder and the path in main.py
To train more data with residual Layer: 
```bash
python train_alphazero.py
```

Pre-trained AlphaZero models (with only convolutional Layer) are available here:
[ Google Drive Link]()
To switch, update the checkpoint folder and the path in main.py

To train more data with convolutional Layer, replace imports in all files (search for 'CNN' to quickly locate the relevant sections): 
```bash
python train_alphazero_CNN.py
```

## Profiling
Performance scripts for benchmarking different agents are available under profiling/.

## Code of Honor & Acknowledgements
- This project was build by Mohammad, Reihaneh, Shokoofeh and Anina
- AI tools (GitHub Copilot and ChatGPT) were used to assist with documentation, code refactoring, and test coverage ideas
- All code was authored by humans first, and AI-generated suggestions were reviewed before integration.

