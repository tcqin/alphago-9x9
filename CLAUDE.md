# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Run tests:**
```bash
python tests/test_go_engine.py
```

**Training:**
```bash
python train_sl.py        # Supervised learning from professional games
python train_rollout.py   # Rollout policy training
python train_rl.py        # Reinforcement learning via self-play
python train_value.py     # Value network training
```

**Play / Evaluate:**
```bash
python human_play.py      # Interactive game vs AI
python mcts.py            # Run a complete MCTS game
python evaluate.py        # Evaluate model performance
python self_play.py       # Generate self-play games
```

**Data preprocessing:**
```bash
python filter_9x9.py
python filter_9x9_winner_elo.py
```

There is no package manager config — dependencies (PyTorch, sgfmill, numpy) must be installed manually. All scripts use `torch.device("mps")` targeting Apple Silicon.

## Architecture

This is an AlphaGo implementation for 9x9 Go following the original paper's three-network design.

### Networks (`policy_net.py`)
- **PolicyNetwork**: 37-plane input (board features) → 81-move probability output. 8 conv layers, 128 filters. Used for both SL and RL.
- **RolloutNetwork**: 5-plane input → 81-move output. 2 conv layers, 32 filters. Lightweight for fast rollout simulation.
- **ValueNetwork**: 37-plane input → scalar output (Tanh). Predicts game outcome.

### Feature Encoding (`features.py`)
Board state is encoded as multi-plane tensors. PolicyNetwork uses 37 planes: our stones, opponent stones, empty, 8 one-hot liberty planes each for self/opponent, 8 one-hot recency planes each for self/opponent, Ko point, bias. RolloutNetwork uses 5 planes: our/opponent/empty/Ko/bias.

### Game Engine (`go_engine.py`)
`GoGame` manages full 9x9 game state: board, Ko tracking, move history, captures. Includes true eye detection (used during rollout/self-play to prevent filling own eyes). Precomputed neighbor/diagonal lookups for speed.

### MCTS (`mcts.py`)
Standard PUCT selection, expansion via top-20 policy network moves, rollout to game end using `RolloutNetwork`, backpropagation. Key parameters: 200 simulations/move, `c=1.4`, max rollout depth 80.

### Training Flow
1. **SL** (`train_sl.py`): Cross-entropy on 1500+ professional 9x9 games. Best checkpoint: `models_filtered/sl_network.pth` (epoch 8).
2. **Rollout** (`train_rollout.py`): Same dataset, 5-plane input. Best: `models_filtered/rollout_network.pth`.
3. **RL** (`train_rl.py`): Self-play policy gradient. Opponent pool = SL network + previous RL checkpoints, weighted toward recent generations. 80 games/iteration, gradient clipping at 1.2, `lr=5e-6`, demeaned rewards.

### Key Utilities (`utils.py`)
Global constants: `DEFAULT_GAME_OVER_EMPTY_COUNT = 15` (game ends when 15 empty squares remain), and model paths `SL_NETWORK_PATH`, `ROLLOUT_NETWORK_PATH`, `RL_NETWORK_PATH`.

### Data Layout
```
data/9x9_filtered/   # ~1500 filtered professional games (SGF)
models_filtered/     # Best SL and rollout checkpoints
models/              # SL epoch checkpoints + RL iteration checkpoints
```
