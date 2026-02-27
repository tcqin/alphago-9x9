# Utils
Stores network paths and meta-information about the AlphaGo 9x9 system.

# Models
models_filtered contains the filtered set of models
models_filtered/sl_network: copy of models/alphago_epoch_8.pth
models_filtered/rl_network: highest version of models_filtered/rl_network_v{i}.pth
    - v1: copy of models/alphago_rl_epoch18.pth, which won 145/200 games against the SL network (2026-02-27)
models_filtered/value_network: copy of models/value_network_epoch_{epoch}.pth
models_filtered/rollout_network: copy of models/alphago_simple_rollout_epoch_45.pth

# data/self_play_games_sl
- dataset_{0-499}.h5 generated from the SL network using DEFAULT_GAME_OVER_EMPTY_COUNT=13
- dataset_{500-999}.h5 generated from the SL network using DEFAULT_GAME_OVER_EMPTY_COUNT=15

# data/self_play_games_rl
- dataset_{0-499}.h5 generated from the RL network using DEFAULT_GAME_OVER_EMPTY_COUNT=16

# Worklog
1. Built a SL (supervised learning) network based on 1500+ elo online games (37x9x9)
2. Built simple rollout network based on the same games with a smaller architecture (5x9x9)
3. Built a value network with a similar architecture as the SL network (37x9x9) that was trained to
predict the result of a game given a situation (37x9x9)
4. Built a RL (reinforcement learning) network that was seeded with the parameters from the original
SL network (37x9x9). We then pit this network against itself to maximize winning (as opposed to mimic'ing
humans' moves). We periodically check if the new network is beating the original SL network, and if so,
we checkpoint the weights of the new network. As training progresses, we pit the network against a pool
consisting of the original SL network as well as previous checkpoints. There were many issues and bugs
that we had to address during this stage:
    - Made sure we're only training on the RL network's perspective
    - Added DEFAULT_GAME_OVER_EMPTY_COUNT (around 15) to prevent games from cycling
    - Used a larger number of games per iteration (80) to denoise the training
    - Clipped the gradients to 1.0 to prevent snowballing effects
    - Played around with the learning rate, ultimately settling to 1e-4
    - We found a bug in mcts.py where select_child was grabbing self.Q from the child instead of the parent,
    which effectively interted the reward metric (since the perspective flips each turn)
    - We used the value network (trained above) to determine how good moves were based on TD (temporal
    differences) advantages in the RL network training loop to incentivize good moves. This was a game-changer.
    - After adding all the above, the RL network was able to beat the original SL network in 140/200 games

# 2026-02-27
RL network training is finally working! I think it's because of the TD (temporal difference)
advantages code that we incorporated. We did this by first training a value network on 500k
self-play games by the SL network. Then, in the RL training loop, we assessed how good each
move was by comparing successive value network outputs, as opposed to merely the game result.

We're generating 500k more self-play games by the SL network, but this time using a
DEFAULT_GAME_OVER_EMPTY_COUNT of 15 (slightly larger than the previous 13).

The benchmarking results so far:
SL only against bare SL: 51/100
SL + rollout against bare SL: 77/100
SL + value network against bare SL: 94/100
SL + value network + rollout against bare SL: 94/100

The plan going forward is to:
- continue to train the RL network
- use the RL network weights to generate self-play games to improve the value network

# 2026-02-26
Changing DEFAULT_GAME_OVER_EMPTY_COUNT to 16 yielded an average game length of ~80
Starting to generate data for the value network (attempting for 500k games not including data augmentation)
Choosing a position from each game with probability (i + 1) ** 0.5 where i is the move number