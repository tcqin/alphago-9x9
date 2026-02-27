# Utils
Stores network paths and meta-information about the AlphaGo 9x9 system.

# Models
models_filtered contains the filtered set of models
models_filtered/sl_network: copy of models/alphago_epoch_8.pth
models_filtered/rl_network: copy of models/alphago_rl_epoch_18.pth
models_filtered/value_network: copy of models/value_network_epoch_{epoch}.pth
models_filtered/rollout_network: copy of models/alphago_simple_rollout_epoch_45.pth

# data/self_play_games_sl
- dataset_{0-499}.h5 generated from the SL network using DEFAULT_GAME_OVER_EMPTY_COUNT=13
- dataset_{500-999}.h5 generated from the SL network using DEFAULT_GAME_OVER_EMPTY_COUNT=15

# data/self_play_games_rl
- dataset_{0-499}.h5 generated from the RL network using DEFAULT_GAME_OVER_EMPTY_COUNT=16

# Worklog
1. Built SL network based on 1500+ elo online games (37x9x9)
2. Built simple rollout network based on the same games with a smaller architecture (5x9x9)
3. Working on RL network
    - Added is_true_eye check to generate_self_play_game, although this is still imperfect
    - Made sure we were only training on the policy_network's perspective
    - Demeaned the rewards so that we're comparing against the baseline
    - Games used to last 200+ moves where the board was just cycling
    - Set the DEFAULT_GAME_OVER_EMPTY_COUNT to 16 to prevent games from going on for too long
    - Used a larger number of games per iteration (80) to denoise the training
    - Clipped the gradients at 1.2 to prevent snowballing effects
    - Reduced the learning rate

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