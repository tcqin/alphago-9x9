# Utils
Stores network paths and meta-information about the AlphaGo 9x9 system.

# Models
models_filtered contains the filtered set of models
models_filtered/sl_network: copy of models/alphago_epoch_8.pth
models_filtered/rollout_network: copy of models/alphago_simple_rollout_epoch_45.pth

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

# 2026-02-26
Changing DEFAULT_GAME_OVER_EMPTY_COUNT to 16 yielded an average game length of ~80
Starting to generate data for the value network (attempting for 500k games not including data augmentation)
Choosing a position from each game with probability (i + 1) ** 0.5 where i is the move number