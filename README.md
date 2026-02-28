# AlphaGo 9x9
AlphaGo 9x9 is a neural network that plays 9x9 games of Go. The architecture mimics the original
implementation of AlphaGo, but this version was trained on a MacBook Pro with an Apple M2 Max chip
over the course of a week.

The final version of AlphaGo 9x9 combines CNNs and MCTS (Monte Carlo Tree Search) in its final algorithm.

# Data
I pulled existing 9x9 games from a couple of links online that I found from the omega_go repository:
   `data/Kifu`  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;https://www.dropbox.com/s/5wsrhr4pbtuygd6/go9-large.tgz  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;https://www.dropbox.com/s/dsdftx4kux2nezj/gokif2.tgz  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;https://www.dropbox.com/s/r4sdo2j4da0zuko/gokif3.tgz  
    `data/Pro`  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;http://homepages.cwi.nl/~aeb/go/games/9x9.tgz  
    `data/Top50`  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;https://www.dropbox.com/s/abpzmqrw7gyvlzt/go9.tgz  

# Directory layout
`filter_9x9.py` `filter_9x9_winner_elo.py`: Filter the raw 9x9 online games using sgfmill and keep only the
games where the ELO is 1500+
`go_engine.py`: Stores the `GoGame` class that is used throughout the project
`features.py`: Takes a game state and converts it into a 3-dimensional tensor used for training
`policy_net.py`: Defines the network architecture for the various networks
`mcts.py`: Defines the core logic behind Monte Carlo Tree Search
`train_sl.py` `train_rollout.py` `train_value.py` `train_rl.py`: Various training scripts
`benchmark.py`: Compares various models against each other
`human_play.py`: Allows a human to play AlphaGo 9x9 via a terminal UI

# Training
The training process follows a step-by-step procedure starting from the filtered 9x9 online games.

## SL (supervised learning) network
We first convert a board state into a 37x9x9 tensor:
- 1 plane of the current player's stones
- 1 plane of the opponent's stones
- 1 plane for empty points
- 1 plane for the ko point
- 1 plane of ones (used as a bias plane)
- 8 planes for liberties of our own groups (one-hot vector)
- 8 planes for liberties of opponent groups (one-hot vector)
- 8 planes for turns since played for our stones (one-hot vector)
- 8 planes for turns since played for opponent stones (one-hot vector)
The intuition behind this is that convolutions are linear operations, and the number of liberties should
be considered more like a categorical rather than a numeric. For instance, a group in atari (1 liberty)
is much more susceptible than a group that has 2 liberties. These are encoded as binary planes rather
than a single plane with integer values.

From here, we build a 10-layer CNN (convolutional neural net) that outputs a 9x9 vector. We train the SL
network over the corpus of existing 1500+ ELO games to try and predict the next move. This network was
able to achieve 57% test accuracy.

## Rollout network
This network is used for fast rollout during MCTS. We build a lighter-weight 4-player CNN that takes in
just a 5x9x9 tensor (current player's stones, opponent's stones, empty points, ko point, and bias plane).
This faster rollout network was able to achieve 47% test accuracy.

## Value network
The original AlphaGo first trained the RL (reinforcement learning) network before the value network. However,
in my replication, I wasn't able to get my RL network to improve from merely self-playing games. Instead,
I had to build the value network first. I first generated ~500k self-playing games from the SL network. I then
initialized a similar network architecture as the SL network for the value network, with the exception of
having it output a single float (used to represent the probability of winning from a given state) instead of a 9x9
vector. I fed this network one randomly chosen position from each of my ~500k self-played games. This network
was able to achieve a MSE loss of around 0.70 on a test set.

## RL (reinforcement learning) network
In RL training, I used the value network above to determine whether a move increased or decreased the winning
probability. This method of TD (temporal differences) was much more stable than merely calling a move "good" or
"bad" based on the outcome of the game. After an initial training run, the RL network was able to achieve a 70%
win rate over the original SL network.

# Overall
AlphaGo 9x9 combines the previous networks in an overall system using MCTS. We initialize the network with either
the SL network or the RL network. We then run MCTS on each position, summing together an "exploitation" term (how
often does this node win) and an "exploration" term (we want to encourage exploration of unexplored moves). Each
leaf node is evaluated using both the value network and the rollout network simulation, which are then averaged.

# Utils
Stores network paths and meta-information about the AlphaGo 9x9 system.

# Models
- models_filtered contains the filtered set of models
- models_filtered/sl_network: copy of models/alphago_epoch_8.pth
- models_filtered/rl_network: highest version of models_filtered/rl_network_v{i}.pth
    - v1: used value_network_v1 and TD advantages, won 145/200 games against the SL network (2026-02-27)
- models_filtered/value_network: highest version of models_filtered/value_network_v{i}.pth
    - v1: value network after training on the first 500k games generated by the SL network (2026-02-26)
- models_filtered/rollout_network: copy of models/alphago_simple_rollout_epoch_45.pth

# data/self_play_games_sl
- dataset_{0-499}.h5 generated from the SL network using DEFAULT_GAME_OVER_EMPTY_COUNT=13
- dataset_{500-999}.h5 generated from the SL network using DEFAULT_GAME_OVER_EMPTY_COUNT=15

# data/self_play_games_rl
- dataset_{0-999}.h5 generated from the RL network using DEFAULT_GAME_OVER_EMPTY_COUNT=16

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
RL network training is finally working. I think it's because of the TD (temporal difference)
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