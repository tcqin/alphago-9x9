import os
import sys
import h5py
import torch
import random
import datetime
import numpy as np

from utils import SL_NETWORK_PATH
from policy_net import PolicyNetwork
from self_play import generate_self_play_game

N = 100
pid = int(sys.argv[1])

data_dir = "data/self_play_games_sl"
os.makedirs(data_dir, exist_ok=True)

device = torch.device("mps")
policy_network = PolicyNetwork()
policy_network.load_state_dict(torch.load(SL_NETWORK_PATH, weights_only=False))

for iteration in range(N):
    policy_network.eval()
    states = []
    outcomes_array = []

    batch_size = 1000
    for i in range(batch_size):
        trajectory = generate_self_play_game(policy_network, device)
        state, move, move_idx, reward = random.choice(trajectory)
        del trajectory
        states.append(state.numpy())
        outcomes_array.append(reward)

    fpath = f"{data_dir}/dataset_{pid * 100 + iteration}.h5"
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Writing {fpath}")
    with h5py.File(fpath, "w") as f:
        f.create_dataset("states", data=np.array(states))
        f.create_dataset("outcomes", data=np.array(outcomes_array))
