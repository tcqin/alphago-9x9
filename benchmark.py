import sys
import argparse
import torch
import datetime
import numpy as np

from go_engine import GoGame, BLACK, WHITE, EMPTY
from features import tensorfy_game
from policy_net import PolicyNetwork, RolloutNetwork, ValueNetwork
from mcts import MCTS
from self_play import generate_self_play_game
from utils import (
    SL_NETWORK_PATH,
    RL_NETWORK_PATH,
    ROLLOUT_NETWORK_PATH,
    VALUE_NETWORK_PATH,
)

NUM_GAMES = 100
device = torch.device("mps")


def play_games(network1, network2, mcts1, mcts2, label=None, num_games=NUM_GAMES):
    """Play num_games between two configurations, return win count for config 1."""
    games_won = 0
    for i in range(num_games):
        policy_color = BLACK if i % 2 == 0 else WHITE
        trajectory = generate_self_play_game(
            network1,
            device,
            opponent=network2,
            policy_color=policy_color,
            mcts_policy=mcts1,
            mcts_opponent=mcts2,
        )
        reward = trajectory[0][3]
        if reward == 1:
            games_won += 1
        if (i + 1) % 20 == 0 and label:
            print(f"  [{label}] {i+1}/{num_games} games, winning {games_won}/{i+1}")
    return games_won, num_games


if __name__ == "__main__":
    # Load networks
    sl_network = PolicyNetwork().to(device)
    sl_network.load_state_dict(torch.load(SL_NETWORK_PATH, weights_only=False))
    sl_network.eval()

    rollout_network = RolloutNetwork().to(device)
    rollout_network.load_state_dict(
        torch.load(ROLLOUT_NETWORK_PATH, weights_only=False)
    )
    rollout_network.eval()

    value_network = ValueNetwork().to(device)
    value_network.load_state_dict(torch.load(VALUE_NETWORK_PATH, weights_only=False))
    value_network.eval()

    # Build MCTS configurations
    mcts_rollout_only = MCTS(
        sl_network,
        rollout_network,
        device,
        num_simulations=100,
        value_network=None,
        value_lambda=0.0,
    )
    mcts_value_only = MCTS(
        sl_network,
        rollout_network,
        device,
        num_simulations=100,
        value_network=value_network,
        value_lambda=1.0,
    )
    mcts_combined = MCTS(
        sl_network,
        rollout_network,
        device,
        num_simulations=100,
        value_network=value_network,
        value_lambda=0.5,
    )

    # Baseline SL network
    baseline_sl = PolicyNetwork().to(device)
    baseline_sl.load_state_dict(torch.load(SL_NETWORK_PATH, weights_only=False))
    baseline_sl.eval()

    configs = [
        ("SL only (Sungmin)", sl_network, baseline_sl, None, None),
        ("SL + rollout (Hyunwoo)", sl_network, baseline_sl, mcts_rollout_only, None),
        ("SL + value only (I-geon)", sl_network, baseline_sl, mcts_value_only, None),
        (
            "SL + rollout + value (Dex)",
            sl_network,
            baseline_sl,
            mcts_combined,
            None,
        ),
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=int, required=True, help="Config index")
    args = parser.parse_args()

    config = configs[args.config]
    label, policy_network, opponent_network, mcts_policy, mcts_opponent = config
    print(f"Benchmarking {label} vs bare SL network ({NUM_GAMES}) games each")
    won, played = play_games(
        policy_network, opponent_network, mcts_policy, mcts_opponent, label
    )
    print(f"Result: {won}/{played} ({100*won/played:.1f}%)")
