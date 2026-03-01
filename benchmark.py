import sys
import argparse
import torch
import datetime
import numpy as np

from go_engine import BLACK, WHITE
from policy_net import PolicyNetwork, RolloutNetwork, ValueNetwork
from mcts import MCTS
from self_play import generate_self_play_game
from utils import (
    SL_NETWORK_PATH,
    RL_NETWORK_PATH,
    ROLLOUT_NETWORK_PATH,
    VALUE_NETWORK_SL_PATH,
    VALUE_NETWORK_RL_PATH,
    VALUE_NETWORK_BOTH_PATH,
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
            print(
                f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Played {i+1}/{num_games} games, won {games_won}/{i+1}"
            )
    return games_won, num_games


if __name__ == "__main__":
    # SL network
    sl_network = PolicyNetwork().to(device)
    sl_network.load_state_dict(torch.load(SL_NETWORK_PATH, weights_only=False))
    sl_network.eval()

    # RL network
    rl_network = PolicyNetwork().to(device)
    rl_network.load_state_dict(torch.load(RL_NETWORK_PATH, weights_only=False))
    rl_network.eval()

    # Rollout network
    rollout_network = RolloutNetwork().to(device)
    rollout_network.load_state_dict(
        torch.load(ROLLOUT_NETWORK_PATH, weights_only=False)
    )
    rollout_network.eval()

    # Value networks
    value_network_sl = ValueNetwork().to(device)
    value_network_sl.load_state_dict(
        torch.load(VALUE_NETWORK_SL_PATH, weights_only=False)
    )
    value_network_sl.eval()
    value_network_rl = ValueNetwork().to(device)
    value_network_rl.load_state_dict(
        torch.load(VALUE_NETWORK_RL_PATH, weights_only=False)
    )
    value_network_rl.eval()
    value_network_both = ValueNetwork().to(device)
    value_network_both.load_state_dict(
        torch.load(VALUE_NETWORK_BOTH_PATH, weights_only=False)
    )
    value_network_both.eval()

    # Build MCTS configurations
    mcts_sl_sl_0p0 = MCTS(
        sl_network,
        rollout_network,
        device,
        num_simulations=100,
        value_network=value_network_sl,
        value_lambda=0.0,
    )
    mcts_sl_sl_0p2 = MCTS(
        sl_network,
        rollout_network,
        device,
        num_simulations=100,
        value_network=value_network_sl,
        value_lambda=0.2,
    )
    mcts_sl_sl_0p5 = MCTS(
        sl_network,
        rollout_network,
        device,
        num_simulations=100,
        value_network=value_network_sl,
        value_lambda=0.5,
    )
    mcts_sl_sl_0p8 = MCTS(
        sl_network,
        rollout_network,
        device,
        num_simulations=100,
        value_network=value_network_sl,
        value_lambda=0.8,
    )
    mcts_sl_sl_1p0 = MCTS(
        sl_network,
        rollout_network,
        device,
        num_simulations=100,
        value_network=value_network_sl,
        value_lambda=1.0,
    )
    # mcts_sl_rl = MCTS(
    #     sl_network,
    #     rollout_network,
    #     device,
    #     num_simulations=100,
    #     value_network=value_network_rl,
    #     value_lambda=1.0,
    # )
    # mcts_sl_both = MCTS(
    #     sl_network,
    #     rollout_network,
    #     device,
    #     num_simulations=100,
    #     value_network=value_network_both,
    #     value_lambda=1.0,
    # )
    # mcts_rl_sl = MCTS(
    #     rl_network,
    #     rollout_network,
    #     device,
    #     num_simulations=100,
    #     value_network=value_network_sl,
    #     value_lambda=1.0,
    # )
    # mcts_rl_rl = MCTS(
    #     rl_network,
    #     rollout_network,
    #     device,
    #     num_simulations=100,
    #     value_network=value_network_rl,
    #     value_lambda=1.0,
    # )
    # mcts_rl_both = MCTS(
    #     rl_network,
    #     rollout_network,
    #     device,
    #     num_simulations=100,
    #     value_network=value_network_both,
    #     value_lambda=1.0,
    # )

    # Baseline SL network
    baseline_sl = PolicyNetwork().to(device)
    baseline_sl.load_state_dict(torch.load(SL_NETWORK_PATH, weights_only=False))
    baseline_sl.eval()

    configs = [
        ("SL vs SL", sl_network, baseline_sl, None, None),
        ("RL vs SL", rl_network, baseline_sl, None, None),
        ("SL + value_sl (0.0) vs SL", sl_network, baseline_sl, mcts_sl_sl_0p0, None),
        ("SL + value_sl (0.2) vs SL", sl_network, baseline_sl, mcts_sl_sl_0p2, None),
        ("SL + value_sl (0.5) vs SL", sl_network, baseline_sl, mcts_sl_sl_0p5, None),
        ("SL + value_sl (0.8) vs SL", sl_network, baseline_sl, mcts_sl_sl_0p8, None),
        ("SL + value_sl (1.0) vs SL", sl_network, baseline_sl, mcts_sl_sl_1p0, None),
        # ("SL + value_rl vs SL", sl_network, baseline_sl, mcts_sl_rl, None),
        # ("SL + value_both vs SL", sl_network, baseline_sl, mcts_sl_both, None),
        # ("RL + value_sl vs SL", rl_network, baseline_sl, mcts_rl_sl, None),
        # ("RL + value_rl vs SL", rl_network, baseline_sl, mcts_rl_rl, None),
        # ("RL + value_both vs SL", rl_network, baseline_sl, mcts_rl_both, None),
        # (
        #     "SL + value_sl vs SL + value_both",
        #     sl_network,
        #     baseline_sl,
        #     mcts_sl_sl,
        #     mcts_sl_both,
        # ),
        # (
        #     "SL + value_sl vs RL + value_both",
        #     sl_network,
        #     rl_network,
        #     mcts_sl_sl,
        #     mcts_rl_both,
        # ),
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=int, required=True, help="Config index")
    args = parser.parse_args()

    config = configs[args.config]
    label, policy_network, opponent_network, mcts_policy, mcts_opponent = config
    print(f"Benchmarking {label} ({NUM_GAMES} games each)")
    won, played = play_games(
        policy_network, opponent_network, mcts_policy, mcts_opponent, label
    )
    print(f"Result: {won}/{played} ({100*won/played:.1f}%)")
