import os
import copy
import random
import datetime

import torch
import torch.nn as nn

from policy_net import PolicyNetwork, ValueNetwork
from self_play import generate_self_play_game

from torch.utils.data import TensorDataset, DataLoader


def train_value(
    value_network, policy_network, device, num_iterations=2500, games_per_iteration=40
):
    """
    RL training loop.
    Each iteration:
    1. Generate games_per_iteration self-play games
    2. Compute policy gradient loss
    3. Update network
    """
    value_network = value_network.to(device)
    policy_network = policy_network.to(device)

    optimizer = torch.optim.Adam(value_network.parameters(), lr=0.001)

    # Generate self-play games
    epoch_length = 50
    for iteration in range(num_iterations):
        value_network.train()

        # Generate self-play games
        all_trajectories = []
        for _ in range(games_per_iteration):
            trajectory = generate_self_play_game(policy_network, device)
            all_trajectories.extend(trajectory)

        # Compute loss and update
        states = torch.stack([t[0] for t in all_trajectories])
        moves = torch.tensor(
            [t[1][0] * 9 + t[1][1] for t in all_trajectories], dtype=torch.long
        )
        rewards = torch.tensor([t[3] for t in all_trajectories], dtype=torch.float32)
        del all_trajectories

        # Replace the single forward pass with mini-batches
        batch_size = 512
        total_loss = 0
        optimizer.zero_grad()

        for i in range(0, len(states), batch_size):
            s = states[i : i + batch_size].to(device)
            m = moves[i : i + batch_size].to(device)
            r = rewards[i : i + batch_size].to(device)

            outputs = policy_network(s)
            log_probs = torch.log_softmax(outputs, dim=1)
            action_log_probs = log_probs[range(len(m)), m]
            loss = -(action_log_probs * r).mean()
            loss.backward()
            total_loss += loss.item()

            del s, m, r, outputs, log_probs, action_log_probs, loss
            torch.mps.empty_cache()

        optimizer.step()

        avg_reward = rewards.mean().item()

        print(
            f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Iteration {iteration}/{num_iterations}, Loss: {total_loss:.4f}, Avg Reward: {avg_reward:.4f}"
        )

        # Every N iterations, add current network to opponent pool
        if iteration % epoch_length == 0:
            os.makedirs("models", exist_ok=True)
            torch.save(
                policy_network.state_dict(),
                f"models/alphago_rl_epoch_{iteration // epoch_length}.pth",
            )
            print(
                f"[{datetime.datetime.now().strftime('%H:%M:%S')}] After checkpoint: MPS allocated: {torch.mps.current_allocated_memory() / 1e9:.2f} GB"
            )

        torch.mps.empty_cache()
        print(
            f"[{datetime.datetime.now().strftime('%H:%M:%S')}] After empty_cache: MPS allocated: {torch.mps.current_allocated_memory() / 1e9:.2f} GB"
        )


if __name__ == "__main__":
    device = torch.device("mps")
    value_network = ValueNetwork()
    policy_network = PolicyNetwork()
    policy_network.load_state_dict(
        torch.load("models/alphago_rl_epoch_49.pth", weights_only=False)
    )
    train_value(value_network, policy_network, device)
