import os
import copy
import torch
import random
import datetime

from benchmark import play_games
from self_play import generate_self_play_game
from policy_net import PolicyNetwork, ValueNetwork
from utils import SL_NETWORK_PATH, RL_NETWORK_PATH, VALUE_NETWORK_PATH


def train_rl(
    policy_network, value_network, device, num_iterations=1000, games_per_iteration=80
):
    """
    RL training loop.
    Each iteration:
    1. Generate games_per_iteration self-play games
    2. Compute policy gradient loss
    3. Update network
    """
    policy_network = policy_network.to(device)
    optimizer = torch.optim.Adam(policy_network.parameters(), lr=0.0001)

    # Initialize opponent pool as the original SL network
    opponent_pool = [
        copy.deepcopy(
            {
                k: v.cpu()
                for k, v in torch.load(SL_NETWORK_PATH, weights_only=False).items()
            }
        )
    ]

    # start with SL network as opponent
    epoch_length = 40
    for iteration in range(num_iterations):
        policy_network.train()

        # Sample opponent from pool
        if random.random() < 0.35 or len(opponent_pool) == 1:
            opponent_state = opponent_pool[0]  # SL network
        else:
            weights = [(i + 1) for i in range(1, len(opponent_pool))]
            opponent_state = random.choices(opponent_pool[1:], weights=weights, k=1)[0]

        opponent = PolicyNetwork().to(device)
        opponent.load_state_dict(opponent_state)
        opponent.eval()

        # Generate self-play games
        all_trajectories = []
        for _ in range(games_per_iteration):
            trajectory = generate_self_play_game(
                policy_network,
                device,
                opponent,
            )
            all_trajectories.extend(trajectory)

        del opponent  # free MPS memory before training step
        torch.mps.empty_cache()
        policy_network.train()

        # Compute loss and update
        states = torch.stack([t[0] for t in all_trajectories])
        moves = torch.tensor(
            [t[1][0] * 9 + t[1][1] for t in all_trajectories], dtype=torch.long
        )
        move_idxs = [t[2] for t in all_trajectories]
        move_idxs_t = torch.tensor(move_idxs, dtype=torch.long)
        rewards = torch.tensor([t[3] for t in all_trajectories], dtype=torch.float32)

        avg_reward = rewards.mean().item()

        total_N = len(states)
        batch_size = 512

        # Compute V(s) for all states
        with torch.no_grad():
            value_preds = []
            for i in range(0, len(states), batch_size):
                s = states[i : i + batch_size].to(device)
                v = value_network(s).squeeze(-1).cpu()
                value_preds.append(v)
            values = torch.cat(value_preds)

        # TD advantages: V(s_{t+1}) - V(s_t) within a game, R - V(s_t) at game end.
        # Consecutive trajectory entries are both from the policy player's perspective
        # so no sign flip is needed. A game boundary occurs when move_idx doesn't advance
        # This is the important step that made the RL network train successfully
        is_game_end = torch.ones(total_N, dtype=torch.bool)
        is_game_end[:-1] = move_idxs_t[1:] < move_idxs_t[:-1]
        next_values = torch.cat([values[1:], torch.zeros(1)])
        advantages = torch.where(is_game_end, rewards - values, next_values - values)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Calculate average game length
        game_lengths = [
            move_idxs[i]
            for i in range(1, len(move_idxs) - 1)
            if move_idxs[i] > move_idxs[i - 1] and move_idxs[i] > move_idxs[i + 1]
        ] + [move_idxs[-1]]
        avg_game_length = sum(game_lengths) / len(game_lengths) + 1

        del all_trajectories

        # Replace the single forward pass with mini-batches
        total_loss = 0
        optimizer.zero_grad()

        for i in range(0, len(states), batch_size):
            s = states[i : i + batch_size].to(device)
            m = moves[i : i + batch_size].to(device)
            a = advantages[i : i + batch_size].to(device)

            outputs = policy_network(s)
            probs = torch.softmax(outputs, dim=1)
            log_probs = torch.log_softmax(outputs, dim=1)
            entropy = -(probs * log_probs).sum(dim=1).mean()
            action_log_probs = log_probs[range(len(m)), m]
            loss = -(action_log_probs * a).sum() / total_N - 0.001 * entropy
            loss.backward()
            total_loss += loss.item()

            del s, m, a, outputs, probs, log_probs, entropy, action_log_probs, loss
            torch.mps.empty_cache()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy_network.parameters(), max_norm=1.0
        )
        optimizer.step()

        print(
            f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Iteration {iteration}/{num_iterations}, Gradient: {grad_norm:.4f}, Loss: {total_loss:.4f}, Avg Reward: {avg_reward:.4f}, Avg Game Length: {avg_game_length:.2f}"
        )

        # Every N iterations, add current network to opponent pool if it beats the SL network
        if iteration % epoch_length == epoch_length - 1:
            os.makedirs("models", exist_ok=True)
            model_fpath = f"models/alphago_rl_epoch_{iteration // epoch_length}.pth"
            print(
                f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Saving new model to {model_fpath}"
            )
            torch.save(policy_network.state_dict(), model_fpath)
            # Compare with original SL network
            sl_network = PolicyNetwork().to(device)
            sl_network.load_state_dict(torch.load(SL_NETWORK_PATH, weights_only=False))
            sl_network.eval()
            policy_network.eval()
            games_won, games_played = play_games(
                policy_network, sl_network, None, None, num_games=200
            )
            del sl_network
            torch.mps.empty_cache()
            print(
                f"[{datetime.datetime.now().strftime('%H:%M:%S')}] New model won {games_won}/{games_played} games against the original SL network"
            )
            if games_won / games_played >= 0.5353:  # One stdev above 100/200 games
                state_dict_cpu = {
                    k: v.cpu() for k, v in policy_network.state_dict().items()
                }
                opponent_pool.append(copy.deepcopy(state_dict_cpu))
            print(
                f"[{datetime.datetime.now().strftime('%H:%M:%S')}] After checkpoint: MPS allocated: {torch.mps.current_allocated_memory() / 1e9:.2f} GB"
            )
        else:
            torch.mps.empty_cache()


if __name__ == "__main__":
    device = torch.device("mps")
    policy_network = PolicyNetwork().to(device)
    policy_network.load_state_dict(torch.load(RL_NETWORK_PATH, weights_only=False))
    value_network = ValueNetwork().to(device)
    value_network.load_state_dict(torch.load(VALUE_NETWORK_PATH, weights_only=False))
    value_network.eval()
    train_rl(policy_network, value_network, device)
