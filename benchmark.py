import torch

from go_engine import BLACK, WHITE
from policy_net import PolicyNetwork
from self_play import generate_self_play_game
from utils import SL_NETWORK_PATH, RL_NETWORK_PATH


NUM_GAMES = 200


def compare_networks(n1, n2, device, iterations=NUM_GAMES):
    n1 = n1.to(device)
    n2 = n2.to(device)
    games_won = 0
    for i in range(iterations):
        policy_color = BLACK if i % 2 == 0 else WHITE
        trajectory = generate_self_play_game(
            n1, device, opponent=n2, policy_color=policy_color
        )
        reward_assignment = trajectory[0][3]
        if policy_color == BLACK and reward_assignment == 1:
            games_won += 1
        if policy_color == WHITE and reward_assignment == -1:
            games_won += 1
    return (games_won, iterations)


if __name__ == "__main__":
    device = torch.device("mps")
    sl_network = PolicyNetwork()
    sl_network.load_state_dict(torch.load(SL_NETWORK_PATH, weights_only=False))
    rl_network = PolicyNetwork()
    rl_network.load_state_dict(torch.load(RL_NETWORK_PATH, weights_only=False))

    games_won = 0

    print("Playing games...")
    for i in range(NUM_GAMES):
        policy_color = BLACK if i % 2 == 0 else WHITE
        trajectory = generate_self_play_game(
            rl_network, device, opponent=sl_network, policy_color=policy_color
        )
        reward_assignment = trajectory[0][3]
        if policy_color == BLACK and reward_assignment == 1:
            games_won += 1
        if policy_color == WHITE and reward_assignment == -1:
            games_won += 1
        if (i + 1) % 10 == 0:
            print(f"Played {i+1} games. Won {games_won}/{i+1} games.")
