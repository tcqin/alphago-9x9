import torch
import random
import datetime
import numpy as np

from go_engine import GoGame, BLACK, WHITE, EMPTY
from mcts import MCTS
from features import tensorfy_game
from policy_net import PolicyNetwork
from utils import DEFAULT_GAME_OVER_EMPTY_COUNT, SL_NETWORK_PATH


def generate_self_play_game(
    policy_network,
    device,
    opponent=None,
    policy_color=None,
    debug=False,
    mcts_policy=None,
    mcts_opponent=None,
):
    """
    Play one self-play game. Returns list of (state_tensor, move, move_idx, reward) tuples.
    opponent: if None, play against self, otherwise play against opponent network.
    """
    game = GoGame()
    trajectory = []  # list of (state_tensor, move, move_idx)

    # Move networks to device
    policy_network.eval()
    policy_network = policy_network.to(device)
    if opponent:
        opponent.eval()
        opponent = opponent.to(device)

    # Randomize who goes first
    if policy_color is None:
        rand_result = random.randint(0, 1)
        if opponent:
            if rand_result == 0:
                policy_color = BLACK
                network = policy_network
            else:
                policy_color = WHITE
                network = opponent
        else:
            network = policy_network
    else:
        network = policy_network if policy_color == BLACK else opponent

    move_idx = 0
    empty_count = int(np.sum(game.board == EMPTY))
    while not game.is_game_over() and empty_count > DEFAULT_GAME_OVER_EMPTY_COUNT:
        features = tensorfy_game(game)
        features_batched = features.unsqueeze(0)
        features_batched = features_batched.to(device)
        legal_moves = set(game.legal_moves())

        # Determine if MCTS is available for current network
        active_mcts = None
        if network is policy_network and mcts_policy is not None:
            active_mcts = mcts_policy
        elif opponent is not None and network is opponent and mcts_opponent is not None:
            active_mcts = mcts_opponent
        if active_mcts is not None:
            move = active_mcts.get_move(game)
            if move is not None and network is policy_network:
                r, c = move
                trajectory.append((features, (r, c), move_idx, game.current_player))
            if move is None:
                game.play(None, None)
            else:
                game.play(move[0], move[1])
        else:
            with torch.no_grad():
                outputs = network(features_batched)
                probs = torch.softmax(outputs[0], dim=0)
                probs_np = probs.cpu().numpy().astype(float)
                probs_np = probs_np / probs_np.sum()

                chosen = False
                while probs_np.sum() > 0:
                    selection = np.random.choice(game.size**2, p=probs_np)
                    r, c = selection // game.size, selection % game.size
                    if (r, c) in legal_moves and not game.is_true_eye(r, c):
                        chosen = True
                        if network is policy_network:
                            trajectory.append(
                                (features, (r, c), move_idx, game.current_player)
                            )
                        game.play(r, c)
                        break
                    else:
                        probs_np[selection] = 0  # Mask the bad move
                        total_prob = probs_np.sum()
                        if total_prob <= 0:
                            break
                        probs_np = probs_np / total_prob
                if not chosen:
                    game.play(None, None)

                del features_batched, outputs, probs

        move_idx += 1
        empty_count = int(np.sum(game.board == EMPTY))
        # Alternate networks
        if opponent:
            if game.current_player == policy_color:
                network = policy_network
            else:
                network = opponent

        if debug:
            print(
                f"[{datetime.datetime.now().strftime('%H:%M:%S')}] State after {move_idx} {'move' if move_idx == 1 else 'moves'}:\nBoard:\n{game}"
            )

    # Determine outcome
    b_score, w_score = game.score()
    black_won = b_score > w_score

    # Print statements for debugging
    if debug:
        print(
            f"policy_color={policy_color}, black_won={black_won}, b={b_score:.1f}, w={w_score:.1f}"
        )
        print(
            f"First trajectory reward: {1 if (trajectory[0][3] == BLACK) == black_won else -1}"
        )
        print(f"Trajectory colors: {[t[3] for t in trajectory[:5]]}")
    # Assign reward to each state
    return [
        (
            trajectory[i][0],  # state tensor
            trajectory[i][1],  # move
            trajectory[i][2],  # move_idx
            1 if (trajectory[i][3] == BLACK) == black_won else -1,  # reward assignment
        )
        for i in range(len(trajectory))
    ]


if __name__ == "__main__":
    device = torch.device("mps")
    policy_network = PolicyNetwork()
    policy_network.load_state_dict(torch.load(SL_NETWORK_PATH, weights_only=False))

    print("Generating self-play game...")
    trajectory = generate_self_play_game(policy_network, device, debug=True)
    print(f"Game length: {len(trajectory)} moves")
    print(
        f"First entry: state shape={trajectory[0][0].shape}, move={trajectory[0][1]}, reward={trajectory[0][3]}"
    )
