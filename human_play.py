import torch
from go_engine import GoGame, BLACK, WHITE
from mcts import MCTS
from policy_net import PolicyNetwork, RolloutNetwork, ValueNetwork
from utils import SL_NETWORK_PATH, ROLLOUT_NETWORK_PATH, VALUE_NETWORK_RL_PATH


def get_human_move(game):
    while True:
        try:
            move_str = input("Your move (row col), or 'pass': ").strip()
            if move_str.lower() == "pass":
                return None
            parts = move_str.split()
            r, c = int(parts[0]), int(parts[1])
            if game.is_legal(r, c, game.current_player):
                return (r, c)
            else:
                print("Illegal move, try again")
        except (ValueError, IndexError):
            print("Invalid input. Enter 'row col' e.g. '3 4', or 'pass'")


device = torch.device("mps")
policy_network = PolicyNetwork()
policy_network.load_state_dict(torch.load(SL_NETWORK_PATH, weights_only=False))
policy_network.eval()
value_network = ValueNetwork().to(device)
value_network.load_state_dict(torch.load(VALUE_NETWORK_RL_PATH, weights_only=False))
value_network.eval()
rollout_network = RolloutNetwork()
rollout_network.load_state_dict(torch.load(ROLLOUT_NETWORK_PATH, weights_only=False))
rollout_network.eval()
mcts = MCTS(
    policy_network,
    rollout_network,
    device,
    num_simulations=1000,
    value_network=value_network,
    value_lambda=1.0,
)

game = GoGame()
human_color = BLACK  # you play black

print("You are BLACK (X). Enter moves as 'row col' (0-indexed).")
print(game)

while not game.is_game_over():
    print()
    if game.current_player == human_color:
        move = get_human_move(game)
        if move is None:
            game.play(None, None)
            print("You passed.")
        else:
            game.play(move[0], move[1])
    else:
        print("Bot thinking...")
        move = mcts.get_move(game)
        if move is None:
            game.play(None, None)
            print("Bot passed.")
        else:
            game.play(move[0], move[1])
            print(f"Bot played: {move}")
    print(game)

b_score, w_score = game.score()
print(f"\nGame over! Black: {b_score}, White: {w_score}")
if b_score > w_score:
    print("Black wins!")
else:
    print("White wins!")
