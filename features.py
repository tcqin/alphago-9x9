import torch
import numpy as np
from go_engine import GoGame, BLACK, WHITE, EMPTY

_BIAS_PLANE = np.ones((9, 9), dtype=np.float32)
_KO_TEMPLATE = np.zeros((9, 9), dtype=np.float32)


def tensorfy_game_rollout(game: GoGame):
    """
    Takes a GoGame board state and converts it into a tensor
    that gets fed into the neural network:
    1 plane for our stones
    1 plane for opponent stones
    1 plane for empty points
    1 plane for ko point
    1 plane for ones (as a bias plane, presumably since corners > middle)"""
    # Basic planes
    our_stones = (game.board == game.current_player).astype(np.float32)
    opponent_stones = (game.board == -game.current_player).astype(np.float32)
    empty_stones = (game.board == EMPTY).astype(np.float32)

    # Ko point and bias planes
    ko_point_plane = _KO_TEMPLATE.copy()
    if game.ko_point:
        r, c = game.ko_point
        ko_point_plane[r, c] = 1

    return torch.from_numpy(
        np.stack(
            [our_stones, opponent_stones, empty_stones, ko_point_plane, _BIAS_PLANE]
        )
    )


def tensorfy_game(game: GoGame):
    """
    Takes a GoGame board state and converts it into a tensor
    that gets fed into the neural network:
    1 plane for our stones
    1 plane for opponent stones
    1 plane for empty points
    8 planes for liberties of our groups (one-hot)
    8 planes for liberties of opponent groups (one-hot)
    8 planes for turns since placed for our stones (one-hot)
    8 planes for turns since placed for opponent stones (one-hot)
    1 plane for ko point
    1 plane for ones (as a bias plane, presumably since corners > middle)"""
    # Basic planes
    our_stones = np.where(game.board == game.current_player, 1, 0)
    opponent_stones = np.where(game.board == -game.current_player, 1, 0)
    empty_stones = np.where(game.board == EMPTY, 1, 0)

    # Get liberty planes
    our_liberties = {}
    opponent_liberties = {}
    for r in range(game.size):
        for c in range(game.size):
            if (r, c) in our_liberties or (r, c) in opponent_liberties:
                continue
            group, libs = game.get_group(r, c)
            for stone in group:
                if game.current_player == game.board[stone[0], stone[1]]:
                    # Our stone
                    our_liberties[stone] = min(len(libs), 8)
                else:
                    # Opponent stone
                    opponent_liberties[stone] = min(len(libs), 8)

    our_liberty_array = np.zeros((game.size, game.size), dtype=np.int8)
    opponent_liberty_array = np.zeros((game.size, game.size), dtype=np.int8)
    for (r, c), value in our_liberties.items():
        our_liberty_array[r, c] = value
    for (r, c), value in opponent_liberties.items():
        opponent_liberty_array[r, c] = value

    our_liberties = np.stack(
        [np.where(our_liberty_array == i, 1, 0) for i in range(1, 9)], axis=0
    )
    opponent_liberties = np.stack(
        [np.where(opponent_liberty_array == i, 1, 0) for i in range(1, 9)], axis=0
    )

    # Get turns since planes
    turns_since_ours = {}
    turns_since_opponent = {}
    for turns_ago in range(1, min(9, len(game.move_history) + 1)):
        current = game.move_history[-turns_ago][0]
        previous = (
            game.move_history[-turns_ago - 1][0]
            if turns_ago < len(game.move_history)
            else np.zeros((game.size, game.size), dtype=np.int8)
        )
        for r in range(game.size):
            for c in range(game.size):
                if (r, c) in turns_since_ours or (r, c) in turns_since_opponent:
                    continue
                # Stone was added at this position
                if current[r, c] != EMPTY and previous[r, c] == EMPTY:
                    if current[r, c] == game.current_player:
                        turns_since_ours[(r, c)] = turns_ago
                    else:
                        turns_since_opponent[(r, c)] = turns_ago

    our_turns_since_array = np.zeros((game.size, game.size), dtype=np.int8)
    opponent_turns_since_array = np.zeros((game.size, game.size), dtype=np.int8)
    for (r, c), value in turns_since_ours.items():
        our_turns_since_array[r, c] = value
    for (r, c), value in turns_since_opponent.items():
        opponent_turns_since_array[r, c] = value

    our_turns_since = np.stack(
        [np.where(our_turns_since_array == i, 1, 0) for i in range(1, 9)], axis=0
    )
    opponent_turns_since = np.stack(
        [np.where(opponent_turns_since_array == i, 1, 0) for i in range(1, 9)], axis=0
    )

    # Ko point and bias planes
    ko_point_plane = np.zeros((game.size, game.size), dtype=np.int8)
    if game.ko_point:
        r, c = game.ko_point
        ko_point_plane[r, c] = 1
    bias_plane = np.ones((game.size, game.size), dtype=np.int8)

    tensor = np.concatenate(
        [
            np.stack([our_stones, opponent_stones, empty_stones]),
            our_liberties,
            opponent_liberties,
            our_turns_since,
            opponent_turns_since,
            np.stack([ko_point_plane, bias_plane]),
        ],
        axis=0,
    )
    return torch.tensor(tensor.astype(np.float32))
