import math
import torch
import random
import datetime

import cProfile
import pstats

import numpy as np

from go_engine import GoGame, BLACK, WHITE, EMPTY
from features import tensorfy_game, tensorfy_game_rollout
from policy_net import PolicyNetwork, ValueNetwork, RolloutNetwork
from utils import (
    DEFAULT_GAME_OVER_EMPTY_COUNT,
    SL_NETWORK_PATH,
    RL_NETWORK_PATH,
    ROLLOUT_NETWORK_PATH,
    VALUE_NETWORK_PATH,
)


class MCTSNode:
    def __init__(self, game, parent=None, move=None, prior=0.0, depth=0):
        self.game = game
        self.parent = parent
        self.move = move
        self.prior = prior
        self.depth = depth
        self.N = 0
        self.W = 0.0
        self.children = {}
        self.is_expanded = False

    @property
    def Q(self):
        if self.N == 0:
            return 0.0
        return self.W / self.N

    def uct_score(self, parent_N, c=1.4):
        exploitation = -self.Q  # Negating because this is from perspective of parent
        exploration = c * self.prior * math.sqrt(parent_N) / (1 + self.N)
        return exploitation + exploration

    def select_child(self):
        return max(self.children.values(), key=lambda n: n.uct_score(self.N))

    def is_leaf(self):
        return not self.is_expanded


class MCTS:
    def __init__(
        self,
        policy_network,
        rollout_network,
        device,
        num_simulations=200,
        value_network=None,
        value_lambda=0.0,
        c=1.4,
    ):
        self.device = device
        self.policy_network = policy_network.to(device)
        self.policy_network.eval()
        self.rollout_network = (
            rollout_network.cpu()
        )  # Keep a CPU copy of the rollout network
        self.rollout_network.eval()
        self.num_simulations = num_simulations
        if value_network:
            self.value_network = value_network.to(device)
            self.value_network.eval()
        else:
            self.value_network = None
        self.value_lambda = value_lambda
        self.c = c  # exploration constant

    def get_move(self, game):
        """Run MCTS and return the best move."""
        empty_count = int(np.sum(game.board == EMPTY))
        if empty_count <= DEFAULT_GAME_OVER_EMPTY_COUNT:
            return None  # Pass
        root = MCTSNode(game.copy())
        self._expand(root)

        for _ in range(self.num_simulations):
            node = self._select(root)
            if not node.game.is_game_over():
                self._expand(node)
                node = self._select(node)  # select among new children
            value = self._evaluate(node.game)
            self._backpropagate(node, value)

        # Pick move with highest visit count
        best_move = max(
            root.children.keys(),
            key=lambda m: (
                root.children[m].N
                if (m is not None and not game.is_true_eye(m[0], m[1]))
                else root.children[m].N * 0.05
            ),
        )

        # Explicitly clear tree
        root.children.clear()
        del root

        return best_move

    def _get_game(self, node):
        """Reconstruct game state by replaying moves from root"""
        moves = []
        n = node
        while n.parent is not None:  # stop when n is root, not when n is None
            moves.append(n.move)
            n = n.parent
        # n is now root
        game = n.game.copy(copy_history=False)
        for move in reversed(moves):
            if move is None:
                game.play(None, None, record_history=False)
            else:
                game.play(move[0], move[1], record_history=False, check_legal=False)
        return game

    def _select(self, node):
        """Traverse tree selecting highest UCT child until leaf."""
        while not node.is_leaf():
            node = node.select_child()
        node.game = self._get_game(node)
        return node

    def _expand(self, node, num_children=20):
        """Add children to node using SL policy priors."""
        # get policy priors from SL network
        # add child node for each legal move
        if node.is_expanded:
            return
        features = tensorfy_game(node.game)
        features_batched = features.unsqueeze(0)
        features_batched = features_batched.to(self.device)
        with torch.no_grad():
            outputs = self.policy_network(features_batched)
            probs = torch.softmax(outputs[0], dim=0)
            probs_np = probs.cpu().numpy().astype(float)
        del features_batched, outputs, probs

        legal_moves_set = set(node.game.legal_moves())

        # Only keep top num_children moves by priors
        scored = []
        for i in range(len(probs_np)):
            move = (i // 9, i % 9)
            if move in legal_moves_set:
                scored.append((probs_np[i], move))
        scored.sort(reverse=True)
        top_moves = scored[:num_children]

        for prior, move in top_moves:
            node.children[move] = MCTSNode(
                None, parent=node, move=move, prior=prior, depth=node.depth + 1
            )
        if None in legal_moves_set:
            node.children[None] = MCTSNode(
                None, parent=node, move=None, prior=0.0, depth=node.depth + 1
            )
        node.is_expanded = True

    def _evaluate(self, game):
        """Evaluate position using value network + rollout network"""
        # Value network evaluation short-circuit
        if self.value_network and self.value_lambda == 1.0:
            features = tensorfy_game(game)
            features_batched = features.unsqueeze(0).to(self.device)
            with torch.no_grad():
                return self.value_network(features_batched).item()  # [-1, 1]

        # Rollout evaluation
        rollout_value = self._rollout(game)

        # Value network evaluation
        if self.value_network:
            features = tensorfy_game(game)
            features_batched = features.unsqueeze(0).to(self.device)
            with torch.no_grad():
                value = self.value_network(features_batched).item()  # [-1, 1]
            return self.value_lambda * value + (1 - self.value_lambda) * rollout_value
        else:
            return rollout_value

    def _rollout(self, game):
        """Play out game based on rollout network"""
        game = game.copy(copy_history=False)
        original_player = game.current_player
        empty_count = int(np.sum(game.board == EMPTY))
        while not game.is_game_over() and empty_count > DEFAULT_GAME_OVER_EMPTY_COUNT:
            features = tensorfy_game_rollout(game)
            features_batched = features.unsqueeze(0)
            with torch.no_grad():
                outputs = self.rollout_network(features_batched)
                probs = torch.softmax(outputs[0], dim=0)
                probs_np = probs.numpy().astype(float)
                probs_np = probs_np / probs_np.sum()
            del features, features_batched, outputs, probs

            # Try rollout network
            played = False
            while probs_np.sum() > 0:
                selection = np.random.choice(81, p=probs_np)
                r, c = selection // game.size, selection % game.size
                if not game.is_legal(
                    r, c, game.current_player, thorough=True
                ) or game.is_true_eye(r, c):
                    probs_np[selection] = 0  # Mask the bad move
                    total_prob = probs_np.sum()
                    if total_prob <= 0:
                        break
                    probs_np = probs_np / total_prob
                else:
                    game.play(r, c, check_legal=False, record_history=False)
                    played = True
                    break
            # Otherwise, just pass
            if not played:
                game.play(None, None, check_legal=False, record_history=False)
            empty_count = int(np.sum(game.board == EMPTY))

        b_score, w_score = game.score()
        if original_player == BLACK:
            return 1 if b_score > w_score else -1
        else:
            return 1 if w_score > b_score else -1

    def _backpropagate(self, node, value):
        """Walk up tree updating N and W."""
        while node is not None:
            node.W += value
            node.N += 1
            node = node.parent
            value = -value


if __name__ == "__main__":
    go_game = GoGame()
    policy_network = PolicyNetwork()
    policy_network.load_state_dict(torch.load(RL_NETWORK_PATH, weights_only=False))
    value_network = ValueNetwork()
    value_network.load_state_dict(torch.load(VALUE_NETWORK_PATH, weights_only=False))
    rollout_network = RolloutNetwork()
    rollout_network.load_state_dict(
        torch.load(ROLLOUT_NETWORK_PATH, weights_only=False)
    )
    device = torch.device("mps")
    mcts = MCTS(
        policy_network,
        rollout_network,
        device,
        num_simulations=100,
        value_network=None,
        value_lambda=None,
    )
    i = 0
    while not go_game.is_game_over():
        i += 1
        # profiler = cProfile.Profile()
        # profiler.enable()
        best_move = mcts.get_move(go_game)
        # profiler.disable()
        # stats = pstats.Stats(profiler)
        # stats.sort_stats("cumulative")
        # stats.print_stats(10)  # top 10 functions by cumulative time
        if best_move is None:
            go_game.play(None, None)
        else:
            go_game.play(best_move[0], best_move[1])
        print(
            f"[{datetime.datetime.now().strftime('%H:%M:%S')}] State after {i} {'move' if i == 1 else 'moves'}:\nMove: {best_move}\nBoard:\n{go_game}"
        )

    b_score, w_score = go_game.score()
    print(f"Game over! Black: {b_score}, White: {w_score}")
    print(go_game)
