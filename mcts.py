import math
import torch
import random
import datetime

import cProfile
import pstats

import numpy as np

from go_engine import GoGame, BLACK, WHITE, EMPTY
from features import tensorfy_game, tensorfy_game_rollout
from policy_net import PolicyNetwork, RolloutNetwork
from utils import DEFAULT_GAME_OVER_EMPTY_COUNT, SL_NETWORK_PATH, ROLLOUT_NETWORK_PATH


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
        exploitation = self.Q
        exploration = c * self.prior * math.sqrt(parent_N) / (1 + self.N)
        return exploitation + exploration

    def select_child(self):
        return max(self.children.values(), key=lambda n: n.uct_score(self.N))

    def is_leaf(self):
        return not self.is_expanded


class MCTS:
    def __init__(
        self, policy_network, rollout_network, device, num_simulations=200, c=1.4
    ):
        self.device = device
        self.policy_network = policy_network
        self.policy_network = self.policy_network.to(device)
        self.policy_network.eval()
        self.rollout_network = (
            rollout_network.cpu()
        )  # Keep a CPU copy of the rollout network
        self.rollout_network.eval()
        self.num_simulations = num_simulations
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
            moves_already_played_approx = int(np.sum(game.board != EMPTY)) + node.depth
            value = self._rollout(
                node.game, moves_already_played=moves_already_played_approx
            )
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

        # # Debug: print top 5 moves by visit count
        # top_moves = sorted(root.children.items(), key=lambda x: x[1].N, reverse=True)[
        #     :5
        # ]
        # for move, node in top_moves:
        #     eye = game.is_true_eye(move[0], move[1]) if move is not None else False
        #     print(f"Move {move}: N={node.N}, Q={node.Q:.3f}, eye={eye}")

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

    def _rollout(self, game, moves_already_played=0):
        """Play random moves to end of game, return outcome."""
        game = game.copy(copy_history=False)
        original_player = game.current_player
        max_moves = 80
        moves_played = moves_already_played
        while not game.is_game_over() and moves_played < max_moves:
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
            moves_played += 1

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
    policy_network.load_state_dict(torch.load(SL_NETWORK_PATH, weights_only=False))
    rollout_network = RolloutNetwork()
    rollout_network.load_state_dict(
        torch.load(ROLLOUT_NETWORK_PATH, weights_only=False)
    )
    device = torch.device("mps")
    mcts = MCTS(policy_network, rollout_network, device, num_simulations=200)
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
