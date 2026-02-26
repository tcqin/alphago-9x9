import numpy as np
from typing import Optional

BLACK = 1
WHITE = -1
EMPTY = 0

# At module level
_NEIGHBORS_CACHE = {}
_DIAGONALS_CACHE = {}


class GoGame:
    def __init__(self, size=9, komi=7):
        self.size = size
        self.komi = komi
        self.board = np.zeros((size, size), dtype=np.int8)
        self.current_player = BLACK
        self.ko_point = None  # (row, col) or None
        self.move_history = []  # list of (board, ko_point) for undo
        self.last_move = None
        self.passes = 0  # consecutive passes
        self.game_over = False

        # Compute neighbors and diagonals
        self._get_neighbors()
        self._get_diagonals()

    def _get_neighbors(self):
        # Precompute neighbors
        if self.size not in _NEIGHBORS_CACHE:
            _NEIGHBORS_CACHE[self.size] = {}
            for r in range(self.size):
                for c in range(self.size):
                    result = []
                    for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < self.size and 0 <= nc < self.size:
                            result.append((nr, nc))
                    _NEIGHBORS_CACHE[self.size][(r, c)] = result

    def _get_diagonals(self):
        # Precompute diagonals
        if self.size not in _DIAGONALS_CACHE:
            _DIAGONALS_CACHE[self.size] = {}
            for r in range(self.size):
                for c in range(self.size):
                    result = []
                    for dr, dc in [(1, 1), (-1, -1), (-1, 1), (1, -1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < self.size and 0 <= nc < self.size:
                            result.append((nr, nc))
                    _DIAGONALS_CACHE[self.size][(r, c)] = result

    def copy(self, copy_history=True):
        g = GoGame(self.size)
        g.komi = self.komi
        g.board = self.board.copy()
        g.current_player = self.current_player
        g.ko_point = self.ko_point
        g.move_history = (
            [(b.copy(), kp) for b, kp in self.move_history] if copy_history else []
        )
        g.passes = self.passes
        g.game_over = self.game_over
        return g

    def neighbors(self, r, c):
        """Return valid neighboring positions."""
        return _NEIGHBORS_CACHE[self.size][(r, c)]

    def diagonals(self, r, c):
        """Return valid diagonal positions."""
        return _DIAGONALS_CACHE[self.size][(r, c)]

    def get_group(self, r, c):
        """Return all stones in the group containing (r,c), and their liberties."""
        color = self.board[r, c]
        if color == EMPTY:
            return set(), set()
        group = set()
        liberties = set()
        to_explore = [(r, c)]
        while len(to_explore) > 0:
            stone = to_explore.pop()
            group.add(stone)
            for nr, nc in self.neighbors(stone[0], stone[1]):
                if self.board[nr, nc] == EMPTY:
                    liberties.add((nr, nc))
                elif self.board[nr, nc] == color:
                    if (nr, nc) not in group:
                        to_explore.append((nr, nc))
        return group, liberties

    def get_liberties(self, r, c):
        _, liberties = self.get_group(r, c)
        return liberties

    def remove_group(self, r, c):
        """Remove the group at (r,c) from the board. Returns captured stones."""
        group, _ = self.get_group(r, c)
        for sr, sc in group:
            self.board[sr, sc] = EMPTY
        return group

    def is_legal(self, r, c, color, thorough=True):
        """Check if playing color at (r,c) is legal."""
        # Lazy legality check
        if self.game_over or self.board[r, c] != EMPTY or self.ko_point == (r, c):
            return False
        # Fast suicide pre-check
        for nr, nc in self.neighbors(r, c):
            if self.board[nr, nc] == EMPTY:
                return True  # Has liberty, definitely legal
        if not thorough:
            return True
        # Temporarily play the point and see if it's legal
        self.board[r, c] = color
        try:
            # Check liberties
            _, liberties = self.get_group(r, c)
            if len(liberties) == 0:
                neighbors = self.neighbors(r, c)
                for nr, nc in neighbors:
                    if self.board[nr, nc] == -color:
                        ng, nl = self.get_group(nr, nc)
                        if len(nl) == 0:
                            # Captures opponent group
                            return True
                # Suicide is illegal
                return False
            else:
                return True
        finally:
            self.board[r, c] = EMPTY

    def is_true_eye(self, r, c):
        """Check if (r, c) is a true eye"""
        for nr, nc in self.neighbors(r, c):
            if self.board[nr, nc] != self.current_player:
                return False

        # Count diagonal neighbors
        diagonals = self.diagonals(r, c)
        bad_diagonals = sum(
            1 for dr, dc in diagonals if self.board[dr, dc] != self.current_player
        )
        if len(diagonals) <= 2:
            return bad_diagonals == 0
        else:
            return bad_diagonals <= 1

    def play(self, r, c, check_legal=True, record_history=True):
        """Play current_player's stone at (r, c).
        Returns number of captures.
        Pass is represented as play(None, None).
        """
        if self.game_over:
            return 0
        if r is None and c is None:
            # Player passed
            self.current_player = -self.current_player
            self.ko_point = None
            if record_history:
                self.move_history.append((self.board.copy(), None))
            self.passes += 1
            if self.passes == 2:
                self.game_over = True
            self.last_move = None
            return 0

        if r is None or c is None:
            raise Exception(f"Illegal move: ({r},{c}) cannot have exactly one None")

        if check_legal:
            assert self.is_legal(
                r, c, self.current_player
            ), f"Illegal move: ({r},{c}), player: {self.current_player}, board:\n{self}"

        # Play the move
        self.board[r, c] = self.current_player
        stones_captured = set()

        # Check if we captured opponent's stones
        for nr, nc in self.neighbors(r, c):
            if self.board[nr, nc] == -self.current_player:
                _, nl = self.get_group(nr, nc)
                if len(nl) == 0:
                    # Captured opponent's stone(s)
                    stones_captured |= self.remove_group(nr, nc)

        group, liberties = self.get_group(r, c)

        # Standard move
        if len(group) == 1 and len(liberties) == 1 and len(stones_captured) == 1:
            self.ko_point = next(iter(stones_captured))
        else:
            self.ko_point = None

        self.current_player = -self.current_player
        if record_history:
            self.move_history.append((self.board.copy(), self.ko_point))
        self.passes = 0

        self.last_move = (r, c)

        return len(stones_captured)

    def legal_moves(self):
        """Return list of legal (r,c) moves plus None for pass."""
        if self.game_over:
            return []
        moves = []
        for r in range(self.size):
            for c in range(self.size):
                if self.is_legal(r, c, self.current_player):
                    moves.append((r, c))
        moves.append(None)
        return moves

    def is_game_over(self):
        return self.game_over

    def score(self):
        """Returns a tuple of (black_score, white_score).
        This is an approximate though because it's nontrivial
        to determine whether a group is alive or dead."""
        mapping = {}  # Maps (r, c) --> BLACK or WHITE
        for r in range(self.size):
            for c in range(self.size):
                if (r, c) in mapping:
                    continue
                if self.board[r, c] == BLACK:
                    mapping[(r, c)] = BLACK
                elif self.board[r, c] == WHITE:
                    mapping[(r, c)] = WHITE
                else:
                    # Attempt flood fill algorithm
                    color = None
                    space = set()
                    to_explore = [(r, c)]
                    while len(to_explore) > 0:
                        intersection = to_explore.pop()
                        if intersection in space:
                            continue
                        space.add(intersection)
                        for nr, nc in self.neighbors(*intersection):
                            if self.board[nr, nc] == BLACK:
                                if color in [WHITE, EMPTY]:
                                    # EMPTY being used as sentinel for neutral territory
                                    color = EMPTY
                                else:
                                    color = BLACK
                            elif self.board[nr, nc] == WHITE:
                                if color in [BLACK, EMPTY]:
                                    # EMPTY being used as sentinel for neutral territory
                                    color = EMPTY
                                else:
                                    color = WHITE
                            else:
                                to_explore.append((nr, nc))
                    for intersection in space:
                        mapping[intersection] = color
        b_score = sum([1 for intersection in mapping if mapping[intersection] == BLACK])
        w_score = (
            sum([1 for intersection in mapping if mapping[intersection] == WHITE])
            + self.komi
        )

        return b_score, w_score

    def __str__(self):
        symbols = {BLACK: "X", WHITE: "O", EMPTY: "."}
        RESET = "\033[0m"
        HIGHLIGHT = "\033[1;92m"

        rows = []
        for r in range(self.size):
            row = []
            for c in range(self.size):
                symbol = symbols[self.board[r, c]]
                if self.last_move and (r, c) == self.last_move:
                    row.append(f"{HIGHLIGHT}{symbol}{RESET}")
                else:
                    row.append(symbol)
            rows.append(" ".join(row))
        return "\n".join(rows)
