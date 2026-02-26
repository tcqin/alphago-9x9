import torch

import numpy as np
from go_engine import GoGame, BLACK, WHITE, EMPTY
from features import tensorfy_game


if __name__ == "__main__":
    print(f"Torch version: {torch.__version__}")
    print(f"mps is available? {torch.backends.mps.is_available()}")

    g = GoGame()
    g.play(4, 4)  # black
    g.play(3, 3)  # white
    g.play(5, 5)  # black

    t = tensorfy_game(g)
    print(t.shape)
    print(t.dtype)

    print("Our stones:")
    print(t[0])

    print("Opponent stones:")
    print(t[1])

    print("Empty stones:")
    print(t[2])
