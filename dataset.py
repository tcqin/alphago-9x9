import os
import re
import torch
import numpy as np
import random
import sgfmill
import sgfmill.sgf

from go_engine import GoGame
from features import tensorfy_game
from torchvision import transforms
from torch.utils.data import Dataset


class GoDataset(Dataset):
    def __init__(self, data_dir, tensorfy_fn):
        self.data_dir = data_dir
        self.tensorfy_fn = tensorfy_fn
        self.index = []  # list of (file_path, move_idx)

        files = [
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith(".sgf")
        ]

        print(f"Indexing {len(files)} files...")

        for i, path in enumerate(files):
            if i % 10000 == 0:
                print(f"  {i}/{len(files)} scanned")
            try:
                with open(path, "rb") as f:
                    game = sgfmill.sgf.Sgf_game.from_bytes(f.read())
                move_idx = 0
                for node in game.get_main_sequence():
                    color, move = node.get_move()
                    if color is None:
                        continue
                    if move:
                        self.index.append((path, move_idx))
                    move_idx += 1
            except Exception as e:
                continue

        print(f"Total positions: {len(self.index)}")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        path, target_move_idx = self.index[idx]

        with open(path, "rb") as f:
            game = sgfmill.sgf.Sgf_game.from_bytes(f.read())

        go_game = GoGame(size=game.get_size(), komi=game.get_komi())
        move_idx = 0
        for node in game.get_main_sequence():
            color, move = node.get_move()
            if color is None:
                continue
            if move_idx == target_move_idx:
                tensor = self.tensorfy_fn(go_game)
                label = torch.tensor(9 * move[0] + move[1], dtype=torch.long)
                return tensor, label
            move = move if move else (None, None)
            # print(f"Playing move_idx={move_idx}, color={color}, move={move}")
            go_game.play(move[0], move[1])
            # print(go_game)
            move_idx += 1


if __name__ == "__main__":
    data_dir = "data/9x9_filtered"
    d = GoDataset(data_dir, tensorfy_game)
    for i in range(5):
        index = random.randint(0, len(d) - 1)
        tensor, label = d[index]
        print(f"Tensor: {tensor}, Label: {label}")
