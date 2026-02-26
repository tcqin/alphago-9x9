import h5py
import numpy as np

with h5py.File("data/self_play_games_sl/dataset_0.h5", "r") as f:
    states = f["states"][:]
    outcomes = f["outcomes"][:]

print(f"States shape: {states.shape}")  # should be (100, 37, 9, 9)
print(f"Outcomes shape: {outcomes.shape}")  # should be (100,)
print(f"Outcomes: {outcomes[:10]}")  # should be +1/-1 values
print(f"Outcome balance: {outcomes.mean():.3f}")  # should be near 0
print(f"State min/max: {states.min():.3f}, {states.max():.3f}")  # should be 0/1
