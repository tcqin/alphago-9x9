import os
import copy
import h5py
import random
import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from policy_net import ValueNetwork
from utils import VALUE_NETWORK_PATH

from torch.utils.data import TensorDataset, DataLoader

data_dir = "data"


def augment_batch(states, outcomes):
    """Apply a random symmetry to a batch of board positions"""
    k = random.randint(0, 3)  # Random rotation
    flip = random.random() < 0.5  # Random flip
    states = torch.rot90(states, k, dims=[2, 3])
    if flip:
        states = torch.flip(states, dims=[3])
    return states, outcomes


device = torch.device("mps")
value_network = ValueNetwork()
value_network = value_network.to(device)

state_tensors = []
outcome_tensors = []

total_files = 0
for root, dirs, files in os.walk(data_dir):
    for fname in files:
        if not fname.endswith(".h5"):
            continue
        with h5py.File(os.path.join(root, fname), "r") as f:
            states = f["states"][:]
            outcomes = f["outcomes"][:]
            state_tensors.append(torch.tensor(states, dtype=torch.float32))
            outcome_tensors.append(torch.tensor(outcomes, dtype=torch.float32))
        total_files += 1

print(f"Processed {total_files} .h5 files")

all_states = torch.cat(state_tensors, dim=0)
all_outcomes = torch.cat(outcome_tensors, dim=0)

print(f"State tensor shape: {all_states.shape}")
print(f"Outcome tensor shape: {all_outcomes.shape}")

dataset = TensorDataset(all_states, all_outcomes)
train_size = int(0.9 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_size, test_size]
)
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=512)

optimizer = torch.optim.Adam(value_network.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=3
)

# Current best value network
best_network = ValueNetwork().to(device)
best_network.load_state_dict(torch.load(VALUE_NETWORK_PATH, weights_only=False))
best_network.eval()

total_test_loss = 0
all_predictions = []
all_outcomes_test = []

with torch.no_grad():
    for states_batch, outcomes_batch in test_loader:
        states_batch = states_batch.to(device)
        outcomes_batch = outcomes_batch.to(device)
        predicted_outcomes = best_network(states_batch).squeeze()
        total_test_loss += F.mse_loss(predicted_outcomes, outcomes_batch).item()
        all_predictions.extend(predicted_outcomes.cpu())
        all_outcomes_test.extend(outcomes_batch.cpu())

best_test_loss = total_test_loss / len(test_loader)
all_predictions = np.array(all_predictions)
all_outcomes_test = np.array(all_outcomes_test)
correct_sign = ((all_predictions > 0) == (all_outcomes_test > 0)).mean()

print(f"Current best test loss: {best_test_loss:.4f}")
print(
    f"Predictions - min: {all_predictions.min():.3f}, max: {all_predictions.max():.3f}, mean: {all_predictions.mean():.3f}, std: {all_predictions.std():.3f}"
)
print(f"% predicted > 0.5: {(all_predictions > 0.5).mean():.3f}")
print(f"% predicted < -0.5: {(all_predictions < -0.5).mean():.3f}")
print(f"% predicted near 0 (|x| < 0.1): {(np.abs(all_predictions) < 0.1).mean():.3f}")
print(f"Sign accuracy: {correct_sign:.3f}")


# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    value_network.train()
    for batch_idx, (states_batch, outcomes_batch) in enumerate(train_loader):
        states_batch, outcomes_batch = augment_batch(states_batch, outcomes_batch)
        states_batch = states_batch.to(device)
        outcomes_batch = outcomes_batch.to(device)
        optimizer.zero_grad()
        predicted_outcomes = value_network(states_batch)
        loss = F.mse_loss(predicted_outcomes.squeeze(), outcomes_batch)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(
                f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Epoch {epoch}/{num_epochs - 1}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}"
            )
    os.makedirs("models", exist_ok=True)
    torch.save(value_network.state_dict(), f"models/value_network_epoch_{epoch}.pth")

    # Evaluation
    value_network.eval()

    with torch.no_grad():
        # Test accuracy
        total_test_loss = 0
        for states_batch, outcomes_batch in test_loader:
            states_batch = states_batch.to(device)
            outcomes_batch = outcomes_batch.to(device)
            predicted_outcomes = value_network(states_batch)
            total_test_loss += F.mse_loss(
                predicted_outcomes.squeeze(), outcomes_batch
            ).item()
        avg_test_loss = total_test_loss / len(test_loader)
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            torch.save(value_network.state_dict(), f"models_filtered/value_network.pth")
        print(
            f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Epoch {epoch}/{num_epochs - 1}, Test Loss: {avg_test_loss:.4f}"
        )

    scheduler.step(avg_test_loss)
