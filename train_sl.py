import os
import math
import random
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms

import numpy as np

from dataset import GoDataset
from policy_net import PolicyNetwork

data_dir = "data/9x9_filtered"
d = GoDataset(data_dir)

test_fraction = 0.20
test_length = math.floor(test_fraction * len(d))
test_indices = set(random.sample(list(range(len(d))), test_length))
train_indices = [i for i in range(len(d)) if i not in test_indices]
test_indices = list(test_indices)

train_dataset = Subset(d, train_indices)
test_dataset = Subset(d, test_indices)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# ------------------------------------------------------------------ #
# Build model                                                        #
# ------------------------------------------------------------------ #

print("Building model...")
device = torch.device("mps")
model = PolicyNetwork()
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=3
)

print(f"Total parameters: {model.get_num_parameters()}")
print(f"Training on device: {device}")
print(next(model.parameters()).device)

# Training loop
start_epoch = 0
num_epochs = 20
checkpoint_num = 0
checkpoint = f"models/alphago_epoch_{checkpoint_num}.pth"
if os.path.exists(checkpoint):
    model.load_state_dict(torch.load(checkpoint, weights_only=False))
    start_epoch = checkpoint_num + 1
    print(f"Resuming from epoch {start_epoch}")

print("Training...")
for epoch in range(start_epoch, num_epochs):
    model.train()
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 2000 == 0:
            print(
                f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Epoch {epoch}/{num_epochs - 1}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}"
            )

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), f"models/alphago_epoch_{epoch}.pth")

    # Evaluation
    model.eval()

    with torch.no_grad():

        # Train accuracy
        eval_train_loader = DataLoader(
            Subset(train_dataset, random.sample(range(len(train_dataset)), 100000)),
            batch_size=512,
        )
        eval_test_loader = DataLoader(
            Subset(test_dataset, random.sample(range(len(test_dataset)), 100000)),
            batch_size=512,
        )
        correct = 0
        total = 0
        for images, labels in eval_train_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print(f"Epoch {epoch}/{num_epochs-1} - Train Accuracy: {accuracy:.2f}%")

        # Test accuracy
        correct = 0
        total = 0
        for images, labels in eval_test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print(f"Epoch {epoch}/{num_epochs-1} - Test Accuracy: {accuracy:.2f}%")

    # Learning rate scheduler
    scheduler.step(accuracy)
