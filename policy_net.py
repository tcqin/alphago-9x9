import torch
import torch.nn as nn
import torch.nn.functional as F


class RolloutNetwork(nn.Module):

    def __init__(self, in_channels=5, filters=32, hidden_layers=2):
        super().__init__()
        # Convolution layers
        self.first = self._make_conv_block(in_channels, filters, 5)
        self.middle = nn.Sequential(
            *[self._make_conv_block(filters, filters, 3) for _ in range(hidden_layers)]
        )
        self.output = nn.Conv2d(filters, 1, kernel_size=1)

    def _make_conv_block(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convolution blocks
        x = self.first(x)
        x = self.middle(x)
        x = self.output(x)
        # Flatten
        x = x.view(x.size(0), -1)
        return x

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class PolicyNetwork(nn.Module):
    def __init__(self, in_channels=37, filters=128, hidden_layers=8):
        super().__init__()
        # Convolution layers
        self.first = self._make_conv_block(in_channels, filters, 5)
        self.middle = nn.Sequential(
            *[self._make_conv_block(filters, filters, 3) for _ in range(hidden_layers)]
        )
        self.output = nn.Conv2d(filters, 1, kernel_size=1)

    def _make_conv_block(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convolution blocks
        x = self.first(x)
        x = self.middle(x)
        x = self.output(x)
        # Flatten
        x = x.view(x.size(0), -1)
        return x

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ValueNetwork(nn.Module):
    def __init__(self, in_channels=37, filters=128, hidden_layers=8):
        super().__init__()
        # Convolution layers
        self.first = self._make_conv_block(in_channels, filters, 5)
        self.middle = nn.Sequential(
            *[self._make_conv_block(filters, filters, 3) for _ in range(hidden_layers)]
        )
        self.conv_output = nn.Conv2d(filters, 1, kernel_size=1)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(81, 1)
        self.tanh = nn.Tanh()

    def _make_conv_block(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convolution blocks
        x = self.first(x)
        x = self.middle(x)
        x = self.conv_output(x)
        x = x.view(x.size(0), -1)  # (batch, 81)
        x = self.dropout(x)
        x = self.fc(x)  # (batch, 1)
        x = self.tanh(x)  # (batch, 1), values in [-1, 1]
        return x

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
