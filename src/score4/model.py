from __future__ import annotations

from typing import Sequence

try:
    import torch
    from torch import Tensor, nn
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover - exercised only without torch.
    raise ImportError(
        "score4.model requires PyTorch. Install with: "
        'python -m pip install -e ".[train]"'
    ) from exc

from score4.game import ACTION_SIZE, BOARD_SIZE, Score4State


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(channels)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(channels)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class AlphaZeroNet(nn.Module):
    def __init__(
        self,
        channels: int = 64,
        residual_blocks: int = 4,
        value_hidden: int = 128,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(2, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(channels),
            nn.ReLU(),
        )
        self.trunk = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(residual_blocks)]
        )

        self.policy_head = nn.Sequential(
            nn.Conv3d(channels, 4, kernel_size=1, bias=False),
            nn.BatchNorm3d(4),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4 * BOARD_SIZE**3, ACTION_SIZE),
        )
        self.value_head = nn.Sequential(
            nn.Conv3d(channels, 2, kernel_size=1, bias=False),
            nn.BatchNorm3d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * BOARD_SIZE**3, value_hidden),
            nn.ReLU(),
            nn.Linear(value_hidden, 1),
            nn.Tanh(),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self.trunk(self.stem(x))
        return self.policy_head(x), self.value_head(x).squeeze(-1)


def encode_batch(states: Sequence[Score4State], device: str | torch.device) -> Tensor:
    observations = [state.observation() for state in states]
    return torch.tensor(observations, dtype=torch.float32, device=device)


class NetworkEvaluator:
    def __init__(self, model: AlphaZeroNet, device: str | torch.device = "cpu") -> None:
        self.model = model
        self.device = torch.device(device)

    def __call__(self, state: Score4State) -> tuple[list[float], float]:
        self.model.eval()
        with torch.no_grad():
            logits, value = self.model(encode_batch([state], self.device))
            policy = torch.softmax(logits[0], dim=0).detach().cpu().tolist()
            return policy, float(value[0].detach().cpu().item())
