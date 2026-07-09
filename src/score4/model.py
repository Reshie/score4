from __future__ import annotations

from collections import OrderedDict
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
    def __init__(
        self,
        model: AlphaZeroNet,
        device: str | torch.device = "cpu",
        cache_size: int = 0,
    ) -> None:
        self.model = model
        self.device = torch.device(device)
        self.cache_size = max(0, cache_size)
        self.cache_hits = 0
        self.cache_misses = 0
        self._cache: OrderedDict[Score4State, tuple[list[float], float]] = (
            OrderedDict()
        )
        self.model.eval()

    def __call__(self, state: Score4State) -> tuple[list[float], float]:
        return self.evaluate_batch([state])[0]

    def evaluate_batch(
        self,
        states: Sequence[Score4State],
    ) -> list[tuple[list[float], float]]:
        results: list[tuple[list[float], float] | None] = [None] * len(states)
        missing_states: list[Score4State] = []
        missing_indexes: list[int] = []

        for index, state in enumerate(states):
            cached = self._cache.get(state)
            if cached is None:
                self.cache_misses += 1
                missing_states.append(state)
                missing_indexes.append(index)
                continue
            self.cache_hits += 1
            self._cache.move_to_end(state)
            policy, value = cached
            results[index] = (list(policy), value)

        if missing_states:
            with torch.inference_mode():
                logits, values = self.model(encode_batch(missing_states, self.device))
                policies = torch.softmax(logits, dim=1).detach().cpu().tolist()
                value_list = values.detach().cpu().tolist()

            for index, state, policy, value in zip(
                missing_indexes,
                missing_states,
                policies,
                value_list,
            ):
                result = (policy, float(value))
                results[index] = result
                self._remember(state, result)

        finalized: list[tuple[list[float], float]] = []
        for result in results:
            if result is None:
                raise RuntimeError("missing network evaluation result")
            policy, value = result
            finalized.append((list(policy), value))
        return finalized

    def _remember(
        self,
        state: Score4State,
        result: tuple[list[float], float],
    ) -> None:
        if self.cache_size:
            self._cache[state] = result
            self._cache.move_to_end(state)
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)

    def cache_info(self) -> dict[str, int]:
        return {
            "size": len(self._cache),
            "max_size": self.cache_size,
            "hits": self.cache_hits,
            "misses": self.cache_misses,
        }
