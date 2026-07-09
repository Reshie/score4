from __future__ import annotations

import argparse
import random
from collections import deque
from pathlib import Path
from typing import Iterable

try:
    import torch
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover - exercised only without torch.
    raise SystemExit(
        "Training requires PyTorch. Install it with:\n"
        '  python -m pip install -e ".[train]"'
    ) from exc

from score4.model import AlphaZeroNet, NetworkEvaluator
from score4.self_play import SelfPlayConfig, TrainingExample, play_game


class ReplayBuffer:
    def __init__(self, capacity: int, rng: random.Random) -> None:
        self.examples: deque[TrainingExample] = deque(maxlen=capacity)
        self.rng = rng

    def __len__(self) -> int:
        return len(self.examples)

    def add_many(self, examples: Iterable[TrainingExample]) -> None:
        self.examples.extend(examples)

    def sample(self, batch_size: int) -> list[TrainingExample]:
        return self.rng.sample(list(self.examples), batch_size)

    def to_checkpoint(self) -> list[tuple[object, object, float]]:
        return [
            (example.observation, example.policy, example.value)
            for example in self.examples
        ]

    def load_checkpoint(self, payload: Iterable[tuple[object, object, float]]) -> None:
        self.examples.clear()
        for observation, policy, value in payload:
            self.examples.append(
                TrainingExample(
                    observation=observation,  # type: ignore[arg-type]
                    policy=policy,  # type: ignore[arg-type]
                    value=float(value),
                )
            )


def train_step(
    model: AlphaZeroNet,
    optimizer: torch.optim.Optimizer,
    batch: list[TrainingExample],
    device: torch.device,
) -> dict[str, float]:
    model.train()
    observations = torch.tensor(
        [example.observation for example in batch],
        dtype=torch.float32,
        device=device,
    )
    target_policy = torch.tensor(
        [example.policy for example in batch],
        dtype=torch.float32,
        device=device,
    )
    target_value = torch.tensor(
        [example.value for example in batch],
        dtype=torch.float32,
        device=device,
    )

    logits, value = model(observations)
    policy_loss = -(target_policy * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
    value_loss = F.mse_loss(value, target_value)
    loss = policy_loss + value_loss

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    optimizer.step()

    return {
        "loss": float(loss.detach().cpu().item()),
        "policy_loss": float(policy_loss.detach().cpu().item()),
        "value_loss": float(value_loss.detach().cpu().item()),
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    rng = random.Random(args.seed)
    device = _resolve_device(args.device)

    model = AlphaZeroNet(
        channels=args.channels,
        residual_blocks=args.res_blocks,
        value_hidden=args.value_hidden,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    replay = ReplayBuffer(args.replay_size, rng)

    start_iteration = 1
    if args.resume:
        checkpoint = _load_checkpoint(Path(args.resume), device)
        model.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "replay_buffer" in checkpoint:
            replay.load_checkpoint(checkpoint["replay_buffer"])
        start_iteration = int(checkpoint.get("iteration", 0)) + 1

    self_play_config = SelfPlayConfig(
        simulations=args.simulations,
        temperature_moves=args.temperature_moves,
        c_puct=args.c_puct,
        dirichlet_alpha=args.dirichlet_alpha,
        exploration_fraction=args.exploration_fraction,
    )
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"device={device}, replay={len(replay)}, start_iteration={start_iteration}")

    for iteration in range(start_iteration, args.iterations + 1):
        evaluator = NetworkEvaluator(model, device)
        generated: list[TrainingExample] = []
        first_player_scores: list[float] = []

        for game_index in range(1, args.games_per_iteration + 1):
            examples = play_game(evaluator, self_play_config, rng)
            generated.extend(examples)
            if examples:
                first_player_scores.append(examples[0].value)
            print(
                f"iteration={iteration} game={game_index}/"
                f"{args.games_per_iteration} moves={len(examples)}"
            )

        replay.add_many(generated)
        p1_score = (
            sum(first_player_scores) / len(first_player_scores)
            if first_player_scores
            else 0.0
        )
        print(
            f"iteration={iteration} generated={len(generated)} "
            f"buffer={len(replay)} p1_score={p1_score:+.3f}"
        )

        metrics = []
        if len(replay) >= args.batch_size:
            for _ in range(args.train_steps):
                metrics.append(
                    train_step(
                        model,
                        optimizer,
                        replay.sample(args.batch_size),
                        device,
                    )
                )
            print(_format_metrics(iteration, metrics))
        else:
            print(
                f"iteration={iteration} train=skipped "
                f"(buffer {len(replay)} < batch_size {args.batch_size})"
            )

        checkpoint_path = checkpoint_dir / f"checkpoint_{iteration:04d}.pt"
        torch.save(
            {
                "iteration": iteration,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "replay_buffer": replay.to_checkpoint(),
                "args": vars(args),
            },
            checkpoint_path,
        )
        print(f"saved={checkpoint_path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--games-per-iteration", type=int, default=16)
    parser.add_argument("--simulations", type=int, default=100)
    parser.add_argument("--train-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--replay-size", type=int, default=50_000)
    parser.add_argument("--temperature-moves", type=int, default=12)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--res-blocks", type=int, default=4)
    parser.add_argument("--value-hidden", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--dirichlet-alpha", type=float, default=0.3)
    parser.add_argument("--exploration-fraction", type=float, default=0.25)
    parser.add_argument("--checkpoint-dir", default="runs/score4")
    parser.add_argument("--resume")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _load_checkpoint(path: Path, device: torch.device) -> dict[str, object]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _format_metrics(iteration: int, metrics: list[dict[str, float]]) -> str:
    if not metrics:
        return f"iteration={iteration} train=no_steps"
    keys = ("loss", "policy_loss", "value_loss")
    averaged = {
        key: sum(metric[key] for metric in metrics) / len(metrics)
        for key in keys
    }
    return (
        f"iteration={iteration} train_steps={len(metrics)} "
        f"loss={averaged['loss']:.4f} "
        f"policy={averaged['policy_loss']:.4f} "
        f"value={averaged['value_loss']:.4f}"
    )


if __name__ == "__main__":
    main()
