from __future__ import annotations

import argparse
import csv
import html
import random
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

try:
    import torch
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover - exercised only without torch.
    raise SystemExit(
        "Training requires PyTorch. Install it with:\n"
        '  python -m pip install -e ".[train]"'
    ) from exc

from score4.model import AlphaZeroNet, NetworkEvaluator
from score4.self_play import (
    SelfPlayConfig,
    TrainingExample,
    native_self_play_available,
    play_game,
    play_games_batched,
)


@dataclass(frozen=True)
class TrainingRecord:
    iteration: int
    generated_examples: int
    buffer_size: int
    p1_score: float
    train_steps: int
    loss: float | None
    policy_loss: float | None
    value_loss: float | None
    elapsed_seconds: float


class ProgressReporter:
    def __init__(self, label: str, total: int, width: int = 28) -> None:
        self.label = label
        self.total = max(0, total)
        self.width = width
        self.started_at = time.perf_counter()
        self.last_draw_at = 0.0
        self.last_log_at = 0.0
        self.last_line_length = 0
        self.interactive = sys.stdout.isatty()

    def update(self, current: int, suffix: str = "", force: bool = False) -> None:
        current = max(0, min(current, self.total))
        now = time.perf_counter()
        if not force and current < self.total and now - self.last_draw_at < 0.25:
            return

        line = self._format_line(current, suffix, now)
        if self.interactive:
            padding = " " * max(0, self.last_line_length - len(line))
            print(f"\r{line}{padding}", end="", flush=True)
            self.last_line_length = len(line)
        elif self._should_log(current, now, force):
            print(line, flush=True)
            self.last_log_at = now

        self.last_draw_at = now

    def finish(self, suffix: str = "") -> None:
        self.update(self.total, suffix=suffix, force=True)
        if self.interactive:
            print(flush=True)

    def _format_line(self, current: int, suffix: str, now: float) -> str:
        progress = 1.0 if self.total == 0 else current / self.total
        filled = int(round(self.width * progress))
        bar = "#" * filled + "-" * (self.width - filled)
        elapsed = now - self.started_at
        parts = [
            f"{self.label} [{bar}] {current}/{self.total}",
            f"{progress * 100:5.1f}%",
            f"elapsed={_format_duration(elapsed)}",
        ]
        if 0 < current < self.total:
            eta = elapsed * (self.total - current) / current
            parts.append(f"eta={_format_duration(eta)}")
        if suffix:
            parts.append(suffix)
        return " ".join(parts)

    def _should_log(self, current: int, now: float, force: bool) -> bool:
        return (
            force
            or current in (0, 1, self.total)
            or now - self.last_log_at >= 15.0
        )


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
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
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
        reuse_tree=args.reuse_tree,
        mcts_threads=args.mcts_threads,
    )
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = checkpoint_dir / "metrics.csv"
    chart_path = checkpoint_dir / "training_progress.svg"
    history = (
        [
            record
            for record in _load_training_history(metrics_path)
            if record.iteration < start_iteration
        ]
        if args.resume
        else []
    )

    print(
        f"device={device}, replay={len(replay)}, "
        f"start_iteration={start_iteration}, checkpoints={checkpoint_dir}, "
        f"self_play_backend={'cpp' if native_self_play_available() else 'python'}"
    )
    run_started_at = time.perf_counter()

    for iteration in range(start_iteration, args.iterations + 1):
        iteration_started_at = time.perf_counter()
        remaining = args.iterations - iteration
        print(
            f"\niteration {iteration}/{args.iterations} "
            f"(remaining={remaining})"
        )
        evaluator = NetworkEvaluator(
            model,
            device,
            cache_size=args.eval_cache_size,
        )
        generated: list[TrainingExample] = []
        first_player_scores: list[float] = []
        self_play_progress = ProgressReporter(
            "self-play",
            args.games_per_iteration,
        )
        self_play_progress.update(
            0,
            suffix=f"simulations={args.simulations}",
            force=True,
        )

        if args.self_play_batch_size <= 1:
            for game_index in range(1, args.games_per_iteration + 1):
                examples = play_game(evaluator, self_play_config, rng)
                generated.extend(examples)
                if examples:
                    first_player_scores.append(examples[0].value)
                self_play_progress.update(
                    game_index,
                    suffix=f"last_moves={len(examples)} generated={len(generated)}",
                )
        else:
            completed_games = 0
            while completed_games < args.games_per_iteration:
                batch_games = min(
                    args.self_play_batch_size,
                    args.games_per_iteration - completed_games,
                )
                batch_examples = play_games_batched(
                    evaluator,
                    self_play_config,
                    batch_games,
                    rng,
                )
                if not batch_examples and batch_games:
                    raise RuntimeError("batched self-play did not finish any games")
                for examples in batch_examples:
                    generated.extend(examples)
                    if examples:
                        first_player_scores.append(examples[0].value)
                completed_games += len(batch_examples)
                average_moves = (
                    sum(len(examples) for examples in batch_examples)
                    / len(batch_examples)
                    if batch_examples
                    else 0.0
                )
                self_play_progress.update(
                    completed_games,
                    suffix=(
                        f"batch={len(batch_examples)} "
                        f"avg_moves={average_moves:.1f} "
                        f"generated={len(generated)}"
                    ),
                )

        replay.add_many(generated)
        p1_score = (
            sum(first_player_scores) / len(first_player_scores)
            if first_player_scores
            else 0.0
        )
        self_play_progress.finish(
            suffix=(
                f"generated={len(generated)} buffer={len(replay)} "
                f"p1_score={p1_score:+.3f} "
                f"{_format_cache_suffix(evaluator.cache_info())}"
            )
        )
        print(
            f"iteration={iteration} generated={len(generated)} "
            f"buffer={len(replay)} p1_score={p1_score:+.3f} "
            f"{_format_cache_suffix(evaluator.cache_info())}"
        )

        metrics = []
        averaged_metrics: dict[str, float] | None = None
        if len(replay) >= args.batch_size:
            train_progress = ProgressReporter("train", args.train_steps)
            if args.train_steps > 0:
                train_progress.update(
                    0,
                    suffix=f"batch_size={args.batch_size}",
                    force=True,
                )
            for step in range(1, args.train_steps + 1):
                metrics.append(
                    train_step(
                        model,
                        optimizer,
                        replay.sample(args.batch_size),
                        device,
                    )
                )
                averaged_metrics = _average_metrics(metrics)
                train_progress.update(
                    step,
                    suffix=_format_loss_suffix(averaged_metrics),
                )
            if args.train_steps > 0:
                if train_progress.interactive:
                    train_progress.finish(
                        suffix=_format_loss_suffix(averaged_metrics),
                    )
            print(_format_metrics(iteration, metrics))
        else:
            print(
                f"iteration={iteration} train=skipped "
                f"(buffer {len(replay)} < batch_size {args.batch_size})"
            )

        averaged_metrics = _average_metrics(metrics)
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
        record = TrainingRecord(
            iteration=iteration,
            generated_examples=len(generated),
            buffer_size=len(replay),
            p1_score=p1_score,
            train_steps=len(metrics),
            loss=averaged_metrics["loss"] if averaged_metrics else None,
            policy_loss=(
                averaged_metrics["policy_loss"] if averaged_metrics else None
            ),
            value_loss=(
                averaged_metrics["value_loss"] if averaged_metrics else None
            ),
            elapsed_seconds=time.perf_counter() - iteration_started_at,
        )
        history.append(record)
        _write_training_history(history, metrics_path)
        print(
            f"iteration={iteration} elapsed="
            f"{_format_duration(record.elapsed_seconds)}"
        )

    print(
        f"\nfinished elapsed={_format_duration(time.perf_counter() - run_started_at)}"
    )
    print(f"metrics={metrics_path}")
    print(f"chart={chart_path}")


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
    parser.add_argument("--eval-cache-size", type=int, default=100_000)
    parser.add_argument("--self-play-batch-size", type=int, default=32)
    parser.add_argument("--reuse-tree", action="store_true")
    parser.add_argument(
        "--mcts-threads",
        type=int,
        default=0,
        help=(
            "CPU threads used by the native batched MCTS backend; "
            "0 selects all available cores"
        ),
    )
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
    averaged = _average_metrics(metrics)
    if averaged is None:
        return f"iteration={iteration} train=no_steps"
    return (
        f"iteration={iteration} train_steps={len(metrics)} "
        f"loss={averaged['loss']:.4f} "
        f"policy={averaged['policy_loss']:.4f} "
        f"value={averaged['value_loss']:.4f}"
    )


def _average_metrics(metrics: list[dict[str, float]]) -> dict[str, float] | None:
    if not metrics:
        return None
    keys = ("loss", "policy_loss", "value_loss")
    return {
        key: sum(metric[key] for metric in metrics) / len(metrics)
        for key in keys
    }


def _format_loss_suffix(metrics: dict[str, float] | None) -> str:
    if metrics is None:
        return "loss=n/a"
    return (
        f"loss={metrics['loss']:.4f} "
        f"policy={metrics['policy_loss']:.4f} "
        f"value={metrics['value_loss']:.4f}"
    )


def _format_cache_suffix(cache_info: dict[str, int]) -> str:
    max_size = cache_info["max_size"]
    if max_size <= 0:
        return "eval_cache=off"
    hits = cache_info["hits"]
    misses = cache_info["misses"]
    total = hits + misses
    hit_rate = hits / total if total else 0.0
    return (
        f"eval_cache={cache_info['size']}/{max_size} "
        f"hit_rate={hit_rate:.1%}"
    )


def _format_duration(seconds: float) -> str:
    seconds = max(0.0, seconds)
    if seconds < 10:
        return f"{seconds:.1f}s"
    total_seconds = int(seconds + 0.5)
    minutes, remaining_seconds = divmod(total_seconds, 60)
    hours, remaining_minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{remaining_minutes:02d}m{remaining_seconds:02d}s"
    if remaining_minutes:
        return f"{remaining_minutes}m{remaining_seconds:02d}s"
    return f"{remaining_seconds}s"


def _load_training_history(path: Path) -> list[TrainingRecord]:
    if not path.exists():
        return []

    records: list[TrainingRecord] = []
    with path.open("r", newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            try:
                records.append(
                    TrainingRecord(
                        iteration=int(row["iteration"]),
                        generated_examples=int(row["generated_examples"]),
                        buffer_size=int(row["buffer_size"]),
                        p1_score=float(row["p1_score"]),
                        train_steps=int(row["train_steps"]),
                        loss=_optional_float(row["loss"]),
                        policy_loss=_optional_float(row["policy_loss"]),
                        value_loss=_optional_float(row["value_loss"]),
                        elapsed_seconds=float(row["elapsed_seconds"]),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
    return records


def _write_training_history(records: Sequence[TrainingRecord], path: Path) -> None:
    fieldnames = (
        "iteration",
        "generated_examples",
        "buffer_size",
        "p1_score",
        "train_steps",
        "loss",
        "policy_loss",
        "value_loss",
        "elapsed_seconds",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "iteration": record.iteration,
                    "generated_examples": record.generated_examples,
                    "buffer_size": record.buffer_size,
                    "p1_score": f"{record.p1_score:.8f}",
                    "train_steps": record.train_steps,
                    "loss": _optional_float_text(record.loss),
                    "policy_loss": _optional_float_text(record.policy_loss),
                    "value_loss": _optional_float_text(record.value_loss),
                    "elapsed_seconds": f"{record.elapsed_seconds:.3f}",
                }
            )


def _write_training_chart(records: Sequence[TrainingRecord], path: Path) -> None:
    width = 920
    height = 560
    svg: list[str] = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        ),
        "<style>",
        "text { font-family: Segoe UI, Arial, sans-serif; fill: #24292f; }",
        ".title { font-size: 22px; font-weight: 700; }",
        ".label { font-size: 12px; fill: #57606a; }",
        ".legend { font-size: 12px; }",
        ".grid { stroke: #d8dee4; stroke-width: 1; }",
        ".axis { stroke: #8c959f; stroke-width: 1.2; }",
        "</style>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="48" y="38" class="title">Training progress</text>',
    ]
    if records:
        svg.append(
            f'<text x="48" y="58" class="label">'
            f'{len(records)} iterations recorded</text>'
        )

    _append_loss_plot(svg, records, left=70, top=88, width=800, height=185)
    _append_score_plot(svg, records, left=70, top=350, width=800, height=140)

    svg.append("</svg>")
    path.write_text("\n".join(svg), encoding="utf-8")


def _append_loss_plot(
    svg: list[str],
    records: Sequence[TrainingRecord],
    left: int,
    top: int,
    width: int,
    height: int,
) -> None:
    series = [
        (
            "loss",
            "#cf222e",
            [(record.iteration, record.loss) for record in records],
        ),
        (
            "policy_loss",
            "#0969da",
            [(record.iteration, record.policy_loss) for record in records],
        ),
        (
            "value_loss",
            "#1a7f37",
            [(record.iteration, record.value_loss) for record in records],
        ),
    ]
    values = [
        value
        for _, _, points in series
        for _, value in points
        if value is not None
    ]
    _append_plot_frame(svg, left, top, width, height, "Loss")
    if not values:
        _append_empty_plot_message(svg, left, top, width, height, "No loss yet")
        return

    y_min, y_max = _value_range(values, include_zero=True)
    x_min, x_max = _iteration_range(records)
    _append_y_grid(svg, left, top, width, height, y_min, y_max)
    for index, (label, color, points) in enumerate(series):
        _append_polyline(
            svg,
            [(iteration, value) for iteration, value in points if value is not None],
            left,
            top,
            width,
            height,
            x_min,
            x_max,
            y_min,
            y_max,
            color,
        )
        _append_legend_item(svg, label, color, left, top, index)


def _append_score_plot(
    svg: list[str],
    records: Sequence[TrainingRecord],
    left: int,
    top: int,
    width: int,
    height: int,
) -> None:
    _append_plot_frame(svg, left, top, width, height, "First-player score")
    if not records:
        _append_empty_plot_message(svg, left, top, width, height, "No games yet")
        return

    x_min, x_max = _iteration_range(records)
    y_min, y_max = -1.0, 1.0
    _append_y_grid(svg, left, top, width, height, y_min, y_max)
    _append_polyline(
        svg,
        [(record.iteration, record.p1_score) for record in records],
        left,
        top,
        width,
        height,
        x_min,
        x_max,
        y_min,
        y_max,
        "#8250df",
    )
    _append_legend_item(svg, "p1_score", "#8250df", left, top, 0)


def _append_plot_frame(
    svg: list[str],
    left: int,
    top: int,
    width: int,
    height: int,
    title: str,
) -> None:
    title = html.escape(title)
    svg.append(f'<text x="{left}" y="{top - 18}" font-size="16" font-weight="700">{title}</text>')
    svg.append(
        f'<rect x="{left}" y="{top}" width="{width}" height="{height}" '
        'fill="#ffffff" stroke="#d0d7de" stroke-width="1"/>'
    )
    svg.append(
        f'<line x1="{left}" y1="{top + height}" '
        f'x2="{left + width}" y2="{top + height}" class="axis"/>'
    )
    svg.append(
        f'<line x1="{left}" y1="{top}" '
        f'x2="{left}" y2="{top + height}" class="axis"/>'
    )


def _append_y_grid(
    svg: list[str],
    left: int,
    top: int,
    width: int,
    height: int,
    y_min: float,
    y_max: float,
) -> None:
    for tick in range(5):
        fraction = tick / 4
        y = top + height - fraction * height
        value = y_min + fraction * (y_max - y_min)
        svg.append(
            f'<line x1="{left}" y1="{y:.1f}" '
            f'x2="{left + width}" y2="{y:.1f}" class="grid"/>'
        )
        svg.append(
            f'<text x="{left - 10}" y="{y + 4:.1f}" '
            f'text-anchor="end" class="label">{value:.3g}</text>'
        )


def _append_polyline(
    svg: list[str],
    points: Sequence[tuple[int, float]],
    left: int,
    top: int,
    width: int,
    height: int,
    x_min: int,
    x_max: int,
    y_min: float,
    y_max: float,
    color: str,
) -> None:
    if not points:
        return

    mapped = [
        (
            _map_range(iteration, x_min, x_max, left, left + width),
            _map_range(value, y_min, y_max, top + height, top),
        )
        for iteration, value in points
    ]
    if len(mapped) == 1:
        x, y = mapped[0]
        svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>')
        return

    point_text = " ".join(f"{x:.1f},{y:.1f}" for x, y in mapped)
    svg.append(
        f'<polyline points="{point_text}" fill="none" '
        f'stroke="{color}" stroke-width="2.4" stroke-linejoin="round" '
        'stroke-linecap="round"/>'
    )


def _append_legend_item(
    svg: list[str],
    label: str,
    color: str,
    left: int,
    top: int,
    index: int,
) -> None:
    x = left + index * 120
    y = top - 42
    svg.append(f'<line x1="{x}" y1="{y}" x2="{x + 20}" y2="{y}" stroke="{color}" stroke-width="3"/>')
    svg.append(
        f'<text x="{x + 26}" y="{y + 4}" class="legend">'
        f'{html.escape(label)}</text>'
    )


def _append_empty_plot_message(
    svg: list[str],
    left: int,
    top: int,
    width: int,
    height: int,
    message: str,
) -> None:
    svg.append(
        f'<text x="{left + width / 2:.1f}" y="{top + height / 2:.1f}" '
        f'text-anchor="middle" class="label">{html.escape(message)}</text>'
    )


def _iteration_range(records: Sequence[TrainingRecord]) -> tuple[int, int]:
    if not records:
        return 0, 1
    x_min = min(record.iteration for record in records)
    x_max = max(record.iteration for record in records)
    if x_min == x_max:
        return x_min, x_min + 1
    return x_min, x_max


def _value_range(values: Sequence[float], include_zero: bool = False) -> tuple[float, float]:
    y_min = min(values)
    y_max = max(values)
    if include_zero:
        y_min = min(0.0, y_min)
    if y_min == y_max:
        padding = max(abs(y_min) * 0.1, 1.0)
    else:
        padding = (y_max - y_min) * 0.08
    return y_min - padding, y_max + padding


def _map_range(
    value: float,
    source_min: float,
    source_max: float,
    target_min: float,
    target_max: float,
) -> float:
    if source_min == source_max:
        return (target_min + target_max) / 2
    progress = (value - source_min) / (source_max - source_min)
    return target_min + progress * (target_max - target_min)


def _optional_float(text: str | None) -> float | None:
    if text is None or text == "":
        return None
    return float(text)


def _optional_float_text(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.8f}"


if __name__ == "__main__":
    main()
