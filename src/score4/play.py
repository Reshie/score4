from __future__ import annotations

import argparse
import random
from pathlib import Path

from score4.game import ACTION_SIZE, BOARD_SIZE, Score4State, action_to_xy, xy_to_action
from score4.mcts import MCTS, MCTSConfig


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    rng = random.Random(args.seed)
    model, device = load_model(args)
    evaluator = make_evaluator(model, device)
    mcts = MCTS(
        evaluator,
        MCTSConfig(simulations=args.simulations, c_puct=args.c_puct),
        rng=rng,
    )

    human_player = 1 if args.human == "first" else -1
    state = Score4State.new()

    print("Score Four: human vs AlphaZero model")
    print(f"human={'X' if human_player == 1 else 'O'} model={'O' if human_player == 1 else 'X'}")
    print("Enter a column as 0..15, or coordinates as 'x y'. Type 'q' to quit.")

    while not state.is_terminal():
        print()
        print(format_board(state))
        print()
        print(format_action_grid(state))

        if state.to_play == human_player:
            action = prompt_human_action(state)
            if action is None:
                print("quit")
                return
        else:
            result = mcts.search(state, add_noise=False)
            action = result.best_action()
            x, y = action_to_xy(action)
            print(
                f"model plays action={action} (x={x}, y={y}) "
                f"value={result.root_value:+.3f}"
            )
            if args.show_policy:
                print(format_policy(result.policy(temperature=1.0), state))

        state = state.play(action)

    print()
    print(format_board(state))
    print()
    print(format_result(state, human_player))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play Score Four against a model.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Path to a checkpoint created by score4.train.",
    )
    parser.add_argument("--simulations", type=int, default=100)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--human", choices=("first", "second"), default="first")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--res-blocks", type=int, default=4)
    parser.add_argument("--value-hidden", type=int, default=128)
    parser.add_argument("--show-policy", action="store_true")
    return parser.parse_args(argv)


def load_model(args: argparse.Namespace) -> tuple[object, object]:
    try:
        import torch

        from score4.model import AlphaZeroNet
    except ImportError as exc:
        raise SystemExit(
            "Playing against a neural model requires PyTorch. Install it with:\n"
            '  python -m pip install -e ".[train]"'
        ) from exc

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model_args = {
        "channels": args.channels,
        "residual_blocks": args.res_blocks,
        "value_hidden": args.value_hidden,
    }
    state_dict = None

    if args.checkpoint:
        checkpoint = _torch_load(torch, args.checkpoint, device)
        checkpoint_args = checkpoint.get("args", {})
        model_args = {
            "channels": int(checkpoint_args.get("channels", args.channels)),
            "residual_blocks": int(checkpoint_args.get("res_blocks", args.res_blocks)),
            "value_hidden": int(
                checkpoint_args.get("value_hidden", args.value_hidden)
            ),
        }
        state_dict = checkpoint["model"]
        print(f"loaded checkpoint={args.checkpoint}")
    else:
        print("no checkpoint supplied; using a randomly initialized model")

    model = AlphaZeroNet(**model_args).to(device)
    if state_dict is not None:
        model.load_state_dict(state_dict)
    model.eval()
    print(f"device={device} simulations={args.simulations}")
    return model, device


def make_evaluator(model: object, device: object) -> object:
    from score4.model import NetworkEvaluator

    return NetworkEvaluator(model, device)


def parse_action(text: str, state: Score4State) -> int | None:
    cleaned = text.strip().lower()
    if cleaned in {"q", "quit", "exit"}:
        return None

    action = _parse_action_number(cleaned)
    if action is None:
        action = _parse_xy(cleaned)
    if action is None:
        raise ValueError("enter 0..15 or coordinates like '2 3'")
    if action not in state.legal_actions():
        raise ValueError(f"column {action} is not legal")
    return action


def prompt_human_action(state: Score4State) -> int | None:
    while True:
        try:
            return parse_action(input("your move> "), state)
        except ValueError as exc:
            print(exc)


def format_board(state: Score4State) -> str:
    return state.render()


def format_action_grid(state: Score4State) -> str:
    legal_actions = set(state.legal_actions())
    rows = ["actions (x across, y down):"]
    for y in range(BOARD_SIZE):
        values = []
        for x in range(BOARD_SIZE):
            action = xy_to_action(x, y)
            label = f"{action:2d}" if action in legal_actions else " -"
            values.append(label)
        rows.append(f"y={y} " + " ".join(values))
    rows.append("    " + " ".join(f"x={x}" for x in range(BOARD_SIZE)))
    return "\n".join(rows)


def format_policy(policy: list[float], state: Score4State) -> str:
    rows = ["policy:"]
    legal_actions = set(state.legal_actions())
    for y in range(BOARD_SIZE):
        values = []
        for x in range(BOARD_SIZE):
            action = xy_to_action(x, y)
            if action in legal_actions:
                values.append(f"{policy[action]:.2f}")
            else:
                values.append(" -- ")
        rows.append(" ".join(values))
    return "\n".join(rows)


def format_result(state: Score4State, human_player: int) -> str:
    winner = state.winner()
    if winner == 0:
        return "draw"
    if winner == human_player:
        return "you win"
    return "model wins"


def _parse_action_number(cleaned: str) -> int | None:
    if not cleaned.isdigit():
        return None
    action = int(cleaned)
    if 0 <= action < ACTION_SIZE:
        return action
    return None


def _parse_xy(cleaned: str) -> int | None:
    parts = cleaned.replace(",", " ").split()
    if len(parts) != 2:
        return None
    if not all(part.isdigit() for part in parts):
        return None
    x, y = (int(part) for part in parts)
    if 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE:
        return xy_to_action(x, y)
    return None


def _torch_load(torch: object, path: Path, device: object) -> dict[str, object]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


if __name__ == "__main__":
    main()
