from __future__ import annotations

import argparse
import json
import mimetypes
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from score4.game import ACTION_SIZE, BOARD_CELLS, BOARD_SIZE, Score4State, action_to_xy
from score4.mcts import MCTS, MCTSConfig
from score4.play import _torch_load


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    app = ModelWebApp(args)
    server = ThreadingHTTPServer((args.host, args.port), app.handler_class())
    print(f"serving http://{args.host}:{args.port}")
    print(f"checkpoint={app.checkpoint_path}")
    print(f"device={app.device} simulations={args.simulations}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the 3D Score Four web game.")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--simulations", type=int, default=120)
    parser.add_argument("--max-simulations", type=int, default=400)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


class ModelWebApp:
    def __init__(self, args: argparse.Namespace) -> None:
        try:
            import torch

            from score4.model import AlphaZeroNet, NetworkEvaluator
        except ImportError as exc:
            raise SystemExit(
                "Web AI requires PyTorch. Install it with:\n"
                '  python -m pip install -e ".[train]"'
            ) from exc

        self.args = args
        self.torch = torch
        self.static_root = Path(__file__).resolve().parents[2] / "web"
        self.checkpoint_path = args.checkpoint or _find_latest_checkpoint()
        if self.checkpoint_path is None:
            raise SystemExit(
                "No checkpoint found. Train a model or pass "
                "--checkpoint runs/.../checkpoint_XXXX.pt"
            )

        self.device = _resolve_device(torch, args.device)
        checkpoint = _torch_load(torch, self.checkpoint_path, self.device)
        checkpoint_args = checkpoint.get("args", {})
        model = AlphaZeroNet(
            channels=int(checkpoint_args.get("channels", 64)),
            residual_blocks=int(checkpoint_args.get("res_blocks", 4)),
            value_hidden=int(checkpoint_args.get("value_hidden", 128)),
        ).to(self.device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        self.evaluator = NetworkEvaluator(model, self.device, cache_size=100_000)
        self.rng = __import__("random").Random(args.seed)

    def handler_class(self) -> type[BaseHTTPRequestHandler]:
        app = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                if self.path == "/api/info":
                    self._send_json(
                        {
                            "checkpoint": str(app.checkpoint_path),
                            "device": str(app.device),
                            "simulations": app.args.simulations,
                            "maxSimulations": app.args.max_simulations,
                        }
                    )
                    return
                self._serve_static()

            def do_POST(self) -> None:
                parsed = urlparse(self.path)
                if parsed.path != "/api/move":
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                try:
                    payload = self._read_json()
                    response = app.choose_move(payload)
                except ValueError as exc:
                    self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                    return
                except Exception as exc:  # pragma: no cover - defensive HTTP boundary.
                    self._send_json(
                        {"error": f"AI move failed: {exc}"},
                        HTTPStatus.INTERNAL_SERVER_ERROR,
                    )
                    return
                self._send_json(response)

            def log_message(self, format: str, *args: object) -> None:
                return

            def _read_json(self) -> dict[str, Any]:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0:
                    raise ValueError("request body is empty")
                raw = self.rfile.read(length)
                try:
                    payload = json.loads(raw.decode("utf-8"))
                except json.JSONDecodeError as exc:
                    raise ValueError("invalid JSON") from exc
                if not isinstance(payload, dict):
                    raise ValueError("JSON body must be an object")
                return payload

            def _serve_static(self) -> None:
                parsed = urlparse(self.path)
                request_path = unquote(parsed.path)
                if request_path == "/":
                    request_path = "/index.html"
                relative = request_path.lstrip("/")
                target = (app.static_root / relative).resolve()
                if not _is_relative_to(target, app.static_root.resolve()):
                    self.send_error(HTTPStatus.FORBIDDEN)
                    return
                if not target.exists() or not target.is_file():
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                content_type = mimetypes.guess_type(target)[0] or "application/octet-stream"
                data = target.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _send_json(
                self,
                payload: dict[str, Any],
                status: HTTPStatus = HTTPStatus.OK,
            ) -> None:
                data = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        return Handler

    def choose_move(self, payload: dict[str, Any]) -> dict[str, Any]:
        state = _state_from_payload(payload)
        if state.is_terminal():
            raise ValueError("state is already terminal")
        legal_actions = state.legal_actions()
        if not legal_actions:
            raise ValueError("state has no legal actions")

        requested_simulations = int(payload.get("simulations", self.args.simulations))
        simulations = max(1, min(requested_simulations, self.args.max_simulations))
        mcts = MCTS(
            self.evaluator,
            MCTSConfig(simulations=simulations, c_puct=self.args.c_puct),
            rng=self.rng,
        )
        result = mcts.search(state, add_noise=False)
        action = result.best_action()
        x, y = action_to_xy(action)
        return {
            "action": action,
            "x": x,
            "y": y,
            "value": result.root_value,
            "policy": result.policy(temperature=1.0),
            "simulations": simulations,
            "cache": self.evaluator.cache_info(),
        }


def _state_from_payload(payload: dict[str, Any]) -> Score4State:
    board = payload.get("board")
    heights = payload.get("heights")
    to_play = payload.get("toPlay")
    if not isinstance(board, list) or len(board) != BOARD_CELLS:
        raise ValueError(f"board must be a list of {BOARD_CELLS} values")
    if not isinstance(heights, list) or len(heights) != ACTION_SIZE:
        raise ValueError(f"heights must be a list of {ACTION_SIZE} values")

    board_values = tuple(_checked_int(value, {-1, 0, 1}, "board") for value in board)
    height_values = tuple(_checked_int(value, set(range(BOARD_SIZE + 1)), "heights") for value in heights)
    to_play_value = _checked_int(to_play, {-1, 1}, "toPlay")
    _validate_heights_match_board(board_values, height_values)
    return Score4State(
        board=board_values,
        heights=height_values,
        to_play=to_play_value,
        ply=sum(height_values),
    )


def _checked_int(value: Any, allowed: set[int], name: str) -> int:
    if not isinstance(value, int) or value not in allowed:
        raise ValueError(f"{name} contains invalid value: {value!r}")
    return value


def _validate_heights_match_board(
    board: tuple[int, ...],
    heights: tuple[int, ...],
) -> None:
    for action, height in enumerate(heights):
        for z in range(BOARD_SIZE):
            occupied = board[z * ACTION_SIZE + action] != 0
            if z < height and not occupied:
                raise ValueError("heights do not match board occupancy")
            if z >= height and occupied:
                raise ValueError("heights do not match board occupancy")


def _resolve_device(torch: object, requested: str) -> object:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _find_latest_checkpoint() -> Path | None:
    candidates = sorted(Path("runs").glob("**/checkpoint_*.pt"))
    if not candidates:
        return None
    return candidates[-1]


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


if __name__ == "__main__":
    main()
