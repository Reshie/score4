from __future__ import annotations

from dataclasses import dataclass

BOARD_SIZE = 4
BOARD_CELLS = BOARD_SIZE**3
ACTION_SIZE = BOARD_SIZE**2


def action_to_xy(action: int) -> tuple[int, int]:
    """Return the x/y column coordinates for an action in 0..15."""
    if action < 0 or action >= ACTION_SIZE:
        raise ValueError(f"action must be in [0, {ACTION_SIZE}), got {action}")
    return action % BOARD_SIZE, action // BOARD_SIZE


def xy_to_action(x: int, y: int) -> int:
    if not (0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE):
        raise ValueError(f"x/y must be in [0, {BOARD_SIZE})")
    return y * BOARD_SIZE + x


def cell_index(x: int, y: int, z: int) -> int:
    if not (
        0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE and 0 <= z < BOARD_SIZE
    ):
        raise ValueError(f"x/y/z must be in [0, {BOARD_SIZE})")
    return z * ACTION_SIZE + y * BOARD_SIZE + x


def _generate_winning_lines() -> tuple[tuple[int, int, int, int], ...]:
    directions = (
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 1, 0),
        (1, -1, 0),
        (1, 0, 1),
        (1, 0, -1),
        (0, 1, 1),
        (0, 1, -1),
        (1, 1, 1),
        (1, 1, -1),
        (1, -1, 1),
        (1, -1, -1),
    )
    lines: list[tuple[int, int, int, int]] = []
    for dz in range(BOARD_SIZE):
        for dy in range(BOARD_SIZE):
            for dx in range(BOARD_SIZE):
                for step_x, step_y, step_z in directions:
                    end_x = dx + (BOARD_SIZE - 1) * step_x
                    end_y = dy + (BOARD_SIZE - 1) * step_y
                    end_z = dz + (BOARD_SIZE - 1) * step_z
                    if (
                        0 <= end_x < BOARD_SIZE
                        and 0 <= end_y < BOARD_SIZE
                        and 0 <= end_z < BOARD_SIZE
                    ):
                        lines.append(
                            tuple(
                                cell_index(
                                    dx + i * step_x,
                                    dy + i * step_y,
                                    dz + i * step_z,
                                )
                                for i in range(BOARD_SIZE)
                            )
                        )
    return tuple(lines)


WINNING_LINES = _generate_winning_lines()


@dataclass(frozen=True)
class Score4State:
    """Immutable 4x4x4 Score Four state.

    The board is stored as a flat tuple indexed by z, then y, then x. Each
    action chooses one x/y column and places the next stone at that column's
    current height.
    """

    board: tuple[int, ...]
    heights: tuple[int, ...]
    to_play: int = 1
    ply: int = 0

    @classmethod
    def new(cls) -> "Score4State":
        return cls(
            board=(0,) * BOARD_CELLS,
            heights=(0,) * ACTION_SIZE,
            to_play=1,
            ply=0,
        )

    def legal_actions(self) -> list[int]:
        if self.is_terminal():
            return []
        return [
            action
            for action, height in enumerate(self.heights)
            if height < BOARD_SIZE
        ]

    def play(self, action: int) -> "Score4State":
        if self.is_terminal():
            raise ValueError("cannot play from a terminal state")
        if action < 0 or action >= ACTION_SIZE:
            raise ValueError(f"action must be in [0, {ACTION_SIZE}), got {action}")

        height = self.heights[action]
        if height >= BOARD_SIZE:
            raise ValueError(f"column {action} is full")

        board = list(self.board)
        board[height * ACTION_SIZE + action] = self.to_play
        heights = list(self.heights)
        heights[action] += 1

        return Score4State(
            board=tuple(board),
            heights=tuple(heights),
            to_play=-self.to_play,
            ply=self.ply + 1,
        )

    def winner(self) -> int:
        for line in WINNING_LINES:
            first = self.board[line[0]]
            if first and all(self.board[index] == first for index in line[1:]):
                return first
        return 0

    def is_full(self) -> bool:
        return self.ply >= BOARD_CELLS

    def is_terminal(self) -> bool:
        return self.winner() != 0 or self.is_full()

    def terminal_value(self) -> float:
        """Return terminal value from the current player's perspective."""
        winner = self.winner()
        if winner == 0:
            return 0.0
        return 1.0 if winner == self.to_play else -1.0

    def outcome_for(self, player: int) -> float:
        if player not in (-1, 1):
            raise ValueError("player must be 1 or -1")
        winner = self.winner()
        if winner == 0:
            return 0.0
        return 1.0 if winner == player else -1.0

    def observation(self) -> list[list[list[list[float]]]]:
        """Encode the state as [channel][z][y][x] from to_play's perspective."""
        own = _empty_plane()
        opponent = _empty_plane()
        for z in range(BOARD_SIZE):
            for y in range(BOARD_SIZE):
                for x in range(BOARD_SIZE):
                    value = self.board[cell_index(x, y, z)]
                    if value == self.to_play:
                        own[z][y][x] = 1.0
                    elif value == -self.to_play:
                        opponent[z][y][x] = 1.0
        return [own, opponent]

    def render(self) -> str:
        marks = {1: "X", -1: "O", 0: "."}
        layers: list[str] = []
        for z in reversed(range(BOARD_SIZE)):
            rows = []
            for y in range(BOARD_SIZE):
                rows.append(
                    " ".join(
                        marks[self.board[cell_index(x, y, z)]]
                        for x in range(BOARD_SIZE)
                    )
                )
            layers.append(f"z={z}\n" + "\n".join(rows))
        return "\n\n".join(layers)


def _empty_plane() -> list[list[list[float]]]:
    return [
        [
            [0.0 for _ in range(BOARD_SIZE)]
            for _ in range(BOARD_SIZE)
        ]
        for _ in range(BOARD_SIZE)
    ]
