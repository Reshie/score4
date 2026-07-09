import unittest

from score4.game import (
    ACTION_SIZE,
    BOARD_SIZE,
    Score4State,
    WINNING_LINES,
    cell_index,
)


class Score4GameTests(unittest.TestCase):
    def test_generates_all_winning_lines(self) -> None:
        self.assertEqual(len(WINNING_LINES), 76)
        self.assertEqual(len(set(WINNING_LINES)), 76)

    def test_vertical_win_with_gravity(self) -> None:
        state = Score4State.new()
        for action in (0, 1, 0, 1, 0, 1, 0):
            state = state.play(action)

        self.assertTrue(state.is_terminal())
        self.assertEqual(state.winner(), 1)
        self.assertEqual(state.terminal_value(), -1.0)

    def test_every_generated_line_can_win(self) -> None:
        for line in WINNING_LINES:
            board = [0 for _ in range(BOARD_SIZE**3)]
            for index in line:
                board[index] = -1
            state = Score4State(
                board=tuple(board),
                heights=(0,) * ACTION_SIZE,
                to_play=1,
                ply=4,
            )
            self.assertEqual(state.winner(), -1, line)

    def test_column_becomes_illegal_when_full(self) -> None:
        state = Score4State.new()
        for _ in range(BOARD_SIZE):
            state = state.play(0)

        self.assertNotIn(0, state.legal_actions())
        self.assertIn(1, state.legal_actions())

    def test_observation_is_from_player_perspective(self) -> None:
        state = Score4State.new().play(0)
        own, opponent = state.observation()
        self.assertEqual(opponent[0][0][0], 1.0)
        self.assertEqual(own[0][0][0], 0.0)

        board_index = cell_index(0, 0, 0)
        self.assertEqual(state.board[board_index], 1)


if __name__ == "__main__":
    unittest.main()
