import unittest

from score4.play import parse_action
from score4.game import Score4State


class PlayCliTests(unittest.TestCase):
    def test_parse_action_number(self) -> None:
        state = Score4State.new()

        self.assertEqual(parse_action("7", state), 7)

    def test_parse_xy_coordinates(self) -> None:
        state = Score4State.new()

        self.assertEqual(parse_action("2 3", state), 14)
        self.assertEqual(parse_action("2,3", state), 14)

    def test_parse_quit(self) -> None:
        state = Score4State.new()

        self.assertIsNone(parse_action("q", state))

    def test_rejects_full_column(self) -> None:
        state = Score4State.new()
        for _ in range(4):
            state = state.play(0)

        with self.assertRaises(ValueError):
            parse_action("0", state)


if __name__ == "__main__":
    unittest.main()
