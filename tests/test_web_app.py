import unittest

from score4.game import Score4State
from score4.web_app import _state_from_payload


class WebAppStateTests(unittest.TestCase):
    def test_state_from_payload_accepts_valid_state(self) -> None:
        state = Score4State.new().play(0).play(5)

        restored = _state_from_payload(
            {
                "board": list(state.board),
                "heights": list(state.heights),
                "toPlay": state.to_play,
            }
        )

        self.assertEqual(restored.board, state.board)
        self.assertEqual(restored.heights, state.heights)
        self.assertEqual(restored.to_play, state.to_play)
        self.assertEqual(restored.ply, state.ply)

    def test_state_from_payload_rejects_mismatched_heights(self) -> None:
        state = Score4State.new().play(0)
        heights = list(state.heights)
        heights[0] = 0

        with self.assertRaises(ValueError):
            _state_from_payload(
                {
                    "board": list(state.board),
                    "heights": heights,
                    "toPlay": state.to_play,
                }
            )


if __name__ == "__main__":
    unittest.main()
