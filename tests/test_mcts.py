import random
import unittest

from score4.game import ACTION_SIZE, Score4State
from score4.mcts import MCTS, MCTSConfig


def uniform_evaluator(state: Score4State) -> tuple[list[float], float]:
    return [1.0 / ACTION_SIZE for _ in range(ACTION_SIZE)], 0.0


class MCTSTests(unittest.TestCase):
    def test_search_returns_normalized_policy(self) -> None:
        mcts = MCTS(
            uniform_evaluator,
            MCTSConfig(simulations=8),
            rng=random.Random(0),
        )
        result = mcts.search(Score4State.new())
        policy = result.policy()

        self.assertAlmostEqual(sum(policy), 1.0)
        self.assertEqual(len(policy), ACTION_SIZE)

    def test_finds_immediate_winning_column(self) -> None:
        state = Score4State.new()
        for action in (0, 1, 0, 1, 0, 2):
            state = state.play(action)

        mcts = MCTS(
            uniform_evaluator,
            MCTSConfig(simulations=32, c_puct=1.0),
            rng=random.Random(0),
        )
        result = mcts.search(state)

        self.assertEqual(result.best_action(), 0)


if __name__ == "__main__":
    unittest.main()
