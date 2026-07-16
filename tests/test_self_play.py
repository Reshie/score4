import random
import unittest
from typing import Sequence

from score4.game import ACTION_SIZE, Score4State
from score4.self_play import (
    SelfPlayConfig,
    native_self_play_available,
    play_games_batched,
)


class BatchUniformEvaluator:
    def __init__(self) -> None:
        self.batch_calls = 0
        self.max_batch_size = 0

    def __call__(self, state: Score4State) -> tuple[list[float], float]:
        raise AssertionError("batched self-play should use evaluate_batch")

    def evaluate_batch(
        self,
        states: Sequence[Score4State],
    ) -> list[tuple[list[float], float]]:
        self.batch_calls += 1
        self.max_batch_size = max(self.max_batch_size, len(states))
        policy = [1.0 / ACTION_SIZE for _ in range(ACTION_SIZE)]
        return [(policy, 0.0) for _ in states]


class BatchedSelfPlayTests(unittest.TestCase):
    def test_mcts_threads_defaults_to_auto(self) -> None:
        self.assertEqual(SelfPlayConfig().mcts_threads, 0)

    def test_native_backend_status_is_boolean(self) -> None:
        self.assertIsInstance(native_self_play_available(), bool)

    def test_play_games_batched_returns_one_history_per_game(self) -> None:
        evaluator = BatchUniformEvaluator()

        games = play_games_batched(
            evaluator,
            SelfPlayConfig(simulations=4),
            games=3,
            rng=random.Random(0),
        )

        self.assertEqual(len(games), 3)
        self.assertGreater(evaluator.batch_calls, 0)
        self.assertGreater(evaluator.max_batch_size, 1)
        for examples in games:
            self.assertGreater(len(examples), 0)
            self.assertAlmostEqual(sum(examples[0].policy), 1.0)


if __name__ == "__main__":
    unittest.main()
