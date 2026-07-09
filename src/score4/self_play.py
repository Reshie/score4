from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

from score4.game import ACTION_SIZE, Score4State
from score4.mcts import MCTS, MCTSConfig, Evaluator


@dataclass(frozen=True)
class TrainingExample:
    observation: list[list[list[list[float]]]]
    policy: list[float]
    value: float


@dataclass
class SelfPlayConfig:
    simulations: int = 100
    temperature_moves: int = 12
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    exploration_fraction: float = 0.25


def play_game(
    evaluator: Evaluator,
    config: SelfPlayConfig,
    rng: random.Random | None = None,
) -> list[TrainingExample]:
    rng = rng or random.Random()
    mcts = MCTS(
        evaluator=evaluator,
        config=MCTSConfig(
            simulations=config.simulations,
            c_puct=config.c_puct,
            dirichlet_alpha=config.dirichlet_alpha,
            exploration_fraction=config.exploration_fraction,
        ),
        rng=rng,
    )

    state = Score4State.new()
    history: list[tuple[Score4State, list[float]]] = []

    while not state.is_terminal():
        result = mcts.search(state, add_noise=True)
        temperature = 1.0 if state.ply < config.temperature_moves else 0.0
        policy = _mask_policy(result.policy(temperature), state.legal_actions())
        history.append((state, policy))
        state = state.play(_sample_action(policy, rng))

    winner = state.winner()
    examples = []
    for past_state, policy in history:
        if winner == 0:
            value = 0.0
        else:
            value = 1.0 if winner == past_state.to_play else -1.0
        examples.append(
            TrainingExample(
                observation=past_state.observation(),
                policy=policy,
                value=value,
            )
        )
    return examples


def _mask_policy(policy: Sequence[float], legal_actions: Sequence[int]) -> list[float]:
    masked = [0.0 for _ in range(ACTION_SIZE)]
    for action in legal_actions:
        masked[action] = max(0.0, float(policy[action]))
    total = sum(masked)
    if total <= 0:
        probability = 1.0 / len(legal_actions)
        for action in legal_actions:
            masked[action] = probability
        return masked
    return [probability / total for probability in masked]


def _sample_action(policy: Sequence[float], rng: random.Random) -> int:
    threshold = rng.random()
    cumulative = 0.0
    best_action = 0
    best_probability = -1.0
    for action, probability in enumerate(policy):
        if probability > best_probability:
            best_probability = probability
            best_action = action
        cumulative += probability
        if threshold <= cumulative:
            return action
    return best_action
