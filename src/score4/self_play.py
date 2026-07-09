from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

from score4.game import ACTION_SIZE, Score4State
from score4.mcts import (
    Edge,
    MCTS,
    MCTSConfig,
    Node,
    Evaluator,
    add_exploration_noise,
    expand_node,
    search_result_from_root,
    select_edge,
)


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
    reuse_tree: bool = False


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
    root = None
    history: list[tuple[Score4State, list[float]]] = []

    while not state.is_terminal():
        result = mcts.search(
            state,
            add_noise=True,
            root=root if config.reuse_tree else None,
        )
        temperature = 1.0 if state.ply < config.temperature_moves else 0.0
        policy = _mask_policy(result.policy(temperature), state.legal_actions())
        action = _sample_action(policy, rng)
        history.append((state, policy))
        root = result.child_for(action) if config.reuse_tree else None
        state = state.play(action)

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


def play_games_batched(
    evaluator: Evaluator,
    config: SelfPlayConfig,
    games: int,
    rng: random.Random | None = None,
) -> list[list[TrainingExample]]:
    rng = rng or random.Random()
    mcts_config = MCTSConfig(
        simulations=config.simulations,
        c_puct=config.c_puct,
        dirichlet_alpha=config.dirichlet_alpha,
        exploration_fraction=config.exploration_fraction,
    )
    active = [
        _BatchedGame(
            state=Score4State.new(),
            root=Node(),
            history=[],
        )
        for _ in range(max(0, games))
    ]
    finished: list[list[TrainingExample]] = []

    while active:
        _prepare_roots(active, evaluator, mcts_config, rng)
        _run_batched_search(active, evaluator, mcts_config)

        next_active: list[_BatchedGame] = []
        for game in active:
            result = search_result_from_root(game.root)
            temperature = 1.0 if game.state.ply < config.temperature_moves else 0.0
            policy = _mask_policy(result.policy(temperature), game.state.legal_actions())
            action = _sample_action(policy, rng)
            game.history.append((game.state, policy))

            next_root = result.child_for(action) if config.reuse_tree else None
            game.state = game.state.play(action)
            game.root = next_root or Node()

            if game.state.is_terminal():
                finished.append(_examples_from_history(game.history, game.state.winner()))
            else:
                next_active.append(game)
        active = next_active

    return finished


@dataclass
class _BatchedGame:
    state: Score4State
    root: Node
    history: list[tuple[Score4State, list[float]]]


@dataclass
class _Leaf:
    state: Score4State
    node: Node
    path: list[Edge]


def _prepare_roots(
    games: Sequence[_BatchedGame],
    evaluator: Evaluator,
    config: MCTSConfig,
    rng: random.Random,
) -> None:
    unexpanded = [
        game
        for game in games
        if not game.state.is_terminal() and not game.root.edges
    ]
    if unexpanded:
        evaluations = _evaluate_batch(
            evaluator,
            [game.state for game in unexpanded],
        )
        for game, (policy, value) in zip(unexpanded, evaluations):
            expand_node(game.root, game.state, policy, value)

    for game in games:
        add_exploration_noise(game.root, config, rng)


def _run_batched_search(
    games: Sequence[_BatchedGame],
    evaluator: Evaluator,
    config: MCTSConfig,
) -> None:
    for _ in range(max(0, config.simulations)):
        leaves: list[_Leaf] = []
        for game in games:
            leaf = _select_leaf(game.root, game.state, config)
            if isinstance(leaf, _Leaf):
                leaves.append(leaf)
            else:
                path, terminal_value = leaf
                _backup(path, terminal_value)

        if not leaves:
            continue

        evaluations = _evaluate_batch(evaluator, [leaf.state for leaf in leaves])
        for leaf, (policy, value) in zip(leaves, evaluations):
            leaf_value = expand_node(leaf.node, leaf.state, policy, value)
            _backup(leaf.path, leaf_value)


def _select_leaf(
    root: Node,
    root_state: Score4State,
    config: MCTSConfig,
) -> _Leaf | tuple[list[Edge], float]:
    node = root
    state = root_state
    path: list[Edge] = []

    while True:
        if state.is_terminal():
            return path, state.terminal_value()
        if not node.edges:
            return _Leaf(state=state, node=node, path=path)

        edge = select_edge(node, config)
        path.append(edge)
        state = state.play(edge.action)
        if edge.child is None:
            edge.child = Node()
        node = edge.child


def _backup(path: Sequence[Edge], leaf_value: float) -> None:
    value = leaf_value
    for edge in reversed(path):
        value = -value
        edge.visit_count += 1
        edge.value_sum += value


def _evaluate_batch(
    evaluator: Evaluator,
    states: Sequence[Score4State],
) -> list[tuple[Sequence[float], float]]:
    evaluate_batch = getattr(evaluator, "evaluate_batch", None)
    if callable(evaluate_batch):
        return evaluate_batch(states)
    return [evaluator(state) for state in states]


def _examples_from_history(
    history: Sequence[tuple[Score4State, list[float]]],
    winner: int,
) -> list[TrainingExample]:
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
