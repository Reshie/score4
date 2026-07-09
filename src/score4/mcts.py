from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Protocol, Sequence

from score4.game import ACTION_SIZE, Score4State


class Evaluator(Protocol):
    def __call__(self, state: Score4State) -> tuple[Sequence[float], float]:
        """Return policy priors for all actions and value for state.to_play."""


@dataclass
class MCTSConfig:
    simulations: int = 100
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    exploration_fraction: float = 0.25


@dataclass
class SearchResult:
    visit_counts: list[int]
    root_value: float
    legal_actions: list[int]

    def policy(self, temperature: float = 1.0) -> list[float]:
        if not self.legal_actions:
            return [0.0 for _ in range(ACTION_SIZE)]

        legal_total = sum(self.visit_counts[action] for action in self.legal_actions)
        if legal_total <= 0:
            probability = 1.0 / len(self.legal_actions)
            return [
                probability if action in self.legal_actions else 0.0
                for action in range(ACTION_SIZE)
            ]

        if temperature <= 0:
            best = max(
                self.legal_actions,
                key=lambda action: self.visit_counts[action],
            )
            return [1.0 if action == best else 0.0 for action in range(ACTION_SIZE)]

        scaled = [0.0 for _ in range(ACTION_SIZE)]
        for action in self.legal_actions:
            count = self.visit_counts[action]
            scaled[action] = count ** (1.0 / temperature) if count > 0 else 0.0
        total = sum(scaled)
        if total <= 0:
            return [
                self.visit_counts[action] / legal_total
                if action in self.legal_actions
                else 0.0
                for action in range(ACTION_SIZE)
            ]
        return [count / total for count in scaled]

    def best_action(self) -> int:
        return max(self.legal_actions, key=lambda action: self.visit_counts[action])


@dataclass
class Edge:
    action: int
    prior: float
    visit_count: int = 0
    value_sum: float = 0.0
    child: "Node | None" = None

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


@dataclass
class Node:
    edges: dict[int, Edge] = field(default_factory=dict)


class MCTS:
    def __init__(
        self,
        evaluator: Evaluator,
        config: MCTSConfig | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self.evaluator = evaluator
        self.config = config or MCTSConfig()
        self.rng = rng or random.Random()

    def search(self, state: Score4State, add_noise: bool = False) -> SearchResult:
        root = Node()
        root_value = self._expand(root, state)
        if add_noise:
            self._add_exploration_noise(root)

        for _ in range(max(0, self.config.simulations)):
            self._search(root, state)

        counts = [0 for _ in range(ACTION_SIZE)]
        value_sum = 0.0
        visits = 0
        for action, edge in root.edges.items():
            counts[action] = edge.visit_count
            value_sum += edge.value_sum
            visits += edge.visit_count

        if visits:
            root_value = value_sum / visits
        return SearchResult(
            visit_counts=counts,
            root_value=root_value,
            legal_actions=list(root.edges),
        )

    def select_action(
        self,
        state: Score4State,
        temperature: float = 1.0,
        add_noise: bool = False,
    ) -> int:
        result = self.search(state, add_noise=add_noise)
        policy = result.policy(temperature=temperature)
        return _sample_action(policy, self.rng)

    def _search(self, node: Node, state: Score4State) -> float:
        if state.is_terminal():
            return state.terminal_value()

        if not node.edges:
            return self._expand(node, state)

        edge = self._select_edge(node)
        next_state = state.play(edge.action)
        if edge.child is None:
            edge.child = Node()

        child_value = self._search(edge.child, next_state)
        value = -child_value
        edge.visit_count += 1
        edge.value_sum += value
        return value

    def _expand(self, node: Node, state: Score4State) -> float:
        if state.is_terminal():
            return state.terminal_value()

        policy, value = self.evaluator(state)
        if len(policy) != ACTION_SIZE:
            raise ValueError(
                f"evaluator policy must have {ACTION_SIZE} entries, got {len(policy)}"
            )

        legal_actions = state.legal_actions()
        masked = [max(0.0, float(policy[action])) for action in legal_actions]
        total = sum(masked)
        if total <= 0:
            prior = 1.0 / len(legal_actions)
            for action in legal_actions:
                node.edges[action] = Edge(action=action, prior=prior)
        else:
            for action, probability in zip(legal_actions, masked):
                node.edges[action] = Edge(action=action, prior=probability / total)

        return max(-1.0, min(1.0, float(value)))

    def _select_edge(self, node: Node) -> Edge:
        total_visits = sum(edge.visit_count for edge in node.edges.values())
        exploration = math.sqrt(total_visits + 1.0)

        def score(edge: Edge) -> float:
            u_value = (
                self.config.c_puct
                * edge.prior
                * exploration
                / (1 + edge.visit_count)
            )
            return edge.q_value + u_value

        return max(node.edges.values(), key=score)

    def _add_exploration_noise(self, node: Node) -> None:
        edges = list(node.edges.values())
        if not edges:
            return

        samples = [
            self.rng.gammavariate(self.config.dirichlet_alpha, 1.0)
            for _ in edges
        ]
        total = sum(samples)
        if total <= 0:
            return

        fraction = self.config.exploration_fraction
        for edge, sample in zip(edges, samples):
            noise = sample / total
            edge.prior = (1.0 - fraction) * edge.prior + fraction * noise


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
