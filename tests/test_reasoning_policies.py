"""Pruebas mínimas de Fase 3 para CI."""

from __future__ import annotations

import numpy as np

from baseline_env import build_default_env
from reasoning_policies import (
    AdaptiveBudgetConfig,
    AdaptiveRolloutBudget,
    BestOfNActions,
    GraphOfWaypoints,
    LearnedReasoningController,
    ReasoningControllerConfig,
    RolloutConfig,
    TreeOfActions,
    TreeSearchConfig,
    WaypointGraphConfig,
    build_reasoning_strategies,
)


def assert_valid_action(action: int) -> None:
    """Verifica que la acción pertenezca al espacio discreto del auto."""
    assert action in {0, 1, 2}


def test_best_of_n_no_mutates_environment() -> None:
    """Best-of-N debe simular en copias y no mover el entorno real."""
    env = build_default_env(seed=7, max_steps=80)
    state = env.reset(seed=7)
    original_position = env.position.copy()
    policy = BestOfNActions(RolloutConfig(depth=2, samples_per_action=1))

    action, info = policy.select_action(env, state)

    assert_valid_action(action)
    assert np.allclose(env.position, original_position)
    assert info["costo_decision"] >= 1.0


def test_tree_of_actions_returns_valid_action() -> None:
    """Tree-of-Actions debe devolver una acción válida y costo de decisión."""
    env = build_default_env(seed=11, max_steps=80)
    state = env.reset(seed=11)
    policy = TreeOfActions(TreeSearchConfig(depth=2, beam_width=2, max_expansions=30))

    action, info = policy.select_action(env, state)

    assert_valid_action(action)
    assert info["costo_decision"] >= 1.0


def test_graph_of_waypoints_returns_valid_action() -> None:
    """Graph-of-Waypoints debe producir una acción de bajo nivel válida."""
    env = build_default_env(seed=13, max_steps=80)
    state = env.reset(seed=13)
    policy = GraphOfWaypoints(WaypointGraphConfig(cell_size=25, waypoint_stride=3))

    action, info = policy.select_action(env, state)

    assert_valid_action(action)
    assert info["costo_decision"] >= 0.0


def test_adaptive_budget_increases_with_difficulty() -> None:
    """El presupuesto adaptativo debe asignar más cómputo a estados difíciles."""
    policy = AdaptiveRolloutBudget(AdaptiveBudgetConfig())

    easy_budget = policy.select_budget(0.10)
    hard_budget = policy.select_budget(0.90)

    assert hard_budget.depth > easy_budget.depth
    assert hard_budget.samples_per_action > easy_budget.samples_per_action


def test_learned_controller_save_and_load(tmp_path) -> None:
    """El controlador aprendido debe guardar y cargar sus pesos."""
    env = build_default_env(seed=19, max_steps=80)
    state = env.reset(seed=19)
    strategies = build_reasoning_strategies()
    controller = LearnedReasoningController(
        strategies, ReasoningControllerConfig(epsilon=0.0, seed=19)
    )

    action, info = controller.select_action(env, state)
    assert_valid_action(action)
    assert "estrategia" in info

    path = tmp_path / "controlador.json"
    controller.save(path)

    loaded = LearnedReasoningController(strategies, ReasoningControllerConfig(epsilon=0.0, seed=19))
    loaded.load(path)
    assert loaded.weights.shape == controller.weights.shape
