"""Pruebas de Fase 5: RLoT y GoT para navegación."""

from __future__ import annotations

import numpy as np

from baseline_env import build_default_env
from rlot_got_navigation import (
    ChainOfActionsConfig,
    ChainOfActionsPolicy,
    GoTNavigationConfig,
    GoTNavigationGraphPolicy,
    RLoTNavigator,
    RLoTNavigatorConfig,
    ReflectiveActionPolicy,
)


def assert_valid_action(action: int) -> None:
    """Verifica que la acción sea válida para el auto."""
    assert action in {0, 1, 2}


def test_chain_of_actions_returns_valid_action_without_mutating() -> None:
    """CHAIN debe simular en copias y no mover el entorno real."""
    env = build_default_env(seed=101, max_steps=60)
    state = env.reset(seed=101)
    original_position = env.position.copy()
    policy = ChainOfActionsPolicy(ChainOfActionsConfig(depth=2))

    action, info = policy.select_action(env, state)

    assert_valid_action(action)
    assert np.allclose(env.position, original_position)
    assert info["bloque_razonamiento"] == "CHAIN"
    assert info["costo_decision"] >= 1.0


def test_reflective_action_policy_returns_risk_metadata() -> None:
    """REFLECT debe devolver acción válida y metadatos de reflexión."""
    env = build_default_env(seed=102, max_steps=60)
    state = env.reset(seed=102)
    policy = ReflectiveActionPolicy()

    action, info = policy.select_action(env, state)

    assert_valid_action(action)
    assert info["bloque_razonamiento"] == "REFLECT"
    assert "costo_decision" in info


def test_got_navigation_graph_builds_nodes_and_edges() -> None:
    """GoT debe construir un grafo explícito de pensamientos."""
    env = build_default_env(seed=103, max_steps=60)
    state = env.reset(seed=103)
    policy = GoTNavigationGraphPolicy(GoTNavigationConfig(chain_depth=2, tree_depth=1, max_tree_expansions=20))

    action, info = policy.select_action(env, state)

    assert_valid_action(action)
    assert info["metodo_razonamiento"] == "GoT Navigation Graph"
    assert info["cantidad_nodos"] >= 5.0
    assert info["cantidad_aristas"] >= 4.0
    assert "grafo_pensamientos" in info


def test_rlot_navigator_update_save_and_load(tmp_path) -> None:
    """RLoT debe seleccionar bloques, actualizar Q-table y persistir estado."""
    env = build_default_env(seed=104, max_steps=60)
    state = env.reset(seed=104)
    navigator = RLoTNavigator(RLoTNavigatorConfig(epsilon=0.0, seed=104, block_names=("ACT", "CHAIN", "REFLECT")))

    action, info = navigator.select_action(env, state, explore=False)
    assert_valid_action(action)
    assert info["metodo_razonamiento"] == "RL-of-Thoughts Navigator"

    next_state, reward, _done, _env_info = env.step(action)
    next_key = navigator.discretize_state(env, next_state)
    navigator.update_from_info(info, reward=reward, next_state_key=next_key)
    assert len(navigator.q_table) >= 1

    path = tmp_path / "rlot.json"
    navigator.save(path)

    loaded = RLoTNavigator(RLoTNavigatorConfig(epsilon=0.0, seed=104, block_names=("ACT", "CHAIN", "REFLECT")))
    loaded.load(path)
    assert loaded.q_table.keys() == navigator.q_table.keys()
