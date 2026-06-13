from __future__ import annotations

from baseline_env import build_default_env
from reproducibility import SeedPlan


def test_seed_plan_generates_stable_values() -> None:
    """El plan de semillas debe producir valores estables y separados."""
    first_plan = SeedPlan(123)
    second_plan = SeedPlan(123)

    assert first_plan.env_seed(1) == second_plan.env_seed(1)
    assert first_plan.reset_seed(1) == second_plan.reset_seed(1)
    assert first_plan.policy_seed() == second_plan.policy_seed()
    assert first_plan.controller_seed() == second_plan.controller_seed()
    assert first_plan.env_seed(1) != first_plan.reset_seed(1)


def test_sample_action_is_reproducible_with_same_seed() -> None:
    """Dos entornos con la misma semilla deben muestrear la misma secuencia."""
    first_env = build_default_env(seed=314, max_steps=20)
    second_env = build_default_env(seed=314, max_steps=20)

    first_actions = [first_env.sample_action() for _ in range(20)]
    second_actions = [second_env.sample_action() for _ in range(20)]

    assert first_actions == second_actions
