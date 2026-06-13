from __future__ import annotations

from baseline_env import build_default_env
from reasoning_policies import random_policy
from reproducibility import SeedPlan
from run_reasoning_experiments import build_policies, evaluate_policy


def strip_real_time_metrics(
    rows: list[dict[str, float | str]],
) -> list[dict[str, float | str]]:
    """Retira métricas de tiempo real para comparar trazas deterministas."""
    return [
        {key: value for key, value in row.items() if not key.startswith("tiempo_decision_ms")}
        for row in rows
    ]


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


def test_random_policy_trace_is_reproducible_with_same_seed() -> None:
    """La política aleatoria debe generar la misma traza lógica con la misma semilla."""
    first_rows = evaluate_policy(
        "Política aleatoria",
        random_policy,
        eval_episodes=2,
        max_steps=15,
        seed=77,
    )
    second_rows = evaluate_policy(
        "Política aleatoria",
        random_policy,
        eval_episodes=2,
        max_steps=15,
        seed=77,
    )

    assert strip_real_time_metrics(first_rows) == strip_real_time_metrics(second_rows)


def test_rlot_trace_is_reproducible_with_same_seed() -> None:
    """RLoT debe producir la misma traza lógica con la misma semilla."""
    first_policy = build_policies(seed=2024)["RL-of-Thoughts Navigator"]
    second_policy = build_policies(seed=2024)["RL-of-Thoughts Navigator"]

    first_rows = evaluate_policy(
        "RL-of-Thoughts Navigator",
        first_policy,
        eval_episodes=2,
        max_steps=20,
        seed=2024,
    )
    second_rows = evaluate_policy(
        "RL-of-Thoughts Navigator",
        second_policy,
        eval_episodes=2,
        max_steps=20,
        seed=2024,
    )

    assert strip_real_time_metrics(first_rows) == strip_real_time_metrics(second_rows)


def test_reasoning_rows_include_decision_metrics() -> None:
    """Las filas de reasoning deben incluir costo lógico y tiempo real."""
    rows = evaluate_policy(
        "Política aleatoria",
        random_policy,
        eval_episodes=1,
        max_steps=10,
        seed=101,
    )

    row = rows[0]

    assert "costo_decision_pasos_total" in row
    assert "costo_decision_pasos_promedio" in row
    assert "tiempo_decision_ms_total" in row
    assert "tiempo_decision_ms_promedio" in row
    assert "costo_decision_total" in row
    assert "costo_decision_promedio" in row

    assert float(row["costo_decision_pasos_total"]) >= 0.0
    assert float(row["tiempo_decision_ms_total"]) >= 0.0
