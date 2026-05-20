"""Pruebas de humo para scripts principales."""

from __future__ import annotations

from pathlib import Path

from run_reasoning_experiments import build_policies, evaluate_policy, summarize


def test_reasoning_experiment_smoke(tmp_path: Path) -> None:
    """Evalúa un método de Fase 3 durante pocos pasos para detectar errores de integración."""
    policies = build_policies(controller_path=tmp_path / "no_existe.json")
    rows = evaluate_policy(
        "Best-of-N actions",
        policies["Best-of-N actions"],
        eval_episodes=1,
        max_steps=12,
        seed=123,
    )
    summary = summarize(rows)

    assert len(rows) == 1
    assert len(summary) == 1
    assert summary[0]["metodo"] == "Best-of-N actions"
