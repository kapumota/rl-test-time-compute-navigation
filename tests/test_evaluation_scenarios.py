"""Pruebas ligeras de Fase 4."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from evaluation_scenarios import build_scenario_env, list_scenarios
from run_evaluation_suite import evaluate_policy_on_scenario, save_csv, summarize


def test_scenarios_are_buildable() -> None:
    """Todos los escenarios estándar deben poder construirse y reiniciarse."""
    names = list_scenarios()
    assert set(names) == {"facil", "obstaculos_densos", "cambios_de_meta", "mapas_nunca_vistos", "sensores_ruidosos"}
    for name in names:
        instance = build_scenario_env(name, seed=7, max_steps=20, episode=1)
        state = instance.env.reset(options=instance.reset_options)
        assert state.shape == (4,)
        assert 0.0 <= float(instance.sand_density) <= 1.0


def test_noisy_sensor_observations_are_clipped() -> None:
    """El escenario de sensores ruidosos debe conservar observaciones dentro del rango esperado."""
    instance = build_scenario_env("sensores_ruidosos", seed=11, max_steps=20, episode=1)
    state = instance.env.reset(options=instance.reset_options)
    for _ in range(8):
        state, _reward, _done, _info = instance.env.step(instance.env.sample_action())
        assert np.all(state[1:4] >= 0.0)
        assert np.all(state[1:4] <= 1.0)


def test_evaluation_suite_quick_run(tmp_path: Path) -> None:
    """La suite debe producir filas y resumen con una política simple."""

    def policy(env, state):
        return 0, {"costo_decision": 1.0, "metodo_razonamiento": "prueba"}

    rows = evaluate_policy_on_scenario(
        method_name="Política de prueba",
        policy=policy,
        scenario_name="facil",
        eval_episodes=1,
        max_steps=5,
        seed=3,
    )
    summary = summarize(rows)
    assert len(rows) == 1
    assert len(summary) == 1
    assert summary[0]["escenario"] == "facil"

    target = tmp_path / "eval.csv"
    save_csv(rows, target)
    with target.open("r", encoding="utf-8") as file:
        loaded = list(csv.DictReader(file))
    assert loaded[0]["metodo"] == "Política de prueba"
