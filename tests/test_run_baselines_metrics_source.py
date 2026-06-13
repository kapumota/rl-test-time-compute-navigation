from __future__ import annotations

from pathlib import Path


def test_run_baselines_contains_decision_metrics_without_importing_torch() -> None:
    """El script de baselines debe integrar métricas sin requerir importarlo."""
    source = Path("run_baselines.py").read_text(encoding="utf-8")

    assert "from decision_metrics import build_decision_metrics, measure_decision" in source
    assert "measurement = measure_decision(" in source
    assert "costo_decision_pasos_total" in source
    assert "costo_decision_pasos_promedio" in source
    assert "tiempo_decision_ms_total" in source
    assert "tiempo_decision_ms_promedio" in source
    assert "costo_decision_total" in source
    assert "costo_decision_promedio" in source
