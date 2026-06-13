from __future__ import annotations

from pathlib import Path

EXPECTED_NEW_FIELDS = (
    "costo_decision_pasos_total",
    "costo_decision_pasos_promedio",
    "tiempo_decision_ms_total",
    "tiempo_decision_ms_promedio",
)

EXPECTED_LEGACY_FIELDS = (
    "costo_decision_total",
    "costo_decision_promedio",
)


def read_source(path: str) -> str:
    """Lee código fuente como texto para evitar dependencias opcionales."""
    return Path(path).read_text(encoding="utf-8")


def test_reasoning_exports_new_and_legacy_decision_metrics() -> None:
    """La evaluación de reasoning debe exponer métricas nuevas y antiguas."""
    source = read_source("run_reasoning_experiments.py")

    assert "from decision_metrics import build_decision_metrics, measure_decision" in source
    assert "measurement = measure_decision(" in source
    for field in EXPECTED_NEW_FIELDS:
        assert field in source
    for field in EXPECTED_LEGACY_FIELDS:
        assert field in source


def test_evaluation_suite_exports_new_and_legacy_decision_metrics() -> None:
    """La suite de evaluación debe exponer métricas nuevas y antiguas."""
    source = read_source("run_evaluation_suite.py")

    assert "from decision_metrics import build_decision_metrics, measure_decision" in source
    assert "measurement = measure_decision(" in source
    for field in EXPECTED_NEW_FIELDS:
        assert field in source
    for field in EXPECTED_LEGACY_FIELDS:
        assert field in source


def test_baselines_exports_new_and_legacy_decision_metrics_without_importing_torch() -> None:
    """Los baselines deben exponer métricas sin requerir importación de PyTorch."""
    source = read_source("run_baselines.py")

    assert "from decision_metrics import build_decision_metrics, measure_decision" in source
    assert "measurement = measure_decision(" in source
    for field in EXPECTED_NEW_FIELDS:
        assert field in source
    for field in EXPECTED_LEGACY_FIELDS:
        assert field in source


def test_dashboard_exports_event_and_summary_decision_metrics() -> None:
    """El dashboard debe exportar métricas nuevas y conservar compatibilidad."""
    source = read_source("gui_dashboard.py")

    assert "from decision_metrics import build_decision_metrics, measure_decision" in source
    assert "measurement = measure_decision(" in source
    assert "costo_decision_pasos" in source
    assert "tiempo_decision_ms" in source
    assert "costo_decision" in source
    assert "costo_total" in source
    for field in EXPECTED_NEW_FIELDS:
        assert field in source


def test_decision_metrics_module_keeps_legacy_cost_field() -> None:
    """El módulo base debe mantener costo_decision para compatibilidad."""
    source = read_source("decision_metrics.py")

    assert '"costo_decision_pasos"' in source
    assert '"tiempo_decision_ms"' in source
    assert '"costo_decision"' in source
