from __future__ import annotations

from gui_dashboard import ReasoningDashboardController


def test_dashboard_single_agent_records_decision_metrics() -> None:
    """El dashboard debe registrar costo lógico y tiempo real por decisión."""
    controller = ReasoningDashboardController(seed=41, max_steps=20)
    controller.set_method("Acción directa geométrica")

    state = controller.step()
    event = controller.episode_log[-1]

    assert state.costo_decision_pasos_total >= 0.0
    assert state.costo_decision_pasos_promedio >= 0.0
    assert state.tiempo_decision_ms_total >= 0.0
    assert state.tiempo_decision_ms_promedio >= 0.0

    assert "costo_decision" in event
    assert "costo_decision_pasos" in event
    assert "tiempo_decision_ms" in event
    assert float(event["costo_decision_pasos"]) >= 0.0
    assert float(event["tiempo_decision_ms"]) >= 0.0


def test_dashboard_comparison_summary_records_decision_metrics() -> None:
    """La comparación lado a lado debe exponer las métricas nuevas."""
    controller = ReasoningDashboardController(seed=42, max_steps=20)
    controller.enable_comparison(
        True,
        methods=("Acción directa geométrica",),
    )

    controller.step_comparison()
    summary = controller.get_comparison_summary()[0]

    assert "costo_decision_pasos_total" in summary
    assert "costo_decision_pasos_promedio" in summary
    assert "tiempo_decision_ms_total" in summary
    assert "tiempo_decision_ms_promedio" in summary
    assert float(summary["costo_decision_pasos_total"]) >= 0.0
    assert float(summary["tiempo_decision_ms_total"]) >= 0.0
