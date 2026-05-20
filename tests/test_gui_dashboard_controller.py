"""Pruebas de Fase 6 para el controlador del dashboard Kivy."""

from pathlib import Path

from gui_dashboard import ReasoningDashboardController, SUBTITULO_INTERFAZ, TITULO_INTERFAZ


def test_dashboard_controller_step() -> None:
    """El controlador debe ejecutar pasos sin depender de Kivy."""
    controller = ReasoningDashboardController(seed=7, max_steps=30)
    assert TITULO_INTERFAZ.startswith("¿Cuándo")
    assert "Kivy" in SUBTITULO_INTERFAZ
    assert controller.state.escenario in controller.scenarios
    assert controller.state.metodo in controller.policies

    state = controller.step()
    assert state.pasos >= 1
    assert state.costo_total > 0.0
    assert isinstance(controller.metrics_text(), str)
    assert "Recompensa total" in controller.metrics_text()


def test_dashboard_controller_switches() -> None:
    """El controlador debe cambiar escenario y método con validación."""
    controller = ReasoningDashboardController(seed=11, max_steps=20)
    controller.set_scenario("sensores_ruidosos")
    assert controller.state.escenario == "sensores_ruidosos"

    controller.set_method("Adaptive rollout budget")
    assert controller.state.metodo == "Adaptive rollout budget"
    controller.step()
    assert controller.state.metodo_razonamiento


def test_dashboard_snapshot_export(tmp_path: Path) -> None:
    """El dashboard debe exportar un snapshot JSON reproducible."""
    controller = ReasoningDashboardController(seed=13, max_steps=20)
    controller.step()
    output = controller.export_snapshot(tmp_path / "snapshot.json")
    assert output.exists()
    text = output.read_text(encoding="utf-8")
    assert "grafo_pensamientos" in text
    assert "recompensa_total" in text
