"""Pruebas de Fase 7 para comparación, heatmap, editor y replay."""

from pathlib import Path

import numpy as np

from gui_dashboard import DEFAULT_COMPARISON_METHODS, ReasoningDashboardController


def test_side_by_side_comparison_and_heatmap(tmp_path: Path) -> None:
    """El controlador debe comparar RLoT, GoT y Adaptive Budget en el mismo mapa."""
    controller = ReasoningDashboardController(seed=21, max_steps=25)
    controller.set_scenario("facil")
    controller.enable_comparison(True, DEFAULT_COMPARISON_METHODS)

    controller.step()
    summary = controller.get_comparison_summary()
    assert [row["metodo"] for row in summary] == list(DEFAULT_COMPARISON_METHODS)
    assert all(row["pasos"] == 1 for row in summary)
    assert len(controller.get_heatmap_cells(threshold=0.0)) >= 1

    csv_path = controller.export_comparison_csv(tmp_path / "comparison.csv")
    assert csv_path.exists()
    assert "RL-of-Thoughts Navigator" in csv_path.read_text(encoding="utf-8")


def test_map_editor_and_replay_roundtrip(tmp_path: Path) -> None:
    """El editor debe modificar el mapa y el replay debe preservar seed y arena."""
    controller = ReasoningDashboardController(seed=33, max_steps=20)
    controller.set_replay_seed(33, scenario_name="facil", method_name="Adaptive rollout budget")
    controller.paint_obstacle_at(200.0, 210.0, radius=6)
    assert float(controller.env.sand[200, 210]) == 1.0

    replay_path = controller.export_replay_config(tmp_path / "replay.json")
    loaded = ReasoningDashboardController(seed=1, max_steps=5)
    loaded.load_replay_config(replay_path)
    assert loaded.seed == 33
    assert loaded.state.escenario == "facil"
    assert loaded.state.metodo == "Adaptive rollout budget"
    assert np.array_equal((loaded.env.sand > 0), (controller.env.sand > 0))


def test_render_frame_without_kivy() -> None:
    """El render de frame para grabación debe funcionar sin abrir Kivy."""
    controller = ReasoningDashboardController(seed=44, max_steps=10)
    controller.step()
    frame = controller.render_frame()
    assert frame.ndim == 3
    assert frame.shape[2] == 3
    assert frame.dtype == np.uint8


def test_generate_paper_demo_includes_overthinking_summary(tmp_path: Path) -> None:
    """El modo paper/demo debe producir análisis de sobrepensamiento junto al GIF y las figuras."""
    controller = ReasoningDashboardController(seed=55, max_steps=10)
    controller.set_scenario("facil")
    artifacts = controller.generate_paper_demo(output_dir=tmp_path / "demo", steps=3, seed=55, fps=4)
    assert artifacts["gif"].exists()
    assert artifacts["overthinking_csv"].exists()
    assert artifacts["overthinking_json"].exists()
    assert "indice_sobrepensamiento" in artifacts["overthinking_csv"].read_text(encoding="utf-8")
