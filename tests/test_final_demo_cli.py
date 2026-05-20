"""Pruebas del comando único de demo final."""

from pathlib import Path

from demo_final import PRESETS, write_demo_report


def test_final_demo_presets_are_available() -> None:
    """La demo final debe ofrecer perfiles para validación y presentación."""
    assert {"quick", "presentation", "full"}.issubset(PRESETS)
    assert PRESETS["presentation"].steps >= PRESETS["quick"].steps
    assert PRESETS["presentation"].output == Path("results/final_demo")


def test_write_demo_report(tmp_path: Path) -> None:
    """El reporte final debe documentar artefactos y comandos de repetición."""
    output = tmp_path / "demo"
    output.mkdir()
    (output / "overthinking_summary.json").write_text(
        """
        [
          {
            "metodo": "RL-of-Thoughts Navigator",
            "pasos": 3,
            "recompensa_total": 1.5,
            "costo_total": 12.0,
            "progreso_por_100_compute": 4.2,
            "indice_sobrepensamiento": 0.7,
            "advertencia": "eficiente"
          }
        ]
        """,
        encoding="utf-8",
    )
    artifacts = {"gif": output / "episode_comparison.gif", "csv": output / "comparison_trace.csv"}
    report = write_demo_report(output, PRESETS["quick"], artifacts, ["RL-of-Thoughts Navigator"])
    text = report.read_text(encoding="utf-8")
    assert "Demo final reproducible" in text
    assert "python demo_final.py --preset presentation" in text
    assert "RL-of-Thoughts Navigator" in text
