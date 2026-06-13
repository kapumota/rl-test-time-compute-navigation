"""Pruebas para catálogo de investigación y análisis de sobrepensamiento."""

from pathlib import Path

from analyze_overthinking import summarize_overthinking, write_summary_csv
from research_catalog import (
    RESEARCH_CATALOG,
    markdown_table,
    references_by_priority,
    references_by_status,
)


def test_research_catalog_has_core_references() -> None:
    """El catálogo debe distinguir lo implementado del roadmap."""
    keys = {reference.key for reference in RESEARCH_CATALOG}
    assert {"rlot", "got", "tot", "agentthink", "rest-rl"}.issubset(keys)
    assert len(references_by_status("implementado")) >= 4
    assert len(references_by_priority("alta")) >= 5
    table = markdown_table()
    assert "RL of Thoughts" in table
    assert "Graph of Thoughts" in table


def test_overthinking_summary_flags_high_compute() -> None:
    """Un método con mucho más coste por el mismo progreso debe quedar marcado."""
    rows = [
        {
            "paso": "0",
            "metodo": "barato",
            "distancia_a_meta": "100",
            "recompensa_total": "0",
            "costo_total": "0",
        },
        {
            "paso": "1",
            "metodo": "barato",
            "distancia_a_meta": "50",
            "recompensa_total": "1",
            "costo_total": "10",
        },
        {
            "paso": "0",
            "metodo": "caro",
            "distancia_a_meta": "100",
            "recompensa_total": "0",
            "costo_total": "0",
        },
        {
            "paso": "1",
            "metodo": "caro",
            "distancia_a_meta": "50",
            "recompensa_total": "1",
            "costo_total": "100",
        },
    ]
    summaries = {summary.metodo: summary for summary in summarize_overthinking(rows)}
    assert summaries["caro"].indice_sobrepensamiento > summaries["barato"].indice_sobrepensamiento
    assert summaries["caro"].progreso_por_100_compute < summaries["barato"].progreso_por_100_compute


def test_write_overthinking_csv(tmp_path: Path) -> None:
    """El análisis debe exportarse como CSV reutilizable en el modo paper/demo."""
    summaries = summarize_overthinking(
        [
            {
                "paso": "0",
                "metodo": "A",
                "distancia_a_meta": "100",
                "recompensa_total": "0",
                "costo_total": "0",
            },
            {
                "paso": "2",
                "metodo": "A",
                "distancia_a_meta": "80",
                "recompensa_total": "2",
                "costo_total": "5",
            },
        ]
    )
    path = write_summary_csv(summaries, tmp_path / "overthinking.csv")
    assert path.exists()
    assert "indice_sobrepensamiento" in path.read_text(encoding="utf-8")
