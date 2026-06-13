"""Analiza cuándo el agente gasta demasiado cómputo para el progreso logrado.

Entrada esperada: `comparison_trace.csv` generado por `generate_demo_artifacts.py`.
Salida: CSV/JSON con métricas por método para documentar coste, progreso y eficiencia.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Iterable

EPS = 1e-9


@dataclass(frozen=True)
class OverthinkingSummary:
    """Resumen por método de la relación entre cómputo y avance."""

    metodo: str
    pasos: int
    recompensa_total: float
    costo_total: float
    costo_promedio: float
    distancia_inicial: float
    distancia_final: float
    progreso_distancia: float
    recompensa_por_100_compute: float
    progreso_por_100_compute: float
    compute_por_progreso: float
    indice_sobrepensamiento: float
    advertencia: str


def _to_float(value: str | int | float | None, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def load_trace(path: str | Path) -> list[dict[str, str]]:
    """Carga una traza CSV de comparación."""
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def summarize_overthinking(rows: Iterable[dict[str, str]]) -> list[OverthinkingSummary]:
    """Calcula métricas de coste/progreso por método."""
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        method = row.get("metodo", "desconocido")
        grouped.setdefault(method, []).append(row)

    raw: list[dict[str, float | int | str]] = []
    for method, method_rows in grouped.items():
        ordered = sorted(method_rows, key=lambda row: int(_to_float(row.get("paso"))))
        first = ordered[0]
        last = ordered[-1]
        steps = max(int(_to_float(last.get("paso"))), 0)
        reward_total = _to_float(last.get("recompensa_total"))
        cost_total = _to_float(last.get("costo_total"))
        initial_distance = _to_float(first.get("distancia_a_meta"))
        final_distance = _to_float(last.get("distancia_a_meta"))
        progress = max(initial_distance - final_distance, 0.0)
        compute_per_progress = cost_total / max(progress, EPS) if progress > 0 else float("inf")
        raw.append(
            {
                "metodo": method,
                "pasos": steps,
                "recompensa_total": reward_total,
                "costo_total": cost_total,
                "costo_promedio": cost_total / max(steps, 1),
                "distancia_inicial": initial_distance,
                "distancia_final": final_distance,
                "progreso_distancia": progress,
                "recompensa_por_100_compute": 100.0 * reward_total / max(cost_total, EPS),
                "progreso_por_100_compute": 100.0 * progress / max(cost_total, EPS),
                "compute_por_progreso": compute_per_progress,
            }
        )

    finite_cpp = [
        float(item["compute_por_progreso"])
        for item in raw
        if item["compute_por_progreso"] != float("inf")
    ]
    baseline = median(finite_cpp) if finite_cpp else 1.0

    summaries: list[OverthinkingSummary] = []
    for item in raw:
        cpp = float(item["compute_por_progreso"])
        if cpp == float("inf"):
            index = float("inf")
            warning = "sin_progreso"
        else:
            index = cpp / max(baseline, EPS)
            if index >= 2.0:
                warning = "posible_sobrepensamiento"
            elif index <= 0.75:
                warning = "eficiente"
            else:
                warning = "normal"
        summaries.append(
            OverthinkingSummary(
                metodo=str(item["metodo"]),
                pasos=int(item["pasos"]),
                recompensa_total=float(item["recompensa_total"]),
                costo_total=float(item["costo_total"]),
                costo_promedio=float(item["costo_promedio"]),
                distancia_inicial=float(item["distancia_inicial"]),
                distancia_final=float(item["distancia_final"]),
                progreso_distancia=float(item["progreso_distancia"]),
                recompensa_por_100_compute=float(item["recompensa_por_100_compute"]),
                progreso_por_100_compute=float(item["progreso_por_100_compute"]),
                compute_por_progreso=cpp,
                indice_sobrepensamiento=float(index),
                advertencia=warning,
            )
        )
    return sorted(summaries, key=lambda item: item.metodo)


def write_summary_csv(summaries: Iterable[OverthinkingSummary], path: str | Path) -> Path:
    """Guarda resumen en CSV."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    summaries = list(summaries)
    with target.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = (
            list(asdict(summaries[0]).keys())
            if summaries
            else list(OverthinkingSummary.__dataclass_fields__.keys())
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            writer.writerow(asdict(summary))
    return target


def write_summary_json(summaries: Iterable[OverthinkingSummary], path: str | Path) -> Path:
    """Guarda resumen en JSON."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = [asdict(summary) for summary in summaries]
    target.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return target


def parse_args() -> argparse.Namespace:
    """Argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Analiza coste, progreso e índice de sobrepensamiento."
    )
    parser.add_argument(
        "--trace", type=Path, default=Path("results/paper_demo/comparison_trace.csv")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/paper_demo/overthinking_summary.csv")
    )
    parser.add_argument("--json", type=Path, default=None, help="Ruta opcional para salida JSON.")
    return parser.parse_args()


def main() -> None:
    """Ejecuta análisis desde CLI."""
    args = parse_args()
    summaries = summarize_overthinking(load_trace(args.trace))
    csv_path = write_summary_csv(summaries, args.output)
    print(f"Resumen CSV: {csv_path}")
    if args.json is not None:
        json_path = write_summary_json(summaries, args.json)
        print(f"Resumen JSON: {json_path}")
    for summary in summaries:
        print(
            f"- {summary.metodo}: índice={summary.indice_sobrepensamiento:.3g}, "
            f"progreso/100c={summary.progreso_por_100_compute:.3g}, advertencia={summary.advertencia}"
        )


if __name__ == "__main__":
    main()
