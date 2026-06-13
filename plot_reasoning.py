"""
Proyecto: ¿Cuándo debe pensar un agente RL?
Gráficas de Fase 3 a partir del resumen de razonamiento.

Uso:
    python plot_reasoning.py --summary results/reasoning_summary.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


def read_rows(path: Path) -> List[Dict[str, str]]:
    """Lee un CSV de resumen."""
    with path.open("r", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def plot_metric(rows: List[Dict[str, str]], metric: str, title: str, output: Path) -> None:
    """Grafica una métrica agregada por método."""
    methods = [row["metodo"] for row in rows]
    values = [float(row[f"{metric}_media"]) for row in rows]

    output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(11, 5))
    plt.bar(methods, values)
    plt.title(title)
    plt.ylabel(metric.replace("_", " "))
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


def parse_args() -> argparse.Namespace:
    """Lee argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Grafica resultados de razonamiento en inferencia."
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("results/reasoning_summary.csv"),
        help="CSV agregado de Fase 3.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/figures"), help="Carpeta de salida."
    )
    return parser.parse_args()


def main() -> None:
    """Genera gráficas principales."""
    args = parse_args()
    rows = read_rows(args.summary)
    plot_metric(
        rows,
        "recompensa",
        "Recompensa promedio por método",
        args.output_dir / "reasoning_recompensa.png",
    )
    plot_metric(
        rows, "meta_alcanzada", "Tasa de éxito por método", args.output_dir / "reasoning_exito.png"
    )
    plot_metric(
        rows,
        "costo_decision_promedio",
        "Costo promedio por decisión",
        args.output_dir / "reasoning_costo.png",
    )
    print(f"Gráficas guardadas en: {args.output_dir}")


if __name__ == "__main__":
    main()
