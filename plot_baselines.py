"""
Proyecto: ¿Cuándo debe pensar un agente RL?
Gráficas simples para comparar baselines de la Fase 2.

Uso:
    python plot_baselines.py --summary results/baselines_summary.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


def load_rows(path: Path) -> List[Dict[str, str]]:
    """Carga un CSV de resumen."""
    with path.open("r", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def plot_metric(rows: List[Dict[str, str]], metric: str, ylabel: str, output: Path) -> None:
    """Crea una gráfica de barras para una métrica agregada."""
    labels = [row["baseline"] for row in rows]
    values = [float(row[metric]) for row in rows]

    plt.figure(figsize=(10, 5))
    plt.bar(labels, values)
    plt.ylabel(ylabel)
    plt.title("Comparación de baselines")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=160)
    plt.close()


def parse_args() -> argparse.Namespace:
    """Lee argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description="Genera gráficas para los baselines de la Fase 2.")
    parser.add_argument("--summary", type=Path, default=Path("results/baselines_summary.csv"), help="CSV agregado de baselines.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/figures"), help="Carpeta de salida para figuras.")
    return parser.parse_args()


def main() -> None:
    """Genera las figuras principales."""
    args = parse_args()
    rows = load_rows(args.summary)
    plot_metric(rows, "recompensa_media", "Recompensa promedio", args.output_dir / "recompensa_promedio.png")
    plot_metric(rows, "meta_alcanzada_media", "Tasa de éxito", args.output_dir / "tasa_exito.png")
    plot_metric(rows, "pasos_media", "Pasos promedio", args.output_dir / "pasos_promedio.png")
    plot_metric(rows, "costo_decision_promedio_media", "Costo promedio por decisión", args.output_dir / "costo_decision.png")
    print(f"Figuras guardadas en: {args.output_dir}")


if __name__ == "__main__":
    main()
