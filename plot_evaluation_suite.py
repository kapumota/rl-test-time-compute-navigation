"""
Proyecto: ¿Cuándo debe pensar un agente RL?
Fase 4: gráficas de evaluación por escenario.

Uso:
    python plot_evaluation_suite.py --summary results/evaluation_summary.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List


def read_csv(path: Path) -> List[Dict[str, str]]:
    """Carga un CSV como lista de diccionarios."""
    with path.open("r", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def plot_metric(rows: List[Dict[str, str]], metric: str, ylabel: str, output: Path) -> None:
    """Genera una gráfica de barras agrupada por escenario."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Instala matplotlib para generar gráficas de Fase 4.") from exc

    scenarios = sorted({row["escenario_visible"] for row in rows})
    methods = sorted({row["metodo"] for row in rows})
    values = {
        (row["escenario_visible"], row["metodo"]): float(row[metric])
        for row in rows
    }

    x = np.arange(len(scenarios))
    width = 0.8 / max(len(methods), 1)

    fig, ax = plt.subplots(figsize=(max(10, len(scenarios) * 2.2), 5.5))
    for index, method in enumerate(methods):
        offsets = x - 0.4 + width / 2 + index * width
        method_values = [values.get((scenario, method), 0.0) for scenario in scenarios]
        ax.bar(offsets, method_values, width, label=method)

    ax.set_title(f"Fase 4 - {ylabel}")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=20, ha="right")
    ax.legend(fontsize=8)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Lee argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description="Grafica resultados agregados de Fase 4.")
    parser.add_argument("--summary", type=Path, default=Path("results/evaluation_summary.csv"), help="CSV agregado.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/figures"), help="Directorio de salida.")
    return parser.parse_args()


def main() -> None:
    """Genera las gráficas principales."""
    args = parse_args()
    rows = read_csv(args.summary)
    metrics = [
        ("meta_alcanzada_media", "Tasa de éxito", "fase4_exito.png"),
        ("recompensa_media", "Recompensa promedio", "fase4_recompensa.png"),
        ("distancia_final_media", "Distancia final promedio", "fase4_distancia_final.png"),
        ("costo_decision_promedio_media", "Costo promedio por decisión", "fase4_costo_decision.png"),
    ]
    for metric, label, filename in metrics:
        plot_metric(rows, metric, label, args.output_dir / filename)
    print(f"Gráficas guardadas en: {args.output_dir}")


if __name__ == "__main__":
    main()
