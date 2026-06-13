"""
Proyecto: ¿Cuándo debe pensar un agente RL?
Fase 5: experimento focalizado en GoT y RLoT.

Este script compara únicamente las políticas nuevas contra una política directa.
Para evaluaciones robustas por escenarios, usa run_evaluation_suite.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from run_reasoning_experiments import main as run_reasoning_main


def parse_args() -> argparse.Namespace:
    """Lee argumentos de línea de comandos y los transforma al formato común."""
    parser = argparse.ArgumentParser(description="Ejecuta una comparación rápida de Fase 5.")
    parser.add_argument("--eval-episodes", type=int, default=3, help="Episodios por método.")
    parser.add_argument("--max-steps", type=int, default=200, help="Máximo de pasos por episodio.")
    parser.add_argument("--seed", type=int, default=123, help="Semilla base.")
    parser.add_argument(
        "--results", type=Path, default=Path("results/rlot_got_eval.csv"), help="CSV por episodio."
    )
    parser.add_argument(
        "--summary", type=Path, default=Path("results/rlot_got_summary.csv"), help="CSV agregado."
    )
    return parser.parse_args()


def main() -> None:
    """Reutiliza el evaluador común con los métodos de Fase 5."""
    args = parse_args()
    import sys

    sys.argv = [
        "run_reasoning_experiments.py",
        "--eval-episodes",
        str(args.eval_episodes),
        "--max-steps",
        str(args.max_steps),
        "--seed",
        str(args.seed),
        "--methods",
        "Acción directa geométrica",
        "GoT Navigation Graph",
        "RL-of-Thoughts Navigator",
        "--results",
        str(args.results),
        "--summary",
        str(args.summary),
    ]
    run_reasoning_main()


if __name__ == "__main__":
    main()
