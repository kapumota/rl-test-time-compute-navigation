"""Genera artefactos paper/demostraciones sin abrir la interfaz Kivy."""

from __future__ import annotations

import argparse
from pathlib import Path

from gui_dashboard import DEFAULT_COMPARISON_METHODS, ReasoningDashboardController


def parse_args() -> argparse.Namespace:
    """Lee argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Genera capturas, CSV, GIF y figuras del dashboard."
    )
    parser.add_argument("--scenario", default="obstaculos_densos", help="Escenario de evaluación.")
    parser.add_argument(
        "--method", default="RL-of-Thoughts Navigator", help="Método base del dashboard."
    )
    parser.add_argument("--seed", type=int, default=123, help="Semilla determinista de replay.")
    parser.add_argument("--steps", type=int, default=120, help="Pasos del episodio demo.")
    parser.add_argument("--fps", type=int, default=12, help="Frames por segundo para GIF.")
    parser.add_argument(
        "--output", type=Path, default=Path("results/paper_demo"), help="Directorio de salida."
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        default=list(DEFAULT_COMPARISON_METHODS),
        help="Métodos del modo comparación lado a lado.",
    )
    return parser.parse_args()


def main() -> None:
    """Ejecuta la generación de artefactos."""
    args = parse_args()
    controller = ReasoningDashboardController(seed=args.seed)
    controller.set_scenario(args.scenario)
    controller.set_method(args.method)
    controller.enable_comparison(True, methods=args.methods)
    artifacts = controller.generate_paper_demo(
        output_dir=args.output, steps=args.steps, seed=args.seed, fps=args.fps
    )
    print("Artefactos generados:")
    for name, path in artifacts.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
