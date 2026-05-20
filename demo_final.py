"""Comando único para correr la demostración final presentable.

Este script genera los artefactos que se mostrarían en GitHub o en una demostracion oral:
GIF, capturas, CSV, replay determinista, heatmap, resumen de sobrepensamiento y
un reporte Markdown con el guion de presentación.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from gui_dashboard import DEFAULT_COMPARISON_METHODS, ReasoningDashboardController


@dataclass(frozen=True)
class DemoPreset:
    """Configuración recomendada para una demo reproducible."""

    scenario: str
    seed: int
    steps: int
    fps: int
    output: Path


PRESETS: dict[str, DemoPreset] = {
    "quick": DemoPreset("facil", 123, 35, 8, Path("results/final_demo_quick")),
    "presentation": DemoPreset("obstaculos_densos", 123, 120, 12, Path("results/final_demo")),
    "full": DemoPreset("obstaculos_densos", 777, 200, 15, Path("results/final_demo_full")),
}


def parse_args() -> argparse.Namespace:
    """Lee parámetros de la CLI."""
    parser = argparse.ArgumentParser(
        description="Genera una demostracion final reproducible: GIF, CSV, heatmap, replay y reporte Markdown.",
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        default="presentation",
        help="Perfil de ejecución. 'quick' es para validar; 'presentation' es para mostrar; 'full' produce más pasos.",
    )
    parser.add_argument("--scenario", default=None, help="Escenario a usar. Sobrescribe el preset.")
    parser.add_argument("--seed", type=int, default=None, help="Semilla determinista. Sobrescribe el preset.")
    parser.add_argument("--steps", type=int, default=None, help="Número de pasos del episodio. Sobrescribe el preset.")
    parser.add_argument("--fps", type=int, default=None, help="FPS del GIF. Sobrescribe el preset.")
    parser.add_argument("--output", type=Path, default=None, help="Directorio de salida. Sobrescribe el preset.")
    parser.add_argument(
        "--method",
        default="RL-of-Thoughts Navigator",
        help="Método base del dashboard para snapshot/replay.",
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        default=list(DEFAULT_COMPARISON_METHODS),
        help="Métodos de comparación lado a lado.",
    )
    parser.add_argument(
        "--mp4",
        action="store_true",
        help="También exporta MP4. Requiere imageio-ffmpeg en algunos entornos.",
    )
    return parser.parse_args()


def _resolve_config(args: argparse.Namespace) -> DemoPreset:
    preset = PRESETS[args.preset]
    return DemoPreset(
        scenario=args.scenario or preset.scenario,
        seed=int(args.seed if args.seed is not None else preset.seed),
        steps=int(args.steps if args.steps is not None else preset.steps),
        fps=int(args.fps if args.fps is not None else preset.fps),
        output=Path(args.output or preset.output),
    )


def _load_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []


def _format_artifact_list(artifacts: dict[str, Path]) -> str:
    lines = []
    for key, path in sorted(artifacts.items()):
        lines.append(f"- `{key}`: `{path.as_posix()}`")
    return "\n".join(lines)


def _best_efficiency_line(overthinking_rows: Iterable[dict[str, object]]) -> str:
    rows = list(overthinking_rows)
    finite_rows = [
        row
        for row in rows
        if isinstance(row.get("progreso_por_100_compute"), (int, float))
        and isinstance(row.get("indice_sobrepensamiento"), (int, float))
    ]
    if not finite_rows:
        return "No se pudo calcular una recomendación automática de eficiencia."
    best_progress = max(finite_rows, key=lambda row: float(row["progreso_por_100_compute"]))
    lowest_overthinking = min(finite_rows, key=lambda row: float(row["indice_sobrepensamiento"]))
    return (
        f"Mayor progreso por cómputo: **{best_progress['metodo']}** "
        f"({float(best_progress['progreso_por_100_compute']):.3g} progreso/100c). "
        f"Menor índice de sobrepensamiento: **{lowest_overthinking['metodo']}** "
        f"({float(lowest_overthinking['indice_sobrepensamiento']):.3g})."
    )


def write_demo_report(
    output: Path,
    config: DemoPreset,
    artifacts: dict[str, Path],
    methods: list[str],
    mp4_path: Path | None = None,
) -> Path:
    """Escribe un reporte Markdown listo para abrir durante la demo."""
    overthinking_path = output / "overthinking_summary.json"
    overthinking_rows = _load_json(overthinking_path)
    if not isinstance(overthinking_rows, list):
        overthinking_rows = []
    efficiency_line = _best_efficiency_line(overthinking_rows)  # type: ignore[arg-type]

    method_rows = "\n".join(
        f"| {row.get('metodo', 'desconocido')} | {row.get('pasos', '')} | "
        f"{float(row.get('recompensa_total', 0.0)):.2f} | "
        f"{float(row.get('costo_total', 0.0)):.2f} | "
        f"{float(row.get('progreso_por_100_compute', 0.0)):.3g} | "
        f"{row.get('advertencia', '')} |"
        for row in overthinking_rows
        if isinstance(row, dict)
    )
    if not method_rows:
        method_rows = "| Sin datos | - | - | - | - | - |"

    mp4_line = f"\n- MP4 opcional: `{mp4_path.as_posix()}`" if mp4_path is not None else ""
    report = f"""# Demo final reproducible

Esta carpeta fue generada automáticamente por `demo_final.py`.

## Configuración usada

| Campo | Valor |
|---|---|
| Escenario | `{config.scenario}` |
| Seed | `{config.seed}` |
| Pasos | `{config.steps}` |
| FPS | `{config.fps}` |
| Métodos | `{', '.join(methods)}` |

## Qué mostrar durante la presentación

1. Abrir `episode_comparison.gif` para enseñar la comparación lado a lado.
2. Abrir `figures/thought_heatmap.png` para explicar dónde el agente decidió gastar más cómputo.
3. Abrir `comparison_trace.csv` para mostrar que cada decisión queda registrada.
4. Abrir `overthinking_summary.csv` para discutir coste, progreso y sobrepensamiento.
5. Abrir `replay_config.json` para mostrar que el episodio es determinista y repetible.

## Lectura rápida de resultados

{efficiency_line}

| Método | Pasos | Recompensa | Costo total | Progreso/100c | Advertencia |
|---|---:|---:|---:|---:|---|
{method_rows}

## Artefactos generados

{_format_artifact_list(artifacts)}{mp4_line}

## Comandos para repetir esta demostracion

```bash
python -m pip install -r requirements-ci.txt
python demo_final.py --preset presentation
```

Para una validación rápida:

```bash
python demo_final.py --preset quick
```

Para abrir la interfaz interactiva con editor de mapas:

```bash
python -m pip install -r requirements-gui.txt
python gui_dashboard.py
```
"""
    path = output / "DEMO_REPORT.md"
    path.write_text(report, encoding="utf-8")
    return path


def main() -> None:
    """Ejecuta la demo final."""
    args = parse_args()
    config = _resolve_config(args)

    controller = ReasoningDashboardController(seed=config.seed)
    controller.set_scenario(config.scenario)
    controller.set_method(args.method)
    controller.enable_comparison(True, methods=args.methods)
    artifacts = controller.generate_paper_demo(
        output_dir=config.output,
        steps=config.steps,
        seed=config.seed,
        fps=config.fps,
    )

    mp4_path: Path | None = None
    if args.mp4:
        mp4_path = controller.record_episode(
            config.output / "episode_comparison.mp4",
            steps=config.steps,
            fps=config.fps,
            comparison=True,
        )
        artifacts["mp4"] = mp4_path

    report_path = write_demo_report(config.output, config, artifacts, list(args.methods), mp4_path)

    print("Demo final generada correctamente.")
    print(f"Directorio: {config.output}")
    print(f"Reporte: {report_path}")
    print(f"GIF principal: {artifacts['gif']}")
    print(f"CSV de trazas: {artifacts['csv']}")
    print(f"Replay determinista: {artifacts['replay']}")


if __name__ == "__main__":
    main()
