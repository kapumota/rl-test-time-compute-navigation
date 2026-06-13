"""
Proyecto: ¿Cuándo debe pensar un agente RL?
Entrenamiento ligero del Learned reasoning controller.

El controlador no aprende a conducir directamente. Aprende a escoger entre bloques de razonamiento:
acción directa, Best-of-N, Tree-of-Actions y Graph-of-Waypoints.

Uso rápido:
    python train_reasoning_controller.py --episodes 20 --max-steps 200
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

from baseline_env import TITULO_PROYECTO, build_default_env
from reproducibility import SeedPlan, set_global_seed
from reasoning_policies import (
    LearnedReasoningController,
    ReasoningControllerConfig,
    build_reasoning_strategies,
    normalize_policy_result,
)


def save_csv(rows: List[Dict[str, float | str]], path: Path) -> None:
    """Guarda métricas de entrenamiento en CSV."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def train_controller(
    episodes: int,
    max_steps: int,
    seed: int,
    output_path: Path,
    metrics_path: Path,
) -> LearnedReasoningController:
    """Entrena el controlador como un bandido contextual sobre recompensas reales."""
    set_global_seed(seed)
    seed_plan = SeedPlan(seed)
    strategies = build_reasoning_strategies()
    controller = LearnedReasoningController(
        strategies=strategies,
        config=ReasoningControllerConfig(epsilon=0.20, learning_rate=0.03, seed=seed),
    )

    rows: List[Dict[str, float | str]] = []

    for episode in range(1, episodes + 1):
        env = build_default_env(seed=seed_plan.env_seed(episode), max_steps=max_steps)
        state = env.reset(seed=seed_plan.reset_seed(episode))
        total_reward = 0.0
        reached_goal = False
        strategy_counts: Dict[str, int] = {}
        decision_cost_total = 0.0

        for _ in range(max_steps):
            trace = controller.select_strategy(env, state, explore=True)
            action, info = normalize_policy_result(strategies[trace.strategy](env, state))
            next_state, reward, done, step_info = env.step(action)

            # Recompensa moldeada: se premia avanzar hacia la meta y se descuenta el costo de razonar.
            decision_cost = float(info.get("costo_decision", 1.0))
            shaped_reward = float(reward) - 0.001 * decision_cost
            if bool(step_info.get("meta_alcanzada", False)):
                shaped_reward += 2.0
            if bool(step_info.get("sobre_arena", False)) or bool(
                step_info.get("colision_borde", False)
            ):
                shaped_reward -= 0.25

            controller.update(trace, shaped_reward)
            total_reward += float(reward)
            decision_cost_total += decision_cost
            strategy_counts[trace.strategy] = strategy_counts.get(trace.strategy, 0) + 1
            reached_goal = bool(step_info.get("meta_alcanzada", False))
            state = next_state

            if done:
                break

        rows.append(
            {
                "episodio": float(episode),
                "recompensa": float(total_reward),
                "pasos": float(env.step_count),
                "meta_alcanzada": float(reached_goal),
                "costo_decision_total": float(decision_cost_total),
                "conteo_estrategias": str(strategy_counts),
            }
        )
        if episode == 1 or episode % 5 == 0:
            print(
                f"Controlador | episodio {episode:04d} | "
                f"recompensa={total_reward:8.2f} | pasos={env.step_count:4d} | estrategias={strategy_counts}"
            )

    controller.save(output_path)
    save_csv(rows, metrics_path)
    print(f"Controlador guardado en: {output_path}")
    print(f"Métricas guardadas en: {metrics_path}")
    return controller


def parse_args() -> argparse.Namespace:
    """Lee argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Entrena el controlador aprendido de razonamiento."
    )
    parser.add_argument("--episodes", type=int, default=20, help="Episodios de entrenamiento.")
    parser.add_argument("--max-steps", type=int, default=200, help="Pasos máximos por episodio.")
    parser.add_argument("--seed", type=int, default=123, help="Semilla base.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/reasoning_controller.json"),
        help="Archivo JSON de pesos.",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("results/reasoning_controller_train.csv"),
        help="CSV de entrenamiento.",
    )
    return parser.parse_args()


def main() -> None:
    """Ejecuta entrenamiento del controlador."""
    args = parse_args()
    set_global_seed(args.seed)
    print(TITULO_PROYECTO)
    print("Entrenamiento del controlador aprendido de razonamiento")
    train_controller(args.episodes, args.max_steps, args.seed, args.output, args.metrics)


if __name__ == "__main__":
    main()
