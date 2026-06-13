"""
Proyecto: ¿Cuándo debe pensar un agente RL?
Fase 3: comparación de métodos de razonamiento en inferencia.

Este script evalúa métodos de razonamiento y navegación adaptativa:
- Best-of-N actions.
- Tree-of-Actions.
- Graph-of-Waypoints.
- Adaptive rollout budget.
- Learned reasoning controller.
- GoT Navigation Graph.
- RL-of-Thoughts Navigator.

Uso rápido:
    python run_reasoning_experiments.py --eval-episodes 5 --max-steps 250
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from astar_planner import AStarConfig, AStarPlanner
from baseline_env import TITULO_PROYECTO, build_default_env
from reasoning_policies import (
    AdaptiveBudgetConfig,
    AdaptiveRolloutBudget,
    BestOfNActions,
    GraphOfWaypoints,
    LearnedReasoningController,
    ReasoningControllerConfig,
    RolloutConfig,
    TreeOfActions,
    TreeSearchConfig,
    WaypointGraphConfig,
    build_reasoning_strategies,
    greedy_goal_policy,
    normalize_policy_result,
)
from rlot_got_navigation import build_rlot_and_got_policies
from reproducibility import SeedPlan, set_global_seed

PolicyCallable = Callable[[Any, np.ndarray], int | Tuple[int, Dict[str, Any]]]


def evaluate_policy(
    name: str,
    policy: PolicyCallable,
    eval_episodes: int,
    max_steps: int,
    seed: int,
) -> List[Dict[str, float | str]]:
    """Evalúa una política y devuelve métricas por episodio."""
    rows: List[Dict[str, float | str]] = []
    seed_plan = SeedPlan(seed)

    for episode in range(1, eval_episodes + 1):
        env = build_default_env(seed=seed_plan.env_seed(episode), max_steps=max_steps)
        state = env.reset(seed=seed_plan.reset_seed(episode))
        total_reward = 0.0
        sand_steps = 0
        border_collisions = 0
        reached_goal = False
        decision_cost_total = 0.0
        reasoning_counts: Dict[str, int] = {}

        for _ in range(max_steps):
            action, decision_info = normalize_policy_result(policy(env, state))
            decision_cost_total += float(decision_info.get("costo_decision", 1.0))
            method = str(decision_info.get("metodo_razonamiento", name))
            reasoning_counts[method] = reasoning_counts.get(method, 0) + 1

            next_state, reward, done, info = env.step(action)
            total_reward += float(reward)
            sand_steps += int(info["sobre_arena"])
            border_collisions += int(info["colision_borde"])
            reached_goal = bool(info["meta_alcanzada"])
            state = next_state

            if done:
                break

        steps = max(env.step_count, 1)
        rows.append(
            {
                "metodo": name,
                "episodio": float(episode),
                "recompensa": float(total_reward),
                "pasos": float(env.step_count),
                "meta_alcanzada": float(reached_goal),
                "pasos_en_arena": float(sand_steps),
                "colisiones_borde": float(border_collisions),
                "costo_decision_total": float(decision_cost_total),
                "costo_decision_promedio": float(decision_cost_total / steps),
                "conteo_razonamiento": str(reasoning_counts),
            }
        )

    return rows


def summarize(rows: List[Dict[str, float | str]]) -> List[Dict[str, float | str]]:
    """Agrega métricas promedio por método."""
    methods = sorted({str(row["metodo"]) for row in rows})
    numeric_keys = [
        "recompensa",
        "pasos",
        "meta_alcanzada",
        "pasos_en_arena",
        "colisiones_borde",
        "costo_decision_total",
        "costo_decision_promedio",
    ]
    summary: List[Dict[str, float | str]] = []

    for method in methods:
        subset = [row for row in rows if row["metodo"] == method]
        item: Dict[str, float | str] = {"metodo": method, "episodios": float(len(subset))}
        for key in numeric_keys:
            values = [float(row[key]) for row in subset]
            item[f"{key}_media"] = float(np.mean(values))
            item[f"{key}_desv"] = float(np.std(values))
        summary.append(item)
    return summary


def save_csv(rows: List[Dict[str, float | str]], path: Path) -> None:
    """Guarda filas en formato CSV."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_summary(summary_rows: List[Dict[str, float | str]]) -> None:
    """Imprime un resumen compacto de resultados."""
    print("\nResumen de razonamiento en inferencia")
    print("-" * 108)
    print(
        f"{'método':<32} {'recompensa':>12} {'éxito':>10} {'pasos':>10} {'arena':>10} {'costo/dec':>12}"
    )
    print("-" * 108)
    for row in summary_rows:
        print(
            f"{str(row['metodo']):<32} "
            f"{float(row['recompensa_media']):12.2f} "
            f"{float(row['meta_alcanzada_media']):10.2f} "
            f"{float(row['pasos_media']):10.2f} "
            f"{float(row['pasos_en_arena_media']):10.2f} "
            f"{float(row['costo_decision_promedio_media']):12.2f}"
        )
    print("-" * 108)


def build_policies(
    controller_path: Path | None = None,
    seed: int = 123,
) -> Dict[str, PolicyCallable]:
    """Construye los métodos comparables de Fase 3."""
    seed_plan = SeedPlan(seed)
    astar = AStarPlanner(AStarConfig(cell_size=20, sand_threshold=0.15))
    best_of_n = BestOfNActions(RolloutConfig(depth=4, samples_per_action=3))
    tree = TreeOfActions(TreeSearchConfig(depth=3, beam_width=3, max_expansions=120))
    graph = GraphOfWaypoints(WaypointGraphConfig(cell_size=20, waypoint_stride=4))
    adaptive = AdaptiveRolloutBudget(AdaptiveBudgetConfig())

    strategies = build_reasoning_strategies()
    controller = LearnedReasoningController(
        strategies=strategies,
        config=ReasoningControllerConfig(
            epsilon=0.0,
            learning_rate=0.05,
            seed=seed_plan.controller_seed(),
        ),
    )
    if controller_path is not None and controller_path.exists():
        controller.load(controller_path)

    policies = {
        "Acción directa A*": astar.select_action,
        "Best-of-N actions": best_of_n.select_action,
        "Tree-of-Actions": tree.select_action,
        "Graph-of-Waypoints": graph.select_action,
        "Adaptive rollout budget": adaptive.select_action,
        "Learned reasoning controller": controller.select_action,
        "Acción directa geométrica": greedy_goal_policy,
    }

    # Fase 5: métodos más cercanos a RLoT y GoT.
    rlot_model_path = Path("models/rlot_navigator.json")
    policies.update(
        build_rlot_and_got_policies(
            rlot_model_path if rlot_model_path.exists() else None, seed=seed_plan.policy_seed()
        )
    )
    return policies


def parse_args() -> argparse.Namespace:
    """Lee argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Evalúa razonamiento en inferencia para navegación autónoma."
    )
    parser.add_argument(
        "--eval-episodes", type=int, default=5, help="Episodios de evaluación por método."
    )
    parser.add_argument("--max-steps", type=int, default=250, help="Máximo de pasos por episodio.")
    parser.add_argument("--seed", type=int, default=123, help="Semilla base.")
    parser.add_argument(
        "--results", type=Path, default=Path("results/reasoning_eval.csv"), help="CSV por episodio."
    )
    parser.add_argument(
        "--summary", type=Path, default=Path("results/reasoning_summary.csv"), help="CSV agregado."
    )
    parser.add_argument(
        "--controller",
        type=Path,
        default=Path("models/reasoning_controller.json"),
        help="Pesos opcionales del controlador aprendido.",
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        default=None,
        help="Subconjunto opcional de métodos. Ejemplo: --methods 'Best-of-N actions' 'Tree-of-Actions'",
    )
    return parser.parse_args()


def main() -> None:
    """Ejecuta la comparación de métodos de razonamiento."""
    args = parse_args()
    set_global_seed(args.seed)
    print(TITULO_PROYECTO)
    print("Fase 3 + Fase 5: razonamiento en inferencia y RL-of-Thoughts")

    policies = build_policies(args.controller, seed=args.seed)
    if args.methods:
        requested = set(args.methods)
        policies = {name: policy for name, policy in policies.items() if name in requested}
        missing = sorted(requested - set(policies))
        if missing:
            raise ValueError(f"Métodos no reconocidos: {missing}")

    all_rows: List[Dict[str, float | str]] = []
    for name, policy in policies.items():
        print(f"Evaluando: {name}")
        all_rows.extend(
            evaluate_policy(name, policy, args.eval_episodes, args.max_steps, args.seed)
        )

    summary_rows = summarize(all_rows)
    save_csv(all_rows, args.results)
    save_csv(summary_rows, args.summary)
    print_summary(summary_rows)
    print(f"Resultados por episodio guardados en: {args.results}")
    print(f"Resumen agregado guardado en: {args.summary}")


if __name__ == "__main__":
    main()
