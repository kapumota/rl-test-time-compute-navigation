"""
Proyecto: ¿Cuándo debe pensar un agente RL?
Fase 4: evaluación en escenarios de generalización y robustez.

Escenarios incluidos:
- mapas fáciles,
- obstáculos densos,
- cambios de meta,
- mapas nunca vistos,
- sensores ruidosos.

Uso rápido:
    python run_evaluation_suite.py --eval-episodes 3 --max-steps 200
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Tuple

import numpy as np

from baseline_env import TITULO_PROYECTO
from decision_metrics import build_decision_metrics, measure_decision
from evaluation_scenarios import apply_dynamic_goal_if_needed, build_scenario_env, list_scenarios
from reasoning_policies import normalize_policy_result
from run_reasoning_experiments import build_policies
from reproducibility import SeedPlan, set_global_seed

PolicyCallable = Callable[[Any, np.ndarray], int | Tuple[int, Dict[str, Any]]]


def evaluate_policy_on_scenario(
    method_name: str,
    policy: PolicyCallable,
    scenario_name: str,
    eval_episodes: int,
    max_steps: int,
    seed: int,
) -> List[Dict[str, float | str]]:
    """Evalúa una política sobre un escenario específico."""
    rows: List[Dict[str, float | str]] = []
    seed_plan = SeedPlan(seed)

    for episode in range(1, eval_episodes + 1):
        instance = build_scenario_env(
            scenario_name=scenario_name,
            seed=seed_plan.env_seed(episode, offset=100_000),
            max_steps=max_steps,
            episode=episode,
        )
        env = instance.env
        state = env.reset(
            seed=seed_plan.reset_seed(episode, offset=200_000),
            options=instance.reset_options,
        )

        total_reward = 0.0
        sand_steps = 0
        border_collisions = 0
        goals_reached = 0
        reached_goal_at_least_once = False
        decision_cost_total = 0.0
        decision_time_ms_total = 0.0
        reasoning_counts: Dict[str, int] = {}
        dynamic_goal_changes = 0

        for _step_index in range(max_steps):
            measurement = measure_decision(
                lambda: normalize_policy_result(policy(env, state)),
                simulated_cost_steps=1.0,
            )
            action, decision_info = measurement.value
            decision_metrics = build_decision_metrics(
                simulated_cost_steps=decision_info.get(
                    "costo_decision",
                    measurement.simulated_cost_steps,
                ),
                real_time_ms=measurement.real_time_ms,
            )
            decision_cost_total += float(decision_metrics["costo_decision_pasos"])
            decision_time_ms_total += float(decision_metrics["tiempo_decision_ms"])
            reasoning_method = str(decision_info.get("metodo_razonamiento", method_name))
            reasoning_counts[reasoning_method] = reasoning_counts.get(reasoning_method, 0) + 1

            next_state, reward, done, info = env.step(action)
            total_reward += float(reward)
            sand_steps += int(info["sobre_arena"])
            border_collisions += int(info["colision_borde"])

            if bool(info["meta_alcanzada"]):
                goals_reached += 1
                reached_goal_at_least_once = True

            if apply_dynamic_goal_if_needed(instance):
                dynamic_goal_changes += 1
                # La meta cambió después del paso; se recalcula el estado para que la orientación sea correcta.
                env._update_sensors()
                next_state = env._get_observation()

            state = next_state
            if done:
                break

        steps = max(env.step_count, 1)
        final_distance = float(np.linalg.norm(env.position - env.goal))
        rows.append(
            {
                "escenario": instance.config.name,
                "escenario_visible": instance.config.display_name,
                "descripcion_escenario": instance.config.description,
                "mapa_id": instance.map_id,
                "metodo": method_name,
                "episodio": float(episode),
                "recompensa": float(total_reward),
                "pasos": float(env.step_count),
                "meta_alcanzada": float(reached_goal_at_least_once),
                "metas_alcanzadas": float(goals_reached),
                "cambios_de_meta": float(dynamic_goal_changes),
                "pasos_en_arena": float(sand_steps),
                "colisiones_borde": float(border_collisions),
                "distancia_final": float(final_distance),
                "densidad_arena": float(instance.sand_density),
                "ruido_sensores_std": float(instance.config.sensor_noise_std),
                "dropout_sensores": float(instance.config.sensor_dropout_prob),
                "costo_decision_pasos_total": float(decision_cost_total),
                "costo_decision_pasos_promedio": float(decision_cost_total / steps),
                "tiempo_decision_ms_total": float(decision_time_ms_total),
                "tiempo_decision_ms_promedio": float(decision_time_ms_total / steps),
                "costo_decision_total": float(decision_cost_total),
                "costo_decision_promedio": float(decision_cost_total / steps),
                "conteo_razonamiento": str(reasoning_counts),
            }
        )

    return rows


def summarize(rows: List[Dict[str, float | str]]) -> List[Dict[str, float | str]]:
    """Agrega métricas promedio por escenario y método."""
    groups = sorted({(str(row["escenario"]), str(row["metodo"])) for row in rows})
    numeric_keys = [
        "recompensa",
        "pasos",
        "meta_alcanzada",
        "metas_alcanzadas",
        "cambios_de_meta",
        "pasos_en_arena",
        "colisiones_borde",
        "distancia_final",
        "densidad_arena",
        "costo_decision_pasos_total",
        "costo_decision_pasos_promedio",
        "tiempo_decision_ms_total",
        "tiempo_decision_ms_promedio",
        "costo_decision_total",
        "costo_decision_promedio",
    ]
    summary: List[Dict[str, float | str]] = []

    for scenario_name, method_name in groups:
        subset = [
            row
            for row in rows
            if row["escenario"] == scenario_name and row["metodo"] == method_name
        ]
        first = subset[0]
        item: Dict[str, float | str] = {
            "escenario": scenario_name,
            "escenario_visible": str(first["escenario_visible"]),
            "metodo": method_name,
            "episodios": float(len(subset)),
        }
        for key in numeric_keys:
            values = [float(row[key]) for row in subset]
            item[f"{key}_media"] = float(np.mean(values))
            item[f"{key}_desv"] = float(np.std(values))
        summary.append(item)
    return summary


def save_csv(rows: List[Dict[str, float | str]], path: Path) -> None:
    """Guarda filas en CSV usando UTF-8."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_summary(summary_rows: List[Dict[str, float | str]]) -> None:
    """Imprime una tabla compacta por escenario y método."""
    print("\nResumen de Fase 4: evaluación por escenario")
    print("-" * 132)
    print(
        f"{'escenario':<24} {'método':<32} {'recompensa':>11} {'éxito':>8} "
        f"{'distancia':>10} {'arena':>8} {'costo/dec':>10}"
    )
    print("-" * 132)
    for row in summary_rows:
        print(
            f"{str(row['escenario_visible']):<24} "
            f"{str(row['metodo']):<32} "
            f"{float(row['recompensa_media']):11.2f} "
            f"{float(row['meta_alcanzada_media']):8.2f} "
            f"{float(row['distancia_final_media']):10.2f} "
            f"{float(row['pasos_en_arena_media']):8.2f} "
            f"{float(row['costo_decision_promedio_media']):10.2f}"
        )
    print("-" * 132)


def parse_args() -> argparse.Namespace:
    """Lee argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Evalúa métodos de razonamiento en cinco escenarios de Fase 4."
    )
    parser.add_argument(
        "--eval-episodes", type=int, default=3, help="Episodios por método y escenario."
    )
    parser.add_argument("--max-steps", type=int, default=250, help="Máximo de pasos por episodio.")
    parser.add_argument("--seed", type=int, default=123, help="Semilla base.")
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results/evaluation_suite.csv"),
        help="CSV por episodio.",
    )
    parser.add_argument(
        "--summary", type=Path, default=Path("results/evaluation_summary.csv"), help="CSV agregado."
    )
    parser.add_argument(
        "--scenarios",
        nargs="*",
        default=None,
        help="Subconjunto opcional de escenarios. Ejemplo: --scenarios facil sensores_ruidosos",
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        default=None,
        help="Subconjunto opcional de métodos. Ejemplo: --methods 'Best-of-N actions' 'Adaptive rollout budget'",
    )
    return parser.parse_args()


def validate_selection(requested: Iterable[str], valid: Iterable[str], label: str) -> List[str]:
    """Valida una selección de nombres y conserva el orden solicitado."""
    valid_set = set(valid)
    requested_list = list(requested)
    missing = [name for name in requested_list if name not in valid_set]
    if missing:
        raise ValueError(f"{label} no reconocidos: {missing}. Valores válidos: {sorted(valid_set)}")
    return requested_list


def main() -> None:
    """Ejecuta la evaluación completa de Fase 4."""
    args = parse_args()
    set_global_seed(args.seed)
    print(TITULO_PROYECTO)
    print("Fase 4: evaluación en mapas fáciles, densos, dinámicos, no vistos y ruidosos")

    scenario_names = list_scenarios()
    if args.scenarios:
        scenario_names = validate_selection(args.scenarios, scenario_names, "Escenarios")

    policies = build_policies(seed=args.seed)
    if args.methods:
        requested_methods = validate_selection(args.methods, policies.keys(), "Métodos")
        policies = {name: policies[name] for name in requested_methods}

    all_rows: List[Dict[str, float | str]] = []
    for scenario_name in scenario_names:
        for method_name, policy in policies.items():
            print(f"Evaluando escenario='{scenario_name}' método='{method_name}'")
            rows = evaluate_policy_on_scenario(
                method_name=method_name,
                policy=policy,
                scenario_name=scenario_name,
                eval_episodes=args.eval_episodes,
                max_steps=args.max_steps,
                seed=args.seed,
            )
            all_rows.extend(rows)

    summary_rows = summarize(all_rows)
    save_csv(all_rows, args.results)
    save_csv(summary_rows, args.summary)
    print_summary(summary_rows)
    print(f"Resultados por episodio guardados en: {args.results}")
    print(f"Resumen agregado guardado en: {args.summary}")


if __name__ == "__main__":
    main()
