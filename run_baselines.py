"""
Proyecto: ¿Cuándo debe pensar un agente RL?
Fase 2: comparación de baselines.

Baselines incluidos:
- Random policy: acciones aleatorias.
- DQN original: red Q del proyecto actualizado a PyTorch 2.x.
- Double DQN: mejora estándar con red objetivo.
- PPO: baseline policy-gradient compacto.
- A* planner: planificador clásico sobre mapa discretizado.

Uso rápido:
    python run_baselines.py --train-episodes 30 --eval-episodes 10
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from astar_planner import AStarConfig, AStarPlanner
from baseline_env import TITULO_PROYECTO, build_default_env
from deep_q_learning import DQNAgent
from double_dqn import DoubleDQNAgent, DoubleDQNConfig
from ppo_agent import PPOAgent, PPOConfig
from reproducibility import SeedPlan, set_global_seed

PolicyResult = int | Tuple[int, Dict[str, Any]]
Policy = Callable[[Any, np.ndarray], PolicyResult]


def unpack_policy_result(result: PolicyResult) -> Tuple[int, Dict[str, Any]]:
    """Normaliza el resultado de una política a acción e información extra."""
    if isinstance(result, tuple):
        action, info = result
        return int(action), dict(info)
    return int(result), {"costo_decision": 1.0}


def evaluate_policy(
    name: str,
    policy: Policy,
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
        plan_length_accum = 0.0
        plan_length_count = 0

        for _ in range(max_steps):
            action, decision_info = unpack_policy_result(policy(env, state))
            decision_cost_total += float(decision_info.get("costo_decision", 1.0))
            if "longitud_plan" in decision_info:
                plan_length_accum += float(decision_info["longitud_plan"])
                plan_length_count += 1

            next_state, reward, done, info = env.step(action)
            total_reward += reward
            sand_steps += int(info["sobre_arena"])
            border_collisions += int(info["colision_borde"])
            reached_goal = bool(info["meta_alcanzada"])
            state = next_state

            if done:
                break

        steps = max(env.step_count, 1)
        rows.append(
            {
                "baseline": name,
                "episodio": float(episode),
                "recompensa": float(total_reward),
                "pasos": float(env.step_count),
                "meta_alcanzada": float(reached_goal),
                "pasos_en_arena": float(sand_steps),
                "colisiones_borde": float(border_collisions),
                "costo_decision_total": float(decision_cost_total),
                "costo_decision_promedio": float(decision_cost_total / steps),
                "longitud_plan_promedio": float(plan_length_accum / max(plan_length_count, 1)),
            }
        )

    return rows


def summarize(rows: List[Dict[str, float | str]]) -> List[Dict[str, float | str]]:
    """Agrega métricas promedio por baseline."""
    baselines = sorted({str(row["baseline"]) for row in rows})
    summary: List[Dict[str, float | str]] = []
    numeric_keys = [
        "recompensa",
        "pasos",
        "meta_alcanzada",
        "pasos_en_arena",
        "colisiones_borde",
        "costo_decision_total",
        "costo_decision_promedio",
        "longitud_plan_promedio",
    ]

    for baseline in baselines:
        subset = [row for row in rows if row["baseline"] == baseline]
        item: Dict[str, float | str] = {"baseline": baseline, "episodios": float(len(subset))}
        for key in numeric_keys:
            values = [float(row[key]) for row in subset]
            item[f"{key}_media"] = float(np.mean(values))
            item[f"{key}_desv"] = float(np.std(values))
        summary.append(item)
    return summary


def train_dqn(train_episodes: int, max_steps: int, seed: int) -> DQNAgent:
    """Entrena el baseline DQN original."""
    env = build_default_env(seed=seed, max_steps=max_steps)
    agent = DQNAgent(input_size=4, nb_action=3, gamma=0.99, learning_rate=1e-3, batch_size=64)
    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_decay = max(train_episodes * 0.70, 1.0)

    for episode in range(1, train_episodes + 1):
        state = env.reset(seed=seed + episode)
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * max(
            0.0, 1.0 - episode / epsilon_decay
        )
        total_reward = 0.0
        for _ in range(max_steps):
            action = agent.select_action(state, epsilon=epsilon)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                break
        if episode == 1 or episode % 10 == 0:
            print(
                f"DQN | episodio {episode:04d} | recompensa={total_reward:8.2f} | epsilon={epsilon:.3f}"
            )
    return agent


def train_double_dqn(train_episodes: int, max_steps: int, seed: int) -> DoubleDQNAgent:
    """Entrena el baseline Double DQN."""
    env = build_default_env(seed=seed, max_steps=max_steps)
    config = DoubleDQNConfig(batch_size=64, target_update_interval=250)
    agent = DoubleDQNAgent(input_size=4, nb_action=3, config=config)
    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_decay = max(train_episodes * 0.70, 1.0)

    for episode in range(1, train_episodes + 1):
        state = env.reset(seed=seed + 1_000 + episode)
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * max(
            0.0, 1.0 - episode / epsilon_decay
        )
        total_reward = 0.0
        for _ in range(max_steps):
            action = agent.select_action(state, epsilon=epsilon)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                break
        if episode == 1 or episode % 10 == 0:
            print(
                f"Double DQN | episodio {episode:04d} | recompensa={total_reward:8.2f} | epsilon={epsilon:.3f}"
            )
    return agent


def train_ppo(train_episodes: int, max_steps: int, seed: int) -> PPOAgent:
    """Entrena el baseline PPO compacto."""
    env = build_default_env(seed=seed, max_steps=max_steps)
    config = PPOConfig(minibatch_size=128, update_epochs=4)
    agent = PPOAgent(input_size=4, nb_action=3, config=config)

    states: List[np.ndarray] = []
    actions: List[int] = []
    rewards: List[float] = []
    dones: List[bool] = []
    log_probs: List[float] = []
    values: List[float] = []

    # PPO necesita lotes de trayectoria; se actualiza cada pocos episodios.
    update_every_episodes = 5

    for episode in range(1, train_episodes + 1):
        state = env.reset(seed=seed + 2_000 + episode)
        total_reward = 0.0
        for _ in range(max_steps):
            action, log_prob, value = agent.act(state, deterministic=False)
            next_state, reward, done, _ = env.step(action)

            states.append(np.asarray(state, dtype=np.float32))
            actions.append(int(action))
            rewards.append(float(reward))
            dones.append(bool(done))
            log_probs.append(float(log_prob))
            values.append(float(value))

            state = next_state
            total_reward += reward
            if done:
                break

        if episode % update_every_episodes == 0 or episode == train_episodes:
            loss = agent.learn(states, actions, rewards, dones, log_probs, values)
            states.clear()
            actions.clear()
            rewards.clear()
            dones.clear()
            log_probs.clear()
            values.clear()
        else:
            loss = float("nan")

        if episode == 1 or episode % 10 == 0:
            print(f"PPO | episodio {episode:04d} | recompensa={total_reward:8.2f} | perdida={loss}")
    return agent


def save_csv(rows: List[Dict[str, float | str]], path: Path) -> None:
    """Guarda filas en CSV."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_summary(summary_rows: List[Dict[str, float | str]]) -> None:
    """Imprime una tabla compacta de resultados."""
    print("\nResumen de baselines")
    print("-" * 92)
    print(
        f"{'baseline':<16} {'recompensa':>12} {'éxito':>10} {'pasos':>10} {'arena':>10} {'costo/dec':>12}"
    )
    print("-" * 92)
    for row in summary_rows:
        print(
            f"{str(row['baseline']):<16} "
            f"{float(row['recompensa_media']):12.2f} "
            f"{float(row['meta_alcanzada_media']):10.2f} "
            f"{float(row['pasos_media']):10.2f} "
            f"{float(row['pasos_en_arena_media']):10.2f} "
            f"{float(row['costo_decision_promedio_media']):12.2f}"
        )
    print("-" * 92)


def parse_args() -> argparse.Namespace:
    """Lee argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description="Compara baselines de navegación autónoma.")
    parser.add_argument(
        "--train-episodes",
        type=int,
        default=30,
        help="Episodios de entrenamiento para DQN, Double DQN y PPO.",
    )
    parser.add_argument(
        "--eval-episodes", type=int, default=10, help="Episodios de evaluación por baseline."
    )
    parser.add_argument("--max-steps", type=int, default=400, help="Máximo de pasos por episodio.")
    parser.add_argument("--seed", type=int, default=123, help="Semilla base.")
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results/baselines_eval.csv"),
        help="CSV de evaluación por episodio.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("results/baselines_summary.csv"),
        help="CSV de resumen agregado.",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Evalúa solo Random y A* para una prueba rápida.",
    )
    return parser.parse_args()


def main() -> None:
    """Ejecuta la comparación completa de baselines."""
    args = parse_args()
    set_global_seed(args.seed)
    print(TITULO_PROYECTO)
    print("Fase 2: baselines comparables")

    all_rows: List[Dict[str, float | str]] = []

    random_policy: Policy = lambda env, state: env.sample_action()
    all_rows.extend(
        evaluate_policy(
            "Random policy", random_policy, args.eval_episodes, args.max_steps, args.seed
        )
    )

    astar = AStarPlanner(AStarConfig(cell_size=20, sand_threshold=0.15))
    astar_policy: Policy = lambda env, state: astar.select_action(env, state)
    all_rows.extend(
        evaluate_policy("A* planner", astar_policy, args.eval_episodes, args.max_steps, args.seed)
    )

    if not args.skip_training:
        dqn = train_dqn(args.train_episodes, args.max_steps, args.seed)
        dqn_policy: Policy = lambda env, state: dqn.select_action(state, epsilon=0.0)
        all_rows.extend(
            evaluate_policy(
                "DQN original", dqn_policy, args.eval_episodes, args.max_steps, args.seed
            )
        )

        double_dqn = train_double_dqn(args.train_episodes, args.max_steps, args.seed)
        double_dqn_policy: Policy = lambda env, state: double_dqn.select_action(state, epsilon=0.0)
        all_rows.extend(
            evaluate_policy(
                "Double DQN", double_dqn_policy, args.eval_episodes, args.max_steps, args.seed
            )
        )

        ppo = train_ppo(args.train_episodes, args.max_steps, args.seed)
        ppo_policy: Policy = lambda env, state: ppo.select_action(state, deterministic=True)
        all_rows.extend(
            evaluate_policy("PPO", ppo_policy, args.eval_episodes, args.max_steps, args.seed)
        )

    summary_rows = summarize(all_rows)
    save_csv(all_rows, args.results)
    save_csv(summary_rows, args.summary)
    print_summary(summary_rows)
    print(f"Resultados por episodio guardados en: {args.results}")
    print(f"Resumen agregado guardado en: {args.summary}")


if __name__ == "__main__":
    main()
