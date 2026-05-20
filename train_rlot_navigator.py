"""
Proyecto: ¿Cuándo debe pensar un agente RL?
Fase 5: entrenamiento del navegador RL-of-Thoughts.

El navegador no aprende directamente a manejar el auto. Aprende qué bloque de razonamiento usar:
ACT, CHAIN, TREE, GRAPH o REFLECT.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List

import numpy as np

from baseline_env import TITULO_PROYECTO, build_default_env
from rlot_got_navigation import RLoTNavigator, RLoTNavigatorConfig


def set_global_seed(seed: int) -> None:
    """Fija semillas para entrenamiento reproducible."""
    random.seed(seed)
    np.random.seed(seed)


def train(
    episodes: int,
    max_steps: int,
    seed: int,
    output: Path,
    epsilon: float,
    alpha: float,
    gamma: float,
    cost_penalty: float,
) -> RLoTNavigator:
    """Entrena el navegador RLoT con Q-learning tabular contextual."""
    navigator = RLoTNavigator(
        RLoTNavigatorConfig(
            epsilon=epsilon,
            alpha=alpha,
            gamma=gamma,
            cost_penalty=cost_penalty,
            seed=seed,
        )
    )
    episode_returns: List[float] = []

    for episode in range(1, episodes + 1):
        env = build_default_env(seed=seed + 10_000 + episode, max_steps=max_steps)
        state = env.reset(seed=seed + 20_000 + episode)
        total_reward = 0.0
        used_blocks: Dict[str, int] = {}

        for _step in range(max_steps):
            action, decision_info = navigator.select_action(env, state, explore=True)
            next_state, reward, done, env_info = env.step(action)
            next_state_key = None if done else navigator.discretize_state(env, next_state)
            navigator.update_from_info(decision_info, reward=float(reward), next_state_key=next_state_key)

            block_name = str(decision_info.get("bloque_razonamiento", "desconocido"))
            used_blocks[block_name] = used_blocks.get(block_name, 0) + 1
            total_reward += float(reward)
            state = next_state
            if done:
                break

        episode_returns.append(total_reward)
        if episode == 1 or episode % max(episodes // 5, 1) == 0:
            mean_return = float(np.mean(episode_returns[-10:]))
            print(
                f"Episodio {episode:04d} | recompensa_media_10={mean_return:.3f} | "
                f"estados_q={len(navigator.q_table)} | bloques={used_blocks}"
            )

    navigator.save(output)
    print(f"Navegador RL-of-Thoughts guardado en: {output}")
    return navigator


def parse_args() -> argparse.Namespace:
    """Lee argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description="Entrena el navegador RL-of-Thoughts para navegación.")
    parser.add_argument("--episodes", type=int, default=30, help="Episodios de entrenamiento.")
    parser.add_argument("--max-steps", type=int, default=200, help="Máximo de pasos por episodio.")
    parser.add_argument("--seed", type=int, default=123, help="Semilla base.")
    parser.add_argument("--epsilon", type=float, default=0.25, help="Exploración epsilon-greedy.")
    parser.add_argument("--alpha", type=float, default=0.15, help="Tasa de aprendizaje.")
    parser.add_argument("--gamma", type=float, default=0.90, help="Descuento temporal.")
    parser.add_argument("--cost-penalty", type=float, default=0.01, help="Penalización por costo de decisión.")
    parser.add_argument("--output", type=Path, default=Path("models/rlot_navigator.json"), help="Ruta de salida JSON.")
    return parser.parse_args()


def main() -> None:
    """Entrena y guarda el navegador RLoT."""
    args = parse_args()
    set_global_seed(args.seed)
    print(TITULO_PROYECTO)
    print("Fase 5: entrenamiento de RL-of-Thoughts para navegación")
    train(
        episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        output=args.output,
        epsilon=args.epsilon,
        alpha=args.alpha,
        gamma=args.gamma,
        cost_penalty=args.cost_penalty,
    )


if __name__ == "__main__":
    main()
