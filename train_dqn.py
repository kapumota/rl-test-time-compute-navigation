"""
Proyecto: ¿Cuándo debe pensar un agente RL?
Entrenamiento base de DQN sobre el entorno limpio de navegación.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional

from deep_q_learning import DQNAgent
from map import NavigationConfig, NavigationEnv


def build_default_env(seed: Optional[int] = 123, max_steps: int = 400) -> NavigationEnv:
    """Crea un entorno con obstáculos simples para pruebas iniciales."""
    config = NavigationConfig(seed=seed, max_steps=max_steps, goal_radius=35.0)
    env = NavigationEnv(config=config)

    # Obstáculos sencillos para que el agente tenga que navegar y no solo avanzar.
    env.add_sand_rect(250, 120, 280, 460)
    env.add_sand_rect(480, 160, 520, 520)
    env.add_sand_line([(120, 500), (350, 380), (640, 420)], radius=8)
    return env


def train(
    episodes: int,
    checkpoint_path: Path,
    results_path: Path,
    seed: Optional[int] = 123,
    max_steps: int = 400,
) -> List[Dict[str, float]]:
    """Entrena el agente y devuelve métricas por episodio."""
    env = build_default_env(seed=seed, max_steps=max_steps)
    agent = DQNAgent(
        input_size=4,
        nb_action=3,
        gamma=0.99,
        learning_rate=1e-3,
        batch_size=64,
        memory_capacity=50_000,
        hidden_size=64,
    )

    history: List[Dict[str, float]] = []
    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_decay = max(episodes * 0.65, 1.0)

    for episode in range(1, episodes + 1):
        state = env.reset(seed=None)
        episode_reward = 0.0
        reached_goal = False
        last_loss = None

        epsilon = epsilon_end + (epsilon_start - epsilon_end) * max(0.0, 1.0 - episode / epsilon_decay)

        for _ in range(env.config.max_steps):
            action = agent.select_action(state, epsilon=epsilon)
            next_state, reward, done, info = env.step(action)
            last_loss = agent.update(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            reached_goal = bool(info["meta_alcanzada"])

            if done:
                break

        row = {
            "episodio": float(episode),
            "recompensa": float(episode_reward),
            "pasos": float(env.step_count),
            "epsilon": float(epsilon),
            "perdida": float(last_loss) if last_loss is not None else float("nan"),
            "meta_alcanzada": float(reached_goal),
        }
        history.append(row)

        if episode % 10 == 0 or episode == 1:
            print(
                f"Episodio {episode:04d} | "
                f"recompensa={episode_reward:8.2f} | "
                f"pasos={env.step_count:4d} | "
                f"epsilon={epsilon:.3f} | "
                f"meta={'sí' if reached_goal else 'no'}"
            )

    agent.save(checkpoint_path)
    save_history(history, results_path)
    print(f"Resultados guardados en: {results_path}")
    return history


def save_history(history: List[Dict[str, float]], path: Path) -> None:
    """Guarda métricas de entrenamiento en CSV."""
    if not history:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def parse_args() -> argparse.Namespace:
    """Lee argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description="Entrena un agente DQN en el entorno de navegación autónoma.")
    parser.add_argument("--episodes", type=int, default=100, help="Número de episodios de entrenamiento.")
    parser.add_argument("--checkpoint", type=Path, default=Path("last_brain.pth"), help="Ruta del checkpoint.")
    parser.add_argument("--results", type=Path, default=Path("results/train_history.csv"), help="Ruta del CSV de métricas.")
    parser.add_argument("--seed", type=int, default=123, help="Semilla aleatoria.")
    parser.add_argument("--max-steps", type=int, default=400, help="Máximo de pasos por episodio.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        episodes=args.episodes,
        checkpoint_path=args.checkpoint,
        results_path=args.results,
        seed=args.seed,
        max_steps=args.max_steps,
    )
