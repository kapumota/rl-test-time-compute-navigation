"""
Proyecto: ¿Cuándo debe pensar un agente RL?
Utilidades comunes para construir el entorno de evaluación de baselines.
"""

from __future__ import annotations

from typing import Optional

from map import NavigationConfig, NavigationEnv


TITULO_PROYECTO = "¿Cuándo debe pensar un agente RL? Cómputo adaptativo en tiempo de inferencia para navegación autónoma"


def build_default_env(seed: Optional[int] = 123, max_steps: int = 400) -> NavigationEnv:
    """Crea el entorno estándar compartido por todos los baselines."""
    config = NavigationConfig(seed=seed, max_steps=max_steps, goal_radius=35.0)
    env = NavigationEnv(config=config)

    # Obstáculos fijos: dos paredes verticales y una línea diagonal parcial.
    # Esto obliga al agente a balancear avance, giro y evitación de arena.
    env.add_sand_rect(250, 120, 280, 460)
    env.add_sand_rect(480, 160, 520, 520)
    env.add_sand_line([(120, 500), (350, 380), (640, 420)], radius=8)
    return env


def build_empty_env(seed: Optional[int] = 123, max_steps: int = 400) -> NavigationEnv:
    """Crea un entorno sin obstáculos para pruebas de sanidad."""
    config = NavigationConfig(seed=seed, max_steps=max_steps, goal_radius=35.0)
    return NavigationEnv(config=config)
