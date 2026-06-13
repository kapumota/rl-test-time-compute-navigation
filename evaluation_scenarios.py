"""
Proyecto: ¿Cuándo debe pensar un agente RL?
Fase 4: escenarios de evaluación para generalización y robustez.

Este archivo define mapas fáciles, mapas densos, metas cambiantes, mapas no vistos
sensores ruidosos. 
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from baseline_env import build_default_env
from map import NavigationConfig, NavigationEnv

Array = np.ndarray
Point = Tuple[float, float]


@dataclass(frozen=True)
class ScenarioConfig:
    """Configuración declarativa de un escenario de evaluación."""

    name: str
    display_name: str
    description: str
    start_position: Point = (400.0, 300.0)
    start_angle: float = 0.0
    goal: Point = (20.0, 580.0)
    dynamic_goals: Tuple[Point, ...] = ()
    dynamic_goal_interval: int = 0
    sensor_noise_std: float = 0.0
    sensor_dropout_prob: float = 0.0
    unseen_map: bool = False
    tags: Tuple[str, ...] = ()


@dataclass
class ScenarioInstance:
    """Entorno construido y metadatos usados por la evaluación."""

    config: ScenarioConfig
    env: NavigationEnv
    reset_options: Dict[str, Any]
    map_id: str
    sand_density: float
    goal_changes: int = 0


def list_scenarios() -> List[str]:
    """Lista los nombres internos de escenarios disponibles."""
    return [scenario.name for scenario in get_scenario_configs()]


def get_scenario_configs() -> List[ScenarioConfig]:
    """Devuelve la lista estándar de escenarios de Fase 4."""
    return [
        ScenarioConfig(
            name="facil",
            display_name="Mapa fácil",
            description="Mapa sin obstáculos para medir control básico y eficiencia hacia la meta.",
            start_position=(120.0, 120.0),
            start_angle=0.0,
            goal=(700.0, 500.0),
            tags=("facil", "sanidad"),
        ),
        ScenarioConfig(
            name="obstaculos_densos",
            display_name="Obstáculos densos",
            description="Mapa con paredes, líneas y pasillos parciales para medir navegación deliberativa.",
            start_position=(60.0, 60.0),
            start_angle=0.0,
            goal=(740.0, 540.0),
            tags=("denso", "obstaculos"),
        ),
        ScenarioConfig(
            name="cambios_de_meta",
            display_name="Cambios de meta",
            description="La meta cambia durante el episodio para medir adaptación en tiempo de inferencia.",
            start_position=(400.0, 300.0),
            start_angle=0.0,
            goal=(720.0, 520.0),
            dynamic_goals=((720.0, 520.0), (80.0, 520.0), (720.0, 80.0), (120.0, 120.0)),
            dynamic_goal_interval=60,
            tags=("dinamico", "meta"),
        ),
        ScenarioConfig(
            name="mapas_nunca_vistos",
            display_name="Mapas nunca vistos",
            description="Cada episodio genera obstáculos nuevos por semilla para medir generalización.",
            start_position=(80.0, 80.0),
            start_angle=0.0,
            goal=(720.0, 520.0),
            unseen_map=True,
            tags=("generalizacion", "semillas"),
        ),
        ScenarioConfig(
            name="sensores_ruidosos",
            display_name="Sensores ruidosos",
            description="Mapa estándar con ruido gaussiano y apagones en sensores para medir robustez perceptual.",
            start_position=(400.0, 300.0),
            start_angle=0.0,
            goal=(20.0, 580.0),
            sensor_noise_std=0.15,
            sensor_dropout_prob=0.05,
            tags=("ruido", "robustez"),
        ),
    ]


def get_scenario_config(name: str) -> ScenarioConfig:
    """Obtiene un escenario por nombre interno."""
    for scenario in get_scenario_configs():
        if scenario.name == name:
            return scenario
    valid = ", ".join(list_scenarios())
    raise ValueError(f"Escenario no reconocido: {name}. Escenarios válidos: {valid}.")


class NoisyNavigationEnv(NavigationEnv):
    """Entorno que añade ruido a las lecturas de sensores observadas por el agente."""

    def __init__(
        self,
        config: Optional[NavigationConfig] = None,
        sand_map: Optional[Array] = None,
        render_mode: str = "text",
        sensor_noise_std: float = 0.0,
        sensor_dropout_prob: float = 0.0,
        noise_seed: Optional[int] = None,
    ) -> None:
        self.sensor_noise_std = float(sensor_noise_std)
        self.sensor_dropout_prob = float(sensor_dropout_prob)
        self.noise_rng = np.random.default_rng(noise_seed)
        super().__init__(config=config, sand_map=sand_map, render_mode=render_mode)

    def _get_observation(self) -> Array:
        """Devuelve el estado observado con ruido solo en sensores."""
        observation = super()._get_observation().astype(np.float32)
        if self.sensor_noise_std <= 0.0 and self.sensor_dropout_prob <= 0.0:
            return observation

        noisy_sensors = observation[1:4].copy()
        if self.sensor_noise_std > 0.0:
            noisy_sensors += self.noise_rng.normal(0.0, self.sensor_noise_std, size=3).astype(np.float32)
        if self.sensor_dropout_prob > 0.0:
            dropout_mask = self.noise_rng.random(3) < self.sensor_dropout_prob
            noisy_sensors[dropout_mask] = 1.0

        observation[1:4] = np.clip(noisy_sensors, 0.0, 1.0)
        return observation


def build_scenario_env(
    scenario_name: str,
    seed: int = 123,
    max_steps: int = 400,
    episode: int = 1,
) -> ScenarioInstance:
    """Construye el entorno correspondiente a un escenario y episodio."""
    scenario = get_scenario_config(scenario_name)
    env_seed = int(seed + 1_000 * episode)
    config = NavigationConfig(
        seed=env_seed,
        max_steps=max_steps,
        goal_radius=35.0,
        terminate_on_goal=not bool(scenario.dynamic_goals),
    )

    if scenario.sensor_noise_std > 0.0 or scenario.sensor_dropout_prob > 0.0:
        env: NavigationEnv = NoisyNavigationEnv(
            config=config,
            sensor_noise_std=scenario.sensor_noise_std,
            sensor_dropout_prob=scenario.sensor_dropout_prob,
            noise_seed=env_seed + 77,
        )
    else:
        env = NavigationEnv(config=config)

    env.clear_sand()
    map_id = scenario.name

    if scenario.name == "facil":
        _build_easy_map(env)
    elif scenario.name == "obstaculos_densos":
        _build_dense_obstacle_map(env)
    elif scenario.name == "cambios_de_meta":
        _build_dynamic_goal_map(env)
    elif scenario.name == "mapas_nunca_vistos":
        map_id = _build_unseen_map(env, env_seed)
    elif scenario.name == "sensores_ruidosos":
        # Se reutiliza el mapa estándar de Fase 2/3 para aislar el efecto del ruido perceptual.
        default_env = build_default_env(seed=env_seed, max_steps=max_steps)
        env.sand = default_env.sand.copy()
    else:
        raise ValueError(f"Escenario no implementado: {scenario.name}.")

    reset_options = {
        "start_position": scenario.start_position,
        "start_angle": scenario.start_angle,
        "goal": scenario.goal,
    }
    sand_density = float(np.mean(env.sand > 0.0))
    return ScenarioInstance(
        config=scenario,
        env=env,
        reset_options=reset_options,
        map_id=map_id,
        sand_density=sand_density,
    )


def apply_dynamic_goal_if_needed(instance: ScenarioInstance) -> bool:
    """Cambia la meta del entorno cuando el escenario lo requiere."""
    scenario = instance.config
    if not scenario.dynamic_goals or scenario.dynamic_goal_interval <= 0:
        return False
    if instance.env.step_count <= 0:
        return False
    if instance.env.step_count % scenario.dynamic_goal_interval != 0:
        return False

    next_index = (instance.goal_changes + 1) % len(scenario.dynamic_goals)
    instance.env.goal = np.array(scenario.dynamic_goals[next_index], dtype=np.float32)
    instance.env.last_distance = float(np.linalg.norm(instance.env.position - instance.env.goal))
    instance.goal_changes += 1
    return True


def _build_easy_map(env: NavigationEnv) -> None:
    """Mapa sin arena para prueba de sanidad."""
    env.clear_sand()


def _build_dense_obstacle_map(env: NavigationEnv) -> None:
    """Construye obstáculos densos con pasillos parciales."""
    env.clear_sand()
    env.add_sand_rect(170, 40, 205, 300)
    env.add_sand_rect(170, 380, 205, 560)
    env.add_sand_rect(360, 100, 400, 430)
    env.add_sand_rect(560, 180, 600, 570)
    env.add_sand_line([(70, 470), (260, 360), (460, 390), (720, 300)], radius=9)
    env.add_sand_line([(260, 80), (420, 170), (650, 130)], radius=8)


def _build_dynamic_goal_map(env: NavigationEnv) -> None:
    """Mapa moderado diseñado para evaluar adaptación ante cambios de meta."""
    env.clear_sand()
    env.add_sand_rect(250, 120, 285, 480)
    env.add_sand_rect(500, 80, 535, 420)
    env.add_sand_line([(90, 520), (250, 430), (440, 470), (700, 420)], radius=7)


def _build_unseen_map(env: NavigationEnv, seed: int) -> str:
    """Genera un mapa no visto con obstáculos aleatorios reproducibles."""
    env.clear_sand()
    rng = np.random.default_rng(seed)

    # Obstáculos rectangulares pequeños. Se evitan zonas cercanas al inicio y a la meta.
    protected_points = [np.array([80.0, 80.0]), np.array([720.0, 520.0])]
    for _ in range(7):
        for _attempt in range(25):
            width = int(rng.integers(30, 75))
            height = int(rng.integers(45, 140))
            x0 = int(rng.integers(120, max(121, env.width - width - 80)))
            y0 = int(rng.integers(70, max(71, env.height - height - 70)))
            center = np.array([x0 + width / 2.0, y0 + height / 2.0])
            if all(float(np.linalg.norm(center - point)) > 110.0 for point in protected_points):
                env.add_sand_rect(x0, y0, x0 + width, y0 + height)
                break

    for _ in range(3):
        points: List[Point] = []
        for _point_index in range(3):
            points.append((float(rng.integers(100, 720)), float(rng.integers(80, 540))))
        env.add_sand_line(points, radius=int(rng.integers(5, 10)))

    return f"mapa_no_visto_seed_{seed}"
