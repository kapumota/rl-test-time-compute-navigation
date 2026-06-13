"""
Proyecto: ¿Cuándo debe pensar un agente RL?
Entorno de navegación autónoma con cómputo adaptativo en tiempo de inferencia.

Este archivo reemplaza la lógica acoplada a Kivy por un entorno limpio tipo Gymnasium.
Las firmas principales se mantienen en inglés: reset(), step() y render().
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple
import copy
import math

import numpy as np

from reproducibility import build_numpy_rng

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # El entorno sigue funcionando sin instalar Gymnasium.
    gym = None
    spaces = None


Array = np.ndarray


@dataclass
class NavigationConfig:
    """Configuración principal del entorno de navegación."""

    width: int = 800
    height: int = 600
    normal_speed: float = 6.0
    sand_speed: float = 1.0
    margin: int = 10
    sensor_distance: float = 30.0
    sensor_angle: float = 30.0
    sensor_radius: int = 10
    goal_radius: float = 35.0
    max_steps: int = 2_000
    goal_bonus: float = 1.0
    terminate_on_goal: bool = True
    action_to_rotation: Tuple[float, float, float] = (0.0, 20.0, -20.0)
    seed: Optional[int] = None


class NavigationEnv(gym.Env if gym is not None else object):
    """
    Entorno de navegación autónoma.

    Estado observado:
        [orientación_hacia_meta, señal_sensor_frontal, señal_sensor_izquierdo, señal_sensor_derecho]

    Acciones:
        0 -> avanzar recto
        1 -> girar a la izquierda
        2 -> girar a la derecha

    La API es una versión simple estilo Gymnasium:
        reset() -> estado
        step(action) -> nuevo_estado, recompensa, terminado, info
        render() -> visualización opcional
    """

    metadata = {"render_modes": ["rgb_array", "human", "text"]}

    def __init__(
        self,
        config: Optional[NavigationConfig] = None,
        sand_map: Optional[Array] = None,
        render_mode: str = "text",
    ) -> None:
        self.config = config or NavigationConfig()
        self.render_mode = render_mode
        self.rng = build_numpy_rng(self.config.seed)

        self.width = int(self.config.width)
        self.height = int(self.config.height)
        self.sand = self._prepare_sand_map(sand_map)

        # Se definen estos espacios si Gymnasium está disponible.
        if spaces is not None:
            self.observation_space = spaces.Box(
                low=np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
                dtype=np.float32,
            )
            self.action_space = spaces.Discrete(len(self.config.action_to_rotation))
        else:
            self.observation_space = None
            self.action_space = None

        self.position = np.zeros(2, dtype=np.float32)
        self.velocity = np.zeros(2, dtype=np.float32)
        self.angle = 0.0
        self.goal = np.zeros(2, dtype=np.float32)
        self.sensor_positions = np.zeros((3, 2), dtype=np.float32)
        self.sensor_signals = np.zeros(3, dtype=np.float32)
        self.last_distance = 0.0
        self.step_count = 0
        self.last_reward = 0.0

        self.reset()

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Array:
        """Reinicia el episodio y devuelve el estado inicial."""
        if seed is not None:
            self.rng = build_numpy_rng(seed)

        options = options or {}
        start_position = options.get("start_position")
        start_angle = float(options.get("start_angle", 0.0))
        goal = options.get("goal")
        sand_map = options.get("sand_map")

        if sand_map is not None:
            self.sand = self._prepare_sand_map(sand_map)

        if start_position is None:
            self.position = np.array([self.width / 2.0, self.height / 2.0], dtype=np.float32)
        else:
            self.position = np.array(start_position, dtype=np.float32)

        self.angle = start_angle
        self.velocity = self._vector_from_angle(self.angle, self.config.normal_speed)

        if goal is None:
            self.goal = np.array([20.0, self.height - 20.0], dtype=np.float32)
        else:
            self.goal = np.array(goal, dtype=np.float32)

        self.step_count = 0
        self.last_reward = 0.0
        self._update_sensors()
        self.last_distance = self._distance_to_goal()
        return self._get_observation()

    def step(self, action: int) -> Tuple[Array, float, bool, Dict[str, Any]]:
        """Ejecuta una acción y devuelve nuevo_estado, recompensa, terminado e información."""
        action = int(action)
        if action < 0 or action >= len(self.config.action_to_rotation):
            raise ValueError(f"Acción inválida: {action}. Usa 0, 1 o 2.")

        rotation = float(self.config.action_to_rotation[action])
        self.angle = (self.angle + rotation) % 360.0
        self.position = self.position + self.velocity
        self._update_sensors()

        self.step_count += 1
        distance = self._distance_to_goal()
        on_sand = self._is_on_sand(self.position)
        boundary_collision = self._enforce_boundaries()
        reached_goal = distance <= self.config.goal_radius

        # Recompensa base heredada del proyecto original.
        if on_sand:
            self.velocity = self._vector_from_angle(self.angle, self.config.sand_speed)
            reward = -1.0
        else:
            self.velocity = self._vector_from_angle(self.angle, self.config.normal_speed)
            reward = -0.2
            if distance < self.last_distance:
                reward = 0.1

        if boundary_collision:
            reward = -1.0

        if reached_goal:
            reward += float(self.config.goal_bonus)

        terminated = bool(reached_goal and self.config.terminate_on_goal)
        truncated = self.step_count >= self.config.max_steps
        done = bool(terminated or truncated)

        if reached_goal and not self.config.terminate_on_goal:
            # Modo continuo: la meta cambia de esquina, como en el código original.
            self.goal = np.array(
                [self.width - self.goal[0], self.height - self.goal[1]], dtype=np.float32
            )
            distance = self._distance_to_goal()

        self.last_distance = distance
        self.last_reward = reward

        observation = self._get_observation()
        info = {
            "mensaje": "Paso ejecutado correctamente.",
            "distancia_a_meta": float(distance),
            "meta_alcanzada": bool(reached_goal),
            "colision_borde": bool(boundary_collision),
            "sobre_arena": bool(on_sand),
            "episodio_truncado": bool(truncated),
            "paso": int(self.step_count),
            "accion": action,
            "rotacion": rotation,
            "posicion": self.position.astype(float).tolist(),
            "meta": self.goal.astype(float).tolist(),
            "angulo": float(self.angle),
            "sensores": self.sensor_positions.astype(float).tolist(),
            "senales": self.sensor_signals.astype(float).tolist(),
        }
        return observation, float(reward), done, info

    def render(self, mode: Optional[str] = None) -> Any:
        """Devuelve una visualización simple del entorno."""
        mode = mode or self.render_mode
        if mode == "text":
            return (
                f"Posición: {self.position.round(2).tolist()} | "
                f"Meta: {self.goal.round(2).tolist()} | "
                f"Ángulo: {round(self.angle, 2)} | "
                f"Última recompensa: {round(self.last_reward, 3)}"
            )

        image = self._build_rgb_array()
        if mode == "rgb_array":
            return image

        if mode == "human":
            try:
                import matplotlib.pyplot as plt
            except ImportError as exc:
                raise RuntimeError("Instala matplotlib para usar render(mode='human').") from exc
            plt.imshow(image)
            plt.title("Entorno de navegación autónoma")
            plt.axis("off")
            plt.show()
            return None

        raise ValueError(f"Modo de render inválido: {mode}.")

    def sample_action(self) -> int:
        """Devuelve una acción aleatoria válida usando el generador del entorno."""
        action_count = len(self.config.action_to_rotation)
        return int(self.rng.integers(0, action_count))

    def clear_sand(self) -> None:
        """Elimina todos los obstáculos de arena."""
        self.sand = np.zeros((self.width, self.height), dtype=np.float32)

    def add_sand_rect(self, x0: int, y0: int, x1: int, y1: int) -> None:
        """Agrega un obstáculo rectangular de arena."""
        x0, x1 = sorted((self._clip_x(x0), self._clip_x(x1)))
        y0, y1 = sorted((self._clip_y(y0), self._clip_y(y1)))
        self.sand[x0:x1, y0:y1] = 1.0

    def add_sand_line(self, points: Iterable[Tuple[float, float]], radius: int = 10) -> None:
        """Dibuja una línea de arena a partir de una secuencia de puntos."""
        points = list(points)
        if len(points) < 2:
            return

        for start, end in zip(points[:-1], points[1:]):
            x0, y0 = start
            x1, y1 = end
            distance = max(int(math.hypot(x1 - x0, y1 - y0)), 1)
            for t in np.linspace(0.0, 1.0, distance + 1):
                x = int(round(x0 + t * (x1 - x0)))
                y = int(round(y0 + t * (y1 - y0)))
                self._paint_sand_disk(x, y, radius)

    def copy(self) -> "NavigationEnv":
        """Crea una copia independiente del entorno para futuros rollouts mentales."""
        return copy.deepcopy(self)

    def _prepare_sand_map(self, sand_map: Optional[Array]) -> Array:
        """Convierte un mapa de arena externo al formato interno."""
        if sand_map is None:
            return np.zeros((self.width, self.height), dtype=np.float32)

        sand = np.asarray(sand_map, dtype=np.float32)
        if sand.shape != (self.width, self.height):
            raise ValueError(
                "El mapa de arena debe tener forma "
                f"({self.width}, {self.height}), pero llegó {sand.shape}."
            )
        return np.clip(sand, 0.0, 1.0)

    def _get_observation(self) -> Array:
        """Calcula el vector de estado observado por el agente."""
        goal_vector = self.goal - self.position
        orientation = self._signed_angle(self.velocity, goal_vector) / 180.0
        return np.array(
            [orientation, self.sensor_signals[0], self.sensor_signals[1], self.sensor_signals[2]],
            dtype=np.float32,
        )

    def _update_sensors(self) -> None:
        """Actualiza la posición y la señal de los tres sensores."""
        angles = [
            self.angle,
            (self.angle + self.config.sensor_angle) % 360.0,
            (self.angle - self.config.sensor_angle) % 360.0,
        ]

        for index, angle in enumerate(angles):
            sensor_vector = self._vector_from_angle(angle, self.config.sensor_distance)
            sensor_position = self.position + sensor_vector
            self.sensor_positions[index] = sensor_position
            self.sensor_signals[index] = self._read_sensor(sensor_position)

    def _read_sensor(self, sensor_position: Array) -> float:
        """Lee la densidad de arena alrededor de un sensor."""
        x, y = int(sensor_position[0]), int(sensor_position[1])
        radius = int(self.config.sensor_radius)

        if x < self.config.margin or x >= self.width - self.config.margin:
            return 1.0
        if y < self.config.margin or y >= self.height - self.config.margin:
            return 1.0

        x0 = max(x - radius, 0)
        x1 = min(x + radius, self.width)
        y0 = max(y - radius, 0)
        y1 = min(y + radius, self.height)
        window = self.sand[x0:x1, y0:y1]
        if window.size == 0:
            return 1.0
        return float(np.mean(window))

    def _distance_to_goal(self) -> float:
        """Calcula la distancia euclidiana entre el auto y la meta."""
        return float(np.linalg.norm(self.position - self.goal))

    def _is_on_sand(self, position: Array) -> bool:
        """Indica si el centro del auto está sobre arena."""
        x = self._clip_x(int(position[0]))
        y = self._clip_y(int(position[1]))
        return bool(self.sand[x, y] > 0.0)

    def _enforce_boundaries(self) -> bool:
        """Mantiene el auto dentro del mapa y detecta colisiones con bordes."""
        collided = False
        margin = self.config.margin

        if self.position[0] < margin:
            self.position[0] = margin
            collided = True
        if self.position[0] > self.width - margin:
            self.position[0] = self.width - margin
            collided = True
        if self.position[1] < margin:
            self.position[1] = margin
            collided = True
        if self.position[1] > self.height - margin:
            self.position[1] = self.height - margin
            collided = True

        return collided

    def _build_rgb_array(self) -> Array:
        """Construye una imagen RGB mínima para depuración visual."""
        image = np.full((self.height, self.width, 3), 255, dtype=np.uint8)

        sand_y, sand_x = np.where(self.sand.T > 0.0)
        image[sand_y, sand_x] = np.array([190, 160, 80], dtype=np.uint8)

        self._draw_disk(image, self.goal, radius=6, color=(60, 180, 75))
        self._draw_disk(image, self.position, radius=5, color=(60, 90, 220))
        for sensor_position in self.sensor_positions:
            self._draw_disk(image, sensor_position, radius=3, color=(220, 60, 60))
        return image

    def _draw_disk(
        self, image: Array, position: Array, radius: int, color: Tuple[int, int, int]
    ) -> None:
        """Dibuja un disco pequeño en una imagen RGB."""
        x_center = int(round(position[0]))
        y_center = int(round(position[1]))
        for x in range(max(0, x_center - radius), min(self.width, x_center + radius + 1)):
            for y in range(max(0, y_center - radius), min(self.height, y_center + radius + 1)):
                if (x - x_center) ** 2 + (y - y_center) ** 2 <= radius**2:
                    image[y, x] = np.array(color, dtype=np.uint8)

    def _paint_sand_disk(self, x_center: int, y_center: int, radius: int) -> None:
        """Pinta un disco de arena en el mapa."""
        for x in range(max(0, x_center - radius), min(self.width, x_center + radius + 1)):
            for y in range(max(0, y_center - radius), min(self.height, y_center + radius + 1)):
                if (x - x_center) ** 2 + (y - y_center) ** 2 <= radius**2:
                    self.sand[x, y] = 1.0

    def _clip_x(self, x: int) -> int:
        """Recorta una coordenada x al rango válido."""
        return int(np.clip(x, 0, self.width - 1))

    def _clip_y(self, y: int) -> int:
        """Recorta una coordenada y al rango válido."""
        return int(np.clip(y, 0, self.height - 1))

    @staticmethod
    def _vector_from_angle(angle_degrees: float, magnitude: float) -> Array:
        """Crea un vector 2D a partir de un ángulo en grados."""
        radians = math.radians(angle_degrees)
        return np.array(
            [math.cos(radians) * magnitude, math.sin(radians) * magnitude], dtype=np.float32
        )

    @staticmethod
    def _signed_angle(source: Array, target: Array) -> float:
        """Calcula el ángulo firmado entre dos vectores en grados."""
        source_norm = float(np.linalg.norm(source))
        target_norm = float(np.linalg.norm(target))
        if source_norm == 0.0 or target_norm == 0.0:
            return 0.0

        source_unit = source / source_norm
        target_unit = target / target_norm
        dot = float(np.clip(np.dot(source_unit, target_unit), -1.0, 1.0))
        cross = float(source_unit[0] * target_unit[1] - source_unit[1] * target_unit[0])
        return math.degrees(math.atan2(cross, dot))


if __name__ == "__main__":
    env = NavigationEnv()
    state = env.reset()
    print("Estado inicial:", state)
    for _ in range(5):
        next_state, reward, done, info = env.step(env.sample_action())
        print("Transición:", next_state, reward, done, info["mensaje"])
        if done:
            break
    print(env.render())
