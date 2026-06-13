"""
Proyecto: ¿Cuándo debe pensar un agente RL?
Fase 3: razonamiento en inferencia para navegación autónoma.

Métodos implementados:
- Best-of-N actions: muestreo y evaluación de acciones en tiempo de prueba.
- Tree-of-Actions: búsqueda deliberativa inspirada en Tree of Thoughts.
- Graph-of-Waypoints: planificación sobre grafo de puntos de ruta inspirada en Graph of Thoughts.
- Adaptive rollout budget: presupuesto de cómputo adaptativo según dificultad local.
- Learned reasoning controller: controlador ligero que aprende qué bloque de razonamiento usar.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple
import json
import math
import random

import numpy as np

from astar_planner import AStarConfig, AStarPlanner
from map import NavigationEnv

PolicyResult = int | Tuple[int, Dict[str, Any]]
PolicyCallable = Callable[[NavigationEnv, np.ndarray], PolicyResult]


@dataclass(frozen=True)
class RolloutConfig:
    """Configuración común para rollouts mentales."""

    depth: int = 5
    samples_per_action: int = 4
    discount: float = 0.95
    distance_penalty: float = 0.002
    terminal_bonus: float = 2.0
    max_simulated_steps: int = 10_000


@dataclass(frozen=True)
class TreeSearchConfig:
    """Configuración para Tree-of-Actions."""

    depth: int = 4
    discount: float = 0.95
    beam_width: int = 3
    distance_penalty: float = 0.002
    max_expansions: int = 250


@dataclass(frozen=True)
class WaypointGraphConfig:
    """Configuración para Graph-of-Waypoints."""

    cell_size: int = 20
    waypoint_stride: int = 4
    sand_threshold: float = 0.15
    turn_tolerance_degrees: float = 12.0
    obstacle_margin_cells: int = 0


@dataclass(frozen=True)
class AdaptiveBudgetConfig:
    """Configuración del presupuesto adaptativo de cómputo."""

    low_depth: int = 1
    medium_depth: int = 3
    high_depth: int = 6
    low_samples: int = 1
    medium_samples: int = 3
    high_samples: int = 5
    medium_threshold: float = 0.35
    high_threshold: float = 0.65
    discount: float = 0.95


@dataclass
class ReasoningControllerConfig:
    """Configuración del controlador aprendido de razonamiento."""

    epsilon: float = 0.15
    learning_rate: float = 0.05
    seed: int = 123
    strategy_names: Tuple[str, ...] = (
        "accion_directa",
        "mejor_de_n",
        "arbol_de_acciones",
        "grafo_de_waypoints",
    )


@dataclass
class StrategyTrace:
    """Registro mínimo para entrenar el controlador aprendido."""

    strategy: str
    features: np.ndarray
    prediction: float


def normalize_policy_result(result: PolicyResult) -> Tuple[int, Dict[str, Any]]:
    """Normaliza una política a una acción entera y un diccionario de información."""
    if isinstance(result, tuple):
        action, info = result
        return int(action), dict(info)
    return int(result), {"costo_decision": 1.0}


def random_policy(env: NavigationEnv, state: np.ndarray) -> Tuple[int, Dict[str, Any]]:
    """Política aleatoria simple usada como respaldo de bajo costo."""
    return env.sample_action(), {"mensaje": "Acción aleatoria usada como respaldo.", "costo_decision": 1.0}


def greedy_goal_policy(env: NavigationEnv, state: np.ndarray) -> Tuple[int, Dict[str, Any]]:
    """Política directa que gira hacia la meta usando la geometría actual del entorno."""
    goal_vector = env.goal - env.position
    desired_angle = math.degrees(math.atan2(goal_vector[1], goal_vector[0]))
    angle_error = normalize_angle(desired_angle - env.angle)

    if abs(angle_error) <= 12.0:
        action = 0
    elif angle_error > 0:
        action = 1
    else:
        action = 2

    return action, {
        "mensaje": "Acción directa orientada a la meta.",
        "costo_decision": 1.0,
        "error_angular": float(angle_error),
    }


def normalize_angle(angle: float) -> float:
    """Normaliza un ángulo al intervalo [-180, 180]."""
    while angle > 180.0:
        angle -= 360.0
    while angle < -180.0:
        angle += 360.0
    return angle


def available_actions(env: NavigationEnv) -> List[int]:
    """Lista las acciones válidas del entorno."""
    return list(range(len(env.config.action_to_rotation)))


def rollout_score(
    env: NavigationEnv,
    first_action: int,
    config: RolloutConfig,
    rollout_policy: PolicyCallable = greedy_goal_policy,
) -> Tuple[float, int, Dict[str, Any]]:
    """Evalúa una acción inicial mediante rollouts simulados sin modificar el entorno real."""
    simulated_env = env.copy()
    total_score = 0.0
    discount = 1.0
    simulated_steps = 0
    done = False

    state, reward, done, info = simulated_env.step(first_action)
    simulated_steps += 1
    total_score += reward

    if bool(info.get("meta_alcanzada", False)):
        total_score += config.terminal_bonus

    for _ in range(max(config.depth - 1, 0)):
        if done:
            break
        action, _ = normalize_policy_result(rollout_policy(simulated_env, state))
        state, reward, done, info = simulated_env.step(action)
        simulated_steps += 1
        discount *= config.discount
        total_score += discount * reward
        if bool(info.get("meta_alcanzada", False)):
            total_score += discount * config.terminal_bonus

    # Moldeado suave: entre dos rutas con recompensas similares, se prefiere quedar más cerca de la meta.
    distance = float(info.get("distancia_a_meta", np.linalg.norm(simulated_env.position - simulated_env.goal)))
    total_score -= config.distance_penalty * distance

    return float(total_score), int(simulated_steps), {
        "distancia_final_simulada": distance,
        "pasos_simulados": float(simulated_steps),
    }


class BestOfNActions:
    """Selecciona la acción con mejor retorno promedio tras N rollouts por acción."""

    def __init__(
        self,
        config: Optional[RolloutConfig] = None,
        rollout_policy: PolicyCallable = greedy_goal_policy,
    ) -> None:
        self.config = config or RolloutConfig()
        self.rollout_policy = rollout_policy

    def select_action(self, env: NavigationEnv, state: np.ndarray) -> Tuple[int, Dict[str, Any]]:
        """Devuelve la mejor acción según muestreo en tiempo de inferencia."""
        action_scores: Dict[int, List[float]] = {action: [] for action in available_actions(env)}
        total_simulated_steps = 0

        for action in action_scores:
            for _ in range(max(self.config.samples_per_action, 1)):
                score, simulated_steps, _ = rollout_score(env, action, self.config, self.rollout_policy)
                action_scores[action].append(score)
                total_simulated_steps += simulated_steps
                if total_simulated_steps >= self.config.max_simulated_steps:
                    break

        mean_scores = {action: float(np.mean(scores)) for action, scores in action_scores.items() if scores}
        selected_action = max(mean_scores, key=mean_scores.get)
        info = {
            "mensaje": "Acción elegida con Best-of-N actions.",
            "metodo_razonamiento": "Best-of-N actions",
            "costo_decision": float(total_simulated_steps),
            "puntajes_promedio": mean_scores,
            "profundidad_rollout": float(self.config.depth),
            "muestras_por_accion": float(self.config.samples_per_action),
        }
        return int(selected_action), info


class TreeOfActions:
    """Búsqueda deliberativa de acciones inspirada en Tree of Thoughts."""

    def __init__(self, config: Optional[TreeSearchConfig] = None) -> None:
        self.config = config or TreeSearchConfig()
        self.last_expansions = 0

    def select_action(self, env: NavigationEnv, state: np.ndarray) -> Tuple[int, Dict[str, Any]]:
        """Explora un árbol de acciones y devuelve la primera acción del mejor camino."""
        self.last_expansions = 0
        scores: Dict[int, float] = {}

        for action in available_actions(env):
            simulated_env = env.copy()
            next_state, reward, done, info = simulated_env.step(action)
            self.last_expansions += 1
            score = float(reward)
            if bool(info.get("meta_alcanzada", False)):
                score += 2.0
            if not done:
                score += self.config.discount * self._value(simulated_env, next_state, self.config.depth - 1)
            score -= self.config.distance_penalty * float(info.get("distancia_a_meta", 0.0))
            scores[action] = score

        selected_action = max(scores, key=scores.get)
        info = {
            "mensaje": "Acción elegida con Tree-of-Actions.",
            "metodo_razonamiento": "Tree-of-Actions",
            "costo_decision": float(self.last_expansions),
            "profundidad_arbol": float(self.config.depth),
            "ancho_haz": float(self.config.beam_width),
            "puntajes_raiz": scores,
        }
        return int(selected_action), info

    def _value(self, env: NavigationEnv, state: np.ndarray, depth: int) -> float:
        """Calcula el valor recursivo de un nodo del árbol de acciones."""
        if depth <= 0 or self.last_expansions >= self.config.max_expansions:
            distance = float(np.linalg.norm(env.position - env.goal))
            return -self.config.distance_penalty * distance

        candidate_scores: List[float] = []
        actions = available_actions(env)[: max(self.config.beam_width, 1)]
        for action in actions:
            if self.last_expansions >= self.config.max_expansions:
                break
            simulated_env = env.copy()
            next_state, reward, done, info = simulated_env.step(action)
            self.last_expansions += 1
            score = float(reward)
            if bool(info.get("meta_alcanzada", False)):
                score += 2.0
            if not done:
                score += self.config.discount * self._value(simulated_env, next_state, depth - 1)
            score -= self.config.distance_penalty * float(info.get("distancia_a_meta", 0.0))
            candidate_scores.append(score)

        if not candidate_scores:
            return 0.0
        return float(max(candidate_scores))


class GraphOfWaypoints:
    """Planificador sobre un grafo discreto de waypoints inspirado en Graph of Thoughts."""

    def __init__(self, config: Optional[WaypointGraphConfig] = None) -> None:
        self.config = config or WaypointGraphConfig()
        self.planner = AStarPlanner(
            AStarConfig(
                cell_size=self.config.cell_size,
                sand_threshold=self.config.sand_threshold,
                turn_tolerance_degrees=self.config.turn_tolerance_degrees,
                obstacle_margin_cells=self.config.obstacle_margin_cells,
            )
        )
        self.last_waypoints: List[Tuple[int, int]] = []

    def select_action(self, env: NavigationEnv, state: np.ndarray) -> Tuple[int, Dict[str, Any]]:
        """Calcula un grafo de waypoints y traduce el siguiente nodo a una acción."""
        path = self.planner.plan(env)
        if not path:
            action, info = greedy_goal_policy(env, state)
            info.update(
                {
                    "mensaje": "Graph-of-Waypoints no encontró ruta; se usa política directa.",
                    "metodo_razonamiento": "Graph-of-Waypoints",
                    "costo_decision": float(self.planner.last_expanded_nodes),
                    "waypoints": [],
                }
            )
            return action, info

        stride = max(int(self.config.waypoint_stride), 1)
        waypoints = path[::stride]
        if waypoints[-1] != path[-1]:
            waypoints.append(path[-1])
        self.last_waypoints = waypoints

        current_cell = self.planner._position_to_cell(env, env.position)
        target_cell = self._choose_next_waypoint(current_cell, waypoints)
        target_position = self.planner._cell_to_position(env, target_cell)
        desired_angle = math.degrees(math.atan2(target_position[1] - env.position[1], target_position[0] - env.position[0]))
        angle_error = self.planner._normalize_angle(desired_angle - env.angle)

        if abs(angle_error) <= self.config.turn_tolerance_degrees:
            action = 0
        elif angle_error > 0:
            action = 1
        else:
            action = 2

        info = {
            "mensaje": "Acción elegida con Graph-of-Waypoints.",
            "metodo_razonamiento": "Graph-of-Waypoints",
            "costo_decision": float(self.planner.last_expanded_nodes),
            "longitud_plan": float(len(path)),
            "cantidad_waypoints": float(len(waypoints)),
            "waypoint_objetivo": tuple(map(int, target_cell)),
            "error_angular": float(angle_error),
        }
        return int(action), info

    def _choose_next_waypoint(self, current_cell: Tuple[int, int], waypoints: Sequence[Tuple[int, int]]) -> Tuple[int, int]:
        """Escoge el primer waypoint que aún no coincide con la celda actual."""
        for waypoint in waypoints:
            if waypoint != current_cell:
                return waypoint
        return waypoints[-1]


class AdaptiveRolloutBudget:
    """Ajusta el presupuesto de rollouts según señales de dificultad local."""

    def __init__(
        self,
        config: Optional[AdaptiveBudgetConfig] = None,
        rollout_policy: PolicyCallable = greedy_goal_policy,
    ) -> None:
        self.config = config or AdaptiveBudgetConfig()
        self.rollout_policy = rollout_policy

    def estimate_difficulty(self, env: NavigationEnv, state: np.ndarray) -> float:
        """Estima dificultad a partir de orientación, sensores y proximidad a obstáculos."""
        orientation_error = float(abs(state[0]))
        sensor_front = float(state[1])
        sensor_left = float(state[2])
        sensor_right = float(state[3])
        max_sensor = max(sensor_front, sensor_left, sensor_right)
        sensor_imbalance = abs(sensor_left - sensor_right)

        raw = 0.45 * max_sensor + 0.35 * orientation_error + 0.20 * sensor_imbalance
        return float(np.clip(raw, 0.0, 1.0))

    def select_budget(self, difficulty: float) -> RolloutConfig:
        """Convierte dificultad estimada en profundidad y número de muestras."""
        if difficulty >= self.config.high_threshold:
            return RolloutConfig(depth=self.config.high_depth, samples_per_action=self.config.high_samples, discount=self.config.discount)
        if difficulty >= self.config.medium_threshold:
            return RolloutConfig(depth=self.config.medium_depth, samples_per_action=self.config.medium_samples, discount=self.config.discount)
        return RolloutConfig(depth=self.config.low_depth, samples_per_action=self.config.low_samples, discount=self.config.discount)

    def select_action(self, env: NavigationEnv, state: np.ndarray) -> Tuple[int, Dict[str, Any]]:
        """Selecciona acción usando más cómputo solo cuando el estado parece difícil."""
        difficulty = self.estimate_difficulty(env, state)
        budget = self.select_budget(difficulty)
        policy = BestOfNActions(config=budget, rollout_policy=self.rollout_policy)
        action, info = policy.select_action(env, state)
        info.update(
            {
                "mensaje": "Acción elegida con presupuesto adaptativo de rollouts.",
                "metodo_razonamiento": "Adaptive rollout budget",
                "dificultad_estimada": float(difficulty),
            }
        )
        return action, info


class LearnedReasoningController:
    """Controlador ligero que aprende a elegir entre bloques de razonamiento."""

    def __init__(
        self,
        strategies: Mapping[str, PolicyCallable],
        config: Optional[ReasoningControllerConfig] = None,
    ) -> None:
        self.config = config or ReasoningControllerConfig()
        self.strategies = dict(strategies)
        self.strategy_names = tuple(name for name in self.config.strategy_names if name in self.strategies)
        if not self.strategy_names:
            raise ValueError("Debes proporcionar al menos una estrategia válida para el controlador.")
        self.rng = random.Random(self.config.seed)
        self.feature_size = 8
        self.weights = np.zeros((len(self.strategy_names), self.feature_size), dtype=np.float32)

        # Sesgo inicial suave: antes de entrenar, se favorecen estrategias de menor costo.
        for index, name in enumerate(self.strategy_names):
            self.weights[index, 0] = -0.02 * index

    def featurize(self, env: NavigationEnv, state: np.ndarray) -> np.ndarray:
        """Convierte un estado en características para seleccionar estrategia."""
        orientation = float(abs(state[0]))
        sensors = np.asarray(state[1:4], dtype=np.float32)
        distance = float(np.linalg.norm(env.position - env.goal)) / max(float(env.width + env.height), 1.0)
        features = np.array(
            [
                1.0,
                orientation,
                float(sensors[0]),
                float(sensors[1]),
                float(sensors[2]),
                float(np.max(sensors)),
                float(np.mean(sensors)),
                distance,
            ],
            dtype=np.float32,
        )
        return features

    def select_strategy(self, env: NavigationEnv, state: np.ndarray, explore: bool = False) -> StrategyTrace:
        """Elige una estrategia por epsilon-greedy sobre un modelo lineal."""
        features = self.featurize(env, state)
        predictions = self.weights @ features
        if explore and self.rng.random() < self.config.epsilon:
            index = self.rng.randrange(len(self.strategy_names))
        else:
            index = int(np.argmax(predictions))
        strategy = self.strategy_names[index]
        return StrategyTrace(strategy=strategy, features=features, prediction=float(predictions[index]))

    def select_action(self, env: NavigationEnv, state: np.ndarray, explore: bool = False) -> Tuple[int, Dict[str, Any]]:
        """Selecciona estrategia de razonamiento y luego acción de bajo nivel."""
        trace = self.select_strategy(env, state, explore=explore)
        action, info = normalize_policy_result(self.strategies[trace.strategy](env, state))
        info.update(
            {
                "mensaje": "Acción elegida por controlador aprendido de razonamiento.",
                "metodo_razonamiento": "Learned reasoning controller",
                "estrategia": trace.strategy,
                "prediccion_controlador": float(trace.prediction),
                "features_controlador": trace.features.astype(float).tolist(),
                "costo_decision": float(info.get("costo_decision", 1.0)) + 1.0,
            }
        )
        return action, info

    def update(self, trace: StrategyTrace, reward: float) -> None:
        """Actualiza la estrategia elegida con una regla lineal tipo bandido contextual."""
        index = self.strategy_names.index(trace.strategy)
        prediction = float(self.weights[index] @ trace.features)
        error = float(reward) - prediction
        self.weights[index] += float(self.config.learning_rate) * error * trace.features

    def update_from_info(self, info: Mapping[str, Any], reward: float) -> None:
        """Actualiza usando la información devuelta por select_action()."""
        strategy = str(info.get("estrategia", ""))
        if strategy not in self.strategy_names:
            return
        features = np.asarray(info.get("features_controlador", []), dtype=np.float32)
        if features.shape != (self.feature_size,):
            return
        prediction = float(info.get("prediccion_controlador", 0.0))
        self.update(StrategyTrace(strategy=strategy, features=features, prediction=prediction), reward)

    def save(self, path: str | Path) -> None:
        """Guarda pesos y metadatos del controlador."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "mensaje": "Controlador de razonamiento guardado correctamente.",
            "strategy_names": list(self.strategy_names),
            "weights": self.weights.astype(float).tolist(),
            "feature_size": self.feature_size,
        }
        target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def load(self, path: str | Path) -> None:
        """Carga pesos del controlador desde JSON."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        loaded_names = tuple(payload["strategy_names"])
        if loaded_names != self.strategy_names:
            raise ValueError(
                "Las estrategias del archivo no coinciden con las estrategias actuales: "
                f"{loaded_names} != {self.strategy_names}"
            )
        weights = np.asarray(payload["weights"], dtype=np.float32)
        if weights.shape != self.weights.shape:
            raise ValueError(f"Forma de pesos inválida: {weights.shape}, esperada {self.weights.shape}.")
        self.weights = weights


def build_reasoning_strategies() -> Dict[str, PolicyCallable]:
    """Construye el conjunto estándar de estrategias de razonamiento de Fase 3."""
    best_of_n = BestOfNActions(RolloutConfig(depth=4, samples_per_action=3))
    tree = TreeOfActions(TreeSearchConfig(depth=3, beam_width=3, max_expansions=120))
    graph = GraphOfWaypoints(WaypointGraphConfig(cell_size=20, waypoint_stride=4))

    return {
        "accion_directa": greedy_goal_policy,
        "mejor_de_n": best_of_n.select_action,
        "arbol_de_acciones": tree.select_action,
        "grafo_de_waypoints": graph.select_action,
    }
