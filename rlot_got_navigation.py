"""
Proyecto: ¿Cuándo debe pensar un agente RL?
Fase 5: RL-of-Thoughts para navegación y Graph-of-Thoughts operativo.

Esta capa vuelve el proyecto más cercano a RLoT y GoT:
- RLoT: un navegador ligero aprende con Q-learning qué bloque de razonamiento usar.
- GoT: las hipótesis de acción, riesgo, waypoint y reflexión se representan como nodos conectados.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import json
import math
import random

import numpy as np

from map import NavigationEnv
from reasoning_policies import (
    BestOfNActions,
    GraphOfWaypoints,
    RolloutConfig,
    TreeOfActions,
    TreeSearchConfig,
    WaypointGraphConfig,
    available_actions,
    greedy_goal_policy,
    normalize_policy_result,
    rollout_score,
)


PolicyInfo = Dict[str, Any]
PolicyReturn = Tuple[int, PolicyInfo]


@dataclass(frozen=True)
class ChainOfActionsConfig:
    """Configuración para el bloque CHAIN."""

    depth: int = 4
    discount: float = 0.95
    distance_penalty: float = 0.002
    terminal_bonus: float = 2.0


@dataclass(frozen=True)
class ReflectivePolicyConfig:
    """Configuración para el bloque REFLECT."""

    lookahead_depth: int = 2
    danger_penalty: float = 1.5
    distance_penalty: float = 0.002
    safe_margin: float = 0.35


@dataclass(frozen=True)
class GoTNavigationConfig:
    """Configuración para el grafo de pensamientos de navegación."""

    chain_depth: int = 3
    tree_depth: int = 2
    tree_beam_width: int = 2
    max_tree_expansions: int = 80
    waypoint_cell_size: int = 25
    waypoint_stride: int = 3
    reflection_depth: int = 2
    cost_penalty: float = 0.01
    risk_penalty: float = 1.25
    agreement_bonus: float = 0.20


@dataclass(frozen=True)
class RLoTNavigatorConfig:
    """Configuración del navegador RL-of-Thoughts."""

    epsilon: float = 0.20
    alpha: float = 0.15
    gamma: float = 0.90
    cost_penalty: float = 0.01
    seed: int = 123
    block_names: Tuple[str, ...] = ("ACT", "CHAIN", "TREE", "GRAPH", "REFLECT")


@dataclass
class ThoughtNode:
    """Nodo de razonamiento usado por Graph-of-Thoughts de navegación."""

    node_id: str
    kind: str
    action: Optional[int]
    score: float
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ThoughtEdge:
    """Arista entre dos nodos de razonamiento."""

    source: str
    target: str
    relation: str
    weight: float = 1.0


@dataclass
class NavigationThoughtGraph:
    """Grafo explícito de pensamientos de navegación."""

    nodes: Dict[str, ThoughtNode] = field(default_factory=dict)
    edges: List[ThoughtEdge] = field(default_factory=list)

    def add_node(self, node: ThoughtNode) -> None:
        """Agrega o reemplaza un nodo del grafo."""
        self.nodes[node.node_id] = node

    def add_edge(self, source: str, target: str, relation: str, weight: float = 1.0) -> None:
        """Agrega una dependencia dirigida entre pensamientos."""
        if source in self.nodes and target in self.nodes:
            self.edges.append(ThoughtEdge(source=source, target=target, relation=relation, weight=float(weight)))

    def action_scores(self) -> Dict[int, float]:
        """Destila el grafo a puntajes por acción de bajo nivel."""
        scores: Dict[int, float] = {}
        votes: Dict[int, int] = {}
        for node in self.nodes.values():
            if node.action is None:
                continue
            action = int(node.action)
            scores[action] = scores.get(action, 0.0) + float(node.score)
            votes[action] = votes.get(action, 0) + 1

        # Bonificación por consenso: varios pensamientos independientes apoyan la misma acción.
        for action, count in votes.items():
            if count > 1:
                scores[action] += 0.05 * float(count - 1)
        return scores

    def best_action(self, fallback: int = 0) -> Tuple[int, float]:
        """Devuelve la acción con mayor puntaje destilado."""
        scores = self.action_scores()
        if not scores:
            return int(fallback), 0.0
        action = max(scores, key=scores.get)
        return int(action), float(scores[action])

    def to_dict(self) -> Dict[str, Any]:
        """Convierte el grafo a diccionario serializable."""
        return {
            "nodos": [
                {
                    "id": node.node_id,
                    "tipo": node.kind,
                    "accion": node.action,
                    "puntaje": float(node.score),
                    "detalles": node.details,
                }
                for node in self.nodes.values()
            ],
            "aristas": [
                {
                    "origen": edge.source,
                    "destino": edge.target,
                    "relacion": edge.relation,
                    "peso": float(edge.weight),
                }
                for edge in self.edges
            ],
            "puntajes_por_accion": self.action_scores(),
        }


class ChainOfActionsPolicy:
    """Bloque CHAIN: razona como una cadena de acciones futuras."""

    def __init__(self, config: Optional[ChainOfActionsConfig] = None) -> None:
        self.config = config or ChainOfActionsConfig()

    def select_action(self, env: NavigationEnv, state: np.ndarray) -> PolicyReturn:
        """Escoge la primera acción de la mejor cadena simulada."""
        rollout_config = RolloutConfig(
            depth=self.config.depth,
            samples_per_action=1,
            discount=self.config.discount,
            distance_penalty=self.config.distance_penalty,
            terminal_bonus=self.config.terminal_bonus,
        )
        scores: Dict[int, float] = {}
        simulated_steps = 0
        for action in available_actions(env):
            score, steps, _details = rollout_score(env, action, rollout_config, greedy_goal_policy)
            scores[int(action)] = float(score)
            simulated_steps += int(steps)
        selected_action = max(scores, key=scores.get)
        return int(selected_action), {
            "mensaje": "Acción elegida con Chain-of-Actions.",
            "metodo_razonamiento": "Chain-of-Actions",
            "bloque_razonamiento": "CHAIN",
            "costo_decision": float(simulated_steps),
            "puntajes_cadena": scores,
            "profundidad_cadena": float(self.config.depth),
        }


class ReflectiveActionPolicy:
    """Bloque REFLECT: propone una acción y la revisa antes de ejecutarla."""

    def __init__(self, config: Optional[ReflectivePolicyConfig] = None) -> None:
        self.config = config or ReflectivePolicyConfig()

    def select_action(self, env: NavigationEnv, state: np.ndarray) -> PolicyReturn:
        """Evita acciones que parecen llevar a arena o borde en un lookahead corto."""
        proposed_action, proposed_info = greedy_goal_policy(env, state)
        proposed_score, proposed_risk, proposed_steps = self._reflect(env, proposed_action)

        if proposed_risk <= self.config.safe_margin:
            return int(proposed_action), {
                "mensaje": "La reflexión aceptó la acción directa.",
                "metodo_razonamiento": "Reflective Action",
                "bloque_razonamiento": "REFLECT",
                "accion_propuesta": int(proposed_action),
                "riesgo_estimado": float(proposed_risk),
                "puntaje_reflexion": float(proposed_score),
                "costo_decision": float(proposed_steps + 1),
                "detalle_accion_directa": proposed_info,
            }

        candidate_scores: Dict[int, float] = {}
        candidate_risks: Dict[int, float] = {}
        total_steps = proposed_steps
        for action in available_actions(env):
            score, risk, steps = self._reflect(env, action)
            candidate_scores[int(action)] = float(score)
            candidate_risks[int(action)] = float(risk)
            total_steps += steps
        selected_action = max(candidate_scores, key=candidate_scores.get)
        return int(selected_action), {
            "mensaje": "La reflexión rechazó la acción directa y eligió una alternativa.",
            "metodo_razonamiento": "Reflective Action",
            "bloque_razonamiento": "REFLECT",
            "accion_propuesta": int(proposed_action),
            "riesgo_propuesto": float(proposed_risk),
            "riesgos_por_accion": candidate_risks,
            "puntajes_reflexion": candidate_scores,
            "costo_decision": float(total_steps + 1),
        }

    def _reflect(self, env: NavigationEnv, action: int) -> Tuple[float, float, int]:
        """Simula una acción y estima riesgo local."""
        simulated_env = env.copy()
        total_reward = 0.0
        risk = 0.0
        steps = 0
        current_action = int(action)
        for depth_index in range(max(self.config.lookahead_depth, 1)):
            state, reward, done, info = simulated_env.step(current_action)
            steps += 1
            total_reward += float(reward)
            if bool(info.get("sobre_arena", False)):
                risk += 0.60
            if bool(info.get("colision_borde", False)):
                risk += 0.75
            if bool(info.get("meta_alcanzada", False)):
                total_reward += 2.0
            if done:
                break
            if depth_index == 0:
                current_action, _ = greedy_goal_policy(simulated_env, state)
        distance = float(np.linalg.norm(simulated_env.position - simulated_env.goal))
        score = total_reward - self.config.danger_penalty * risk - self.config.distance_penalty * distance
        return float(score), float(min(risk, 1.0)), int(steps)


class GoTNavigationGraphPolicy:
    """Graph-of-Thoughts aplicado a navegación autónoma."""

    def __init__(self, config: Optional[GoTNavigationConfig] = None) -> None:
        self.config = config or GoTNavigationConfig()
        self.chain_policy = ChainOfActionsPolicy(ChainOfActionsConfig(depth=self.config.chain_depth))
        self.tree_policy = TreeOfActions(
            TreeSearchConfig(
                depth=self.config.tree_depth,
                beam_width=self.config.tree_beam_width,
                max_expansions=self.config.max_tree_expansions,
            )
        )
        self.waypoint_policy = GraphOfWaypoints(
            WaypointGraphConfig(
                cell_size=self.config.waypoint_cell_size,
                waypoint_stride=self.config.waypoint_stride,
            )
        )
        self.reflect_policy = ReflectiveActionPolicy(ReflectivePolicyConfig(lookahead_depth=self.config.reflection_depth))
        self.last_graph = NavigationThoughtGraph()

    def select_action(self, env: NavigationEnv, state: np.ndarray) -> PolicyReturn:
        """Construye un grafo de pensamientos y destila una acción."""
        graph = NavigationThoughtGraph()
        graph.add_node(ThoughtNode("estado", "estado_actual", None, 0.0, self._state_details(env, state)))

        total_cost = 0.0
        recommendations: List[Tuple[str, int, float, Dict[str, Any]]] = []

        action, info = greedy_goal_policy(env, state)
        recommendations.append(("actuar", action, self._immediate_score(env, action), info))
        total_cost += 1.0

        action, info = self.chain_policy.select_action(env, state)
        recommendations.append(("cadena", action, self._score_from_info(info, "puntajes_cadena", action), info))
        total_cost += float(info.get("costo_decision", 1.0))

        action, info = self.tree_policy.select_action(env, state)
        recommendations.append(("arbol", action, self._score_from_info(info, "puntajes_raiz", action), info))
        total_cost += float(info.get("costo_decision", 1.0))

        action, info = self.waypoint_policy.select_action(env, state)
        waypoint_score = self._immediate_score(env, action) + 0.10 * float(info.get("longitud_plan", 0.0) > 0.0)
        recommendations.append(("grafo_waypoints", action, waypoint_score, info))
        total_cost += float(info.get("costo_decision", 1.0))

        action, info = self.reflect_policy.select_action(env, state)
        reflect_score = self._score_from_info(info, "puntajes_reflexion", action, default=float(info.get("puntaje_reflexion", 0.0)))
        recommendations.append(("reflexion", action, reflect_score, info))
        total_cost += float(info.get("costo_decision", 1.0))

        # Nodos de recomendación: cada bloque aporta un pensamiento parcial.
        for index, (kind, action, score, info) in enumerate(recommendations):
            node_id = f"{kind}_{index}"
            adjusted_score = float(score) - self.config.cost_penalty * float(info.get("costo_decision", 1.0))
            graph.add_node(
                ThoughtNode(
                    node_id=node_id,
                    kind=kind,
                    action=int(action),
                    score=adjusted_score,
                    details=self._compact_info(info),
                )
            )
            graph.add_edge("estado", node_id, "genera", 1.0)

            risk_score, risk_level, risk_steps = self._risk_node(env, action)
            risk_node_id = f"riesgo_{index}"
            graph.add_node(
                ThoughtNode(
                    node_id=risk_node_id,
                    kind="evaluacion_riesgo",
                    action=int(action),
                    score=float(risk_score),
                    details={"riesgo_estimado": float(risk_level), "pasos_simulados": float(risk_steps)},
                )
            )
            graph.add_edge(node_id, risk_node_id, "critica", 1.0)
            total_cost += risk_steps

        # Aristas de consenso: si dos bloques recomiendan lo mismo, se refuerzan mutuamente.
        for left_index, (left_kind, left_action, _left_score, _left_info) in enumerate(recommendations):
            for right_index, (right_kind, right_action, _right_score, _right_info) in enumerate(recommendations):
                if left_index >= right_index or left_action != right_action:
                    continue
                left_id = f"{left_kind}_{left_index}"
                right_id = f"{right_kind}_{right_index}"
                graph.add_edge(left_id, right_id, "consenso", self.config.agreement_bonus)
                if left_id in graph.nodes:
                    graph.nodes[left_id].score += self.config.agreement_bonus
                if right_id in graph.nodes:
                    graph.nodes[right_id].score += self.config.agreement_bonus

        fallback_action, _ = greedy_goal_policy(env, state)
        selected_action, selected_score = graph.best_action(fallback=fallback_action)
        self.last_graph = graph
        return int(selected_action), {
            "mensaje": "Acción elegida con Graph-of-Thoughts de navegación.",
            "metodo_razonamiento": "GoT Navigation Graph",
            "bloque_razonamiento": "GRAPH",
            "accion_destilada": int(selected_action),
            "puntaje_destilado": float(selected_score),
            "costo_decision": float(total_cost),
            "grafo_pensamientos": graph.to_dict(),
            "cantidad_nodos": float(len(graph.nodes)),
            "cantidad_aristas": float(len(graph.edges)),
        }

    def _state_details(self, env: NavigationEnv, state: np.ndarray) -> Dict[str, Any]:
        """Resume el estado actual para el nodo raíz."""
        return {
            "posicion": env.position.astype(float).round(3).tolist(),
            "meta": env.goal.astype(float).round(3).tolist(),
            "angulo": float(env.angle),
            "estado": np.asarray(state, dtype=float).round(4).tolist(),
            "distancia_meta": float(np.linalg.norm(env.position - env.goal)),
        }

    def _compact_info(self, info: Mapping[str, Any]) -> Dict[str, Any]:
        """Reduce información para que el grafo sea legible y serializable."""
        compact: Dict[str, Any] = {}
        keep = [
            "metodo_razonamiento",
            "bloque_razonamiento",
            "costo_decision",
            "error_angular",
            "longitud_plan",
            "cantidad_waypoints",
            "riesgo_estimado",
            "accion_propuesta",
        ]
        for key in keep:
            if key in info:
                compact[key] = info[key]
        return compact

    def _score_from_info(self, info: Mapping[str, Any], key: str, action: int, default: float = 0.0) -> float:
        """Extrae puntaje de un diccionario de resultados."""
        raw = info.get(key, {})
        if isinstance(raw, Mapping):
            return float(raw.get(int(action), raw.get(str(action), default)))
        return float(default)

    def _immediate_score(self, env: NavigationEnv, action: int) -> float:
        """Evalúa una acción con una simulación de un paso."""
        simulated_env = env.copy()
        _state, reward, _done, info = simulated_env.step(action)
        score = float(reward)
        if bool(info.get("meta_alcanzada", False)):
            score += 2.0
        score -= 0.002 * float(info.get("distancia_a_meta", 0.0))
        return float(score)

    def _risk_node(self, env: NavigationEnv, action: int) -> Tuple[float, float, int]:
        """Genera el pensamiento crítico de riesgo para una acción."""
        score, risk, steps = self.reflect_policy._reflect(env, action)
        risk_score = -self.config.risk_penalty * float(risk) + 0.25 * float(score)
        return float(risk_score), float(risk), int(steps)


class RLoTNavigator:
    """Navegador ligero que aprende a seleccionar bloques de razonamiento en inferencia."""

    def __init__(self, config: Optional[RLoTNavigatorConfig] = None) -> None:
        self.config = config or RLoTNavigatorConfig()
        self.rng = random.Random(self.config.seed)
        self.blocks = self._build_blocks()
        self.block_names = tuple(name for name in self.config.block_names if name in self.blocks)
        if not self.block_names:
            raise ValueError("Debes configurar al menos un bloque de razonamiento válido.")
        self.q_table: Dict[str, List[float]] = {}

    def select_action(self, env: NavigationEnv, state: np.ndarray, explore: bool = False) -> PolicyReturn:
        """Selecciona bloque de razonamiento y luego acción de navegación."""
        state_key = self.discretize_state(env, state)
        block_index = self.select_block_index(state_key, explore=explore)
        block_name = self.block_names[block_index]
        action, info = normalize_policy_result(self.blocks[block_name](env, state))
        base_cost = float(info.get("costo_decision", 1.0))
        prediction = self._values(state_key)[block_index]
        info.update(
            {
                "mensaje": "Acción elegida por RL-of-Thoughts para navegación.",
                "metodo_razonamiento": "RL-of-Thoughts Navigator",
                "bloque_razonamiento": block_name,
                "estado_discreto_rlot": state_key,
                "indice_bloque_rlot": float(block_index),
                "valor_estimado_rlot": float(prediction),
                "costo_decision": float(base_cost + 1.0),
            }
        )
        return int(action), info

    def select_block_index(self, state_key: str, explore: bool = False) -> int:
        """Escoge un bloque con epsilon-greedy."""
        values = self._values(state_key)
        if explore and self.rng.random() < self.config.epsilon:
            return self.rng.randrange(len(self.block_names))
        return int(np.argmax(values))

    def discretize_state(self, env: NavigationEnv, state: np.ndarray) -> str:
        """Convierte el estado continuo en una clave discreta para Q-learning."""
        orientation = float(abs(state[0]))
        front = float(state[1])
        left = float(state[2])
        right = float(state[3])
        max_sensor = max(front, left, right)
        imbalance = abs(left - right)
        distance = float(np.linalg.norm(env.position - env.goal)) / max(float(env.width + env.height), 1.0)
        orientation_bin = min(int(orientation * 4.0), 3)
        sensor_bin = min(int(max_sensor * 4.0), 3)
        imbalance_bin = min(int(imbalance * 4.0), 3)
        distance_bin = min(int(distance * 5.0), 4)
        return f"o{orientation_bin}_s{sensor_bin}_i{imbalance_bin}_d{distance_bin}"

    def update(self, state_key: str, block_index: int, reward: float, next_state_key: Optional[str] = None) -> None:
        """Actualiza la Q-table con una regla Q-learning compacta."""
        values = self._values(state_key)
        current_value = values[int(block_index)]
        if next_state_key is None:
            target = float(reward)
        else:
            target = float(reward) + float(self.config.gamma) * float(np.max(self._values(next_state_key)))
        values[int(block_index)] = current_value + float(self.config.alpha) * (target - current_value)
        self.q_table[state_key] = values

    def update_from_info(
        self,
        info: Mapping[str, Any],
        reward: float,
        next_state_key: Optional[str] = None,
    ) -> None:
        """Actualiza el navegador usando la información devuelta por select_action()."""
        state_key = str(info.get("estado_discreto_rlot", ""))
        if not state_key:
            return
        block_index = int(float(info.get("indice_bloque_rlot", 0.0)))
        adjusted_reward = float(reward) - float(self.config.cost_penalty) * float(info.get("costo_decision", 1.0))
        self.update(state_key, block_index, adjusted_reward, next_state_key=next_state_key)

    def save(self, path: str | Path) -> None:
        """Guarda el navegador RLoT en JSON."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "mensaje": "Navegador RL-of-Thoughts guardado correctamente.",
            "block_names": list(self.block_names),
            "config": {
                "epsilon": self.config.epsilon,
                "alpha": self.config.alpha,
                "gamma": self.config.gamma,
                "cost_penalty": self.config.cost_penalty,
                "seed": self.config.seed,
            },
            "q_table": {key: [float(value) for value in values] for key, values in self.q_table.items()},
        }
        target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def load(self, path: str | Path) -> None:
        """Carga la Q-table desde JSON."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        loaded_blocks = tuple(payload.get("block_names", []))
        if loaded_blocks != self.block_names:
            raise ValueError(f"Bloques incompatibles: {loaded_blocks} != {self.block_names}.")
        q_table = payload.get("q_table", {})
        self.q_table = {str(key): [float(value) for value in values] for key, values in q_table.items()}

    def _values(self, state_key: str) -> List[float]:
        """Obtiene valores Q para un estado discreto, inicializando sesgos suaves."""
        if state_key not in self.q_table:
            # Sesgo inicial: actuar cuesta menos, GRAPH/REFLECT se reservan para estados más difíciles.
            initial = [-0.01 * index for index in range(len(self.block_names))]
            self.q_table[state_key] = [float(value) for value in initial]
        return self.q_table[state_key]

    def _build_blocks(self) -> Dict[str, Any]:
        """Construye los bloques disponibles para el navegador."""
        chain = ChainOfActionsPolicy(ChainOfActionsConfig(depth=4))
        tree = TreeOfActions(TreeSearchConfig(depth=3, beam_width=3, max_expansions=120))
        graph = GoTNavigationGraphPolicy(GoTNavigationConfig())
        reflect = ReflectiveActionPolicy(ReflectivePolicyConfig())
        return {
            "ACT": greedy_goal_policy,
            "CHAIN": chain.select_action,
            "TREE": tree.select_action,
            "GRAPH": graph.select_action,
            "REFLECT": reflect.select_action,
        }


def build_rlot_and_got_policies(model_path: Optional[Path] = None) -> Dict[str, Any]:
    """Construye políticas nuevas de Fase 5 para scripts de evaluación."""
    got_policy = GoTNavigationGraphPolicy(GoTNavigationConfig())
    rlot = RLoTNavigator(RLoTNavigatorConfig(epsilon=0.0, seed=123))
    if model_path is not None and model_path.exists():
        rlot.load(model_path)
    return {
        "GoT Navigation Graph": got_policy.select_action,
        "RL-of-Thoughts Navigator": rlot.select_action,
    }
