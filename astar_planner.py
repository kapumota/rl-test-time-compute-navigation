"""
Proyecto: ¿Cuándo debe pensar un agente RL?
Baseline A* para navegación con mapa discretizado.

A* no aprende. Usa el mapa de arena como ocupación y calcula una ruta discreta
hacia la meta. Después traduce el siguiente waypoint a una acción del auto:
recto, giro a la izquierda o giro a la derecha.
"""

from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from typing import Dict, Iterable, List, Optional, Tuple
import math

import numpy as np

from map import NavigationEnv

GridCell = Tuple[int, int]


@dataclass(frozen=True)
class AStarConfig:
    """Configuración del planificador A*."""

    cell_size: int = 20
    sand_threshold: float = 0.15
    replan_each_step: bool = True
    turn_tolerance_degrees: float = 12.0
    obstacle_margin_cells: int = 0


class AStarPlanner:
    """Planificador clásico A* sobre una discretización del mapa."""

    def __init__(self, config: Optional[AStarConfig] = None) -> None:
        self.config = config or AStarConfig()
        self.cached_path: List[GridCell] = []
        self.last_expanded_nodes = 0
        self.last_path_length = 0

    def select_action(
        self, env: NavigationEnv, state: Optional[np.ndarray] = None
    ) -> Tuple[int, Dict[str, float]]:
        """Devuelve la acción de bajo nivel sugerida por A*."""
        if self.config.replan_each_step or not self.cached_path:
            self.cached_path = self.plan(env)

        if len(self.cached_path) <= 1:
            return 0, {
                "mensaje": "A* no encontró una ruta útil; se avanza recto como respaldo.",
                "costo_decision": float(self.last_expanded_nodes),
                "longitud_plan": float(self.last_path_length),
            }

        current_cell = self._position_to_cell(env, env.position)
        path = self.cached_path

        # Se descartan celdas ya alcanzadas para evitar perseguir un waypoint atrasado.
        while len(path) > 1 and path[0] == current_cell:
            path = path[1:]
        self.cached_path = path

        target_cell = path[0] if path[0] != current_cell else path[min(1, len(path) - 1)]
        target_position = self._cell_to_position(env, target_cell)
        desired_angle = math.degrees(
            math.atan2(target_position[1] - env.position[1], target_position[0] - env.position[0])
        )
        angle_error = self._normalize_angle(desired_angle - env.angle)

        if abs(angle_error) <= self.config.turn_tolerance_degrees:
            action = 0
        elif angle_error > 0:
            action = 1
        else:
            action = 2

        info = {
            "mensaje": "Acción calculada con A* sobre mapa discretizado.",
            "costo_decision": float(self.last_expanded_nodes),
            "longitud_plan": float(self.last_path_length),
            "error_angular": float(angle_error),
        }
        return action, info

    def plan(self, env: NavigationEnv) -> List[GridCell]:
        """Calcula una ruta desde la posición actual hasta la meta."""
        grid = self._build_occupancy_grid(env)
        start = self._position_to_cell(env, env.position)
        goal = self._position_to_cell(env, env.goal)

        # La meta del entorno original está cerca del borde. Se liberan explícitamente
        # las celdas de inicio y meta para no bloquear un episodio válido por margen.
        if 0 <= start[0] < grid.shape[0] and 0 <= start[1] < grid.shape[1]:
            grid[start] = False
        if 0 <= goal[0] < grid.shape[0] and 0 <= goal[1] < grid.shape[1]:
            grid[goal] = False

        if not self._is_valid_cell(grid, start) or not self._is_valid_cell(grid, goal):
            self.last_expanded_nodes = 0
            self.last_path_length = 0
            return []

        frontier: List[Tuple[float, GridCell]] = []
        heappush(frontier, (0.0, start))
        came_from: Dict[GridCell, Optional[GridCell]] = {start: None}
        cost_so_far: Dict[GridCell, float] = {start: 0.0}
        self.last_expanded_nodes = 0

        while frontier:
            _, current = heappop(frontier)
            self.last_expanded_nodes += 1

            if current == goal:
                break

            for neighbor, move_cost in self._neighbors(grid, current):
                new_cost = cost_so_far[current] + move_cost
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + self._heuristic(neighbor, goal)
                    heappush(frontier, (priority, neighbor))
                    came_from[neighbor] = current

        if goal not in came_from:
            self.last_path_length = 0
            return []

        path = self._reconstruct_path(came_from, goal)
        self.last_path_length = len(path)
        return path

    def _build_occupancy_grid(self, env: NavigationEnv) -> np.ndarray:
        """Convierte el mapa continuo de arena en una celda ocupada/libre."""
        cell = int(self.config.cell_size)
        grid_width = int(math.ceil(env.width / cell))
        grid_height = int(math.ceil(env.height / cell))
        grid = np.zeros((grid_width, grid_height), dtype=bool)

        for gx in range(grid_width):
            for gy in range(grid_height):
                x0 = gx * cell
                x1 = min((gx + 1) * cell, env.width)
                y0 = gy * cell
                y1 = min((gy + 1) * cell, env.height)
                sand_density = float(np.mean(env.sand[x0:x1, y0:y1]))
                near_border = (
                    x0 < env.config.margin
                    or y0 < env.config.margin
                    or x1 > env.width - env.config.margin
                    or y1 > env.height - env.config.margin
                )
                grid[gx, gy] = sand_density > self.config.sand_threshold or near_border

        if self.config.obstacle_margin_cells > 0:
            grid = self._dilate_obstacles(grid, self.config.obstacle_margin_cells)
        return grid

    def _dilate_obstacles(self, grid: np.ndarray, margin: int) -> np.ndarray:
        """Aumenta el margen alrededor de obstáculos para rutas más conservadoras."""
        expanded = grid.copy()
        occupied = np.argwhere(grid)
        for gx, gy in occupied:
            for dx in range(-margin, margin + 1):
                for dy in range(-margin, margin + 1):
                    nx, ny = gx + dx, gy + dy
                    if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1]:
                        expanded[nx, ny] = True
        return expanded

    def _neighbors(self, grid: np.ndarray, cell: GridCell) -> Iterable[Tuple[GridCell, float]]:
        """Enumera vecinos libres con movimientos de 8 direcciones."""
        x, y = cell
        moves = [
            (-1, 0, 1.0),
            (1, 0, 1.0),
            (0, -1, 1.0),
            (0, 1, 1.0),
            (-1, -1, math.sqrt(2.0)),
            (-1, 1, math.sqrt(2.0)),
            (1, -1, math.sqrt(2.0)),
            (1, 1, math.sqrt(2.0)),
        ]
        for dx, dy, cost in moves:
            neighbor = (x + dx, y + dy)
            if self._is_valid_cell(grid, neighbor):
                yield neighbor, cost

    def _is_valid_cell(self, grid: np.ndarray, cell: GridCell) -> bool:
        """Indica si una celda existe y no está ocupada."""
        x, y = cell
        return 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1] and not bool(grid[x, y])

    def _position_to_cell(self, env: NavigationEnv, position: np.ndarray) -> GridCell:
        """Convierte coordenadas continuas a celda de grilla."""
        cell = int(self.config.cell_size)
        gx = int(np.clip(position[0] // cell, 0, math.ceil(env.width / cell) - 1))
        gy = int(np.clip(position[1] // cell, 0, math.ceil(env.height / cell) - 1))
        return gx, gy

    def _cell_to_position(self, env: NavigationEnv, cell: GridCell) -> np.ndarray:
        """Convierte una celda a su centro aproximado en coordenadas continuas."""
        size = int(self.config.cell_size)
        x = min(cell[0] * size + size / 2.0, env.width - env.config.margin)
        y = min(cell[1] * size + size / 2.0, env.height - env.config.margin)
        return np.array([x, y], dtype=np.float32)

    def _reconstruct_path(
        self, came_from: Dict[GridCell, Optional[GridCell]], goal: GridCell
    ) -> List[GridCell]:
        """Reconstruye la ruta desde el diccionario de padres."""
        current: Optional[GridCell] = goal
        path: List[GridCell] = []
        while current is not None:
            path.append(current)
            current = came_from[current]
        path.reverse()
        return path

    def _heuristic(self, a: GridCell, b: GridCell) -> float:
        """Heurística euclidiana admisible para A*."""
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _normalize_angle(self, angle: float) -> float:
        """Normaliza un ángulo al rango [-180, 180]."""
        return (float(angle) + 180.0) % 360.0 - 180.0
