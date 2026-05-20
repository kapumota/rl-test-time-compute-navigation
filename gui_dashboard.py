"""
Proyecto: ¿Cuándo debe pensar un agente RL?
Fase 6: interfaz gráfica en Kivy para visualizar razonamiento en inferencia.

Este módulo mantiene la lógica de RL separada de la interfaz. La clase
ReasoningDashboardController no depende de Kivy y puede probarse en CI. La app
Kivy solo se importa cuando se ejecuta este archivo directamente.

La versión extendida agrega comparación lado a lado, heatmap de cómputo,
grabación de episodios, editor de mapas, modo paper/demostracion y replay determinista.

"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import ast
import csv
import json
import math

import numpy as np

from evaluation_scenarios import apply_dynamic_goal_if_needed, build_scenario_env, list_scenarios
from map import NavigationEnv
from reasoning_policies import greedy_goal_policy, normalize_policy_result
from run_reasoning_experiments import build_policies
from analyze_overthinking import summarize_overthinking, write_summary_csv, write_summary_json

PolicyCallable = Callable[[NavigationEnv, np.ndarray], int | Tuple[int, Dict[str, Any]]]

TITULO_INTERFAZ = "¿Cuándo debe pensar un agente RL?"
SUBTITULO_INTERFAZ = "Panel Kivy para RLoT, GoT y cómputo adaptativo en navegación autónoma"
DEFAULT_COMPARISON_METHODS: Tuple[str, str, str] = (
    "RL-of-Thoughts Navigator",
    "GoT Navigation Graph",
    "Adaptive rollout budget",
)


@dataclass
class DashboardState:
    """Estado visible del panel de control."""

    escenario: str = "facil"
    metodo: str = "RL-of-Thoughts Navigator"
    recompensa_total: float = 0.0
    pasos: int = 0
    metas_alcanzadas: int = 0
    pasos_en_arena: int = 0
    colisiones_borde: int = 0
    costo_total: float = 0.0
    accion: int = 0
    bloque_razonamiento: str = "ACT"
    metodo_razonamiento: str = "Acción directa"
    mensaje: str = "Panel inicializado."
    ultima_info: Dict[str, Any] = field(default_factory=dict)

    @property
    def costo_promedio(self) -> float:
        """Devuelve el costo promedio por decisión."""
        return float(self.costo_total / max(self.pasos, 1))


@dataclass
class AgentTrack:
    """Estado independiente de un agente usado en comparación lado a lado."""

    method: str
    env: NavigationEnv
    observation: np.ndarray
    scenario_instance: Any
    recompensa_total: float = 0.0
    pasos: int = 0
    metas_alcanzadas: int = 0
    pasos_en_arena: int = 0
    colisiones_borde: int = 0
    costo_total: float = 0.0
    accion: int = 0
    bloque_razonamiento: str = "ACT"
    metodo_razonamiento: str = ""
    mensaje: str = ""
    done: bool = False
    path: List[Tuple[float, float]] = field(default_factory=list)
    events: List[Dict[str, Any]] = field(default_factory=list)
    ultima_info: Dict[str, Any] = field(default_factory=dict)

    @property
    def costo_promedio(self) -> float:
        """Devuelve el costo promedio por decisión para la pista."""
        return float(self.costo_total / max(self.pasos, 1))

    def to_summary(self) -> Dict[str, Any]:
        """Convierte la pista a un resumen serializable."""
        distance = float(np.linalg.norm(self.env.position - self.env.goal))
        return {
            "metodo": self.method,
            "pasos": int(self.pasos),
            "recompensa_total": float(self.recompensa_total),
            "costo_total": float(self.costo_total),
            "costo_promedio": float(self.costo_promedio),
            "metas_alcanzadas": int(self.metas_alcanzadas),
            "pasos_en_arena": int(self.pasos_en_arena),
            "colisiones_borde": int(self.colisiones_borde),
            "distancia_a_meta": distance,
            "accion": int(self.accion),
            "bloque_razonamiento": self.bloque_razonamiento,
            "done": bool(self.done),
            "posicion": self.env.position.astype(float).round(3).tolist(),
        }


class ReasoningDashboardController:
    """Controlador independiente de Kivy para la interfaz de razonamiento."""

    def __init__(self, seed: int = 123, max_steps: int = 500) -> None:
        self.seed = int(seed)
        self.max_steps = int(max_steps)
        self.episode = 1
        self.scenarios = list_scenarios()
        self.policies = self._build_dashboard_policies()
        self.state = DashboardState(
            escenario=self.scenarios[0] if self.scenarios else "facil",
            metodo=self._default_method(),
        )
        self.env: NavigationEnv
        self.observation: np.ndarray
        self.scenario_instance: Any = None

        self.comparison_methods: Tuple[str, ...] = tuple(
            method for method in DEFAULT_COMPARISON_METHODS if method in self.policies
        )
        self.comparison_enabled = False
        self.comparison_tracks: Dict[str, AgentTrack] = {}

        self.show_heatmap = True
        self.heatmap_cols = 48
        self.heatmap_rows = 36
        self.heatmap = np.zeros((self.heatmap_cols, self.heatmap_rows), dtype=np.float32)
        self.heatmap_counts = np.zeros_like(self.heatmap)
        self.episode_log: List[Dict[str, Any]] = []

        self.map_editor_enabled = False
        self.editor_tool = "paint"
        self.editor_radius = 12
        self.custom_sand_map: Optional[np.ndarray] = None
        self.last_exported_artifacts: Dict[str, str] = {}
        self.reset()

    def reset(self) -> np.ndarray:
        """Reinicia el entorno del escenario actual."""
        self._reset_heatmap()
        self.episode_log = []
        self.scenario_instance = self._build_scenario_instance(self.episode)
        self.env = self.scenario_instance.env
        self._apply_custom_map(self.env)
        self.observation = self.env.reset(
            seed=self._reset_seed(self.episode),
            options=self.scenario_instance.reset_options,
        )
        self._reset_visible_state()
        self._record_path_event(
            method=self.state.metodo,
            env=self.env,
            step=0,
            reward=0.0,
            cost=0.0,
            action=self.state.accion,
            decision_info={"mensaje": "Inicio de episodio."},
            env_info={"mensaje": "Inicio de episodio."},
        )
        if self.comparison_enabled:
            self._reset_comparison_tracks()
            self._sync_primary_env_from_comparison()
        return self.observation

    def step(self) -> DashboardState:
        """Ejecuta una decisión de razonamiento y un paso del entorno."""
        if self.comparison_enabled:
            return self.step_comparison()

        policy = self.policies.get(self.state.metodo, greedy_goal_policy)
        action, decision_info = normalize_policy_result(policy(self.env, self.observation))
        position_before = self.env.position.copy()
        next_state, reward, done, env_info = self.env.step(action)
        dynamic_goal_changed = apply_dynamic_goal_if_needed(self.scenario_instance)
        if dynamic_goal_changed:
            env_info["meta_dinamica_cambiada"] = True

        cost = float(decision_info.get("costo_decision", 1.0))
        self._record_thought_cost(self.env, position_before, cost, self.state.metodo)
        self._update_visible_state(action, reward, decision_info, env_info)
        self._record_path_event(
            method=self.state.metodo,
            env=self.env,
            step=self.state.pasos,
            reward=reward,
            cost=cost,
            action=action,
            decision_info=decision_info,
            env_info=env_info,
        )
        self.observation = next_state

        if done:
            self.episode += 1
            final_message = "Episodio terminado; se reinicia automáticamente."
            self.reset()
            self.state.mensaje = final_message
        return self.state

    def step_comparison(self) -> DashboardState:
        """Avanza RLoT, GoT y Adaptive Budget sobre el mismo mapa inicial."""
        if not self.comparison_tracks:
            self._reset_comparison_tracks()

        for track in self.comparison_tracks.values():
            self._step_track(track)

        self._sync_primary_env_from_comparison()
        self.state.ultima_info = {"comparacion": self.get_comparison_summary()}
        self.state.mensaje = "Comparación lado a lado actualizada."
        return self.state

    def set_scenario(self, scenario_name: str) -> None:
        """Cambia el escenario activo."""
        if scenario_name not in self.scenarios:
            raise ValueError(f"Escenario no reconocido: {scenario_name}")
        self.state.escenario = scenario_name
        self.episode += 1
        self.custom_sand_map = None
        self.reset()

    def set_method(self, method_name: str) -> None:
        """Cambia el método de razonamiento activo."""
        if method_name not in self.policies:
            raise ValueError(f"Método no reconocido: {method_name}")
        self.state.metodo = method_name
        self.state.metodo_razonamiento = method_name
        self.state.mensaje = f"Método activo: {method_name}"

    def set_replay_seed(
        self,
        seed: int,
        scenario_name: Optional[str] = None,
        method_name: Optional[str] = None,
        episode: int = 1,
    ) -> None:
        """Carga una semilla y reinicia el episodio de forma determinista."""
        if scenario_name is not None:
            if scenario_name not in self.scenarios:
                raise ValueError(f"Escenario no reconocido: {scenario_name}")
            self.state.escenario = scenario_name
        if method_name is not None:
            if method_name not in self.policies:
                raise ValueError(f"Método no reconocido: {method_name}")
            self.state.metodo = method_name
        self.seed = int(seed)
        self.episode = int(episode)
        self.reset()
        self.state.mensaje = f"Replay determinista cargado con seed={self.seed}, episodio={self.episode}."

    def enable_comparison(self, enabled: bool = True, methods: Optional[Sequence[str]] = None) -> None:
        """Activa o desactiva la comparación lado a lado en el mismo mapa."""
        if methods is not None:
            missing = [method for method in methods if method not in self.policies]
            if missing:
                raise ValueError(f"Métodos no reconocidos para comparación: {missing}")
            self.comparison_methods = tuple(methods)
        self.comparison_enabled = bool(enabled)
        if self.comparison_enabled:
            self._reset_comparison_tracks()
            self._sync_primary_env_from_comparison()
            self.state.mensaje = "Modo comparación lado a lado activado."
        else:
            self.comparison_tracks = {}
            self.reset()
            self.state.mensaje = "Modo comparación lado a lado desactivado."

    def get_comparison_summary(self) -> List[Dict[str, Any]]:
        """Devuelve métricas actuales de RLoT, GoT y Adaptive Budget."""
        return [track.to_summary() for track in self.comparison_tracks.values()]

    def export_comparison_csv(self, path: str | Path) -> Path:
        """Exporta el historial de comparación a CSV."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "paso",
            "metodo",
            "escenario",
            "seed",
            "recompensa_paso",
            "recompensa_total",
            "costo_decision",
            "costo_total",
            "accion",
            "bloque_razonamiento",
            "posicion_x",
            "posicion_y",
            "distancia_a_meta",
            "sobre_arena",
            "colision_borde",
            "meta_alcanzada",
        ]
        with target.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for event in self.episode_log:
                writer.writerow({key: event.get(key, "") for key in fieldnames})
        return target

    def export_snapshot(self, path: str | Path) -> Path:
        """Guarda un resumen JSON del estado actual del dashboard."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "titulo": TITULO_INTERFAZ,
            "escenario": self.state.escenario,
            "metodo": self.state.metodo,
            "seed": self.seed,
            "episodio": self.episode,
            "comparacion_activada": self.comparison_enabled,
            "metodos_comparacion": list(self.comparison_methods),
            "comparacion": self.get_comparison_summary(),
            "pasos": self.state.pasos,
            "recompensa_total": self.state.recompensa_total,
            "costo_promedio": self.state.costo_promedio,
            "bloque_razonamiento": self.state.bloque_razonamiento,
            "posicion": self.env.position.astype(float).round(3).tolist(),
            "meta": self.env.goal.astype(float).round(3).tolist(),
            "heatmap": self.get_heatmap_matrix(normalize=True).round(4).tolist(),
            "grafo_pensamientos": self.get_thought_graph(),
        }
        target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return target

    def export_replay_config(self, path: str | Path, include_map: bool = True) -> Path:
        """Guarda semilla, escenario, método y mapa para repetir exactamente el episodio."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload: Dict[str, Any] = {
            "version": 1,
            "titulo": TITULO_INTERFAZ,
            "seed": int(self.seed),
            "episodio": int(self.episode),
            "max_steps": int(self.max_steps),
            "escenario": self.state.escenario,
            "metodo": self.state.metodo,
            "comparacion_activada": bool(self.comparison_enabled),
            "metodos_comparacion": list(self.comparison_methods),
            "editor_radius": int(self.editor_radius),
        }
        if include_map:
            payload["sand_rle"] = self._encode_sand_rle(self.env.sand)
        target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return target

    def load_replay_config(self, path: str | Path) -> None:
        """Carga una configuración de replay determinista desde JSON."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        scenario_name = str(payload.get("escenario", self.state.escenario))
        method_name = str(payload.get("metodo", self.state.metodo))
        if scenario_name not in self.scenarios:
            raise ValueError(f"Escenario no reconocido en replay: {scenario_name}")
        if method_name not in self.policies:
            raise ValueError(f"Método no reconocido en replay: {method_name}")

        self.seed = int(payload.get("seed", self.seed))
        self.episode = int(payload.get("episodio", 1))
        self.max_steps = int(payload.get("max_steps", self.max_steps))
        self.state.escenario = scenario_name
        self.state.metodo = method_name
        if "sand_rle" in payload:
            self.custom_sand_map = self._decode_sand_rle(payload["sand_rle"])
        methods = tuple(str(method) for method in payload.get("metodos_comparacion", self.comparison_methods))
        self.comparison_methods = tuple(method for method in methods if method in self.policies)
        self.comparison_enabled = bool(payload.get("comparacion_activada", False))
        self.reset()
        self.state.mensaje = f"Replay cargado desde {path}."

    def metrics_text(self) -> str:
        """Construye el texto compacto de métricas para la interfaz."""
        if self.comparison_enabled:
            return self._comparison_metrics_text()
        distance = float(np.linalg.norm(self.env.position - self.env.goal))
        active_heat = int(np.count_nonzero(self.heatmap))
        return (
            f"Escenario: {self.state.escenario}\n"
            f"Seed replay: {self.seed}\n"
            f"Método: {self.state.metodo}\n"
            f"Bloque elegido: {self.state.bloque_razonamiento}\n"
            f"Acción: {self.state.accion}\n"
            f"Pasos: {self.state.pasos}\n"
            f"Recompensa total: {self.state.recompensa_total:.2f}\n"
            f"Distancia a meta: {distance:.1f}\n"
            f"Costo promedio: {self.state.costo_promedio:.2f}\n"
            f"Celdas heatmap: {active_heat}\n"
            f"Arena: {self.state.pasos_en_arena} | Bordes: {self.state.colisiones_borde}\n"
            f"Metas alcanzadas: {self.state.metas_alcanzadas}\n"
            f"Editor mapa: {'ON' if self.map_editor_enabled else 'OFF'} ({self.editor_tool})\n"
            f"Mensaje: {self.state.mensaje}"
        )

    def get_thought_graph(self) -> Dict[str, Any]:
        """Obtiene el grafo de pensamientos si el método actual lo produjo."""
        decision = self.state.ultima_info.get("decision", {}) if self.state.ultima_info else {}
        graph = decision.get("grafo_pensamientos", {}) if isinstance(decision, Mapping) else {}
        if not graph and self.comparison_enabled:
            for track in self.comparison_tracks.values():
                decision = track.ultima_info.get("decision", {}) if track.ultima_info else {}
                if isinstance(decision, Mapping) and decision.get("grafo_pensamientos"):
                    graph = decision.get("grafo_pensamientos", {})
                    break
        if isinstance(graph, str):
            try:
                graph = ast.literal_eval(graph)
            except (ValueError, SyntaxError):
                graph = {}
        if isinstance(graph, Mapping):
            return dict(graph)
        return {}

    def get_heatmap_matrix(self, normalize: bool = False) -> np.ndarray:
        """Devuelve la matriz de cómputo acumulado por celda."""
        matrix = self.heatmap.copy()
        if normalize and float(np.max(matrix)) > 0.0:
            matrix = matrix / float(np.max(matrix))
        return matrix

    def get_heatmap_cells(self, threshold: float = 0.02) -> List[Dict[str, float]]:
        """Devuelve celdas del heatmap en coordenadas del entorno."""
        env = self.env
        matrix = self.get_heatmap_matrix(normalize=True)
        cell_width = float(env.width) / float(self.heatmap_cols)
        cell_height = float(env.height) / float(self.heatmap_rows)
        cells: List[Dict[str, float]] = []
        for x_index in range(self.heatmap_cols):
            for y_index in range(self.heatmap_rows):
                value = float(matrix[x_index, y_index])
                if value <= threshold:
                    continue
                cells.append(
                    {
                        "x0": x_index * cell_width,
                        "y0": y_index * cell_height,
                        "x1": (x_index + 1) * cell_width,
                        "y1": (y_index + 1) * cell_height,
                        "value": value,
                    }
                )
        return cells

    def set_map_editor(self, enabled: bool, tool: Optional[str] = None, radius: Optional[int] = None) -> None:
        """Activa el editor de mapas y configura herramienta/radio."""
        self.map_editor_enabled = bool(enabled)
        if tool is not None:
            if tool not in {"paint", "erase"}:
                raise ValueError("La herramienta debe ser 'paint' o 'erase'.")
            self.editor_tool = tool
        if radius is not None:
            self.editor_radius = max(int(radius), 1)
        self.state.mensaje = f"Editor de mapas {'activado' if enabled else 'desactivado'} ({self.editor_tool})."

    def paint_obstacle_at(self, x: float, y: float, radius: Optional[int] = None, erase: bool = False) -> None:
        """Dibuja o borra arena en una coordenada del entorno."""
        radius = int(radius or self.editor_radius)
        value = 0.0 if erase else 1.0
        self._paint_sand_disk(self.env, x, y, radius, value)
        for track in self.comparison_tracks.values():
            self._paint_sand_disk(track.env, x, y, radius, value)
        self.custom_sand_map = self.env.sand.copy()
        self._refresh_observation_after_map_edit()
        self.state.mensaje = "Mapa editado: obstáculo borrado." if erase else "Mapa editado: obstáculo pintado."

    def clear_obstacles(self) -> None:
        """Elimina todos los obstáculos del mapa activo."""
        self.env.clear_sand()
        for track in self.comparison_tracks.values():
            track.env.clear_sand()
        self.custom_sand_map = self.env.sand.copy()
        self._refresh_observation_after_map_edit()
        self.state.mensaje = "Mapa limpiado desde el editor."

    def save_map(self, path: str | Path) -> Path:
        """Guarda el mapa de arena editado en NPZ comprimido."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target,
            sand=self.env.sand.astype(np.float32),
            scenario=np.array([self.state.escenario]),
            seed=np.array([self.seed], dtype=np.int64),
        )
        return target

    def load_map(self, path: str | Path) -> None:
        """Carga un mapa de arena NPZ producido por save_map()."""
        data = np.load(Path(path), allow_pickle=False)
        sand = np.asarray(data["sand"], dtype=np.float32)
        if sand.shape != self.env.sand.shape:
            raise ValueError(f"Mapa incompatible: se esperaba {self.env.sand.shape}, llegó {sand.shape}.")
        self.custom_sand_map = np.clip(sand, 0.0, 1.0)
        self._apply_custom_map(self.env)
        for track in self.comparison_tracks.values():
            self._apply_custom_map(track.env)
        self._refresh_observation_after_map_edit()
        self.state.mensaje = f"Mapa cargado desde {path}."

    def render_frame(self, include_heatmap: bool = True, include_paths: bool = True) -> np.ndarray:
        """Renderiza un frame RGB sin depender de Kivy."""
        image = self.env.render(mode="rgb_array").copy()
        if include_heatmap and self.show_heatmap:
            self._overlay_heatmap(image)
        if include_paths:
            self._overlay_paths(image)
        if self.comparison_enabled and self.comparison_tracks:
            for index, track in enumerate(self.comparison_tracks.values()):
                color = self._track_color(index)
                self._draw_disk_rgb(image, track.env.position, radius=7, color=color)
                heading = track.env.position + track.env._vector_from_angle(track.env.angle, 22.0)
                self._draw_line_rgb(image, track.env.position, heading, color=color, radius=1)
        else:
            self._draw_disk_rgb(image, self.env.position, radius=7, color=(30, 80, 230))
        return image

    def record_episode(
        self,
        path: str | Path,
        steps: int = 160,
        fps: int = 12,
        comparison: bool = True,
    ) -> Path:
        """Ejecuta un episodio y exporta GIF o MP4 para README/GitHub."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        imageio = self._load_imageio()

        if comparison:
            self.enable_comparison(True)
        else:
            if self.comparison_enabled:
                self.enable_comparison(False)
            self.reset()

        frames: List[np.ndarray] = [self.render_frame()]
        for _ in range(max(int(steps), 1)):
            self.step()
            frames.append(self.render_frame())
            if self.comparison_enabled and self.comparison_tracks and all(track.done for track in self.comparison_tracks.values()):
                break
        if target.suffix.lower() == ".mp4":
            imageio.mimsave(target, frames, fps=int(fps), macro_block_size=1)
        else:
            imageio.mimsave(target, frames, duration=1000.0 / max(int(fps), 1))
        self.state.mensaje = f"Episodio grabado en {target}."
        return target

    def generate_paper_demo(
        self,
        output_dir: str | Path = "results/paper_demo",
        steps: int = 120,
        seed: Optional[int] = None,
        fps: int = 12,
    ) -> Dict[str, Path]:
        """Genera capturas, CSV, GIF y figuras para documentación o paper/demo."""
        output = Path(output_dir)
        figures_dir = output / "figures"
        captures_dir = output / "captures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        captures_dir.mkdir(parents=True, exist_ok=True)

        if seed is not None:
            self.set_replay_seed(seed, scenario_name=self.state.escenario, method_name=self.state.metodo, episode=1)
        self.enable_comparison(True)

        imageio = self._load_imageio()
        frames: List[np.ndarray] = []
        capture_steps = {0, max(int(steps) // 2, 1), max(int(steps), 1)}
        for step_index in range(max(int(steps), 1) + 1):
            frame = self.render_frame()
            frames.append(frame)
            if step_index in capture_steps:
                imageio.imwrite(captures_dir / f"dashboard_step_{step_index:04d}.png", frame)
            if step_index < steps:
                self.step()

        csv_path = self.export_comparison_csv(output / "comparison_trace.csv")
        overthinking_summaries = summarize_overthinking(self.episode_log)
        overthinking_csv_path = write_summary_csv(overthinking_summaries, output / "overthinking_summary.csv")
        overthinking_json_path = write_summary_json(overthinking_summaries, output / "overthinking_summary.json")
        snapshot_path = self.export_snapshot(output / "dashboard_snapshot.json")
        replay_path = self.export_replay_config(output / "replay_config.json")
        gif_path = output / "episode_comparison.gif"
        imageio.mimsave(gif_path, frames, duration=1000.0 / max(int(fps), 1))

        artifacts: Dict[str, Path] = {
            "csv": csv_path,
            "overthinking_csv": overthinking_csv_path,
            "overthinking_json": overthinking_json_path,
            "snapshot": snapshot_path,
            "replay": replay_path,
            "gif": gif_path,
        }
        artifacts.update(self._generate_demo_figures(figures_dir))
        manifest_path = output / "manifest.json"
        manifest_path.write_text(
            json.dumps({key: str(value) for key, value in artifacts.items()}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        artifacts["manifest"] = manifest_path
        self.last_exported_artifacts = {key: str(value) for key, value in artifacts.items()}
        self.state.mensaje = f"Modo paper/demo generado en {output}."
        return artifacts

    def _default_method(self) -> str:
        """Selecciona un método por defecto útil para demostrar RLoT/GoT."""
        preferred = ["RL-of-Thoughts Navigator", "GoT Navigation Graph", "Adaptive rollout budget"]
        for name in preferred:
            if name in self.policies:
                return name
        return next(iter(self.policies))

    def _build_dashboard_policies(self) -> Dict[str, PolicyCallable]:
        """Construye políticas sin mezclar código visual con lógica de razonamiento."""
        policies = build_policies()
        policies["Acción directa geométrica"] = greedy_goal_policy
        return dict(sorted(policies.items(), key=lambda item: item[0]))

    def _build_scenario_instance(self, episode: int) -> Any:
        """Construye el escenario con las mismas semillas que el dashboard original."""
        return build_scenario_env(
            self.state.escenario,
            seed=self.seed + 10_000 + int(episode),
            max_steps=self.max_steps,
            episode=int(episode),
        )

    def _reset_seed(self, episode: int) -> int:
        """Semilla usada para resetear posición/sensores."""
        return int(self.seed + 20_000 + int(episode))

    def _reset_visible_state(self) -> None:
        """Reinicia métricas visibles del panel."""
        self.state.recompensa_total = 0.0
        self.state.pasos = 0
        self.state.metas_alcanzadas = 0
        self.state.pasos_en_arena = 0
        self.state.colisiones_borde = 0
        self.state.costo_total = 0.0
        self.state.accion = 0
        self.state.bloque_razonamiento = "ACT"
        self.state.metodo_razonamiento = self.state.metodo
        self.state.mensaje = "Episodio reiniciado."
        self.state.ultima_info = {}

    def _update_visible_state(
        self,
        action: int,
        reward: float,
        decision_info: Mapping[str, Any],
        env_info: Mapping[str, Any],
    ) -> None:
        """Actualiza métricas del modo de un solo agente."""
        self.state.accion = int(action)
        self.state.recompensa_total += float(reward)
        self.state.pasos = int(self.env.step_count)
        self.state.costo_total += float(decision_info.get("costo_decision", 1.0))
        self.state.metodo_razonamiento = str(decision_info.get("metodo_razonamiento", self.state.metodo))
        self.state.bloque_razonamiento = str(decision_info.get("bloque_razonamiento", self.state.metodo_razonamiento))
        self.state.mensaje = str(decision_info.get("mensaje", "Decisión ejecutada."))
        self.state.ultima_info = {"decision": dict(decision_info), "entorno": dict(env_info)}
        self.state.pasos_en_arena += int(bool(env_info.get("sobre_arena", False)))
        self.state.colisiones_borde += int(bool(env_info.get("colision_borde", False)))
        self.state.metas_alcanzadas += int(bool(env_info.get("meta_alcanzada", False)))

    def _reset_comparison_tracks(self) -> None:
        """Crea agentes separados que comparten escenario, semilla y mapa inicial."""
        self.comparison_tracks = {}
        for method in self.comparison_methods:
            instance = self._build_scenario_instance(self.episode)
            env = instance.env
            self._apply_custom_map(env)
            observation = env.reset(seed=self._reset_seed(self.episode), options=instance.reset_options)
            track = AgentTrack(
                method=method,
                env=env,
                observation=observation,
                scenario_instance=instance,
                metodo_razonamiento=method,
                path=[tuple(env.position.astype(float))],
            )
            self.comparison_tracks[method] = track
        self.episode_log = []
        self._reset_heatmap()
        for track in self.comparison_tracks.values():
            self._record_path_event(
                method=track.method,
                env=track.env,
                step=0,
                reward=0.0,
                cost=0.0,
                action=0,
                decision_info={"mensaje": "Inicio de comparación."},
                env_info={"mensaje": "Inicio de comparación."},
            )

    def _sync_primary_env_from_comparison(self) -> None:
        """Usa la primera pista como mapa base para dibujar fondo, meta y heatmap."""
        if not self.comparison_tracks:
            return
        primary = next(iter(self.comparison_tracks.values()))
        self.env = primary.env
        self.observation = primary.observation
        self.state.pasos = max((track.pasos for track in self.comparison_tracks.values()), default=0)
        self.state.recompensa_total = float(primary.recompensa_total)
        self.state.costo_total = float(primary.costo_total)
        self.state.accion = int(primary.accion)
        self.state.bloque_razonamiento = primary.bloque_razonamiento
        self.state.metodo_razonamiento = "Comparación lado a lado"
        self.state.pasos_en_arena = sum(track.pasos_en_arena for track in self.comparison_tracks.values())
        self.state.colisiones_borde = sum(track.colisiones_borde for track in self.comparison_tracks.values())
        self.state.metas_alcanzadas = sum(track.metas_alcanzadas for track in self.comparison_tracks.values())

    def _step_track(self, track: AgentTrack) -> None:
        """Ejecuta un paso de una pista de comparación."""
        if track.done:
            return
        policy = self.policies.get(track.method, greedy_goal_policy)
        action, decision_info = normalize_policy_result(policy(track.env, track.observation))
        position_before = track.env.position.copy()
        next_state, reward, done, env_info = track.env.step(action)
        dynamic_goal_changed = apply_dynamic_goal_if_needed(track.scenario_instance)
        if dynamic_goal_changed:
            env_info["meta_dinamica_cambiada"] = True

        cost = float(decision_info.get("costo_decision", 1.0))
        track.accion = int(action)
        track.recompensa_total += float(reward)
        track.pasos = int(track.env.step_count)
        track.costo_total += cost
        track.metodo_razonamiento = str(decision_info.get("metodo_razonamiento", track.method))
        track.bloque_razonamiento = str(decision_info.get("bloque_razonamiento", track.metodo_razonamiento))
        track.mensaje = str(decision_info.get("mensaje", "Decisión ejecutada."))
        track.ultima_info = {"decision": dict(decision_info), "entorno": dict(env_info)}
        track.pasos_en_arena += int(bool(env_info.get("sobre_arena", False)))
        track.colisiones_borde += int(bool(env_info.get("colision_borde", False)))
        track.metas_alcanzadas += int(bool(env_info.get("meta_alcanzada", False)))
        track.observation = next_state
        track.done = bool(done)
        track.path.append(tuple(track.env.position.astype(float)))

        self._record_thought_cost(track.env, position_before, cost, track.method)
        event = self._record_path_event(
            method=track.method,
            env=track.env,
            step=track.pasos,
            reward=reward,
            cost=cost,
            action=action,
            decision_info=decision_info,
            env_info=env_info,
        )
        event["recompensa_total"] = float(track.recompensa_total)
        event["costo_total"] = float(track.costo_total)
        track.events.append(event)

    def _comparison_metrics_text(self) -> str:
        """Texto compacto para comparación lado a lado."""
        lines = [
            f"Escenario: {self.state.escenario}",
            f"Seed replay: {self.seed}",
            "Modo: comparación lado a lado",
            "Métodos: RLoT vs GoT vs Adaptive",
            f"Heatmap: {'ON' if self.show_heatmap else 'OFF'} | Editor: {'ON' if self.map_editor_enabled else 'OFF'}",
            "",
        ]
        for summary in self.get_comparison_summary():
            lines.append(
                f"{summary['metodo'][:24]} | pasos {summary['pasos']:3d} | "
                f"R {summary['recompensa_total']:7.2f} | C {summary['costo_promedio']:5.2f} | "
                f"bloque {summary['bloque_razonamiento']}"
            )
        lines.append("")
        lines.append(f"Mensaje: {self.state.mensaje}")
        return "\n".join(lines)

    def _record_thought_cost(self, env: NavigationEnv, position: np.ndarray, cost: float, method: str) -> None:
        """Acumula costo de decisión por celda espacial."""
        del method  # El heatmap actual agrega todos los métodos; el CSV conserva el método.
        x_index = int(np.clip(float(position[0]) / max(env.width, 1) * self.heatmap_cols, 0, self.heatmap_cols - 1))
        y_index = int(np.clip(float(position[1]) / max(env.height, 1) * self.heatmap_rows, 0, self.heatmap_rows - 1))
        self.heatmap[x_index, y_index] += float(max(cost, 0.0))
        self.heatmap_counts[x_index, y_index] += 1.0

    def _record_path_event(
        self,
        method: str,
        env: NavigationEnv,
        step: int,
        reward: float,
        cost: float,
        action: int,
        decision_info: Mapping[str, Any],
        env_info: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Registra una fila de trayectoria para CSV/demo."""
        distance = float(np.linalg.norm(env.position - env.goal))
        event = {
            "paso": int(step),
            "metodo": method,
            "escenario": self.state.escenario,
            "seed": int(self.seed),
            "recompensa_paso": float(reward),
            "recompensa_total": float(getattr(self.state, "recompensa_total", 0.0)),
            "costo_decision": float(cost),
            "costo_total": float(getattr(self.state, "costo_total", 0.0)),
            "accion": int(action),
            "bloque_razonamiento": str(decision_info.get("bloque_razonamiento", decision_info.get("metodo_razonamiento", ""))),
            "posicion_x": float(env.position[0]),
            "posicion_y": float(env.position[1]),
            "distancia_a_meta": distance,
            "sobre_arena": bool(env_info.get("sobre_arena", False)),
            "colision_borde": bool(env_info.get("colision_borde", False)),
            "meta_alcanzada": bool(env_info.get("meta_alcanzada", False)),
        }
        self.episode_log.append(event)
        return event

    def _reset_heatmap(self) -> None:
        """Limpia el heatmap de cómputo."""
        self.heatmap = np.zeros((self.heatmap_cols, self.heatmap_rows), dtype=np.float32)
        self.heatmap_counts = np.zeros_like(self.heatmap)

    def _apply_custom_map(self, env: NavigationEnv) -> None:
        """Aplica el mapa editado al entorno si existe."""
        if self.custom_sand_map is not None:
            if self.custom_sand_map.shape != env.sand.shape:
                raise ValueError(f"Mapa editado incompatible: {self.custom_sand_map.shape} vs {env.sand.shape}.")
            env.sand = self.custom_sand_map.copy()

    def _paint_sand_disk(self, env: NavigationEnv, x: float, y: float, radius: int, value: float) -> None:
        """Pinta un disco directamente sobre la matriz de arena."""
        x_center = int(round(float(x)))
        y_center = int(round(float(y)))
        radius = max(int(radius), 1)
        for x_coord in range(max(0, x_center - radius), min(env.width, x_center + radius + 1)):
            for y_coord in range(max(0, y_center - radius), min(env.height, y_center + radius + 1)):
                if (x_coord - x_center) ** 2 + (y_coord - y_center) ** 2 <= radius**2:
                    env.sand[x_coord, y_coord] = float(value)
        env._update_sensors()

    def _refresh_observation_after_map_edit(self) -> None:
        """Actualiza sensores y observaciones después de editar arena."""
        self.env._update_sensors()
        self.observation = self.env._get_observation()
        for track in self.comparison_tracks.values():
            track.env._update_sensors()
            track.observation = track.env._get_observation()

    def _encode_sand_rle(self, sand: np.ndarray) -> Dict[str, Any]:
        """Codifica la arena como RLE binario para replay JSON."""
        flat = (np.asarray(sand) > 0.0).astype(np.uint8).ravel()
        runs: List[List[int]] = []
        if flat.size == 0:
            return {"shape": list(sand.shape), "runs": runs}
        current = int(flat[0])
        count = 1
        for value in flat[1:]:
            value_int = int(value)
            if value_int == current:
                count += 1
            else:
                runs.append([current, count])
                current = value_int
                count = 1
        runs.append([current, count])
        return {"shape": list(sand.shape), "runs": runs}

    def _decode_sand_rle(self, payload: Mapping[str, Any]) -> np.ndarray:
        """Decodifica arena RLE desde replay JSON."""
        shape = tuple(int(value) for value in payload.get("shape", []))
        if len(shape) != 2:
            raise ValueError("Replay inválido: shape de sand_rle debe tener dos dimensiones.")
        values: List[int] = []
        for value, count in payload.get("runs", []):
            values.extend([int(value)] * int(count))
        flat = np.asarray(values, dtype=np.float32)
        expected = int(shape[0] * shape[1])
        if flat.size != expected:
            raise ValueError(f"Replay inválido: sand_rle tiene {flat.size} valores; se esperaban {expected}.")
        return flat.reshape(shape)

    def _overlay_heatmap(self, image: np.ndarray) -> None:
        """Pinta el heatmap de cómputo sobre un frame RGB."""
        matrix = self.get_heatmap_matrix(normalize=True)
        height, width = image.shape[:2]
        for x_index in range(self.heatmap_cols):
            for y_index in range(self.heatmap_rows):
                value = float(matrix[x_index, y_index])
                if value <= 0.02:
                    continue
                x0 = int(x_index * width / self.heatmap_cols)
                x1 = int((x_index + 1) * width / self.heatmap_cols)
                y0 = int(y_index * height / self.heatmap_rows)
                y1 = int((y_index + 1) * height / self.heatmap_rows)
                alpha = min(0.60, 0.12 + 0.48 * value)
                overlay = np.array([255, 80, 20], dtype=np.float32)
                block = image[y0:y1, x0:x1].astype(np.float32)
                image[y0:y1, x0:x1] = np.clip(block * (1.0 - alpha) + overlay * alpha, 0, 255).astype(np.uint8)

    def _overlay_paths(self, image: np.ndarray) -> None:
        """Dibuja trayectorias acumuladas en el frame."""
        if self.comparison_enabled:
            for index, track in enumerate(self.comparison_tracks.values()):
                color = self._track_color(index)
                for start, end in zip(track.path[:-1], track.path[1:]):
                    self._draw_line_rgb(image, np.array(start), np.array(end), color=color, radius=1)
        else:
            positions = [
                np.array([event["posicion_x"], event["posicion_y"]], dtype=np.float32)
                for event in self.episode_log
                if event.get("metodo") == self.state.metodo
            ]
            for start, end in zip(positions[:-1], positions[1:]):
                self._draw_line_rgb(image, start, end, color=(30, 80, 230), radius=1)

    def _draw_disk_rgb(self, image: np.ndarray, position: np.ndarray, radius: int, color: Tuple[int, int, int]) -> None:
        """Dibuja un disco en una imagen RGB."""
        height, width = image.shape[:2]
        x_center = int(round(float(position[0])))
        y_center = int(round(float(position[1])))
        for x_coord in range(max(0, x_center - radius), min(width, x_center + radius + 1)):
            for y_coord in range(max(0, y_center - radius), min(height, y_center + radius + 1)):
                if (x_coord - x_center) ** 2 + (y_coord - y_center) ** 2 <= radius**2:
                    image[y_coord, x_coord] = np.asarray(color, dtype=np.uint8)

    def _draw_line_rgb(
        self,
        image: np.ndarray,
        start: np.ndarray,
        end: np.ndarray,
        color: Tuple[int, int, int],
        radius: int = 1,
    ) -> None:
        """Dibuja una línea simple en una imagen RGB."""
        distance = max(int(np.linalg.norm(end - start)), 1)
        for t in np.linspace(0.0, 1.0, distance + 1):
            point = start * (1.0 - t) + end * t
            self._draw_disk_rgb(image, point, radius=radius, color=color)

    def _track_color(self, index: int) -> Tuple[int, int, int]:
        """Color estable para pistas de comparación."""
        palette = [(30, 80, 230), (30, 170, 80), (230, 120, 30), (150, 80, 210), (220, 60, 80)]
        return palette[index % len(palette)]

    def _generate_demo_figures(self, figures_dir: Path) -> Dict[str, Path]:
        """Genera figuras matplotlib a partir del historial."""
        artifacts: Dict[str, Path] = {}
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return artifacts

        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for event in self.episode_log:
            grouped.setdefault(str(event["metodo"]), []).append(event)

        reward_path = figures_dir / "reward_by_method.png"
        plt.figure(figsize=(7, 4))
        for method, rows in grouped.items():
            steps = [int(row["paso"]) for row in rows]
            rewards = np.cumsum([float(row["recompensa_paso"]) for row in rows])
            plt.plot(steps, rewards, label=method)
        plt.xlabel("Paso")
        plt.ylabel("Recompensa acumulada")
        plt.title("Comparación de retorno por método")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(reward_path, dpi=160)
        plt.close()
        artifacts["reward_figure"] = reward_path

        cost_path = figures_dir / "compute_cost_by_method.png"
        plt.figure(figsize=(7, 4))
        for method, rows in grouped.items():
            steps = [int(row["paso"]) for row in rows]
            costs = np.cumsum([float(row["costo_decision"]) for row in rows])
            plt.plot(steps, costs, label=method)
        plt.xlabel("Paso")
        plt.ylabel("Costo acumulado de decisión")
        plt.title("Gasto de cómputo en inferencia")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(cost_path, dpi=160)
        plt.close()
        artifacts["cost_figure"] = cost_path

        heatmap_path = figures_dir / "thought_heatmap.png"
        plt.figure(figsize=(6, 4))
        plt.imshow(self.get_heatmap_matrix(normalize=True).T, origin="lower", aspect="auto")
        plt.xlabel("Celda X")
        plt.ylabel("Celda Y")
        plt.title("Heatmap de pensamiento/cómputo")
        plt.colorbar(label="Costo normalizado")
        plt.tight_layout()
        plt.savefig(heatmap_path, dpi=160)
        plt.close()
        artifacts["heatmap_figure"] = heatmap_path
        return artifacts

    def _load_imageio(self) -> Any:
        """Carga imageio bajo demanda para no exigirlo en CI básico."""
        try:
            import imageio.v2 as imageio
        except ImportError as exc:
            raise RuntimeError(
                "Instala imageio para grabar episodios o generar modo paper/demo: "
                "python -m pip install imageio imageio-ffmpeg"
            ) from exc
        return imageio


def run_app() -> None:
    """Lanza la aplicación Kivy."""
    try:
        from kivy.app import App
        from kivy.clock import Clock
        from kivy.graphics import Color, Ellipse, Line, Rectangle
        from kivy.metrics import dp
        from kivy.properties import BooleanProperty, NumericProperty, ObjectProperty, StringProperty
        from kivy.uix.boxlayout import BoxLayout
        from kivy.uix.button import Button
        from kivy.uix.label import Label
        from kivy.uix.slider import Slider
        from kivy.uix.spinner import Spinner
        from kivy.uix.textinput import TextInput
        from kivy.uix.widget import Widget
    except ImportError as exc:
        raise RuntimeError(
            "Kivy no está instalado. Ejecuta: python -m pip install -r requirements-gui.txt"
        ) from exc

    class MapCanvas(Widget):  # type: ignore[misc, valid-type]
        """Lienzo que dibuja mapa, obstáculos, heatmap, sensores, agentes y meta."""

        controller = ObjectProperty(None)

        def _geometry(self) -> Tuple[NavigationEnv, float, float, float]:
            env = self.controller.env
            scale = min(self.width / max(env.width, 1), self.height / max(env.height, 1))
            offset_x = self.x + (self.width - env.width * scale) / 2.0
            offset_y = self.y + (self.height - env.height * scale) / 2.0
            return env, scale, offset_x, offset_y

        def _to_screen(self, x_value: float, y_value: float, scale: float, offset_x: float, offset_y: float) -> Tuple[float, float]:
            return offset_x + float(x_value) * scale, offset_y + float(y_value) * scale

        def _to_env(self, x_value: float, y_value: float) -> Optional[Tuple[float, float]]:
            if self.controller is None:
                return None
            env, scale, offset_x, offset_y = self._geometry()
            env_x = (x_value - offset_x) / max(scale, 1e-9)
            env_y = (y_value - offset_y) / max(scale, 1e-9)
            if 0.0 <= env_x <= env.width and 0.0 <= env_y <= env.height:
                return float(env_x), float(env_y)
            return None

        def refresh(self) -> None:
            """Redibuja el entorno actual."""
            self.canvas.clear()
            if self.controller is None:
                return
            env, scale, offset_x, offset_y = self._geometry()

            def tx(x_value: float) -> float:
                return offset_x + x_value * scale

            def ty(y_value: float) -> float:
                return offset_y + y_value * scale

            with self.canvas:
                Color(0.96, 0.96, 0.96, 1.0)
                Rectangle(pos=(offset_x, offset_y), size=(env.width * scale, env.height * scale))

                # Obstáculos: se submuestrea la arena para mantener fluida la interfaz.
                Color(0.74, 0.62, 0.34, 1.0)
                step = max(int(min(env.width, env.height) / 90), 5)
                sand = env.sand
                for x_index in range(0, env.width, step):
                    for y_index in range(0, env.height, step):
                        window = sand[x_index : min(x_index + step, env.width), y_index : min(y_index + step, env.height)]
                        if window.size and float(np.mean(window)) > 0.10:
                            Rectangle(pos=(tx(x_index), ty(y_index)), size=(step * scale, step * scale))

                # Heatmap de pensamiento: rojo/naranja indica más gasto de cómputo.
                if self.controller.show_heatmap:
                    for cell in self.controller.get_heatmap_cells(threshold=0.03):
                        intensity = float(cell["value"])
                        Color(1.0, 0.20, 0.04, 0.10 + 0.45 * intensity)
                        Rectangle(
                            pos=(tx(cell["x0"]), ty(cell["y0"])),
                            size=((cell["x1"] - cell["x0"]) * scale, (cell["y1"] - cell["y0"]) * scale),
                        )

                # Meta.
                Color(0.18, 0.70, 0.30, 1.0)
                goal_radius = max(6.0, env.config.goal_radius * scale * 0.35)
                Ellipse(
                    pos=(tx(env.goal[0]) - goal_radius, ty(env.goal[1]) - goal_radius),
                    size=(2 * goal_radius, 2 * goal_radius),
                )

                if self.controller.comparison_enabled:
                    self._draw_comparison_agents(tx, ty, scale)
                else:
                    self._draw_single_agent(env, tx, ty, scale)

                # Marco.
                Color(0.10, 0.10, 0.10, 1.0)
                Line(rectangle=(offset_x, offset_y, env.width * scale, env.height * scale), width=1.2)

        def _draw_single_agent(self, env: NavigationEnv, tx: Callable[[float], float], ty: Callable[[float], float], scale: float) -> None:
            """Dibuja sensores y agente en modo individual."""
            Color(0.95, 0.20, 0.20, 1.0)
            for sensor_position in env.sensor_positions:
                Line(points=[tx(env.position[0]), ty(env.position[1]), tx(sensor_position[0]), ty(sensor_position[1])], width=1.1)
                sensor_radius = max(3.0, 4.0 * scale)
                Ellipse(
                    pos=(tx(sensor_position[0]) - sensor_radius, ty(sensor_position[1]) - sensor_radius),
                    size=(2 * sensor_radius, 2 * sensor_radius),
                )

            Color(0.12, 0.25, 0.90, 1.0)
            car_radius = max(7.0, 8.0 * scale)
            Ellipse(
                pos=(tx(env.position[0]) - car_radius, ty(env.position[1]) - car_radius),
                size=(2 * car_radius, 2 * car_radius),
            )
            heading = env.position + env._vector_from_angle(env.angle, 28.0)
            Line(points=[tx(env.position[0]), ty(env.position[1]), tx(heading[0]), ty(heading[1])], width=2.0)

        def _draw_comparison_agents(self, tx: Callable[[float], float], ty: Callable[[float], float], scale: float) -> None:
            """Dibuja trayectorias y agentes del modo comparación."""
            palette = [
                (0.12, 0.25, 0.90, 1.0),
                (0.10, 0.62, 0.25, 1.0),
                (0.90, 0.45, 0.12, 1.0),
                (0.55, 0.20, 0.75, 1.0),
            ]
            for index, track in enumerate(self.controller.comparison_tracks.values()):
                color = palette[index % len(palette)]
                Color(*color)
                if len(track.path) > 1:
                    points: List[float] = []
                    for point in track.path[-120:]:
                        points.extend([tx(point[0]), ty(point[1])])
                    Line(points=points, width=1.3)
                car_radius = max(7.0, 8.0 * scale)
                Ellipse(
                    pos=(tx(track.env.position[0]) - car_radius, ty(track.env.position[1]) - car_radius),
                    size=(2 * car_radius, 2 * car_radius),
                )
                heading = track.env.position + track.env._vector_from_angle(track.env.angle, 24.0)
                Line(points=[tx(track.env.position[0]), ty(track.env.position[1]), tx(heading[0]), ty(heading[1])], width=2.0)

        def on_touch_down(self, touch: Any) -> bool:
            """Permite dibujar obstáculos desde la interfaz."""
            if self.controller is not None and self.controller.map_editor_enabled and self.collide_point(*touch.pos):
                point = self._to_env(touch.x, touch.y)
                if point is not None:
                    erase = self.controller.editor_tool == "erase" or getattr(touch, "button", "") == "right"
                    self.controller.paint_obstacle_at(point[0], point[1], erase=erase)
                    self.refresh()
                    return True
            return super().on_touch_down(touch)

        def on_touch_move(self, touch: Any) -> bool:
            """Dibuja continuamente mientras se arrastra el cursor/dedo."""
            if self.controller is not None and self.controller.map_editor_enabled and self.collide_point(*touch.pos):
                point = self._to_env(touch.x, touch.y)
                if point is not None:
                    erase = self.controller.editor_tool == "erase" or getattr(touch, "button", "") == "right"
                    self.controller.paint_obstacle_at(point[0], point[1], erase=erase)
                    self.refresh()
                    return True
            return super().on_touch_move(touch)

    class ThoughtGraphCanvas(Widget):  # type: ignore[misc, valid-type]
        """Lienzo pequeño para visualizar nodos/aristas de Graph-of-Thoughts."""

        controller = ObjectProperty(None)

        def refresh(self) -> None:
            """Redibuja el grafo de pensamientos si existe."""
            self.canvas.clear()
            if self.controller is None:
                return
            graph = self.controller.get_thought_graph()
            nodes = list(graph.get("nodos", [])) if isinstance(graph, Mapping) else []
            edges = list(graph.get("aristas", [])) if isinstance(graph, Mapping) else []

            with self.canvas:
                Color(0.98, 0.98, 0.98, 1.0)
                Rectangle(pos=self.pos, size=self.size)
                Color(0.15, 0.15, 0.15, 1.0)
                Line(rectangle=(self.x, self.y, self.width, self.height), width=1.0)

                if not nodes:
                    return

                max_nodes = min(len(nodes), 9)
                spacing = self.width / max(max_nodes + 1, 2)
                y_mid = self.y + self.height * 0.52
                positions: Dict[str, Tuple[float, float]] = {}
                for index, node in enumerate(nodes[:max_nodes], start=1):
                    node_id = str(node.get("id", index))
                    positions[node_id] = (self.x + spacing * index, y_mid + math.sin(index) * self.height * 0.15)

                Color(0.45, 0.45, 0.45, 1.0)
                for edge in edges[:18]:
                    source = str(edge.get("origen", ""))
                    target = str(edge.get("destino", ""))
                    if source in positions and target in positions:
                        x0, y0 = positions[source]
                        x1, y1 = positions[target]
                        Line(points=[x0, y0, x1, y1], width=1.0)

                for node in nodes[:max_nodes]:
                    node_id = str(node.get("id", ""))
                    kind = str(node.get("tipo", "pensamiento"))
                    score = float(node.get("puntaje", 0.0))
                    x, y = positions[node_id]
                    radius = 10.0 + min(abs(score), 3.0) * 2.0
                    if "riesgo" in kind or score < -0.2:
                        Color(0.88, 0.30, 0.25, 1.0)
                    elif "waypoint" in kind or "grafo" in kind:
                        Color(0.25, 0.62, 0.78, 1.0)
                    elif score > 0.2:
                        Color(0.25, 0.72, 0.36, 1.0)
                    else:
                        Color(0.55, 0.55, 0.70, 1.0)
                    Ellipse(pos=(x - radius, y - radius), size=(2 * radius, 2 * radius))

    class DashboardRoot(BoxLayout):  # type: ignore[misc, valid-type]
        """Layout principal del panel."""

        running = BooleanProperty(False)
        steps_per_tick = NumericProperty(1)
        status_text = StringProperty("Inicializando...")

        def __init__(self, **kwargs: Any) -> None:
            super().__init__(orientation="horizontal", spacing=dp(8), padding=dp(8), **kwargs)
            self.controller = ReasoningDashboardController()
            self.clock_event = None
            self._build_layout()
            self.refresh()

        def _build_layout(self) -> None:
            """Construye controles, mapa y paneles informativos."""
            left = BoxLayout(orientation="vertical", spacing=dp(6), size_hint=(0.70, 1.0))
            header = Label(
                text=f"[b]{TITULO_INTERFAZ}[/b]\n{SUBTITULO_INTERFAZ}",
                markup=True,
                size_hint=(1.0, None),
                height=dp(58),
                halign="center",
                valign="middle",
            )
            header.bind(size=lambda instance, _value: setattr(instance, "text_size", instance.size))
            self.map_canvas = MapCanvas(controller=self.controller, size_hint=(1.0, 0.78))
            self.graph_canvas = ThoughtGraphCanvas(controller=self.controller, size_hint=(1.0, 0.22))
            left.add_widget(header)
            left.add_widget(self.map_canvas)
            left.add_widget(self.graph_canvas)

            right = BoxLayout(orientation="vertical", spacing=dp(6), size_hint=(0.30, 1.0))
            self.metrics_label = Label(text="", size_hint=(1.0, 0.44), halign="left", valign="top")
            self.metrics_label.bind(size=lambda instance, _value: setattr(instance, "text_size", instance.size))

            self.scenario_spinner = Spinner(
                text=self.controller.state.escenario,
                values=self.controller.scenarios,
                size_hint=(1.0, None),
                height=dp(38),
            )
            self.scenario_spinner.bind(text=self.on_scenario_selected)

            self.method_spinner = Spinner(
                text=self.controller.state.metodo,
                values=list(self.controller.policies.keys()),
                size_hint=(1.0, None),
                height=dp(38),
            )
            self.method_spinner.bind(text=self.on_method_selected)

            controls = BoxLayout(orientation="horizontal", spacing=dp(5), size_hint=(1.0, None), height=dp(40))
            self.play_button = Button(text="Iniciar")
            step_button = Button(text="Paso")
            reset_button = Button(text="Reiniciar")
            self.play_button.bind(on_release=self.toggle_run)
            step_button.bind(on_release=lambda _instance: self.manual_step())
            reset_button.bind(on_release=lambda _instance: self.reset_episode())
            controls.add_widget(self.play_button)
            controls.add_widget(step_button)
            controls.add_widget(reset_button)

            comparison_controls = BoxLayout(orientation="horizontal", spacing=dp(5), size_hint=(1.0, None), height=dp(40))
            self.comparison_button = Button(text="Comparación OFF")
            self.heatmap_button = Button(text="Heatmap ON")
            self.comparison_button.bind(on_release=lambda _instance: self.toggle_comparison())
            self.heatmap_button.bind(on_release=lambda _instance: self.toggle_heatmap())
            comparison_controls.add_widget(self.comparison_button)
            comparison_controls.add_widget(self.heatmap_button)

            editor_controls = BoxLayout(orientation="horizontal", spacing=dp(5), size_hint=(1.0, None), height=dp(40))
            self.editor_button = Button(text="Editor OFF")
            self.editor_tool_button = Button(text="Pintar")
            clear_map_button = Button(text="Limpiar")
            self.editor_button.bind(on_release=lambda _instance: self.toggle_editor())
            self.editor_tool_button.bind(on_release=lambda _instance: self.toggle_editor_tool())
            clear_map_button.bind(on_release=lambda _instance: self.clear_map())
            editor_controls.add_widget(self.editor_button)
            editor_controls.add_widget(self.editor_tool_button)
            editor_controls.add_widget(clear_map_button)

            speed_label = Label(text="Velocidad de simulación", size_hint=(1.0, None), height=dp(22))
            self.speed_slider = Slider(min=1, max=12, value=1, step=1, size_hint=(1.0, None), height=dp(36))
            self.speed_slider.bind(value=self.on_speed_changed)

            replay_controls = BoxLayout(orientation="horizontal", spacing=dp(5), size_hint=(1.0, None), height=dp(40))
            self.seed_input = TextInput(text=str(self.controller.seed), multiline=False, input_filter="int")
            seed_button = Button(text="Cargar seed")
            seed_button.bind(on_release=lambda _instance: self.load_seed())
            replay_controls.add_widget(self.seed_input)
            replay_controls.add_widget(seed_button)

            export_controls = BoxLayout(orientation="horizontal", spacing=dp(5), size_hint=(1.0, None), height=dp(40))
            snapshot_button = Button(text="JSON")
            replay_button = Button(text="Replay")
            csv_button = Button(text="CSV")
            snapshot_button.bind(on_release=lambda _instance: self.export_snapshot())
            replay_button.bind(on_release=lambda _instance: self.export_replay())
            csv_button.bind(on_release=lambda _instance: self.export_csv())
            export_controls.add_widget(snapshot_button)
            export_controls.add_widget(replay_button)
            export_controls.add_widget(csv_button)

            media_controls = BoxLayout(orientation="horizontal", spacing=dp(5), size_hint=(1.0, None), height=dp(40))
            gif_button = Button(text="Grabar GIF")
            demo_button = Button(text="Paper/demo")
            gif_button.bind(on_release=lambda _instance: self.record_gif())
            demo_button.bind(on_release=lambda _instance: self.generate_demo())
            media_controls.add_widget(gif_button)
            media_controls.add_widget(demo_button)

            right.add_widget(self.scenario_spinner)
            right.add_widget(self.method_spinner)
            right.add_widget(controls)
            right.add_widget(comparison_controls)
            right.add_widget(editor_controls)
            right.add_widget(speed_label)
            right.add_widget(self.speed_slider)
            right.add_widget(replay_controls)
            right.add_widget(export_controls)
            right.add_widget(media_controls)
            right.add_widget(self.metrics_label)
            self.add_widget(left)
            self.add_widget(right)

        def refresh(self) -> None:
            """Actualiza métricas y lienzos."""
            self.metrics_label.text = self.controller.metrics_text()
            self.comparison_button.text = "Comparación ON" if self.controller.comparison_enabled else "Comparación OFF"
            self.heatmap_button.text = "Heatmap ON" if self.controller.show_heatmap else "Heatmap OFF"
            self.editor_button.text = "Editor ON" if self.controller.map_editor_enabled else "Editor OFF"
            self.editor_tool_button.text = "Borrar" if self.controller.editor_tool == "erase" else "Pintar"
            self.map_canvas.refresh()
            self.graph_canvas.refresh()

        def manual_step(self) -> None:
            """Avanza un paso manual."""
            self.controller.step()
            self.refresh()

        def toggle_run(self, _instance: Any) -> None:
            """Inicia o pausa la simulación."""
            self.running = not self.running
            self.play_button.text = "Pausar" if self.running else "Iniciar"
            if self.running and self.clock_event is None:
                self.clock_event = Clock.schedule_interval(self.update_loop, 1.0 / 30.0)
            elif not self.running and self.clock_event is not None:
                self.clock_event.cancel()
                self.clock_event = None

        def update_loop(self, _dt: float) -> None:
            """Bucle visual llamado por Kivy Clock."""
            for _ in range(int(self.steps_per_tick)):
                self.controller.step()
            self.refresh()

        def reset_episode(self) -> None:
            """Reinicia el episodio activo."""
            self.controller.reset()
            self.refresh()

        def toggle_comparison(self) -> None:
            """Activa/desactiva comparación RLoT vs GoT vs Adaptive Budget."""
            self.controller.enable_comparison(not self.controller.comparison_enabled)
            self.refresh()

        def toggle_heatmap(self) -> None:
            """Muestra u oculta el heatmap de cómputo."""
            self.controller.show_heatmap = not self.controller.show_heatmap
            self.controller.state.mensaje = f"Heatmap {'activado' if self.controller.show_heatmap else 'oculto'}."
            self.refresh()

        def toggle_editor(self) -> None:
            """Activa o desactiva el editor visual de obstáculos."""
            self.controller.set_map_editor(not self.controller.map_editor_enabled)
            self.refresh()

        def toggle_editor_tool(self) -> None:
            """Alterna pintar/borrar obstáculos."""
            tool = "erase" if self.controller.editor_tool == "paint" else "paint"
            self.controller.set_map_editor(self.controller.map_editor_enabled, tool=tool)
            self.refresh()

        def clear_map(self) -> None:
            """Limpia la arena del mapa activo."""
            self.controller.clear_obstacles()
            self.refresh()

        def load_seed(self) -> None:
            """Carga una semilla desde el input de replay."""
            try:
                seed = int(self.seed_input.text or "0")
                self.controller.set_replay_seed(seed, scenario_name=self.controller.state.escenario, method_name=self.controller.state.metodo)
            except ValueError as exc:
                self.controller.state.mensaje = f"Seed inválida: {exc}"
            self.refresh()

        def export_snapshot(self) -> None:
            """Exporta un snapshot reproducible del panel."""
            path = self.controller.export_snapshot(Path("results/dashboard_snapshot.json"))
            self.controller.state.mensaje = f"Estado exportado en {path}"
            self.refresh()

        def export_replay(self) -> None:
            """Exporta configuración de replay determinista."""
            path = self.controller.export_replay_config(Path("results/replay_config.json"))
            self.controller.state.mensaje = f"Replay exportado en {path}"
            self.refresh()

        def export_csv(self) -> None:
            """Exporta trazas CSV de comparación."""
            path = self.controller.export_comparison_csv(Path("results/dashboard_comparison_trace.csv"))
            self.controller.state.mensaje = f"CSV exportado en {path}"
            self.refresh()

        def record_gif(self) -> None:
            """Graba un GIF corto de la simulación."""
            try:
                path = self.controller.record_episode(Path("results/dashboard_episode.gif"), steps=90, fps=12, comparison=True)
                self.controller.state.mensaje = f"GIF exportado en {path}"
            except RuntimeError as exc:
                self.controller.state.mensaje = str(exc)
            self.refresh()

        def generate_demo(self) -> None:
            """Genera artefactos de documentación."""
            try:
                artifacts = self.controller.generate_paper_demo(Path("results/paper_demo"), steps=90, fps=12)
                self.controller.state.mensaje = f"Paper/demo generado: {artifacts.get('manifest')}"
            except RuntimeError as exc:
                self.controller.state.mensaje = str(exc)
            self.refresh()

        def on_speed_changed(self, _instance: Any, value: float) -> None:
            """Actualiza pasos por tick visual."""
            self.steps_per_tick = int(value)

        def on_scenario_selected(self, _instance: Any, value: str) -> None:
            """Maneja cambio de escenario."""
            if value != self.controller.state.escenario:
                self.controller.set_scenario(value)
                self.refresh()

        def on_method_selected(self, _instance: Any, value: str) -> None:
            """Maneja cambio de método."""
            if value != self.controller.state.metodo:
                self.controller.set_method(value)
                self.refresh()

    class ReasoningDashboardApp(App):  # type: ignore[misc, valid-type]
        """Aplicación Kivy principal."""

        title = TITULO_INTERFAZ

        def build(self) -> DashboardRoot:
            """Construye el árbol de widgets."""
            return DashboardRoot()

    ReasoningDashboardApp().run()


if __name__ == "__main__":
    run_app()
