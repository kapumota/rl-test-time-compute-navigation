### Interfaz Kivy, demostración y reproducibilidad


La interfaz permite observar cómo cambia el comportamiento del agente cuando actúa directo, cuando usa búsqueda, cuando construye un grafo de pensamientos o cuando un navegador RLoT decide qué bloque de razonamiento conviene activar.

La lógica se mantiene separada de Kivy mediante `ReasoningDashboardController`, por lo que las funciones principales se prueban en CI sin abrir ventanas gráficas.

#### Controles disponibles

| Control | Función |
|---|---|
| Selector de escenario | Cambia entre mapa fácil, obstáculos densos, meta cambiante, mapas no vistos y sensores ruidosos |
| Selector de método | Cambia entre A*, Best-of-N, Tree-of-Actions, Graph-of-Waypoints, Adaptive Budget, GoT y RLoT |
| `Iniciar/Pausar` | Ejecuta la simulación automáticamente |
| `Paso` | Ejecuta una decisión manual |
| `Reiniciar` | Reinicia el episodio actual |
| Velocidad | Cambia cuántos pasos se ejecutan por tick visual |
| Exportar JSON | Guarda un snapshot en `results/dashboard_snapshot.json` |
| Comparación ON/OFF | Ejecuta RLoT, GoT y Adaptive Budget en paralelo sobre el mismo mapa |
| Heatmap ON/OFF | Muestra u oculta el gasto espacial de cómputo |
| Editor ON/OFF | Permite pintar o borrar obstáculos desde el mapa |
| Cargar seed | Reinicia el episodio con una semilla exacta |
| Replay | Exporta `results/replay_config.json` con mapa y semilla |
| CSV | Exporta trazas de comparación a `results/dashboard_comparison_trace.csv` |
| Grabar GIF | Exporta `results/dashboard_episode.gif` |
| Paper/demo | Genera capturas, CSV, GIF, JSON y figuras en `results/paper_demo/` |

#### Funciones 

| Función | Estado | Detalle |
|---|---|---|
| Comparación lado a lado | Implementada | Ejecuta RLoT, GoT y Adaptive Budget sobre el mismo mapa y semilla |
| Heatmap de pensamiento | Implementada | Acumula `costo_decision` por celda espacial |
| Grabación de episodios | Implementada | `record_episode()` exporta GIF/MP4 mediante `imageio` |
| Editor de mapas | Implementada | El canvas permite pintar/borrar arena cuando el editor está activo |
| Paper/demo mode | Implementada | `generate_paper_demo()` crea capturas PNG, CSV, GIF, JSON y figuras |
| Replay determinista | Implementada | `export_replay_config()` y `load_replay_config()` preservan seed, escenario, método y mapa |
| Análisis de sobrepensamiento | Implementada | `analyze_overthinking.py` calcula coste por progreso y recompensa por cómputo |

#### Uso desde CLI sin abrir Kivy

```bash
python generate_demo_artifacts.py \
  --scenario obstaculos_densos \
  --seed 123 \
  --steps 120 \
  --output results/paper_demo
```

Luego:

```bash
python analyze_overthinking.py \
  --trace results/paper_demo/comparison_trace.csv \
  --output results/paper_demo/overthinking_summary.csv \
  --json results/paper_demo/overthinking_summary.json
```

#### Relación con investigación

La implementación está alineada con RLoT, GoT, ToT, cómputo adaptativo y análisis de sobrepensamiento. Ver [`alineamiento_investigacion.md`](alineamiento_investigacion.md) para distinguir qué está implementado, qué está parcialmente representado y qué queda como roadmap.
