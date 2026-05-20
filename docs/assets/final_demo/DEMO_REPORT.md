# Demo final reproducible

Esta carpeta fue generada automáticamente por `demo_final.py`.

## Configuración usada

| Campo | Valor |
|---|---|
| Escenario | `obstaculos_densos` |
| Seed | `123` |
| Pasos | `120` |
| FPS | `12` |
| Métodos | `RL-of-Thoughts Navigator, GoT Navigation Graph, Adaptive rollout budget` |

## Qué mostrar durante la presentación

1. Abrir `episode_comparison.gif` para enseñar la comparación lado a lado.
2. Abrir `figures/thought_heatmap.png` para explicar dónde el agente decidió gastar más cómputo.
3. Abrir `comparison_trace.csv` para mostrar que cada decisión queda registrada.
4. Abrir `overthinking_summary.csv` para discutir coste, progreso y sobrepensamiento.
5. Abrir `replay_config.json` para mostrar que el episodio es determinista y repetible.

## Lectura rápida de resultados

Mayor progreso por cómputo: **RL-of-Thoughts Navigator** (169 progreso/100c). Menor índice de sobrepensamiento: **RL-of-Thoughts Navigator** (0.111).

| Método | Pasos | Recompensa | Costo total | Progreso/100c | Advertencia |
|---|---:|---:|---:|---:|---|
| Adaptive rollout budget | 120 | -61.70 | 1983.00 | 18.7 | normal |
| GoT Navigation Graph | 120 | -43.10 | 38887.00 | 0.474 | posible_sobrepensamiento |
| RL-of-Thoughts Navigator | 120 | -57.30 | 240.00 | 169 | eficiente |

## Artefactos generados

- `cost_figure`: `docs/assets/final_demo/figures/compute_cost_by_method.png`
- `csv`: `docs/assets/final_demo/comparison_trace.csv`
- `gif`: `docs/assets/final_demo/episode_comparison.gif`
- `heatmap_figure`: `docs/assets/final_demo/figures/thought_heatmap.png`
- `manifest`: `docs/assets/final_demo/manifest.json`
- `overthinking_csv`: `docs/assets/final_demo/overthinking_summary.csv`
- `overthinking_json`: `docs/assets/final_demo/overthinking_summary.json`
- `replay`: `docs/assets/final_demo/replay_config.json`
- `reward_figure`: `docs/assets/final_demo/figures/reward_by_method.png`
- `snapshot`: `docs/assets/final_demo/dashboard_snapshot.json`

## Comandos para repetir esta demo

```bash
python -m pip install -r requirements-ci.txt
python demo_final.py --preset presentation
```

Para una validación rápida:

```bash
python demo_final.py --preset quick
```

Para abrir la interfaz interactiva con editor de mapas:

```bash
python -m pip install -r requirements-gui.txt
python gui_dashboard.py
```
