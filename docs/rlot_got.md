### RL-of-Thoughts y Graph-of-Thoughts para navegación

Esta fase convierte el proyecto en una arquitectura de razonamiento adaptativo. En lugar de comparar políticas aisladas, el agente aprende a seleccionar un bloque de razonamiento en tiempo de inferencia.

#### Bloques disponibles

| Bloque | Función |
|---|---|
| `ACT` | Acción directa de bajo costo hacia la meta |
| `CHAIN` | Simulación secuencial de acciones futuras |
| `TREE` | Búsqueda deliberativa en árbol de acciones |
| `GRAPH` | Grafo de pensamientos con acciones, waypoints, críticas y consenso |
| `REFLECT` | Revisión de una acción propuesta antes de ejecutarla |

#### Diferencia entre GoT y RLoT en este repositorio

**GoT Navigation Graph** construye un grafo explícito en cada decisión. Sus nodos representan recomendaciones de distintos mecanismos, estimaciones de riesgo y consensos entre bloques. Luego destila el grafo a una acción.

**RL-of-Thoughts Navigator** aprende una política sobre bloques. Su acción de alto nivel no es girar o avanzar, sino escoger `ACT`, `CHAIN`, `TREE`, `GRAPH` o `REFLECT`. Después, el bloque seleccionado produce la acción de bajo nivel.

#### Hipótesis experimental

Un agente no debería usar siempre el razonamiento más caro. En estados fáciles, `ACT` puede ser suficiente. En estados con obstáculos, ruido o incertidumbre, `TREE`, `GRAPH` o `REFLECT` pueden compensar su mayor costo computacional.

#### Archivos principales

| Archivo | Rol |
|---|---|
| `rlot_got_navigation.py` | Implementa GoT Navigation Graph y RLoT Navigator |
| `train_rlot_navigator.py` | Entrena el selector de bloques con Q-learning |
| `run_rlot_got_experiments.py` | Ejecuta una evaluación focalizada de Fase 5 |
| `run_evaluation_suite.py` | Evalúa RLoT/GoT en los escenarios de Fase 4 |

#### Referencias de inspiración

- RL of Thoughts: Navigating LLM Reasoning with Inference-time Reinforcement Learning. arXiv:2505.14140.
- Graph of Thoughts: Solving Elaborate Problems with Large Language Models. AAAI 2024 / arXiv:2308.09687.
- Beyond Chain-of-Thought, Effective Graph-of-Thought Reasoning in Language Models. arXiv:2305.16582.
