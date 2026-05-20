### Alineación con investigación reciente

Este documento separa tres niveles:

1. **Implementado:** existe código ejecutable y pruebas.
2. **Parcial:** la idea está representada, pero no reproduce el paper completo.
3. **Roadmap:** es una buena dirección futura, pero no debe presentarse como implementada.

#### Selección recomendada

La lista original es amplia. Para que el proyecto quede fuerte y honesto, conviene priorizar estos ejes:

| Eje | Trabajos usados | Estado en el repo |
|---|---|---|
| Estructuras de razonamiento | Tree of Thoughts, Graph of Thoughts, Demystifying Chains/Trees/Graphs | Implementado/parcial |
| Selección aprendida de razonamiento | RL of Thoughts | Implementado |
| Presupuesto adaptativo | Knowing Before Saying, test-time compute | Parcial |
| Riesgo de sobrepensamiento | Reasoning-Action Dilemma / Agent overthinking | Implementado como métrica de análisis |
| Verificación y búsqueda futura | ReST-RL, Satori, STaR, SkyThought | Roadmap |

#### Catálogo

| Prioridad | Estado | Trabajo | Mapeo en el proyecto |
|---|---|---|---|
| alta | implementado | [RL of Thoughts](https://arxiv.org/abs/2505.14140) | `RLoTNavigator` aprende a elegir `ACT`, `CHAIN`, `TREE`, `GRAPH` o `REFLECT` en inferencia. |
| alta | implementado | [Graph of Thoughts](https://arxiv.org/abs/2308.09687) | `GoTNavigationGraphPolicy` representa hipótesis, riesgos, consenso y waypoints como nodos/aristas. |
| alta | parcial | [Demystifying Chains, Trees, and Graphs of Thoughts](https://arxiv.org/abs/2401.14295) | La documentación organiza Chain/Tree/Graph/reflexión/presupuesto como topologías de razonamiento. |
| alta | implementado | [Tree of Thoughts](https://arxiv.org/abs/2305.10601) | `TreeOfActions` implementa branching, lookahead y coste de expansión. |
| alta | parcial | [Knowing Before Saying](https://arxiv.org/abs/2505.24362) | `AdaptiveRolloutBudget` aproxima asignación de cómputo según dificultad local. |
| alta | implementado | [Reasoning-Action Dilemma / Agent overthinking](https://arxiv.org/abs/2502.08235) | `analyze_overthinking.py` calcula coste por progreso e índice de sobrepensamiento. |
| media | roadmap | [ReST-RL](https://arxiv.org/abs/2508.19576) | Próximo paso: verificador ligero + VM-MCTS sobre rollouts. |
| media | roadmap | [STaR](https://arxiv.org/abs/2203.14465) | Próximo paso: guardar trazas exitosas y reentrenar RLoT con autocurriculum. |
| media | roadmap | [Satori / Chain-of-Action-Thought](https://arxiv.org/abs/2502.02508) | Próximo paso: internalizar búsqueda/reflexión en una política entrenada. |
| media | roadmap | [SkyThought](https://arxiv.org/abs/2502.07374) | Próximo paso: evaluar si la estructura de trazas importa más que el contenido literal. |
| baja | roadmap | [DSPy](https://arxiv.org/abs/2310.03714) | Útil si el proyecto crece hacia pipelines con LLMs/herramientas reales. |

#### Qué no afirmamos

- No se afirma que este proyecto reproduzca benchmarks de LLMs.
- No se afirma que use modelos fundacionales externos.
- No se afirma que implemente GRPO, PRM o MCTS con value model todavía.
- Sí se afirma que traduce las ideas de test-time compute y razonamiento estructurado a un entorno RL controlado y reproducible.

#### Roadmap técnico recomendado

1. **Verificador de trayectorias:** puntuar rollouts por seguridad, progreso y consistencia antes de elegir acción.
2. **VM-MCTS compacto:** entrenar un value model pequeño para guiar búsqueda sobre acciones.
3. **Autocurriculum estilo STaR:** guardar episodios exitosos y reentrenar el selector de bloques.
4. **Ablación de estructura:** comparar trazas con misma recompensa pero distinta topología: chain vs tree vs graph.
5. **Early stopping real:** aprender cuándo detener rollouts antes de gastar todo el presupuesto.
