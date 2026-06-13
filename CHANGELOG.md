### Changelog

#### 0.1.0a0 - En desarrollo

Cambios iniciales de la Fase A:

- Se agrega tooling base de calidad con pytest, pytest-cov, mypy, ruff y black.
- Se incorporan comandos de Makefile para instalación de desarrollo, pruebas, cobertura, typing, lint y verificación de formato.
- Se actualiza requirements-ci.txt para reproducir el entorno de validación local y de CI.
- Se inicia el registro formal de cambios del proyecto con versionado semántico preliminar.

#### A1.0 - Base de seed determinista

- Se agrega `reproducibility.py` con `SeedPlan`, `set_global_seed`, `build_numpy_rng` y `build_python_rng`.
- Se reemplaza el muestreo de acciones aleatorias por un generador controlado por la seed del entorno.
- Se agregan pruebas iniciales para validar semillas derivadas y muestreo reproducible.

#### A1.1 - Propagación de seed en evaluación y RLoT

- Se propaga la semilla base hacia `build_policies`.
- Se propaga la semilla base hacia el controlador aprendido y RLoT.
- Se reemplazan offsets manuales de evaluación por `SeedPlan`.
- Se agregan pruebas de traza lógica reproducible para política aleatoria y RLoT.

#### A1.2 - Propagación de seed en entrenamiento y dashboard

- Se propaga `SeedPlan` a baselines y scripts de entrenamiento.
- Se centraliza `set_global_seed` para entrenamiento DQN, RLoT y controlador de razonamiento.
- Se propaga seed al dashboard para reconstruir políticas y resets reproducibles.
- Se agrega prueba de traza reproducible para baseline aleatorio.\n\n#### A2.0 - Base de métricas de decisión

- Se agrega `decision_metrics.py` para separar costo lógico y tiempo real de decisión.
- Se introduce `costo_decision_pasos` como métrica explícita de costo simulado.
- Se introduce `tiempo_decision_ms` como métrica explícita de tiempo real.
- Se mantiene `costo_decision` como campo compatible con reportes previos.
- Se agregan pruebas unitarias para conversión de tiempo, normalización de costo y compatibilidad de métricas.

#### A2.1 - Métricas de decisión en reasoning

- Se integra `measure_decision` en la evaluación de métodos de reasoning.
- Se agrega `costo_decision_pasos_total` y `costo_decision_pasos_promedio`.
- Se agrega `tiempo_decision_ms_total` y `tiempo_decision_ms_promedio`.
- Se mantiene `costo_decision_total` y `costo_decision_promedio` como compatibilidad histórica.
- Se ajustan pruebas de determinismo para no comparar tiempo real entre corridas.

#### A2.2 - Métricas de decisión en suite de evaluación

- Se integra `measure_decision` en la evaluación por escenarios.
- Se agrega `costo_decision_pasos_total` y `costo_decision_pasos_promedio`.
- Se agrega `tiempo_decision_ms_total` y `tiempo_decision_ms_promedio`.
- Se mantiene `costo_decision_total` y `costo_decision_promedio` como compatibilidad histórica.
- Se agregan pruebas para validar que la suite exporta y resume las nuevas métricas.

#### A2.3 - Métricas de decisión en baselines

- Se integra `measure_decision` en la evaluación de baselines.
- Se agrega `costo_decision_pasos_total` y `costo_decision_pasos_promedio`.
- Se agrega `tiempo_decision_ms_total` y `tiempo_decision_ms_promedio`.
- Se mantiene `costo_decision_total` y `costo_decision_promedio` como compatibilidad histórica.
- Se agrega una prueba de fuente para validar la integración sin importar módulos dependientes de PyTorch.

#### A2.4 - Métricas de decisión en dashboard

- Se integra `measure_decision` en el paso del dashboard de un solo agente.
- Se integra `measure_decision` en la comparación lado a lado.
- Se agrega `costo_decision_pasos` y `tiempo_decision_ms` a eventos exportables.
- Se mantienen `costo_decision` y `costo_total` como compatibilidad histórica.
- Se agregan pruebas para validar métricas en estado visible, eventos y resumen de comparación.\n\n#### A2.5 - Cierre de compatibilidad de métricas de decisión

- Se agrega una prueba de integración por fuente para validar la presencia de métricas nuevas.
- Se verifica que reasoning, suite de evaluación, baselines y dashboard usen `measure_decision`.
- Se verifica que los reportes mantengan campos históricos de costo.
- Se evita importar módulos con dependencias opcionales durante la prueba de compatibilidad.
