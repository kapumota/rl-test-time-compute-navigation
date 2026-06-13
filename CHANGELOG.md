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
