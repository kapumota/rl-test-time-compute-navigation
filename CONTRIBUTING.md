### Contribuir

Este repositorio está organizado como un proyecto de investigación pequeño y reproducible.

#### Flujo recomendado

1. Crea una rama nueva.
2. Ejecutar pruebas locales:

```bash
python -m pip install -r requirements-ci.txt
pytest -q
```

3. Ejecutar una prueba corta de Fase 3:

```bash
python run_reasoning_experiments.py --eval-episodes 1 --max-steps 30 --methods "Best-of-N actions"
```

4. Abrir un pull request contra `main`.

#### Convenciones

- Firmas de funciones en inglés.
- Comentarios, cadenas de texto y documentación en español.
- Métricas guardadas en `results/`.
- Modelos y controladores guardados en `models/`.
