"""
Métricas de decisión para separar costo lógico y tiempo real.

El costo lógico representa pasos simulados, expansiones o unidades internas.
El tiempo real representa duración observada en milisegundos.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter_ns
from typing import Callable, Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class DecisionMeasurement(Generic[T]):
    """Resultado de una decisión junto con sus métricas de ejecución."""

    value: T
    simulated_cost_steps: float
    real_time_ms: float


def ns_to_ms(elapsed_ns: int) -> float:
    """Convierte nanosegundos a milisegundos."""
    if elapsed_ns < 0:
        raise ValueError("El tiempo transcurrido no puede ser negativo.")
    return float(elapsed_ns) / 1_000_000.0


def normalize_simulated_cost(simulated_cost_steps: float | int) -> float:
    """Normaliza el costo lógico de una decisión."""
    value = float(simulated_cost_steps)
    if value < 0:
        raise ValueError("El costo lógico de decisión no puede ser negativo.")
    return value


def measure_decision(
    decision_fn: Callable[[], T],
    simulated_cost_steps: float | int = 1.0,
) -> DecisionMeasurement[T]:
    """Ejecuta una decisión y mide costo lógico más tiempo real."""
    normalized_cost = normalize_simulated_cost(simulated_cost_steps)
    start_ns = perf_counter_ns()
    value = decision_fn()
    elapsed_ns = perf_counter_ns() - start_ns

    return DecisionMeasurement(
        value=value,
        simulated_cost_steps=normalized_cost,
        real_time_ms=ns_to_ms(elapsed_ns),
    )


def build_decision_metrics(
    simulated_cost_steps: float | int,
    real_time_ms: float,
) -> dict[str, float]:
    """Construye métricas serializables para reportes y CSV."""
    normalized_cost = normalize_simulated_cost(simulated_cost_steps)
    measured_time = float(real_time_ms)
    if measured_time < 0:
        raise ValueError("El tiempo real de decisión no puede ser negativo.")

    return {
        "costo_decision_pasos": normalized_cost,
        "tiempo_decision_ms": measured_time,
        "costo_decision": normalized_cost,
    }
