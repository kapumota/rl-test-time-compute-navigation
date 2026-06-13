from __future__ import annotations

import pytest

from decision_metrics import (
    build_decision_metrics,
    measure_decision,
    normalize_simulated_cost,
    ns_to_ms,
)


def test_ns_to_ms_converts_nanoseconds_to_milliseconds() -> None:
    """La conversión de nanosegundos a milisegundos debe ser estable."""
    assert ns_to_ms(0) == 0.0
    assert ns_to_ms(1_000_000) == 1.0
    assert ns_to_ms(2_500_000) == 2.5


def test_ns_to_ms_rejects_negative_values() -> None:
    """El tiempo transcurrido no debe aceptar valores negativos."""
    with pytest.raises(ValueError, match="no puede ser negativo"):
        ns_to_ms(-1)


def test_normalize_simulated_cost_accepts_int_and_float() -> None:
    """El costo lógico debe normalizar enteros y flotantes."""
    assert normalize_simulated_cost(3) == 3.0
    assert normalize_simulated_cost(2.5) == 2.5


def test_normalize_simulated_cost_rejects_negative_values() -> None:
    """El costo lógico no debe aceptar valores negativos."""
    with pytest.raises(ValueError, match="no puede ser negativo"):
        normalize_simulated_cost(-0.1)


def test_measure_decision_returns_value_and_metrics() -> None:
    """La medición debe preservar el valor de decisión y agregar métricas."""
    measurement = measure_decision(lambda: 7, simulated_cost_steps=4)

    assert measurement.value == 7
    assert measurement.simulated_cost_steps == 4.0
    assert measurement.real_time_ms >= 0.0


def test_build_decision_metrics_keeps_backward_compatibility() -> None:
    """Las métricas deben mantener el campo histórico de costo."""
    metrics = build_decision_metrics(
        simulated_cost_steps=5,
        real_time_ms=1.25,
    )

    assert metrics == {
        "costo_decision_pasos": 5.0,
        "tiempo_decision_ms": 1.25,
        "costo_decision": 5.0,
    }


def test_build_decision_metrics_rejects_negative_real_time() -> None:
    """El tiempo real de decisión no debe aceptar valores negativos."""
    with pytest.raises(ValueError, match="no puede ser negativo"):
        build_decision_metrics(simulated_cost_steps=1, real_time_ms=-0.01)
