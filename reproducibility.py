"""
Utilidades de reproducibilidad para experimentos de navegación.

Toda corrida debe derivar sus generadores desde una semilla base explícita.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class SeedPlan:
    """Plan determinista de semillas derivadas desde una semilla base."""

    base_seed: int

    def env_seed(self, episode: int, offset: int = 10_000) -> int:
        """Devuelve la semilla usada para construir entornos."""
        return int(self.base_seed + offset + int(episode))

    def reset_seed(self, episode: int, offset: int = 20_000) -> int:
        """Devuelve la semilla usada para reiniciar episodios."""
        return int(self.base_seed + offset + int(episode))

    def policy_seed(self, offset: int = 30_000) -> int:
        """Devuelve una semilla estable para políticas internas."""
        return int(self.base_seed + offset)

    def controller_seed(self, offset: int = 40_000) -> int:
        """Devuelve una semilla estable para controladores aprendidos."""
        return int(self.base_seed + offset)


def set_global_seed(seed: int) -> None:
    """Fija semillas globales usadas por Python, NumPy y PyTorch si está instalado."""
    value = int(seed)
    random.seed(value)
    np.random.seed(value)

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def build_numpy_rng(seed: Optional[int]) -> np.random.Generator:
    """Construye un generador NumPy reproducible."""
    return np.random.default_rng(None if seed is None else int(seed))


def build_python_rng(seed: Optional[int]) -> random.Random:
    """Construye un generador Python reproducible."""
    return random.Random(None if seed is None else int(seed))
