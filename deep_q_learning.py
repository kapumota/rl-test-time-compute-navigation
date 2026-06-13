"""
Proyecto: ¿Cuándo debe pensar un agente RL?
Implementación moderna de Deep Q-Learning con PyTorch 2.x.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from pathlib import Path
from typing import Deque, Iterable, List, Optional
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


@dataclass(frozen=True)
class Transition:
    """Transición individual almacenada en memoria de repetición."""

    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class Network(nn.Module):
    """Red Q pequeña para estados de baja dimensión."""

    def __init__(self, input_size: int, nb_action: int, hidden_size: int = 64) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.nb_action = int(nb_action)
        self.fc1 = nn.Linear(self.input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, self.nb_action)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ReplayMemory:
    """Memoria de repetición para estabilizar el aprendizaje Q."""

    def __init__(self, capacity: int) -> None:
        self.capacity = int(capacity)
        self.memory: Deque[Transition] = deque(maxlen=self.capacity)

    def __len__(self) -> int:
        return len(self.memory)

    def push(self, transition: Transition) -> None:
        """Agrega una transición a la memoria."""
        self.memory.append(transition)

    def sample(self, batch_size: int) -> List[Transition]:
        """Toma una muestra aleatoria de transiciones."""
        return random.sample(self.memory, int(batch_size))


class DQNAgent:
    """Agente DQN actualizado para experimentos de navegación autónoma."""

    def __init__(
        self,
        input_size: int,
        nb_action: int,
        gamma: float = 0.99,
        learning_rate: float = 1e-3,
        batch_size: int = 128,
        memory_capacity: int = 100_000,
        hidden_size: int = 64,
        device: Optional[str] = None,
    ) -> None:
        self.input_size = int(input_size)
        self.nb_action = int(nb_action)
        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.model = Network(input_size, nb_action, hidden_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=float(learning_rate))
        self.memory = ReplayMemory(memory_capacity)
        self.loss_history: List[float] = []

    def select_action(self, state: Iterable[float], epsilon: float = 0.05) -> int:
        """Selecciona una acción con política epsilon-greedy."""
        if random.random() < float(epsilon):
            return random.randrange(self.nb_action)

        state_tensor = self._to_state_tensor(state)
        self.model.eval()
        with torch.no_grad():
            q_values = self.model(state_tensor)
            action = int(torch.argmax(q_values, dim=1).item())
        self.model.train()
        return action

    def select_action_softmax(self, state: Iterable[float], temperature: float = 1.0) -> int:
        """Selecciona una acción por muestreo softmax para exploración estocástica."""
        temperature = max(float(temperature), 1e-6)
        state_tensor = self._to_state_tensor(state)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(state_tensor) / temperature
            probabilities = torch.softmax(logits, dim=1)
            action = int(torch.multinomial(probabilities, num_samples=1).item())
        self.model.train()
        return action

    def update(
        self,
        state: Iterable[float],
        action: int,
        reward: float,
        next_state: Iterable[float],
        done: bool,
    ) -> Optional[float]:
        """Guarda una transición y actualiza la red si hay datos suficientes."""
        transition = Transition(
            state=np.asarray(state, dtype=np.float32),
            action=int(action),
            reward=float(reward),
            next_state=np.asarray(next_state, dtype=np.float32),
            done=bool(done),
        )
        self.memory.push(transition)

        if len(self.memory) < self.batch_size:
            return None

        loss = self.learn(self.memory.sample(self.batch_size))
        self.loss_history.append(loss)
        return loss

    def learn(self, batch: List[Transition]) -> float:
        """Ejecuta un paso de aprendizaje Q sobre un lote de transiciones."""
        states = torch.as_tensor(np.stack([item.state for item in batch]), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor([item.action for item in batch], dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.as_tensor([item.reward for item in batch], dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(np.stack([item.next_state for item in batch]), dtype=torch.float32, device=self.device)
        dones = torch.as_tensor([item.done for item in batch], dtype=torch.float32, device=self.device)

        q_values = self.model(states).gather(1, actions).squeeze(1)
        with torch.no_grad():
            next_q_values = self.model(next_states).max(dim=1).values
            targets = rewards + self.gamma * next_q_values * (1.0 - dones)

        loss = F.smooth_l1_loss(q_values, targets)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.optimizer.step()
        return float(loss.item())

    def save(self, path: str | Path = "last_brain.pth") -> None:
        """Guarda el estado entrenado del agente."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "input_size": self.input_size,
            "nb_action": self.nb_action,
            "gamma": self.gamma,
            "batch_size": self.batch_size,
            "loss_history": self.loss_history,
        }
        torch.save(checkpoint, Path(path))
        print(f"Modelo guardado en: {path}")

    def load(self, path: str | Path = "last_brain.pth", load_optimizer: bool = True) -> bool:
        """Carga un checkpoint si existe y devuelve si la carga fue exitosa."""
        path = Path(path)
        if not path.is_file():
            print(f"No se encontró checkpoint en: {path}")
            return False

        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        model_state = checkpoint.get("model_state_dict") or checkpoint.get("state_dict")
        if model_state is None:
            raise KeyError("El checkpoint no contiene pesos del modelo.")

        self.model.load_state_dict(model_state)
        if load_optimizer:
            optimizer_state = checkpoint.get("optimizer_state_dict") or checkpoint.get("optimizer")
            if optimizer_state is not None:
                self.optimizer.load_state_dict(optimizer_state)

        self.loss_history = list(checkpoint.get("loss_history", []))
        print(f"Checkpoint cargado desde: {path}")
        return True

    def _to_state_tensor(self, state: Iterable[float]) -> torch.Tensor:
        """Convierte un estado en tensor de lote con tamaño [1, input_size]."""
        array = np.asarray(state, dtype=np.float32)
        if array.shape != (self.input_size,):
            raise ValueError(f"El estado debe tener forma ({self.input_size},), pero llegó {array.shape}.")
        return torch.as_tensor(array, dtype=torch.float32, device=self.device).unsqueeze(0)


# Alias para mantener compatibilidad parcial con código anterior.
Dqn = DQNAgent
