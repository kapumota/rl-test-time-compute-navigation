"""
Proyecto: ¿Cuándo debe pensar un agente RL?
Baseline Double DQN para navegación autónoma.

Double DQN separa la selección de la acción futura de su evaluación:
- la red online selecciona argmax_a Q_online(s', a)
- la red objetivo evalúa Q_target(s', argmax_a)

Esto reduce el sesgo de sobreestimación típico de DQN estándar.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from deep_q_learning import Network, ReplayMemory, Transition


@dataclass(frozen=True)
class DoubleDQNConfig:
    """Hiperparámetros principales del baseline Double DQN."""

    gamma: float = 0.99
    learning_rate: float = 1e-3
    batch_size: int = 64
    memory_capacity: int = 50_000
    hidden_size: int = 64
    target_update_interval: int = 250
    gradient_clip: float = 10.0


class DoubleDQNAgent:
    """Agente Double DQN con red online y red objetivo."""

    def __init__(
        self,
        input_size: int,
        nb_action: int,
        config: Optional[DoubleDQNConfig] = None,
        device: Optional[str] = None,
    ) -> None:
        self.input_size = int(input_size)
        self.nb_action = int(nb_action)
        self.config = config or DoubleDQNConfig()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.online_model = Network(self.input_size, self.nb_action, self.config.hidden_size).to(
            self.device
        )
        self.target_model = Network(self.input_size, self.nb_action, self.config.hidden_size).to(
            self.device
        )
        self.target_model.load_state_dict(self.online_model.state_dict())
        self.target_model.eval()

        self.optimizer = optim.Adam(self.online_model.parameters(), lr=self.config.learning_rate)
        self.memory = ReplayMemory(self.config.memory_capacity)
        self.loss_history: List[float] = []
        self.training_steps = 0

    def select_action(self, state: Iterable[float], epsilon: float = 0.05) -> int:
        """Selecciona una acción con política epsilon-greedy."""
        if random.random() < float(epsilon):
            return random.randrange(self.nb_action)

        state_tensor = self._to_state_tensor(state)
        self.online_model.eval()
        with torch.no_grad():
            q_values = self.online_model(state_tensor)
            action = int(torch.argmax(q_values, dim=1).item())
        self.online_model.train()
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
        self.memory.push(
            Transition(
                state=np.asarray(state, dtype=np.float32),
                action=int(action),
                reward=float(reward),
                next_state=np.asarray(next_state, dtype=np.float32),
                done=bool(done),
            )
        )

        if len(self.memory) < self.config.batch_size:
            return None

        loss = self.learn(self.memory.sample(self.config.batch_size))
        self.loss_history.append(loss)
        self.training_steps += 1

        if self.training_steps % self.config.target_update_interval == 0:
            self.update_target()

        return loss

    def learn(self, batch: List[Transition]) -> float:
        """Ejecuta un paso Double DQN sobre un lote de transiciones."""
        states = torch.as_tensor(
            np.stack([item.state for item in batch]), dtype=torch.float32, device=self.device
        )
        actions = torch.as_tensor(
            [item.action for item in batch], dtype=torch.long, device=self.device
        ).unsqueeze(1)
        rewards = torch.as_tensor(
            [item.reward for item in batch], dtype=torch.float32, device=self.device
        )
        next_states = torch.as_tensor(
            np.stack([item.next_state for item in batch]), dtype=torch.float32, device=self.device
        )
        dones = torch.as_tensor(
            [item.done for item in batch], dtype=torch.float32, device=self.device
        )

        q_values = self.online_model(states).gather(1, actions).squeeze(1)
        with torch.no_grad():
            next_actions = self.online_model(next_states).argmax(dim=1, keepdim=True)
            next_q_values = self.target_model(next_states).gather(1, next_actions).squeeze(1)
            targets = rewards + self.config.gamma * next_q_values * (1.0 - dones)

        loss = F.smooth_l1_loss(q_values, targets)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_model.parameters(), max_norm=self.config.gradient_clip)
        self.optimizer.step()
        return float(loss.item())

    def update_target(self) -> None:
        """Sincroniza la red objetivo con la red online."""
        self.target_model.load_state_dict(self.online_model.state_dict())

    def save(self, path: str | Path = "double_dqn_brain.pth") -> None:
        """Guarda el checkpoint del baseline Double DQN."""
        checkpoint = {
            "online_model_state_dict": self.online_model.state_dict(),
            "target_model_state_dict": self.target_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "input_size": self.input_size,
            "nb_action": self.nb_action,
            "config": self.config.__dict__,
            "loss_history": self.loss_history,
            "training_steps": self.training_steps,
        }
        torch.save(checkpoint, Path(path))
        print(f"Modelo Double DQN guardado en: {path}")

    def load(self, path: str | Path = "double_dqn_brain.pth", load_optimizer: bool = True) -> bool:
        """Carga un checkpoint Double DQN si existe."""
        path = Path(path)
        if not path.is_file():
            print(f"No se encontró checkpoint Double DQN en: {path}")
            return False

        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.online_model.load_state_dict(checkpoint["online_model_state_dict"])
        self.target_model.load_state_dict(
            checkpoint.get("target_model_state_dict", checkpoint["online_model_state_dict"])
        )
        if load_optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.loss_history = list(checkpoint.get("loss_history", []))
        self.training_steps = int(checkpoint.get("training_steps", 0))
        print(f"Checkpoint Double DQN cargado desde: {path}")
        return True

    def _to_state_tensor(self, state: Iterable[float]) -> torch.Tensor:
        """Convierte un estado en tensor de lote."""
        array = np.asarray(state, dtype=np.float32)
        if array.shape != (self.input_size,):
            raise ValueError(
                f"El estado debe tener forma ({self.input_size},), pero llegó {array.shape}."
            )
        return torch.as_tensor(array, dtype=torch.float32, device=self.device).unsqueeze(0)
