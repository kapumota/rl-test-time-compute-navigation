"""
Proyecto: ¿Cuándo debe pensar un agente RL?
Baseline PPO compacto para acciones discretas.

Este PPO fue escrito desde cero para evitar dependencias pesadas y para que el
experimento sea fácil de leer. No busca competir con implementaciones industriales;
solo sirve como baseline policy-gradient reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


@dataclass(frozen=True)
class PPOConfig:
    """Hiperparámetros principales de PPO."""

    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    learning_rate: float = 3e-4
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    minibatch_size: int = 128
    hidden_size: int = 64


class ActorCritic(nn.Module):
    """Red actor-crítico pequeña para estados de baja dimensión."""

    def __init__(self, input_size: int, nb_action: int, hidden_size: int = 64) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden_size, nb_action)
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.shared(state)
        logits = self.actor(features)
        value = self.critic(features).squeeze(-1)
        return logits, value


class PPOAgent:
    """Agente PPO para el entorno de navegación autónoma."""

    def __init__(
        self,
        input_size: int,
        nb_action: int,
        config: Optional[PPOConfig] = None,
        device: Optional[str] = None,
    ) -> None:
        self.input_size = int(input_size)
        self.nb_action = int(nb_action)
        self.config = config or PPOConfig()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = ActorCritic(self.input_size, self.nb_action, self.config.hidden_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        self.loss_history: List[float] = []

    def select_action(self, state: Iterable[float], deterministic: bool = False) -> int:
        """Selecciona una acción para evaluación o uso externo."""
        action, _, _ = self.act(state, deterministic=deterministic)
        return action

    def act(self, state: Iterable[float], deterministic: bool = False) -> Tuple[int, float, float]:
        """Selecciona una acción y devuelve acción, log-probabilidad y valor."""
        state_tensor = self._to_state_tensor(state)
        self.model.eval()
        with torch.no_grad():
            logits, value = self.model(state_tensor)
            distribution = torch.distributions.Categorical(logits=logits)
            if deterministic:
                action_tensor = torch.argmax(logits, dim=1)
            else:
                action_tensor = distribution.sample()
            log_prob = distribution.log_prob(action_tensor)
        self.model.train()
        return int(action_tensor.item()), float(log_prob.item()), float(value.item())

    def learn(
        self,
        states: Sequence[Iterable[float]],
        actions: Sequence[int],
        rewards: Sequence[float],
        dones: Sequence[bool],
        old_log_probs: Sequence[float],
        values: Sequence[float],
    ) -> float:
        """Actualiza la política usando una trayectoria o lote de trayectorias."""
        if len(states) == 0:
            return 0.0

        states_t = torch.as_tensor(np.asarray(states, dtype=np.float32), dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        old_log_probs_t = torch.as_tensor(old_log_probs, dtype=torch.float32, device=self.device)
        values_np = np.asarray(values, dtype=np.float32)

        returns_np, advantages_np = self._compute_gae(rewards, dones, values_np)
        returns_t = torch.as_tensor(returns_np, dtype=torch.float32, device=self.device)
        advantages_t = torch.as_tensor(advantages_np, dtype=torch.float32, device=self.device)
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std(unbiased=False) + 1e-8)

        total_loss = 0.0
        total_updates = 0
        indices = np.arange(len(states))

        for _ in range(self.config.update_epochs):
            np.random.shuffle(indices)
            for start in range(0, len(indices), self.config.minibatch_size):
                batch_idx = indices[start : start + self.config.minibatch_size]
                batch_idx_t = torch.as_tensor(batch_idx, dtype=torch.long, device=self.device)

                logits, current_values = self.model(states_t.index_select(0, batch_idx_t))
                distribution = torch.distributions.Categorical(logits=logits)
                new_log_probs = distribution.log_prob(actions_t.index_select(0, batch_idx_t))
                entropy = distribution.entropy().mean()

                ratio = torch.exp(new_log_probs - old_log_probs_t.index_select(0, batch_idx_t))
                adv = advantages_t.index_select(0, batch_idx_t)
                unclipped = ratio * adv
                clipped = torch.clamp(ratio, 1.0 - self.config.clip_epsilon, 1.0 + self.config.clip_epsilon) * adv
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = F.mse_loss(current_values, returns_t.index_select(0, batch_idx_t))
                loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                total_loss += float(loss.item())
                total_updates += 1

        average_loss = total_loss / max(total_updates, 1)
        self.loss_history.append(average_loss)
        return average_loss

    def save(self, path: str | Path = "ppo_brain.pth") -> None:
        """Guarda el checkpoint del baseline PPO."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "input_size": self.input_size,
            "nb_action": self.nb_action,
            "config": self.config.__dict__,
            "loss_history": self.loss_history,
        }
        torch.save(checkpoint, Path(path))
        print(f"Modelo PPO guardado en: {path}")

    def load(self, path: str | Path = "ppo_brain.pth") -> bool:
        """Carga un checkpoint PPO si existe."""
        path = Path(path)
        if not path.is_file():
            print(f"No se encontró checkpoint PPO en: {path}")
            return False
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.loss_history = list(checkpoint.get("loss_history", []))
        print(f"Checkpoint PPO cargado desde: {path}")
        return True

    def _compute_gae(
        self,
        rewards: Sequence[float],
        dones: Sequence[bool],
        values: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calcula retornos y ventajas con GAE."""
        advantages = np.zeros(len(rewards), dtype=np.float32)
        last_advantage = 0.0
        next_value = 0.0

        for t in reversed(range(len(rewards))):
            non_terminal = 0.0 if dones[t] else 1.0
            delta = float(rewards[t]) + self.config.gamma * next_value * non_terminal - float(values[t])
            last_advantage = delta + self.config.gamma * self.config.gae_lambda * non_terminal * last_advantage
            advantages[t] = last_advantage
            next_value = float(values[t])

        returns = advantages + values.astype(np.float32)
        return returns.astype(np.float32), advantages.astype(np.float32)

    def _to_state_tensor(self, state: Iterable[float]) -> torch.Tensor:
        """Convierte un estado en tensor de lote."""
        array = np.asarray(state, dtype=np.float32)
        if array.shape != (self.input_size,):
            raise ValueError(f"El estado debe tener forma ({self.input_size},), pero llegó {array.shape}.")
        return torch.as_tensor(array, dtype=torch.float32, device=self.device).unsqueeze(0)
