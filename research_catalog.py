"""Catálogo curado de trabajos que justifican el diseño del proyecto.

El objetivo no es depender de APIs externas ni reproducir papers completos, sino
explicitar qué idea de investigación inspira cada componente del agente de
navegación. El catálogo se usa para documentación y para mantener separado lo
implementado de lo que queda como roadmap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

Status = Literal["implementado", "parcial", "roadmap"]
Priority = Literal["alta", "media", "baja"]


@dataclass(frozen=True)
class ResearchReference:
    """Referencia técnica resumida para alinear papers con el repo."""

    key: str
    title: str
    institution: str
    url: str
    priority: Priority
    status: Status
    project_mapping: str
    why_it_matters: str


RESEARCH_CATALOG: tuple[ResearchReference, ...] = (
    ResearchReference(
        key="rlot",
        title="RL of Thoughts",
        institution="Tsinghua",
        url="https://arxiv.org/abs/2505.14140",
        priority="alta",
        status="implementado",
        project_mapping="RLoTNavigator aprende a elegir ACT, CHAIN, TREE, GRAPH o REFLECT en inferencia.",
        why_it_matters="Es la conexión más directa con test-time compute aprendido.",
    ),
    ResearchReference(
        key="got",
        title="Graph of Thoughts",
        institution="ETH Zürich",
        url="https://arxiv.org/abs/2308.09687",
        priority="alta",
        status="implementado",
        project_mapping="GoTNavigationGraphPolicy representa hipótesis, riesgos, consenso y waypoints como nodos y aristas.",
        why_it_matters="Da una estructura explícita y visualizable para el razonamiento.",
    ),
    ResearchReference(
        key="demystifying",
        title="Demystifying Chains, Trees, and Graphs of Thoughts",
        institution="ETH Zürich",
        url="https://arxiv.org/abs/2401.14295",
        priority="alta",
        status="parcial",
        project_mapping="La documentación separa Chain, Tree, Graph, reflexión y presupuesto adaptativo como topologías de razonamiento.",
        why_it_matters="Funciona como marco teórico para explicar el coste/rendimiento de cada método.",
    ),
    ResearchReference(
        key="tot",
        title="Tree of Thoughts",
        institution="Princeton",
        url="https://arxiv.org/abs/2305.10601",
        priority="alta",
        status="implementado",
        project_mapping="TreeOfActions hace búsqueda deliberativa con profundidad, beam width y coste de expansión.",
        why_it_matters="Es la base clásica para branching, lookahead y backtracking.",
    ),
    ResearchReference(
        key="knowing-before-saying",
        title="Knowing Before Saying",
        institution="TUM / colaboradores",
        url="https://arxiv.org/abs/2505.24362",
        priority="alta",
        status="parcial",
        project_mapping="AdaptiveRolloutBudget aproxima la idea de gastar más cómputo solo cuando el estado parece difícil.",
        why_it_matters="Justifica early stopping y asignación adaptativa de cómputo.",
    ),
    ResearchReference(
        key="rest-rl",
        title="ReST-RL",
        institution="THUDM / KEG",
        url="https://arxiv.org/abs/2508.19576",
        priority="media",
        status="roadmap",
        project_mapping="Roadmap: añadir verificador ligero de trayectorias y selección tipo VM-MCTS sobre rollouts.",
        why_it_matters="Conecta RL, verificación y búsqueda en árbol en tiempo de inferencia.",
    ),
    ResearchReference(
        key="agentthink",
        title="Reasoning-Action Dilemma / Agent overthinking",
        institution="UC Berkeley / colaboradores",
        url="https://arxiv.org/abs/2502.08235",
        priority="alta",
        status="implementado",
        project_mapping="analyze_overthinking.py mide coste por progreso, recompensa por cómputo e índice de sobrepensamiento.",
        why_it_matters="Ayuda a mostrar cuándo pensar más deja de pagar en tareas agentic.",
    ),
    ResearchReference(
        key="star",
        title="STaR: Self-Taught Reasoner",
        institution="Stanford",
        url="https://arxiv.org/abs/2203.14465",
        priority="media",
        status="roadmap",
        project_mapping="Roadmap: guardar trazas exitosas del agente y reentrenar el selector RLoT con autocurriculum.",
        why_it_matters="Aporta una vía de bootstrapping a partir de episodios correctos.",
    ),
    ResearchReference(
        key="dspy",
        title="DSPy",
        institution="Stanford",
        url="https://arxiv.org/abs/2310.03714",
        priority="baja",
        status="roadmap",
        project_mapping="Roadmap: envolver experimentos como pipelines optimizables, sin hacerlo dependencia obligatoria.",
        why_it_matters="Puede ayudar si el proyecto crece hacia agentes con herramientas o LLMs reales.",
    ),
    ResearchReference(
        key="satori",
        title="Satori / Chain-of-Action-Thought",
        institution="MIT / MIT-IBM / colaboradores",
        url="https://arxiv.org/abs/2502.02508",
        priority="media",
        status="roadmap",
        project_mapping="Roadmap: internalizar búsqueda/reflexión en una política entrenada, no solo usar módulos externos.",
        why_it_matters="Refuerza la idea de búsqueda autoregresiva y reflexión aprendida.",
    ),
    ResearchReference(
        key="skythought",
        title="LLMs Can Easily Learn to Reason from Demonstrations / SkyThought",
        institution="UC Berkeley / NovaSky",
        url="https://arxiv.org/abs/2502.07374",
        priority="media",
        status="roadmap",
        project_mapping="Roadmap: comparar si la estructura de trazas CHAIN/TREE/GRAPH importa más que sus detalles numéricos.",
        why_it_matters="Sugiere evaluar estructura de razonamiento, no solo recompensa final.",
    ),
)


def references_by_status(status: Status) -> tuple[ResearchReference, ...]:
    """Filtra referencias por estado dentro del proyecto."""
    return tuple(reference for reference in RESEARCH_CATALOG if reference.status == status)


def references_by_priority(priority: Priority) -> tuple[ResearchReference, ...]:
    """Filtra referencias por prioridad de integración."""
    return tuple(reference for reference in RESEARCH_CATALOG if reference.priority == priority)


def markdown_table(references: Iterable[ResearchReference] = RESEARCH_CATALOG) -> str:
    """Genera una tabla Markdown compacta del catálogo."""
    rows = [
        "| Prioridad | Estado | Trabajo | Mapeo en el proyecto |",
        "|---|---|---|---|",
    ]
    for reference in references:
        title = f"[{reference.title}]({reference.url})"
        rows.append(
            f"| {reference.priority} | {reference.status} | {title} | {reference.project_mapping} |"
        )
    return "\n".join(rows)
