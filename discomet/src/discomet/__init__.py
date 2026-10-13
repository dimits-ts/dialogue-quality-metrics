"""discomet: metrics for evaluating the quality of online discussions."""

from .cpu import (
    linguistic_coordination,
    plot_conversation_structure,
    politeness_strategies,
    reciprocity_motifs,
)

__version__ = "0.1.0.dev0"

__all__ = [
    "linguistic_coordination",
    "plot_conversation_structure",
    "politeness_strategies",
    "reciprocity_motifs",
]
