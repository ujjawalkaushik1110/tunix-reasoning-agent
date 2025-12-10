"""
Tunix Reasoning Agent - A multi-step reasoning agent using Google's Tunix library.
"""

__version__ = "0.1.0"
__author__ = "Ujjawal Kaushik"

from .core.reasoning_agent import ReasoningAgent
from .core.problem_decomposer import ProblemDecomposer
from .evaluation.metrics import ReasoningMetrics
from .fine_tuning.trainer import TunixTrainer

__all__ = [
    "ReasoningAgent",
    "ProblemDecomposer",
    "ReasoningMetrics",
    "TunixTrainer",
]
