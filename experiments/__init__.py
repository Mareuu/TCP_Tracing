"""
TCP Experiments Module

Tools for running and analyzing ablation studies.
"""

from .ablation_runner import ABLATION_EXPERIMENT_CONFIGS
from .analyze_ablation import (
    TaskResult,
    ExperimentSummary,
    load_experiment_results,
    compute_experiment_summary,
    load_all_experiments,
    compute_contributions,
)

__all__ = [
    'ABLATION_EXPERIMENT_CONFIGS',
    'TaskResult',
    'ExperimentSummary',
    'load_experiment_results',
    'compute_experiment_summary',
    'load_all_experiments',
    'compute_contributions',
]
