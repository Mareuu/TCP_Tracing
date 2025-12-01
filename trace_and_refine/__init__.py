"""
TCP Trace and Refine Package

TCP (Tracing & Correcting Program)

Iterative code refinement with dynamic feedback for ARC tasks.

Main modules:
- tcp_refine: Main script for iterative code refinement
- tcp_evaluation_utils: Evaluation and feedback utilities
- tcp_dataset: Dataset loading utilities
"""

__version__ = "1.0.0"
__author__ = "TCP Team"

from .tcp_evaluation_utils import (
    Color,
    NumpyEncoder,
    GridComparisonResult,
    parse_code_from_response,
    compare_grids,
    get_feedback_for_challenger_dynamic,
)
from tcp_core.tcp_dataset import get_dataset

__all__ = [
    'Color',
    'NumpyEncoder',
    'GridComparisonResult',
    'parse_code_from_response',
    'compare_grids',
    'get_feedback_for_challenger_dynamic',
    'get_dataset',
]
