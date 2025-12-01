"""
TCP Generate and Evaluate Package

Code generation and feedback evaluation for ARC tasks.

Main modules:
- generate_and_evaluate: Main script for generation and evaluation
- tcp_utils: Utility functions (Color, connected components)
- tcp_object_detection: Object detection for ARC grids
- tcp_dataset: Dataset loading utilities
- create_seed_file: Pre-processing for seed example creation
"""

__version__ = "1.0.0"
__author__ = "TCP Team"

# Make key utilities available at package level
from .tcp_utils import Color, find_connected_components
from .tcp_object_detection import extract_objects_from_grid
from tcp_core.tcp_dataset import get_dataset

__all__ = [
    'Color',
    'find_connected_components',
    'extract_objects_from_grid',
    'get_dataset',
]
