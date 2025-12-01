"""
TCP Evaluation Utilities

TCP (Tracing & Correcting Program)

Evaluation and feedback utilities for ARC code refinement.
"""

import json
import numpy as np
import ast
import signal
import sys
import math
import copy
from types import FrameType
from typing import List, Dict, Tuple, Optional, Any, Union
from enum import Enum
from func_timeout import func_timeout, FunctionTimedOut
from scipy.ndimage import label
import re
import os


class Color:
    """Color constants for ARC grids."""
    BLACK = 0
    BLUE = 1
    RED = 2
    GREEN = 3
    YELLOW = 4
    GREY = 5
    PINK = 6
    ORANGE = 7
    TEAL = 8
    MAROON = 9
    ALL_COLORS = [BLACK, BLUE, RED, GREEN, YELLOW, GREY, PINK, ORANGE, TEAL, MAROON]


def find_connected_components(grid, monochromatic=True, connectivity=4):
    """Find connected components in a grid."""
    if connectivity == 4:
        structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    else:  # Default to 8-way
        structure = np.ones((3, 3))
    labeled, num = label(grid > 0, structure=structure)
    return labeled, num


def parse_code_from_response(response: str) -> str:
    """Extract Python code blocks from response, returning the first one."""
    print("--- RAW LLM RESPONSE ---")
    print(response)
    print("--- END RAW LLM RESPONSE ---")
    code_pattern = r'```python(.*?)```'
    matches = re.findall(code_pattern, response, re.DOTALL)
    if matches:
        return matches[0].strip()
    code_pattern_fallback = r'```(.*?)```'
    matches = re.findall(code_pattern_fallback, response, re.DOTALL)
    if matches:
        return matches[0].strip()
    if "def transform" in response:
        return response.strip()
    return ""


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


class GridComparisonResult(Enum):
    """Result types for grid comparison."""
    EQUAL = 0
    SHAPE_MISMATCH = 1
    CONTENT_MISMATCH = 2
    TYPE_MISMATCH = 3
    ERROR = 4
    NON_2D_ARRAY = 5
    TIMEOUT = 6


def compare_grids(output_grid: Any, expected_output_grid: np.ndarray) -> Tuple[GridComparisonResult, Optional[float]]:
    """Compare two grids and return the result and pixel accuracy."""
    if isinstance(output_grid, str) and output_grid.startswith("ERROR:"):
        if "timed out" in output_grid:
            return GridComparisonResult.TIMEOUT, None
        return GridComparisonResult.ERROR, None

    if not isinstance(output_grid, np.ndarray):
        try:
            output_grid = np.array(output_grid)
        except (ValueError, TypeError):
            return GridComparisonResult.TYPE_MISMATCH, 0.0

    if len(output_grid.shape) != 2:
        return GridComparisonResult.NON_2D_ARRAY, 0.0

    if output_grid.shape != expected_output_grid.shape:
        return GridComparisonResult.SHAPE_MISMATCH, 0.0

    if np.array_equal(output_grid, expected_output_grid):
        return GridComparisonResult.EQUAL, 1.0

    ratio = np.sum(output_grid == expected_output_grid) / np.prod(expected_output_grid.shape)
    return GridComparisonResult.CONTENT_MISMATCH, ratio


def count_objects(grid, connectivity=8):
    """Count distinct objects in a grid."""
    if not isinstance(grid, np.ndarray) or grid.size == 0:
        return 0
    if connectivity == 4:
        structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    else:
        structure = np.ones((3, 3))
    _, num = label(grid > 0, structure=structure)
    return num


def execute_single_code_with_timeout(code: str, input_grid: np.ndarray, timeout: int, puzzle_properties: Dict[str, Any]):
    """Execute code with a timeout and return the result."""
    def execute_code():
        namespace = {
            'np': np, 'numpy': np, 'math': math, 'copy': copy,
            'input_grid': input_grid.copy(), 'Color': Color,
            'find_connected_components': find_connected_components
        }
        if puzzle_properties:
            namespace.update(puzzle_properties)

        try:
            exec(code, namespace)
            if 'transform' in namespace:
                return namespace['transform'](input_grid.copy())
            elif 'main' in namespace:
                return namespace['main'](input_grid.copy())
            else:
                return "ERROR: transform/main function not found"
        except Exception as e:
            return f"ERROR: {str(e)}"

    try:
        result = func_timeout(timeout, execute_code)
        if isinstance(result, list):  # Ensure output is numpy array
            result = np.array(result)
        return result
    except FunctionTimedOut:
        return "ERROR: Execution timed out"
    except Exception as e:
        return f"ERROR: {str(e)}"


def generate_dynamic_feedback(
    all_pair_results: List[Dict[str, Any]],
    avg_pixel_accuracy: float
) -> List[str]:
    """
    Generates feedback with a dynamic level of detail based on average accuracy.
    """
    feedback_points = []

    # Determine feedback level
    feedback_level = 'summary'
    if avg_pixel_accuracy >= 0.9:
        feedback_level = 'pixel-level'
    elif avg_pixel_accuracy >= 0.5:
        feedback_level = 'detailed'

    # Generate high-level summary
    verdicts = {res.get("verdict") for res in all_pair_results}
    if "SHAPE_MISMATCH" in verdicts:
        high_level_summary = "The code consistently fails with a SHAPE_MISMATCH error. The output grid dimensions are incorrect."
    elif "CONTENT_MISMATCH" in verdicts:
        high_level_summary = "The code produces grids with the correct dimensions, but the content is wrong. There are pixel mismatches."
    elif "ERROR" in verdicts or "TIMEOUT" in verdicts:
        high_level_summary = f"The code is failing to execute correctly, with verdicts: {', '.join(filter(None, verdicts))}."
    else:
        high_level_summary = "The code is failing for an unknown reason. Please review the detailed feedback below."
    feedback_points.append(high_level_summary)

    if feedback_level == 'summary':
        return feedback_points

    # Generate detailed breakdown
    for result in all_pair_results:
        pair_index = result.get("train_pair_index", "N/A")
        verdict = result.get("verdict", "N/A")
        pixel_accuracy = result.get("pixel_accuracy", 0.0)

        actual_output = result.get("actual_output")
        expected_output = result.get("expected_output")

        missing_colors, extra_colors, mismatched_pixels = [], [], []

        if isinstance(actual_output, np.ndarray) and isinstance(expected_output, np.ndarray) and actual_output.shape == expected_output.shape:
            target_colors = set(np.unique(expected_output))
            # Filter out None and non-numeric values
            actual_flat = actual_output.flatten()
            actual_clean = [x for x in actual_flat if x is not None and isinstance(x, (int, float, np.integer, np.floating))]
            actual_colors = set(actual_clean) if len(actual_clean) > 0 else set()
            missing_colors = sorted(list(target_colors - actual_colors))
            extra_colors = sorted(list(actual_colors - target_colors))
            mismatch_coords = np.argwhere(actual_output != expected_output)
            mismatched_pixels = []
            for r, c in mismatch_coords:
                try:
                    actual_val = actual_output[r, c]
                    if actual_val is not None and np.isscalar(actual_val) and np.isfinite(actual_val):
                        actual_color = int(actual_val)
                    else:
                        actual_color = str(actual_val)

                    mismatched_pixels.append({
                        "coords": [int(r), int(c)],
                        "expected_color": int(expected_output[r, c]),
                        "actual_color": actual_color
                    })
                except (TypeError, ValueError):
                    # Handle case where actual_output contains non-numeric values
                    mismatched_pixels.append({
                        "coords": [int(r), int(c)],
                        "expected_color": int(expected_output[r, c]),
                        "actual_color": str(actual_output[r, c])
                    })

        feedback_points.append(f"\n--- Feedback on Training Example {pair_index + 1} ---")
        if pixel_accuracy is not None:
            feedback_points.append(f"The code resulted in a '{verdict}' with {pixel_accuracy * 100:.1f}% pixel accuracy.")
        else:
            feedback_points.append(f"The code resulted in a '{verdict}' (Code failed to execute or produce a valid grid).")

        if missing_colors:
            feedback_points.append(f"It failed to produce the required colors: {missing_colors}")
        if extra_colors:
            feedback_points.append(f"It incorrectly introduced new colors: {extra_colors}")

        if feedback_level == 'pixel-level' and mismatched_pixels:
            feedback_points.append(f"There were {len(mismatched_pixels)} incorrect pixels:")
            for pixel in mismatched_pixels[:10]:  # Limit to 10 examples for brevity
                feedback_points.append(f"  - At position {pixel['coords']}, expected color {pixel['expected_color']}, but got {pixel['actual_color']}")
        elif mismatched_pixels:
            feedback_points.append(f"There were {len(mismatched_pixels)} incorrect pixels.")

    return feedback_points


def get_feedback_for_challenger_dynamic(challenger_code: str, train_pairs: List[Dict[str, Any]], puzzle_properties: Dict[str, Any], timeout: int = 5) -> Tuple[Optional[float], List[str]]:
    """
    Generates feedback for a challenger code with a dynamic level of detail.
    """
    if not train_pairs:
        return 0.0, ["No training pairs to evaluate."]

    all_pair_results = []
    all_accuracies = []

    for i, pair in enumerate(train_pairs):
        input_grid_np = np.array(pair['input'])
        expected_output_np = np.array(pair['output'])

        actual_output = execute_single_code_with_timeout(challenger_code, input_grid_np, timeout, puzzle_properties)
        verdict, pixel_accuracy = compare_grids(actual_output, expected_output_np)

        # Object-level feedback removed for simplicity
        output_objects = []
        target_objects = []

        result_item = {
            "train_pair_index": i,
            "verdict": verdict.name,
            "pixel_accuracy": pixel_accuracy,
            "actual_output": actual_output,
            "expected_output": expected_output_np,
            "output_objects": output_objects,
            "target_objects": target_objects,
        }
        all_pair_results.append(result_item)

        if pixel_accuracy is not None:
            all_accuracies.append(pixel_accuracy)

    if not all_accuracies:
        return 0.0, ["ERROR: Code failed to execute or produce valid output on all training pairs."]

    average_pixel_accuracy = sum(all_accuracies) / len(all_accuracies)

    feedback_strings = generate_dynamic_feedback(all_pair_results, average_pixel_accuracy)

    return average_pixel_accuracy, feedback_strings
