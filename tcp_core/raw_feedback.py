"""
TCP Raw Feedback Module

Generates domain-agnostic numerical feedback for ablation studies.
This module provides pure numerical feedback without domain interpretation,
allowing clear separation of "method contribution" vs "domain knowledge contribution".

Raw feedback includes only:
- Execution status (success/error/timeout)
- Shape match (boolean)
- Accuracy (0.0 to 1.0)
- Element counts (total, correct, incorrect)
- Error positions (list of coordinates)

This addresses reviewer feedback about attributing gains to method vs domain knowledge.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np


@dataclass
class RawFeedbackResult:
    """Raw numerical feedback without domain interpretation."""
    execution_status: str  # "success", "error", "timeout"
    shape_match: bool
    accuracy: float  # 0.0 to 1.0
    total_elements: int
    correct_elements: int
    incorrect_elements: int
    error_positions: List[Tuple[int, int]] = field(default_factory=list)
    error_message: Optional[str] = None

    # Shape information (for debugging)
    expected_shape: Optional[Tuple[int, int]] = None
    actual_shape: Optional[Tuple[int, int]] = None


class RawFeedbackGenerator:
    """
    Generates raw numerical feedback without domain interpretation.

    This class provides purely quantitative feedback about code execution results,
    specifically designed for ablation studies to measure method contribution
    independent of domain knowledge.
    """

    @staticmethod
    def generate_raw_feedback(
        actual_output: Any,
        expected_output: np.ndarray,
        execution_status: str = "success"
    ) -> RawFeedbackResult:
        """
        Generate raw numerical feedback comparing actual vs expected output.

        Args:
            actual_output: The output from code execution (may be error string, array, etc.)
            expected_output: The expected output as numpy array
            execution_status: Status of execution ("success", "error", "timeout")

        Returns:
            RawFeedbackResult with numerical metrics only
        """
        # Handle error cases
        if isinstance(actual_output, str):
            if "ERROR" in actual_output:
                if "timed out" in actual_output.lower():
                    return RawFeedbackResult(
                        execution_status="timeout",
                        shape_match=False,
                        accuracy=0.0,
                        total_elements=int(np.prod(expected_output.shape)),
                        correct_elements=0,
                        incorrect_elements=int(np.prod(expected_output.shape)),
                        error_message=actual_output,
                        expected_shape=expected_output.shape,
                        actual_shape=None
                    )
                else:
                    return RawFeedbackResult(
                        execution_status="error",
                        shape_match=False,
                        accuracy=0.0,
                        total_elements=int(np.prod(expected_output.shape)),
                        correct_elements=0,
                        incorrect_elements=int(np.prod(expected_output.shape)),
                        error_message=actual_output,
                        expected_shape=expected_output.shape,
                        actual_shape=None
                    )

        # Convert to numpy array if needed
        if not isinstance(actual_output, np.ndarray):
            try:
                actual_output = np.array(actual_output)
            except (ValueError, TypeError) as e:
                return RawFeedbackResult(
                    execution_status="error",
                    shape_match=False,
                    accuracy=0.0,
                    total_elements=int(np.prod(expected_output.shape)),
                    correct_elements=0,
                    incorrect_elements=int(np.prod(expected_output.shape)),
                    error_message=f"Cannot convert output to array: {str(e)}",
                    expected_shape=expected_output.shape,
                    actual_shape=None
                )

        # Check for valid 2D array
        if len(actual_output.shape) != 2:
            return RawFeedbackResult(
                execution_status="error",
                shape_match=False,
                accuracy=0.0,
                total_elements=int(np.prod(expected_output.shape)),
                correct_elements=0,
                incorrect_elements=int(np.prod(expected_output.shape)),
                error_message=f"Output is not 2D: shape={actual_output.shape}",
                expected_shape=expected_output.shape,
                actual_shape=actual_output.shape
            )

        # Check shape match
        shape_match = actual_output.shape == expected_output.shape

        if not shape_match:
            return RawFeedbackResult(
                execution_status=execution_status,
                shape_match=False,
                accuracy=0.0,
                total_elements=int(np.prod(expected_output.shape)),
                correct_elements=0,
                incorrect_elements=int(np.prod(expected_output.shape)),
                expected_shape=expected_output.shape,
                actual_shape=actual_output.shape
            )

        # Calculate accuracy metrics
        total_elements = int(np.prod(expected_output.shape))
        correct_mask = actual_output == expected_output
        correct_elements = int(np.sum(correct_mask))
        incorrect_elements = total_elements - correct_elements
        accuracy = correct_elements / total_elements if total_elements > 0 else 0.0

        # Get error positions (limit to prevent memory issues)
        error_positions = []
        if incorrect_elements > 0:
            error_coords = np.argwhere(~correct_mask)
            # Limit to first 100 error positions to prevent huge lists
            for r, c in error_coords[:100]:
                error_positions.append((int(r), int(c)))

        return RawFeedbackResult(
            execution_status=execution_status,
            shape_match=True,
            accuracy=accuracy,
            total_elements=total_elements,
            correct_elements=correct_elements,
            incorrect_elements=incorrect_elements,
            error_positions=error_positions,
            expected_shape=expected_output.shape,
            actual_shape=actual_output.shape
        )

    @staticmethod
    def format_as_string(feedback: RawFeedbackResult, include_positions: bool = True) -> str:
        """
        Format raw feedback as a concise string for LLM consumption.

        This format is deliberately minimal and domain-agnostic:
        - No interpretation of what errors mean
        - No hints or suggestions
        - Just pure numerical facts

        Args:
            feedback: RawFeedbackResult to format
            include_positions: Whether to include error positions (default: True)

        Returns:
            Formatted string with raw metrics
        """
        lines = [
            f"Execution Status: {feedback.execution_status}",
            f"Shape Match: {feedback.shape_match}",
        ]

        if feedback.expected_shape:
            lines.append(f"Expected Shape: {feedback.expected_shape}")
        if feedback.actual_shape:
            lines.append(f"Actual Shape: {feedback.actual_shape}")

        lines.extend([
            f"Accuracy: {feedback.accuracy:.4f}",
            f"Total Elements: {feedback.total_elements}",
            f"Correct Elements: {feedback.correct_elements}",
            f"Incorrect Elements: {feedback.incorrect_elements}",
        ])

        if feedback.error_message:
            lines.append(f"Error: {feedback.error_message}")

        if include_positions and feedback.error_positions:
            lines.append(f"Error Positions (first {len(feedback.error_positions)}): {feedback.error_positions[:20]}")

        return "\n".join(lines)

    @staticmethod
    def aggregate_feedback(
        feedbacks: List[RawFeedbackResult]
    ) -> Dict[str, Any]:
        """
        Aggregate multiple feedback results into summary statistics.

        Args:
            feedbacks: List of RawFeedbackResult objects

        Returns:
            Dictionary with aggregate statistics
        """
        if not feedbacks:
            return {
                "num_examples": 0,
                "avg_accuracy": 0.0,
                "min_accuracy": 0.0,
                "max_accuracy": 0.0,
                "all_shapes_match": False,
                "num_errors": 0,
                "num_timeouts": 0,
                "total_incorrect_elements": 0,
            }

        accuracies = [f.accuracy for f in feedbacks]

        return {
            "num_examples": len(feedbacks),
            "avg_accuracy": sum(accuracies) / len(accuracies),
            "min_accuracy": min(accuracies),
            "max_accuracy": max(accuracies),
            "all_shapes_match": all(f.shape_match for f in feedbacks),
            "num_errors": sum(1 for f in feedbacks if f.execution_status == "error"),
            "num_timeouts": sum(1 for f in feedbacks if f.execution_status == "timeout"),
            "total_incorrect_elements": sum(f.incorrect_elements for f in feedbacks),
        }

    @staticmethod
    def format_aggregate_as_string(feedbacks: List[RawFeedbackResult]) -> str:
        """
        Format aggregated feedback from multiple examples as a string.

        Args:
            feedbacks: List of RawFeedbackResult objects

        Returns:
            Formatted string with aggregate statistics and per-example summaries
        """
        if not feedbacks:
            return "No examples to evaluate."

        aggregate = RawFeedbackGenerator.aggregate_feedback(feedbacks)

        lines = [
            "=== Raw Feedback Summary ===",
            f"Examples Evaluated: {aggregate['num_examples']}",
            f"Average Accuracy: {aggregate['avg_accuracy']:.4f}",
            f"Min/Max Accuracy: {aggregate['min_accuracy']:.4f} / {aggregate['max_accuracy']:.4f}",
            f"All Shapes Match: {aggregate['all_shapes_match']}",
            f"Execution Errors: {aggregate['num_errors']}",
            f"Timeouts: {aggregate['num_timeouts']}",
            f"Total Incorrect Elements: {aggregate['total_incorrect_elements']}",
            "",
            "=== Per-Example Results ==="
        ]

        for i, feedback in enumerate(feedbacks):
            lines.append(
                f"Example {i+1}: status={feedback.execution_status}, "
                f"accuracy={feedback.accuracy:.4f}, "
                f"shape_match={feedback.shape_match}"
            )

        return "\n".join(lines)
