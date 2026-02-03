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

## Feedback Granularity Levels (for ablation experiments)

The module supports different levels of feedback granularity to study
the minimal sufficient feedback required for effective refinement:

- Level 0 (NONE): No feedback - just "try again"
- Level 1 (BINARY): Pass/Fail only
- Level 2 (ACCURACY): Accuracy score only
- Level 3 (SHAPE): Accuracy + shape match info
- Level 4 (COUNT): + error counts
- Level 5 (POSITION): + error positions (raw coordinates)
- Level 6 (FULL_RAW): All raw metrics
- Level 7 (INTERPRETED): + domain-specific interpretation (handled elsewhere)
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np


class FeedbackGranularity(IntEnum):
    """
    Feedback granularity levels for ablation experiments.

    Lower levels provide less information, enabling study of
    minimal sufficient feedback for effective refinement.

    Information content (approximate bits per example):
    - NONE: 0 bits
    - BINARY: 1 bit
    - ACCURACY: ~7 bits (0-100 range)
    - SHAPE: ~8 bits
    - COUNT: ~15 bits
    - POSITION: ~variable (depends on error count)
    - FULL_RAW: all numerical info
    - INTERPRETED: + domain knowledge
    """
    NONE = 0           # "Your code failed. Try again."
    BINARY = 1         # "Pass" or "Fail"
    ACCURACY = 2       # "Accuracy: 0.75"
    SHAPE = 3          # + "Shape: Match/Mismatch"
    COUNT = 4          # + "Errors: 12"
    POSITION = 5       # + "Error positions: [(0,1), ...]"
    FULL_RAW = 6       # All raw metrics
    INTERPRETED = 7    # + Domain-specific interpretation


# Mapping of granularity levels to human-readable names
GRANULARITY_NAMES = {
    FeedbackGranularity.NONE: "none",
    FeedbackGranularity.BINARY: "binary",
    FeedbackGranularity.ACCURACY: "accuracy_only",
    FeedbackGranularity.SHAPE: "accuracy_shape",
    FeedbackGranularity.COUNT: "accuracy_shape_count",
    FeedbackGranularity.POSITION: "with_positions",
    FeedbackGranularity.FULL_RAW: "full_raw",
    FeedbackGranularity.INTERPRETED: "interpreted",
}


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
    def format_at_granularity(
        feedback: RawFeedbackResult,
        granularity: FeedbackGranularity
    ) -> str:
        """
        Format feedback at a specific granularity level.

        This method enables systematic ablation of feedback information,
        allowing study of the minimal sufficient feedback for effective refinement.

        Args:
            feedback: RawFeedbackResult to format
            granularity: FeedbackGranularity level

        Returns:
            Formatted string with information limited to the specified granularity
        """
        if granularity == FeedbackGranularity.NONE:
            if feedback.execution_status != "success":
                return "Your code failed to execute. Please try again."
            elif feedback.accuracy >= 1.0:
                return "Your code passed all tests."
            else:
                return "Your code did not produce the correct output. Please try again."

        if granularity == FeedbackGranularity.BINARY:
            if feedback.execution_status != "success":
                return "Result: FAIL (execution error)"
            elif feedback.accuracy >= 1.0:
                return "Result: PASS"
            else:
                return "Result: FAIL"

        if granularity == FeedbackGranularity.ACCURACY:
            if feedback.execution_status != "success":
                return f"Result: FAIL (execution error)\nAccuracy: 0.0"
            return f"Accuracy: {feedback.accuracy:.4f}"

        if granularity == FeedbackGranularity.SHAPE:
            lines = []
            if feedback.execution_status != "success":
                lines.append(f"Execution: {feedback.execution_status}")
                lines.append("Accuracy: 0.0")
            else:
                lines.append(f"Accuracy: {feedback.accuracy:.4f}")
            lines.append(f"Shape Match: {feedback.shape_match}")
            if not feedback.shape_match and feedback.expected_shape and feedback.actual_shape:
                lines.append(f"Expected Shape: {feedback.expected_shape}")
                lines.append(f"Actual Shape: {feedback.actual_shape}")
            return "\n".join(lines)

        if granularity == FeedbackGranularity.COUNT:
            lines = []
            if feedback.execution_status != "success":
                lines.append(f"Execution: {feedback.execution_status}")
                if feedback.error_message:
                    lines.append(f"Error: {feedback.error_message[:100]}")
            lines.append(f"Accuracy: {feedback.accuracy:.4f}")
            lines.append(f"Shape Match: {feedback.shape_match}")
            lines.append(f"Total Elements: {feedback.total_elements}")
            lines.append(f"Correct: {feedback.correct_elements}")
            lines.append(f"Incorrect: {feedback.incorrect_elements}")
            return "\n".join(lines)

        if granularity == FeedbackGranularity.POSITION:
            lines = []
            if feedback.execution_status != "success":
                lines.append(f"Execution: {feedback.execution_status}")
                if feedback.error_message:
                    lines.append(f"Error: {feedback.error_message[:100]}")
            lines.append(f"Accuracy: {feedback.accuracy:.4f}")
            lines.append(f"Shape Match: {feedback.shape_match}")
            lines.append(f"Correct: {feedback.correct_elements}/{feedback.total_elements}")
            if feedback.error_positions:
                # Limit to first 20 positions to avoid huge prompts
                positions = feedback.error_positions[:20]
                lines.append(f"Error Positions (first {len(positions)}): {positions}")
            return "\n".join(lines)

        # FULL_RAW or higher - use the standard format_as_string
        return RawFeedbackGenerator.format_as_string(feedback, include_positions=True)

    @staticmethod
    def format_aggregate_at_granularity(
        feedbacks: List[RawFeedbackResult],
        granularity: FeedbackGranularity
    ) -> str:
        """
        Format aggregated feedback from multiple examples at a specific granularity.

        Args:
            feedbacks: List of RawFeedbackResult objects
            granularity: FeedbackGranularity level

        Returns:
            Formatted string with information limited to the specified granularity
        """
        if not feedbacks:
            return "No examples to evaluate."

        if granularity == FeedbackGranularity.NONE:
            all_pass = all(f.accuracy >= 1.0 and f.execution_status == "success" for f in feedbacks)
            if all_pass:
                return "All tests passed."
            else:
                return "Some tests failed. Please revise your code."

        if granularity == FeedbackGranularity.BINARY:
            passed = sum(1 for f in feedbacks if f.accuracy >= 1.0 and f.execution_status == "success")
            return f"Passed: {passed}/{len(feedbacks)}"

        # For higher granularities, show per-example results
        lines = []
        aggregate = RawFeedbackGenerator.aggregate_feedback(feedbacks)

        if granularity >= FeedbackGranularity.ACCURACY:
            lines.append(f"Average Accuracy: {aggregate['avg_accuracy']:.4f}")

        if granularity >= FeedbackGranularity.SHAPE:
            lines.append(f"All Shapes Match: {aggregate['all_shapes_match']}")

        if granularity >= FeedbackGranularity.COUNT:
            lines.append(f"Total Incorrect Elements: {aggregate['total_incorrect_elements']}")
            lines.append(f"Execution Errors: {aggregate['num_errors']}")

        lines.append("")
        lines.append("Per-Example Results:")

        for i, feedback in enumerate(feedbacks):
            example_str = RawFeedbackGenerator.format_at_granularity(feedback, granularity)
            # Indent and prefix with example number
            indented = "\n".join(f"  {line}" for line in example_str.split("\n"))
            lines.append(f"Example {i+1}:")
            lines.append(indented)

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
