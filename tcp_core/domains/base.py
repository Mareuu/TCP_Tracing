"""
TCP Domain Abstraction Layer - Base Interfaces

This module defines the abstract interfaces for domain adapters in the TCP framework.
Domain adapters enable TCP's iterative refinement methodology to work across different
problem domains (ARC, HumanEval, GSM8K, etc.) while maintaining a consistent interface.

The abstraction addresses reviewer feedback about demonstrating that TCP's methodology
works independently of domain-specific knowledge by:
1. Separating domain-agnostic feedback (raw metrics) from domain-specific interpretation
2. Providing a common interface for loading problems, evaluating code, and generating prompts
3. Enabling systematic ablation studies across domains
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple, Optional, Any, Union


class EvaluationStatus(Enum):
    """Status of code evaluation."""
    SUCCESS = "success"          # Code executed successfully
    ERROR = "error"              # Runtime error during execution
    TIMEOUT = "timeout"          # Execution timed out
    COMPILATION_ERROR = "compilation_error"  # Code failed to compile/parse
    INVALID_OUTPUT = "invalid_output"        # Output format is invalid


class ProgressState(Enum):
    """
    Domain-agnostic progress state for strategy selection.

    This abstraction allows strategy selection without domain-specific
    accuracy thresholds. Each domain maps its metrics to these states.
    """
    FAILING = "failing"          # No meaningful progress (errors, very low accuracy)
    STRUGGLING = "struggling"    # Some progress but significant issues remain
    PROGRESSING = "progressing"  # Making steady progress
    CLOSE = "close"              # Near solution, minor issues
    SOLVED = "solved"            # Problem solved


class RepairStrategy(Enum):
    """
    Repair strategies for code refinement.

    These are domain-agnostic strategies that can be applied
    regardless of the problem domain.
    """
    COMPLETE_REWRITE = "complete_rewrite"    # Start fresh with new approach
    MAJOR_RESTRUCTURE = "major_restructure"  # Significant changes to logic
    TARGETED_FIX = "targeted_fix"            # Fix specific issues
    PRECISION_FIX = "precision_fix"          # Minor adjustments
    CONTINUE = "continue"                    # Keep current approach
    PERFECT = "perfect"                      # Already solved


class StrategyMode(Enum):
    """
    Mode for selecting repair strategy.

    - ADAPTIVE_THRESHOLD: Use domain-specific accuracy thresholds (original behavior)
    - HISTORY_BASED: Use improvement history patterns (domain-agnostic)
    - FIXED: Always use same strategy (baseline for ablation)
    """
    ADAPTIVE_THRESHOLD = "adaptive_threshold"
    HISTORY_BASED = "history_based"
    FIXED = "fixed"


@dataclass
class RefinementHistory:
    """
    Track refinement history for history-based strategy selection.

    This enables domain-agnostic strategy decisions based on
    improvement patterns rather than absolute accuracy values.
    """
    accuracies: List[float] = field(default_factory=list)
    strategies_used: List[str] = field(default_factory=list)
    improvements: List[float] = field(default_factory=list)  # delta from previous

    def add_iteration(self, accuracy: float, strategy: str):
        """Record a refinement iteration."""
        if self.accuracies:
            improvement = accuracy - self.accuracies[-1]
        else:
            improvement = accuracy
        self.accuracies.append(accuracy)
        self.strategies_used.append(strategy)
        self.improvements.append(improvement)

    @property
    def current_accuracy(self) -> float:
        return self.accuracies[-1] if self.accuracies else 0.0

    @property
    def iterations_without_improvement(self) -> int:
        """Count consecutive iterations without improvement."""
        count = 0
        for imp in reversed(self.improvements):
            if imp <= 0.001:  # Tiny threshold for floating point
                count += 1
            else:
                break
        return count

    @property
    def recent_trend(self) -> str:
        """Analyze recent improvement trend."""
        if len(self.improvements) < 2:
            return "unknown"
        recent = self.improvements[-3:]  # Last 3 iterations
        avg_improvement = sum(recent) / len(recent)
        if avg_improvement > 0.05:
            return "improving"
        elif avg_improvement < -0.01:
            return "regressing"
        else:
            return "stagnant"

    @property
    def best_accuracy(self) -> float:
        return max(self.accuracies) if self.accuracies else 0.0

    @property
    def is_stuck(self) -> bool:
        """Check if refinement is stuck (no progress for multiple iterations)."""
        return self.iterations_without_improvement >= 3


@dataclass
class Problem:
    """
    Domain-agnostic representation of a problem.

    This is the common format for problems across all domains.
    Domain-specific adapters are responsible for converting their
    native format to this structure.
    """
    uid: str                              # Unique identifier
    description: str                      # Human-readable problem description
    examples: List[Dict[str, Any]]        # Training/demonstration examples
    test_cases: List[Dict[str, Any]]      # Test cases for evaluation
    metadata: Dict[str, Any] = field(default_factory=dict)  # Domain-specific metadata


@dataclass
class RawMetrics:
    """
    Domain-agnostic raw metrics from evaluation.

    These metrics are purely numerical and contain no domain interpretation.
    They are designed to enable fair comparison of method contribution
    across different domains.
    """
    execution_status: EvaluationStatus
    passed: bool                          # Did the code pass all test cases?
    pass_rate: float                      # Fraction of test cases passed (0.0 to 1.0)
    total_cases: int                      # Total number of test cases
    passed_cases: int                     # Number of passed test cases
    failed_cases: int                     # Number of failed test cases
    error_message: Optional[str] = None   # Error message if any


@dataclass
class Feedback:
    """
    Feedback structure containing both raw metrics and interpreted feedback.

    The separation of raw_metrics from interpreted_feedback enables:
    - Raw feedback mode: only provide raw_metrics (domain-agnostic)
    - Interpreted mode: provide both raw and interpreted feedback
    """
    raw_metrics: RawMetrics               # Always provided (domain-agnostic)
    interpreted_feedback: List[str] = field(default_factory=list)  # Optional domain-specific


@dataclass
class EvaluationResult:
    """
    Complete result of evaluating code on a problem.

    Contains the code's output, feedback, and evaluation details.
    """
    problem_uid: str
    code: str
    feedback: Feedback
    outputs: List[Any] = field(default_factory=list)  # Actual outputs for each test case
    expected_outputs: List[Any] = field(default_factory=list)  # Expected outputs
    execution_time: float = 0.0           # Total execution time in seconds
    metadata: Dict[str, Any] = field(default_factory=dict)


class DomainAdapter(ABC):
    """
    Abstract base class for domain adapters.

    A domain adapter provides the bridge between TCP's domain-agnostic
    refinement loop and a specific problem domain. Each adapter must implement:
    - Dataset loading
    - Code evaluation
    - Prompt generation
    - Optional: domain-specific hints
    """

    @property
    @abstractmethod
    def domain_name(self) -> str:
        """
        Return the name of this domain (e.g., 'arc', 'humaneval', 'gsm8k').

        Returns:
            String identifier for the domain
        """
        pass

    @abstractmethod
    def load_dataset(
        self,
        path: str,
        split: str = "train"
    ) -> Dict[str, Problem]:
        """
        Load problems from a dataset.

        Args:
            path: Path to the dataset file or directory
            split: Dataset split to load ('train', 'val', 'test')

        Returns:
            Dictionary mapping problem UIDs to Problem objects
        """
        pass

    @abstractmethod
    def evaluate(
        self,
        code: str,
        problem: Problem,
        timeout: int = 5
    ) -> EvaluationResult:
        """
        Evaluate code against a problem.

        Args:
            code: The Python code to evaluate
            problem: The problem to evaluate against
            timeout: Execution timeout in seconds

        Returns:
            EvaluationResult containing feedback and metrics
        """
        pass

    @abstractmethod
    def get_refinement_prompt(
        self,
        problem: Problem,
        code: str,
        feedback: Feedback,
        iteration: int,
        enable_hints: bool = True
    ) -> str:
        """
        Generate a refinement prompt for the LLM.

        Args:
            problem: The problem being solved
            code: Current code that needs refinement
            feedback: Feedback from the last evaluation
            iteration: Current refinement iteration (1-indexed)
            enable_hints: Whether to include domain-specific hints

        Returns:
            Prompt string for the LLM
        """
        pass

    def get_structural_hints(self, problem: Problem) -> List[str]:
        """
        Generate domain-specific structural hints for a problem.

        This is the injection point for domain knowledge. By default,
        returns an empty list (no hints). Subclasses can override this
        to provide domain-specific guidance.

        Args:
            problem: The problem to analyze

        Returns:
            List of hint strings (empty by default)
        """
        return []

    def get_initial_prompt(
        self,
        problem: Problem,
        enable_hints: bool = True
    ) -> str:
        """
        Generate an initial code generation prompt.

        Args:
            problem: The problem to solve
            enable_hints: Whether to include domain-specific hints

        Returns:
            Prompt string for initial code generation
        """
        # Default implementation - subclasses should override for better results
        examples_str = self._format_examples(problem.examples)
        hints = self.get_structural_hints(problem) if enable_hints else []
        hints_str = "\n".join(f"- {h}" for h in hints) if hints else "No hints available."

        return f"""You are an expert Python programmer.

# Problem Description
{problem.description}

# Examples
{examples_str}

# Hints
{hints_str}

# Instructions
Write a Python function called `transform` that solves this problem.
The function should take the input and return the expected output.

```python
def transform(input_data):
    # Your solution here
    pass
```
"""

    def _format_examples(self, examples: List[Dict[str, Any]]) -> str:
        """
        Format examples for display in prompts.

        Args:
            examples: List of example dictionaries

        Returns:
            Formatted string representation
        """
        lines = []
        for i, ex in enumerate(examples, 1):
            lines.append(f"## Example {i}")
            for key, value in ex.items():
                lines.append(f"### {key.title()}")
                lines.append(str(value))
            lines.append("")
        return "\n".join(lines)

    def get_raw_feedback_string(self, feedback: Feedback) -> str:
        """
        Format raw metrics as a string for LLM consumption.

        This method provides domain-agnostic feedback only.

        Args:
            feedback: Feedback object

        Returns:
            String containing only raw metrics
        """
        metrics = feedback.raw_metrics
        lines = [
            f"Execution Status: {metrics.execution_status.value}",
            f"Passed: {metrics.passed}",
            f"Pass Rate: {metrics.pass_rate:.4f}",
            f"Total Cases: {metrics.total_cases}",
            f"Passed Cases: {metrics.passed_cases}",
            f"Failed Cases: {metrics.failed_cases}",
        ]
        if metrics.error_message:
            lines.append(f"Error: {metrics.error_message}")
        return "\n".join(lines)

    def get_interpreted_feedback_string(self, feedback: Feedback) -> str:
        """
        Format both raw metrics and interpreted feedback as a string.

        This method provides full domain-specific feedback.

        Args:
            feedback: Feedback object

        Returns:
            String containing raw metrics and domain interpretation
        """
        raw_str = self.get_raw_feedback_string(feedback)
        if feedback.interpreted_feedback:
            interpreted_str = "\n".join(feedback.interpreted_feedback)
            return f"{raw_str}\n\n# Detailed Feedback\n{interpreted_str}"
        return raw_str

    # =========================================================================
    # Strategy Selection Methods (Domain-agnostic)
    # =========================================================================

    def classify_progress(self, metrics: RawMetrics) -> ProgressState:
        """
        Classify current progress state from metrics.

        This is the ONLY place where domain-specific thresholds should be used.
        Override this method in subclasses for domain-specific classification.

        Default implementation uses pass_rate thresholds (can be overridden).

        Args:
            metrics: Raw metrics from evaluation

        Returns:
            ProgressState classification
        """
        if metrics.passed:
            return ProgressState.SOLVED
        if metrics.execution_status in (EvaluationStatus.ERROR, EvaluationStatus.TIMEOUT):
            return ProgressState.FAILING
        if metrics.pass_rate >= 0.9:
            return ProgressState.CLOSE
        if metrics.pass_rate >= 0.5:
            return ProgressState.PROGRESSING
        if metrics.pass_rate >= 0.2:
            return ProgressState.STRUGGLING
        return ProgressState.FAILING

    def get_strategy_from_progress(
        self,
        progress: ProgressState,
        iteration: int
    ) -> RepairStrategy:
        """
        Select repair strategy based on progress state (threshold-based mode).

        This is domain-agnostic - it works with ProgressState abstraction.

        Args:
            progress: Current progress state
            iteration: Current iteration number

        Returns:
            RepairStrategy to use
        """
        strategy_map = {
            ProgressState.SOLVED: RepairStrategy.PERFECT,
            ProgressState.CLOSE: RepairStrategy.PRECISION_FIX,
            ProgressState.PROGRESSING: RepairStrategy.TARGETED_FIX,
            ProgressState.STRUGGLING: RepairStrategy.MAJOR_RESTRUCTURE,
            ProgressState.FAILING: RepairStrategy.COMPLETE_REWRITE if iteration > 2 else RepairStrategy.MAJOR_RESTRUCTURE,
        }
        return strategy_map.get(progress, RepairStrategy.TARGETED_FIX)

    def get_strategy_from_history(
        self,
        history: RefinementHistory,
        iteration: int
    ) -> RepairStrategy:
        """
        Select repair strategy based on improvement history (history-based mode).

        This is completely domain-agnostic - no accuracy thresholds used.
        Strategy is determined solely by improvement patterns.

        Args:
            history: Refinement history tracking improvements
            iteration: Current iteration number

        Returns:
            RepairStrategy to use
        """
        # Check if solved
        if history.current_accuracy >= 1.0:
            return RepairStrategy.PERFECT

        # Check if stuck for multiple iterations
        if history.is_stuck:
            # Stuck for 3+ iterations - need drastic change
            if history.iterations_without_improvement >= 5:
                return RepairStrategy.COMPLETE_REWRITE
            else:
                return RepairStrategy.MAJOR_RESTRUCTURE

        # Check recent trend
        trend = history.recent_trend
        if trend == "improving":
            # Making progress - continue current approach
            return RepairStrategy.CONTINUE
        elif trend == "regressing":
            # Getting worse - try different approach
            return RepairStrategy.MAJOR_RESTRUCTURE
        else:  # stagnant
            # Not improving but not regressing - targeted fixes
            return RepairStrategy.TARGETED_FIX

    def get_repair_strategy(
        self,
        metrics: RawMetrics,
        history: RefinementHistory,
        iteration: int,
        mode: StrategyMode = StrategyMode.HISTORY_BASED
    ) -> RepairStrategy:
        """
        Main entry point for strategy selection.

        Args:
            metrics: Current evaluation metrics
            history: Refinement history
            iteration: Current iteration number
            mode: Strategy selection mode

        Returns:
            RepairStrategy to use
        """
        if mode == StrategyMode.FIXED:
            return RepairStrategy.TARGETED_FIX  # Always same strategy

        elif mode == StrategyMode.ADAPTIVE_THRESHOLD:
            # Use domain-specific thresholds via classify_progress
            progress = self.classify_progress(metrics)
            return self.get_strategy_from_progress(progress, iteration)

        else:  # HISTORY_BASED (default, most domain-agnostic)
            return self.get_strategy_from_history(history, iteration)

    def get_feedback_detail_level(
        self,
        metrics: RawMetrics,
        history: RefinementHistory,
        mode: StrategyMode = StrategyMode.HISTORY_BASED
    ) -> str:
        """
        Determine feedback detail level.

        Args:
            metrics: Current evaluation metrics
            history: Refinement history
            mode: Strategy selection mode

        Returns:
            Feedback level: "summary", "detailed", or "pixel-level"
        """
        if mode == StrategyMode.FIXED:
            return "detailed"  # Always same level

        elif mode == StrategyMode.ADAPTIVE_THRESHOLD:
            # Threshold-based (original behavior)
            if metrics.pass_rate >= 0.9:
                return "pixel-level"
            elif metrics.pass_rate >= 0.5:
                return "detailed"
            return "summary"

        else:  # HISTORY_BASED
            # History-based: more detail when close to solution or stuck
            if metrics.pass_rate >= 0.95:
                return "pixel-level"
            elif history.is_stuck:
                return "detailed"  # Need more info when stuck
            elif history.recent_trend == "improving":
                return "summary"  # Brief when making progress
            return "detailed"

    def get_num_samples(
        self,
        strategy: RepairStrategy,
        iteration: int,
        mode: StrategyMode = StrategyMode.HISTORY_BASED
    ) -> int:
        """
        Determine number of samples to generate (best-of-N).

        Args:
            strategy: Current repair strategy
            iteration: Current iteration number
            mode: Strategy selection mode

        Returns:
            Number of samples to generate
        """
        if mode == StrategyMode.FIXED:
            return 2  # Always same

        # More samples for drastic strategies or early iterations
        sample_map = {
            RepairStrategy.COMPLETE_REWRITE: 4,
            RepairStrategy.MAJOR_RESTRUCTURE: 3,
            RepairStrategy.TARGETED_FIX: 2,
            RepairStrategy.PRECISION_FIX: 3,  # More samples when close
            RepairStrategy.CONTINUE: 2,
            RepairStrategy.PERFECT: 1,
        }
        base = sample_map.get(strategy, 2)

        # Boost early iterations
        if iteration < 3:
            return min(base + 1, 5)
        return base
