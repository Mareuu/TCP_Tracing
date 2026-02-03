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
