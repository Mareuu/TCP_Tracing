"""
TCP Domain Adapter for ARC (Abstraction and Reasoning Corpus)

This adapter wraps the existing ARC evaluation logic in the DomainAdapter interface,
enabling TCP's methodology to be compared across domains while maintaining
backward compatibility with existing ARC-specific code.
"""

import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

from .base import (
    DomainAdapter, Problem, Feedback, EvaluationResult,
    RawMetrics, EvaluationStatus
)


class ARCDomainAdapter(DomainAdapter):
    """
    Domain adapter for ARC (Abstraction and Reasoning Corpus) tasks.

    This adapter implements the DomainAdapter interface for ARC tasks,
    wrapping the existing evaluation logic from tcp_evaluation_utils.py
    and analyze_transformation_pattern() from tcp_refine.py.
    """

    def __init__(self, base_path: str = "/data/TCP_Tracing"):
        """
        Initialize the ARC domain adapter.

        Args:
            base_path: Base path to the ARC dataset
        """
        self.base_path = base_path

    @property
    def domain_name(self) -> str:
        return "arc"

    def load_dataset(
        self,
        path: str = None,
        split: str = "train"
    ) -> Dict[str, Problem]:
        """
        Load ARC problems from the dataset.

        Args:
            path: Path to dataset (uses base_path if None)
            split: Dataset split ('train', 'val', 'eval', 'test')

        Returns:
            Dictionary mapping task UIDs to Problem objects
        """
        from tcp_core.tcp_dataset import get_dataset

        data_path = path or self.base_path
        train_data, val_data, test_data = get_dataset(data_path=data_path)

        # Select appropriate split
        if split == "train":
            raw_data = train_data
        elif split in ["val", "eval"]:
            raw_data = val_data
        elif split == "test":
            raw_data = test_data
        else:
            # Combine train and val for general use
            raw_data = {**train_data, **val_data}

        # Convert to Problem format
        problems = {}
        for uid, task_data in raw_data.items():
            # Format examples from train pairs
            examples = []
            for i, pair in enumerate(task_data.get('train', [])):
                examples.append({
                    'input': pair['input'],
                    'output': pair['output'],
                })

            # Format test cases
            test_cases = []
            for i, pair in enumerate(task_data.get('test', [])):
                test_cases.append({
                    'input': pair['input'],
                    'output': pair.get('output'),  # May be None for actual test set
                })

            # Generate description
            description = self._generate_task_description(task_data)

            problems[uid] = Problem(
                uid=uid,
                description=description,
                examples=examples,
                test_cases=test_cases,
                metadata={
                    'raw_task_data': task_data,
                    'num_train': len(examples),
                    'num_test': len(test_cases),
                }
            )

        return problems

    def _generate_task_description(self, task_data: Dict) -> str:
        """Generate a description for an ARC task."""
        train_pairs = task_data.get('train', [])
        if not train_pairs:
            return "ARC visual reasoning task"

        # Analyze grid dimensions
        input_shapes = [np.array(p['input']).shape for p in train_pairs]
        output_shapes = [np.array(p['output']).shape for p in train_pairs]

        desc_parts = [
            "This is an ARC (Abstraction and Reasoning Corpus) task.",
            f"The task has {len(train_pairs)} training example(s).",
        ]

        # Check if shapes change
        if all(i == o for i, o in zip(input_shapes, output_shapes)):
            desc_parts.append("The output grid has the same dimensions as the input.")
        else:
            desc_parts.append("The output grid may have different dimensions than the input.")

        return " ".join(desc_parts)

    def evaluate(
        self,
        code: str,
        problem: Problem,
        timeout: int = 5,
        use_train: bool = True
    ) -> EvaluationResult:
        """
        Evaluate code against an ARC problem.

        Args:
            code: Python code containing a transform() function
            problem: ARC problem to evaluate against
            timeout: Execution timeout in seconds
            use_train: If True, evaluate on training examples; if False, on test cases

        Returns:
            EvaluationResult with raw metrics and interpreted feedback
        """
        # Get pairs to evaluate
        if use_train:
            pairs = [{'input': ex['input'], 'output': ex['output']}
                     for ex in problem.examples]
        else:
            pairs = [{'input': tc['input'], 'output': tc['output']}
                     for tc in problem.test_cases if tc.get('output') is not None]

        if not pairs:
            return EvaluationResult(
                problem_uid=problem.uid,
                code=code,
                feedback=Feedback(
                    raw_metrics=RawMetrics(
                        execution_status=EvaluationStatus.ERROR,
                        passed=False,
                        pass_rate=0.0,
                        total_cases=0,
                        passed_cases=0,
                        failed_cases=0,
                        error_message="No evaluation pairs available"
                    )
                )
            )

        # Execute code on each pair
        all_results = []
        accuracies = []
        outputs = []
        expected = []

        for i, pair in enumerate(pairs):
            input_grid = np.array(pair['input'])
            expected_output = np.array(pair['output'])

            # Execute code
            actual_output = self._execute_code(code, input_grid, timeout)

            # Compare results
            result = self._compare_grids(actual_output, expected_output)
            all_results.append(result)

            if result['accuracy'] is not None:
                accuracies.append(result['accuracy'])

            outputs.append(actual_output)
            expected.append(expected_output)

        # Compute aggregate metrics
        if accuracies:
            avg_accuracy = sum(accuracies) / len(accuracies)
            all_passed = all(a == 1.0 for a in accuracies)
        else:
            avg_accuracy = 0.0
            all_passed = False

        # Determine execution status
        if any(r['status'] == 'timeout' for r in all_results):
            exec_status = EvaluationStatus.TIMEOUT
        elif any(r['status'] == 'error' for r in all_results):
            exec_status = EvaluationStatus.ERROR
        else:
            exec_status = EvaluationStatus.SUCCESS

        # Build raw metrics
        raw_metrics = RawMetrics(
            execution_status=exec_status,
            passed=all_passed,
            pass_rate=avg_accuracy,
            total_cases=len(pairs),
            passed_cases=sum(1 for a in accuracies if a == 1.0),
            failed_cases=len(pairs) - sum(1 for a in accuracies if a == 1.0),
            error_message=all_results[0].get('error') if all_results else None
        )

        # Build interpreted feedback
        interpreted = self._generate_interpreted_feedback(all_results, avg_accuracy)

        return EvaluationResult(
            problem_uid=problem.uid,
            code=code,
            feedback=Feedback(
                raw_metrics=raw_metrics,
                interpreted_feedback=interpreted
            ),
            outputs=outputs,
            expected_outputs=expected,
            metadata={'per_pair_results': all_results}
        )

    def _execute_code(
        self,
        code: str,
        input_grid: np.ndarray,
        timeout: int
    ) -> Any:
        """Execute code with timeout and return result."""
        from func_timeout import func_timeout, FunctionTimedOut
        import math
        import copy

        def execute():
            namespace = {
                'np': np, 'numpy': np, 'math': math, 'copy': copy,
                'input_grid': input_grid.copy(),
            }
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
            result = func_timeout(timeout, execute)
            if isinstance(result, list):
                result = np.array(result)
            return result
        except FunctionTimedOut:
            return "ERROR: Execution timed out"
        except Exception as e:
            return f"ERROR: {str(e)}"

    def _compare_grids(
        self,
        actual: Any,
        expected: np.ndarray
    ) -> Dict[str, Any]:
        """Compare actual output with expected output."""
        result = {
            'status': 'success',
            'accuracy': None,
            'shape_match': False,
            'error': None,
            'actual': actual,
            'expected': expected,
        }

        # Handle error strings
        if isinstance(actual, str) and "ERROR" in actual:
            result['status'] = 'timeout' if 'timed out' in actual.lower() else 'error'
            result['error'] = actual
            result['accuracy'] = 0.0
            return result

        # Convert to numpy
        if not isinstance(actual, np.ndarray):
            try:
                actual = np.array(actual)
            except:
                result['status'] = 'error'
                result['error'] = 'Cannot convert output to array'
                result['accuracy'] = 0.0
                return result

        # Check dimensions
        if len(actual.shape) != 2:
            result['status'] = 'error'
            result['error'] = f'Output not 2D: shape={actual.shape}'
            result['accuracy'] = 0.0
            return result

        # Check shape
        if actual.shape != expected.shape:
            result['status'] = 'success'  # Code ran, just wrong shape
            result['shape_match'] = False
            result['accuracy'] = 0.0
            return result

        result['shape_match'] = True

        # Calculate accuracy
        total = np.prod(expected.shape)
        correct = np.sum(actual == expected)
        result['accuracy'] = float(correct / total) if total > 0 else 0.0
        result['actual'] = actual

        return result

    def _generate_interpreted_feedback(
        self,
        results: List[Dict],
        avg_accuracy: float
    ) -> List[str]:
        """Generate interpreted feedback from evaluation results."""
        feedback = []

        # High-level summary
        if any(r['status'] == 'error' or r['status'] == 'timeout' for r in results):
            feedback.append("The code failed to execute on one or more examples.")
        elif not all(r.get('shape_match', False) for r in results):
            feedback.append("The output grid dimensions are incorrect for one or more examples.")
        elif avg_accuracy < 0.5:
            feedback.append("The transformation logic appears fundamentally incorrect.")
        elif avg_accuracy < 0.9:
            feedback.append("The code captures some patterns but has significant errors.")
        elif avg_accuracy < 1.0:
            feedback.append("The code is close but has minor errors.")
        else:
            feedback.append("The code appears to be correct.")

        # Per-example feedback
        for i, r in enumerate(results):
            feedback.append(f"\n--- Example {i+1} ---")
            if r['status'] == 'error':
                feedback.append(f"Error: {r.get('error', 'Unknown error')}")
            elif r['status'] == 'timeout':
                feedback.append("Execution timed out")
            elif not r.get('shape_match', False):
                feedback.append(f"Shape mismatch: expected {r['expected'].shape}, got {r.get('actual', 'N/A')}")
            else:
                acc = r.get('accuracy', 0)
                feedback.append(f"Pixel accuracy: {acc*100:.1f}%")

                # Color analysis if shapes match
                if acc < 1.0 and isinstance(r.get('actual'), np.ndarray):
                    actual = r['actual']
                    expected = r['expected']
                    actual_colors = set(actual.flatten())
                    expected_colors = set(expected.flatten())
                    missing = expected_colors - actual_colors
                    extra = actual_colors - expected_colors
                    if missing:
                        feedback.append(f"Missing colors: {sorted(missing)}")
                    if extra:
                        feedback.append(f"Extra colors: {sorted(extra)}")

        return feedback

    def get_structural_hints(self, problem: Problem) -> List[str]:
        """
        Generate structural hints by analyzing the training examples.

        This replicates the analyze_transformation_pattern() functionality.

        Args:
            problem: ARC problem to analyze

        Returns:
            List of structural hints
        """
        hints = []
        examples = problem.examples

        if not examples:
            return hints

        # Check if grid size changes
        size_changes = False
        for ex in examples:
            input_shape = np.array(ex['input']).shape
            output_shape = np.array(ex['output']).shape
            if input_shape != output_shape:
                size_changes = True
                break

        if size_changes:
            hints.append("Grid dimensions change between input and output")
        else:
            hints.append("Grid dimensions are preserved")

        # Check color usage
        input_colors = set()
        output_colors = set()
        for ex in examples:
            input_colors.update(np.array(ex['input']).flatten().tolist())
            output_colors.update(np.array(ex['output']).flatten().tolist())

        if output_colors - input_colors:
            hints.append(f"New colors appear in output: {output_colors - input_colors}")
        if input_colors - output_colors:
            hints.append(f"Colors removed from output: {input_colors - output_colors}")

        return hints

    def get_refinement_prompt(
        self,
        problem: Problem,
        code: str,
        feedback: Feedback,
        iteration: int,
        enable_hints: bool = True
    ) -> str:
        """
        Generate a refinement prompt for ARC tasks.

        Args:
            problem: ARC problem
            code: Current code to refine
            feedback: Feedback from evaluation
            iteration: Current iteration number
            enable_hints: Whether to include structural hints

        Returns:
            Prompt string
        """
        # Format training examples
        examples_str = ""
        for i, ex in enumerate(problem.examples):
            examples_str += f"\n## Training Example {i+1}:\n"
            examples_str += f"### Input:\n{json.dumps(ex['input'])}\n"
            examples_str += f"### Output:\n{json.dumps(ex['output'])}\n"

        # Get hints
        hints = self.get_structural_hints(problem) if enable_hints else []
        hints_str = "\n".join(f"- {h}" for h in hints) if hints else "(Hints disabled)"

        # Format feedback based on type
        if enable_hints:
            feedback_str = self.get_interpreted_feedback_string(feedback)
        else:
            feedback_str = self.get_raw_feedback_string(feedback)

        return f"""You are an AI assistant specialized in repairing Python code for ARC-AGI tasks.

# Task Description
{problem.description}

# Pattern Analysis
{hints_str}

# Training Examples
{examples_str}

# Current Code to Repair (Iteration {iteration}):
```python
{code}
```

# Current Accuracy: {feedback.raw_metrics.pass_rate:.1%}

# Evaluation Feedback:
{feedback_str}

# Instructions
Analyze the feedback and fix the code. Provide the corrected implementation:

```python
def transform(input_grid):
    # Your corrected solution
    pass
```"""

    def get_initial_prompt(
        self,
        problem: Problem,
        enable_hints: bool = True
    ) -> str:
        """Generate initial code generation prompt for ARC."""
        examples_str = ""
        for i, ex in enumerate(problem.examples):
            examples_str += f"\n## Training Example {i+1}:\n"
            examples_str += f"### Input:\n{json.dumps(ex['input'])}\n"
            examples_str += f"### Output:\n{json.dumps(ex['output'])}\n"

        hints = self.get_structural_hints(problem) if enable_hints else []
        hints_str = "\n".join(f"- {h}" for h in hints) if hints else "(No hints)"

        return f"""You are an expert at solving ARC (Abstraction and Reasoning Corpus) tasks.

# Task Description
{problem.description}

# Pattern Hints
{hints_str}

# Training Examples
{examples_str}

# Instructions
Write a Python function called `transform` that takes an input grid (2D list of integers 0-9)
and returns the transformed output grid.

```python
def transform(input_grid):
    import numpy as np
    grid = np.array(input_grid)
    # Your solution here
    return output_grid.tolist()
```"""
