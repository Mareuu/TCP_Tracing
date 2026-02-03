"""
TCP Domain Adapter for HumanEval

This adapter implements the DomainAdapter interface for HumanEval tasks,
demonstrating that TCP's iterative refinement methodology generalizes
beyond ARC to code generation tasks.

HumanEval evaluates code correctness through test case execution,
providing binary pass/fail feedback per test case.
"""

import json
import re
from typing import Dict, List, Tuple, Optional, Any

from .base import (
    DomainAdapter, Problem, Feedback, EvaluationResult,
    RawMetrics, EvaluationStatus
)


class HumanEvalDomainAdapter(DomainAdapter):
    """
    Domain adapter for HumanEval code generation tasks.

    HumanEval problems consist of:
    - A function signature and docstring
    - Test cases that verify the function
    - The task is to complete the function body

    Feedback is test-based (pass/fail) rather than pixel-accuracy.
    """

    def __init__(self, timeout: int = 5):
        """
        Initialize the HumanEval domain adapter.

        Args:
            timeout: Default execution timeout in seconds
        """
        self.timeout = timeout

    @property
    def domain_name(self) -> str:
        return "humaneval"

    def load_dataset(
        self,
        path: str,
        split: str = "test"
    ) -> Dict[str, Problem]:
        """
        Load HumanEval problems from a JSONL file.

        Expected format (HumanEval standard):
        {
            "task_id": "HumanEval/0",
            "prompt": "def function_name(...) -> ...:\\n    \"\"\"docstring\"\"\"\\n",
            "canonical_solution": "...",
            "test": "def check(candidate):\\n    assert ...",
            "entry_point": "function_name"
        }

        Args:
            path: Path to HumanEval JSONL file
            split: Ignored for HumanEval (single split)

        Returns:
            Dictionary mapping task IDs to Problem objects
        """
        problems = {}

        try:
            with open(path, 'r') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        problem = self._convert_to_problem(data)
                        problems[problem.uid] = problem
        except FileNotFoundError:
            # Return empty dict if file not found
            pass

        return problems

    def _convert_to_problem(self, data: Dict) -> Problem:
        """Convert HumanEval JSON entry to Problem format."""
        task_id = data.get('task_id', data.get('name', 'unknown'))
        prompt = data.get('prompt', '')
        test_code = data.get('test', '')
        entry_point = data.get('entry_point', self._extract_entry_point(prompt))
        canonical = data.get('canonical_solution', '')

        # Extract function signature and docstring
        description = self._format_description(prompt)

        # Parse test cases from test code
        test_cases = self._parse_test_cases(test_code, entry_point)

        return Problem(
            uid=task_id,
            description=description,
            examples=[],  # HumanEval doesn't have explicit examples
            test_cases=test_cases,
            metadata={
                'prompt': prompt,
                'entry_point': entry_point,
                'test_code': test_code,
                'canonical_solution': canonical,
            }
        )

    def _format_description(self, prompt: str) -> str:
        """Format the HumanEval prompt as a description."""
        # Extract docstring if present
        docstring_match = re.search(r'"""(.*?)"""', prompt, re.DOTALL)
        if docstring_match:
            docstring = docstring_match.group(1).strip()
            return f"Complete the following Python function:\n\n{prompt}\n\nDescription:\n{docstring}"
        return f"Complete the following Python function:\n\n{prompt}"

    def _extract_entry_point(self, prompt: str) -> str:
        """Extract function name from prompt."""
        match = re.search(r'def\s+(\w+)\s*\(', prompt)
        return match.group(1) if match else 'solution'

    def _parse_test_cases(self, test_code: str, entry_point: str) -> List[Dict]:
        """
        Parse test cases from HumanEval test code.

        Returns list of test case dictionaries with 'assertion' key.
        """
        test_cases = []

        # Find all assert statements
        assert_pattern = r'assert\s+(.+?)(?:\n|$)'
        matches = re.findall(assert_pattern, test_code)

        for i, assertion in enumerate(matches):
            test_cases.append({
                'assertion': assertion.strip(),
                'test_id': i,
            })

        return test_cases

    def evaluate(
        self,
        code: str,
        problem: Problem,
        timeout: int = None
    ) -> EvaluationResult:
        """
        Evaluate code against HumanEval test cases.

        Args:
            code: Python code (complete function implementation)
            problem: HumanEval problem
            timeout: Execution timeout (uses default if None)

        Returns:
            EvaluationResult with pass/fail metrics
        """
        from func_timeout import func_timeout, FunctionTimedOut

        timeout = timeout or self.timeout
        entry_point = problem.metadata.get('entry_point', 'solution')
        test_code = problem.metadata.get('test_code', '')

        # Build complete code to execute
        full_code = self._build_test_code(code, test_code, entry_point)

        # Execute and collect results
        test_results = []
        all_passed = True
        error_message = None

        def run_tests():
            namespace = {}
            exec(full_code, namespace)
            # If we get here without exception, tests passed
            return True

        try:
            result = func_timeout(timeout, run_tests)
            passed = True
        except FunctionTimedOut:
            passed = False
            all_passed = False
            error_message = "Execution timed out"
            exec_status = EvaluationStatus.TIMEOUT
        except AssertionError as e:
            passed = False
            all_passed = False
            error_message = f"Assertion failed: {str(e)}"
            exec_status = EvaluationStatus.SUCCESS  # Code ran, test failed
        except SyntaxError as e:
            passed = False
            all_passed = False
            error_message = f"Syntax error: {str(e)}"
            exec_status = EvaluationStatus.COMPILATION_ERROR
        except Exception as e:
            passed = False
            all_passed = False
            error_message = f"Runtime error: {str(e)}"
            exec_status = EvaluationStatus.ERROR

        if passed:
            exec_status = EvaluationStatus.SUCCESS

        # Calculate metrics
        # For HumanEval, it's typically all-or-nothing
        num_tests = len(problem.test_cases) or 1
        passed_tests = num_tests if passed else 0

        raw_metrics = RawMetrics(
            execution_status=exec_status,
            passed=passed,
            pass_rate=1.0 if passed else 0.0,
            total_cases=num_tests,
            passed_cases=passed_tests,
            failed_cases=num_tests - passed_tests,
            error_message=error_message
        )

        # Generate interpreted feedback
        interpreted = self._generate_interpreted_feedback(
            passed, error_message, code, problem
        )

        return EvaluationResult(
            problem_uid=problem.uid,
            code=code,
            feedback=Feedback(
                raw_metrics=raw_metrics,
                interpreted_feedback=interpreted
            ),
            metadata={
                'passed': passed,
                'error': error_message,
            }
        )

    def _build_test_code(self, code: str, test_code: str, entry_point: str) -> str:
        """Build complete executable code with tests."""
        # Replace 'candidate' with the actual function name in tests
        test_code_modified = test_code.replace('candidate', entry_point)

        return f"""
{code}

{test_code_modified}

check({entry_point})
"""

    def _generate_interpreted_feedback(
        self,
        passed: bool,
        error: Optional[str],
        code: str,
        problem: Problem
    ) -> List[str]:
        """Generate interpreted feedback for HumanEval evaluation."""
        feedback = []

        if passed:
            feedback.append("All test cases passed successfully.")
        elif error:
            if "timed out" in error.lower():
                feedback.append("The code execution timed out. Consider optimizing for efficiency.")
            elif "Syntax error" in error:
                feedback.append(f"The code has a syntax error: {error}")
            elif "Assertion failed" in error:
                feedback.append("One or more test cases failed.")
                feedback.append(f"Error: {error}")
                feedback.append("Review the function logic and edge cases.")
            else:
                feedback.append(f"The code raised an error during execution: {error}")
        else:
            feedback.append("The code failed for unknown reasons.")

        return feedback

    def get_structural_hints(self, problem: Problem) -> List[str]:
        """
        Generate hints for HumanEval problems.

        Unlike ARC, HumanEval hints are extracted from the docstring
        and function signature.

        Args:
            problem: HumanEval problem

        Returns:
            List of hints
        """
        hints = []
        prompt = problem.metadata.get('prompt', '')

        # Extract input/output types from signature
        type_match = re.search(r'def\s+\w+\s*\((.*?)\)\s*->\s*(\w+)', prompt)
        if type_match:
            params = type_match.group(1)
            return_type = type_match.group(2)
            hints.append(f"Function parameters: {params}")
            hints.append(f"Return type: {return_type}")

        # Extract examples from docstring if present
        example_match = re.search(r'>>>\s*(.+?)(?:\n|$)', prompt)
        if example_match:
            hints.append("Docstring contains usage examples")

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
        Generate refinement prompt for HumanEval.

        Args:
            problem: HumanEval problem
            code: Current code
            feedback: Evaluation feedback
            iteration: Iteration number
            enable_hints: Whether to include hints

        Returns:
            Prompt string
        """
        prompt = problem.metadata.get('prompt', problem.description)
        hints = self.get_structural_hints(problem) if enable_hints else []
        hints_str = "\n".join(f"- {h}" for h in hints) if hints else "(No hints)"

        if enable_hints:
            feedback_str = self.get_interpreted_feedback_string(feedback)
        else:
            feedback_str = self.get_raw_feedback_string(feedback)

        return f"""You are an expert Python programmer. Fix the failing code.

# Problem
{prompt}

# Hints
{hints_str}

# Current Code (Iteration {iteration}):
```python
{code}
```

# Test Results:
- Passed: {feedback.raw_metrics.passed}
- Pass Rate: {feedback.raw_metrics.pass_rate:.1%}

# Feedback:
{feedback_str}

# Instructions
Fix the code to pass all test cases. Provide the complete corrected function:

```python
{problem.metadata.get('prompt', 'def solution():')}
    # Your implementation
    pass
```"""

    def get_initial_prompt(
        self,
        problem: Problem,
        enable_hints: bool = True
    ) -> str:
        """Generate initial code generation prompt for HumanEval."""
        prompt = problem.metadata.get('prompt', problem.description)
        hints = self.get_structural_hints(problem) if enable_hints else []
        hints_str = "\n".join(f"- {h}" for h in hints) if hints else "(No hints)"

        return f"""You are an expert Python programmer.

# Problem
{prompt}

# Hints
{hints_str}

# Instructions
Complete the function implementation. Make sure to handle edge cases.

```python
{prompt}
    # Your implementation here
    pass
```"""
