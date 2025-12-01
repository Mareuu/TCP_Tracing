#!/usr/bin/env python3
"""
TCP Refine - Iterative Code Refinement with Dynamic Feedback

TCP (Tracing & Correcting Program)

This script performs iterative code refinement using dynamic feedback.
"""

import sys
import os

# Add parent directory to path for tcp_core imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import random
import argparse
import json
import pickle
import re
from collections import defaultdict
import time

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from tcp_core import get_dataset, LLM_serv
from tcp_evaluation_utils import get_feedback_for_challenger_dynamic, NumpyEncoder, parse_code_from_response


def analyze_transformation_pattern(train_pairs):
    """Analyze the transformation pattern to provide structural hints."""
    hints = []

    # Check if grid size changes
    size_changes = False
    for pair in train_pairs:
        if np.array(pair['input']).shape != np.array(pair['output']).shape:
            size_changes = True
            break

    if size_changes:
        hints.append("Grid dimensions change between input and output")
    else:
        hints.append("Grid dimensions are preserved")

    # Check color usage
    input_colors = set()
    output_colors = set()
    for pair in train_pairs:
        input_colors.update(np.array(pair['input']).flatten().tolist())
        output_colors.update(np.array(pair['output']).flatten().tolist())

    if output_colors - input_colors:
        hints.append(f"New colors appear: {output_colors - input_colors}")
    if input_colors - output_colors:
        hints.append(f"Colors removed: {input_colors - output_colors}")

    return hints


def parse_arguments():
    """Parse command-line arguments for ARC refinement configuration."""
    parser = argparse.ArgumentParser(
        description="TCP refinement - iterative code improvement with adaptive strategies",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    path_group = parser.add_argument_group("Path configurations")
    path_group.add_argument("--base_path", type=str, default="/data/TCP_Tracing",
                          help="Path to data directory")
    path_group.add_argument("--path_feedback", type=str, required=True,
                          help="Path to load feedback for failing solutions.")
    path_group.add_argument("--path_save_res", type=str, required=True,
                          help="Results saving path")

    model_group = parser.add_argument_group("Model parameters")
    model_group.add_argument("--path_model", type=str, default="qwen",
                           help="HF model identifier")
    model_group.add_argument("--n_gpu", type=int, default=1,
                           help="Number of GPUs")
    model_group.add_argument("--model_len", type=int, default=30000,
                           help="Context length limit")
    model_group.add_argument("--fp8", action=argparse.BooleanOptionalAction, default=True,
                           help="Use FP8 precision")

    inference_group = parser.add_argument_group("Inference parameters")
    inference_group.add_argument("--temperature", type=float, default=0.4,
                               help="Base sampling temperature")
    inference_group.add_argument("--top_p", type=float, default=1.0,
                               help="Top-p nucleus sampling")
    inference_group.add_argument("--min_p", type=float, default=0.05,
                                help="Min_p sampling")
    inference_group.add_argument("--top_k", type=float, default=-1,
                                help="Top-k sampling")
    inference_group.add_argument("--max_tokens", type=int, default=2048,
                               help="max gen tokens")

    dataset_group = parser.add_argument_group("Experimental flags")
    dataset_group.add_argument("--split", type=str, default="val",
                            help="'train' or 'val'")
    dataset_group.add_argument("--arc2", action=argparse.BooleanOptionalAction,
                            help="arc 2")

    rex_group = parser.add_argument_group("Refinement parameters")
    rex_group.add_argument("--max_refinement_retries", type=int, default=10,
                        help="Maximum number of refinement retries (default: 10)")

    advanced_group = parser.add_argument_group("Advanced parameters")
    advanced_group.add_argument("--seed", type=int, default=42,
                            help="Random seed")
    advanced_group.add_argument("--gpu_mem", type=float, default=0.9,
                            help="GPU memory utilization ratio")
    advanced_group.add_argument("--num_tasks", type=int, default=-1,
                            help="Number of tasks to process. -1 for all.")
    return parser.parse_args()


args = parse_arguments()

os.makedirs(args.path_save_res, exist_ok=True)

train_data, val_data, test_data = get_dataset(data_path=args.base_path, arc_2=args.arc2)
data2test = train_data.copy()
data2test.update(val_data)

with open(args.path_feedback, 'r') as f:
    feedback_data = [json.loads(line) for line in f]

if args.num_tasks != -1:
    feedback_data = feedback_data[:args.num_tasks]

print("--- TCP REFINEMENT: Structured Reasoning + Adaptive Strategies ---")
print(json.dumps(feedback_data, indent=4))

MAX_TIMEOUT = int(60 * 60 * 3)
llm = LLM_serv(args.path_model, model_len=args.model_len, seed=args.seed, n_gpu=args.n_gpu,
                temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
                max_timeout=MAX_TIMEOUT, min_p=args.min_p, gpu_mem=args.gpu_mem,
                fp8=args.fp8, max_tokens=args.max_tokens)

for item in feedback_data:
    task_id = item['problem_uid']
    task_data = data2test.get(task_id)
    if not task_data:
        print(f"Skipping task {task_id}: No data found.")
        continue

    puzzle_properties = item.get('inferred_puzzle_properties', {})

    # Handle both string and dict formats for failed_code
    failed_code_raw = item['failed_code']
    if isinstance(failed_code_raw, dict):
        # Extract code from dictionary format (with 'text' and 'code' fields)
        champion_code = failed_code_raw.get('code', failed_code_raw.get('text', ''))
    else:
        # Handle plain string format
        champion_code = failed_code_raw

    # Get initial champion accuracy
    champion_accuracy, champion_feedback = get_feedback_for_challenger_dynamic(
        champion_code, task_data['train'], puzzle_properties)
    if champion_accuracy is None:
        champion_accuracy = 0.0

    print(f"\n=== Task {task_id} ===")
    print(f"Initial accuracy: {champion_accuracy:.2%}")

    # Analyze transformation patterns
    structural_hints = analyze_transformation_pattern(task_data['train'])
    print(f"Pattern analysis: {', '.join(structural_hints)}")

    log_file_path = os.path.join(args.path_save_res, f"{task_id}_refinement_log.jsonl")
    debug_file_path = log_file_path.replace('.jsonl', '_debug.txt')

    # Initialize debug file
    with open(debug_file_path, 'w') as debug_file:
        debug_file.write(f"=== Task {task_id} TCP Refinement Debug Log ===\n")
        debug_file.write(f"Initial champion accuracy: {champion_accuracy:.4f}\n")
        debug_file.write(f"Max iterations: {args.max_refinement_retries}\n")
        debug_file.write(f"Base temperature: {args.temperature}\n")
        debug_file.write(f"Pattern hints: {', '.join(structural_hints)}\n\n")

    task_start_time = time.time()
    total_refinement_time = 0.0

    with open(log_file_path, 'w') as log_file:
        initial_log_entry = {
            "iteration": 0,
            "type": "original",
            "code": champion_code,
            "pixel_accuracy": champion_accuracy,
            "feedback": champion_feedback
        }
        log_file.write(json.dumps(initial_log_entry, cls=NumpyEncoder) + '\n')

        # Track improvements
        no_improvement_count = 0
        previous_champion_accuracy = 0.0
        previous_champion_code = ""
        previous_hypotheses = []  # Track tried hypotheses to avoid repetition

        for i in range(args.max_refinement_retries):
            iteration_start_time = time.time()

            # Early stopping if no improvement for many iterations
            if no_improvement_count >= 5 and i >= 7:  # Balanced patience
                print(f"\n--- Task: {task_id} - Early stopping after {i} iterations ---")
                with open(debug_file_path, 'a') as debug_file:
                    debug_file.write(f"Early stopping triggered at iteration {i}\n")
                break

            print(f"\n--- Task: {task_id}, Iteration: {i+1}/{args.max_refinement_retries} ---")
            print(f"Current Champion Accuracy: {champion_accuracy:.4f}")

            # Determine repair strategy based on accuracy and history
            if champion_accuracy < 0.25 and i > 2:
                repair_strategy = "complete_rewrite"
                print("Strategy: COMPLETE REWRITE (fundamentally wrong approach)")
            elif champion_accuracy < 0.5:
                repair_strategy = "major_restructure"
                print("Strategy: Major restructuring needed")
            elif champion_accuracy < 0.8:
                repair_strategy = "targeted_fix"
                print("Strategy: Targeted fixes")
            elif champion_accuracy < 0.95:
                repair_strategy = "fine_tune"
                print("Strategy: Fine-tuning edge cases")
            elif champion_accuracy < 1.0:
                repair_strategy = "precision_fix"
                print("Strategy: PRECISION FIX (very close to solution)")
            else:
                repair_strategy = "perfect"
                print("Strategy: Already perfect!")

            # Adaptive temperature
            if i < 3:
                current_temperature = min(0.7, args.temperature * 1.75)
            elif champion_accuracy > 0.85:
                current_temperature = max(0.15, args.temperature * 0.4)
            elif champion_accuracy > 0.7:
                current_temperature = max(0.25, args.temperature * 0.6)
            else:
                current_temperature = args.temperature

            print(f"Using temperature: {current_temperature:.2f}")

            # Build training examples
            training_examples_str = ""
            for j, pair in enumerate(task_data['train']):
                input_grid = np.array(pair['input']).tolist()
                output_grid = np.array(pair['output']).tolist()
                training_examples_str += f"\n## Training Example {j+1}:\n"
                training_examples_str += f"### Input:\n{json.dumps(input_grid)}\n"
                training_examples_str += f"### Correct Output:\n{json.dumps(output_grid)}\n"

            puzzle_properties_str = json.dumps(puzzle_properties, indent=4)

            # Progressive feedback detail
            if champion_accuracy < 0.3:
                feedback_detail = "very_detailed"
                if isinstance(champion_feedback, list):
                    final_feedback_payload = "\n".join(champion_feedback[:15])
                    final_feedback_payload += "\n\nCRITICAL: Very low accuracy. The core transformation logic is wrong."
                else:
                    final_feedback_payload = str(champion_feedback)
            elif champion_accuracy < 0.6:
                feedback_detail = "detailed"
                if isinstance(champion_feedback, list):
                    final_feedback_payload = "\n".join(champion_feedback[:10])
                    final_feedback_payload += "\n\nPARTIAL SUCCESS: Some patterns are being recognized correctly."
                else:
                    final_feedback_payload = str(champion_feedback)
            else:
                feedback_detail = "targeted"
                if isinstance(champion_feedback, list):
                    final_feedback_payload = "\n".join(champion_feedback[:5])
                    final_feedback_payload += "\n\nCLOSE: Focus on edge cases and boundary conditions."
                else:
                    final_feedback_payload = str(champion_feedback)

            print(f"Feedback detail level: {feedback_detail}")

            # Build prompt based on strategy
            if repair_strategy == "precision_fix":
                # Special prompt for very high accuracy (95-99%)
                # Check if we've been stuck at high accuracy for too long
                if no_improvement_count >= 2 and champion_accuracy >= 0.95:
                    # Force pattern-level thinking instead of pixel patches
                    prompt_template = f"""You are an AI assistant specialized in perfecting Python code for ARC-AGI tasks.

CRITICAL: The solution has {champion_accuracy:.1%} accuracy but has been stuck for {no_improvement_count} iterations.
This suggests you're missing a PATTERN, not just individual pixels.

# Training Examples:
{training_examples_str}

# Current Code (STUCK AT HIGH ACCURACY):
```python
{champion_code}
```

# Evaluation Feedback:
{final_feedback_payload}

⚠️ IMPORTANT: DO NOT add hardcoded fixes like:
- if grid[x, y] == value: new_grid[x, y] = value
- Special cases for specific coordinates
- Patches for individual pixels

Instead, identify the GENERAL RULE that would handle ALL these failures:

<hypothesis>
State your hypothesis about what PATTERN or RULE is being missed.
</hypothesis>

<plan>
Describe how to implement this pattern generally, not with specific coordinates.
</plan>

Then provide the corrected code:
```python
def transform(input_grid):
    # Your solution with the general pattern
    pass
```"""
                else:
                    prompt_template = f"""You are an AI assistant specialized in perfecting Python code for ARC-AGI tasks.

CRITICAL: The current solution has {champion_accuracy:.1%} accuracy - it's VERY CLOSE to perfect!
Only {(1-champion_accuracy)*100:.1f}% of pixels are wrong. Focus on the SPECIFIC failures.

# Training Examples:
{training_examples_str}

# Current Code (ALMOST PERFECT):
```python
{champion_code}
```

# Evaluation Feedback (FOCUS ON THESE SPECIFIC FAILURES):
{final_feedback_payload}

Your task is to identify the pattern in the failures:

<hypothesis>
What pattern or rule explains why these specific pixels are wrong?
</hypothesis>

<plan>
How will you fix this pattern (avoid hardcoding specific coordinates)?
</plan>

Provide the corrected Python code:
```python
def transform(input_grid):
    # Your minimally modified solution
    pass
```"""
            elif repair_strategy == "complete_rewrite":
                prompt_template = f"""You are an AI assistant specialized in solving ARC-AGI tasks.

⚠️ CRITICAL: The current solution has only {champion_accuracy:.1%} accuracy after {i+1} attempts.
This indicates the approach is FUNDAMENTALLY WRONG.

# Task to solve:
{item['problem_description']}

# Pattern Analysis:
{chr(10).join('- ' + hint for hint in structural_hints)}

# Training Examples:
{training_examples_str}

# FAILED APPROACH (DO NOT COPY THIS):
```python
{champion_code}
```

# Feedback showing why it fails:
{final_feedback_payload}

# Previously tried hypotheses (AVOID THESE):
{chr(10).join('- ' + h for h in previous_hypotheses[-3:])}

Your task is to act as an expert problem solver. Follow these steps:

1. **Analyze the Pattern:** Inside `<pattern_analysis>` tags, describe what actually changes between input and output grids. Look for:
   - Object detection and manipulation
   - Color replacement rules
   - Pattern filling or flood fill
   - Symmetry operations
   - Counting and replication

2. **Formulate a NEW Hypothesis:** Inside `<hypothesis>` tags, describe a COMPLETELY NEW approach that is different from the failed code above and different from previously tried hypotheses.

3. **Plan the Implementation:** Inside `<plan>` tags, describe how you will implement this new hypothesis step by step.

4. **Write the Code:** Provide a complete NEW implementation of the `transform` function.

Remember: The current approach is fundamentally wrong. Start fresh with a different algorithm."""

            else:  # Standard repair with structured reasoning
                prompt_template = f"""You are an AI assistant specialized in repairing Python code for ARC-AGI tasks.
Your goal is to analyze input-output grid pairs and repair a previously implemented transformation function.

# STRICT OUTPUT REQUIREMENTS:
1. You MUST implement a function called `transform`.
2. The function MUST take one argument: the input grid as `list[list[int]]`.
3. The function MUST return the transformed grid as a `list[list[int]]`.
4. DO NOT return a raw numpy array or any other data type.

# Task to solve:
{item['problem_description']}

# Pattern Analysis from Examples:
{chr(10).join('- ' + hint for hint in structural_hints)}

# Inferred Puzzle Properties:
```json
{puzzle_properties_str}
```

# Training Examples:
{training_examples_str}

# Current Code to Repair (Champion):
```python
{champion_code}
```

# Current Accuracy: {champion_accuracy:.1%}

# Evaluation Feedback:
{final_feedback_payload}

# Previously tried hypotheses (AVOID REPEATING THESE):
{chr(10).join('- ' + h for h in previous_hypotheses[-3:])}

Your task is to act as an expert code debugger. Follow these steps:

1. **Analyze the Feedback:** Inside `<analysis>` tags, summarize the key failures. Identify patterns in what's failing.

2. **Formulate a New Hypothesis:** Inside `<hypothesis>` tags, state a DIFFERENT hypothesis from previous attempts. This MUST fix the identified issues.

3. **Plan the Code Changes:** Inside `<plan>` tags, describe specific changes you'll make.

4. **Write the Corrected Code:** Provide the complete corrected implementation.

Your new hypothesis MUST be different from the current code's logic and from previously tried approaches."""

            # Add hints based on accuracy
            if champion_accuracy < 0.25:
                prompt_template += "\n\nHINT: The transformation logic is fundamentally wrong. Consider a simpler approach."
            elif champion_accuracy < 0.5:
                prompt_template += "\n\nHINT: Check transformation order, array indexing, and boundary conditions."
            elif champion_accuracy < 0.8:
                prompt_template += "\n\nHINT: Focus on edge cases and special conditions that might be failing."
            else:
                prompt_template += "\n\nHINT: Almost there! Look for off-by-one errors or corner cases."

            # Best-of-N sampling
            if repair_strategy == "complete_rewrite":
                num_samples = 4  # Samples for complete rewrites
            elif repair_strategy == "major_restructure":
                num_samples = 3
            elif repair_strategy == "precision_fix":
                num_samples = 5  # Many samples for final push to 100%
            elif i < 3 or champion_accuracy < 0.5:
                num_samples = 3
            elif i < 8:  # Keep generating multiple samples longer
                num_samples = 2
            else:
                num_samples = 1

            if num_samples > 1:
                print(f"Generating {num_samples} solutions and selecting best...")

            best_challenger_code = None
            best_challenger_accuracy = -1.0
            best_challenger_feedback = []
            best_hypothesis = ""

            for sample_idx in range(num_samples):
                # Vary temperature for diversity
                sample_temperature = current_temperature * (1 + 0.1 * (sample_idx - num_samples/2))

                # Generate response
                original_temp = llm.temperature if hasattr(llm, 'temperature') else args.temperature
                if hasattr(llm, 'temperature'):
                    llm.temperature = sample_temperature

                raw_response = llm.generate([prompt_template])

                if hasattr(llm, 'temperature'):
                    llm.temperature = original_temp

                # Extract hypothesis for tracking
                response_text = raw_response[0][0]

                # Improved hypothesis extraction using regex
                hypothesis_match = re.search(r'<hypothesis>(.*?)</hypothesis>', response_text, re.DOTALL)
                if hypothesis_match:
                    hypothesis = hypothesis_match.group(1).strip()
                    # Truncate long hypotheses for logging
                    if len(hypothesis) > 200:
                        hypothesis = hypothesis[:197] + "..."
                else:
                    # Fallback: try to extract any statement about the approach
                    analysis_match = re.search(r'<analysis>(.*?)</analysis>', response_text, re.DOTALL)
                    if analysis_match:
                        hypothesis = f"Analysis-based: {analysis_match.group(1).strip()[:100]}..."
                    else:
                        hypothesis = f"Strategy: {repair_strategy}, Sample {sample_idx+1}"

                # Parse code
                sample_code = parse_code_from_response(response_text)

                if sample_code:
                    sample_accuracy, sample_feedback = get_feedback_for_challenger_dynamic(
                        sample_code, task_data['train'], puzzle_properties)
                    if sample_accuracy is None:
                        sample_accuracy = 0.0

                    print(f"  Sample {sample_idx+1}/{num_samples}: {sample_accuracy:.4f}")
                    if num_samples <= 3:  # Only show hypotheses for small sample counts to avoid clutter
                        print(f"    Hypothesis: {hypothesis[:100]}...")

                    if sample_accuracy > best_challenger_accuracy:
                        best_challenger_accuracy = sample_accuracy
                        best_challenger_code = sample_code
                        best_challenger_feedback = sample_feedback
                        best_hypothesis = hypothesis

            # Update champion if improved
            is_better = best_challenger_accuracy > champion_accuracy

            iteration_end_time = time.time()
            iteration_duration = iteration_end_time - iteration_start_time
            total_refinement_time += iteration_duration

            # Log iteration
            iteration_log_entry = {
                "iteration": i + 1,
                "loop": 1,
                "type": "challenger",
                "code": best_challenger_code,
                "pixel_accuracy": best_challenger_accuracy,
                "feedback": best_challenger_feedback,
                "strategy": repair_strategy,
                "hypothesis": best_hypothesis,
                "iteration_duration_seconds": iteration_duration
            }

            if is_better:
                print(f"IMPROVEMENT! {champion_accuracy:.4f} -> {best_challenger_accuracy:.4f}")
                print(f"  Winning hypothesis: {best_hypothesis[:150]}...")
                previous_champion_accuracy = champion_accuracy
                previous_champion_code = champion_code
                champion_code = best_challenger_code
                champion_accuracy = best_challenger_accuracy
                champion_feedback = best_challenger_feedback
                iteration_log_entry["status"] = "became_new_champion"
                iteration_log_entry["time_to_new_champion_seconds"] = total_refinement_time
                iteration_log_entry["improvement"] = best_challenger_accuracy - previous_champion_accuracy
                no_improvement_count = 0

                # Track successful hypothesis
                if best_hypothesis and not best_hypothesis.startswith("Strategy:"):
                    previous_hypotheses.append(best_hypothesis[:100])  # Keep first 100 chars
            else:
                print(f"No improvement (best: {best_challenger_accuracy:.4f})")
                if best_hypothesis and not best_hypothesis.startswith("Strategy:"):
                    print(f"  Failed hypothesis: {best_hypothesis[:150]}...")
                iteration_log_entry["status"] = "discarded_worse_than_champion"
                no_improvement_count += 1
                if i == 0:
                    previous_champion_accuracy = champion_accuracy
                    previous_champion_code = champion_code

            log_file.write(json.dumps(iteration_log_entry, cls=NumpyEncoder) + '\n')

            # Write debug info
            with open(debug_file_path, 'a') as debug_file:
                debug_file.write(f"Iteration {i+1}: Strategy={repair_strategy}, ")
                debug_file.write(f"Temp={current_temperature:.2f}, ")
                debug_file.write(f"Samples={num_samples}, ")
                debug_file.write(f"Best accuracy={best_challenger_accuracy:.4f}, ")
                debug_file.write(f"Improved={is_better}\n")

            # Check for perfect score
            if champion_accuracy >= 1.0:
                print(f"Perfect score on training set!")
                if task_data.get('test'):
                    test_accuracy, test_feedback = get_feedback_for_challenger_dynamic(
                        champion_code, task_data['test'], puzzle_properties)
                    if test_accuracy == 1.0:
                        print(f"SUCCESS: Task {task_id} solved on test set!")
                        log_entry = {
                            "iteration": i + 1,
                            "status": "solved_on_test",
                            "test_accuracy": test_accuracy
                        }
                        log_file.write(json.dumps(log_entry, cls=NumpyEncoder) + '\n')
                        break
                    else:
                        print(f"Overfitting: Test accuracy {test_accuracy:.4f}")
                        champion_feedback = ["Overfitting detected: Perfect on training but fails on test"]
                        champion_accuracy = 0.99  # Continue refinement
                else:
                    break

        # Save final code
        final_save_path = os.path.join(args.path_save_res, f"{task_id}_final.py")
        with open(final_save_path, 'w') as f_out:
            f_out.write(champion_code)
        print(f"Saved final code for task {task_id} to {final_save_path}")
        print(f"Final accuracy: {champion_accuracy:.4f}")

        # Print hypothesis summary for analysis
        if previous_hypotheses:
            print(f"\n--- Hypothesis Summary for {task_id} ---")
            print(f"Total hypotheses tried: {len(previous_hypotheses)}")
            print("Successful hypotheses that led to improvements:")
            for idx, hyp in enumerate(previous_hypotheses[-5:], 1):  # Show last 5
                print(f"  {idx}. {hyp}")

        # Write summary to debug file
        with open(debug_file_path, 'a') as debug_file:
            debug_file.write(f"\n--- Final Summary ---\n")
            debug_file.write(f"Final accuracy: {champion_accuracy:.4f}\n")
            debug_file.write(f"Total iterations: {i+1}\n")
            debug_file.write(f"Total time: {total_refinement_time:.2f}s\n")
            debug_file.write(f"Hypotheses generated: {len(previous_hypotheses)}\n")

print("\n--- TCP Refinement Complete ---")
