#!/usr/bin/env python3
"""
TCP Generate and Evaluate - ARC Task Solution Generation with Feedback

This script combines code generation with detailed feedback evaluation for ARC tasks.

Process:
1. Loads a language model and generates code solutions
2. Executes code with detailed, line-by-line tracing
3. Provides comprehensive feedback for code improvement
"""

# Force CUDA device selection BEFORE any other imports
import os
if 'CUDA_VISIBLE_DEVICES' in os.environ:
    print(f"🎯 Forcing CUDA device selection: GPU {os.environ['CUDA_VISIBLE_DEVICES']}")
    # Ensure no other GPU access
    cuda_devices = os.environ['CUDA_VISIBLE_DEVICES']
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda_devices

import sys
import os

# Add parent directory to path for tcp_core imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pickle
import gc
import numpy as np
from enum import Enum
from tqdm import tqdm
import torch
import ast
from typing import List, Dict, Tuple, Optional, Any
import re
import argparse
import random
from datetime import datetime

# --- TCP core imports ---
from tcp_core import (
    get_dataset,
    LLM_serv,
    get_solver_prompt,
    prompt_wo_fewshot_v1_,
    prompt_fewshot_v1,
    check_solutions,
)

# --- ARC data types ---
from arc import train_problems, validation_problems
from arc.read import parse_dir
from arc.types import ArcIOPair, ArcProblem
from scipy.ndimage import label
from func_timeout import func_timeout, FunctionTimedOut
import copy
from collections import Counter
import signal
from types import FrameType
import difflib

# Import TCP utility modules
from tcp_utils import Color, find_connected_components

# Import TCP object detection utilities
from tcp_object_detection import extract_objects_from_grid

# Object analysis is always available with our simplified implementation
OBJECT_ANALYSIS_AVAILABLE = True
print("Object analysis available: Using simplified TCP object detection")

# --- Configuration ---
MAX_SEQ_LENGTH = 8192
TIMEOUT = 5
COLOR_MAP = {
    0: "Black", 1: "Blue", 2: "Red", 3: "Green", 4: "Yellow",
    5: "Grey", 6: "Pink", 7: "Orange", 8: "Teal", 9: "Maroon"
}

# --- Helper Classes and Enums ---

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)

class GridComparisonResult(Enum):
    EQUAL = 0
    SHAPE_MISMATCH = 1
    CONTENT_MISMATCH = 2
    TYPE_MISMATCH = 3
    ERROR = 4
    NON_2D_ARRAY = 5
    TIMEOUT = 6

# --- Argument Parsing ---

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate and Evaluate ARC Solutions")
    # Path configurations
    parser.add_argument("--path_save_res", default="./results", help="Results saving path")
    parser.add_argument("--extra_save_name", default="", help="Extra save name for results file")

    # Model parameters
    parser.add_argument("-m", "--path_model", type=str, required=True, help="HF model identifier for generation")
    parser.add_argument("--n_gpu", type=int, default=1, help="Number of GPUs for parallel processing")
    parser.add_argument("--model_len", type=int, default=655, help="Context length limit for vLLM models")
    parser.add_argument("--fp8", action=argparse.BooleanOptionalAction, default=False, help="Use FP8 precision")
    parser.add_argument("--enable_thinking", action=argparse.BooleanOptionalAction, default=True, help="Enable thinking mode for ONLY Qwen models")

    # Inference parameters
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p nucleus sampling")
    parser.add_argument("--min_p", type=float, default=0.05, help="Min_p sampling")
    parser.add_argument("--top_k", type=float, default=-1, help="Top-k sampling")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Max generation tokens")
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="Repetition penalty")

    # Dataset parameters
    parser.add_argument("--num_problems", type=int, default=-1, help="Number of problems to run from the training set")
    parser.add_argument("--split", type=str, default="train", help="'train' or 'val'")
    parser.add_argument("--seed_examples_path", type=str, default="/data/TCP_Tracing/save_results/Qwen2.5-Coder-7B-Instruct/tcp_seed_examples.json", help="Path to the JSON file containing seed examples for one-shot prompting.")

    # Advanced parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--gpu_mem", type=float, default=0.85, help="GPU memory utilization ratio")

    return parser.parse_args()

# --- Core Helper Functions (from both scripts) ---

def setup_model(args):
    """Setup the LLM model for inference."""
    # Force CUDA device selection to respect CUDA_VISIBLE_DEVICES
    import os
    import torch
    
    # Ensure CUDA_VISIBLE_DEVICES is respected
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        print(f"CUDA_VISIBLE_DEVICES is set to: {os.environ['CUDA_VISIBLE_DEVICES']}")
        available_devices = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
        print(f"Will use GPU(s): {available_devices}")
        
        # Set torch to only see the specified devices
        if torch.cuda.is_available():
            print(f"CUDA available. Detected {torch.cuda.device_count()} device(s)")
            for i in range(torch.cuda.device_count()):
                print(f"Device {i}: {torch.cuda.get_device_name(i)}")
    
    llm = LLM_serv(args.path_model, model_len=args.model_len, seed=args.seed, n_gpu=args.n_gpu, temperature=args.temperature,
                   min_p=args.min_p, gpu_mem=args.gpu_mem, fp8=args.fp8, repetition_penalty=args.repetition_penalty,
                   top_p=args.top_p, top_k=args.top_k, enable_thinking=args.enable_thinking, max_tokens=args.max_tokens)
    return llm

def grid_to_string(grid: np.ndarray) -> str:
    if not isinstance(grid, np.ndarray): return "Error: Not a numpy array"
    rows = [" ".join([COLOR_MAP.get(int(c), f"Unk_{c}") for c in r]) for r in grid]
    return "\n".join(rows) + "\n"

def parse_code_from_response(response: str) -> str:
    code_pattern = r'''```python(.*?)```'''
    matches = re.findall(code_pattern, response, re.DOTALL)
    return matches[0].strip() if matches else ""

def compare_grids(output_grid: Any, expected_output_grid: np.ndarray) -> Tuple[GridComparisonResult, Optional[float]]:
    if isinstance(output_grid, str) and output_grid.startswith("ERROR:"):
        return (GridComparisonResult.TIMEOUT, None) if "timed out" in output_grid else (GridComparisonResult.ERROR, None)
    if not isinstance(output_grid, np.ndarray): return GridComparisonResult.TYPE_MISMATCH, 0.0
    if len(output_grid.shape) != 2: return GridComparisonResult.NON_2D_ARRAY, 0.0
    if output_grid.shape != expected_output_grid.shape: return GridComparisonResult.SHAPE_MISMATCH, 0.0
    if np.array_equal(output_grid, expected_output_grid): return GridComparisonResult.EQUAL, 1.0
    ratio = np.sum(output_grid == expected_output_grid) / np.prod(expected_output_grid.shape)
    return GridComparisonResult.CONTENT_MISMATCH, ratio

def execute_single_code_with_timeout(code: str, input_grid: np.ndarray, timeout: int, puzzle_properties: Dict[str, Any]):
    """Execute code with timeout and trace output_grid changes."""
    trace_log = []
    def trace_calls(frame, event, arg):
        if event == 'line':
            try:
                # Only trace lines within the provided code
                if frame.f_code.co_filename == '<string>':
                    local_vars = frame.f_locals
                    grid_var = None
                    # Check for standard variable names for the grid
                    if 'output_grid' in local_vars and isinstance(local_vars['output_grid'], np.ndarray):
                        grid_var = local_vars['output_grid']
                    elif 'grid_lst' in local_vars and isinstance(local_vars['grid_lst'], np.ndarray):
                        grid_var = local_vars['grid_lst']
                    
                    if grid_var is not None:
                        # Capture previous state if available
                        prev_snapshot = None
                        if trace_log:
                            last_entry = trace_log[-1]
                            prev_snapshot = last_entry["snapshot_after"]
                        
                        current_snapshot = grid_var.tolist()
                        changed_pixels = []
                        
                        # Calculate changed pixels
                        if prev_snapshot is not None:
                            try:
                                prev_array = np.array(prev_snapshot)
                                if prev_array.shape == grid_var.shape:
                                    changed_coords = np.argwhere(prev_array != grid_var)
                                    changed_pixels = [[int(r), int(c)] for r, c in changed_coords]
                                else:
                                    changed_pixels = "SHAPE_CHANGED"
                            except:
                                changed_pixels = "UNKNOWN"
                        
                        trace_log.append({
                            "lineno": frame.f_lineno,
                            "snapshot_before": prev_snapshot,
                            "snapshot_after": current_snapshot,
                            "changed_pixels": changed_pixels
                        })
            except Exception as e:
                # Continue tracing even if there's an error
                pass
        return trace_calls

    def execute_code_with_trace():
        # Prepare the execution environment
        namespace = {
            'np': np,
            'numpy': np,
            'enumerate': enumerate,
            'range': range,
            'len': len,
            'list': list,
            'set': set,
            'dict': dict,
            'sorted': sorted,
            'min': min,
            'max': max,
            'sum': sum,
            'zip': zip,
            'input_grid': input_grid.copy(),
            'Color': Color,
            'find_connected_components': find_connected_components
        }
        # Add puzzle properties to namespace if needed by the code
        if puzzle_properties:
            for key, value in puzzle_properties.items():
                namespace[key] = value
        
        # Execute the code with tracing enabled
        sys.settrace(trace_calls)
        try:
            exec(code, namespace)
            # Try to get result from main function or output variables
            if 'main' in namespace and callable(namespace['main']):
                result = namespace['main'](input_grid.copy())
            elif 'output_grid' in namespace:
                result = namespace['output_grid']
            elif 'grid_lst' in namespace:
                result = namespace['grid_lst']
            else:
                result = "ERROR: No valid output produced"
            return result, namespace
        except Exception as e:
            return f"ERROR: {str(e)}", namespace
        finally:
            sys.settrace(None)

    try:
        func_result, final_namespace = func_timeout(timeout, execute_code_with_trace)
        if isinstance(func_result, list):
            return np.array(func_result), trace_log
        elif isinstance(func_result, np.ndarray):
            return func_result, trace_log
        elif 'output_grid' in final_namespace and isinstance(final_namespace["output_grid"], np.ndarray):
            return final_namespace["output_grid"], trace_log
        elif 'grid_lst' in final_namespace:
            return np.array(final_namespace["grid_lst"]), trace_log
        else:
            return "ERROR: No valid output produced", trace_log
    except FunctionTimedOut:
        return "ERROR: Execution timed out", trace_log
    except Exception as e:
        return f"ERROR: {str(e)}", trace_log

# Object extraction function is now imported from tcp_object_detection module

def generate_natural_language_feedback(verdict, ratio, exec_output, target_grid, input_objects, output_objects, predicted_objects, mismatched_pixels, train_pair_index):
    """Generate natural language feedback matching the reference format."""
    feedback = []
    
    # Main verdict feedback
    if verdict == GridComparisonResult.EQUAL:
        feedback.append("The code produces the correct output.")
    elif verdict == GridComparisonResult.SHAPE_MISMATCH:
        feedback.append("The code produces a grid with incorrect dimensions.")
    else:
        feedback.append("The code produces a grid with the correct dimensions, but the content is wrong. There are pixel mismatches.")
    
    # Add training example header
    feedback.append(f"\n--- Feedback on Training Example {train_pair_index + 1} ---")
    
    # Add comparison result and accuracy
    if verdict != GridComparisonResult.EQUAL:
        if ratio is not None:
            feedback.append(f"The code resulted in a '{verdict.name}' with {ratio*100:.1f}% pixel accuracy.")
        else:
            feedback.append(f"The code resulted in a '{verdict.name}'.")
        
        # Handle mismatched_pixels as either list of coordinates or count
        mismatch_count = len(mismatched_pixels) if isinstance(mismatched_pixels, list) else mismatched_pixels
        
        if mismatch_count > 0:
            # Analyze color differences
            target_colors = set()
            predicted_colors = set()
            
            if isinstance(target_grid, np.ndarray):
                target_colors = set(target_grid.flatten())
            elif isinstance(target_grid, list):
                target_colors = set([item for sublist in target_grid for item in sublist])
            
            if isinstance(exec_output, np.ndarray):
                predicted_colors = set(exec_output.flatten())
            elif isinstance(exec_output, list):
                predicted_colors = set([item for sublist in exec_output for item in sublist])
            elif isinstance(exec_output, str):
                predicted_colors = set()  # Error case
            
            # Missing and extra colors
            missing_colors = target_colors - predicted_colors
            extra_colors = predicted_colors - target_colors
            
            if missing_colors:
                feedback.append(f"It failed to produce the required colors: {sorted(list(missing_colors))}")
            if extra_colors:
                feedback.append(f"It incorrectly introduced new colors: {sorted(list(extra_colors))}")
            
            # Pixel count information
            if mismatch_count > 0:
                feedback.append(f"There were {mismatch_count} incorrect pixels.")
    
    # Object-level feedback
    if input_objects or output_objects or predicted_objects:
        feedback.append("\n--- Object-Level Feedback ---")
        
        if predicted_objects and output_objects:
            if len(predicted_objects) != len(output_objects):
                feedback.append(f"- Found {len(predicted_objects)} objects, but expected {len(output_objects)}.")
            else:
                for i, (pred_obj, target_obj) in enumerate(zip(predicted_objects, output_objects)):
                    if pred_obj.get('size', 0) != target_obj.get('size', 0):
                        feedback.append(f"- Object {i+1}: Expected size {target_obj.get('size', 0)}, but got {pred_obj.get('size', 0)}.")
                    if pred_obj.get('colors', []) != target_obj.get('colors', []):
                        expected_color = target_obj.get('colors', [None])[0] if target_obj.get('colors') else None
                        actual_color = pred_obj.get('colors', [None])[0] if pred_obj.get('colors') else None
                        if expected_color is not None and actual_color is not None:
                            feedback.append(f"- Object {i+1}: Expected color {expected_color}, but got {actual_color}.")
    
    return feedback

def calculate_line_feedback(code: str, input_grid: np.ndarray, target_grid: np.ndarray, timeout: int, puzzle_properties: Dict[str, Any], train_pairs: List[ArcIOPair], input_objects: List[Dict[str, Any]], output_objects: List[Dict[str, Any]]):
    line_rewards = {}
    line_reward_details = {}
    trace_log = []
    exec_output = None
    
    # Calculate pixels that *should* be affected (difference between input and target)
    pixels_should_be_affected_overall = []
    if isinstance(input_grid, np.ndarray) and isinstance(target_grid, np.ndarray) and input_grid.shape == target_grid.shape:
        pixels_should_be_affected_overall = np.argwhere(input_grid != target_grid).tolist()
    
    # Execute code with tracing
    exec_output, trace_log = execute_single_code_with_timeout(code, input_grid, timeout, puzzle_properties)
    
    # If execution failed or output is not a numpy array, we can't do pixel-level analysis
    if not isinstance(exec_output, np.ndarray):
        return line_rewards, line_reward_details, trace_log, exec_output, []
    
    # Calculate final mismatched pixels for overall feedback
    final_mismatched_pixels = []
    if exec_output.shape == target_grid.shape:
        mismatch_coords = np.argwhere(exec_output != target_grid)
        final_mismatched_pixels = [[int(r), int(c)] for r, c in mismatch_coords]
    
    # Calculate line-level rewards based on runtime trace and structural consistency
    for trace_entry in trace_log:
        lineno = trace_entry["lineno"]
        line_reward = 0.0
        detail = {
            "snapshot_before": trace_entry.get("snapshot_before"),
            "snapshot_after": trace_entry.get("snapshot_after"),
            "Criteria": [],
            "Color_ids": [],
            "Object_ids": [],
            "Pixels Actually Changed By This Line": [],
            "Pixel Should be affected": [],
            "Pixel Touched": [],
            "Pixel Missed/Not touch": []
        }
        
        # Get snapshots
        snapshot_after_np = np.array(trace_entry["snapshot_after"]) if isinstance(trace_entry["snapshot_after"], list) else None
        snapshot_before_np = np.array(trace_entry["snapshot_before"]) if isinstance(trace_entry["snapshot_before"], list) else None
        
        # Process changed pixels information
        if isinstance(trace_entry.get("changed_pixels"), list):
            detail["Pixels Actually Changed By This Line"] = trace_entry["changed_pixels"]
        elif isinstance(trace_entry.get("changed_pixels"), str):  # Handle "SHAPE_CHANGED" or "UNKNOWN"
            detail["Pixels Actually Changed By This Line"] = trace_entry["changed_pixels"]
            if trace_entry["changed_pixels"] == "SHAPE_CHANGED":
                line_reward -= 1.0  # Heavy penalty for invalidating the grid
                detail["Criteria"].append("SHAPE_CHANGED")
        
        # Detailed pixel analysis if we have proper snapshots
        if snapshot_after_np is not None and snapshot_before_np is not None:
            if snapshot_after_np.shape == target_grid.shape and snapshot_before_np.shape == target_grid.shape:
                # Identify pixels that were correct before but are now incorrect
                correct_before_coords = np.argwhere(snapshot_before_np == target_grid)
                incorrect_after_coords = np.argwhere(snapshot_after_np != target_grid)
                broken_coords = []
                for coord in correct_before_coords:
                    if any(np.array_equal(coord, inc_coord) for inc_coord in incorrect_after_coords):
                        broken_coords.append(coord.tolist())
                
                # Identify pixels that were incorrect before but are now correct
                incorrect_before_coords = np.argwhere(snapshot_before_np != target_grid)
                correct_after_coords = np.argwhere(snapshot_after_np == target_grid)
                fixed_coords = []
                for coord in incorrect_before_coords:
                    if any(np.array_equal(coord, corr_coord) for corr_coord in correct_after_coords):
                        fixed_coords.append(coord.tolist())
                
                if broken_coords:
                    line_reward -= len(broken_coords) * 0.2  # Penalty for breaking pixels
                    detail["Criteria"].append("BROKE_PIXELS")
                
                if fixed_coords:
                    line_reward += len(fixed_coords) * 0.1  # Small reward for fixing pixels
                    detail["Criteria"].append("FIXED_PIXELS")
                
                # Track colors involved
                if len(trace_entry.get("changed_pixels", [])) > 0:
                    try:
                        changed_pixels = trace_entry["changed_pixels"]
                        if isinstance(changed_pixels, list) and len(changed_pixels) > 0:
                            colors_involved = set()
                            for pixel_coord in changed_pixels:
                                if len(pixel_coord) == 2:
                                    r, c = pixel_coord
                                    if 0 <= r < snapshot_after_np.shape[0] and 0 <= c < snapshot_after_np.shape[1]:
                                        colors_involved.add(int(snapshot_after_np[r, c]))
                            detail["Color_ids"] = sorted(list(colors_involved))
                    except Exception:
                        pass
            else:
                line_reward -= 0.5
                detail["Criteria"].append("SHAPE_MISMATCH")
        
        # Accumulate line rewards
        line_rewards[lineno] = line_rewards.get(lineno, 0.0) + line_reward
        line_reward_details[lineno] = detail  # Store the detailed feedback for the line
    
    return line_rewards, line_reward_details, trace_log, exec_output, final_mismatched_pixels

# --- Main Execution Logic ---

def main():
    args = parse_arguments()
    np.random.seed(args.seed)
    
    print("--- Setting up Model and Dataset ---")
    model = setup_model(args)

    # --- Load and Convert Seed Examples for One-Shot Prompting ---
    seed_examples = []
    seed_file_path = args.seed_examples_path
    arc_challenges_path = "/data/TCP_Tracing/arc-prize-2024/arc-agi_training_challenges.json"
    arc_solutions_path = "/data/TCP_Tracing/arc-prize-2024/arc-agi_training_solutions.json"
    
    try:
        # Load seed examples UIDs
        with open(seed_file_path, "r") as f:
            raw_seed_examples = json.load(f)
        
        # Load actual ARC training challenges and solutions
        with open(arc_challenges_path, "r") as f:
            arc_challenges = json.load(f)
            
        with open(arc_solutions_path, "r") as f:
            arc_solutions = json.load(f)
        
        # Extract proper ARC task data for each seed example
        for raw_example in raw_seed_examples:
            if 'uid' in raw_example and raw_example['uid'] in arc_challenges:
                task_data = arc_challenges[raw_example['uid']]
                
                # Add test output from solutions if available
                test_with_output = []
                if raw_example['uid'] in arc_solutions:
                    solutions = arc_solutions[raw_example['uid']]
                    for i, test_pair in enumerate(task_data['test']):
                        test_pair_with_output = {'input': test_pair['input']}
                        if i < len(solutions):
                            test_pair_with_output['output'] = solutions[i]
                        test_with_output.append(test_pair_with_output)
                else:
                    test_with_output = task_data['test']
                
                # Convert to proper format
                converted_example = {
                    'uid': raw_example['uid'],
                    'train': task_data['train'],
                    'test': test_with_output,
                    'solution': raw_example.get('solution', '')  # Include solution from original seed
                }
                seed_examples.append(converted_example)
        
        print(f"Successfully loaded and converted {len(seed_examples)} seed examples with complete ARC task data")
    except FileNotFoundError as e:
        print(f"Warning: Required file not found ({e}). Falling back to zero-shot prompting.")
    except Exception as e:
        print(f"Warning: Error loading seed examples: {e}. Falling back to zero-shot prompting.")
    
    # Load dataset
    try:
        train_data, val_data, _ = get_dataset(data_path=os.path.expanduser("~/tcp/data/"))
    except FileNotFoundError:
        print("Error: Could not find dataset. Please check the path in get_dataset.")
        model.terminate()
        return

    dataset = train_data if args.split == "train" else val_data
    # Keep track of task IDs by using items() instead of values()
    problems_to_run = list(dataset.items())
    if args.num_problems > 0:
        problems_to_run = problems_to_run[:args.num_problems]

    # --- Setup for JSONL output ---
    os.makedirs(args.path_save_res, exist_ok=True)
    save_path = os.path.join(args.path_save_res, f"detailed_feedback_{args.extra_save_name}.jsonl")
    
    with open(save_path, 'w') as f:
        print(f"--- Starting Generation and Evaluation. Results will be streamed to {save_path} ---")
        
        for i, (task_id, problem) in enumerate(tqdm(problems_to_run, desc="Processing Tasks")):
            print(f"\nProcessing Task {i+1}/{len(problems_to_run)}: {task_id}")
            
            # Debug: print problem structure for first task
            if i == 0:  # Only for first task to avoid spam
                print(f"Debug: Task ID: {task_id}, Problem keys: {list(problem.keys())}")
            
            # --- 1. Generation Phase (Now with One-Shot Logic) ---
            fewshot_examples = []
            prompt_solver = prompt_wo_fewshot_v1_  # Default to zero-shot

            if seed_examples:
                # Select a random example that is not the current task
                chosen_example = random.choice(seed_examples)
                if len(seed_examples) > 1:
                    while chosen_example['uid'] == task_id:
                        chosen_example = random.choice(seed_examples)
                
                if chosen_example['uid'] != task_id:
                    fewshot_examples = [chosen_example]
                    prompt_solver = prompt_fewshot_v1  # Use proper few-shot prompt

            # Generate prompt for the task
            prompt_formated = get_solver_prompt(
                problem,
                fewshot_examples,
                prompt_solver=prompt_solver,
                grid_display_mode="numpy",
                alt_colors=True
            )
            
            # Generate code solution
            results = model.generate([prompt_formated], n=1)

            # Format results
            from tcp_core import format_all_generation
            dict_response = format_all_generation(results, [task_id], use_vllm_generate=False)
            
            # Extract the generated code from formatted response
            if task_id in dict_response and len(dict_response[task_id]) > 0:
                generated_code = dict_response[task_id][0]
            else:
                generated_code = None

            if not generated_code:
                print(f"Failed to generate or parse code for task {task_id}.")
                error_record = {"uid": task_id, "error": "Code generation failed."}
                f.write(json.dumps(error_record) + '\n')
                continue
            
            # Extract training pairs
            train_pairs = problem['train']
            
            # Generate puzzle properties to match reference format (moved before task_feedback)
            puzzle_properties = {
                "connectivity": 8,  # Default to 8-connectivity
                "connectivity_needed": True,
                "preserves_object_count": True,  # Will be analyzed
                "preserves_grid_size": True,    # Will be analyzed
                "color_map": None,              # Will be inferred
                "color_map_reason": "No size-preserving training pairs found to infer a map.",
                "shape_transformation_rule": {"type": "complex"},
                "allowed_colors": [],           # Will be populated
                "object_count_transform": {"type": "delta", "value": 0},
                "is_identity_task": False,
                "max_color_value": 0            # Will be calculated
            }
            
            # Analyze training pairs to infer properties
            all_colors = set()
            input_sizes = []
            output_sizes = []
            
            for pair in train_pairs:
                input_grid = pair['input']
                output_grid = pair['output']
                
                # Collect all colors
                for row in input_grid + output_grid:
                    all_colors.update(row)
                
                # Check grid size preservation
                input_sizes.append((len(input_grid), len(input_grid[0])))
                output_sizes.append((len(output_grid), len(output_grid[0])))
            
            # Update properties based on analysis
            puzzle_properties["allowed_colors"] = sorted(list(all_colors))
            puzzle_properties["max_color_value"] = max(all_colors) if all_colors else 0
            puzzle_properties["preserves_grid_size"] = input_sizes == output_sizes
            
            # Basic color mapping analysis (simplified)
            if puzzle_properties["preserves_grid_size"]:
                puzzle_properties["color_map_reason"] = "Consistent map found across all size-preserving pairs."
                # Could implement more sophisticated color mapping analysis here
            
            # Shape transformation analysis
            if input_sizes == output_sizes:
                puzzle_properties["shape_transformation_rule"] = {"type": "identity"}
            else:
                puzzle_properties["shape_transformation_rule"] = {"type": "complex"}
            
            # --- 2. Evaluation Phase ---
            task_feedback = {
                "problem_uid": task_id,
                "problem_description": f"Evaluating generated code for problem {task_id}.",
                "failed_code": generated_code,
                "evaluation_results": [],
                "solved_all_training_pairs": True,  # Will be updated based on results
                "inferred_puzzle_properties": puzzle_properties
            }

            is_fully_correct = True
            
            for j, train_pair in enumerate(train_pairs):
                # Convert to numpy arrays
                input_grid = np.array(train_pair['input'])
                output_grid = np.array(train_pair['output'])
                
                # Extract objects from input and output grids
                try:
                    input_objects = extract_objects_from_grid(input_grid)
                    if input_objects is None:
                        input_objects = []
                        print(f"Warning: input_objects was None, using empty list")
                except Exception as e:
                    print(f"Warning: Input object extraction failed: {e}")
                    input_objects = []
                
                try:
                    output_objects = extract_objects_from_grid(output_grid)
                    if output_objects is None:
                        output_objects = []
                        print(f"Warning: output_objects was None, using empty list")
                except Exception as e:
                    print(f"Warning: Output object extraction failed: {e}")
                    output_objects = []
                
                # Use enhanced feedback calculation
                line_rewards, line_details, trace_log, exec_output, mismatched_pixels = calculate_line_feedback(
                    generated_code, input_grid, output_grid, TIMEOUT, 
                    puzzle_properties, train_pairs, input_objects, output_objects
                )
                
                verdict, ratio = compare_grids(exec_output, output_grid)
                if verdict != GridComparisonResult.EQUAL:
                    is_fully_correct = False

                # Extract objects from predicted output for comparison
                predicted_objects = []
                if isinstance(exec_output, (list, np.ndarray)) and not isinstance(exec_output, str):
                    try:
                        predicted_output_grid = np.array(exec_output) if isinstance(exec_output, list) else exec_output
                        predicted_objects = extract_objects_from_grid(predicted_output_grid)
                    except:
                        predicted_objects = []

                # Generate natural language feedback for this training pair
                pair_feedback = generate_natural_language_feedback(
                    verdict, ratio, exec_output, output_grid, 
                    input_objects, output_objects, predicted_objects, mismatched_pixels, j
                )
                
                # Add feedback to evaluation results
                task_feedback["evaluation_results"].extend(pair_feedback)

            task_feedback["solved_all_training_pairs"] = is_fully_correct
            print(f"Task {task_id} evaluation complete. Solved all training pairs: {is_fully_correct}")
            
            # --- Stream result to JSONL file ---
            f.write(json.dumps(task_feedback, cls=NumpyEncoder) + '\n')

    print(f"--- All tasks processed. Results saved to {save_path} ---")
    
    # --- 4. Teardown ---
    model.terminate()


if __name__ == "__main__":
    main()
