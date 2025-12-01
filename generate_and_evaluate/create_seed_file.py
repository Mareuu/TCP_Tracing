"""
Create Seed File - Pre-processing for Few-Shot Examples

This script extracts perfect solutions from solved tasks to create
seed examples for few-shot prompting in the generate_and_evaluate pipeline.
"""

import sys
import os

# Add parent directory to path for tcp_core imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import json
import argparse
import numpy as np
from tcp_core import get_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Create seed file from solved tasks")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen2.5-Coder-7B-Instruct",
        help="Model name (e.g., Qwen2.5-Coder-7B-Instruct)"
    )
    parser.add_argument(
        "--base_save_path",
        type=str,
        default="/data/TCP_tracing/save_results",
        help="Base path for save results"
    )
    parser.add_argument(
        "--pickle_file",
        type=str,
        default=None,
        help="Override pickle file path (optional)"
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Override output JSON path (optional)"
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=5,
        help="Maximum number of seed examples to find"
    )
    return parser.parse_args()


args = parse_args()

# Define file paths - take from model name or overridden
pickle_file_path = args.pickle_file or f"{args.base_save_path}/{args.model}/gen-0/solved_gen_0_with_train.pkl"
output_json_path = args.output_json or f"{args.base_save_path}/{args.model}/tcp_seed_examples.json"

# --- Helper function to reformat task data into a prompt-like structure ---
def get_task_prompt(task_data):
    prompt_parts = []
    # The task_data is a dict, so use dict access.
    train_pairs = task_data.get('train', [])
    test_pairs = task_data.get('test', [])

    for i, train_pair in enumerate(train_pairs):
        prompt_parts.append(f"Example {i+1} Input:")
        prompt_parts.append(str(train_pair.get('input')))
        prompt_parts.append(f"Example {i+1} Output:")
        prompt_parts.append(str(train_pair.get('output')))
        prompt_parts.append("")
    
    if test_pairs:
        prompt_parts.append("Test Input:")
        prompt_parts.append(str(test_pairs[0].get('input')))
        prompt_parts.append("Test Output:")
    return "\n".join(prompt_parts)


# --- Main script logic ---
def main():
    seed_examples = []
    found_task_ids = set()
    max_examples_to_find = args.max_examples

    print("--- Step 1: Loading ARC Dataset ---")
    try:
        arc_train_data, _, _ = get_dataset(data_path="/data/TCP_Tracing")
        print(f"Loaded {len(arc_train_data)} tasks from the main dataset.")
    except Exception as e:
        print(f"Fatal Error: Could not load main ARC dataset. {e}")
        return

    print(f"--- Step 2: Loading Solved Results Pickle File ---")
    try:
        with open(pickle_file_path, "rb") as f:
            solved_data = pickle.load(f)
        all_solved_tasks = solved_data[0]['dict_response']
    except Exception as e:
        print(f"An error occurred while loading the pickle file: {e}")
        return

    print("--- Step 3: Searching for Perfect Solutions ---")
    for task_id, solutions in all_solved_tasks.items():
        if len(seed_examples) >= max_examples_to_find:
            break
        if task_id in found_task_ids or task_id not in arc_train_data:
            continue

        for solution in solutions:
            correct_train = solution.get('correct_train_input', [])
            correct_test = solution.get('correct_test_input', [])
            
            # FINAL CORRECTED LOGIC: Check for list OR numpy.ndarray
            is_perfect = (isinstance(correct_train, (list, np.ndarray)) and len(correct_train) > 0 and all(correct_train) and
                          isinstance(correct_test, (list, np.ndarray)) and len(correct_test) > 0 and all(correct_test))

            if is_perfect:
                code_body = solution.get('code')
                if code_body:
                    print(f"Found a perfect solution for task: {task_id}")
                    task_data = arc_train_data[task_id]
                    formatted_example = {
                        "uid": task_id,
                        "prompt": get_task_prompt(task_data),
                        "solution": code_body
                    }
                    seed_examples.append(formatted_example)
                    found_task_ids.add(task_id)
                    break

    # --- Final Saving Step ---
    if len(seed_examples) > 0:
        print(f"\n--- Step 4: Saving Seed File ---")
        print(f"Found {len(seed_examples)} perfect solutions. Saving to {output_json_path}")
        with open(output_json_path, "w") as f:
            json.dump(seed_examples, f, indent=2)
        print("Successfully created the seed example file.")
    else:
        print("\nCould not find any perfect solutions in the file.")

if __name__ == "__main__":
    main()
