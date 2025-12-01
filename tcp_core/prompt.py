"""
TCP Prompt Templates - Prompt utilities for ARC task solving
"""

import numpy as np
import copy

# Color scheme mappings
alt_color_scheme_name = {
    0: "black",
    1: "blue",
    2: "red",
    3: "green",
    4: "yellow",
    5: "grey",
    6: "pink",
    7: "orange",
    8: "purple",
    9: "brown",
}

color_scheme_name = {
    0: "black",
    1: "blue",
    2: "purple",
    3: "green",
    4: "yellow",
    5: "grey",
    6: "fuchsia",
    7: "orange",
    8: "teal",
    9: "brown",
}

# Spreadsheet column labels
spreadsheet_col_labels = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z", "AA", "AB", "AC", "AD",
]


def grid_formatting(grid, mode="ascii"):
    """
    Take a grid and return a string representation.

    Args:
        grid: 2D list or numpy array
        mode: Format mode - "ascii", "spreadsheet", "numpy", or "colors"

    Returns:
        String representation of the grid
    """
    grid = np.array(grid)

    if mode == "ascii":
        return "\n".join("|".join(str(x) for x in row) for row in grid)
    elif mode == "spreadsheet":
        rows, cols = grid.shape
        assert cols <= 30
        assert rows <= 30
        cols_header_line = "|".join([" "] + spreadsheet_col_labels[:cols])
        rest = "\n".join(
            "|".join([str(i + 1)] + [str(x) for x in row])
            for i, row in enumerate(grid)
        )
        return f"{cols_header_line}\n{rest}"
    elif mode == "numpy":
        return str(grid)
    elif mode == "colors":
        return "[\n" + "\n".join(
            " " * 4 + "[" + ", ".join(f'"{alt_color_scheme_name[x]}"' for x in row) + "],"
            for row in grid
        ) + "\n]"
    else:
        raise ValueError(f"Invalid mode {mode}")


def format_task(example, max_example=-1, grid_display_mode="ascii", include_test=True, show_output_test=False, randomize_pair_order=False):
    """Format a task's input-output pairs for prompting."""
    if max_example == -1:
        max_example = len(example['train'])
    else:
        assert max_example > 0, "max_example should be positive"
        max_example = min(max_example, len(example['train']))

    prompt = ""
    if randomize_pair_order:
        list_idx = np.random.permutation(max_example)
    else:
        list_idx = range(max_example)

    for i in list_idx:
        input_i = example['train'][i]["input"]
        x_shape, y_shape = np.array(input_i).shape
        prompt += f"## Input {i+1} (grid shape: {x_shape} by {y_shape}):\n" + grid_formatting(input_i, mode=grid_display_mode) + "\n\n"
        output_i = example['train'][i]["output"]
        x_shape, y_shape = np.array(output_i).shape
        prompt += f"## Output {i+1} (grid shape: {x_shape} by {y_shape}):\n" + grid_formatting(output_i, mode=grid_display_mode) + "\n\n"

    if include_test:
        max_example = min(max_example, len(example['test']))
        for i in range(max_example):
            input_i = example['test'][i]["input"]
            x_shape, y_shape = np.array(input_i).shape
            prompt += f"## Test Input {i+1} (grid shape: {x_shape} by {y_shape}):\n" + grid_formatting(input_i, mode=grid_display_mode) + "\n\n"
            if show_output_test:
                output_i = example['test'][i]["output"]
                x_shape, y_shape = np.array(output_i).shape
                prompt += f"## Test Output {i+1} (grid shape: {x_shape} by {y_shape}):\n" + grid_formatting(output_i, mode=grid_display_mode) + "\n\n"

    return prompt


def format_fewshot_tasks(examples, max_example=-1, grid_display_mode="ascii", randomize_pair_order=False, show_output_test=False):
    """Format multiple tasks for few-shot prompting."""
    prompt = ""
    for id, example in enumerate(examples):
        prompt += f"# Task {id+1}:\n"
        prompt += format_task(example, max_example=max_example, grid_display_mode=grid_display_mode, include_test=True, randomize_pair_order=randomize_pair_order, show_output_test=show_output_test)
        assert "solution" in example, "example should have a solution"
        prompt += "## Solution:\n" + example["solution"] + "\n"
    return prompt


def get_list_fewshot_example(examples_fewshot, solution):
    """Return a list of fewshot examples with solutions added."""
    fewshot_example = []
    for i, example in enumerate(examples_fewshot):
        ex = copy.deepcopy(example)
        ex["solution"] = solution[i]
        fewshot_example.append(ex)
    return fewshot_example


def get_additional_info_prompt(alt_colors=False, grid_shape="ascii", prompt_colors=True):
    """Get additional info string for prompts."""
    map_colors = """The number in the input grid can be mapped to the following colors: 0:Black; 1:Blue; 2:Purple; 3:Green; 4:Yellow; 5:Grey; 6:Fuschia; 7:Orange; 8:Teal; 9:Brown\n"""
    map_colors_alt = """The number in the input grid can be mapped to the following colors: 0:Black; 1:Blue; 2:Red; 3:Green; 4:Yellow; 5:Grey; 6:Pink; 7:Orange; 8:Purple; 9:Brown\n"""

    additional_info = ""

    colors_mapping = map_colors_alt if alt_colors else map_colors
    if prompt_colors:
        additional_info += colors_mapping

    if grid_shape in ["ascii", "spreadsheet"]:
        additional_info += "Grid are represented by elements separated by '|' and each each rows are separated by a Newline ('\\n').\n"
    if grid_shape == "spreadsheet":
        additional_info += "Locations are denoted like A7 or D3, where columns are denoted with A, B, C, etc., and rows are denoted with 1, 2, 3, etc. So, D3 corresponds to the cell in the 4th column and the 3rd row. Note that rows are 1-indexed.\n"

    return additional_info


def get_solver_prompt(task2solve, fewshot_examples, prompt_solver, grid_display_mode="numpy", alt_colors=True, prompt_colors=True, max_example=-1, randomize_pair_order=False, show_output_test=False):
    """Get a task and return the prompt for the solver."""
    additional_info = get_additional_info_prompt(alt_colors=alt_colors, grid_shape=grid_display_mode, prompt_colors=prompt_colors)
    fewshot_example = format_fewshot_tasks(fewshot_examples, max_example=max_example, grid_display_mode=grid_display_mode, randomize_pair_order=randomize_pair_order)
    task2solve_formated = "# Task to solve:\n" + format_task(task2solve, grid_display_mode=grid_display_mode, max_example=max_example, randomize_pair_order=randomize_pair_order, show_output_test=show_output_test, include_test=True)

    assert "additional_info" in prompt_solver, "prompt_solver should have {additional_info}"
    assert "fewshot_example" in prompt_solver, "prompt_solver should have {fewshot_example}"
    assert "task" in prompt_solver, "prompt_solver should have {task}"

    prompt_formated = prompt_solver.format(additional_info=additional_info, fewshot_example=fewshot_example, task=task2solve_formated)
    return prompt_formated.strip()


# Prompt templates
prompt_fewshot_v1 = """You are an AI assistant specialized in solving Abstract Reasoning Corpus (ARC-AGI) tasks by generating Python code.
Your goal is to analyze input-output grid pairs. The outputs were produced by applying a transformation rule to the inputs. Implement the transformation rules as a Python function.
You should only write the implemented the transformation in code.
You must write code in triple backticks (```python and then ```). You must write a function called `transform` which takes a single argument, the input grid as `list[list[int]]`, and returns the transformed grid (also as `list[list[int]]`).
You should make sure that you implement a version of the transformation that works in general (at least for all given input-output pairs and test input pairs).
{additional_info}
Here are examples of ARC-AGI tasks, the reasoning step to find the solution, and the Python solution (`transform`):
{fewshot_example}
Now, solve the following ARC-AGI task:

{task}"""

prompt_wo_fewshot_v1_ = """You are an AI assistant specialized in solving Abstract Reasoning Corpus (ARC-AGI) tasks by generating Python code.
Your goal is to analyze input-output grid pairs. The outputs were produced by applying a transformation rule to the inputs. Implement the transformation rules as a Python function.
You should only write the implemented the transformation in code.
You must write code in triple backticks (```python and then ```). You must write a function called `transform` which takes a single argument, the input grid as `list[list[int]]`, and returns the transformed grid (also as `list[list[int]]`).
You should make sure that you implement a version of the transformation that works in general (at least for all given input-output pairs and test input pairs).
{additional_info}{fewshot_example}
Now, solve the following ARC-AGI task:

{task}"""
