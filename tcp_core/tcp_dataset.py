"""
TCP Dataset Utilities

TCP (Tracing & Correcting Program)

Dataset loading utilities for ARC tasks.
"""

import copy
import json
import os


def merge_GT(data, data_GT):
    """Merge ground truth data to data (correct output for the test example)."""
    data_merged = copy.deepcopy(data)
    for puzzle in data_GT:
        assert len(data_GT[puzzle]) == len(data_merged[puzzle]['test'])
        for i in range(len(data_GT[puzzle])):
            data_merged[puzzle]['test'][i]['output'] = data_GT[puzzle][i]
    return data_merged


def get_dataset(data_path="/data/TCP_Tracing", arc_2=False):
    """
    Get train, val, test dataset from ARC challenge files.

    Args:
        data_path: Base path to data directory
        arc_2: If True, use arc-prize-2025 folder, else arc-prize-2024

    Returns:
        tuple: (train_data, val_data, test_data) dictionaries
    """
    folder_arc_1 = "arc-prize-2024/"
    folder_arc_2 = "arc-prize-2025/"

    if arc_2:
        data_path = os.path.join(data_path, folder_arc_2)
    else:
        data_path = os.path.join(data_path, folder_arc_1)

    path_trainset = os.path.join(data_path, "arc-agi_training_challenges.json")
    path_trainset_GT = os.path.join(data_path, "arc-agi_training_solutions.json")
    path_valset = os.path.join(data_path, "arc-agi_evaluation_challenges.json")
    path_valset_GT = os.path.join(data_path, "arc-agi_evaluation_solutions.json")
    path_testset = os.path.join(data_path, "arc-agi_test_challenges.json")

    print(f"Loading dataset from: {data_path}")
    print(f"Training set: {path_trainset}")

    with open(path_trainset) as f:
        data_train = json.load(f)
    with open(path_trainset_GT) as f:
        data_train_GT = json.load(f)
    with open(path_valset) as f:
        data_val = json.load(f)
    with open(path_valset_GT) as f:
        data_val_GT = json.load(f)
    with open(path_testset) as f:
        data_test = json.load(f)

    train_data = merge_GT(data_train, data_train_GT)
    val_data = merge_GT(data_val, data_val_GT)
    test_data = data_test

    return train_data, val_data, test_data


def convert_to_list(arrays):
    """Convert an array of array of array of int64 to a list of square list of int."""
    result = []
    for array in arrays:
        converted_array = []
        for sub_array in array:
            converted_array.append(sub_array.astype(int).tolist())
        result.append(converted_array)
    return result


def reformat_solutions_parquet(df):
    """Reformat solutions from a DataFrame to a dictionary of task_id -> list of solutions."""
    dict_solutions = {}
    for task_id, group in df.groupby('task_id'):
        dict_solutions[task_id] = group.to_dict(orient='records')
    for task_id, solutions in dict_solutions.items():
        for solution in solutions:
            solution['predicted_train_output'] = convert_to_list(solution['predicted_train_output'])
            solution['predicted_test_output'] = convert_to_list(solution['predicted_test_output'])
    return dict_solutions
