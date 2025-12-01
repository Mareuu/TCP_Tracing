"""
TCP Sandbox - Safe code execution for ARC task solutions
Provides sandboxed execution of generated code with memory limits and timeouts.
"""

import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
from typing import List, Tuple
import os
from multiprocessing import Pool
import builtins
import importlib
import re

Result = Tuple[str, List[bool]]

PASS = "pass"
FAIL = "fail"
TIMEOUT = "timeout"

Timeout_sandbox = 3  # seconds
Memory_limit = 1  # GB

try:
    import resource
    RESOURCE_MODULE_AVAILABLE = True
except ImportError:
    RESOURCE_MODULE_AVAILABLE = False


def limit_memory(max_memory):
    if RESOURCE_MODULE_AVAILABLE:
        actual_limit = max_memory + (512 * 1024 * 1024)  # Add 512MB buffer
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        resource.setrlimit(resource.RLIMIT_AS, (actual_limit, hard))


def extract_transform(markdown_text):
    """Extract the transform function from the LLM generated text."""
    pattern = r'```python\n(.*?)```'

    code_blocks = re.findall(pattern, markdown_text, re.DOTALL)
    transform_function = None
    flag_def = False

    for block in code_blocks:
        if 'def transform(' in block:
            transform_function = block
            flag_def = True
        if not flag_def:
            transform_function = block

    if transform_function is not None:
        transform_function = postprocess_transform(transform_function)
    else:
        transform_function = ""
    return transform_function


def postprocess_transform(transform):
    """Postprocess the transform function."""
    # Add import numpy as np if not present and np is used
    if 'np.' in transform and 'import numpy' not in transform:
        transform = 'import numpy as np\n' + transform
    return transform


class Sandbox:
    """A sandbox for executing code with limited imports and memory usage."""

    def __init__(self, banned_modules=None, timeout=5, max_memory=1 * 1000 * 1024 * 1024):
        self.banned_modules = banned_modules or ['os', 'sys', 'subprocess', 'shutil', 'socket']
        self.timeout = timeout
        self.max_memory = max_memory
        self.globals = {'__builtins__': {}}

        try:
            import scipy
            import scipy.ndimage
            self.globals['scipy'] = scipy
            self.globals['ndimage'] = scipy.ndimage
        except ImportError:
            pass

        for name in dir(builtins):
            if name not in ['eval', 'exec', 'open']:
                self.globals['__builtins__'][name] = getattr(builtins, name)
        self.globals['__import__'] = self.safe_import

    def safe_import(self, name, *args, **kwargs):
        if name in self.banned_modules:
            raise ImportError(f"Import of '{name}' is not allowed for security reasons")
        if name == 'scipy.ndimage':
            import scipy.ndimage
            return scipy.ndimage
        elif name == 'scipy':
            import scipy
            return scipy
        return importlib.import_module(name)

    @staticmethod
    def _initialize_worker():
        """Initialize the worker process with required modules"""
        try:
            import numpy as np
            globals()['np'] = np

            import scipy
            globals()['scipy'] = scipy

            import math
            globals()['math'] = math

        except ImportError as e:
            print(f"Warning: Failed to initialize some modules: {e}")

    @staticmethod
    def _run_with_memory_limit(code, dict_io_train_test, max_memory, entry_point='transform'):
        limit_memory(max_memory)

        error_message = "Unknown error"
        try:
            exec(code, globals())
            fn = globals()[entry_point]

            for key, values in dict_io_train_test.items():
                list_input_matrix = values["inputs"]
                for id_input, input_matrix in enumerate(list_input_matrix):
                    try:
                        result = fn(input_matrix)
                        if isinstance(result, np.ndarray):
                            result = result.tolist()
                        dict_io_train_test[key]["prediction"][id_input] = result

                        try:
                            if isinstance(dict_io_train_test[key]["outputs"][id_input], np.ndarray):
                                dict_io_train_test[key]["outputs"][id_input] = dict_io_train_test[key]["outputs"][id_input].tolist()
                            if result == dict_io_train_test[key]["outputs"][id_input]:
                                dict_io_train_test[key]["list_correct"][id_input] = True
                        except Exception:
                            pass
                    except Exception as e:
                        error_message = f"ERROR while executing the code, Error: {str(e)}"
                        dict_io_train_test[key]["prediction"][id_input] = "ERROR: while executing the code, Error: " + str(e)

        except MemoryError:
            error_message = f"Error: Memory limit of {max_memory / (1024 * 1024):.2f} MB exceeded"
        except Exception as e:
            error_message = f"Error: {str(e)}"

        for key, values in dict_io_train_test.items():
            list_input_matrix = values["inputs"]
            for id_input, input_matrix in enumerate(list_input_matrix):
                if dict_io_train_test[key]["prediction"][id_input] is None:
                    dict_io_train_test[key]["prediction"][id_input] = error_message

        return dict_io_train_test

    def run(self, code, dict_io_train_test, timeout=None, entry_point='transform'):
        if timeout is None:
            timeout = self.timeout

        with Pool(processes=1, initializer=self._initialize_worker) as pool:
            try:
                result = pool.apply_async(
                    self._run_with_memory_limit,
                    (code, dict_io_train_test, self.max_memory, entry_point)
                )
                res = result.get(timeout=timeout)
                return res
            except TimeoutError:
                pass
            except Exception:
                pass

            return {
                key: {
                    "prediction": [f"Error: Code execution failed"] * len(values["inputs"]),
                    "list_correct": [False] * len(values["inputs"])
                }
                for key, values in dict_io_train_test.items()
            }


def check_solutions(dict_response, dataset, max_workers=None, timeout=5, max_memory=4, keep_only_correct_results=False):
    """Check if code generates the correct output for the 'train' input and 'test' input"""
    list_keys = list(dict_response.keys())
    if max_workers is None:
        max_workers = min(os.cpu_count(), int(2 * os.cpu_count() / 5))
    results = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for key in list_keys:
            data_train_input = dataset[key]["train"]
            list_train_input = [data["input"] for data in data_train_input]
            list_train_output = [data["output"] for data in data_train_input]

            train_data_test_input = dataset[key]["test"]
            list_test_input = [data["input"] for data in train_data_test_input]
            list_test_output = [data["output"] for data in train_data_test_input]

            for j in range(len(dict_response[key])):
                if "code" in dict_response[key][j]:
                    code = dict_response[key][j]["code"]
                else:
                    code = extract_transform(dict_response[key][j]["transform"])

                futures.append(executor.submit(
                    check_solution_batch,
                    list_train_input, list_train_output,
                    list_test_input, list_test_output,
                    code, key, j
                ))

        for future in as_completed(futures):
            try:
                result = future.result(timeout=timeout)
                results.append(result)
            except TimeoutError:
                pass
            except Exception:
                pass

    for result in results:
        key, j, train_results, test_results = result
        if keep_only_correct_results:
            train_results = {k: v for k, v in train_results.items() if k.startswith("correct")}
            test_results = {k: v for k, v in test_results.items() if k.startswith("correct")}
        dict_response[key][j].update(train_results)
        dict_response[key][j].update(test_results)

    return dict_response


def check_solution_batch(train_inputs, train_outputs, test_inputs, test_outputs, code, key, j):
    dict_io_train_test = {
        "train": {
            "inputs": train_inputs,
            "outputs": train_outputs,
            "prediction": [None for _ in range(len(train_inputs))],
            "list_correct": [False for _ in range(len(train_inputs))]
        },
        "test": {
            "inputs": test_inputs,
            "outputs": test_outputs,
            "prediction": [None for _ in range(len(test_outputs))],
            "list_correct": [False for _ in range(len(test_outputs))]
        }
    }
    train_results, test_results = check_solution_set(dict_io_train_test, code)
    return key, j, train_results, test_results


def check_solution_set(dict_io_train_test, code):
    """Results for train or test set"""
    count_matrix = 0
    for _, value in dict_io_train_test.items():
        count_matrix += len(value["inputs"])
    timeout = 1 * count_matrix
    timeout = max(2, timeout)
    timeout = min(20, timeout)

    sandbox = Sandbox(timeout=Timeout_sandbox, max_memory=Memory_limit * 1000 * 1024 * 1024)
    dict_io_train_test_results = sandbox.run(code, dict_io_train_test, timeout=timeout)

    correct_train_input, predicted_train_output = process_result_set(dict_io_train_test_results["train"])
    correct_test_input, predicted_test_output = process_result_set(dict_io_train_test_results["test"])

    train_results = {
        "correct_train_input": correct_train_input,
        "predicted_train_output": remove_circular_refs(predicted_train_output)
    }
    test_results = {
        "correct_test_input": correct_test_input,
        "predicted_test_output": remove_circular_refs(predicted_test_output)
    }
    del sandbox
    return train_results, test_results


def remove_circular_refs(ob, _seen=None):
    """Remove circular references in predicted output"""
    if _seen is None:
        _seen = set()
    if id(ob) in _seen:
        return None
    _seen.add(id(ob))
    res = ob
    if isinstance(ob, dict):
        res = {
            remove_circular_refs(k, _seen): remove_circular_refs(v, _seen)
            for k, v in ob.items()
        }
    elif isinstance(ob, (list, tuple, set, frozenset)):
        res = type(ob)(remove_circular_refs(v, _seen) for v in ob)
    _seen.remove(id(ob))
    return res


def process_result_set(dict_io_train_test_results):
    correct_input = dict_io_train_test_results["list_correct"]
    predicted_output = dict_io_train_test_results["prediction"]
    return correct_input, predicted_output


def check_and_remove_test_transform(code_list):
    """Check if the code contains a test_transform function and remove it if it exists."""
    if not isinstance(code_list, list):
        code_list = [code_list]
        flag_not_list = True
    else:
        flag_not_list = False
    result = []

    for code in code_list:
        has_test_transform = False
        try:
            has_test_transform = "def test_transform" in code or "def check_transform" in code
        except:
            pass

        if has_test_transform:
            # Remove test_transform function
            lines = code.split('\n')
            new_lines = []
            skip = False
            for line in lines:
                if 'def test_transform' in line or 'def check_transform' in line:
                    skip = True
                elif skip and line and not line[0].isspace():
                    skip = False
                if not skip:
                    new_lines.append(line)
            code = '\n'.join(new_lines)
        result.append(code)

    if flag_not_list:
        return result[0]
    return result


def format_all_generation(out, list_task_id, use_vllm_generate=False, keep_full_text=False, use_cot=False):
    """
    Format LLM generation output into a structured dictionary.

    Args:
        out: list[list[str]] - raw output from the model
        list_task_id: list of task identifiers
        use_vllm_generate: bool - if True, process vLLM generate output format
        keep_full_text: bool - if True, keep the full text in output
        use_cot: bool - if True, process chain-of-thought response

    Returns:
        dict_response: dict mapping task_id to list of response dicts
    """
    outputs_copy = out

    # Handle single solution case
    if not isinstance(outputs_copy[0], list):
        outputs_copy = [[outputs_copy[i]] for i in range(len(outputs_copy))]

    count_none = 0
    dict_response = {}

    for i in range(len(outputs_copy)):
        res_task_i = []
        for j in range(len(outputs_copy[i])):
            if outputs_copy[i][j] is None:
                count_none += 1
                output_dict = {"text": "", "code": ""}
            else:
                output_dict = {"text": outputs_copy[i][j]}
                if keep_full_text:
                    output_dict["full_text"] = outputs_copy[i][j]
                output_dict["code"] = extract_transform(output_dict["text"])

            output_dict["code"] = check_and_remove_test_transform(output_dict["code"])
            res_task_i.append(output_dict)

        dict_response[list_task_id[i]] = res_task_i

    if count_none > 0:
        print(f"Number of None: {count_none}")

    return dict_response
