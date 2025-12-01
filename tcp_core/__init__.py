"""
TCP Core - Shared Utilities

TCP (Tracing & Correcting Program)

Shared utilities used across TCP packages.

Modules:
- tcp_dataset: Dataset loading for ARC tasks
- llm_service: Language model serving for TCP pipeline
- prompt: Prompt templates for ARC task solving
- sandbox: Safe code execution for ARC solutions
"""

__version__ = "1.0.0"
__author__ = "TCP Team"

from .tcp_dataset import get_dataset, merge_GT, convert_to_list, reformat_solutions_parquet
from .llm_service import LLM_serv
from .prompt import (
    get_solver_prompt,
    prompt_fewshot_v1,
    prompt_wo_fewshot_v1_,
    format_task,
    format_fewshot_tasks,
    get_list_fewshot_example,
    grid_formatting,
)
from .sandbox import check_solutions, Sandbox, extract_transform, format_all_generation

__all__ = [
    # Dataset
    'get_dataset',
    'merge_GT',
    'convert_to_list',
    'reformat_solutions_parquet',
    'LLM_serv',
    'get_solver_prompt',
    'prompt_fewshot_v1',
    'prompt_wo_fewshot_v1_',
    'format_task',
    'format_fewshot_tasks',
    'get_list_fewshot_example',
    'grid_formatting',
    'check_solutions',
    'Sandbox',
    'extract_transform',
]
