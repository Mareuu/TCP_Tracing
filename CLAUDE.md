# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TCP (Tracing & Correcting Program) is a two-stage framework for solving ARC (Abstraction and Reasoning Corpus) tasks using LLM-based code generation with iterative refinement through detailed feedback.

## Commands

### Run Full Pipeline
```bash
./run_pipeline.sh                    # All stages
./run_pipeline.sh --stage 1          # Stage 1: Generate and Evaluate
./run_pipeline.sh --stage 2          # Stage 2: Create Seed File (optional)
./run_pipeline.sh --stage 3          # Stage 3: Trace and Refine
```

### Manual Execution
```bash
# Stage 1: Generate initial solutions
cd generate_and_evaluate
python generate_and_evaluate.py \
    --path_model "Qwen/Qwen2.5-Coder-7B-Instruct" \
    --num_problems 50 \
    --split train \
    --path_save_res "/data/TCP_Tracing/save_results/model/gen-0"

# Stage 3: Refine solutions
cd trace_and_refine
python tcp_refine.py \
    --path_feedback "/path/to/detailed_feedback_.jsonl" \
    --path_model "Qwen/Qwen2.5-Coder-7B-Instruct" \
    --path_save_res "/path/to/refined"
```

### Environment Setup
```bash
pip install -r requirements.txt
export PYTHONPATH="/path/to/TCP_Tracing:$PYTHONPATH"  # If module import fails
```

## Architecture

```
TCP_Tracing/
├── tcp_core/                    # Core shared library
│   ├── tcp_dataset.py          # ARC dataset loading (train/eval/test splits)
│   ├── llm_service.py          # SGLang LLM serving (ports 30100-30200)
│   ├── prompt.py               # Prompt templates, grid formatting
│   └── sandbox.py              # Safe code execution (3s timeout, 1GB memory)
├── generate_and_evaluate/       # Stage 1: Code generation
│   ├── generate_and_evaluate.py # Main entry - generates code + detailed feedback
│   ├── tcp_utils.py            # Grid analysis, connected components
│   ├── tcp_object_detection.py # Object extraction from grids
│   └── create_seed_file.py     # Extract solved tasks for few-shot examples
└── trace_and_refine/           # Stage 3: Iterative refinement
    ├── tcp_refine.py           # Main entry - feedback-guided code improvement
    └── tcp_evaluation_utils.py # Grid comparison, feedback generation
```

## Data Flow

1. **Stage 1 (Generate)**: ARC tasks → LLM generates Python code → Sandbox executes → Pixel/object-level feedback → `detailed_feedback_.jsonl`
2. **Stage 2 (Seed)**: Extract perfectly solved tasks → `tcp_seed_examples.json` (optional few-shot examples)
3. **Stage 3 (Refine)**: Feedback + failed code → LLM refines → Re-evaluate → Loop up to `max_refinement_retries`

## Key Components

- **LLM_serv** (`tcp_core/llm_service.py`): Manages SGLang server, provides `get_completion()` and `get_multiple_completions()` with retry logic and thread pooling
- **Sandbox** (`tcp_core/sandbox.py`): Safe execution via `check_solutions()`, extracts code with `extract_transform()`
- **Grid formatting** (`tcp_core/prompt.py`): Multiple display modes (ascii, spreadsheet, numpy, colors) via `grid_formatting()`
- **Feedback system**: Pixel-level comparison + object detection (connected components, bounding boxes)

## Configuration

Edit `run_pipeline.sh` configuration section:
- `MODEL_NAME` / `MODEL_PATH`: HuggingFace model identifier
- `NUM_PROBLEMS`: -1 for all tasks
- `SPLIT`: train, eval, or test
- `N_GPU`, `GPU_MEM`, `FP8`: GPU settings
- `MAX_REFINEMENT_RETRIES`: Default 10

## ARC Dataset Location

```
/data/TCP_Tracing/arc-prize-2024/
├── arc-agi_training_challenges.json
├── arc-agi_training_solutions.json
├── arc-agi_evaluation_challenges.json
├── arc-agi_evaluation_solutions.json
└── arc-agi_test_challenges.json
```
