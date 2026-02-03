# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TCP (Tracing & Correcting Program) is a two-stage framework for solving ARC (Abstraction and Reasoning Corpus) tasks using LLM-based code generation with iterative refinement through detailed feedback.

## Research Goals

### Main Research Question

**"What is the minimal sufficient feedback for effective LLM self-correction?"**

This project investigates how much feedback information is actually necessary for LLMs to successfully refine their outputs through iterative correction.

### Motivation

Reviewer feedback on the original TCP paper raised important questions:
- "How much is domain heuristics responsible for performance vs the iterative feedback mechanism itself?"
- "It is hard to attribute gains to the method rather than to evaluation or prompt/setup choices"

### Research Approach

We address these questions through systematic ablation studies:

1. **Feedback Granularity Ablation**: Test 8 levels of feedback information
   - Level 0 (NONE): "Your code failed. Try again."
   - Level 1 (BINARY): Pass/Fail only
   - Level 2 (ACCURACY): Accuracy score (e.g., "75%")
   - Level 3 (SHAPE): + Shape match info
   - Level 4 (COUNT): + Error counts
   - Level 5 (POSITION): + Error coordinates
   - Level 6 (FULL_RAW): All numerical metrics
   - Level 7 (INTERPRETED): + Domain-specific interpretation

2. **Domain Transfer**: Validate across multiple domains (ARC, HumanEval) to prove method generalization

3. **Domain Knowledge Separation**: Compare raw feedback vs interpreted feedback to isolate method contribution

### Key Hypotheses

1. There exists a "knee point" where additional feedback provides diminishing returns
2. This minimal sufficient feedback level may vary by domain and model capability
3. The iterative feedback mechanism itself (not domain knowledge) is the primary contributor

### Expected Contributions

- Empirical evidence for minimal sufficient feedback across domains
- Domain-agnostic refinement framework
- Guidelines for efficient feedback design in LLM self-correction systems

## Ablation Study Support

### Running Ablation Experiments

```bash
# Feedback granularity ablation
python experiments/ablation_runner.py \
    --experiments feedback_none feedback_binary feedback_accuracy feedback_count feedback_full_raw full_system \
    --base_args "--path_feedback data/feedback.jsonl --path_model model_name --num_tasks 100" \
    --output_dir results/feedback_ablation/

# Domain knowledge ablation
python experiments/ablation_runner.py \
    --experiments full_system raw_feedback_only heuristic_free no_feedback \
    --base_args "--path_feedback data/feedback.jsonl --path_model model_name" \
    --output_dir results/domain_knowledge_ablation/
```

### Key Ablation Configurations

| Config | Domain Knowledge | Feedback Type | Purpose |
|--------|------------------|---------------|---------|
| `full_system` | ON | Interpreted | Baseline (best performance) |
| `raw_feedback_only` | OFF | Raw numerical | Method contribution only |
| `heuristic_free` | OFF | Raw + Fixed strategy | Pure mechanism |
| `no_feedback` | ON | None | Domain heuristics only |
| `feedback_none` ~ `feedback_full_raw` | OFF | Granularity levels 0-6 | Information ablation |

### Key Files for Ablation

- `tcp_core/ablation_config.py`: AblationConfig class with presets
- `tcp_core/raw_feedback.py`: FeedbackGranularity enum, RawFeedbackGenerator
- `tcp_core/domains/`: Domain abstraction layer (ARC, HumanEval adapters)
- `experiments/ablation_runner.py`: Automated experiment runner

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
│   ├── sandbox.py              # Safe code execution (3s timeout, 1GB memory)
│   ├── ablation_config.py      # Ablation study configuration & presets
│   ├── raw_feedback.py         # Domain-agnostic numerical feedback & granularity
│   └── domains/                # Domain abstraction layer
│       ├── base.py             # DomainAdapter interface, ProgressState, RefinementHistory
│       ├── arc_domain.py       # ARC domain adapter
│       └── humaneval_domain.py # HumanEval domain adapter
├── generate_and_evaluate/       # Stage 1: Code generation
│   ├── generate_and_evaluate.py # Main entry - generates code + detailed feedback
│   ├── tcp_utils.py            # Grid analysis, connected components
│   ├── tcp_object_detection.py # Object extraction from grids
│   └── create_seed_file.py     # Extract solved tasks for few-shot examples
├── trace_and_refine/           # Stage 3: Iterative refinement
│   ├── tcp_refine.py           # Main entry - feedback-guided code improvement
│   └── tcp_evaluation_utils.py # Grid comparison, feedback generation
└── experiments/                 # Ablation study experiments
    ├── ablation_runner.py      # Automated experiment runner
    └── analyze_ablation.py     # Results analysis & visualization
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
