# TCP (Tracing & Correcting Program)

A framework for ARC task solution generation with detailed feedback and iterative refinement.

## Overview

TCP is a two-stage system for solving Abstraction and Reasoning Corpus (ARC) tasks:

1. **Generate and Evaluate**: Creates initial code solutions with detailed pixel-level and object-level feedback
2. **Trace and Refine**: Iteratively improves solutions using dynamic feedback and adaptive strategies

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
  - 7B models: ~16GB VRAM
  - 14B models: ~32GB VRAM
  - 32B models: ~64GB VRAM (or multi-GPU)

### Setup

```bash
# Clone the repository
git clone https://github.com/Mareuu/TCP_Tracing.git
cd TCP_Tracing

# (Optional) Create virtual environment
python -m venv tcp_env
source tcp_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download ARC dataset from Kaggle
mkdir -p /data/TCP_Tracing/arc-prize-2024
# Place dataset files in arc-prize-2024/
```

**Required dataset files:**
```
/data/TCP_Tracing/arc-prize-2024/
├── arc-agi_training_challenges.json
├── arc-agi_training_solutions.json
├── arc-agi_evaluation_challenges.json
├── arc-agi_evaluation_solutions.json
└── arc-agi_test_challenges.json
```

## Quick Start

### Using the Pipeline Runner (Recommended)

Edit `run_pipeline.sh` to set your model, then run:

```bash
# Configure your model at the top of run_pipeline.sh
MODEL_NAME="Qwen2.5-Coder-7B-Instruct"
MODEL_PATH="Qwen/Qwen2.5-Coder-7B-Instruct"

# Run all stages
./run_pipeline.sh

# Or run individual stages
./run_pipeline.sh --stage 1    # Generate and Evaluate
./run_pipeline.sh --stage 2    # Create Seed File (optional)
./run_pipeline.sh --stage 3    # Trace and Refine
```

### Manual Execution

```bash
# 1. Generate initial solutions
cd generate_and_evaluate
CUDA_VISIBLE_DEVICES=0 python generate_and_evaluate.py \
    --path_model "Qwen/Qwen2.5-Coder-7B-Instruct" \
    --num_problems 50 \
    --split train \
    --path_save_res "/data/TCP_Tracing/save_results/Qwen2.5-Coder-7B-Instruct/gen-0"

# 2. Refine solutions
cd ../trace_and_refine
python tcp_refine.py \
    --path_feedback "/data/TCP_Tracing/save_results/Qwen2.5-Coder-7B-Instruct/gen-0/detailed_feedback_.jsonl" \
    --path_save_res "/data/TCP_Tracing/save_results/Qwen2.5-Coder-7B-Instruct/refined" \
    --path_model "Qwen/Qwen2.5-Coder-7B-Instruct" \
    --max_refinement_retries 10
```

## Configuration

### Key Parameters

**Generate and Evaluate:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--path_model` | required | HuggingFace model identifier |
| `--num_problems` | -1 | Number of tasks (-1 for all) |
| `--split` | train | Dataset split: train, eval, test |
| `--temperature` | 0.7 | Sampling temperature |
| `--max_tokens` | 2048 | Maximum generation tokens |
| `--fp8` | false | Use FP8 precision |

**Trace and Refine:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--path_model` | required | HuggingFace model identifier |
| `--path_feedback` | required | Path to feedback JSONL from Stage 1 |
| `--max_refinement_retries` | 10 | Maximum iterations per task |
| `--temperature` | 0.4 | Sampling temperature |

## Output

Results are organized by model name:

```
/data/TCP_Tracing/save_results/
├── Qwen2.5-Coder-7B-Instruct/
│   ├── gen-0/
│   │   └── detailed_feedback_.jsonl
│   └── refined/
│       ├── {task_id}_refinement_log.jsonl
│       ├── {task_id}_refinement_log_debug.txt
│       └── {task_id}_final.py
```

## Repository Structure

```
TCP_Tracing/
├── run_pipeline.sh              # Main pipeline runner
├── requirements.txt             # Python dependencies
├── tcp_core/                    # Shared utilities
│   ├── tcp_dataset.py          # ARC dataset loading
│   └── llm_service.py          # LLM serving
├── generate_and_evaluate/       # Stage 1
│   ├── generate_and_evaluate.py
│   ├── tcp_utils.py
│   ├── tcp_object_detection.py
│   └── create_seed_file.py
└── trace_and_refine/           # Stage 2
    ├── tcp_refine.py
    └── tcp_evaluation_utils.py
```

## Troubleshooting

### ImportError: No module named 'tcp_core'
```bash
export PYTHONPATH="/path/to/TCP_Tracing:$PYTHONPATH"
```

### CUDA out of memory
- Use a smaller model (7B instead of 32B)
- Enable FP8: `--fp8`
- Reduce `--gpu_mem` (default: 0.85)

### SGLang server fails to start
- Check if port 30100-30200 is in use
- Reduce `--gpu_mem` parameter

## Contact

- GitHub: https://github.com/Mareuu/TCP_Tracing
- Issues: https://github.com/Mareuu/TCP_Tracing/issues
