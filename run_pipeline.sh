#!/bin/bash
#
# TCP Pipeline Runner Script
# ==========================
# This script runs the complete TCP (Trace-Code-Perfect) pipeline for ARC tasks.
# Set your model configuration once at the top, and run all stages automatically.
#
# Pipeline Stages:
#   Stage 1: Generate and Evaluate - Generate code solutions and evaluate them
#   Stage 2: Create Seed File - Extract perfect solutions for few-shot examples (optional)
#   Stage 3: Trace and Refine - Iteratively improve failed solutions
#
# Usage:
#   ./run_pipeline.sh                    # Run all stages
#   ./run_pipeline.sh --stage 1          # Run only stage 1 (generate)
#   ./run_pipeline.sh --stage 2          # Run only stage 2 (create seed - optional)
#   ./run_pipeline.sh --stage 3          # Run only stage 3 (refine)
#

set -e  # Exit on error

# =============================================================================
# CONFIGURATION - SET YOUR MODEL AND PARAMETERS HERE (CHANGE ONLY THIS SECTION)
# =============================================================================

# Model configuration (used for both generation and refinement stages)
MODEL_NAME="Qwen2.5-Coder-7B-Instruct"           # Model folder name for results
MODEL_PATH="Qwen/Qwen2.5-Coder-7B-Instruct"      # HuggingFace model ID

# Paths
BASE_SAVE_PATH="/data/barc_feedback/TCP_Tracing/save_results"
DATA_PATH="/data/barc_feedback/TCP_Tracing"      # Contains arc-prize-2024 or arc-prize-2025

# Generation parameters
NUM_PROBLEMS=50                  # Number of ARC tasks to process (-1 for all)
SPLIT="train"                    # Dataset split: train, eval, or test
TEMPERATURE=0.7                  # Sampling temperature for generation
MAX_TOKENS=2048                  # Max tokens for generation
N_GPU=1                          # Number of GPUs
GPU_MEM=0.85                     # GPU memory utilization (0.0-1.0)
FP8="--fp8"                      # Use FP8 precision (set to "" to disable)
ENABLE_THINKING=""               # Set to "--enable_thinking" for Qwen thinking mode

# Refinement parameters
MAX_REFINEMENT_RETRIES=10        # Maximum refinement iterations
REFINE_TEMPERATURE=0.4           # Temperature for refinement

# GPU configuration
export CUDA_VISIBLE_DEVICES=0

# Seed examples (optional - leave empty to use default or if not available yet)
SEED_EXAMPLES_PATH=""            # Set to path if you have pre-existing seed examples

# =============================================================================
# DERIVED PATHS (DO NOT MODIFY)
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GENERATE_DIR="${SCRIPT_DIR}/generate_and_evaluate"
REFINE_DIR="${SCRIPT_DIR}/trace_and_refine"

MODEL_SAVE_DIR="${BASE_SAVE_PATH}/${MODEL_NAME}"
GEN_RESULTS_DIR="${MODEL_SAVE_DIR}/gen-0"
SEED_FILE="${MODEL_SAVE_DIR}/tcp_seed_examples.json"
FEEDBACK_FILE="${GEN_RESULTS_DIR}/detailed_feedback_.jsonl"
REFINED_RESULTS_DIR="${MODEL_SAVE_DIR}/refined"

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

print_header() {
    echo ""
    echo "============================================================================="
    echo "$1"
    echo "============================================================================="
    echo ""
}

print_config() {
    print_header "TCP Pipeline Configuration"
    echo "Model Name:           ${MODEL_NAME}"
    echo "Model Path:           ${MODEL_PATH}"
    echo "Base Save Path:       ${BASE_SAVE_PATH}"
    echo "Data Path:            ${DATA_PATH}"
    echo "Results Directory:    ${MODEL_SAVE_DIR}"
    echo ""
    echo "Hardware Settings:"
    echo "  - GPUs:             ${N_GPU}"
    echo "  - GPU Memory:       ${GPU_MEM}"
    echo "  - FP8:              ${FP8:-disabled}"
    echo "  - CUDA Devices:     ${CUDA_VISIBLE_DEVICES}"
    echo ""
    echo "Generation Settings:"
    echo "  - Problems:         ${NUM_PROBLEMS}"
    echo "  - Split:            ${SPLIT}"
    echo "  - Temperature:      ${TEMPERATURE}"
    echo "  - Max Tokens:       ${MAX_TOKENS}"
    echo ""
    echo "Refinement Settings:"
    echo "  - Max Retries:      ${MAX_REFINEMENT_RETRIES}"
    echo "  - Temperature:      ${REFINE_TEMPERATURE}"
    echo ""
}

# =============================================================================
# STAGE 1: Generate and Evaluate
# =============================================================================

run_stage_1_generate() {
    print_header "STAGE 1: Generate and Evaluate Solutions"

    # Create output directory
    mkdir -p "${GEN_RESULTS_DIR}"

    cd "${GENERATE_DIR}"

    echo "Running generate_and_evaluate.py..."
    echo "Output will be saved to: ${GEN_RESULTS_DIR}"
    echo ""

    # Build optional arguments
    SEED_ARG=""
    if [ -n "${SEED_EXAMPLES_PATH}" ] && [ -f "${SEED_EXAMPLES_PATH}" ]; then
        SEED_ARG="--seed_examples_path ${SEED_EXAMPLES_PATH}"
        echo "Using seed examples from: ${SEED_EXAMPLES_PATH}"
    fi

    python generate_and_evaluate.py \
        --path_model "${MODEL_PATH}" \
        --num_problems ${NUM_PROBLEMS} \
        --split ${SPLIT} \
        --temperature ${TEMPERATURE} \
        --max_tokens ${MAX_TOKENS} \
        --n_gpu ${N_GPU} \
        --gpu_mem ${GPU_MEM} \
        --path_save_res "${GEN_RESULTS_DIR}" \
        ${FP8} \
        ${ENABLE_THINKING} \
        ${SEED_ARG}

    echo ""
    echo "Stage 1 complete. Results saved to: ${GEN_RESULTS_DIR}"
}

# =============================================================================
# STAGE 2: Create Seed File (Optional - for few-shot examples)
# =============================================================================

run_stage_2_create_seed() {
    print_header "STAGE 2: Create Seed File for Few-Shot Examples"

    cd "${GENERATE_DIR}"

    echo "Creating seed examples from solved tasks..."
    echo "Input pickle: ${GEN_RESULTS_DIR}/solved_gen_0_with_train.pkl"
    echo "Output JSON:  ${SEED_FILE}"
    echo ""

    python create_seed_file.py \
        --model "${MODEL_NAME}" \
        --base_save_path "${BASE_SAVE_PATH}"

    echo ""
    echo "Stage 2 complete. Seed file saved to: ${SEED_FILE}"
}

# =============================================================================
# STAGE 3: Trace and Refine
# =============================================================================

run_stage_3_refine() {
    print_header "STAGE 3: Trace and Refine Solutions"

    # Check if feedback file exists
    if [ ! -f "${FEEDBACK_FILE}" ]; then
        echo "ERROR: Feedback file not found at: ${FEEDBACK_FILE}"
        echo "Please run Stage 1 first."
        exit 1
    fi

    # Create output directory
    mkdir -p "${REFINED_RESULTS_DIR}"

    cd "${REFINE_DIR}"

    echo "Running tcp_refine.py..."
    echo "Input feedback: ${FEEDBACK_FILE}"
    echo "Output:         ${REFINED_RESULTS_DIR}"
    echo ""

    python tcp_refine.py \
        --path_model "${MODEL_PATH}" \
        --path_feedback "${FEEDBACK_FILE}" \
        --path_save_res "${REFINED_RESULTS_DIR}" \
        --base_path "${DATA_PATH}" \
        --n_gpu ${N_GPU} \
        --gpu_mem ${GPU_MEM} \
        --max_refinement_retries ${MAX_REFINEMENT_RETRIES} \
        --temperature ${REFINE_TEMPERATURE} \
        ${FP8}

    echo ""
    echo "Stage 3 complete. Refined results saved to: ${REFINED_RESULTS_DIR}"
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================

# Parse command line arguments
STAGE=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --stage)
            STAGE="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [--stage N]"
            echo ""
            echo "Options:"
            echo "  --stage N    Run only stage N (1, 2, or 3)"
            echo "               1 = Generate and Evaluate"
            echo "               2 = Create Seed File"
            echo "               3 = Trace and Refine"
            echo ""
            echo "If no stage is specified, all stages run sequentially."
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Print configuration
print_config

# Run stages
if [ -z "${STAGE}" ]; then
    # Run all stages
    run_stage_1_generate
    run_stage_2_create_seed
    run_stage_3_refine

    print_header "PIPELINE COMPLETE"
    echo "All results saved to: ${MODEL_SAVE_DIR}"
else
    # Run specific stage
    case ${STAGE} in
        1)
            run_stage_1_generate
            ;;
        2)
            run_stage_2_create_seed
            ;;
        3)
            run_stage_3_refine
            ;;
        *)
            echo "Invalid stage: ${STAGE}. Must be 1, 2, or 3."
            exit 1
            ;;
    esac
fi

echo ""
echo "Done!"
