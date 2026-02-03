#!/usr/bin/env python3
"""
TCP Ablation Experiment Runner

Automates systematic ablation experiments to separate the contribution of
domain heuristics vs feedback mechanism.

Usage:
    python experiments/ablation_runner.py \
        --base_args "--path_feedback data/feedback.jsonl --path_model Qwen/Qwen2.5-Coder-7B-Instruct" \
        --output_dir results/ablation_study/
"""

import argparse
import os
import sys
import subprocess
import json
import time
from datetime import datetime
from typing import Dict, List, Optional

# Add parent directory to path for tcp_core imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tcp_core.ablation_config import AblationConfig, ABLATION_EXPERIMENTS, list_ablation_configs


# Predefined ablation experiments
ABLATION_EXPERIMENT_CONFIGS = {
    "full_system": {
        "description": "Full TCP system with all heuristics",
        "args": ["--ablation_mode", "full"],
    },
    "no_heuristics": {
        "description": "Feedback-only: all domain heuristics disabled",
        "args": ["--ablation_mode", "no_heuristics"],
    },
    "feedback_only": {
        "description": "Pure feedback mechanism (alias for no_heuristics)",
        "args": ["--ablation_mode", "feedback_only"],
    },
    "heuristics_only": {
        "description": "Domain heuristics with minimal feedback",
        "args": ["--ablation_mode", "heuristics_only"],
    },
    "no_feedback": {
        "description": "Domain heuristics only, no evaluation feedback",
        "args": ["--ablation_mode", "no_feedback"],
    },
    "raw_feedback_only": {
        "description": "Pure numerical feedback without domain interpretation",
        "args": ["--ablation_mode", "raw_feedback_only", "--feedback_style", "raw"],
    },
    "heuristic_free": {
        "description": "Completely heuristic-free: raw feedback + history-based strategy + fixed temperature",
        "args": ["--ablation_mode", "heuristic_free"],
    },
    "history_based_strategy": {
        "description": "History-based strategy selection (no accuracy thresholds)",
        "args": ["--ablation_mode", "history_based_strategy", "--strategy_mode", "history_based"],
    },
    "no_structural_hints": {
        "description": "Structural hints disabled",
        "args": ["--disable_structural_hints"],
    },
    "no_adaptive_strategy": {
        "description": "Adaptive strategy selection disabled",
        "args": ["--disable_adaptive_strategy"],
    },
    "no_accuracy_hints": {
        "description": "Accuracy-based hints disabled",
        "args": ["--disable_accuracy_hints"],
    },
    "fixed_temperature": {
        "description": "Fixed temperature (no adaptive adjustment)",
        "args": ["--temperature_mode", "fixed"],
    },
    # =========================================================================
    # Feedback Granularity Ablation Ladder
    # These experiments test the minimal sufficient feedback hypothesis
    # =========================================================================
    "feedback_none": {
        "description": "Level 0: No feedback - just 'your code failed, try again'",
        "args": ["--ablation_mode", "feedback_none", "--feedback_granularity", "0"],
    },
    "feedback_binary": {
        "description": "Level 1: Binary pass/fail feedback only",
        "args": ["--ablation_mode", "feedback_binary", "--feedback_granularity", "1"],
    },
    "feedback_accuracy": {
        "description": "Level 2: Accuracy score only",
        "args": ["--ablation_mode", "feedback_accuracy", "--feedback_granularity", "2"],
    },
    "feedback_shape": {
        "description": "Level 3: Accuracy + shape match info",
        "args": ["--ablation_mode", "feedback_shape", "--feedback_granularity", "3"],
    },
    "feedback_count": {
        "description": "Level 4: Accuracy + shape + error counts",
        "args": ["--ablation_mode", "feedback_count", "--feedback_granularity", "4"],
    },
    "feedback_position": {
        "description": "Level 5: Full raw metrics including error positions",
        "args": ["--ablation_mode", "feedback_position", "--feedback_granularity", "5"],
    },
    "feedback_full_raw": {
        "description": "Level 6: All raw metrics (no domain interpretation)",
        "args": ["--ablation_mode", "feedback_full_raw", "--feedback_granularity", "6"],
    },
}


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="TCP Ablation Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available experiments:
  full_system           Full TCP system with all heuristics (baseline)
  no_heuristics         Feedback-only: all domain heuristics disabled
  feedback_only         Pure feedback mechanism (alias for no_heuristics)
  heuristics_only       Domain heuristics with minimal feedback
  no_feedback           Domain heuristics only, no evaluation feedback
  raw_feedback_only     Pure numerical feedback without domain interpretation
  heuristic_free        Completely heuristic-free (raw + history-based + fixed temp)
  history_based_strategy  History-based strategy (no accuracy thresholds)
  no_structural_hints   Structural hints disabled
  no_adaptive_strategy  Adaptive strategy selection disabled
  no_accuracy_hints     Accuracy-based hints disabled
  fixed_temperature     Fixed temperature (no adaptive adjustment)

Feedback Granularity Ablation (minimal sufficient feedback study):
  feedback_none         Level 0: No feedback - just 'try again'
  feedback_binary       Level 1: Binary pass/fail only
  feedback_accuracy     Level 2: Accuracy score only
  feedback_shape        Level 3: Accuracy + shape match
  feedback_count        Level 4: + error counts
  feedback_position     Level 5: + error positions
  feedback_full_raw     Level 6: All raw metrics (no interpretation)

Example:
  python experiments/ablation_runner.py \\
      --experiments full_system heuristic_free history_based_strategy no_feedback \\
      --base_args "--path_feedback data/feedback.jsonl --path_model model_name" \\
      --output_dir results/ablation/

Example (feedback granularity ablation):
  python experiments/ablation_runner.py \\
      --experiments feedback_none feedback_binary feedback_accuracy feedback_position full_system \\
      --base_args "--path_feedback data/feedback.jsonl --path_model model_name" \\
      --output_dir results/feedback_ablation/
        """
    )

    parser.add_argument("--experiments", nargs="+", default=["full_system", "no_heuristics", "no_feedback"],
                       choices=list(ABLATION_EXPERIMENT_CONFIGS.keys()) + ["all"],
                       help="Experiments to run (default: full_system, no_heuristics, no_feedback)")
    parser.add_argument("--base_args", type=str, required=True,
                       help="Base arguments for tcp_refine.py (quoted string)")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for experiment results")
    parser.add_argument("--tcp_refine_path", type=str, default=None,
                       help="Path to tcp_refine.py (default: auto-detect)")
    parser.add_argument("--dry_run", action="store_true",
                       help="Print commands without executing")
    parser.add_argument("--sequential", action="store_true",
                       help="Run experiments sequentially (default: parallel where possible)")
    parser.add_argument("--timeout", type=int, default=None,
                       help="Timeout in seconds for each experiment")

    return parser.parse_args()


def get_tcp_refine_path(provided_path: Optional[str] = None) -> str:
    """Get path to tcp_refine.py."""
    if provided_path:
        return provided_path

    # Try to find tcp_refine.py relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    possible_paths = [
        os.path.join(script_dir, "..", "trace_and_refine", "tcp_refine.py"),
        os.path.join(script_dir, "trace_and_refine", "tcp_refine.py"),
    ]

    for path in possible_paths:
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path):
            return abs_path

    raise FileNotFoundError("Could not find tcp_refine.py. Please specify --tcp_refine_path")


def build_command(tcp_refine_path: str, base_args: str, experiment_name: str, output_dir: str) -> List[str]:
    """Build command for running an experiment."""
    experiment_config = ABLATION_EXPERIMENT_CONFIGS[experiment_name]
    experiment_output_dir = os.path.join(output_dir, experiment_name)

    cmd = [
        sys.executable,
        tcp_refine_path,
    ]

    # Add base args (split by whitespace, handling quoted strings)
    import shlex
    cmd.extend(shlex.split(base_args))

    # Add experiment-specific args
    cmd.extend(experiment_config["args"])

    # Set output directory for this experiment
    # Check if --path_save_res is already in base_args
    if "--path_save_res" not in base_args:
        cmd.extend(["--path_save_res", experiment_output_dir])
    else:
        # Replace existing path_save_res
        for i, arg in enumerate(cmd):
            if arg == "--path_save_res" and i + 1 < len(cmd):
                cmd[i + 1] = experiment_output_dir
                break

    return cmd


def run_experiment(cmd: List[str], experiment_name: str, output_dir: str,
                   dry_run: bool = False, timeout: Optional[int] = None) -> Dict:
    """Run a single experiment and return results."""
    result = {
        "experiment": experiment_name,
        "command": " ".join(cmd),
        "start_time": datetime.now().isoformat(),
        "status": "pending",
    }

    experiment_output_dir = os.path.join(output_dir, experiment_name)
    os.makedirs(experiment_output_dir, exist_ok=True)

    if dry_run:
        print(f"\n[DRY RUN] Would execute: {' '.join(cmd)}")
        result["status"] = "dry_run"
        return result

    print(f"\n{'='*60}")
    print(f"Running experiment: {experiment_name}")
    print(f"Description: {ABLATION_EXPERIMENT_CONFIGS[experiment_name]['description']}")
    print(f"Command: {' '.join(cmd)}")
    print(f"Output dir: {experiment_output_dir}")
    print(f"{'='*60}\n")

    try:
        start_time = time.time()
        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        end_time = time.time()

        result["duration_seconds"] = end_time - start_time
        result["return_code"] = process.returncode
        result["status"] = "success" if process.returncode == 0 else "failed"

        # Save stdout and stderr
        log_path = os.path.join(experiment_output_dir, "experiment_log.txt")
        with open(log_path, 'w') as f:
            f.write(f"=== {experiment_name} Experiment Log ===\n")
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"Return code: {process.returncode}\n")
            f.write(f"Duration: {result['duration_seconds']:.2f}s\n\n")
            f.write("=== STDOUT ===\n")
            f.write(process.stdout)
            f.write("\n=== STDERR ===\n")
            f.write(process.stderr)

        if process.returncode == 0:
            print(f"[SUCCESS] {experiment_name} completed in {result['duration_seconds']:.2f}s")
        else:
            print(f"[FAILED] {experiment_name} failed with return code {process.returncode}")
            print(f"See log at: {log_path}")

    except subprocess.TimeoutExpired:
        result["status"] = "timeout"
        result["error"] = f"Experiment timed out after {timeout}s"
        print(f"[TIMEOUT] {experiment_name} timed out after {timeout}s")

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        print(f"[ERROR] {experiment_name} failed with error: {e}")

    result["end_time"] = datetime.now().isoformat()
    return result


def main():
    args = parse_arguments()

    # Expand 'all' to all experiments
    if "all" in args.experiments:
        experiments = list(ABLATION_EXPERIMENT_CONFIGS.keys())
    else:
        experiments = args.experiments

    # Get tcp_refine.py path
    tcp_refine_path = get_tcp_refine_path(args.tcp_refine_path)
    print(f"Using tcp_refine.py at: {tcp_refine_path}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save experiment configuration
    config_path = os.path.join(args.output_dir, "ablation_config.json")
    with open(config_path, 'w') as f:
        json.dump({
            "experiments": experiments,
            "base_args": args.base_args,
            "output_dir": args.output_dir,
            "tcp_refine_path": tcp_refine_path,
            "start_time": datetime.now().isoformat(),
        }, f, indent=2)

    print(f"\n{'#'*60}")
    print(f"TCP Ablation Study Runner")
    print(f"{'#'*60}")
    print(f"Experiments to run: {', '.join(experiments)}")
    print(f"Output directory: {args.output_dir}")
    print(f"Base args: {args.base_args}")
    print(f"Dry run: {args.dry_run}")
    print(f"{'#'*60}\n")

    # Run experiments
    results = []
    for experiment_name in experiments:
        cmd = build_command(tcp_refine_path, args.base_args, experiment_name, args.output_dir)
        result = run_experiment(cmd, experiment_name, args.output_dir,
                               dry_run=args.dry_run, timeout=args.timeout)
        results.append(result)

    # Save results
    results_path = os.path.join(args.output_dir, "ablation_results.json")
    with open(results_path, 'w') as f:
        json.dump({
            "experiments": results,
            "end_time": datetime.now().isoformat(),
            "summary": {
                "total": len(results),
                "success": sum(1 for r in results if r["status"] == "success"),
                "failed": sum(1 for r in results if r["status"] == "failed"),
                "timeout": sum(1 for r in results if r["status"] == "timeout"),
                "error": sum(1 for r in results if r["status"] == "error"),
            }
        }, f, indent=2)

    # Print summary
    print(f"\n{'#'*60}")
    print("Ablation Study Complete")
    print(f"{'#'*60}")
    print(f"Results saved to: {results_path}")

    success_count = sum(1 for r in results if r["status"] == "success")
    failed_count = sum(1 for r in results if r["status"] in ["failed", "timeout", "error"])

    print(f"\nSummary:")
    print(f"  - Success: {success_count}/{len(results)}")
    print(f"  - Failed: {failed_count}/{len(results)}")

    if not args.dry_run:
        print(f"\nNext step: Run analyze_ablation.py to analyze results:")
        print(f"  python experiments/analyze_ablation.py --results_dir {args.output_dir}")


if __name__ == "__main__":
    main()
