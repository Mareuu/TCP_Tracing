#!/usr/bin/env python3
"""
TCP Ablation Results Analyzer

Analyzes ablation experiment results and calculates component contributions.

Usage:
    python experiments/analyze_ablation.py --results_dir results/ablation_study/
"""

import argparse
import os
import sys
import json
import glob
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# Add parent directory to path for tcp_core imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class TaskResult:
    """Result for a single task."""
    task_id: str
    final_accuracy: float
    iterations: int
    solved: bool  # accuracy == 1.0
    initial_accuracy: float = 0.0
    improvement: float = 0.0


@dataclass
class ExperimentSummary:
    """Summary statistics for an experiment."""
    experiment_name: str
    description: str
    total_tasks: int
    solved_tasks: int
    solve_rate: float
    avg_accuracy: float
    avg_iterations: float
    avg_improvement: float
    task_results: List[TaskResult]


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="TCP Ablation Results Analyzer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--results_dir", type=str, required=True,
                       help="Directory containing ablation experiment results")
    parser.add_argument("--output_format", type=str, default="markdown",
                       choices=["markdown", "json", "csv"],
                       help="Output format for the report")
    parser.add_argument("--output_file", type=str, default=None,
                       help="Output file path (default: stdout)")
    parser.add_argument("--baseline", type=str, default="full_system",
                       help="Baseline experiment for comparison")

    return parser.parse_args()


def load_experiment_results(experiment_dir: str) -> List[TaskResult]:
    """Load results from an experiment directory."""
    results = []

    # Find all refinement log files
    log_files = glob.glob(os.path.join(experiment_dir, "*_refinement_log.jsonl"))

    for log_file in log_files:
        task_id = os.path.basename(log_file).replace("_refinement_log.jsonl", "")

        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()

            if not lines:
                continue

            # Parse entries
            entries = [json.loads(line) for line in lines if line.strip()]

            if not entries:
                continue

            # Get initial accuracy from first entry
            initial_entry = entries[0]
            initial_accuracy = initial_entry.get("pixel_accuracy", 0.0)

            # Find final accuracy (last non-metadata entry)
            final_accuracy = initial_accuracy
            iterations = 0

            for entry in entries:
                if entry.get("type") in ["original", "challenger"]:
                    accuracy = entry.get("pixel_accuracy", 0.0)
                    if accuracy is not None:
                        final_accuracy = accuracy
                iteration = entry.get("iteration", 0)
                if iteration > iterations:
                    iterations = iteration

            result = TaskResult(
                task_id=task_id,
                final_accuracy=final_accuracy,
                iterations=iterations,
                solved=(final_accuracy >= 1.0),
                initial_accuracy=initial_accuracy,
                improvement=final_accuracy - initial_accuracy
            )
            results.append(result)

        except Exception as e:
            print(f"Warning: Failed to parse {log_file}: {e}", file=sys.stderr)

    return results


def compute_experiment_summary(experiment_name: str, results: List[TaskResult],
                                description: str = "") -> ExperimentSummary:
    """Compute summary statistics for an experiment."""
    if not results:
        return ExperimentSummary(
            experiment_name=experiment_name,
            description=description,
            total_tasks=0,
            solved_tasks=0,
            solve_rate=0.0,
            avg_accuracy=0.0,
            avg_iterations=0.0,
            avg_improvement=0.0,
            task_results=[]
        )

    total_tasks = len(results)
    solved_tasks = sum(1 for r in results if r.solved)
    solve_rate = solved_tasks / total_tasks if total_tasks > 0 else 0.0
    avg_accuracy = sum(r.final_accuracy for r in results) / total_tasks
    avg_iterations = sum(r.iterations for r in results) / total_tasks
    avg_improvement = sum(r.improvement for r in results) / total_tasks

    return ExperimentSummary(
        experiment_name=experiment_name,
        description=description,
        total_tasks=total_tasks,
        solved_tasks=solved_tasks,
        solve_rate=solve_rate,
        avg_accuracy=avg_accuracy,
        avg_iterations=avg_iterations,
        avg_improvement=avg_improvement,
        task_results=results
    )


def load_all_experiments(results_dir: str) -> Dict[str, ExperimentSummary]:
    """Load all experiments from results directory."""
    summaries = {}

    # Load experiment config if available
    config_path = os.path.join(results_dir, "ablation_config.json")
    experiment_descriptions = {}

    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)

    # Find all experiment directories
    for entry in os.listdir(results_dir):
        experiment_dir = os.path.join(results_dir, entry)
        if os.path.isdir(experiment_dir) and not entry.startswith('.'):
            results = load_experiment_results(experiment_dir)
            description = experiment_descriptions.get(entry, "")
            summaries[entry] = compute_experiment_summary(entry, results, description)

    return summaries


def compute_contributions(summaries: Dict[str, ExperimentSummary],
                          baseline: str = "full_system") -> Dict[str, Dict]:
    """Compute contribution of each component relative to baseline."""
    contributions = {}

    if baseline not in summaries:
        print(f"Warning: Baseline '{baseline}' not found in experiments", file=sys.stderr)
        return contributions

    baseline_summary = summaries[baseline]

    for exp_name, summary in summaries.items():
        if exp_name == baseline:
            continue

        solve_rate_diff = baseline_summary.solve_rate - summary.solve_rate
        accuracy_diff = baseline_summary.avg_accuracy - summary.avg_accuracy

        contributions[exp_name] = {
            "solve_rate_contribution": solve_rate_diff,
            "accuracy_contribution": accuracy_diff,
            "solve_rate_pct": solve_rate_diff * 100,
            "accuracy_pct": accuracy_diff * 100,
        }

    return contributions


def generate_markdown_report(summaries: Dict[str, ExperimentSummary],
                             contributions: Dict[str, Dict],
                             baseline: str) -> str:
    """Generate markdown report."""
    lines = []

    lines.append("# TCP Ablation Study Results")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append("This report analyzes the contribution of domain heuristics vs feedback mechanism")
    lines.append("in the TCP (Tracing & Correcting Program) framework.")
    lines.append("")

    # Main results table
    lines.append("## Experiment Results")
    lines.append("")
    lines.append("| Experiment | Tasks | Solved | Solve Rate | Avg Accuracy | Diff vs Baseline |")
    lines.append("|------------|-------|--------|------------|--------------|------------------|")

    # Sort by solve rate descending
    sorted_experiments = sorted(summaries.items(), key=lambda x: x[1].solve_rate, reverse=True)

    for exp_name, summary in sorted_experiments:
        if exp_name == baseline:
            diff_str = "baseline"
        elif exp_name in contributions:
            diff = contributions[exp_name]["solve_rate_pct"]
            diff_str = f"{diff:+.1f}%"
        else:
            diff_str = "N/A"

        lines.append(f"| {exp_name} | {summary.total_tasks} | {summary.solved_tasks} | "
                    f"{summary.solve_rate:.1%} | {summary.avg_accuracy:.1%} | {diff_str} |")

    lines.append("")

    # Component contributions
    lines.append("## Component Contributions")
    lines.append("")
    lines.append("Contribution = Baseline performance - Ablated performance")
    lines.append("(Higher values indicate more important components)")
    lines.append("")

    if contributions:
        lines.append("| Component Removed | Solve Rate Impact | Accuracy Impact |")
        lines.append("|-------------------|-------------------|-----------------|")

        # Sort by solve rate contribution
        sorted_contributions = sorted(contributions.items(),
                                       key=lambda x: x[1]["solve_rate_contribution"],
                                       reverse=True)

        for exp_name, contrib in sorted_contributions:
            lines.append(f"| {exp_name} | {contrib['solve_rate_pct']:+.1f}% | "
                        f"{contrib['accuracy_pct']:+.1f}% |")

    lines.append("")

    # Key findings
    lines.append("## Key Findings")
    lines.append("")

    if contributions:
        # Find most important components
        sorted_by_solve = sorted(contributions.items(),
                                 key=lambda x: x[1]["solve_rate_contribution"],
                                 reverse=True)

        if sorted_by_solve:
            most_important = sorted_by_solve[0]
            lines.append(f"1. **Most important component**: `{most_important[0]}` "
                        f"(removing it decreases solve rate by {most_important[1]['solve_rate_pct']:.1f}%)")

        # Compare heuristics vs feedback
        heuristics_contrib = contributions.get("no_heuristics", {}).get("solve_rate_pct", 0)
        feedback_contrib = contributions.get("no_feedback", {}).get("solve_rate_pct", 0)

        if heuristics_contrib or feedback_contrib:
            lines.append("")
            lines.append("### Heuristics vs Feedback Mechanism")
            lines.append("")
            lines.append(f"- **Feedback mechanism contribution**: {feedback_contrib:.1f}% solve rate")
            lines.append(f"- **Domain heuristics contribution**: {heuristics_contrib:.1f}% solve rate")

            if feedback_contrib > heuristics_contrib:
                lines.append("")
                lines.append(f"*The feedback mechanism contributes more ({feedback_contrib:.1f}%) "
                            f"than domain heuristics ({heuristics_contrib:.1f}%) to overall performance.*")
            else:
                lines.append("")
                lines.append(f"*Domain heuristics contribute more ({heuristics_contrib:.1f}%) "
                            f"than the feedback mechanism ({feedback_contrib:.1f}%) to overall performance.*")

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- **no_heuristics**: Tests pure feedback mechanism without domain knowledge")
    lines.append("- **no_feedback**: Tests domain heuristics without iterative feedback")
    lines.append("- **no_structural_hints**: Tests impact of grid/color pattern analysis")
    lines.append("- **no_adaptive_strategy**: Tests impact of accuracy-based strategy selection")
    lines.append("- **no_accuracy_hints**: Tests impact of accuracy-based prompt hints")
    lines.append("")

    return "\n".join(lines)


def generate_json_report(summaries: Dict[str, ExperimentSummary],
                         contributions: Dict[str, Dict],
                         baseline: str) -> str:
    """Generate JSON report."""
    report = {
        "baseline": baseline,
        "experiments": {},
        "contributions": contributions,
    }

    for exp_name, summary in summaries.items():
        report["experiments"][exp_name] = {
            "total_tasks": summary.total_tasks,
            "solved_tasks": summary.solved_tasks,
            "solve_rate": summary.solve_rate,
            "avg_accuracy": summary.avg_accuracy,
            "avg_iterations": summary.avg_iterations,
            "avg_improvement": summary.avg_improvement,
        }

    return json.dumps(report, indent=2)


def generate_csv_report(summaries: Dict[str, ExperimentSummary],
                        contributions: Dict[str, Dict],
                        baseline: str) -> str:
    """Generate CSV report."""
    lines = []
    lines.append("experiment,total_tasks,solved_tasks,solve_rate,avg_accuracy,solve_rate_diff,accuracy_diff")

    for exp_name, summary in summaries.items():
        if exp_name in contributions:
            solve_diff = contributions[exp_name]["solve_rate_pct"]
            acc_diff = contributions[exp_name]["accuracy_pct"]
        else:
            solve_diff = 0.0
            acc_diff = 0.0

        lines.append(f"{exp_name},{summary.total_tasks},{summary.solved_tasks},"
                    f"{summary.solve_rate:.4f},{summary.avg_accuracy:.4f},"
                    f"{solve_diff:.2f},{acc_diff:.2f}")

    return "\n".join(lines)


def main():
    args = parse_arguments()

    print(f"Loading experiments from: {args.results_dir}", file=sys.stderr)

    # Load all experiments
    summaries = load_all_experiments(args.results_dir)

    if not summaries:
        print("Error: No experiment results found", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(summaries)} experiments", file=sys.stderr)

    # Compute contributions
    contributions = compute_contributions(summaries, args.baseline)

    # Generate report
    if args.output_format == "markdown":
        report = generate_markdown_report(summaries, contributions, args.baseline)
    elif args.output_format == "json":
        report = generate_json_report(summaries, contributions, args.baseline)
    elif args.output_format == "csv":
        report = generate_csv_report(summaries, contributions, args.baseline)
    else:
        report = generate_markdown_report(summaries, contributions, args.baseline)

    # Output report
    if args.output_file:
        with open(args.output_file, 'w') as f:
            f.write(report)
        print(f"Report saved to: {args.output_file}", file=sys.stderr)
    else:
        print(report)


if __name__ == "__main__":
    main()
