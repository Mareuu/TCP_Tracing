"""
TCP Ablation Configuration Module

Provides configuration for ablation studies to separate the contribution of
domain heuristics vs feedback mechanism in the TCP framework.

This module addresses reviewer feedback:
- Reviewer 1: "how much this information [domain heuristics] actually is responsible
  for the performance improvements, rather than the iterative feedback mechanism itself"
- Reviewer 2: "it is hard to attribute gains to the method rather than to evaluation
  or prompt/setup choices"
"""

from dataclasses import dataclass
from typing import Optional
import json


@dataclass
class AblationConfig:
    """
    Configuration class for controlling domain heuristics in ablation studies.

    Attributes:
        enable_structural_hints: Enable analyze_transformation_pattern() hints
        enable_size_change_hints: Enable grid size change hints in prompts
        enable_color_change_hints: Enable color change hints in prompts
        enable_adaptive_strategy: Enable accuracy-based repair strategy selection
        enable_accuracy_hints: Enable "HINT: Almost there!" style hints
        enable_color_mapping: Enable color scheme information in prompts
        feedback_level: Force specific feedback level (-1=adaptive, 0-3=forced)
        temperature_mode: Temperature mode ("adaptive" or "fixed")
        feedback_style: Feedback style ("interpreted" or "raw")
            - "interpreted": Domain-specific feedback with hints and explanations
            - "raw": Pure numerical feedback without domain interpretation
    """

    # Structural Hints Control
    enable_structural_hints: bool = True
    enable_size_change_hints: bool = True
    enable_color_change_hints: bool = True

    # Strategy Engine Control
    enable_adaptive_strategy: bool = True
    enable_accuracy_hints: bool = True

    # Color/Grid Heuristics Control
    enable_color_mapping: bool = True

    # Existing controls (already implemented in tcp_refine.py)
    feedback_level: int = -1  # -1=adaptive, 0=none, 1=summary, 2=detailed, 3=pixel-level
    temperature_mode: str = "adaptive"  # "adaptive" or "fixed"

    # Feedback style control (NEW: for raw feedback ablation)
    feedback_style: str = "interpreted"  # "interpreted" or "raw"

    # Strategy selection mode (NEW: for heuristic-free ablation)
    # - "adaptive_threshold": Use domain-specific accuracy thresholds (original behavior)
    # - "history_based": Use improvement history patterns (domain-agnostic, no thresholds)
    # - "fixed": Always use same strategy (baseline for ablation)
    strategy_mode: str = "adaptive_threshold"

    # Feedback granularity (NEW: for minimal feedback ablation study)
    # Controls how much information is provided in feedback
    # 0=none, 1=binary, 2=accuracy, 3=shape, 4=count, 5=position, 6=full_raw, 7=interpreted
    feedback_granularity: int = 7  # Default: full interpreted feedback

    # Metadata for experiment tracking
    config_name: str = "custom"
    description: str = ""

    def __post_init__(self):
        """Validate configuration values."""
        if self.feedback_level not in [-1, 0, 1, 2, 3]:
            raise ValueError(f"feedback_level must be -1, 0, 1, 2, or 3, got {self.feedback_level}")
        if self.temperature_mode not in ["adaptive", "fixed"]:
            raise ValueError(f"temperature_mode must be 'adaptive' or 'fixed', got {self.temperature_mode}")
        if self.feedback_style not in ["interpreted", "raw"]:
            raise ValueError(f"feedback_style must be 'interpreted' or 'raw', got {self.feedback_style}")
        if self.strategy_mode not in ["adaptive_threshold", "history_based", "fixed"]:
            raise ValueError(f"strategy_mode must be 'adaptive_threshold', 'history_based', or 'fixed', got {self.strategy_mode}")
        if self.feedback_granularity not in range(8):
            raise ValueError(f"feedback_granularity must be 0-7, got {self.feedback_granularity}")

    @classmethod
    def full_system(cls) -> "AblationConfig":
        """Full TCP system with all heuristics enabled (default)."""
        return cls(
            config_name="full_system",
            description="Full TCP system with all domain heuristics and adaptive feedback"
        )

    @classmethod
    def no_heuristics(cls) -> "AblationConfig":
        """
        Feedback-only configuration: disables all domain heuristics.

        This isolates the contribution of the iterative feedback mechanism
        without any domain-specific knowledge injection.
        """
        return cls(
            enable_structural_hints=False,
            enable_size_change_hints=False,
            enable_color_change_hints=False,
            enable_adaptive_strategy=False,
            enable_accuracy_hints=False,
            enable_color_mapping=False,
            config_name="no_heuristics",
            description="Feedback-only: all domain heuristics disabled"
        )

    @classmethod
    def feedback_only(cls) -> "AblationConfig":
        """
        Alias for no_heuristics() - pure feedback mechanism.

        Uses only iterative feedback without strategy adaptation or hints.
        """
        config = cls.no_heuristics()
        config.config_name = "feedback_only"
        config.description = "Pure feedback mechanism without domain heuristics"
        return config

    @classmethod
    def heuristics_only(cls) -> "AblationConfig":
        """
        Minimal feedback configuration: domain heuristics with reduced feedback.

        This isolates the contribution of domain heuristics with minimal
        feedback (summary level only).
        """
        return cls(
            enable_structural_hints=True,
            enable_size_change_hints=True,
            enable_color_change_hints=True,
            enable_adaptive_strategy=True,
            enable_accuracy_hints=True,
            enable_color_mapping=True,
            feedback_level=1,  # Summary level only
            config_name="heuristics_only",
            description="Domain heuristics with minimal (summary) feedback"
        )

    @classmethod
    def no_structural_hints(cls) -> "AblationConfig":
        """Disable structural hints only (grid size/color change analysis)."""
        return cls(
            enable_structural_hints=False,
            enable_size_change_hints=False,
            enable_color_change_hints=False,
            config_name="no_structural_hints",
            description="Structural hints disabled (grid/color analysis)"
        )

    @classmethod
    def no_adaptive_strategy(cls) -> "AblationConfig":
        """Disable adaptive strategy selection based on accuracy."""
        return cls(
            enable_adaptive_strategy=False,
            config_name="no_adaptive_strategy",
            description="Adaptive strategy selection disabled"
        )

    @classmethod
    def no_accuracy_hints(cls) -> "AblationConfig":
        """Disable accuracy-based hints in prompts."""
        return cls(
            enable_accuracy_hints=False,
            config_name="no_accuracy_hints",
            description="Accuracy-based hints disabled"
        )

    @classmethod
    def fixed_temperature(cls) -> "AblationConfig":
        """Use fixed temperature instead of adaptive temperature."""
        return cls(
            temperature_mode="fixed",
            config_name="fixed_temperature",
            description="Fixed temperature (no adaptive adjustment)"
        )

    @classmethod
    def no_feedback(cls) -> "AblationConfig":
        """
        No feedback configuration: heuristics only with no feedback.

        This is the extreme case where only domain heuristics are used
        without any feedback from evaluation.
        """
        return cls(
            enable_structural_hints=True,
            enable_size_change_hints=True,
            enable_color_change_hints=True,
            enable_adaptive_strategy=True,
            enable_accuracy_hints=True,
            enable_color_mapping=True,
            feedback_level=0,  # No feedback
            config_name="no_feedback",
            description="Domain heuristics only, no evaluation feedback"
        )

    @classmethod
    def raw_feedback_only(cls) -> "AblationConfig":
        """
        Raw feedback configuration: pure numerical feedback without domain interpretation.

        This isolates the contribution of the iterative feedback mechanism
        by providing only domain-agnostic numerical metrics:
        - Accuracy (0.0 to 1.0)
        - Shape match (True/False)
        - Error counts and positions

        No domain-specific hints, explanations, or adaptive strategies.
        This answers: "How much does the method work without domain knowledge?"
        """
        return cls(
            enable_structural_hints=False,
            enable_size_change_hints=False,
            enable_color_change_hints=False,
            enable_adaptive_strategy=False,
            enable_accuracy_hints=False,
            enable_color_mapping=False,
            feedback_style="raw",  # Use raw numerical feedback
            strategy_mode="history_based",  # No threshold-based heuristics
            config_name="raw_feedback_only",
            description="Pure numerical feedback without domain interpretation"
        )

    @classmethod
    def heuristic_free(cls) -> "AblationConfig":
        """
        Completely heuristic-free configuration.

        All domain knowledge and heuristics disabled:
        - No structural hints
        - No accuracy-based hints
        - Raw numerical feedback only
        - History-based strategy selection (no accuracy thresholds)
        - Fixed temperature (no adaptive adjustment)

        This is the most domain-agnostic configuration possible.
        It answers: "Does the iterative feedback loop work at all without any heuristics?"
        """
        return cls(
            enable_structural_hints=False,
            enable_size_change_hints=False,
            enable_color_change_hints=False,
            enable_adaptive_strategy=False,
            enable_accuracy_hints=False,
            enable_color_mapping=False,
            feedback_style="raw",
            strategy_mode="history_based",
            temperature_mode="fixed",
            config_name="heuristic_free",
            description="Completely heuristic-free: raw feedback + history-based strategy + fixed temperature"
        )

    @classmethod
    def history_based_strategy(cls) -> "AblationConfig":
        """
        Use history-based strategy selection instead of accuracy thresholds.

        Keeps interpreted feedback but removes accuracy threshold heuristics
        from strategy selection. Strategy is determined by improvement patterns:
        - Stuck for 3+ iterations → complete rewrite
        - Improving → continue current approach
        - Stagnant → targeted fix

        This answers: "Do we need accuracy thresholds for strategy selection?"
        """
        return cls(
            enable_structural_hints=True,
            enable_size_change_hints=True,
            enable_color_change_hints=True,
            enable_adaptive_strategy=True,  # Still use adaptive, but history-based
            enable_accuracy_hints=True,
            enable_color_mapping=True,
            strategy_mode="history_based",
            config_name="history_based_strategy",
            description="History-based strategy selection (no accuracy thresholds)"
        )

    # =========================================================================
    # Feedback Granularity Presets (for minimal feedback ablation study)
    # =========================================================================

    @classmethod
    def feedback_none(cls) -> "AblationConfig":
        """
        Level 0: No feedback - just "your code failed, try again".

        This is the extreme baseline to measure pure iteration effect
        without any information about what went wrong.
        """
        return cls(
            enable_structural_hints=False,
            enable_accuracy_hints=False,
            feedback_style="raw",
            feedback_granularity=0,
            strategy_mode="fixed",  # Can't adapt without info
            config_name="feedback_none",
            description="No feedback (Level 0): pure iteration baseline"
        )

    @classmethod
    def feedback_binary(cls) -> "AblationConfig":
        """
        Level 1: Binary pass/fail feedback only.

        Provides 1 bit of information per example.
        """
        return cls(
            enable_structural_hints=False,
            enable_accuracy_hints=False,
            feedback_style="raw",
            feedback_granularity=1,
            strategy_mode="fixed",
            config_name="feedback_binary",
            description="Binary feedback (Level 1): pass/fail only"
        )

    @classmethod
    def feedback_accuracy(cls) -> "AblationConfig":
        """
        Level 2: Accuracy score only.

        Provides ~7 bits of information (0-100% range).
        """
        return cls(
            enable_structural_hints=False,
            enable_accuracy_hints=False,
            feedback_style="raw",
            feedback_granularity=2,
            strategy_mode="history_based",  # Can use accuracy trend
            config_name="feedback_accuracy",
            description="Accuracy feedback (Level 2): score only"
        )

    @classmethod
    def feedback_shape(cls) -> "AblationConfig":
        """
        Level 3: Accuracy + shape match info.
        """
        return cls(
            enable_structural_hints=False,
            enable_accuracy_hints=False,
            feedback_style="raw",
            feedback_granularity=3,
            strategy_mode="history_based",
            config_name="feedback_shape",
            description="Shape feedback (Level 3): accuracy + shape"
        )

    @classmethod
    def feedback_count(cls) -> "AblationConfig":
        """
        Level 4: Accuracy + shape + error counts.
        """
        return cls(
            enable_structural_hints=False,
            enable_accuracy_hints=False,
            feedback_style="raw",
            feedback_granularity=4,
            strategy_mode="history_based",
            config_name="feedback_count",
            description="Count feedback (Level 4): + error counts"
        )

    @classmethod
    def feedback_position(cls) -> "AblationConfig":
        """
        Level 5: Full raw metrics including error positions.
        """
        return cls(
            enable_structural_hints=False,
            enable_accuracy_hints=False,
            feedback_style="raw",
            feedback_granularity=5,
            strategy_mode="history_based",
            config_name="feedback_position",
            description="Position feedback (Level 5): + error positions"
        )

    @classmethod
    def feedback_full_raw(cls) -> "AblationConfig":
        """
        Level 6: All raw metrics (no interpretation).

        Maximum numerical information without domain knowledge.
        """
        return cls(
            enable_structural_hints=False,
            enable_accuracy_hints=False,
            feedback_style="raw",
            feedback_granularity=6,
            strategy_mode="history_based",
            config_name="feedback_full_raw",
            description="Full raw feedback (Level 6): all metrics, no interpretation"
        )

    @classmethod
    def from_dict(cls, config_dict: dict) -> "AblationConfig":
        """Create config from dictionary."""
        return cls(**config_dict)

    @classmethod
    def from_json(cls, json_path: str) -> "AblationConfig":
        """Load config from JSON file."""
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def from_args(cls, args) -> "AblationConfig":
        """
        Create config from argparse namespace.

        Args:
            args: argparse.Namespace with ablation-related arguments

        Returns:
            AblationConfig instance based on args
        """
        # Check for preset mode first
        ablation_mode = getattr(args, 'ablation_mode', 'full')

        if ablation_mode == 'no_heuristics':
            config = cls.no_heuristics()
        elif ablation_mode == 'feedback_only':
            config = cls.feedback_only()
        elif ablation_mode == 'heuristics_only':
            config = cls.heuristics_only()
        elif ablation_mode == 'no_feedback':
            config = cls.no_feedback()
        elif ablation_mode == 'raw_feedback_only':
            config = cls.raw_feedback_only()
        elif ablation_mode == 'heuristic_free':
            config = cls.heuristic_free()
        elif ablation_mode == 'history_based_strategy':
            config = cls.history_based_strategy()
        # Feedback granularity presets
        elif ablation_mode == 'feedback_none':
            config = cls.feedback_none()
        elif ablation_mode == 'feedback_binary':
            config = cls.feedback_binary()
        elif ablation_mode == 'feedback_accuracy':
            config = cls.feedback_accuracy()
        elif ablation_mode == 'feedback_shape':
            config = cls.feedback_shape()
        elif ablation_mode == 'feedback_count':
            config = cls.feedback_count()
        elif ablation_mode == 'feedback_position':
            config = cls.feedback_position()
        elif ablation_mode == 'feedback_full_raw':
            config = cls.feedback_full_raw()
        else:  # 'full' or 'custom'
            config = cls.full_system()

        # Apply individual overrides
        if getattr(args, 'disable_structural_hints', False):
            config.enable_structural_hints = False
            config.enable_size_change_hints = False
            config.enable_color_change_hints = False

        if getattr(args, 'disable_adaptive_strategy', False):
            config.enable_adaptive_strategy = False

        if getattr(args, 'disable_accuracy_hints', False):
            config.enable_accuracy_hints = False

        if getattr(args, 'disable_color_mapping', False):
            config.enable_color_mapping = False

        # Override feedback_level, temperature_mode, and feedback_style from args
        if hasattr(args, 'feedback_level') and args.feedback_level >= 0:
            config.feedback_level = args.feedback_level

        if hasattr(args, 'temperature_mode'):
            config.temperature_mode = args.temperature_mode

        if hasattr(args, 'feedback_style') and args.feedback_style:
            config.feedback_style = args.feedback_style

        if hasattr(args, 'strategy_mode') and args.strategy_mode:
            config.strategy_mode = args.strategy_mode

        if hasattr(args, 'feedback_granularity') and args.feedback_granularity is not None:
            config.feedback_granularity = args.feedback_granularity

        # Update config name if using custom overrides
        if ablation_mode == 'custom' or any([
            getattr(args, 'disable_structural_hints', False),
            getattr(args, 'disable_adaptive_strategy', False),
            getattr(args, 'disable_accuracy_hints', False),
            getattr(args, 'disable_color_mapping', False),
        ]):
            config.config_name = "custom"
            config.description = cls._generate_description(config)

        return config

    @staticmethod
    def _generate_description(config: "AblationConfig") -> str:
        """Generate description based on enabled/disabled features."""
        disabled = []
        if not config.enable_structural_hints:
            disabled.append("structural_hints")
        if not config.enable_adaptive_strategy:
            disabled.append("adaptive_strategy")
        if not config.enable_accuracy_hints:
            disabled.append("accuracy_hints")
        if not config.enable_color_mapping:
            disabled.append("color_mapping")
        if config.feedback_level == 0:
            disabled.append("feedback")
        elif config.feedback_level > 0:
            disabled.append(f"adaptive_feedback(forced_level={config.feedback_level})")
        if config.temperature_mode == "fixed":
            disabled.append("adaptive_temperature")
        if config.feedback_style == "raw":
            disabled.append("interpreted_feedback(using_raw)")
        if config.strategy_mode == "history_based":
            disabled.append("threshold_strategy(using_history)")
        elif config.strategy_mode == "fixed":
            disabled.append("adaptive_strategy(using_fixed)")

        if disabled:
            return f"Custom config: disabled [{', '.join(disabled)}]"
        return "Custom config: all features enabled"

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            'enable_structural_hints': self.enable_structural_hints,
            'enable_size_change_hints': self.enable_size_change_hints,
            'enable_color_change_hints': self.enable_color_change_hints,
            'enable_adaptive_strategy': self.enable_adaptive_strategy,
            'enable_accuracy_hints': self.enable_accuracy_hints,
            'enable_color_mapping': self.enable_color_mapping,
            'feedback_level': self.feedback_level,
            'temperature_mode': self.temperature_mode,
            'feedback_style': self.feedback_style,
            'strategy_mode': self.strategy_mode,
            'feedback_granularity': self.feedback_granularity,
            'config_name': self.config_name,
            'description': self.description,
        }

    def to_json(self, json_path: str):
        """Save config to JSON file."""
        with open(json_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    def get_cli_args(self) -> list:
        """
        Generate CLI arguments to reproduce this configuration.

        Returns:
            List of CLI argument strings
        """
        args = []

        if not self.enable_structural_hints:
            args.append("--disable_structural_hints")
        if not self.enable_adaptive_strategy:
            args.append("--disable_adaptive_strategy")
        if not self.enable_accuracy_hints:
            args.append("--disable_accuracy_hints")
        if not self.enable_color_mapping:
            args.append("--disable_color_mapping")
        if self.feedback_level >= 0:
            args.append(f"--feedback_level {self.feedback_level}")
        if self.temperature_mode == "fixed":
            args.append("--temperature_mode fixed")
        if self.feedback_style == "raw":
            args.append("--feedback_style raw")
        if self.strategy_mode != "adaptive_threshold":
            args.append(f"--strategy_mode {self.strategy_mode}")
        if self.feedback_granularity != 7:  # 7 is default (interpreted)
            args.append(f"--feedback_granularity {self.feedback_granularity}")

        return args

    def summary(self) -> str:
        """Generate human-readable summary of configuration."""
        lines = [
            f"=== Ablation Config: {self.config_name} ===",
            f"Description: {self.description}",
            "",
            "Domain Heuristics:",
            f"  - Structural hints: {'ON' if self.enable_structural_hints else 'OFF'}",
            f"  - Size change hints: {'ON' if self.enable_size_change_hints else 'OFF'}",
            f"  - Color change hints: {'ON' if self.enable_color_change_hints else 'OFF'}",
            f"  - Adaptive strategy: {'ON' if self.enable_adaptive_strategy else 'OFF'}",
            f"  - Accuracy hints: {'ON' if self.enable_accuracy_hints else 'OFF'}",
            f"  - Color mapping: {'ON' if self.enable_color_mapping else 'OFF'}",
            "",
            "Feedback Settings:",
            f"  - Feedback level: {self.feedback_level} ({'adaptive' if self.feedback_level == -1 else ['none', 'summary', 'detailed', 'pixel-level'][self.feedback_level]})",
            f"  - Feedback style: {self.feedback_style}",
            f"  - Feedback granularity: {self.feedback_granularity} ({['none', 'binary', 'accuracy', 'shape', 'count', 'position', 'full_raw', 'interpreted'][self.feedback_granularity]})",
            f"  - Temperature mode: {self.temperature_mode}",
            f"  - Strategy mode: {self.strategy_mode}",
        ]
        return "\n".join(lines)


# Predefined experiment configurations for ablation study
ABLATION_EXPERIMENTS = {
    # Main ablation modes
    "full_system": AblationConfig.full_system,
    "no_heuristics": AblationConfig.no_heuristics,
    "feedback_only": AblationConfig.feedback_only,
    "heuristics_only": AblationConfig.heuristics_only,
    "no_feedback": AblationConfig.no_feedback,
    "raw_feedback_only": AblationConfig.raw_feedback_only,
    "heuristic_free": AblationConfig.heuristic_free,
    "history_based_strategy": AblationConfig.history_based_strategy,
    # Component ablations
    "no_structural_hints": AblationConfig.no_structural_hints,
    "no_adaptive_strategy": AblationConfig.no_adaptive_strategy,
    "no_accuracy_hints": AblationConfig.no_accuracy_hints,
    "fixed_temperature": AblationConfig.fixed_temperature,
    # Feedback granularity ablation ladder
    "feedback_none": AblationConfig.feedback_none,
    "feedback_binary": AblationConfig.feedback_binary,
    "feedback_accuracy": AblationConfig.feedback_accuracy,
    "feedback_shape": AblationConfig.feedback_shape,
    "feedback_count": AblationConfig.feedback_count,
    "feedback_position": AblationConfig.feedback_position,
    "feedback_full_raw": AblationConfig.feedback_full_raw,
}


def get_ablation_config(name: str) -> AblationConfig:
    """
    Get predefined ablation configuration by name.

    Args:
        name: Name of the ablation configuration

    Returns:
        AblationConfig instance

    Raises:
        ValueError: If configuration name is not found
    """
    if name not in ABLATION_EXPERIMENTS:
        available = ", ".join(ABLATION_EXPERIMENTS.keys())
        raise ValueError(f"Unknown ablation config '{name}'. Available: {available}")
    return ABLATION_EXPERIMENTS[name]()


def list_ablation_configs() -> list:
    """Return list of available ablation configuration names."""
    return list(ABLATION_EXPERIMENTS.keys())
