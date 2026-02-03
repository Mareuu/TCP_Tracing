"""
TCP Domain Abstraction Layer

This module provides the domain adapter framework for TCP, enabling the
iterative refinement methodology to work across different problem domains.

Supported domains:
- ARC (Abstraction and Reasoning Corpus): Visual grid transformation tasks
- HumanEval: Code generation with test-based evaluation

Usage:
    from tcp_core.domains import get_domain_adapter, list_domains

    # Get an adapter for ARC
    adapter = get_domain_adapter('arc')
    problems = adapter.load_dataset('/path/to/arc')

    # Evaluate code
    result = adapter.evaluate(code, problem)

    # Get feedback
    raw_feedback = adapter.get_raw_feedback_string(result.feedback)
    full_feedback = adapter.get_interpreted_feedback_string(result.feedback)
"""

from typing import Dict, List, Optional, Type

from .base import (
    DomainAdapter,
    Problem,
    Feedback,
    EvaluationResult,
    RawMetrics,
    EvaluationStatus,
)
from .arc_domain import ARCDomainAdapter
from .humaneval_domain import HumanEvalDomainAdapter


# Domain registry
_DOMAIN_REGISTRY: Dict[str, Type[DomainAdapter]] = {
    'arc': ARCDomainAdapter,
    'humaneval': HumanEvalDomainAdapter,
}


def register_domain(name: str, adapter_class: Type[DomainAdapter]) -> None:
    """
    Register a new domain adapter.

    Args:
        name: Name to register the adapter under
        adapter_class: DomainAdapter subclass to register
    """
    if not issubclass(adapter_class, DomainAdapter):
        raise TypeError(f"{adapter_class} must be a subclass of DomainAdapter")
    _DOMAIN_REGISTRY[name.lower()] = adapter_class


def get_domain_adapter(name: str, **kwargs) -> DomainAdapter:
    """
    Get a domain adapter by name.

    Args:
        name: Name of the domain ('arc', 'humaneval', etc.)
        **kwargs: Additional arguments to pass to the adapter constructor

    Returns:
        Instantiated DomainAdapter

    Raises:
        ValueError: If domain name is not registered
    """
    name_lower = name.lower()
    if name_lower not in _DOMAIN_REGISTRY:
        available = ', '.join(_DOMAIN_REGISTRY.keys())
        raise ValueError(f"Unknown domain '{name}'. Available: {available}")

    adapter_class = _DOMAIN_REGISTRY[name_lower]
    return adapter_class(**kwargs)


def list_domains() -> List[str]:
    """
    List all registered domain names.

    Returns:
        List of domain name strings
    """
    return list(_DOMAIN_REGISTRY.keys())


def get_domain_info() -> Dict[str, str]:
    """
    Get information about all registered domains.

    Returns:
        Dictionary mapping domain names to their adapter class names
    """
    return {name: cls.__name__ for name, cls in _DOMAIN_REGISTRY.items()}


__all__ = [
    # Base interfaces
    'DomainAdapter',
    'Problem',
    'Feedback',
    'EvaluationResult',
    'RawMetrics',
    'EvaluationStatus',

    # Domain adapters
    'ARCDomainAdapter',
    'HumanEvalDomainAdapter',

    # Registry functions
    'register_domain',
    'get_domain_adapter',
    'list_domains',
    'get_domain_info',
]
