"""LLM-as-judge evaluation toolkit.

This package implements the modules described in ``docs/architecture.md``:
- data loading and input templating
- prompt management
- sampling control and caching
- LLM scoring hooks
- metric computation and reporting utilities
"""

__all__ = [
    "data_loader",
    "templates",
    "prompt_manager",
    "sampling",
    "cache",
    "llm_scorer",
    "metrics",
    "reporting",
]
