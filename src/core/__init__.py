"""AI engine helpers."""

from .formatting import format_results

__all__ = ["AIWorker", "format_results"]


def __getattr__(name):
    if name == "AIWorker":
        from .ai_worker import AIWorker

        return AIWorker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
