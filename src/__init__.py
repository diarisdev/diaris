"""Audio-process package."""

__all__ = ["run"]


def __getattr__(name):
    if name == "run":
        from .pipeline import run

        return run
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
