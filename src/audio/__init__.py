"""Audio hardware helpers."""

__all__ = ["auto_detect_device", "VADEngine"]


def __getattr__(name):
    if name == "auto_detect_device":
        from .device import auto_detect_device

        return auto_detect_device
    if name == "VADEngine":
        from .vad import VADEngine

        return VADEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
