"""Small audio helpers that do not depend on hardware or model libraries."""


def calculate_chunk_duration_ms(chunk_bytes: bytes, rate: int, channels: int, sample_width: int = 2) -> float:
    """Return the duration of interleaved PCM audio bytes in milliseconds."""
    if rate <= 0:
        raise ValueError("rate must be greater than zero")
    if channels <= 0:
        raise ValueError("channels must be greater than zero")
    if sample_width <= 0:
        raise ValueError("sample_width must be greater than zero")

    bytes_per_sample_frame = sample_width * channels
    num_samples = len(chunk_bytes) // bytes_per_sample_frame
    return (num_samples / rate) * 1000

