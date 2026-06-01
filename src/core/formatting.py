"""Formatting helpers for transcription results."""


def format_results(results, return_str=False):
    """
    Format AI results for terminal output or GUI callbacks.

    Args:
        results: list of dictionaries returned by AIWorker.process_chunk.
        return_str: when True, return the formatted text instead of printing it.
    """
    if not results:
        return "" if return_str else None

    lines = ["-" * 50]
    for result in results:
        lines.append(
            f"[{result['speaker']}] {result['start']:.1f}s - "
            f"{result['end']:.1f}s: {result['text']}"
        )
    lines.append("-" * 50)

    output = "\n".join(lines)

    if return_str:
        return output

    print("\n" + output + "\n")
    return None
