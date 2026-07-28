"""Formatting helpers for transcription results."""

import re

# format_results'ın ürettiği satırın TERSİ. Biçim ile ayrıştırıcısı yan yana
# duruyor ki biri değişince diğeri unutulmasın — arayüzün iki ayrı yerinde
# (günlük paneli + altyazı overlay'i) aynı satırlar çözülüyor.
_RESULT_LINE_RE = re.compile(
    r"^\[(?P<speaker>.*)\]\s+(?P<start>\d+(?:\.\d+)?)s\s*-\s*"
    r"(?P<end>\d+(?:\.\d+)?)s:\s*(?P<text>.*)$"
)


def parse_result_line(line: str):
    """Biçimlenmiş bir sonuç satırını bileşenlerine ayırır.

    Returns:
        {"speaker", "start", "end", "text"} ya da satır bu biçimde değilse None.
        Konuşmacı etiketindeki iç içe parantez soyulur: warm-up etiketi
        "[Calibrating... 35s]" olarak yazıldığında dış parantez zaten biçimden
        gelir, içteki etiketin parçasıdır.
    """
    match = _RESULT_LINE_RE.match((line or "").strip())
    if not match:
        return None
    speaker = match.group("speaker").strip()
    if speaker.startswith("[") and speaker.endswith("]"):
        speaker = speaker[1:-1].strip()
    return {
        "speaker": speaker,
        "start": float(match.group("start")),
        "end": float(match.group("end")),
        "text": match.group("text").strip(),
    }


def format_results(results, return_str=False):
    """
    Format AI results for terminal output or GUI callbacks.

    Args:
        results: list of dictionaries returned by AIWorker.process_chunk.
        return_str: when True, return the formatted text instead of printing it.
    """
    if not results:
        return "" if return_str else None

    lines = []
    for result in results:
        lines.append(
            f"[{result['speaker']}] {result['start']:.1f}s - "
            f"{result['end']:.1f}s: {result['text']}"
        )

    output = "\n".join(lines)

    if return_str:
        return output

    print("\n" + output + "\n")
    return None
