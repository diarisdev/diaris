from tests.evaluator import TranscriptionEvaluator


def test_normalize_text_lowercases_and_removes_punctuation():
    evaluator = TranscriptionEvaluator()

    assert evaluator.normalize_text(" Hello,   WORLD! ") == "hello world"


def test_evaluate_records_word_error_rate():
    evaluator = TranscriptionEvaluator()

    result = evaluator.evaluate("hello world", "hello word")

    assert result.substitutions == 1
    assert result.insertions == 0
    assert result.deletions == 0
    assert result.wer == 0.5
    assert evaluator.report.successful_samples == 1

