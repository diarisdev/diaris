import os
import pytest
from src.translation.engine import (
    get_translation_engine,
    GoogleTranslationEngine,
    DeepLTranslationEngine,
    CTranslate2TranslationEngine
)

def test_google_translation_engine():
    engine = get_translation_engine("google")
    assert isinstance(engine, GoogleTranslationEngine)
    # Perform a quick live test translation if connection permits, else check it doesn't crash
    result = engine.translate("Hello", source_lang="en", target_lang="tr")
    assert isinstance(result, str)
    assert len(result) > 0

def test_deepl_translation_engine_fallback():
    # If no API key is specified, it should fallback to Google Translate
    engine = get_translation_engine("deepl", api_key="")
    assert isinstance(engine, DeepLTranslationEngine)
    result = engine.translate("Hello", source_lang="en", target_lang="tr")
    assert isinstance(result, str)
    assert len(result) > 0

def test_ctranslate2_translation_engine_fallback():
    # Without a valid path, it should fallback to Google Translate
    engine = get_translation_engine("ctranslate2", model_path="invalid_path")
    assert isinstance(engine, CTranslate2TranslationEngine)
    result = engine.translate("Hello", source_lang="en", target_lang="tr")
    assert isinstance(result, str)
    assert len(result) > 0

def test_local_nllb_translation_engine():
    pytest.importorskip("ctranslate2")
    model_path = os.path.join("models", "ctranslate2-nllb-200-distilled-600M")
    if os.path.exists(model_path):
        engine = get_translation_engine("ctranslate2", model_path=model_path)
        assert isinstance(engine, CTranslate2TranslationEngine)
        assert engine.is_nllb is True
        assert engine.tokenizer is not None
        result = engine.translate("Hello", source_lang="en", target_lang="tr")
        assert isinstance(result, str)
        assert len(result) > 0
        print(f"NLLB translation result: {result}")
