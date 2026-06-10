import urllib.request
import urllib.parse
import json
import os
import logging

logger = logging.getLogger(__name__)

class TranslationEngine:
    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        raise NotImplementedError

class GoogleTranslationEngine(TranslationEngine):
    """
    Lightweight, zero-dependency translation engine using the free Google Translate API.
    """
    def translate(self, text: str, source_lang: str = "en", target_lang: str = "tr") -> str:
        if not text.strip():
            return ""
        try:
            # Normalize language codes (e.g. English -> en, Turkish -> tr)
            sl = source_lang.split("-")[0].lower()
            tl = target_lang.split("-")[0].lower()
            
            url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl={sl}&tl={tl}&dt=t&q={urllib.parse.quote(text)}"
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            
            with urllib.request.urlopen(req, timeout=5) as response:
                data = json.loads(response.read().decode('utf-8'))
                translated = "".join(item[0] for item in data[0] if item and item[0])
                return translated
        except Exception as e:
            logger.error(f"Google translation failed: {e}")
            return f"[Ceviri Hatası: {e}]"

class CTranslate2TranslationEngine(TranslationEngine):
    """
    Local translation engine using CTranslate2 and OPUS-MT / MarianMT models.
    Falls back to Google Translate if models are not loaded.
    """
    def __init__(self, model_path: str = None):
        self.model_path = model_path
        self.translator = None
        self.sp_source = None
        self.sp_target = None
        self.fallback = GoogleTranslationEngine()
        
        if model_path and os.path.exists(model_path):
            try:
                import ctranslate2
                import sentencepiece as spm
                
                # Check for source/target sentencepiece models in the directory
                sp_src_path = os.path.join(model_path, "source.spm")
                sp_tgt_path = os.path.join(model_path, "target.spm")
                
                if os.path.exists(sp_src_path) and os.path.exists(sp_tgt_path):
                    self.translator = ctranslate2.Translator(model_path, device="cpu") # default to CPU for translation
                    self.sp_source = spm.SentencePieceProcessor(model_file=sp_src_path)
                    self.sp_target = spm.SentencePieceProcessor(model_file=sp_tgt_path)
                    logger.info(f"Loaded CTranslate2 translation model from {model_path}")
            except Exception as e:
                logger.error(f"Failed to load CTranslate2 translation model: {e}")

    def translate(self, text: str, source_lang: str = "en", target_lang: str = "tr") -> str:
        if not text.strip():
            return ""
            
        if self.translator is None:
            # Fallback to Google Translate if local model not loaded
            return self.fallback.translate(text, source_lang, target_lang)
            
        try:
            # Tokenize source text
            source_tokens = self.sp_source.encode(text, out_type=str)
            
            # Run translation
            results = self.translator.translate_batch([source_tokens])
            target_tokens = results[0].hypotheses[0]
            
            # Decode target text
            translated = self.sp_target.decode(target_tokens)
            return translated
        except Exception as e:
            logger.error(f"CTranslate2 translation failed: {e}, falling back to Google")
            return self.fallback.translate(text, source_lang, target_lang)

def get_translation_engine(engine_name: str = "google", model_path: str = None) -> TranslationEngine:
    if engine_name.lower() == "ctranslate2":
        return CTranslate2TranslationEngine(model_path)
    return GoogleTranslationEngine()
