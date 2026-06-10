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

class DeepLTranslationEngine(TranslationEngine):
    """
    Online translation engine using DeepL API (Free/Pro).
    Falls back to Google Translate if key is missing or calls fail.
    """
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("DEEPL_API_KEY")
        self.url = "https://api-free.deepl.com/v2/translate"
        self.fallback = GoogleTranslationEngine()

    def translate(self, text: str, source_lang: str = "en", target_lang: str = "tr") -> str:
        if not text.strip():
            return ""
        if not self.api_key:
            return self.fallback.translate(text, source_lang, target_lang)
        try:
            sl = source_lang.split("-")[0].upper()
            tl = target_lang.split("-")[0].upper()
            
            # Check if it's pro or free key (pro keys don't end with :fx)
            if not self.api_key.endswith(":fx"):
                self.url = "https://api.deepl.com/v2/translate"
                
            data = urllib.parse.urlencode({
                "text": text,
                "source_lang": sl,
                "target_lang": tl
            }).encode("utf-8")
            
            req = urllib.request.Request(
                self.url,
                data=data,
                headers={
                    "Authorization": f"DeepL-Auth-Key {self.api_key}",
                    "Content-Type": "application/x-www-form-urlencoded"
                },
                method="POST"
            )
            
            with urllib.request.urlopen(req, timeout=5) as response:
                res_data = json.loads(response.read().decode("utf-8"))
                translations = res_data.get("translations", [])
                if translations:
                    return translations[0].get("text", "").strip()
                return ""
        except Exception as e:
            logger.error(f"DeepL translation failed: {e}, falling back to Google")
            return self.fallback.translate(text, source_lang, target_lang)


class CTranslate2TranslationEngine(TranslationEngine):
    """
    Local translation engine using CTranslate2 and NLLB-200 model.
    Falls back to Google Translate if model is not loaded.
    """
    def __init__(self, model_path: str = None):
        self.model_path = model_path
        self.translator = None
        self.tokenizer = None
        self.sp_source = None
        self.tokenizer_type = None
        self.is_nllb = True
        self.fallback = GoogleTranslationEngine()
        
        self.lang_map = {
            "en": "eng_Latn",
            "tr": "tur_Latn",
            "de": "deu_Latn",
            "fr": "fra_Latn",
            "es": "spa_Latn",
            "it": "ita_Latn",
            "pt": "por_Latn",
            "ru": "rus_Cyrl",
            "zh": "zho_Hans",
            "ar": "ary_Arab"
        }
        
        if model_path and os.path.exists(model_path):
            try:
                import ctranslate2
                import sentencepiece as spm
                
                # First choice: tokenizer.json (HF tokenizers library)
                json_path = os.path.join(model_path, "tokenizer.json")
                sp_model_files = [f for f in os.listdir(model_path) if f.endswith(".model") or "sentencepiece" in f.lower()]
                
                if os.path.exists(json_path):
                    try:
                        from tokenizers import Tokenizer
                        self.translator = ctranslate2.Translator(model_path, device="cpu")
                        self.tokenizer = Tokenizer.from_file(json_path)
                        self.tokenizer_type = "tokenizers"
                        logger.info(f"Loaded CTranslate2 NLLB-200 model from {model_path} using tokenizers library")
                    except Exception as e:
                        logger.error(f"Failed to load tokenizer.json with tokenizers library: {e}")
                
                if self.tokenizer_type is None and sp_model_files:
                    try:
                        sp_model_path = os.path.join(model_path, sp_model_files[0])
                        self.translator = ctranslate2.Translator(model_path, device="cpu")
                        self.sp_source = spm.SentencePieceProcessor(model_file=sp_model_path)
                        self.tokenizer_type = "sentencepiece"
                        logger.info(f"Loaded CTranslate2 NLLB-200 model from {model_path} with SentencePiece tokenizer {sp_model_files[0]}")
                    except Exception as e:
                        logger.error(f"Failed to load SentencePiece model: {e}")
                
                if self.tokenizer_type is None:
                    # Try to load tokenizer using transformers
                    try:
                        from transformers import AutoTokenizer
                        self.translator = ctranslate2.Translator(model_path, device="cpu")
                        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                        self.tokenizer_type = "transformers"
                        logger.info(f"Loaded CTranslate2 NLLB-200 model from {model_path} using transformers AutoTokenizer")
                    except Exception as e:
                        logger.error(f"Failed to find tokenizer.json, SentencePiece file or transformers AutoTokenizer for NLLB: {e}")
            except Exception as e:
                logger.error(f"Failed to load CTranslate2 translation model: {e}")

    def translate(self, text: str, source_lang: str = "en", target_lang: str = "tr") -> str:
        if not text.strip():
            return ""
            
        if self.translator is None:
            return self.fallback.translate(text, source_lang, target_lang)
            
        try:
            sl = source_lang.split("-")[0].lower()
            tl = target_lang.split("-")[0].lower()
            src_code = self.lang_map.get(sl, "eng_Latn")
            tgt_code = self.lang_map.get(tl, "tur_Latn")
            
            if self.tokenizer_type == "tokenizers" and self.tokenizer is not None:
                # Using tokenizers Tokenizer
                encoded = self.tokenizer.encode(text)
                source_tokens = encoded.tokens
                if source_tokens and source_tokens[0] != src_code:
                    source_tokens = [src_code] + source_tokens
                if source_tokens and source_tokens[-1] != "</s>":
                    source_tokens = source_tokens + ["</s>"]
                    
                results = self.translator.translate_batch([source_tokens], target_prefix=[[tgt_code]])
                target_tokens = results[0].hypotheses[0]
                if target_tokens and target_tokens[0] == tgt_code:
                    target_tokens = target_tokens[1:]
                if target_tokens and target_tokens[-1] == "</s>":
                    target_tokens = target_tokens[:-1]
                    
                ids = [self.tokenizer.token_to_id(t) for t in target_tokens]
                ids = [i for i in ids if i is not None]
                translated = self.tokenizer.decode(ids)
                return translated
            elif self.tokenizer_type == "transformers" and self.tokenizer is not None:
                # Using transformers AutoTokenizer
                self.tokenizer.src_lang = src_code
                source_tokens = self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(text))
                if source_tokens and source_tokens[0] != src_code:
                    source_tokens = [src_code] + source_tokens
                if source_tokens and source_tokens[-1] != "</s>":
                    source_tokens = source_tokens + ["</s>"]
                    
                results = self.translator.translate_batch([source_tokens], target_prefix=[[tgt_code]])
                target_tokens = results[0].hypotheses[0]
                if target_tokens and target_tokens[0] == tgt_code:
                    target_tokens = target_tokens[1:]
                translated = self.tokenizer.decode(self.tokenizer.convert_tokens_to_ids(target_tokens), skip_special_tokens=True)
                return translated
            elif self.tokenizer_type == "sentencepiece" and self.sp_source is not None:
                # Using pure sentencepiece
                tokens = self.sp_source.encode(text, out_type=str)
                source_tokens = [src_code] + tokens + ["</s>"]
                results = self.translator.translate_batch([source_tokens], target_prefix=[[tgt_code]])
                target_tokens = results[0].hypotheses[0]
                if target_tokens and target_tokens[0] == tgt_code:
                    target_tokens = target_tokens[1:]
                if target_tokens and target_tokens[-1] == "</s>":
                    target_tokens = target_tokens[:-1]
                translated = self.sp_source.decode(target_tokens)
                return translated
            else:
                raise ValueError("No tokenizer loaded for NLLB model")
        except Exception as e:
            logger.error(f"CTranslate2 translation failed: {e}, falling back to Google")
            return self.fallback.translate(text, source_lang, target_lang)

def get_translation_engine(engine_name: str = "google", **kwargs) -> TranslationEngine:
    name = engine_name.lower()
    if name == "ctranslate2":
        return CTranslate2TranslationEngine(kwargs.get("model_path"))
    elif name == "deepl":
        return DeepLTranslationEngine(kwargs.get("api_key"))
    return GoogleTranslationEngine()
