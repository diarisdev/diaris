import urllib.request
import urllib.parse
import json
import os
import logging

logger = logging.getLogger(__name__)


def is_same_language(source_lang: str, target_lang: str) -> bool:
    """Kaynak ve hedef dil aynı mı? (bölge kodu yok sayılır: en-US == en)"""
    if not source_lang or not target_lang:
        return False
    return source_lang.split("-")[0].lower() == target_lang.split("-")[0].lower()


class TranslationEngine:
    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        raise NotImplementedError

    def translate_many(self, texts, source_lang: str = "en", target_lang: str = "tr"):
        """
        Birden çok metni TEK çağrıda çevirir (mümkünse). Canlı diarization'da
        bir chunk birden çok konuşmacı-segmentine bölününce, her segmenti ayrı
        ayrı çevirmek (özellikle yerel CPU NLLB ile) sistemi dondurur. Bu metot
        çağrı sayısını 1'e indirir.

        Varsayılan: metinleri satır sonuyla birleştirip tek çeviri yapar, sonra
        geri böler. Satır sayısı tutmazsa (model satırları birleştirmiş olabilir)
        güvenli biçimde tek tek çevirmeye düşer.
        """
        texts = list(texts)
        if not texts:
            return []
        # Kaynak = hedef → çeviri yok, metinler aynen döner (maliyetsiz)
        if is_same_language(source_lang, target_lang):
            return [t or "" for t in texts]
        idx = [i for i, t in enumerate(texts) if t and t.strip()]
        if not idx:
            return ["" for _ in texts]

        SEP = "\n"
        joined = SEP.join(texts[i].strip() for i in idx)
        translated = self.translate(joined, source_lang, target_lang)
        parts = translated.split(SEP)

        out = ["" for _ in texts]
        if len(parts) == len(idx):
            for k, i in enumerate(idx):
                out[i] = parts[k].strip()
            return out

        # Satır sayısı tutmadı → güvenli tek-tek çeviri
        for i in idx:
            out[i] = self.translate(texts[i], source_lang, target_lang)
        return out

class GoogleTranslationEngine(TranslationEngine):
    """
    Lightweight, zero-dependency translation engine using the free Google Translate API.
    """
    def translate(self, text: str, source_lang: str = "en", target_lang: str = "tr") -> str:
        if not text.strip():
            return ""
        if is_same_language(source_lang, target_lang):
            return text
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
        if is_same_language(source_lang, target_lang):
            return text
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
        
        # NLLB-200 FLORES dil kodları. UI'daki TÜM dilleri kapsamalı; aksi halde
        # eksik dil sessizce yanlış koda düşer (eski hata: ja/ko/nl yoktu →
        # hedef Türkçeye, kaynak İngilizceye düşüyordu). Burada olmayan diller
        # otomatik Google'a yönlendirilir (aşağıdaki _supports kontrolü).
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
            "ar": "arb_Arab",   # standart Arapça (eski ary_Arab = Faslı Arapça idi)
            "ja": "jpn_Jpan",   # Japonca (UI'da vardı, map'te yoktu)
            "ko": "kor_Hang",   # Korece (UI'da vardı, map'te yoktu)
            "nl": "nld_Latn",   # Felemenkçe (UI'da vardı, map'te yoktu)
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

    def _supports(self, source_lang: str, target_lang: str) -> bool:
        """Hem kaynak hem hedef NLLB lang_map'te var mı?"""
        sl = source_lang.split("-")[0].lower()
        tl = target_lang.split("-")[0].lower()
        return sl in self.lang_map and tl in self.lang_map

    def _encode_source(self, text: str, src_code: str):
        """Metni NLLB kaynak token dizisine çevirir (tokenizer tipinden bağımsız)."""
        if self.tokenizer_type == "tokenizers" and self.tokenizer is not None:
            source_tokens = self.tokenizer.encode(text).tokens
        elif self.tokenizer_type == "transformers" and self.tokenizer is not None:
            self.tokenizer.src_lang = src_code
            source_tokens = self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(text))
        elif self.tokenizer_type == "sentencepiece" and self.sp_source is not None:
            source_tokens = self.sp_source.encode(text, out_type=str)
        else:
            raise ValueError("No tokenizer loaded for NLLB model")

        if not source_tokens or source_tokens[0] != src_code:
            source_tokens = [src_code] + source_tokens
        if not source_tokens or source_tokens[-1] != "</s>":
            source_tokens = source_tokens + ["</s>"]
        return source_tokens

    def _decode_target(self, target_tokens, tgt_code: str) -> str:
        """NLLB hedef token dizisini metne çevirir."""
        if target_tokens and target_tokens[0] == tgt_code:
            target_tokens = target_tokens[1:]
        if target_tokens and target_tokens[-1] == "</s>":
            target_tokens = target_tokens[:-1]

        if self.tokenizer_type == "tokenizers":
            ids = [self.tokenizer.token_to_id(t) for t in target_tokens]
            ids = [i for i in ids if i is not None]
            return self.tokenizer.decode(ids)
        elif self.tokenizer_type == "transformers":
            return self.tokenizer.decode(
                self.tokenizer.convert_tokens_to_ids(target_tokens),
                skip_special_tokens=True,
            )
        else:  # sentencepiece
            return self.sp_source.decode(target_tokens)

    def translate(self, text: str, source_lang: str = "en", target_lang: str = "tr") -> str:
        if not text.strip():
            return ""
        if is_same_language(source_lang, target_lang):
            return text

        if self.translator is None:
            return self.fallback.translate(text, source_lang, target_lang)

        # NLLB desteklemiyorsa sessizce yanlış çevirmek yerine Google'a düş
        if not self._supports(source_lang, target_lang):
            logger.warning(
                "NLLB '%s->%s' desteklemiyor, Google'a düşülüyor", source_lang, target_lang
            )
            return self.fallback.translate(text, source_lang, target_lang)

        try:
            sl = source_lang.split("-")[0].lower()
            tl = target_lang.split("-")[0].lower()
            src_code = self.lang_map.get(sl, "eng_Latn")
            tgt_code = self.lang_map.get(tl, "tur_Latn")

            source_tokens = self._encode_source(text, src_code)
            results = self.translator.translate_batch([source_tokens], target_prefix=[[tgt_code]])
            return self._decode_target(results[0].hypotheses[0], tgt_code)
        except Exception as e:
            logger.error(f"CTranslate2 translation failed: {e}, falling back to Google")
            return self.fallback.translate(text, source_lang, target_lang)

    def translate_many(self, texts, source_lang: str = "en", target_lang: str = "tr"):
        """Tüm metinleri TEK translate_batch çağrısında çevirir (CPU dostu)."""
        texts = list(texts)
        if not texts:
            return []
        # Kaynak = hedef → çeviri yok (maliyetsiz)
        if is_same_language(source_lang, target_lang):
            return [t or "" for t in texts]
        if self.translator is None:
            return [self.translate(t, source_lang, target_lang) for t in texts]

        # NLLB desteklemiyorsa Google'a düş (sessiz yanlış çeviri yerine)
        if not self._supports(source_lang, target_lang):
            logger.warning(
                "NLLB '%s->%s' desteklemiyor, Google'a düşülüyor", source_lang, target_lang
            )
            return self.fallback.translate_many(texts, source_lang, target_lang)

        idx = [i for i, t in enumerate(texts) if t and t.strip()]
        if not idx:
            return ["" for _ in texts]

        try:
            sl = source_lang.split("-")[0].lower()
            tl = target_lang.split("-")[0].lower()
            src_code = self.lang_map.get(sl, "eng_Latn")
            tgt_code = self.lang_map.get(tl, "tur_Latn")

            sources = [self._encode_source(texts[i], src_code) for i in idx]
            results = self.translator.translate_batch(
                sources, target_prefix=[[tgt_code]] * len(sources)
            )
            out = ["" for _ in texts]
            for k, i in enumerate(idx):
                out[i] = self._decode_target(results[k].hypotheses[0], tgt_code)
            return out
        except Exception as e:
            logger.error(f"CTranslate2 batch translation failed: {e}, falling back")
            return [self.translate(t, source_lang, target_lang) for t in texts]

def get_translation_engine(engine_name: str = "google", **kwargs) -> TranslationEngine:
    name = engine_name.lower()
    if name == "ctranslate2":
        return CTranslate2TranslationEngine(kwargs.get("model_path"))
    elif name == "deepl":
        return DeepLTranslationEngine(kwargs.get("api_key"))
    return GoogleTranslationEngine()
