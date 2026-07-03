"""
Voice Activity Detection (VAD) modülü.
WebRTC (hızlı ön filtre) + Silero (doğru sinir ağı) ikili katmanlı ses algılama.

Silero doğru kullanım notları (bu modülün tasarım gerekçesi):
  * Silero 16 kHz'de TAM 512 örneklik pencereler bekler ve RNN durumunu
    çağrılar arasında taşır. Eski kural (480 örneklik 30 ms frame'i sıfırla
    doldurmak + yalnızca WebRTC'nin geçirdiği frame'leri göstermek) hem her
    pencereyi bozuyor hem de RNN durumunda delikler açıyordu.
  * Bu sürümde HER frame (WebRTC sonucundan bağımsız) anti-alias filtreli
    stateful bir resampler'dan geçirilip 16 kHz FIFO'ya eklenir; FIFO'dan
    tam 512'lik pencereler kesintisiz olarak Silero'ya akar. Böylece RNN
    durumu gerçek, sürekli bir ses akışı görür.
  * 48 kHz → 16 kHz için eski `[::3]` decimation'ı alias'lanmış spektrumu
    Silero'ya yediriyordu; artık pencereli-sinc FIR alçak geçiren filtre +
    kesirli fazlı interpolasyon kullanılır (frame sınırlarında durum korunur).
"""

import numpy as np
import torch
import webrtcvad

from ..config import VAD_AGGRESSIVENESS, SILERO_THRESHOLD

# WebRTC yalnızca bu örnekleme hızlarını destekler; diğer hızlarda WebRTC
# katmanı atlanır (karar tamamen Silero'ya kalır).
_WEBRTC_RATES = (8000, 16000, 32000, 48000)
_SILERO_RATE = 16000
_SILERO_WINDOW = 512  # Silero'nun 16 kHz'de beklediği örnek sayısı


def load_silero_vad():
    """Load Silero VAD through one mockable boundary."""
    model, utils = torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)
    return model, utils


class StreamingResampler:
    """Stateful, anti-alias filtreli mono float32 resampler (src → 16 kHz).

    Frame frame beslenir; FIR filtre kuyruğu ve kesirli okuma fazı çağrılar
    arasında korunur, böylece frame sınırlarında süreksizlik oluşmaz.
    Kaynak hız 16 kHz ise passthrough'tur.
    """

    NUM_TAPS = 129  # pencereli-sinc FIR uzunluğu (tek sayı → simetrik/lineer faz)

    def __init__(self, src_rate: int, dst_rate: int = _SILERO_RATE):
        self.src_rate = int(src_rate)
        self.dst_rate = int(dst_rate)
        self.ratio = self.src_rate / self.dst_rate
        self._passthrough = self.src_rate == self.dst_rate
        if self._passthrough:
            return

        # Alçak geçiren FIR: kesim, hedef Nyquist'in güvenli altında (0.45 * dst).
        cutoff_hz = 0.45 * self.dst_rate
        t = np.arange(self.NUM_TAPS) - (self.NUM_TAPS - 1) / 2.0
        taps = np.sinc(2.0 * cutoff_hz / self.src_rate * t) * np.hamming(self.NUM_TAPS)
        self._taps = (taps / taps.sum()).astype(np.float32)

        self._filt_hist = np.zeros(self.NUM_TAPS - 1, dtype=np.float32)  # filtre kuyruğu
        self._carry = np.zeros(0, dtype=np.float32)  # henüz tüketilmemiş filtreli örnekler
        self._phase = 0.0  # carry içine kesirli okuma konumu

    def reset(self) -> None:
        if self._passthrough:
            return
        self._filt_hist = np.zeros(self.NUM_TAPS - 1, dtype=np.float32)
        self._carry = np.zeros(0, dtype=np.float32)
        self._phase = 0.0

    def process(self, samples: np.ndarray) -> np.ndarray:
        """Bir frame'lik float32 mono örnekleri 16 kHz'e çevirir."""
        samples = np.asarray(samples, dtype=np.float32)
        if self._passthrough or samples.size == 0:
            return samples

        buf = np.concatenate([self._filt_hist, samples])
        filtered = np.convolve(buf, self._taps, mode="valid")  # len == len(samples)
        self._filt_hist = buf[-(self.NUM_TAPS - 1):]

        stream = np.concatenate([self._carry, filtered]) if self._carry.size else filtered
        # Lineer interpolasyon için idx+1 gerekir → son kullanılabilir konum len-2.
        if stream.size < 2:
            self._carry = stream
            return np.zeros(0, dtype=np.float32)

        n_out = int(np.floor((stream.size - 2 - self._phase) / self.ratio)) + 1
        if n_out <= 0:
            self._carry = stream
            return np.zeros(0, dtype=np.float32)

        pos = self._phase + self.ratio * np.arange(n_out)
        idx = pos.astype(np.int64)
        frac = (pos - idx).astype(np.float32)
        out = stream[idx] * (1.0 - frac) + stream[idx + 1] * frac

        next_pos = self._phase + self.ratio * n_out
        keep_from = min(int(np.floor(next_pos)), stream.size - 1)
        self._carry = stream[keep_from:]
        self._phase = next_pos - keep_from
        return out.astype(np.float32)


class VADEngine:
    """
    İkili katmanlı ses algılama motoru.

    Katman 1 - WebRTC: Çok hızlı, kaba filtre (CPU-only, C tabanlı)
    Katman 2 - Silero: Doğru sinir ağı doğrulaması (PyTorch)

    Karar = WebRTC VE Silero (WebRTC desteklemeyen hızlarda yalnız Silero).
    Silero, WebRTC kararından bağımsız olarak HER frame'i görür — RNN durumu
    kesintisiz kalır; WebRTC yalnızca nihai karara oy verir.
    """

    def __init__(self, aggressiveness=None, threshold=None):
        """
        Args:
            aggressiveness: WebRTC agresiflik seviyesi (0-3). None ise config'den alınır.
            threshold: Silero güven eşiği. None ise config'den alınır.
        """
        self.aggressiveness = aggressiveness if aggressiveness is not None else VAD_AGGRESSIVENESS
        self.threshold = threshold if threshold is not None else SILERO_THRESHOLD

        self.webrtc_vad = webrtcvad.Vad(self.aggressiveness)
        self.silero_model, _ = load_silero_vad()

        self._resampler: StreamingResampler | None = None
        self._fifo = np.zeros(0, dtype=np.float32)
        self._last_prob = 0.0

    def reset_stream(self) -> None:
        """Uzun sessizlik sonrası akış durumunu sıfırlar.

        Silero RNN durumu sonsuza dek taşınmamalı; bir chunk sessizlikle
        kapandığında (konuşma arası) çağrılır. Konuşmanın ORTASINDAN kesilen
        (max-süre) chunk'larda ÇAĞRILMAZ — akış devam eder.
        """
        if hasattr(self.silero_model, "reset_states"):
            self.silero_model.reset_states()
        if self._resampler is not None:
            self._resampler.reset()
        self._fifo = np.zeros(0, dtype=np.float32)
        self._last_prob = 0.0

    def _to_mono_float(self, data: bytes, channels: int) -> np.ndarray:
        audio_np = np.frombuffer(data, dtype=np.int16)
        if channels > 1:
            audio_np = audio_np.reshape(-1, channels)
            # Kanal ORTALAMASI — tek kanal seçmek stereo-pan'lı seslerde bir
            # konuşmacıyı tamamen kaybettirebiliyordu.
            mono = audio_np.astype(np.float32).mean(axis=1)
        else:
            mono = audio_np.astype(np.float32)
        return mono / 32768.0

    def _webrtc_vote(self, mono_float: np.ndarray, rate: int) -> bool:
        """WebRTC oyu. Desteklenmeyen hız/frame boyutunda 'geçir' (True) döner."""
        if rate not in _WEBRTC_RATES:
            return True
        expected = int(rate * 30 / 1000)
        if mono_float.shape[0] != expected:
            return True
        pcm = np.clip(mono_float * 32768.0, -32768, 32767).astype(np.int16)
        try:
            return self.webrtc_vad.is_speech(pcm.tobytes(), rate)
        except Exception:
            return True

    def _silero_prob(self, mono_float: np.ndarray, rate: int) -> float:
        """Frame'i 16 kHz akışa ekler, tamamlanan 512'lik pencereleri işler.

        Bu çağrıda hiç pencere tamamlanmadıysa son bilinen olasılık döner
        (30 ms frame ≈ 480 örnek < 512; pencereler frame'lere hizalı değildir).
        """
        if self._resampler is None or self._resampler.src_rate != rate:
            self._resampler = StreamingResampler(rate)
            self._fifo = np.zeros(0, dtype=np.float32)

        resampled = self._resampler.process(mono_float)
        if resampled.size:
            self._fifo = np.concatenate([self._fifo, resampled])

        prob = None
        while self._fifo.size >= _SILERO_WINDOW:
            window = self._fifo[:_SILERO_WINDOW]
            self._fifo = self._fifo[_SILERO_WINDOW:]
            # inference_mode: no_grad'dan daha ucuz (version-counter/autograd
            # kaydı tamamen kapalı) — frame başına çağrılan sıcak yol.
            with torch.inference_mode():
                p = self.silero_model(
                    torch.from_numpy(np.ascontiguousarray(window)), _SILERO_RATE
                ).item()
            prob = p if prob is None else max(prob, p)

        if prob is None:
            return self._last_prob
        self._last_prob = prob
        return prob

    def check_speech(self, data, rate, channels):
        """
        Ses verisinde konuşma olup olmadığını kontrol eder.

        Args:
            data: Ham ses verisi (bytes, int16 formatında)
            rate: Örnekleme hızı (Hz)
            channels: Kanal sayısı

        Returns:
            tuple[bool, float]: (konuşma_var_mı, güven_skoru)
        """
        try:
            mono = self._to_mono_float(data, channels)
            if mono.size == 0:
                return False, 0.0

            # Silero HER frame'i görür (durum sürekliliği); WebRTC paralel oy verir.
            confidence = self._silero_prob(mono, rate)
            webrtc_ok = self._webrtc_vote(mono, rate)

            return (webrtc_ok and confidence > self.threshold), confidence
        except Exception:
            return False, 0.0
