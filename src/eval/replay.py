#!/usr/bin/env python3
"""
Faz 2 — Offline replay köprüsü.

AMI Mix-Headset dosyalarını, sistemin GERÇEK canlı işleme yolundan (VADEngine +
RecordingState + _flush_chunk_if_ready + AIWorker) geçirir; ama mikrofon/thread
yerine dosyadan senkron besler. Her toplantı için hipotez çıktıları üretir:

  hyp/rttm/<meeting>.rttm         → DER için (Faz 4)
  hyp/transcripts/<meeting>.json  → cpWER + WER için (Faz 4)
  hyp_summary.json                → süre, RTF (compute), konuşmacı sayısı

Yerleşim
--------
Bu dosyayı paketin içine koyun, ör.  src/eval/replay.py  (yeni 'eval' alt-paketi,
içine boş bir __init__.py ekleyin). Aşağıdaki göreli importlar buna göredir;
farklı yere koyarsanız import yollarını siz ayarlarsınız.

Çalıştırma (paket kökünün BİR ÜSTÜNDEN, modül olarak):
    python -m src.eval.replay
    # veya belirli dizinleri belirterek:
    python -m src.eval.replay --refs ./tests/ami_data/ami_refs --out ./tests/ami_data/ami_hyp
    # tek toplantıda sağlama (Faz 5.18):
    python -m src.eval.replay --only IS1009a

Mantık (pipeline.run() ile birebir)
-----------------------------------
* WAV 30 ms (480 örnek @16k) frame'lere bölünür.
* Her frame VADEngine.check_speech'ten geçer; RecordingState aynı şekilde güncellenir.
* _flush_chunk_if_ready 'final' chunk üretince → AIWorker.process_chunk + run_diarization.
* GLOBAL OFFSET: dosyadan okunan frame sayacı tutulur; bir chunk'ın ilk (konuşma)
  frame'i geldiği an global başlangıç saniyesi kaydedilir ve Whisper'ın chunk-göreli
  zaman damgalarına eklenir. (En kritik adım — bu olmadan DER/cpWER anlamsız çıkar.)
"""

from __future__ import annotations

# Zararsız üçüncü-parti uyarılarını sustur (boş-dilim → nan downstream'de tutuluyor;
# torchcodec/pyannote uyarıları replay'i etkilemiyor). Terminali temiz tutar.
import warnings as _warnings
for _msg in (r".*Mean of empty slice.*", r".*invalid value encountered.*",
             r".*torchcodec.*", r".*degrees of freedom.*", r".*TensorFloat-32.*"):
    _warnings.filterwarnings("ignore", message=_msg)
_warnings.filterwarnings("ignore", category=UserWarning, module="pyannote")

import argparse
import contextlib
import gc
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

# --- Paket içi göreli importlar (src/eval/replay.py varsayımı) ---
from ..config import FRAME_DURATION_MS
from ..audio.vad import VADEngine
from ..core.ai_worker import AIWorker
from .eval_worker import EvalAIWorker
from ..pipeline import RecordingState, _update_recording_state, _flush_chunk_if_ready

TARGET_RATE = 16000   # AMI ihm-mix zaten 16k mono
TARGET_CH = 1


# --------------------------------------------------------------------------- #
# Saf yardımcılar (paketten bağımsız → ayrıca test edilebilir)
# --------------------------------------------------------------------------- #
def load_int16_mono(path: str, rate: int = TARGET_RATE) -> np.ndarray:
    """WAV'ı 16k mono int16 dizisine yükler (gerekirse resample)."""
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio[:, 0]
    if sr != rate:
        import torch
        import torchaudio
        audio = torchaudio.functional.resample(
            torch.from_numpy(audio), sr, rate
        ).numpy()
    return np.clip(audio * 32768.0, -32768, 32767).astype(np.int16)


def normalize_speaker(label: str) -> str:
    """Warm-up etiketlerini tek bir 'CALIBRATING' altında topla.

    Canlı sistem warm-up sırasında '[Calibrating... 30s]', '[Calibrating... 15s]'
    gibi geri-sayımlı etiketler üretir; bunlar farklı string oldukları için
    konuşmacı sayısını şişirir. Hepsini tek etikete indirger.
    """
    if label.startswith("[Calibrating"):
        return "CALIBRATING"
    return label


def prune_tiny_speakers(results: list, min_sec: float) -> list:
    """Tüm toplantıda toplam süresi `min_sec`'in altındaki gerçek konuşmacıları
    (hayalet) eler: her segmentini zaman olarak en yakın 'kalıcı' konuşmacıya
    devreder. Pseudo-etiketler (Unknown/CALIBRATING) dokunulmaz ve konuşmacı
    sayılmaz. Aday kapısından sızan tek-tük kısa hayalet konuşmacıları temizler;
    metni en yakın gerçek kişiye verdiği için cpWER'i Unknown'a atmaktan iyidir.
    """
    if min_sec <= 0:
        return results

    pseudo = {"Unknown", "CALIBRATING"}
    dur: dict[str, float] = {}
    for r in results:
        spk = normalize_speaker(r["speaker"])
        if spk in pseudo:
            continue
        dur[spk] = dur.get(spk, 0.0) + max(0.0, r["end"] - r["start"])

    tiny = {s for s, d in dur.items() if d < min_sec}
    survivors = [s for s in dur if s not in tiny]
    if not tiny or not survivors:
        return results

    import bisect
    surv = sorted(((r["start"] + r["end"]) / 2.0, normalize_speaker(r["speaker"]))
                  for r in results
                  if normalize_speaker(r["speaker"]) in survivors)
    centers = [c for c, _ in surv]

    out = []
    for r in results:
        if normalize_speaker(r["speaker"]) in tiny:
            mid = (r["start"] + r["end"]) / 2.0
            idx = bisect.bisect_left(centers, mid)
            best, best_d = None, None
            for k in (idx - 1, idx):
                if 0 <= k < len(surv):
                    dd = abs(surv[k][0] - mid)
                    if best_d is None or dd < best_d:
                        best_d, best = dd, surv[k][1]
            if best:
                r = {**r, "speaker": best}
        out.append(r)
    return out


def results_to_rttm(meeting_id: str, results: list) -> str:
    """{speaker,start,end} listesini NIST RTTM'e (zaman sıralı)."""
    lines = []
    for r in sorted(results, key=lambda x: x["start"]):
        dur = max(0.0, r["end"] - r["start"])
        if dur <= 0:
            continue
        spk = normalize_speaker(r["speaker"])
        lines.append(
            f"SPEAKER {meeting_id} 1 {r['start']:.3f} {dur:.3f} "
            f"<NA> <NA> {spk} <NA> <NA>"
        )
    return "\n".join(lines) + "\n"


def results_to_speaker_transcript(results: list) -> dict:
    """Konuşmacı-bazlı birleştirilmiş hipotez (cpWER için) + düz metin (WER)."""
    per_spk: dict[str, list[str]] = defaultdict(list)
    flat: list[str] = []
    for r in sorted(results, key=lambda x: x["start"]):
        txt = (r.get("text") or "").strip()
        if not txt:
            continue
        per_spk[normalize_speaker(r["speaker"])].append(txt)
        flat.append(txt)
    return {
        "speakers": {spk: " ".join(parts) for spk, parts in per_spk.items()},
        "flat": " ".join(flat),
    }


class _ListQueue:
    """_flush_chunk_if_ready'nin .put() çağrısını yakalayan minimal kuyruk shim'i."""
    def __init__(self):
        self.items = []

    def put(self, item):
        self.items.append(item)

    def pop_last(self):
        return self.items.pop()


# --------------------------------------------------------------------------- #
def _fmt_dur(sec: float) -> str:
    """Saniyeyi okunur süreye çevir (ETA için)."""
    sec = int(max(0, sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}sa {m:02d}dk"
    if m:
        return f"{m}dk {s:02d}sn"
    return f"{s}sn"


class _NullSink:
    """stdout bastırma hedefi — yazıları yutar. os.devnull (cp1252) worker'ın
    bastığı '→' gibi karakterlerde UnicodeEncodeError veriyordu; bu sink hiç
    encode etmez ve bellekte biriktirmez."""
    def write(self, *_args, **_kwargs):
        return 0

    def flush(self):
        pass


# --------------------------------------------------------------------------- #
# Tek toplantıyı işle
# --------------------------------------------------------------------------- #
def process_meeting(worker: AIWorker, vad: VADEngine, audio_path: str,
                    language: str, progress_cb=None) -> tuple[list, float, float, float]:
    """
    Returns: (global_results, audio_duration_sec, compute_time_sec, vad_speech_sec)
    global_results: [{speaker,start,end,text}, ...]  (mutlak zaman damgalı)
    vad_speech_sec: VAD'in 'konuşma' saydığı toplam süre (kapsama teşhisi için).
    """
    int16 = load_int16_mono(audio_path)
    frame_n = int(TARGET_RATE * FRAME_DURATION_MS / 1000)   # 480
    n_frames = len(int16) // frame_n
    audio_dur = len(int16) / TARGET_RATE

    worker.speaker_tracker.reset()          # önceki toplantı sızmasın
    state = RecordingState()
    shim = _ListQueue()
    global_results: list = []
    pending_offset = 0.0
    speech_frames = 0                        # VAD'in konuşma saydığı frame sayısı

    def _emit_chunk(chunk_bytes: bytes, offset: float):
        out = worker.process_chunk(chunk_bytes, is_final=True, language=language)
        if not out:
            return
        segs = out.get("results", [])
        # SADAKAT: Güncel canlı _worker_loop, diarization'a ÇEVRİLMEMİŞ per-segment
        # sonuçları KELİME ZAMAN DAMGALARIYLA verir (pipeline.py: original_segments).
        # Eskiden burada tüm chunk tek segmente birleştiriliyordu; bu, 10-20 sn'lik
        # bir pencereyi tek konuşmacıya çökertiyordu (AMI'de %80 tek-konuşmacı
        # sızıntısının asıl nedeni). Artık birleştirmiyoruz; run_diarization
        # kelime-seviyesinde konuşmacı sınırından bölme yapar.
        # Çeviri adımını BİLEREK atlıyoruz — AMI İngilizce ASR değerlendirmesinde
        # çeviri olmamalı, yoksa İngilizce referansa karşı WER bozulur.
        diarized = worker.run_diarization(
            out["waveform_16k"], out["sample_rate"],
            out["chunk_duration_ms"], segs,
        )
        for r in diarized:
            global_results.append({
                "speaker": r["speaker"],
                "start": r["start"] + offset,
                "end": r["end"] + offset,
                "text": r["text"],
            })
        # RAM: chunk'a ait ağır tensörleri (waveform vb.) hemen serbest bırak.
        del out, segs, diarized

    t0 = time.perf_counter()
    for idx in range(n_frames):
        if progress_cb is not None and idx % 300 == 0:
            progress_cb(idx / n_frames if n_frames else 1.0)
        data = int16[idx * frame_n:(idx + 1) * frame_n].tobytes()
        is_speech, _ = vad.check_speech(data, TARGET_RATE, TARGET_CH)
        if is_speech:
            speech_frames += 1

        # Yeni chunk'ın ilk konuşma frame'i → global başlangıcı kaydet
        if not state.has_spoken and is_speech:
            pending_offset = idx * FRAME_DURATION_MS / 1000.0

        _update_recording_state(state, data, is_speech)

        if _flush_chunk_if_ready(state, shim):
            chunk = shim.pop_last()
            _emit_chunk(chunk["data"], pending_offset)

    # Dosya sonu: flush olmadan kalan son chunk'ı zorla işle
    if state.has_spoken and state.chunk_buffer:
        _emit_chunk(b"".join(state.chunk_buffer), pending_offset)

    compute_time = time.perf_counter() - t0
    vad_speech_sec = speech_frames * FRAME_DURATION_MS / 1000.0
    return global_results, audio_dur, compute_time, vad_speech_sec


# --------------------------------------------------------------------------- #
def main() -> None:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    default_refs = PROJECT_ROOT / "tests" / "ami_data" / "ami_refs"
    default_out = PROJECT_ROOT / "tests" / "ami_data" / "ami_hyp"

    ap = argparse.ArgumentParser(description="AMI offline replay köprüsü.")
    ap.add_argument("--refs", type=Path, default=default_refs,
                    help="Faz 1 çıktı dizini (meetings.json içerir).")
    ap.add_argument("--out", type=Path, default=default_out,
                    help="Hipotez çıktılarının yazılacağı dizin.")
    ap.add_argument("--lang", default="en", help="Transkripsiyon dili (AMI: en).")
    ap.add_argument("--only", nargs="*", default=None,
                    help="Sadece bu toplantı(lar) işlensin (sağlama için).")
    ap.add_argument("--prune-speaker-sec", type=float, default=3.0,
                    help="Toplam süresi bu eşiğin (sn) altındaki hayalet konuşmacıları "
                         "en yakın gerçek konuşmacıya birleştir. 0 = kapalı.")
    ap.add_argument("--vad-aggressiveness", type=int, default=1, choices=[0, 1, 2, 3],
                    help="WebRTC VAD agresifliği (0=en hassas, 3=en katı). AMI mix-headset "
                         "için canlı varsayılandan (3) daha hassas; .env/canlı etkilenmez.")
    ap.add_argument("--vad-silero-threshold", type=float, default=0.2,
                    help="Silero VAD güven eşiği (düşük = daha hassas).")
    ap.add_argument("--embedding-threshold", type=float, default=0.70,
                    help="Konuşmacı eşleştirme eşiği (eval'e özel; .env/canlı etkilenmez). "
                         "4-toplantı süpürmesinde 0.70 aggregate-optimal bulundu.")
    ap.add_argument("--verbose", action="store_true",
                    help="Chunk-başı worker loglarını göster. Varsayılan: bastırılır "
                         "(temiz ilerleme çubuğu + daha az stdout I/O).")
    args = ap.parse_args()

    meetings_file = args.refs / "meetings.json"
    if not meetings_file.exists():
        # Kullanıcı elle göreli yol geçip bu dizini bulamadıysa tests/ami_data altından eşleştirmeye çalış
        fallback_refs = PROJECT_ROOT / "tests" / "ami_data" / args.refs.name
        if (fallback_refs / "meetings.json").exists():
            print(f"[Info] '{args.refs}' doğrudan bulunamadı. '{fallback_refs}' konumundaki veriler kullanılacak.")
            args.refs = fallback_refs
            meetings_file = args.refs / "meetings.json"
            # Çıktı klasörünü de otomatik yönlendir
            args.out = PROJECT_ROOT / "tests" / "ami_data" / args.out.name
        else:
            raise SystemExit(
                f"Hata: meetings.json bulunamadı: {meetings_file.resolve()}\n"
                f"Lütfen doğru '--refs' dizinini belirtin ya da verilerin tests/ami_data/ami_refs altında olduğundan emin olun."
            )

    meetings = json.loads(meetings_file.read_text(encoding="utf-8"))
    if args.only:
        wanted = set(args.only)
        meetings = [m for m in meetings if m["meeting_id"] in wanted]
        if not meetings:
            raise SystemExit(f"--only ile eşleşen toplantı yok: {args.only}")

    (args.out / "rttm").mkdir(parents=True, exist_ok=True)
    (args.out / "transcripts").mkdir(parents=True, exist_ok=True)

    print(f"[Replay] Referans dizini: {args.refs.resolve()}")
    print(f"[Replay] Çıktı dizini: {args.out.resolve()}")
    print(f"[Replay] Modeller yükleniyor... (EvalAIWorker, "
          f"embedding_threshold={args.embedding_threshold or 'config'})")
    worker = EvalAIWorker(rate=TARGET_RATE, channels=TARGET_CH,
                          embedding_threshold=args.embedding_threshold)
    if not worker.load_models():
        raise SystemExit("AIWorker modelleri yüklenemedi.")
    # Eval'e özel VAD hassasiyeti (canlı .env'i etkilemez). AMI mix-headset'te
    # canlı varsayılanı (aggressiveness=3) konuşmanın yarısını kaçırıyordu.
    vad = VADEngine(aggressiveness=args.vad_aggressiveness,
                    threshold=args.vad_silero_threshold)
    print(f"[Replay] VAD: webrtc_aggr={args.vad_aggressiveness}, "
          f"silero_thr={args.vad_silero_threshold}")

    summary = []
    t_start = time.perf_counter()                         # model yüklemeden SONRA
    total_audio = sum(float(m.get("duration_sec", 0.0)) for m in meetings) or 1.0
    done_audio = 0.0
    n_meet = len(meetings)
    real_out = sys.stdout    # ilerleme çubuğu, stdout redirect'e rağmen terminale yazsın

    for i_m, m in enumerate(meetings, 1):
        mid = m["meeting_id"]

        # Ses dosyasının konumunu çöz (tests/ami_data altındaki yapıya göre)
        audio_path_raw = m["audio_path"]
        audio_path = Path(audio_path_raw)
        if not audio_path.is_absolute():
            # Farklı aday konumları kontrol et
            candidate1 = args.refs.parent / audio_path
            candidate2 = args.refs / audio_path
            candidate3 = PROJECT_ROOT / "tests" / "ami_data" / audio_path

            if candidate1.exists():
                audio_path = candidate1
            elif candidate2.exists():
                audio_path = candidate2
            elif candidate3.exists():
                audio_path = candidate3
            else:
                # Bulunamazsa varsayılan adayı ata (aşağıdaki exists kontrolünde yakalanıp atlanacak)
                audio_path = candidate1

        if not audio_path.exists():
            print(f"[Warning] Ses dosyası bulunamadı, atlanıyor: {audio_path.resolve()}")
            continue

        meeting_audio = float(m.get("duration_sec", 0.0)) or 1.0
        print(f"\n[{i_m}/{n_meet}] {mid} ({meeting_audio / 60:.1f} dk) işleniyor...")

        def _progress(frac, _da=done_audio, _ma=meeting_audio, _mid=mid):
            cur = _da + frac * _ma
            el = time.perf_counter() - t_start
            eta = el * (total_audio - cur) / cur if cur > 0 else 0.0
            filled = int(26 * frac)
            bar = "#" * filled + "-" * (26 - filled)
            real_out.write(f"\r  [{bar}] {_mid} {frac * 100:4.1f}% | "
                           f"genel {cur / total_audio * 100:4.1f}% | "
                           f"geçen {_fmt_dur(el)} | ETA {_fmt_dur(eta)}   ")
            real_out.flush()

        # Worker'ın chunk-başı spam'ini bastır (temiz çubuk + daha az stdout I/O);
        # --verbose ile açılır.
        if args.verbose:
            results, audio_dur, compute_time, vad_speech_sec = process_meeting(
                worker, vad, str(audio_path), args.lang, progress_cb=_progress)
        else:
            with contextlib.redirect_stdout(_NullSink()):
                results, audio_dur, compute_time, vad_speech_sec = process_meeting(
                    worker, vad, str(audio_path), args.lang, progress_cb=_progress)
        real_out.write("\n")
        real_out.flush()

        # Hayalet (çok kısa) konuşmacıları en yakın gerçek konuşmacıya birleştir.
        results = prune_tiny_speakers(results, args.prune_speaker_sec)

        (args.out / "rttm" / f"{mid}.rttm").write_text(
            results_to_rttm(mid, results), encoding="utf-8"
        )
        (args.out / "transcripts" / f"{mid}.json").write_text(
            json.dumps(results_to_speaker_transcript(results),
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # Konuşmacı sayısı: pseudo-etiketler (Unknown, warm-up CALIBRATING)
        # gerçek konuşmacı değil — tally'den çıkar ki sayı dürüst olsun.
        labels = {normalize_speaker(r["speaker"]) for r in results}
        labels -= {"Unknown", "CALIBRATING"}
        n_spk = len(labels)
        rtf = compute_time / audio_dur if audio_dur else 0.0
        # Kapsama teşhisi: VAD'in konuşma saydığı süre vs hyp'e dökülen süre.
        hyp_speech_sec = sum(max(0.0, r["end"] - r["start"]) for r in results)
        summary.append({
            "meeting_id": mid,
            "audio_sec": round(audio_dur, 1),
            "compute_sec": round(compute_time, 1),
            "rtf": round(rtf, 3),
            "hyp_segments": len(results),
            "hyp_speakers": n_spk,
            "vad_speech_sec": round(vad_speech_sec, 1),
            "hyp_speech_sec": round(hyp_speech_sec, 1),
        })
        print(f"   segment={len(results)}  konuşmacı={n_spk}  "
              f"RTF(compute)={rtf:.2f}")
        print(f"   kapsama: VAD konuşma={vad_speech_sec:.0f}s  "
              f"hyp çıktı={hyp_speech_sec:.0f}s  (ses={audio_dur:.0f}s)")

        # İlerleme + RAM: tamamlanan sesi say, GPU/CPU belleğini topla (uzun koşuda OOM önler).
        done_audio += audio_dur
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    (args.out / "hyp_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nTamam. Hipotezler: {args.out.resolve()}")
    print("  - rttm/*.rttm           (DER hipotezi)")
    print("  - transcripts/*.json    (cpWER + WER hipotezi)")
    print("  - hyp_summary.json       (süre / RTF / konuşmacı sayısı)")


if __name__ == "__main__":
    main()
