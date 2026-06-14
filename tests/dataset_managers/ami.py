"""AMI Meeting Corpus (küçük alt küme) — diarization benchmark.

İki meeting indirir (audio + RTTM referans) ve DER testleri için hazırlar.
RTTM'ler BUTSpeechFIT/AMI-diarization-setup reposundan alınır.
"""

from __future__ import annotations

import json
import os
import shutil
from collections import defaultdict

from .base import DATASETS_ROOT, _download_with_progress, sf


class AmiDiarizationManager:
    def __init__(self, data_dir=None):
        # data_dir = datasets/ami ; meetings + setup repo bunun altında
        self.data_dir = data_dir or os.path.join(DATASETS_ROOT, "ami")
        self.ami_dir = os.path.join(self.data_dir, "AMI_mini")
        self.meetings = ["EN2001a", "ES2002a"]

    def is_downloaded(self):
        if not os.path.isdir(self.ami_dir):
            return False
        for meeting in self.meetings:
            wav_path = os.path.join(self.ami_dir, f"{meeting}.Mix-Headset.wav")
            rttm_path = os.path.join(self.ami_dir, f"{meeting}.rttm")
            if not os.path.exists(wav_path) or not os.path.exists(rttm_path):
                return False
        return True

    def download(self, force=False):
        if self.is_downloaded() and not force:
            print("✅ AMI verileri zaten mevcut.")
            return True

        os.makedirs(self.ami_dir, exist_ok=True)
        print("📥 AMI Diarization veri seti hazırlanıyor...")

        setup_repo_dir = os.path.join(self.data_dir, "AMI-diarization-setup")
        repo_rttm_dir = os.path.join(setup_repo_dir, "only_words", "rttms", "train")

        if not os.path.exists(setup_repo_dir):
            print("   📥 AMI-diarization-setup reposu klonlanıyor...")
            import subprocess
            try:
                subprocess.run(
                    ["git", "clone",
                     "https://github.com/BUTSpeechFIT/AMI-diarization-setup.git",
                     setup_repo_dir],
                    check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
            except Exception as e:
                print(f"❌ Repo klonlama hatası: {e}")
                return False

        for meeting in self.meetings:
            wav_path = os.path.join(self.ami_dir, f"{meeting}.Mix-Headset.wav")
            rttm_path = os.path.join(self.ami_dir, f"{meeting}.rttm")
            if not os.path.exists(wav_path) or force:
                url = (f"https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/"
                       f"{meeting}/audio/{meeting}.Mix-Headset.wav")
                print(f"   İndiriliyor: {meeting} audio...")
                try:
                    _download_with_progress(url, wav_path)
                except Exception as e:
                    print(f"❌ Ses indirme hatası ({meeting}): {e}")
                    return False
            if not os.path.exists(rttm_path) or force:
                src_rttm = os.path.join(repo_rttm_dir, f"{meeting}.rttm")
                if os.path.exists(src_rttm):
                    shutil.copy2(src_rttm, rttm_path)
                else:
                    print(f"❌ RTTM bulunamadı: {src_rttm}")
                    return False

        print("✅ AMI veri seti başarıyla hazırlandı.")
        return True

    def get_samples(self):
        if not self.is_downloaded():
            print("⚠️ AMI verisi bulunamadı. Önce download() çağırın.")
            return []
        if sf is None:
            raise ImportError("AMI örnekleri için 'soundfile' gerekli: pip install soundfile")

        samples = []
        for meeting in self.meetings:
            wav_path = os.path.join(self.ami_dir, f"{meeting}.Mix-Headset.wav")
            rttm_path = os.path.join(self.ami_dir, f"{meeting}.rttm")
            try:
                duration = sf.info(wav_path).duration
            except Exception:
                duration = 0.0
            samples.append({
                "audio_path": wav_path,
                "rttm_path": rttm_path,
                "meeting_id": meeting,
                "duration": duration,
                "annotations": self._parse_rttm(rttm_path),
            })
        return samples

    def _parse_rttm(self, rttm_path):
        annotations = []
        with open(rttm_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 8 and parts[0] == "SPEAKER":
                    start_time = float(parts[3])
                    duration = float(parts[4])
                    annotations.append({
                        "start": start_time,
                        "end": start_time + duration,
                        "speaker": parts[7],
                    })
        return annotations


# --------------------------------------------------------------------------- #
# AMI referans hazırlama (replay/score pipeline'ı için ami_refs/ üretir)
# --------------------------------------------------------------------------- #
def _supervisions_to_rttm(file_id: str, segments: list) -> str:
    """SupervisionSegment listesini NIST RTTM'e (zaman sıralı)."""
    lines = []
    for s in sorted(segments, key=lambda x: x.start):
        lines.append(
            f"SPEAKER {file_id} 1 {s.start:.3f} {s.duration:.3f} "
            f"<NA> <NA> {s.speaker} <NA> <NA>"
        )
    return "\n".join(lines) + "\n"


def _supervisions_to_speaker_transcript(segments: list) -> dict:
    """Konuşmacı-bazlı birleştirilmiş referans (cpWER için) -> {speaker: text}."""
    per_spk: dict[str, list] = defaultdict(list)
    for s in sorted(segments, key=lambda x: x.start):
        if s.text:
            per_spk[s.speaker].append(s.text.strip())
    return {spk: " ".join(parts) for spk, parts in per_spk.items()}


class AmiReferenceManager:
    """AMI ihm-mix (Mix-Headset) test referanslarını lhotse ile indirir/üretir.

    `ami_replay` ve `ami_score` runner'larının işlediği veriyi hazırlar:
        datasets/ami/ami_refs/rttm/<meeting>.rttm          (DER referansı)
        datasets/ami/ami_refs/transcripts/<meeting>.json   (cpWER + WER referansı)
        datasets/ami/ami_refs/meetings.json                (ses yolları + meta)

    Ham AMI sesi datasets/ami/ami_corpus/ altına iner (birkaç GB, gitignore'lı).
    lhotse gerektirir: pip install lhotse

    Kullanım:
        m = AmiReferenceManager(); m.download()              # ilk kez (indirir)
        m = AmiReferenceManager(); m.download(download_audio=False)  # ses zaten varsa
    """

    MIC = "ihm-mix"                  # Mix-Headset = tek karışık akış
    PARTITION = "full-corpus-asr"    # test bölmesi = 16 toplantı (literatürle uyumlu)

    def __init__(self, data_dir=None):
        self.root = data_dir or os.path.join(DATASETS_ROOT, "ami")
        self.ami_corpus = os.path.join(self.root, "ami_corpus")   # lhotse indirme hedefi
        self.refs_dir = os.path.join(self.root, "ami_refs")       # üretilen referanslar

    def is_downloaded(self):
        return os.path.exists(os.path.join(self.refs_dir, "meetings.json"))

    def download(self, force=False, split="test", partition=None, download_audio=True):
        """AMI test referanslarını hazırlar.

        Args:
            force: True ise mevcut referansları yok sayıp yeniden üretir.
            split: 'test' | 'dev' | 'train' (varsayılan test).
            partition: lhotse AMI bölmesi (varsayılan full-corpus-asr).
            download_audio: False ise ham sesi indirmeyi atlar (zaten indiyse).
        """
        if self.is_downloaded() and not force:
            print(f"✅ AMI referansları zaten mevcut: {self.refs_dir}")
            return True

        try:
            from lhotse.recipes.ami import download_ami, prepare_ami
        except ImportError:
            print("❌ Bu adım 'lhotse' gerektirir: pip install lhotse")
            return False

        partition = partition or self.PARTITION
        os.makedirs(self.ami_corpus, exist_ok=True)
        os.makedirs(os.path.join(self.refs_dir, "rttm"), exist_ok=True)
        os.makedirs(os.path.join(self.refs_dir, "transcripts"), exist_ok=True)

        if download_audio:
            print(f"[1/3] AMI '{self.MIC}' indiriliyor → {self.ami_corpus}")
            print("      (Tüm oturumlar + annotations. Birkaç GB, sabırlı olun.)")
            download_ami(self.ami_corpus, mic=self.MIC)
            print("      İndirme tamam.")

        print(f"[2/3] Manifestler hazırlanıyor (mic={self.MIC}, partition={partition})")
        manifests = prepare_ami(
            data_dir=self.ami_corpus,
            annotations_dir=None,            # data_dir içindeki zip otomatik bulunur
            mic=self.MIC,
            partition=partition,
            normalize_text="kaldi",
        )

        sp = manifests[split]
        recordings = sp["recordings"]
        supervisions = sp["supervisions"]

        by_meeting: dict[str, list] = defaultdict(list)
        for seg in supervisions:
            by_meeting[seg.recording_id].append(seg)

        print(f"[3/3] '{split}' bölmesi işleniyor: {len(by_meeting)} toplantı")
        index = []
        for meeting_id in sorted(by_meeting):
            segs = by_meeting[meeting_id]
            with open(os.path.join(self.refs_dir, "rttm", f"{meeting_id}.rttm"),
                      "w", encoding="utf-8") as f:
                f.write(_supervisions_to_rttm(meeting_id, segs))

            spk_trans = _supervisions_to_speaker_transcript(segs)
            flat_text = " ".join(
                s.text.strip() for s in sorted(segs, key=lambda x: x.start) if s.text
            )
            with open(os.path.join(self.refs_dir, "transcripts", f"{meeting_id}.json"),
                      "w", encoding="utf-8") as f:
                json.dump({"speakers": spk_trans, "flat": flat_text}, f,
                          ensure_ascii=False, indent=2)

            rec = recordings[meeting_id]
            index.append({
                "meeting_id": meeting_id,
                "audio_path": str(rec.sources[0].source),
                "sampling_rate": rec.sampling_rate,
                "num_channels": rec.num_channels,
                "duration_sec": round(rec.duration, 1),
                "num_speakers": len(spk_trans),
            })
            flag = "" if (rec.sampling_rate == 16000 and rec.num_channels == 1) else "  <-- DİKKAT: 16k mono değil!"
            print(f"   {meeting_id}: {rec.duration / 60:5.1f} dk, {len(spk_trans)} konuşmacı, "
                  f"{rec.sampling_rate} Hz/{rec.num_channels}ch{flag}")

        with open(os.path.join(self.refs_dir, "meetings.json"), "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=2)

        total_min = sum(m["duration_sec"] for m in index) / 60
        print(f"\n✅ Tamam. {len(index)} toplantı, toplam {total_min:.0f} dk → {self.refs_dir}")
        return True

    def get_meetings(self):
        """meetings.json'u liste olarak döndürür (yoksa boş)."""
        p = os.path.join(self.refs_dir, "meetings.json")
        if not os.path.exists(p):
            return []
        with open(p, encoding="utf-8") as f:
            return json.load(f)

    def get_summary(self):
        if not self.is_downloaded():
            return {"status": "not_ready", "count": 0}
        meetings = self.get_meetings()
        return {"status": "ready", "meetings": len(meetings),
                "total_min": round(sum(m["duration_sec"] for m in meetings) / 60, 1)}
