"""AMI Meeting Corpus (küçük alt küme) — diarization benchmark.

İki meeting indirir (audio + RTTM referans) ve DER testleri için hazırlar.
RTTM'ler BUTSpeechFIT/AMI-diarization-setup reposundan alınır.
"""

from __future__ import annotations

import os
import shutil

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
