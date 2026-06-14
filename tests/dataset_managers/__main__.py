"""`python -m tests.dataset_managers [flores200|librispeech|ami|chime6|all]`

Varsayılan: HEPSİ. Veri setlerini datasets/ altına indirir ve özet basar.
"""

import argparse
import sys

from . import available_datasets, get_dataset


def _demo_flores200():
    m = get_dataset("flores200")
    m.download()
    print("📊", m.get_summary())
    for p in m.get_pairs("en", "tr", split="devtest", limit=3):
        print(f"   [{p['index']}] SRC: {p['source'][:70]}")
        print(f"        REF: {p['reference'][:70]}")


def _demo_librispeech():
    m = get_dataset("librispeech")
    m.download()
    try:
        print("📊", m.get_summary())
    except ImportError as e:
        print(f"⚠️  Özet atlandı: {e}")


def _demo_ami():
    m = get_dataset("ami")
    m.download()
    try:
        samples = m.get_samples()
        print(f"📊 meeting_count: {len(samples)}")
        for s in samples:
            print(f"   🔈 {s['meeting_id']} | süre: {s['duration']:.1f}s | "
                  f"annotation: {len(s['annotations'])}")
    except ImportError as e:
        print(f"⚠️  Özet atlandı: {e}")


def _demo_chime6():
    m = get_dataset("chime6")
    m.download()  # lisanslı: sadece varlık kontrolü
    print("📊", m.get_summary())


def _demo_ami_refs():
    m = get_dataset("ami_refs")
    m.download()  # lhotse ile ~birkaç GB indirir
    print("📊", m.get_summary())


_DEMOS = {
    "flores200": _demo_flores200,
    "librispeech": _demo_librispeech,
    "ami": _demo_ami,
    "ami_refs": _demo_ami_refs,
    "chime6": _demo_chime6,
}

# 'all' ile otomatik inecekler — ami_refs (birkaç GB + lhotse) bilerek HARİÇ;
# açıkça istenince iner: python -m tests.dataset_managers ami_refs
_ALL = ["flores200", "librispeech", "ami", "chime6"]


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # Windows cp1254 emoji çökmesini önle
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="Test veri setlerini indir (varsayılan: all).")
    parser.add_argument("dataset", nargs="?", default="all",
                        choices=["all"] + available_datasets())
    args = parser.parse_args()

    selected = _ALL if args.dataset == "all" else [args.dataset]
    failures = []
    for name in selected:
        print("\n" + "=" * 60)
        print(name)
        print("=" * 60)
        try:
            _DEMOS[name]()
        except Exception as e:
            failures.append((name, e))
            print(f"❌ '{name}' hazırlanamadı: {e}")

    print("\n" + "=" * 60)
    if failures:
        print(f"⚠️  {len(failures)} başarısız: {', '.join(n for n, _ in failures)}")
    else:
        print("✅ Tüm veri setleri hazır.")
    print("=" * 60)


if __name__ == "__main__":
    main()
