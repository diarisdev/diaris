"""pytest yapılandırması — marker kaydı ve proje kökü path'i.

Hızlı birim testleri tests/unit/ altındadır. Ağır benchmark runner'ları
(tests/benchmarks/) test_*.py olmadığından pytest tarafından TOPLANMAZ; onlar
elle çalıştırılır.
"""

import os
import sys

# Proje kökünü path'e ekle (src.* ve tests.* importları için)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "requires_model: model/ağırlık indirme gerektiren testler")
    config.addinivalue_line(
        "markers", "slow: yavaş (ağır bağımlılık veya I/O) testler")
