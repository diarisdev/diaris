"""
Ses cihazı bulma modülü.
Sistem ses çıkışını yakalamak için uygun loopback/WASAPI cihazını otomatik bulur.
Hiçbir cihaz adı hardcoded değildir — her bilgisayarda otomatik çalışır.
"""


def list_loopback_devices(p):
    """
    Tüm WASAPI loopback cihazlarını listeler.

    Args:
        p: PyAudio instance (pyaudiowpatch)

    Returns:
        list[dict]: Bulunan loopback cihazlarının bilgileri.
    """
    try:
        return list(p.get_loopback_device_info_generator())
    except Exception:
        return []


def find_loopback_device(p):
    """
    WASAPI Loopback cihazını otomatik olarak bulur.
    pyaudiowpatch'in get_loopback_device_info_generator() metodu ile
    varsayılan ses çıkışının loopback'ini yakalar.

    Args:
        p: PyAudio instance (pyaudiowpatch)

    Returns:
        dict veya None: Bulunan loopback cihaz bilgisi, bulunamazsa None.
    """
    try:
        # pyaudiowpatch'in özel metodu — varsayılan çıkış cihazının loopback'ini döndürür
        loopback_devices = list(p.get_loopback_device_info_generator())
        if loopback_devices:
            # Varsayılan çıkış cihazının loopback'ini tercih et
            default_output = p.get_default_output_device_info()
            default_name = default_output["name"].lower()

            for dev in loopback_devices:
                if default_name in dev["name"].lower():
                    return dev

            # Varsayılanla eşleşme yoksa ilk loopback'i döndür
            return loopback_devices[0]
    except Exception:
        pass

    return None


def find_audio_device_by_keywords(p, keywords):
    """
    Anahtar kelimelere göre ses cihazı arar (fallback yöntemi).

    Args:
        p: PyAudio instance
        keywords: Aranacak anahtar kelimeler listesi

    Returns:
        int veya None: Bulunan cihazın indeksi, bulunamazsa None.
    """
    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        name = dev["name"].lower()
        if any(key in name for key in keywords):
            if dev["maxInputChannels"] > 0:
                return i
    return None


def list_input_devices(p):
    """
    Tüm kullanılabilir giriş cihazlarını listeler.

    Args:
        p: PyAudio instance

    Returns:
        list[dict]: Her cihaz için {index, name, channels, rate} bilgisi.
    """
    devices = []
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info["maxInputChannels"] > 0:
            devices.append({
                "index": i,
                "name": info["name"],
                "channels": info["maxInputChannels"],
                "rate": int(info["defaultSampleRate"]),
            })
    return devices


def select_device_interactive(p):
    """
    Otomatik algılama başarısız olursa, kullanıcıya cihaz listesi gösterip
    seçim yaptırır.

    Args:
        p: PyAudio instance

    Returns:
        dict veya None: Seçilen cihaz bilgisi, iptal edilirse None.
    """
    devices = list_input_devices(p)
    if not devices:
        return None

    print("\n[Kullanılabilir ses giriş cihazları]:")
    print("-" * 60)
    for idx, dev in enumerate(devices):
        print(f"  [{idx}] {dev['name']}")
        print(f"       Kanal: {dev['channels']} | Hız: {dev['rate']} Hz")
    print("-" * 60)

    while True:
        try:
            choice = input("\n[Seçim] Cihaz numarasını seçin (iptal için 'q'): ").strip()
            if choice.lower() == 'q':
                return None
            choice_idx = int(choice)
            if 0 <= choice_idx < len(devices):
                selected = devices[choice_idx]
                return p.get_device_info_by_index(selected["index"])
            print(f"[Hata] Lütfen 0-{len(devices)-1} arası bir sayı girin.")
        except ValueError:
            print("[Hata] Geçersiz giriş.")


def auto_detect_device(p, allow_interactive=True):
    """
    Tam otomatik cihaz algılama akışı:
    1. WASAPI Loopback dene
    2. Bulamazsa, izin verildiyse kullanıcıya sor

    Args:
        p: PyAudio instance
        allow_interactive: False ise otomatik algılama başarısız olduğunda input() çağırmaz.

    Returns:
        tuple[dict, int, int] veya None:
            (cihaz_bilgisi, kanal_sayısı, örnekleme_hızı) veya hata durumunda None.
    """
    # 1. Otomatik loopback algılama
    loopback = find_loopback_device(p)
    if loopback:
        channels = max(int(loopback["maxInputChannels"]), 1)
        rate = int(loopback["defaultSampleRate"])
        print(f"[Loopback cihazı bulundu]: {loopback['name']}")
        print(f"   Kanal: {channels} | Hız: {rate} Hz")
        return loopback, channels, rate

    # 2. Fallback: Kullanıcıya sor
    print("[Uyarı] Otomatik loopback cihazı bulunamadı.")
    if not allow_interactive:
        return None

    selected = select_device_interactive(p)
    if selected:
        channels = max(int(selected["maxInputChannels"]), 1)
        rate = int(selected["defaultSampleRate"])
        print(f"\n[Seçilen cihaz]: {selected['name']}")
        return selected, channels, rate

    return None
