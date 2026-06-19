# UI ikonu

Uygulama ikonunu bu klasöre koyun. Çözücü (`src/ui/resources.py`) şu sırayla arar:

1. `icon.png`  ← önerilen (Qt'de en sağlamı; 256×256 veya daha büyük, kare, şeffaf arka plan)
2. `icon.svg`
3. `icon.ico`

Yani indirdiğiniz dosyayı **`icon.png`** adıyla buraya kaydetmeniz yeterli — pencere
başlığı, görev çubuğu ve uygulama ikonu otomatik onu kullanır. Dosya yoksa uygulama
ikonsuz ama sorunsuz çalışır.
