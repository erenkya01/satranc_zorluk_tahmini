# Satranç Bulmacalarının Zorluk Derecesinin Makine Öğrenmesi Yöntemleri ile Tahmini

**Sivas Cumhuriyet Üniversitesi — Mühendislik Fakültesi, Bilgisayar Mühendisliği Bölümü Bitirme Projesi**  
**Hazırlayan:** Eren Kaya

## Proje Özeti
Bu çalışma, 4.823.268 adet Lichess satranç bulmacasının Glicko-2 zorluk puanlarını (Elo) **çok adımlı bilişsel insan politika olasılıkları (Maia-2 Rapid & Blitz)** ile **alan bilgisine dayalı yapısal tahta özniteliklerini** birleştiren hibrit bir makine öğrenmesi mimarisiyle tahmin eder. Sistem; **LightGBM, XGBoost ve CatBoost** topluluğu üzerine kurulu **5-katlı Sızıntısız Artık Öğrenme (Residual GBDT)**, **Küplü B-Spline + Ridge kuyruk kalibrasyonu** ve **Heteroscedastic hata varyansı (belirsizlik)** katmanlarından oluşur.

## Mimari Evrim ve Test Kümesi Sonuçları (N = 482.428)

| Sürüm | Öznitelik ve Eğitim Mimarisi | Test RMSE (Elo) | Test MAE (Elo) | Test $R^2$ | %90 Masked RMSE |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Baseline** | Ortalama Puan Tahmini | 476,32 | 396,99 | 0,0000 | — |
| **v1 (Yapısal)** | 56 Yapısal Tahta & Taktik Özniteliği | 282,71 | 220,57 | 0,6477 | — |
| **v2 (Hibrit Ply 1)** | 70 Öznitelik (56 Yapısal + 14 Ply 1 Maia-2) | 261,68 | 201,49 | 0,6982 | — |
| **v3 (Çok Adımlı)** | 84 Öznitelik (Ply 1 & Ply 3 Maia-2 + Zincir Etkileşim) | 241,38 | 185,54 | 0,7432 | 237,07 |
| **v3.1 (Stabilize)** | 84 Öznitelik + Glicko-2 RD Ağırlık + Heteroscedastic Belirsizlik | 241,15 | 185,26 | 0,7437 | 229,55 |
| **v4 (Nihai Bilişsel)** | **107 Öznitelik (Blitz/Rapid Tuzak + Ply 2 Savunma + Ply 1/3/5)** | **231,97** | **178,74** | **0,7628** | **222,20** |

## Dizin Yapısı

```text
├── app/
│   └── streamlit_app.py             # Etkileşimli web arayüzü ve v1–v4 model karşılaştırma paneli
├── reports/
│   ├── figures/                     # 300 DPI tez ve FedCSIS 2025 doğrulama grafikleri (fig1–fig6)
│   └── tables/                      # Zorluk kademesi ve taktik tema bazlı hata tabloları
├── src/
│   ├── 01_build_db.py               # Lichess verisinin SQLite veritabanına aktarımı (80/10/10 ayrım)
│   ├── 02_features.py               # 56 yapısal ve taktiksel tahta özniteliği çıkarımı
│   ├── 02b_maia2_features.py        # Çözüm 1. hamlesi (Ply 1) Maia-2 insan politika çıkarımı
│   ├── 02c_maia2_ply3_features.py   # Çözüm 2. hamlesi (Ply 3) Maia-2 insan politika çıkarımı
│   ├── 02d_maia2_v4_features.py     # Blitz-Rapid tuzak farkı, Ply 2 savunma ve Ply 5 GPU çıkarımı
│   ├── 03_train_v4_gbdt.py          # 107 öznitelikli LightGBM + XGBoost + CatBoost eğitimi
│   ├── 04_v4_residual_cal.py        # 5-katlı artık öğrenme, B-Spline kalibrasyon ve belirsizlik modeli
│   ├── 05_eval_external_v4.py       # 2.235 bulmacalık kör yarışma kümesi dış doğrulaması
│   ├── 05_thesis_evaluation.py      # Kademe/tema hata analizi ve 300 DPI tez grafikleri üretimi
│   └── inference.py                 # Veritabanı ve canlı GPU/CPU Maia-2 destekli tahmin motoru
└── requirements.txt
