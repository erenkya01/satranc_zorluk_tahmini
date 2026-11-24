# Satranç Bulmacalarının Zorluk Derecesinin Makine Öğrenmesi Yöntemleri ile Tahmini

**Sivas Cumhuriyet Üniversitesi — Mühendislik Fakültesi, Bilgisayar Mühendisliği Bölümü Bitirme Projesi**  
**Hazırlayan:** Eren Kaya

## Proje Özeti
Bu çalışma, 4.823.268 adet Lichess satranç bulmacasının Glicko-2 zorluk puanlarını (Elo) **çok adımlı insan politika olasılıkları (Maia-2)** ile **alan bilgisine dayalı yapısal tahta özniteliklerini** birleştiren hibrit bir makine öğrenmesi mimarisiyle tahmin eder. Sistem; **LightGBM, XGBoost ve CatBoost** topluluğu üzerine kurulu **5-katlı Artık Öğrenme (Residual GBDT)** ve **doğrusal kalibrasyon** katmanlarından oluşur.

## Mimari Evrim ve Test Kümesi Sonuçları (N = 482.428)

| Sürüm | Öznitelik Mimarisi | Test RMSE (Elo) | Test MAE (Elo) | Test $R^2$ |
| :--- | :--- | :---: | :---: | :---: |
| **Baseline** | Ortalama Puan Tahmini | 476,32 | 396,99 | 0,0000 |
| **v1 (Yapısal)** | 56 Yapısal Tahta & Taktik Özniteliği | 282,71 | 220,57 | 0,6477 |
| **v2 (Hibrit Ply 1)** | 70 Öznitelik (56 Yapısal + 14 Ply 1 Maia-2) | 261,68 | 201,49 | 0,6982 |
| **v3 (Tam Hibrit)** | **84 Öznitelik (56 Yapısal + Ply 1 & Ply 3 Maia-2 + Etkileşim)** | **241,38** | **185,54** | **0,7432** |

## Dizin Yapısı

```text
├── app/
│   └── streamlit_app.py             # Etkileşimli web arayüzü ve v1/v2/v3 karşılaştırma paneli
├── src/
│   ├── 01_build_db.py               # Lichess bulmaca verisinin SQLite veritabanına aktarımı (80/10/10 ayrım)
│   ├── 02_features.py               # 56 yapısal ve taktiksel öznitelik çıkarımı
│   ├── 02b_maia2_features.py        # Çözüm 1. hamlesi (Ply 1) Maia-2 insan politika çıkarımı
│   ├── 02c_maia2_ply3_features.py   # Çözüm 2. hamlesi (Ply 3) Maia-2 insan politika çıkarımı
│   ├── 03_train_v3_gbdt.py          # 84 öznitelikli LightGBM + XGBoost + CatBoost eğitimi
│   ├── 04_v3_residual_cal.py        # Artık öğrenme (Residual GBDT) ve kalibrasyon katmanı
│   ├── 05_thesis_evaluation.py      # Zorluk kademelerine göre hata analizi ve 300 DPI tez grafikleri
│   └── inference.py                 # Veritabanı ve canlı Maia-2 destekli tahmin motoru
└── requirements.txt
