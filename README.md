# ♟️ Satranç Bulmacalarının Zorluk Derecesinin Makine Öğrenmesi Yöntemleri ile Tahmini

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%20Enabled-ee4c2c.svg)](https://pytorch.org/)
[![LightGBM](https://img.shields.io/badge/GBDT-LightGBM%20%7C%20XGBoost%20%7C%20CatBoost-2ea44f.svg)](https://lightgbm.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Web%20App-Streamlit-ff4b4b.svg)](https://streamlit.io/)

**Sivas Cumhuriyet Üniversitesi — Bilgisayar Mühendisliği Bölümü Bitirme Projesi**  
**Hazırlayan:** Eren Kaya 

---

## 📌 Proje Özeti

Bu proje, satranç bulmacalarının (taktik pozisyonların) zorluk derecesini (Glicko-2 Elo puanını) yalnızca tahta üzerindeki taş dizilimine bakarak değil, **insan oyuncuların bilişsel karar süreçlerini ve yanılgı eğilimlerini** modelleyerek tahmin eden uçtan uca bir makine öğrenmesi sistemidir.

* **Veri Ölçeği:** Lichess açık bulmaca veri tabanından Glicko-2 Derece Sapması (`RatingDeviation <= 90`) filtresiyle seçilen **4.823.268** yüksek güvenilirlikli bulmaca (`data/puzzles.db`).
* **Hibrit Mimari (`v4` — 107 Öznitelik):** Alan bilgisine dayalı **56 yapısal satranç özniteliği**, insan davranışını taklit eden **Maia-2 (Rapid & Blitz)** sinir ağlarından çok adımlı (`Ply 1`, `Ply 2`, `Ply 3`, `Ply 5`) çıkarılan **45 bilişsel politika sinyali** ve **6 derin varyant etkileşimi**.
* **Sızıntısız (Leak-Free) Topluluk & Kalibrasyon:** Ters-varyans (`1 / RD²`) ağırlıklı **LightGBM + XGBoost + CatBoost** topluluğu, 5-katlı *out-of-fold* artık öğrenme (*residual learning*) ve küplü B-Spline kalibrasyonu.
* **Nihai Performans (`N = 482.428` Test Kümesi):** **231,968 Elo Tam Test RMSE** (`178,740 MAE`, **$R^2 = 0,7628$**) ve belirsizlik tabanlı seçici tahminde (%90 kapsam) **222,201 Elo Masked RMSE**.

---

## 📊 Mimari Evrim ve Test Sonuçları (`N = 482.428`)

Tüm modeller sabit tohum (`seed=42`) ile oluşturulan **%80 Eğitim (`3.858.440`) / %10 Doğrulama (`482.400`) / %10 Test (`482.428`)** ayrımı üzerinde değerlendirilmiştir.

| Sürüm | Mimari ve Öznitelik Kümesi | Tam Test RMSE | Tam Test MAE | Test $R^2$ | Yardımcı (%90 Masked RMSE) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Baseline** | Sabit Eğitim Ortalaması Tahmini | 476,322 | 396,989 | 0,0000 | — |
| **v1** | Saf Yapısal Tahta Modeli (56 Öznitelik) | 282,710 | 220,570 | 0,6477 | — |
| **v2** | Hibrit Ply 1 Maia-2 Rapid (70 Öznitelik) | 261,680 | 201,490 | 0,6982 | — |
| **v3** | Çok Adımlı Ply 1 & Ply 3 Maia-2 (84 Öznitelik) | 241,378 | 185,536 | 0,7432 | 237,070 |
| **v3.1** | RD Ters-Varyans Ağırlıklı + Heteroscedastic Unc (84 Öznitelik) | 241,146 | 185,263 | 0,7437 | 229,547 |
| **v4 (Nihai)** | **Tam Bilişsel Tuzak & Ply 5 Derin Varyant (107 Öznitelik)** | **231,968** | **178,740** | **0,7628** | **222,201** |

### v4 Modelleme Katmanları Ayrıntısı

| v4 Modelleme Aşaması | Val RMSE | Test MSE | Tam Test RMSE | Test MAE | Test $R^2$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 1. Tekil LightGBM (`9.500 ağaç, num_leaves=319, lr=0,042`) | 232,497 | 54.322,45 | 233,072 | 179,803 | 0,7606 |
| 2. Tekil XGBoost (`CUDA, depth=9, 8.000 ağaç, lr=0,045`) | 233,288 | 54.697,85 | 233,876 | 180,564 | 0,7589 |
| 3. Tekil CatBoost (`GPU, depth=9, 7.500 iter., lr=0,060`) | 233,883 | 54.961,57 | 234,439 | 181,429 | 0,7578 |
| 4. 3'lü v4 GBDT Topluluğu (`0,525 LGB + 0,285 XGB + 0,190 CAT`) | 231,845 | 54.020,53 | 232,423 | 179,426 | 0,7619 |
| 5. + 5-Katlı Sızıntısız Artık Öğrenme (`Residual, α=0,920`) | 231,367 | 53.768,67 | 231,881 | 178,584 | 0,7630 |
| **6. + 5-Katlı Küplü B-Spline + Ridge Kalibrasyon (Nihai v4)** | **231,466** | **53.809,23** | **231,968** | **178,740** | **0,7628** |

---

## 🔬 Kontrollü Öznitelik Ablasyonu ve Permutation Importance

### 1. Sabit Hiperparametrelerle Öznitelik Grubu Ablasyonu

`v3.1` (`241,15 RMSE`) $\rightarrow$ `v4` geçişindeki kazancın model kapasitesi artışından (`-4,790 Elo`) ve yeni bilişsel özniteliklerden (`-3,540 Elo`) gelen paylarını ayrıştırmak için LightGBM hiperparametreleri sabit tutularak kontrollü ablasyon yapılmıştır:

| Deney Grubu (Sabit LightGBM Hiperparametreleri) | Özellik ($k$) | Test RMSE | Test MAE | Fark (vs. A) |
| :--- | :---: | :---: | :---: | :---: |
| **A. v3.1 Taban Öznitelikleri** (Aynı v4 Hiperparametreleri) | 86 | 236,356 | 182,246 | 0,000 |
| **B. + Blitz vs. Rapid Bilişsel Tuzak Ayrışması** | 91 | 236,026 | 181,918 | -0,330 |
| **C. + 1900 Elo Aday Hamle Sırası & Top-1/Top-2 Payı** | 92 | 236,562 | 182,449 | +0,206 |
| **D. + Rakip Savunma Dallanması (Ply 2 Entropi)** | 89 | 235,876 | 181,847 | -0,480 |
| **E. + Derin Varyant Hesaplaması (Ply 5)** | 89 | 233,277 | 179,895 | -3,079 |
| **F. Tam v4 (Tüm Yeni Gruplar + Derin Etkileşimler)** | **107** | **232,816** | **179,606** | **-3,540** |

### 2. Permutation Importance (En Kritik İlk 10 Öznitelik — `N = 25.000` Test)

| Sıra | Öznitelik | Öznitelik Ailesi | Karıştırıldığında RMSE Artışı (Elo) | Std |
| :---: | :--- | :--- | :---: | :---: |
| 1 | `maia_p3_prob_1100` | Ply 3 Maia-2 Rapid | **+48,424** | 1,169 |
| 2 | `maia_p5_prob_1500` | v4 Ply 5 Derin Varyant | **+47,433** | 0,121 |
| 3 | `avail_checks_start` | Yapısal Tahta | **+35,568** | 0,149 |
| 4 | `maia_blitz_prob_1500` | v4 Blitz vs. Rapid Tuzak | **+23,384** | 0,037 |
| 5 | `maia_entropy_1500` | Ply 1 Karar Entropisi | **+20,801** | 0,054 |
| 6 | `ends_in_mate` | Yapısal Tahta | **+19,976** | 0,343 |
| 7 | `maia_full_chain_prob_1900` | v4 Tam Zincir Etkileşimi | **+15,535** | 0,461 |
| 8 | `first_move_is_check` | Yapısal Tahta | **+14,701** | 0,112 |
| 9 | `maia_prob_1100` | Ply 1 Maia-2 Rapid | **+12,943** | 0,206 |
| 10 | `first_move_piece` | Yapısal Tahta | **+11,484** | 0,311 |

---

## 🧪 Örtüşme Denetimli Davranışsal Dış Doğrulama (`testing_data_cropped.csv`)

2.235 bulmacalık harici davranışsal test seti (`testing_data_cropped.csv`), `PuzzleId` ve normalize `FEN` üzerinden 4,82 milyonluk veritabanı ile taranmıştır. Kayıtların **%99,46'sının (`2.223 / 2.235`)** eğitim/doğrulama verisinde hiç yer almadığı (`unseen_db`) doğrulanmıştır. Örtüşen 12 kayıt çıkarılarak elde edilen kesin sızıntısız **`N = 2.223`** alt kümede model tahminleri ile gerçek 1550 Rapid insan çözüm olasılığı arasında güçlü korelasyon elde edilmiştir:

* **Pearson Korelasyonu:** $r = -0,6865$ ($p = 1,15 \times 10^{-309}$)
* **Spearman Sıra Korelasyonu:** $\rho = -0,6676$

| Train-Dışı Temiz Alt Küme Grubu (`N = 2.223`) | Bulmaca Adedi | Ort. `v4` Tahmini | Ort. 1550 Rapid İnsan Başarısı |
| :--- | :---: | :---: | :---: |
| **Kolay Grup** (`<1050` IRT eşiği; tüm seviyelerde > %50 başarı) | 341 | 1164,2 Elo | %81,6 |
| **Orta / İleri Grup** (`1050–2050` bandında %50 başarı eşiği) | 312 | 1424,7 Elo | %56,5 |
| **Zor Grup** (`>2050` IRT eşiği; 2050 Elo'da < %50 başarı) | 1.570 | 1767,6 Elo | %14,7 |

---

## 🗂️ Proje Klasör Yapısı

```text
satranc_zorluk_tahmini/
├── app/                          # Streamlit interaktif web paneli ve canlı pozisyon analizi
├── data/
│   └── puzzles.db                # SQLite veritabanı (4.823.268 bulmaca & öznitelik tabloları)
├── models_v4/                    # Eğitilmiş v4 LightGBM, XGBoost, CatBoost ve kalibrasyon modelleri
├── reports/
│   ├── figures/                  # Evrim, kalibrasyon, hata dağılımı ve ablasyon grafikleri
│   └── tables/                   # CSV formatında ablasyon, permutation importance ve tema tabloları
├── src/
│   ├── 01_build_database.py      # Lichess veri filtreleme (RD <= 90) ve sızıntısız split kurulumu
│   ├── 02_extract_features.py    # Yapısal (python-chess) ve Maia-2 (Rapid/Blitz) öznitelik çıkarımı
│   ├── 03_train_v4_gbdt.py       # RD ağırlıklı v4 GBDT topluluğu, residual öğrenme ve kalibrasyon
│   ├── 04_uncertainty_model.py   # Heteroscedastic belirsizlik modeli ve seçici tahmin analizi
│   ├── 05_external_validation.py # Harici davranışsal test kümesi değerlendirmesi
│   ├── 06_ablation_and_overlap.py# Dış küme örtüşme denetimi ve kontrollü öznitelik grubu ablasyonu
│   └── 06b_fast_perm.py          # Test kümesi üzerinde Permutation Importance hesabı
├── requirements.txt
└── README.md