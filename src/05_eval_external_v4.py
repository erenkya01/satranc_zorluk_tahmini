import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.inference import PuzzleDifficultyPredictor


def find_external_csv() -> Path:
    for candidate in [
        ROOT_DIR / "testing_data_cropped.csv",
        ROOT_DIR / "data" / "testing_data_cropped.csv",
    ]:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("testing_data_cropped.csv proje kökünde veya data/ klasöründe bulunamadı!")


def evaluate_model_on_external(model_dir: Path, df: pd.DataFrame) -> pd.DataFrame:
    predictor = PuzzleDifficultyPredictor(model_dir=model_dir)
    preds = []
    margins = []
    blitz_gaps = []

    for fen, moves in tqdm(
        zip(df["FEN"], df["Moves"]),
        total=len(df),
        desc=f"{model_dir.name} Dış Doğrulama",
        unit="bulmaca",
    ):
        res = predictor.predict(fen, moves, puzzle_id=None)
        preds.append(res["predicted_rating"])
        margins.append(res["expected_margin"])
        blitz_gaps.append(res["features"].get("maia_blitz_trap_gap_1900", np.nan))

    out = df.copy()
    out["pred_rating"] = preds
    out["expected_margin"] = margins
    out["blitz_trap_gap"] = blitz_gaps
    return out


def main():
    csv_path = find_external_csv()
    df = pd.read_csv(csv_path)
    print(f"Dış doğrulama dosyası yüklendi: {csv_path} ({len(df):,} bulmaca)")

    t0 = time.time()
    v4_dir = ROOT_DIR / "models_v4"
    res_df = evaluate_model_on_external(v4_dir, df)
    elapsed = time.time() - t0

    out_csv = v4_dir / "external_validation_v4.csv"
    res_df.to_csv(out_csv, index=False)

    p_rapid_1550 = res_df["success_prob_rapid_1550"].to_numpy()
    p_blitz_1550 = res_df["success_prob_blitz_1550"].to_numpy()
    y_pred = res_df["pred_rating"].to_numpy()

    r_rapid, _ = pearsonr(y_pred, p_rapid_1550)
    rho_rapid, _ = spearmanr(y_pred, p_rapid_1550)
    r_blitz, _ = pearsonr(y_pred, p_blitz_1550)
    rho_blitz, _ = spearmanr(y_pred, p_blitz_1550)

    print(f"\n================ v4 DIŞ DOĞRULAMA SONUÇLARI ({elapsed:.1f} sn) ================")
    print(f"1550 Rapid Çözüm Olasılığı Korelasyonu -> Pearson r: {r_rapid:+.4f} | Spearman rho: {rho_rapid:+.4f}")
    print(f"1550 Blitz Çözüm Olasılığı Korelasyonu -> Pearson r: {r_blitz:+.4f} | Spearman rho: {rho_blitz:+.4f}")

    # İnsan çözüm oranına göre 3 zorluk grubu ayrımı
    easy_mask = p_rapid_1550 > 0.70
    mid_mask = (p_rapid_1550 >= 0.35) & (p_rapid_1550 <= 0.70)
    hard_mask = p_rapid_1550 < 0.35

    print("\nİnsan Başarı Oranına Göre Tahmin Dağılımı (Rapid 1550):")
    for label, mask in [
        ("Kolay (> %70 çözüm)", easy_mask),
        ("Orta (%35 – %70 çözüm)", mid_mask),
        ("Zor (< %35 çözüm)", hard_mask),
    ]:
        sub = res_df[mask]
        print(
            f"  - {label:24s} (n={len(sub):4d}): Ort. Tahmin = {sub['pred_rating'].mean():6.1f} Elo "
            f"(Std: {sub['pred_rating'].std():5.1f} | Ort. Belirsizlik: ±{sub['expected_margin'].mean():5.1f} Elo)"
        )

    # Eğer v3 modeli de duruyorsa yan yana kıyasla
    v3_dir = ROOT_DIR / "models_v3_full"
    if v3_dir.exists():
        v3_df = evaluate_model_on_external(v3_dir, df)
        r_v3, _ = pearsonr(v3_df["pred_rating"], p_rapid_1550)
        rho_v3, _ = spearmanr(v3_df["pred_rating"], p_rapid_1550)
        print("\n================ v3 vs. v4 DIŞ KÜME KIYASI ================")
        print(f"v3 (84 özellik)  -> Pearson r: {r_v3:+.4f} | Spearman rho: {rho_v3:+.4f}")
        print(f"v4 (107 özellik) -> Pearson r: {r_rapid:+.4f} | Spearman rho: {rho_rapid:+.4f}")

    print(f"\nDetaylı tahminler kaydedildi: {out_csv}")


if __name__ == "__main__":
    main()