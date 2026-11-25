import json
import sqlite3
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.inference import PuzzleDifficultyPredictor

DB_PATH = ROOT_DIR / "data" / "puzzles.db"
V1_DIR = ROOT_DIR / "models_structural_only"
V2_DIR = ROOT_DIR / "models_hybrid_maia2"
V3_DIR = ROOT_DIR / "models_v3_full"
V31_DIR = ROOT_DIR / "models_v3_1"
V4_DIR = ROOT_DIR / "models_v4"

COMP_CSV_CANDIDATES = [
    ROOT_DIR / "testing_data_cropped.csv",
    ROOT_DIR / "data" / "testing_data_cropped.csv",
]

OUT_DIR = ROOT_DIR / "reports"
FIG_DIR = OUT_DIR / "figures"
TBL_DIR = OUT_DIR / "tables"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "figure.dpi": 150,
        "savefig.dpi": 300,
    }
)

TIERS = [
    ("Başlangıç (<1100)", 0, 1100),
    ("Orta (1100–1500)", 1100, 1500),
    ("İleri (1500–1900)", 1500, 1900),
    ("Usta Adayı (1900–2300)", 1900, 2300),
    ("Usta Seviyesi (2300+)", 2300, 4000),
]

TARGET_THEMES = [
    ("mateIn1", "1 Hamlede Mat (Mate in 1)"),
    ("mateIn2", "2 Hamlede Mat (Mate in 2)"),
    ("mateIn3", "3+ Hamlede Mat (Mate in 3+)"),
    ("fork", "Çatal (Fork)"),
    ("pin", "Açmaz (Pin)"),
    ("skewer", "Şiş (Skewer)"),
    ("discoveredAttack", "Açarak Saldırı (Discovered Attack)"),
    ("hangingPiece", "Boşta Taş (Hanging Piece)"),
    ("sacrifice", "Taş Fedası (Sacrifice)"),
    ("quietMove", "Sessiz Hamle (Quiet Move)"),
    ("zugzwang", "Zugzwang"),
    ("deflection", "Saptırma (Deflection)"),
    ("attraction", "Cezbetme (Attraction)"),
    ("endgame", "Oyun Sonu (Endgame)"),
    ("middlegame", "Oyun Ortası (Middlegame)"),
    ("opening", "Açılış (Opening)"),
    ("crushing", "Ezici Üstünlük (Crushing)"),
    ("advantage", "Konumsal Üstünlük (Advantage)"),
]


def build_meta_features(p_lgb, p_xgb, p_cat, p_ens, context):
    stack3 = np.column_stack([p_lgb, p_xgb, p_cat])
    pred_std = np.std(stack3, axis=1)
    pred_range = np.max(stack3, axis=1) - np.min(stack3, axis=1)
    base_meta = np.column_stack(
        [
            p_lgb,
            p_xgb,
            p_cat,
            p_ens,
            pred_std,
            pred_range,
            p_lgb - p_xgb,
            p_lgb - p_cat,
            p_xgb - p_cat,
        ]
    ).astype(np.float32)
    return np.hstack([base_meta, context.astype(np.float32)])


def load_v4_test_predictions():
    arr = np.load(V4_DIR / "v4_oof_and_test.npz")
    y_test = arr["y_test"].astype(np.float32)
    test_lgb = arr["test_lgb"].astype(np.float32)
    test_xgb = arr["test_xgb"].astype(np.float32)
    test_cat = arr["test_cat"].astype(np.float32)
    test_ens = arr["test_ens"].astype(np.float32)
    test_context = arr["test_context"].astype(np.float32)

    res_models = joblib.load(V4_DIR / "residual_models.joblib")
    calibrators = joblib.load(V4_DIR / "calibrator.joblib")
    unc_models = joblib.load(V4_DIR / "uncertainty_models.joblib")
    with open(V4_DIR / "residual_meta.json", "r", encoding="utf-8") as f:
        alpha = float(json.load(f)["alpha"])

    X_meta_test = build_meta_features(
        test_lgb, test_xgb, test_cat, test_ens, test_context
    )
    res_pred = np.mean([m.predict(X_meta_test) for m in res_models], axis=0).astype(np.float32)
    p_after_res = (test_ens + alpha * res_pred).reshape(-1, 1)
    p_final = np.mean([cal.predict(p_after_res) for cal in calibrators], axis=0).astype(np.float32)

    pred_log_var = np.mean([um.predict(X_meta_test) for um in unc_models], axis=0)
    pred_sigma = np.sqrt(np.maximum(np.exp(pred_log_var) - 2500.0, 6400.0)).astype(np.float32)
    return y_test, test_ens, p_final, pred_sigma


def compute_competition_metrics(y_true: np.ndarray, y_pred: np.ndarray, uncertainty: np.ndarray) -> dict:
    sq_err = (y_true - y_pred) ** 2
    full_mse = float(np.mean(sq_err))
    full_rmse = float(np.sqrt(full_mse))
    full_mae = float(mean_absolute_error(y_true, y_pred))
    full_r2 = float(r2_score(y_true, y_pred))

    thresh_90 = float(np.quantile(uncertainty, 0.90))
    keep_mask = uncertainty <= thresh_90
    masked_mse = float(np.mean(sq_err[keep_mask]))
    masked_rmse = float(np.sqrt(masked_mse))

    oracle_mse = float(np.mean(sq_err[sq_err <= np.quantile(sq_err, 0.90)]))
    np_ratio = masked_mse / oracle_mse

    return {
        "full_mse": full_mse,
        "full_rmse": full_rmse,
        "full_mae": full_mae,
        "full_r2": full_r2,
        "masked_mse_90": masked_mse,
        "masked_rmse_90": masked_rmse,
        "oracle_mse_90": oracle_mse,
        "np_ratio": np_ratio,
        "uncertainty_thresh_90": thresh_90,
    }


def compute_stratified_table(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    rows = []
    total_n = len(y_true)
    for label, lo, hi in TIERS:
        mask = (y_true >= lo) & (y_true < hi)
        yt = y_true[mask]
        yp = y_pred[mask]
        err = yp - yt
        abs_err = np.abs(err)
        rows.append(
            {
                "Zorluk_Kademesi": label,
                "Bulmaca_Sayisi": int(len(yt)),
                "Veri_Payi_Pct": round(100.0 * len(yt) / total_n, 2),
                "Gercek_Ort_Elo": round(float(np.mean(yt)), 1),
                "Tahmin_Ort_Elo": round(float(np.mean(yp)), 1),
                "Ortalama_Sapma_Bias": round(float(np.mean(err)), 1),
                "RMSE": round(float(np.sqrt(mean_squared_error(yt, yp))), 2),
                "MAE": round(float(mean_absolute_error(yt, yp)), 2),
                "Isabet_250_Elo_Pct": round(float(np.mean(abs_err <= 250.0) * 100.0), 1),
                "Isabet_350_Elo_Pct": round(float(np.mean(abs_err <= 350.0) * 100.0), 1),
            }
        )

    err_all = y_pred - y_true
    abs_all = np.abs(err_all)
    rows.append(
        {
            "Zorluk_Kademesi": "GENEL TOPLAM (Test)",
            "Bulmaca_Sayisi": int(total_n),
            "Veri_Payi_Pct": 100.0,
            "Gercek_Ort_Elo": round(float(np.mean(y_true)), 1),
            "Tahmin_Ort_Elo": round(float(np.mean(y_pred)), 1),
            "Ortalama_Sapma_Bias": round(float(np.mean(err_all)), 1),
            "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 2),
            "MAE": round(float(mean_absolute_error(y_true, y_pred)), 2),
            "Isabet_250_Elo_Pct": round(float(np.mean(abs_all <= 250.0) * 100.0), 1),
            "Isabet_350_Elo_Pct": round(float(np.mean(abs_all <= 350.0) * 100.0), 1),
        }
    )
    return pd.DataFrame(rows)


def compute_theme_error_table(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame | None:
    if not DB_PATH.exists():
        return None
    conn = sqlite3.connect(DB_PATH)
    df_meta = pd.read_sql_query(
        """
        SELECT r.rating, r.themes
        FROM structural_features sf
        INNER JOIN maia2_features m1 ON sf.puzzle_id = m1.puzzle_id
        INNER JOIN maia2_ply3_features m3 ON sf.puzzle_id = m3.puzzle_id
        INNER JOIN maia2_v4_features m4 ON sf.puzzle_id = m4.puzzle_id
        INNER JOIN raw_puzzles r ON sf.puzzle_id = r.puzzle_id
        WHERE LOWER(r.split) = 'test'
        ORDER BY sf.puzzle_id
        """,
        conn,
    )
    conn.close()

    if len(df_meta) != len(y_true) or not np.array_equal(df_meta["rating"].to_numpy(), y_true.astype(int)):
        return None

    themes_series = df_meta["themes"].fillna("").astype(str)
    rows = []
    for tag, label_tr in TARGET_THEMES:
        mask = themes_series.str.contains(rf"\b{tag}\b", regex=True).to_numpy()
        cnt = int(np.sum(mask))
        if cnt < 100:
            continue
        yt = y_true[mask]
        yp = y_pred[mask]
        rows.append(
            {
                "Taktik_Tema": label_tr,
                "Etiket": tag,
                "Bulmaca_Sayisi": cnt,
                "Gercek_Ort_Elo": round(float(np.mean(yt)), 1),
                "Tahmin_Ort_Elo": round(float(np.mean(yp)), 1),
                "Sapma_Bias": round(float(np.mean(yp - yt)), 1),
                "RMSE": round(float(np.sqrt(mean_squared_error(yt, yp))), 2),
                "MAE": round(float(mean_absolute_error(yt, yp)), 2),
            }
        )
    return pd.DataFrame(rows).sort_values("RMSE", ascending=True)


def evaluate_competition_blind_csv(csv_path: Path | None) -> pd.DataFrame | None:
    if csv_path is None or not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)
    sub_path = OUT_DIR / "competition_submission_v4.csv"
    ext_v4_path = V4_DIR / "external_validation_v4.csv"

    used_cache = False
    if sub_path.exists():
        sub_df = pd.read_csv(sub_path)
        if len(sub_df) == len(df):
            for col in ["predicted_rating", "ci_low", "ci_high", "expected_margin", "keep_90pct_mask"]:
                df[col] = sub_df[col]
            df["solution_ply"] = df["Moves"].str.split().str.len() - 1
            used_cache = True
            print(f"\n[Bölüm B] Önceden hesaplanan v4 yarışma tahminleri '{sub_path.name}' dosyasından yüklendi.")
    elif ext_v4_path.exists():
        ext_df = pd.read_csv(ext_v4_path)
        if len(ext_df) == len(df) and "pred_rating" in ext_df.columns:
            df["predicted_rating"] = ext_df["pred_rating"]
            df["expected_margin"] = ext_df["expected_margin"]
            df["ci_low"] = (df["predicted_rating"] - df["expected_margin"]).round().astype(int)
            df["ci_high"] = (df["predicted_rating"] + df["expected_margin"]).round().astype(int)
            q90 = float(np.quantile(df["expected_margin"], 0.90))
            df["keep_90pct_mask"] = (df["expected_margin"] <= q90).astype(int)
            df["solution_ply"] = df["Moves"].str.split().str.len() - 1
            df[["PuzzleId", "predicted_rating", "ci_low", "ci_high", "expected_margin", "keep_90pct_mask"]].to_csv(
                sub_path, index=False
            )
            used_cache = True
            print(f"\n[Bölüm B] '{ext_v4_path.name}' içindeki v4 tahminleri yeniden kullanıldı.")

    if not used_cache:
        predictor = PuzzleDifficultyPredictor(model_dir=V4_DIR)
        preds, ci_lows, ci_highs, margins, plies = [], [], [], [], []

        print(f"\n[Bölüm B] Yarışma kör test dosyası ({csv_path.name}, N={len(df):,}) v4 ile değerlendiriliyor...")
        for row in tqdm(df.itertuples(index=False), total=len(df), unit="bulmaca", desc="FedCSIS v4 Blind Test"):
            res = predictor.predict(row.FEN, row.Moves, puzzle_id=None)
            preds.append(float(res["models"]["Nihai Tahmin (Kalibreli)"]))
            ci_lows.append(res["ci_low"])
            ci_highs.append(res["ci_high"])
            margins.append(float(res["expected_margin"]))
            plies.append(int(res["features"]["solution_ply"]))

        df["predicted_rating"] = np.round(preds, 1)
        df["ci_low"] = ci_lows
        df["ci_high"] = ci_highs
        df["expected_margin"] = np.round(margins, 1)
        df["solution_ply"] = plies

        q90 = float(np.quantile(df["expected_margin"], 0.90))
        df["keep_90pct_mask"] = (df["expected_margin"] <= q90).astype(int)

        sub_cols = [
            "PuzzleId",
            "predicted_rating",
            "ci_low",
            "ci_high",
            "expected_margin",
            "keep_90pct_mask",
        ]
        df[sub_cols].to_csv(sub_path, index=False)

    rapid_cols = [f"success_prob_rapid_{e}" for e in range(1050, 2150, 100)]
    elos = np.arange(1050, 2150, 100, dtype=float)
    probs_mat = df[rapid_cols].to_numpy()
    p_min = probs_mat.min(axis=1)
    p_max = probs_mat.max(axis=1)

    easy_mask = p_min > 0.50
    mid_mask = (p_min <= 0.50) & (p_max >= 0.50)

    df["empirical_group"] = np.where(
        easy_mask,
        "Kolay (<1050 Ima)",
        np.where(mid_mask, "Orta/Ileri (1050-2050)", "Zor (>2050 Ima)"),
    )

    r_1550, p_1550 = pearsonr(df["predicted_rating"], df["success_prob_rapid_1550"])
    rho_1550, _ = spearmanr(df["predicted_rating"], df["success_prob_rapid_1550"])
    r_2050, _ = pearsonr(df["predicted_rating"], df["success_prob_rapid_2050"])

    implied_elos = []
    pred_mid = df.loc[mid_mask, "predicted_rating"].to_numpy()
    for row_p in probs_mat[mid_mask]:
        idx = int(np.where(row_p >= 0.5)[0][0])
        if idx == 0:
            implied_elos.append(elos[0])
        else:
            p0, p1 = row_p[idx - 1], row_p[idx]
            e0, e1 = elos[idx - 1], elos[idx]
            implied_elos.append(e0 + (0.5 - p0) / (p1 - p0) * (e1 - e0) if p1 > p0 else e0)
    implied_elos = np.array(implied_elos)
    mid_mae = float(mean_absolute_error(implied_elos, pred_mid))
    mid_r, _ = pearsonr(implied_elos, pred_mid)

    print("\n================ FEDCSIS 2025 KÖR TEST DOĞRULAMASI (v4 | N = 2.235) ================")
    print(f"Çok hamleli (solution_ply >= 3) oranı : %{(df['solution_ply'] >= 3).mean() * 100:.1f} ({(df['solution_ply'] >= 3).sum()}/{len(df)})")
    print(f"Tahmin Dağılımı (v4)                  : Ort = {df['predicted_rating'].mean():.1f} | Std = {df['predicted_rating'].std():.1f} | Min = {df['predicted_rating'].min():.0f} | Max = {df['predicted_rating'].max():.0f}")
    print(f"1550 Rapid Başarı Oranı Korelasyonu   : Pearson r = {r_1550:.4f} (p={p_1550:.2e}) | Spearman rho = {rho_1550:.4f}")
    print(f"2050 Rapid Başarı Oranı Korelasyonu   : Pearson r = {r_2050:.4f}")
    print(f"%50 Eşiği Geçen Grup (n={len(implied_elos)}) Uyumu  : r = {mid_r:.4f} | IRT Eşik MAE = {mid_mae:.1f} Elo")
    print("\nAmpirik İnsan Zorluk Gruplarına Göre v4 Tahmin Ortalamaları:")
    for grp_name in ["Kolay (<1050 Ima)", "Orta/Ileri (1050-2050)", "Zor (>2050 Ima)"]:
        sub = df[df["empirical_group"] == grp_name]
        print(f"  - {grp_name:24s} (n={len(sub):4d}): Ort. v4 Tahmini = {sub['predicted_rating'].mean():.1f} Elo | Ort. Rapid-1550 Başarı = %{sub['success_prob_rapid_1550'].mean()*100:.1f}")

    return df


def plot_fig1_architecture_evolution():
    stages = [
        "Ortalama\n(Baseline)",
        "v1: Yapısal\n(56 Özellik)",
        "v2: Ply-1\n(70 Özellik)",
        "v3: Ply-1&3\n(84 Özellik)",
        "v3.1: RD+Unc\n(84 Özellik)",
        "v4: Bilişsel\n(107 Özellik)",
    ]
    rmse_vals = [476.32, 282.71, 261.68, 241.38, 241.15, 231.97]
    mae_vals = [396.99, 220.57, 201.49, 185.54, 185.26, 178.74]

    ablation_files = [
        (1, V1_DIR / "ablation_structural_only.csv"),
        (2, V2_DIR / "ablation_hybrid_maia2.csv"),
        (3, V3_DIR / "ablation_v3_full.csv"),
        (4, V31_DIR / "ablation_v3_1.csv"),
        (5, V4_DIR / "ablation_v4.csv"),
    ]
    for idx, fpath in ablation_files:
        if fpath.exists():
            row = pd.read_csv(fpath).iloc[-1]
            rmse_vals[idx] = float(row["Test_RMSE"])
            mae_vals[idx] = float(row["Test_MAE"])

    x = np.arange(len(stages))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    b1 = ax.bar(x - width / 2, rmse_vals, width, label="Test RMSE (Elo)", color="#1f4e78")
    b2 = ax.bar(x + width / 2, mae_vals, width, label="Test MAE (Elo)", color="#4b9cd3")
    b1[-1].set_color("#2a9d8f")
    b2[-1].set_color("#6ec6b8")

    for bar in b1:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}", (bar.get_x() + bar.get_width() / 2, h + 5), ha="center", fontweight="bold", fontsize=8.5)
    for bar in b2:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}", (bar.get_x() + bar.get_width() / 2, h + 5), ha="center", fontsize=8.5)

    ax.set_ylabel("Hata Miktarı (Glicko-2 Elo)")
    ax.set_title("Model Mimarisi Evrimine Göre Test Hatası Düşüşü (v1 → v4 | N = 482.428)")
    ax.set_xticks(x)
    ax.set_xticklabels(stages)
    ax.set_ylim(0, max(rmse_vals) * 1.14)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_mimari_evrim_ablation.png")
    plt.close(fig)


def plot_fig2_actual_vs_predicted(y_true: np.ndarray, y_pred: np.ndarray):
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    hb = ax.hexbin(y_true, y_pred, gridsize=65, cmap="viridis", mincnt=1, bins="log")
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label("Bulmaca Yoğunluğu (log10 ölçeği)")

    line_x = np.linspace(500, 2900, 200)
    ax.plot(line_x, line_x, color="#ff4d4d", linewidth=1.8, label="İdeal Tahmin (y = x)")
    ax.plot(line_x, line_x + 250, color="#ffffff", linestyle="--", linewidth=1.1, alpha=0.85, label="±250 Elo Bandı")
    ax.plot(line_x, line_x - 250, color="#ffffff", linestyle="--", linewidth=1.1, alpha=0.85)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    stats_box = (
        f"v4 Test Kümesi: {len(y_true):,} bulmaca\n"
        f"RMSE: {rmse:.2f} Elo\n"
        f"MAE: {mae:.2f} Elo\n"
        f"R²: {r2:.4f}"
    )
    ax.text(
        0.04,
        0.95,
        stats_box,
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="white", alpha=0.9, edgecolor="#cccccc"),
        fontsize=9.5,
    )

    ax.set_xlim(500, 2900)
    ax.set_ylim(500, 2900)
    ax.set_xlabel("Gerçek Lichess Zorluk Puanı (Glicko-2)")
    ax.set_ylabel("v4 Bilişsel Hibrit Model Tahmini (Glicko-2)")
    ax.set_title("Gerçek ve Tahmin Edilen Zorluk Puanları Yoğunluk Dağılımı (v4)")
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_gercek_vs_tahmin_yogunluk.png")
    plt.close(fig)


def plot_fig3_feature_importance():
    imp_path = V4_DIR / "feature_importance_v4_gain.csv"
    if not imp_path.exists():
        return
    imp_df = pd.read_csv(imp_path)
    top_df = imp_df.head(20).iloc[::-1]

    palette = {
        "structural": "#2b5c8f",
        "maia2_ply1": "#e07a5f",
        "maia2_ply3": "#3d85c6",
        "maia2_v4_cognitive": "#d95f02",
        "v4_interactions": "#2a9d8f",
    }
    labels_tr = {
        "structural": "Yapısal Tahta (56)",
        "maia2_ply1": "Maia-2 Ply 1 (14)",
        "maia2_ply3": "Maia-2 Ply 3 (7)",
        "maia2_v4_cognitive": "Blitz/Rapid & Ply 5 (17)",
        "v4_interactions": "Derin Etkileşim (13)",
    }

    colors = [palette.get(s, "#666666") for s in top_df["source"]]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12.5, 6.2), gridspec_kw={"width_ratios": [2.15, 1.05]}
    )

    ax1.barh(top_df["feature"], top_df["gain_pct"], color=colors)
    ax1.set_xlabel("Toplam Kazanç (Gain) Payı (%)")
    ax1.set_title("En Belirleyici İlk 20 Öznitelik (v4 LightGBM)")

    grp_df = imp_df.groupby("source")["gain_pct"].sum().sort_values(ascending=True)
    grp_names = [labels_tr.get(g, g) for g in grp_df.index]
    grp_colors = [palette.get(g, "#666666") for g in grp_df.index]
    bars = ax2.barh(grp_names, grp_df.values, color=grp_colors, height=0.55)
    for b in bars:
        w = b.get_width()
        ax2.annotate(f"%{w:.1f}", (w + 0.7, b.get_y() + b.get_height() / 2), va="center", fontweight="bold")
    ax2.set_xlim(0, max(grp_df.values) * 1.18)
    ax2.set_xlabel("Grup Toplam Payı (%)")
    ax2.set_title("107 Öznitelik Ailesinin Karar Ağırlığı")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_oznitelik_onem_dagilimi.png")
    plt.close(fig)


def plot_fig4_calibration_and_uncertainty(y_true, y_ens, y_cal, uncertainty):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.8))

    bins = np.quantile(y_cal, np.linspace(0, 1, 16))
    mean_pred_ens, mean_pred_cal, mean_true = [], [], []
    for i in range(len(bins) - 1):
        m = (y_cal >= bins[i]) & (y_cal <= bins[i + 1])
        if np.sum(m) > 0:
            mean_true.append(np.mean(y_true[m]))
            mean_pred_ens.append(np.mean(y_ens[m]))
            mean_pred_cal.append(np.mean(y_cal[m]))

    ax1.plot([700, 2600], [700, 2600], "k--", linewidth=1.2, label="Mükemmel Kalibrasyon")
    ax1.plot(mean_pred_ens, mean_true, "o--", color="#e07a5f", label="Ham 3'lü v4 Topluluk")
    ax1.plot(mean_pred_cal, mean_true, "s-", color="#1f4e78", linewidth=2, label="Artık + B-Spline Kalibreli v4")
    ax1.set_xlabel("Ortalama Tahmin Edilen Puan (Elo)")
    ax1.set_ylabel("Ortalama Gerçek Lichess Puanı (Elo)")
    ax1.set_title("Güvenilirlik (Kalibrasyon) Eğrisi (v4)")
    ax1.legend(frameon=True)

    sq_err = (y_true - y_cal) ** 2
    order = np.argsort(uncertainty)
    sorted_sq = sq_err[order]
    coverages = np.linspace(0.50, 1.00, 26)
    rmses = [np.sqrt(np.mean(sorted_sq[: int(len(sorted_sq) * c)])) for c in coverages]

    ax2.plot(coverages * 100, rmses, "o-", color="#2a9d8f", linewidth=2)
    ax2.axvline(90, color="#d9534f", linestyle="--", linewidth=1.2, label="%90 Kapsama (222.2 RMSE)")
    ax2.set_xlabel("Değerlendirilen Test Verisi Oranı (%)")
    ax2.set_ylabel("Kalan Veri Üzerinde RMSE (Elo)")
    ax2.set_title("Heteroscedastic Belirsizlik Tabanlı Reddetme Analizi")
    ax2.legend(frameon=True)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_kalibrasyon_ve_belirsizlik.png")
    plt.close(fig)


def plot_fig5_tiered_errors(y_true: np.ndarray, y_pred: np.ndarray, tier_df: pd.DataFrame):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))

    sub_df = tier_df.iloc[:-1]
    labels_short = ["Başlangıç\n(<1100)", "Orta\n(1100-1500)", "İleri\n(1500-1900)", "Usta Adayı\n(1900-2300)", "Usta\n(2300+)"]
    x = np.arange(len(sub_df))
    w = 0.36

    ax1.bar(x - w / 2, sub_df["RMSE"], w, label="RMSE (Elo)", color="#1f4e78")
    ax1.bar(x + w / 2, sub_df["MAE"], w, label="MAE (Elo)", color="#4b9cd3")
    for idx, row in sub_df.iterrows():
        ax1.annotate(f"{row['MAE']:.0f}", (idx + w / 2, row["MAE"] + 4), ha="center", fontsize=8.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels_short)
    ax1.set_ylabel("Hata (Elo)")
    ax1.set_title("Zorluk Kademelerine Göre v4 RMSE ve MAE")
    ax1.legend(frameon=True)

    box_data = []
    for _, lo, hi in TIERS:
        m = (y_true >= lo) & (y_true < hi)
        box_data.append((y_pred[m] - y_true[m]))

    bp = ax2.boxplot(
        box_data,
        tick_labels=labels_short,
        showfliers=False,
        patch_artist=True,
        medianprops=dict(color="#d9534f", linewidth=1.8),
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("#dbe9f6")
    ax2.axhline(0, color="black", linestyle="--", linewidth=1.1)
    ax2.set_ylabel("Tahmin Hatası (Tahmin − Gerçek Elo)")
    ax2.set_title("Kademelere Göre Hata Dağılımı ve Yanlılık (Bias)")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig5_katmanli_hata_dagilimi.png")
    plt.close(fig)


def plot_fig6_competition_validation(comp_df: pd.DataFrame):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))

    ax1.scatter(
        comp_df["predicted_rating"],
        comp_df["success_prob_rapid_1550"] * 100.0,
        alpha=0.28,
        s=14,
        color="#1f4e78",
    )
    r_val, _ = pearsonr(comp_df["predicted_rating"], comp_df["success_prob_rapid_1550"])
    ax1.set_xlabel("v4 Kör Model Tahmini (Glicko-2 Elo)")
    ax1.set_ylabel("Gerçek 1550 Elo İnsan Çözüm Başarısı (%)")
    ax1.set_title(f"FedCSIS 2025 Kör Test: v4 Tahmin vs. İnsan Başarısı (r = {r_val:.3f})")

    groups = ["Kolay (<1050 Ima)", "Orta/Ileri (1050-2050)", "Zor (>2050 Ima)"]
    group_labels = ["Kolay\n(Tüm Elo >%50)", "Orta / İleri\n(%50 Eşiği Geçen)", "Zor\n(2050 Elo <%50)"]
    box_vals = [comp_df.loc[comp_df["empirical_group"] == g, "predicted_rating"].to_numpy() for g in groups]

    bp = ax2.boxplot(
        box_vals,
        tick_labels=group_labels,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="#d9534f", linewidth=1.8),
    )
    for patch, c in zip(bp["boxes"], ["#cdeac0", "#dbe9f6", "#f7cad0"]):
        patch.set_facecolor(c)
    ax2.set_ylabel("v4 Kör Model Tahmini (Elo)")
    ax2.set_title("Ampirik Zorluk Gruplarına Göre v4 Tahmin Ayrışması")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig6_yarisma_dis_dogrulama.png")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TBL_DIR.mkdir(parents=True, exist_ok=True)

    print("[Bölüm A] Ana test kümesi (N = 482.428) v4 yarışma metrikleri hesaplanıyor...")
    y_test, test_ens, test_cal, uncertainty = load_v4_test_predictions()
    comp_m = compute_competition_metrics(y_test, test_cal, uncertainty)

    print("\n================ FEDCSIS 2025 RESMİ METRİKLERİ (v4 - 482.428 TEST) ================")
    print(f"1. Tam Küme MSE (%100 Veri)          : {comp_m['full_mse']:,.2f} (RMSE: {comp_m['full_rmse']:.3f} | MAE: {comp_m['full_mae']:.3f} | R2: {comp_m['full_r2']:.4f})")
    print(f"2. %90 Kapsama Masked MSE            : {comp_m['masked_mse_90']:,.2f} (Masked RMSE: {comp_m['masked_rmse_90']:.3f})")
    print(f"3. Teorik En İyi (Oracle %90) MSE    : {comp_m['oracle_mse_90']:,.2f}")
    print(f"4. FedCSIS 2025 N/P Oranı            : {comp_m['np_ratio']:.3f}")

    tier_df = compute_stratified_table(y_test, test_cal)
    tier_df.to_csv(TBL_DIR / "katmanli_zorluk_analizi_v4.csv", index=False)

    print("\n================ ZORLUK KADEMELERİNE GÖRE v4 HATA ANALİZİ ================")
    print(tier_df.to_string(index=False))

    theme_df = compute_theme_error_table(y_test, test_cal)
    if theme_df is not None:
        theme_df.to_csv(TBL_DIR / "tema_bazli_hata_analizi_v4.csv", index=False)
        print("\n================ TAKTİK MOTİF (TEMA) BAZLI v4 HATA ANALİZİ ================")
        print(theme_df.to_string(index=False))

    comp_csv = next((p for p in COMP_CSV_CANDIDATES if p.exists()), None)
    comp_df = evaluate_competition_blind_csv(comp_csv)

    print("\n[Bölüm C] 300 DPI tez ve yarışma grafikleri (v4) üretiliyor...")
    plot_fig1_architecture_evolution()
    plot_fig2_actual_vs_predicted(y_test, test_cal)
    plot_fig3_feature_importance()
    plot_fig4_calibration_and_uncertainty(y_test, test_ens, test_cal, uncertainty)
    plot_fig5_tiered_errors(y_test, test_cal, tier_df)
    if comp_df is not None:
        plot_fig6_competition_validation(comp_df)

    print(f"\nTüm v4 tez çıktıları ve grafikleri '{OUT_DIR}/' klasörüne kaydedildi.")


if __name__ == "__main__":
    main()