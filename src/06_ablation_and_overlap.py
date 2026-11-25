import sqlite3
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT_DIR = Path(__file__).resolve().parent.parent
DB_PATH = ROOT_DIR / "data" / "puzzles.db"
V4_DIR = ROOT_DIR / "models_v4"
COMP_CSV_CANDIDATES = [
    ROOT_DIR / "testing_data_cropped.csv",
    ROOT_DIR / "data" / "testing_data_cropped.csv",
]


def check_external_overlap():
    print("=" * 72)
    print("1. DIŞ KÜME (testing_data_cropped.csv) VERİ ÖRTÜŞMESİ DENETİMİ")
    print("=" * 72)

    comp_csv = next((p for p in COMP_CSV_CANDIDATES if p.exists()), None)
    ext_v4 = V4_DIR / "external_validation_v4.csv"
    if comp_csv is None or not ext_v4.exists():
        print("Dış doğrulama dosyaları bulunamadı, bu adım atlandı.")
        return

    df_comp = pd.read_csv(comp_csv)
    df_pred = pd.read_csv(ext_v4)
    df_comp["pred_rating"] = df_pred["pred_rating"]
    df_comp["fen_norm"] = df_comp["FEN"].str.split().str[:4].str.join(" ")

    conn = sqlite3.connect(DB_PATH)
    db_df = pd.read_sql_query("SELECT puzzle_id, fen, split FROM raw_puzzles", conn)
    conn.close()

    db_df["split"] = db_df["split"].str.lower().replace({"validation": "val"})
    db_df["fen_norm"] = db_df["fen"].str.split().str[:4].str.join(" ")

    id_to_split = dict(zip(db_df["puzzle_id"], db_df["split"]))
    fen_to_split = dict(zip(db_df["fen_norm"], db_df["split"]))

    df_comp["match_id_split"] = df_comp["PuzzleId"].map(id_to_split).fillna("unseen_db")
    df_comp["match_fen_split"] = df_comp["fen_norm"].map(fen_to_split).fillna("unseen_db")

    print("\n[A] PuzzleId Bazlı Eşleşme Dağılımı (N = 2.235):")
    print(df_comp["match_id_split"].value_counts().to_string())

    print("\n[B] Normalize FEN Pozisyon Bazlı Eşleşme Dağılımı (N = 2.235):")
    print(df_comp["match_fen_split"].value_counts().to_string())

    # Eğitim kümesinde (train) hiç yer almayan kesin sızıntısız alt küme
    clean_mask = (df_comp["match_id_split"] != "train") & (df_comp["match_fen_split"] != "train")
    df_clean = df_comp[clean_mask].copy()

    r_all, _ = pearsonr(df_comp["pred_rating"], df_comp["success_prob_rapid_1550"])
    rho_all, _ = spearmanr(df_comp["pred_rating"], df_comp["success_prob_rapid_1550"])

    r_clean, p_clean = pearsonr(df_clean["pred_rating"], df_clean["success_prob_rapid_1550"])
    rho_clean, _ = spearmanr(df_clean["pred_rating"], df_clean["success_prob_rapid_1550"])

    print(f"\nToplam Dış Küme (N={len(df_comp):,})          -> Pearson r = {r_all:.4f} | Spearman rho = {rho_all:.4f}")
    print(f"Train Dışı Temiz Küme (N={len(df_clean):,})  -> Pearson r = {r_clean:.4f} (p={p_clean:.2e}) | Spearman rho = {rho_clean:.4f}")

    rapid_cols = [f"success_prob_rapid_{e}" for e in range(1050, 2150, 100)]
    probs_mat = df_clean[rapid_cols].to_numpy()
    easy = probs_mat.min(axis=1) > 0.50
    mid = (probs_mat.min(axis=1) <= 0.50) & (probs_mat.max(axis=1) >= 0.50)
    hard = probs_mat.max(axis=1) < 0.50

    print("\nTrain-Dışı Temiz Alt Kümede Ampirik Grup Ayrışması:")
    for name, m in [("Kolay (<1050)", easy), ("Orta/İleri (1050-2050)", mid), ("Zor (>2050)", hard)]:
        sub = df_clean[m]
        print(f"  - {name:22s} (n={len(sub):4d}): Ort. v4 = {sub['pred_rating'].mean():.1f} Elo | 1550 Rapid Başarı = %{sub['success_prob_rapid_1550'].mean()*100:.1f}")


def get_table_cols(conn, table_name):
    info = pd.read_sql_query(f"PRAGMA table_info({table_name})", conn)
    return [c for c in info["name"].tolist() if c != "puzzle_id"]


def run_controlled_feature_ablation():
    print("\n" + "=" * 72)
    print("2. KONTROLLÜ ÖZNİTELİK GRUBU ABLASYONU (AYNI LIGHTGBM HİPERPARAMETRELERİ)")
    print("=" * 72)

    conn = sqlite3.connect(DB_PATH)
    sf_cols = get_table_cols(conn, "structural_features")
    m1_cols = get_table_cols(conn, "maia2_features")
    m3_cols = get_table_cols(conn, "maia2_ply3_features")
    m4_cols = get_table_cols(conn, "maia2_v4_features")

    sel_sf = ", ".join(f"sf.{c}" for c in sf_cols)
    sel_m1 = ", ".join(f"m1.{c}" for c in m1_cols)
    sel_m3 = ", ".join(f"m3.{c}" for c in m3_cols)
    sel_m4 = ", ".join(f"m4.{c}" for c in m4_cols)

    query = f"""
        SELECT r.rating, r.rating_deviation, r.split,
               {sel_sf}, {sel_m1}, {sel_m3}, {sel_m4}
        FROM structural_features sf
        INNER JOIN maia2_features m1 ON sf.puzzle_id = m1.puzzle_id
        INNER JOIN maia2_ply3_features m3 ON sf.puzzle_id = m3.puzzle_id
        INNER JOIN maia2_v4_features m4 ON sf.puzzle_id = m4.puzzle_id
        INNER JOIN raw_puzzles r ON sf.puzzle_id = r.puzzle_id
    """
    t0 = time.time()
    df = pd.read_sql_query(query, conn)
    conn.close()
    print(f"Veri yüklendi: {len(df):,} satır ({time.time() - t0:.1f} sn)")

    # v3 + v4 etkileşim özniteliklerini üret (03_train_v4_gbdt.py ile birebir aynı)
    from importlib.util import module_from_spec, spec_from_file_location
    spec = spec_from_file_location("train_v4", ROOT_DIR / "src" / "03_train_v4_gbdt.py")
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.add_v4_interactions(df)

    v3_inter_cols = [c for c in mod.ENGINEERED_COLS_V4 if not any(k in c for k in ["full_chain", "trap_severity", "calc_burden", "blitz", "p5", "opp_p2"])]
    v4_inter_cols = [c for c in mod.ENGINEERED_COLS_V4 if c not in v3_inter_cols]

    base_v31_cols = sf_cols + m1_cols + m3_cols + v3_inter_cols

    # Yeni v4 ham özniteliklerini 4 bilişsel alt gruba ayır
    grp_blitz = [c for c in m4_cols if "blitz" in c]
    grp_p2 = [c for c in m4_cols if "opp_p2" in c]
    grp_p5 = [c for c in m4_cols if "_p5_" in c]
    grp_rank_gap = [c for c in m4_cols if c not in grp_blitz + grp_p2 + grp_p5]

    df["split"] = df["split"].str.lower().replace({"validation": "val"})
    tr = df["split"] == "train"
    va = df["split"] == "val"
    te = df["split"] == "test"

    y_tr = df.loc[tr, "rating"].to_numpy(dtype=np.float32)
    y_va = df.loc[va, "rating"].to_numpy(dtype=np.float32)
    y_te = df.loc[te, "rating"].to_numpy(dtype=np.float32)

    rd_tr = np.clip(df.loc[tr, "rating_deviation"].to_numpy(dtype=np.float32), 40.0, 120.0)
    rd_med = float(np.median(rd_tr))
    w_tr = np.clip((rd_med / rd_tr) ** 2, 0.35, 2.25)
    w_tr = w_tr / np.mean(w_tr)

    experiments = [
        ("A. v3.1 Taban (84 Özellik, Aynı v4 Hiperparametreleri)", base_v31_cols),
        ("B. + Blitz vs. Rapid Tuzak Ayrışması", base_v31_cols + grp_blitz),
        ("C. + 1900 Elo Aday Hamle Sırası & Top1-Top2 Payı", base_v31_cols + grp_rank_gap),
        ("D. + Rakip Savunma Dallanması (Ply 2 Entropi)", base_v31_cols + grp_p2),
        ("E. + Derin Varyant Hesaplaması (Ply 5)", base_v31_cols + grp_p5),
        ("F. Tam v4 (Tüm Yeni Gruplar + Derin Etkileşimler = 107)", base_v31_cols + m4_cols + v4_inter_cols),
    ]

    lgb_params = dict(
        objective="regression",
        metric="rmse",
        n_estimators=9500,
        learning_rate=0.042,
        num_leaves=319,
        max_depth=13,
        min_child_samples=85,
        subsample=0.82,
        subsample_freq=1,
        colsample_bytree=0.72,
        reg_alpha=12.0,
        reg_lambda=35.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    results = []
    last_model = None
    last_cols = None

    for label, cols in experiments:
        t1 = time.time()
        model = lgb.LGBMRegressor(**lgb_params)
        model.fit(
            df.loc[tr, cols],
            y_tr,
            sample_weight=w_tr,
            eval_set=[(df.loc[va, cols], y_va)],
            callbacks=[lgb.early_stopping(150, verbose=False)],
        )
        preds = model.predict(df.loc[te, cols])
        rmse = float(np.sqrt(mean_squared_error(y_te, preds)))
        mae = float(mean_absolute_error(y_te, preds))
        r2 = float(r2_score(y_te, preds))
        results.append({
            "Deney_Grubu": label,
            "Ozellik_Sayisi": len(cols),
            "Test_RMSE": round(rmse, 3),
            "Test_MAE": round(mae, 3),
            "Test_R2": round(r2, 4),
            "Sure_sn": round(time.time() - t1, 1),
        })
        print(f"{label:56s} | k={len(cols):3d} | RMSE={rmse:.3f} | MAE={mae:.3f}")
        last_model = model
        last_cols = cols

    res_df = pd.DataFrame(results)
    res_df["Fark_vs_v31_RMSE"] = (res_df["Test_RMSE"] - res_df.loc[0, "Test_RMSE"]).round(3)
    res_df.to_csv(ROOT_DIR / "reports" / "tables" / "v4_feature_group_ablation.csv", index=False)

    # 3. Permutation Importance (N = 50.000 test alt kümesi üzerinde)
    print("\n50.000 test örneği üzerinde Permutation Importance hesaplanıyor...")
    rng = np.random.default_rng(42)
    te_idx = np.where(te)[0]
    sub_idx = rng.choice(te_idx, size=min(50_000, len(te_idx)), replace=False)
    perm = permutation_importance(
        last_model,
        df.iloc[sub_idx][last_cols],
        df.iloc[sub_idx]["rating"].to_numpy(dtype=np.float32),
        scoring="neg_root_mean_squared_error",
        n_repeats=3,
        random_state=42,
        n_jobs=-1,
    )
    perm_df = pd.DataFrame({
        "feature": last_cols,
        "rmse_increase_when_shuffled": np.round(perm.importances_mean, 3),
        "std": np.round(perm.importances_std, 3),
    }).sort_values("rmse_increase_when_shuffled", ascending=False)
    perm_df.to_csv(ROOT_DIR / "reports" / "tables" / "v4_permutation_importance.csv", index=False)
    print("\nEn Kritik İlk 15 Öznitelik (Permutation Importance - RMSE Artışı):")
    print(perm_df.head(15).to_string(index=False))


if __name__ == "__main__":
    check_external_overlap()
    run_controlled_feature_ablation()