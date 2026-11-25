import gc
import json
import sqlite3
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostRegressor, Pool
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DB_PATH = Path("data") / "puzzles.db"
OUT_DIR = Path("models_v4")

ENGINEERED_COLS_V4 = [
    # v3 etkileşimleri (7)
    "maia_chain_prob_1500",
    "maia_chain_prob_1900",
    "maia_min_prob_1500",
    "maia_log_prob_1900",
    "maia_p3_log_prob_1900",
    "maia_p3_slope",
    "maia_rank_sum_1500",
    # v4 derin varyant, tuzak ve savunma dallanma etkileşimleri (6)
    "maia_full_chain_prob_1900",
    "maia_full_chain_log_1900",
    "maia_min_chain_prob_1900",
    "maia_rank_sum_1900",
    "maia_trap_severity_1900",
    "maia_calc_burden_1900",
]

CONTEXT_CANDIDATES_V4 = [
    "solution_ply",
    "legal_moves_start",
    "total_quiet_solver_moves",
    "ends_in_mate",
    "is_mate_in_1",
    "is_sacrifice",
    "quiet_despite_checks",
    "first_move_is_quiet",
    "material_gain",
    "opp_king_attackers",
    "maia_prob_1100",
    "maia_prob_1500",
    "maia_prob_1900",
    "maia_prob_slope",
    "maia_entropy_1500",
    "maia_entropy_1900",
    "maia_move_rank_1500",
    "maia_move_rank_1900",
    "maia_top1_top2_gap_1900",
    "maia_rapid_minus_blitz_1900",
    "maia_blitz_trap_gap_1900",
    "opp_p2_entropy_1900",
    "opp_p2_sol_prob_1900",
    "has_ply3",
    "maia_p3_prob_1500",
    "maia_p3_prob_1900",
    "has_ply5",
    "maia_p5_prob_1500",
    "maia_p5_prob_1900",
    "maia_chain_prob_1900",
    "maia_full_chain_prob_1900",
    "maia_full_chain_log_1900",
    "maia_min_chain_prob_1900",
    "maia_rank_sum_1900",
    "maia_trap_severity_1900",
    "maia_calc_burden_1900",
]


def eval_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def add_v4_interactions(df: pd.DataFrame) -> None:
    p1_1500 = df["maia_prob_1500"].to_numpy(dtype=np.float32)
    p1_1900 = df["maia_prob_1900"].to_numpy(dtype=np.float32)
    p3_1100 = df["maia_p3_prob_1100"].to_numpy(dtype=np.float32)
    p3_1500 = df["maia_p3_prob_1500"].to_numpy(dtype=np.float32)
    p3_1900 = df["maia_p3_prob_1900"].to_numpy(dtype=np.float32)
    p5_1900 = df["maia_p5_prob_1900"].to_numpy(dtype=np.float32)

    has_p3 = df["has_ply3"].to_numpy(dtype=np.float32)
    has_p5 = df["has_ply5"].to_numpy(dtype=np.float32)

    r1_1500 = df["maia_move_rank_1500"].to_numpy(dtype=np.float32)
    r3_1500 = df["maia_p3_rank_1500"].to_numpy(dtype=np.float32)
    r1_1900 = df["maia_move_rank_1900"].to_numpy(dtype=np.float32)
    r3_1900 = np.maximum(1.0, df["maia_p3_rank_1500"].to_numpy(dtype=np.float32) * 0.75)
    r5_1900 = df["maia_p5_rank_1900"].to_numpy(dtype=np.float32)

    blitz_trap = df["maia_blitz_trap_gap_1900"].to_numpy(dtype=np.float32)
    top1_match_1900 = df["maia_top1_matches_1900"].to_numpy(dtype=np.float32)
    opp_ent = df["opp_p2_entropy_1900"].to_numpy(dtype=np.float32)
    opp_sol = df["opp_p2_sol_prob_1900"].to_numpy(dtype=np.float32)

    # v3 etkileşimleri
    df["maia_chain_prob_1500"] = p1_1500 * p3_1500
    df["maia_chain_prob_1900"] = p1_1900 * p3_1900
    df["maia_min_prob_1500"] = np.minimum(p1_1500, p3_1500)
    df["maia_log_prob_1900"] = np.log(np.clip(p1_1900, 1e-6, 1.0))
    df["maia_p3_log_prob_1900"] = np.log(np.clip(p3_1900, 1e-6, 1.0))
    df["maia_p3_slope"] = p3_1900 - p3_1100
    df["maia_rank_sum_1500"] = r1_1500 + (r3_1500 - 1.0) * has_p3

    # v4 etkileşimleri
    full_chain = p1_1900 * p3_1900 * p5_1900
    df["maia_full_chain_prob_1900"] = full_chain
    df["maia_full_chain_log_1900"] = np.log(np.clip(full_chain, 1e-9, 1.0))
    df["maia_min_chain_prob_1900"] = np.minimum(np.minimum(p1_1900, p3_1900), p5_1900)
    df["maia_rank_sum_1900"] = r1_1900 + (r3_1900 - 1.0) * has_p3 + (r5_1900 - 1.0) * has_p5
    df["maia_trap_severity_1900"] = blitz_trap * (1.0 - top1_match_1900)
    df["maia_calc_burden_1900"] = (1.0 - opp_sol) * opp_ent


def load_v4_dataset():
    if not DB_PATH.exists():
        raise FileNotFoundError("data/puzzles.db bulunamadı!")

    conn = sqlite3.connect(DB_PATH)
    sf_cols = [r[1] for r in conn.execute("PRAGMA table_info(structural_features);").fetchall()]
    m1_cols = [r[1] for r in conn.execute("PRAGMA table_info(maia2_features);").fetchall()]
    m3_cols = [r[1] for r in conn.execute("PRAGMA table_info(maia2_ply3_features);").fetchall()]
    m4_cols = [r[1] for r in conn.execute("PRAGMA table_info(maia2_v4_features);").fetchall()]

    if not sf_cols or not m1_cols or not m3_cols or not m4_cols:
        conn.close()
        raise RuntimeError("Gerekli veritabanı tablolarından biri eksik (maia2_v4_features kontrol edin)!")

    meta_skip = {"puzzle_id", "rating", "split", "rating_deviation"}
    struct_cols = [c for c in sf_cols if c not in meta_skip]
    maia_p1_cols = [c for c in m1_cols if c not in meta_skip]
    maia_p3_cols = [c for c in m3_cols if c not in meta_skip]
    maia_v4_cols = [c for c in m4_cols if c not in meta_skip]

    select_sf = ", ".join([f"sf.{c}" for c in struct_cols])
    select_m1 = ", ".join([f"m1.{c}" for c in maia_p1_cols])
    select_m3 = ", ".join([f"m3.{c}" for c in maia_p3_cols])
    select_m4 = ", ".join([f"m4.{c}" for c in maia_v4_cols])

    query = f"""
        SELECT sf.puzzle_id, r.rating AS rating, r.rating_deviation AS rating_deviation,
               r.split AS split, {select_sf}, {select_m1}, {select_m3}, {select_m4}
        FROM structural_features sf
        INNER JOIN maia2_features m1 ON sf.puzzle_id = m1.puzzle_id
        INNER JOIN maia2_ply3_features m3 ON sf.puzzle_id = m3.puzzle_id
        INNER JOIN maia2_v4_features m4 ON sf.puzzle_id = m4.puzzle_id
        INNER JOIN raw_puzzles r ON sf.puzzle_id = r.puzzle_id
        ORDER BY sf.puzzle_id
    """

    total_raw = len(struct_cols) + len(maia_p1_cols) + len(maia_p3_cols) + len(maia_v4_cols)
    print(f"Veritabanından {total_raw} ham öznitelik yükleniyor (56 yapısal + 14 Ply1 + 7 Ply3 + 17 v4)...")
    t0 = time.time()
    df = pd.read_sql_query(query, conn)
    conn.close()
    print(f"Toplam {len(df):,} satır yüklendi ({time.time() - t0:.1f} sn).")

    add_v4_interactions(df)
    feature_cols = struct_cols + maia_p1_cols + maia_p3_cols + maia_v4_cols + ENGINEERED_COLS_V4
    print(f"v4 etkileşim öznitelikleri eklendi -> Toplam özellik sayısı: {len(feature_cols)}")

    df["split"] = df["split"].astype(str).str.lower().replace({"validation": "val"})

    X_all = df[feature_cols].to_numpy(dtype=np.float32, copy=True)
    np.nan_to_num(X_all, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    y_all = df["rating"].to_numpy(dtype=np.float32)
    rd_all = df["rating_deviation"].to_numpy(dtype=np.float32)

    train_mask = (df["split"] == "train").to_numpy()
    val_mask = (df["split"] == "val").to_numpy()
    test_mask = (df["split"] == "test").to_numpy()

    context_cols = [c for c in CONTEXT_CANDIDATES_V4 if c in feature_cols]
    context_idx = [feature_cols.index(c) for c in context_cols]

    X_train, y_train, rd_train = X_all[train_mask], y_all[train_mask], rd_all[train_mask]
    X_val, y_val = X_all[val_mask], y_all[val_mask]
    X_test, y_test = X_all[test_mask], y_all[test_mask]

    rd_med = float(np.median(rd_train))
    raw_w = (rd_med / np.clip(rd_train, 40.0, 120.0)) ** 2
    clipped_w = np.clip(raw_w, 0.35, 2.25)
    w_train = (clipped_w / np.mean(clipped_w)).astype(np.float32)

    val_context = X_val[:, context_idx].copy()
    test_context = X_test[:, context_idx].copy()

    del df, X_all, y_all, rd_all
    gc.collect()

    print(f"Bölünme -> Train: {len(y_train):,} | Val: {len(y_val):,} | Test: {len(y_test):,}")
    return (
        feature_cols,
        struct_cols,
        maia_p1_cols,
        maia_p3_cols,
        maia_v4_cols,
        X_train,
        y_train,
        w_train,
        X_val,
        y_val,
        X_test,
        y_test,
        context_cols,
        val_context,
        test_context,
    )


def optimize_ensemble_weights(val_preds_matrix: np.ndarray, y_val: np.ndarray) -> np.ndarray:
    n_models = val_preds_matrix.shape[1]
    init_w = np.ones(n_models, dtype=np.float64) / n_models

    def loss_fn(weights):
        blend = val_preds_matrix @ weights
        return float(np.sqrt(np.mean((y_val - blend) ** 2)))

    res = minimize(
        loss_fn,
        init_w,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_models,
        constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
    )
    return res.x if res.success else init_w


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (
        feature_cols,
        struct_cols,
        maia_p1_cols,
        maia_p3_cols,
        maia_v4_cols,
        X_train,
        y_train,
        w_train,
        X_val,
        y_val,
        X_test,
        y_test,
        context_cols,
        val_context,
        test_context,
    ) = load_v4_dataset()

    # 1. LightGBM (107 Öznitelik, 9.500 İterasyon)
    print("\n[1/3] v4 LightGBM eğitiliyor (107 öznitelik, 9.500 iterasyon)...")
    t0 = time.time()
    lgb_model = lgb.LGBMRegressor(
        n_estimators=9500,
        learning_rate=0.042,
        num_leaves=319,
        min_child_samples=45,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.70,
        reg_alpha=1.0,
        reg_lambda=5.0,
        n_jobs=-1,
        random_state=42,
        verbose=-1,
    )
    lgb_model.fit(
        X_train,
        y_train,
        sample_weight=w_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(500)],
    )
    val_lgb = lgb_model.predict(X_val).astype(np.float32)
    test_lgb = lgb_model.predict(X_test).astype(np.float32)
    print(f"LightGBM tamamlandı ({(time.time() - t0)/60:.1f} dk) -> Test RMSE: {eval_metrics(y_test, test_lgb)['rmse']:.3f}")

    gains = lgb_model.booster_.feature_importance(importance_type="gain")
    sources = []
    for c in feature_cols:
        if c in struct_cols:
            sources.append("structural")
        elif c in maia_p1_cols:
            sources.append("maia2_ply1")
        elif c in maia_p3_cols:
            sources.append("maia2_ply3")
        elif c in maia_v4_cols:
            sources.append("maia2_v4_cognitive")
        else:
            sources.append("v4_interactions")

    imp_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "source": sources,
            "gain": gains,
            "gain_pct": 100.0 * gains / np.sum(gains),
        }
    ).sort_values("gain", ascending=False)
    imp_df.to_csv(OUT_DIR / "feature_importance_v4_gain.csv", index=False)

    print("Öznitelik Gruplarına Göre Karar Ağırlığı Payları:")
    for grp, pct in imp_df.groupby("source")["gain_pct"].sum().sort_values(ascending=False).items():
        print(f"  - {grp:22s}: %{pct:.2f}")

    joblib.dump(lgb_model, OUT_DIR / "lgb_v4.joblib")
    del lgb_model
    gc.collect()

    # 2. XGBoost (RTX 3050 CUDA, max_depth=9, 8.000 İterasyon)
    print("\n[2/3] v4 XGBoost eğitiliyor (RTX 3050 CUDA, depth=9, 8.000 iterasyon)...")
    t0 = time.time()
    xgb_params = dict(
        n_estimators=8000,
        learning_rate=0.045,
        max_depth=9,
        min_child_weight=45,
        subsample=0.8,
        colsample_bytree=0.70,
        reg_alpha=1.0,
        reg_lambda=5.0,
        tree_method="hist",
        device="cuda",
        early_stopping_rounds=150,
        eval_metric="rmse",
        random_state=42,
    )
    try:
        xgb_model = xgb.XGBRegressor(**xgb_params)
        xgb_model.fit(X_train, y_train, sample_weight=w_train, eval_set=[(X_val, y_val)], verbose=500)
    except Exception as e:
        print(f"CUDA uyarısı ({e}), depth=8 ile yeniden deneniyor...")
        xgb_params["max_depth"] = 8
        xgb_model = xgb.XGBRegressor(**xgb_params)
        xgb_model.fit(X_train, y_train, sample_weight=w_train, eval_set=[(X_val, y_val)], verbose=500)

    val_xgb = xgb_model.predict(X_val).astype(np.float32)
    test_xgb = xgb_model.predict(X_test).astype(np.float32)
    print(f"XGBoost tamamlandı ({(time.time() - t0)/60:.1f} dk) -> Test RMSE: {eval_metrics(y_test, test_xgb)['rmse']:.3f}")

    xgb_model.save_model(str(OUT_DIR / "xgb_v4.json"))
    del xgb_model
    gc.collect()

    # 3. CatBoost (RTX 3050 GPU, depth=9, 7.500 İterasyon)
    print("\n[3/3] v4 CatBoost eğitiliyor (RTX 3050 GPU, depth=9, 7.500 iterasyon)...")
    t0 = time.time()
    train_pool = Pool(X_train, y_train, weight=w_train)
    val_pool = Pool(X_val, y_val)
    try:
        cat_model = CatBoostRegressor(
            iterations=7500,
            learning_rate=0.06,
            depth=9,
            l2_leaf_reg=5.0,
            loss_function="RMSE",
            eval_metric="RMSE",
            task_type="GPU",
            random_seed=42,
            early_stopping_rounds=150,
            verbose=500,
        )
        cat_model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    except Exception as e:
        print(f"CatBoost GPU depth=9 uyarısı ({e}), depth=8 ile devam ediliyor...")
        cat_model = CatBoostRegressor(
            iterations=7500,
            learning_rate=0.06,
            depth=8,
            l2_leaf_reg=5.0,
            loss_function="RMSE",
            eval_metric="RMSE",
            task_type="GPU",
            random_seed=42,
            early_stopping_rounds=150,
            verbose=500,
        )
        cat_model.fit(train_pool, eval_set=val_pool, use_best_model=True)

    val_cat = cat_model.predict(X_val).astype(np.float32)
    test_cat = cat_model.predict(X_test).astype(np.float32)
    print(f"CatBoost tamamlandı ({(time.time() - t0)/60:.1f} dk) -> Test RMSE: {eval_metrics(y_test, test_cat)['rmse']:.3f}")

    cat_model.save_model(str(OUT_DIR / "cat_v4.cbm"))
    del cat_model, train_pool, val_pool, X_train, X_val, X_test
    gc.collect()

    val_mat = np.column_stack([val_lgb, val_xgb, val_cat])
    test_mat = np.column_stack([test_lgb, test_xgb, test_cat])
    weights = optimize_ensemble_weights(val_mat, y_val)

    val_ens = (val_mat @ weights).astype(np.float32)
    test_ens = (test_mat @ weights).astype(np.float32)
    ens_test_m = eval_metrics(y_test, test_ens)

    print("\n================ v4 TAM BİLİŞSEL & DERİN VARYANT (107 ÖZNİTELİK) ÖZETİ ================")
    print(f"Optimal Ağırlıklar -> LGB: {weights[0]:.3f} | XGB: {weights[1]:.3f} | CAT: {weights[2]:.3f}")
    print(f"3'lü v4 Topluluk Test RMSE: {ens_test_m['rmse']:.3f} | MAE: {ens_test_m['mae']:.3f} | R2: {ens_test_m['r2']:.4f}")

    with open(OUT_DIR / "feature_cols.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "feature_cols": feature_cols,
                "struct_cols": struct_cols,
                "maia_p1_cols": maia_p1_cols,
                "maia_p3_cols": maia_p3_cols,
                "maia_v4_cols": maia_v4_cols,
                "engineered_cols": ENGINEERED_COLS_V4,
                "context_cols": context_cols,
            },
            f,
            indent=2,
        )

    with open(OUT_DIR / "ensemble_weights.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "lgb": float(weights[0]),
                "xgb": float(weights[1]),
                "cat": float(weights[2]),
                "train_mean": float(np.mean(y_train)),
            },
            f,
            indent=2,
        )

    np.savez_compressed(
        OUT_DIR / "v4_oof_and_test.npz",
        y_val=y_val,
        val_lgb=val_lgb,
        val_xgb=val_xgb,
        val_cat=val_cat,
        val_ens=val_ens,
        val_context=val_context,
        y_test=y_test,
        test_lgb=test_lgb,
        test_xgb=test_xgb,
        test_cat=test_cat,
        test_ens=test_ens,
        test_context=test_context,
    )
    print(f"Tüm v4 modeller ve ara çıktılar '{OUT_DIR}/' klasörüne kaydedildi.")


if __name__ == "__main__":
    main()