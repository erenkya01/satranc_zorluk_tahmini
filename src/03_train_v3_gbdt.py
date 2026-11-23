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
from catboost import CatBoostRegressor
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DB_PATH = Path("data") / "puzzles.db"
OUT_DIR = Path("models_v3_full")

ENGINEERED_COLS = [
    "maia_chain_prob_1500",
    "maia_chain_prob_1900",
    "maia_min_prob_1500",
    "maia_log_prob_1900",
    "maia_p3_log_prob_1900",
    "maia_p3_slope",
    "maia_rank_sum_1500",
]

CONTEXT_CANDIDATES = [
    "solution_ply",
    "legal_moves_start",
    "total_quiet_solver_moves",
    "ends_in_mate",
    "is_sacrifice",
    "maia_prob_1500",
    "maia_prob_slope",
    "maia_entropy_1500",
    "maia_move_rank_1500",
    "has_ply3",
    "maia_p3_prob_1500",
    "maia_chain_prob_1500",
    "maia_min_prob_1500",
]


def eval_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def add_v3_interactions(df: pd.DataFrame) -> None:
    """Ply 1 ve Ply 3 Maia-2 sinyallerini birleştiren 7 etkileşim özniteliğini üretir."""
    p1_1500 = df["maia_prob_1500"].to_numpy(dtype=np.float32)
    p1_1900 = df["maia_prob_1900"].to_numpy(dtype=np.float32)
    p3_1100 = df["maia_p3_prob_1100"].to_numpy(dtype=np.float32)
    p3_1500 = df["maia_p3_prob_1500"].to_numpy(dtype=np.float32)
    p3_1900 = df["maia_p3_prob_1900"].to_numpy(dtype=np.float32)
    has_p3 = df["has_ply3"].to_numpy(dtype=np.float32)
    r1_1500 = df["maia_move_rank_1500"].to_numpy(dtype=np.float32)
    r3_1500 = df["maia_p3_rank_1500"].to_numpy(dtype=np.float32)

    df["maia_chain_prob_1500"] = p1_1500 * p3_1500
    df["maia_chain_prob_1900"] = p1_1900 * p3_1900
    df["maia_min_prob_1500"] = np.minimum(p1_1500, p3_1500)
    df["maia_log_prob_1900"] = np.log(np.clip(p1_1900, 1e-6, 1.0))
    df["maia_p3_log_prob_1900"] = np.log(np.clip(p3_1900, 1e-6, 1.0))
    df["maia_p3_slope"] = p3_1900 - p3_1100
    df["maia_rank_sum_1500"] = r1_1500 + (r3_1500 - 1.0) * has_p3


def load_v3_dataset():
    if not DB_PATH.exists():
        raise FileNotFoundError("data/puzzles.db bulunamadı!")

    conn = sqlite3.connect(DB_PATH)
    sf_cols = [r[1] for r in conn.execute("PRAGMA table_info(structural_features);").fetchall()]
    m1_cols = [r[1] for r in conn.execute("PRAGMA table_info(maia2_features);").fetchall()]
    m3_cols = [r[1] for r in conn.execute("PRAGMA table_info(maia2_ply3_features);").fetchall()]

    if not sf_cols or not m1_cols or not m3_cols:
        conn.close()
        raise RuntimeError("Gerekli veritabanı tablolarından biri eksik!")

    meta_skip = {"puzzle_id", "rating", "split"}
    struct_cols = [c for c in sf_cols if c not in meta_skip]
    maia_p1_cols = [c for c in m1_cols if c not in meta_skip]
    maia_p3_cols = [c for c in m3_cols if c not in meta_skip]

    select_sf = ", ".join([f"sf.{c}" for c in struct_cols])
    select_m1 = ", ".join([f"m1.{c}" for c in maia_p1_cols])
    select_m3 = ", ".join([f"m3.{c}" for c in maia_p3_cols])

    query = f"""
        SELECT sf.puzzle_id, r.rating AS rating, r.split AS split,
               {select_sf}, {select_m1}, {select_m3}
        FROM structural_features sf
        INNER JOIN maia2_features m1 ON sf.puzzle_id = m1.puzzle_id
        INNER JOIN maia2_ply3_features m3 ON sf.puzzle_id = m3.puzzle_id
        INNER JOIN raw_puzzles r ON sf.puzzle_id = r.puzzle_id
        ORDER BY sf.puzzle_id
    """

    total_raw = len(struct_cols) + len(maia_p1_cols) + len(maia_p3_cols)
    print(f"Veritabanından {total_raw} ham öznitelik yükleniyor (56 yapısal + 14 Ply1 + 7 Ply3)...")
    t0 = time.time()
    df = pd.read_sql_query(query, conn)
    conn.close()
    print(f"Toplam {len(df):,} satır yüklendi ({time.time() - t0:.1f} sn).")

    add_v3_interactions(df)
    feature_cols = struct_cols + maia_p1_cols + maia_p3_cols + ENGINEERED_COLS
    print(f"Etkileşim öznitelikleri eklendi -> Toplam özellik sayısı: {len(feature_cols)}")

    df["split"] = df["split"].astype(str).str.lower().replace({"validation": "val"})

    X_all = df[feature_cols].to_numpy(dtype=np.float32, copy=True)
    np.nan_to_num(X_all, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    y_all = df["rating"].to_numpy(dtype=np.float32)

    train_mask = (df["split"] == "train").to_numpy()
    val_mask = (df["split"] == "val").to_numpy()
    test_mask = (df["split"] == "test").to_numpy()

    context_cols = [c for c in CONTEXT_CANDIDATES if c in feature_cols]
    context_idx = [feature_cols.index(c) for c in context_cols]

    X_train, y_train = X_all[train_mask], y_all[train_mask]
    X_val, y_val = X_all[val_mask], y_all[val_mask]
    X_test, y_test = X_all[test_mask], y_all[test_mask]

    val_context = X_val[:, context_idx].copy()
    test_context = X_test[:, context_idx].copy()

    del df, X_all, y_all
    gc.collect()

    print(f"Bölünme -> Train: {len(y_train):,} | Val: {len(y_val):,} | Test: {len(y_test):,}")
    return (
        feature_cols,
        struct_cols,
        maia_p1_cols,
        maia_p3_cols,
        X_train,
        y_train,
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
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        context_cols,
        val_context,
        test_context,
    ) = load_v3_dataset()

    # 1. LightGBM (84 Öznitelik, 7.500 İterasyon Kapasitesi)
    print("\n[1/3] v3 LightGBM eğitiliyor (7.500 iterasyon kapasitesi)...")
    t0 = time.time()
    lgb_model = lgb.LGBMRegressor(
        n_estimators=7500,
        learning_rate=0.04,
        num_leaves=255,
        min_child_samples=80,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.72,
        reg_alpha=1.0,
        reg_lambda=5.0,
        n_jobs=-1,
        random_state=42,
        verbose=-1,
    )
    lgb_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(120, verbose=False), lgb.log_evaluation(500)],
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
        else:
            sources.append("maia2_interactions")

    imp_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "source": sources,
            "gain": gains,
            "gain_pct": 100.0 * gains / np.sum(gains),
        }
    ).sort_values("gain", ascending=False)
    imp_df.to_csv(OUT_DIR / "feature_importance_v3_gain.csv", index=False)

    print("Öznitelik Gruplarına Göre Karar Ağırlığı Payları:")
    for grp, pct in imp_df.groupby("source")["gain_pct"].sum().sort_values(ascending=False).items():
        print(f"  - {grp:20s}: %{pct:.2f}")

    joblib.dump(lgb_model, OUT_DIR / "lgb_v3.joblib")
    del lgb_model
    gc.collect()

    # 2. XGBoost (RTX 3050 CUDA)
    print("\n[2/3] v3 XGBoost eğitiliyor (RTX 3050 CUDA)...")
    t0 = time.time()
    xgb_params = dict(
        n_estimators=6500,
        learning_rate=0.04,
        max_depth=8,
        min_child_weight=80,
        subsample=0.8,
        colsample_bytree=0.72,
        reg_alpha=1.0,
        reg_lambda=5.0,
        tree_method="hist",
        device="cuda",
        early_stopping_rounds=120,
        eval_metric="rmse",
        random_state=42,
    )
    try:
        xgb_model = xgb.XGBRegressor(**xgb_params)
        xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=500)
    except Exception as e:
        print(f"CUDA uyarısı ({e}), CPU hist moduna geçiliyor...")
        xgb_params["device"] = "cpu"
        xgb_params["n_jobs"] = -1
        xgb_model = xgb.XGBRegressor(**xgb_params)
        xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=500)

    val_xgb = xgb_model.predict(X_val).astype(np.float32)
    test_xgb = xgb_model.predict(X_test).astype(np.float32)
    print(f"XGBoost tamamlandı ({(time.time() - t0)/60:.1f} dk) -> Test RMSE: {eval_metrics(y_test, test_xgb)['rmse']:.3f}")

    xgb_model.save_model(str(OUT_DIR / "xgb_v3.json"))
    del xgb_model
    gc.collect()

    # 3. CatBoost (RTX 3050 GPU, Derinlik 9)
    print("\n[3/3] v3 CatBoost eğitiliyor (RTX 3050 GPU, depth=9)...")
    t0 = time.time()
    try:
        cat_model = CatBoostRegressor(
            iterations=6000,
            learning_rate=0.055,
            depth=9,
            l2_leaf_reg=5.0,
            loss_function="RMSE",
            eval_metric="RMSE",
            task_type="GPU",
            random_seed=42,
            early_stopping_rounds=150,
            verbose=500,
        )
        cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
    except Exception as e:
        print(f"CatBoost GPU depth=9 uyarısı ({e}), depth=8 ile devam ediliyor...")
        cat_model = CatBoostRegressor(
            iterations=6000,
            learning_rate=0.055,
            depth=8,
            l2_leaf_reg=5.0,
            loss_function="RMSE",
            eval_metric="RMSE",
            task_type="GPU",
            random_seed=42,
            early_stopping_rounds=150,
            verbose=500,
        )
        cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)

    val_cat = cat_model.predict(X_val).astype(np.float32)
    test_cat = cat_model.predict(X_test).astype(np.float32)
    print(f"CatBoost tamamlandı ({(time.time() - t0)/60:.1f} dk) -> Test RMSE: {eval_metrics(y_test, test_cat)['rmse']:.3f}")

    cat_model.save_model(str(OUT_DIR / "cat_v3.cbm"))
    del cat_model, X_train, X_val, X_test
    gc.collect()

    # Optimal Topluluk Ağırlıkları
    val_mat = np.column_stack([val_lgb, val_xgb, val_cat])
    test_mat = np.column_stack([test_lgb, test_xgb, test_cat])
    weights = optimize_ensemble_weights(val_mat, y_val)

    val_ens = (val_mat @ weights).astype(np.float32)
    test_ens = (test_mat @ weights).astype(np.float32)
    ens_test_m = eval_metrics(y_test, test_ens)

    print("\n================ v3 HİBRİT (84 ÖZNİTELİK) FAZ 3 ÖZETİ ================")
    print(f"Optimal Ağırlıklar -> LGB: {weights[0]:.3f} | XGB: {weights[1]:.3f} | CAT: {weights[2]:.3f}")
    print(f"3'lü v3 Topluluk Test RMSE: {ens_test_m['rmse']:.3f} | MAE: {ens_test_m['mae']:.3f} | R2: {ens_test_m['r2']:.4f}")

    with open(OUT_DIR / "feature_cols.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "feature_cols": feature_cols,
                "struct_cols": struct_cols,
                "maia_p1_cols": maia_p1_cols,
                "maia_p3_cols": maia_p3_cols,
                "engineered_cols": ENGINEERED_COLS,
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
        OUT_DIR / "v3_oof_and_test.npz",
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
    print(f"Tüm v3 modeller ve ara çıktılar '{OUT_DIR}/' klasörüne kaydedildi.")


if __name__ == "__main__":
    main()