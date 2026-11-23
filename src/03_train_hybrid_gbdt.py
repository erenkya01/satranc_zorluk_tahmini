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
OUT_DIR = Path("models_hybrid_maia2")

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
]


def eval_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def load_hybrid_dataset():
    if not DB_PATH.exists():
        raise FileNotFoundError("data/puzzles.db bulunamadı!")

    conn = sqlite3.connect(DB_PATH)
    sf_cols = [r[1] for r in conn.execute("PRAGMA table_info(structural_features);").fetchall()]
    m2_cols = [r[1] for r in conn.execute("PRAGMA table_info(maia2_features);").fetchall()]

    if not sf_cols or not m2_cols:
        conn.close()
        raise RuntimeError("structural_features veya maia2_features tablosu eksik!")

    meta_skip = {"puzzle_id", "rating", "split"}
    struct_cols = [c for c in sf_cols if c not in meta_skip]
    maia_cols = [c for c in m2_cols if c not in meta_skip]
    feature_cols = struct_cols + maia_cols

    rating_expr = "sf.rating" if "rating" in sf_cols else "r.rating"
    split_expr = "sf.split" if "split" in sf_cols else "r.split"

    select_sf = ", ".join([f"sf.{c}" for c in struct_cols])
    select_m2 = ", ".join([f"m.{c}" for c in maia_cols])

    query = f"""
        SELECT sf.puzzle_id, {rating_expr} AS rating, {split_expr} AS split,
               {select_sf}, {select_m2}
        FROM structural_features sf
        INNER JOIN maia2_features m ON sf.puzzle_id = m.puzzle_id
        INNER JOIN raw_puzzles r ON sf.puzzle_id = r.puzzle_id
        ORDER BY sf.puzzle_id
    """

    print(f"Veritabanından {len(feature_cols)} öznitelik ({len(struct_cols)} yapısal + {len(maia_cols)} Maia-2) yükleniyor...")
    t0 = time.time()
    df = pd.read_sql_query(query, conn)
    conn.close()
    print(f"Toplam {len(df):,} satır yüklendi ({time.time() - t0:.1f} sn).")

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
        maia_cols,
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
        maia_cols,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        context_cols,
        val_context,
        test_context,
    ) = load_hybrid_dataset()

    # 1. LightGBM (70 Öznitelik, Genişletilmiş Kapasite)
    print("\n[1/3] Hibrit LightGBM eğitiliyor (5.500 iterasyon kapasitesi)...")
    t0 = time.time()
    lgb_model = lgb.LGBMRegressor(
        n_estimators=5500,
        learning_rate=0.045,
        num_leaves=255,
        min_child_samples=80,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.75,
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
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(500)],
    )
    val_lgb = lgb_model.predict(X_val).astype(np.float32)
    test_lgb = lgb_model.predict(X_test).astype(np.float32)
    print(f"LightGBM tamamlandı ({(time.time() - t0)/60:.1f} dk) -> Test RMSE: {eval_metrics(y_test, test_lgb)['rmse']:.3f}")

    gains = lgb_model.booster_.feature_importance(importance_type="gain")
    imp_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "source": ["maia2" if c in maia_cols else "structural" for c in feature_cols],
            "gain": gains,
            "gain_pct": 100.0 * gains / np.sum(gains),
        }
    ).sort_values("gain", ascending=False)
    imp_df.to_csv(OUT_DIR / "feature_importance_hybrid_gain.csv", index=False)
    maia_share = imp_df.loc[imp_df["source"] == "maia2", "gain_pct"].sum()
    print(f"LightGBM toplam karar ağırlığında Maia-2 payı: %{maia_share:.2f}")

    joblib.dump(lgb_model, OUT_DIR / "lgb_hybrid.joblib")
    del lgb_model
    gc.collect()

    # 2. XGBoost (RTX 3050 CUDA)
    print("\n[2/3] Hibrit XGBoost eğitiliyor (RTX 3050 CUDA)...")
    t0 = time.time()
    xgb_params = dict(
        n_estimators=5500,
        learning_rate=0.045,
        max_depth=8,
        min_child_weight=80,
        subsample=0.8,
        colsample_bytree=0.75,
        reg_alpha=1.0,
        reg_lambda=5.0,
        tree_method="hist",
        device="cuda",
        early_stopping_rounds=100,
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

    xgb_model.save_model(str(OUT_DIR / "xgb_hybrid.json"))
    del xgb_model
    gc.collect()

    # 3. CatBoost (RTX 3050 GPU)
    print("\n[3/3] Hibrit CatBoost eğitiliyor (RTX 3050 GPU)...")
    t0 = time.time()
    try:
        cat_model = CatBoostRegressor(
            iterations=5500,
            learning_rate=0.055,
            depth=8,
            l2_leaf_reg=5.0,
            loss_function="RMSE",
            eval_metric="RMSE",
            task_type="GPU",
            random_seed=42,
            early_stopping_rounds=120,
            verbose=500,
        )
        cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
    except Exception as e:
        print(f"CatBoost GPU uyarısı ({e}), CPU moduna geçiliyor...")
        cat_model = CatBoostRegressor(
            iterations=4000,
            learning_rate=0.06,
            depth=8,
            l2_leaf_reg=5.0,
            loss_function="RMSE",
            eval_metric="RMSE",
            thread_count=-1,
            random_seed=42,
            early_stopping_rounds=120,
            verbose=500,
        )
        cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)

    val_cat = cat_model.predict(X_val).astype(np.float32)
    test_cat = cat_model.predict(X_test).astype(np.float32)
    print(f"CatBoost tamamlandı ({(time.time() - t0)/60:.1f} dk) -> Test RMSE: {eval_metrics(y_test, test_cat)['rmse']:.3f}")

    cat_model.save_model(str(OUT_DIR / "cat_hybrid.cbm"))
    del cat_model, X_train, X_val, X_test
    gc.collect()

    # Optimal Topluluk Ağırlıkları
    val_mat = np.column_stack([val_lgb, val_xgb, val_cat])
    test_mat = np.column_stack([test_lgb, test_xgb, test_cat])
    weights = optimize_ensemble_weights(val_mat, y_val)

    val_ens = (val_mat @ weights).astype(np.float32)
    test_ens = (test_mat @ weights).astype(np.float32)
    ens_test_m = eval_metrics(y_test, test_ens)

    print("\n================ HİBRİT (70 ÖZNİTELİK) FAZ 3 ÖZETİ ================")
    print(f"Optimal Ağırlıklar -> LGB: {weights[0]:.3f} | XGB: {weights[1]:.3f} | CAT: {weights[2]:.3f}")
    print(f"3'lü Hibrit Topluluk Test RMSE: {ens_test_m['rmse']:.3f} | MAE: {ens_test_m['mae']:.3f} | R2: {ens_test_m['r2']:.4f}")

    with open(OUT_DIR / "feature_cols.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "feature_cols": feature_cols,
                "struct_cols": struct_cols,
                "maia_cols": maia_cols,
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
        OUT_DIR / "hybrid_oof_and_test.npz",
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
    print(f"Tüm hibrit modeller ve ara çıktılar '{OUT_DIR}/' klasörüne kaydedildi.")


if __name__ == "__main__":
    main()