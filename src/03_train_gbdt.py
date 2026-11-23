import gc
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
from sklearn.metrics import mean_squared_error

DB_PATH = Path("data") / "puzzles.db"
MODELS_DIR = Path("models")
RANDOM_SEED = 42

FEATURE_COLS = [
    "solution_ply",
    "solver_moves_count",
    "ends_in_mate",
    "is_mate_in_1",
    "total_pieces",
    "solver_pawns",
    "solver_minors",
    "solver_rooks",
    "solver_queens",
    "opp_pawns",
    "opp_minors",
    "opp_rooks",
    "opp_queens",
    "solver_material",
    "opp_material",
    "material_diff_start",
    "material_diff_end",
    "material_gain",
    "in_check_at_start",
    "setup_is_capture",
    "setup_is_check",
    "first_move_is_check",
    "first_move_is_capture",
    "first_move_is_quiet",
    "is_sacrifice",
    "first_move_piece",
    "first_move_dist",
    "solver_checks",
    "solver_captures",
    "solver_promotions",
    "legal_moves_start",
    "solver_king_attackers",
    "opp_king_attackers",
]


def load_datasets():
    print("Veritabanından öznitelikler ve etiketler yükleniyor...")
    t0 = time.time()
    conn = sqlite3.connect(DB_PATH)
    cols_sql = ", ".join([f"f.{c}" for c in FEATURE_COLS])
    query = f"""
        SELECT r.puzzle_id, r.rating, r.split, {cols_sql}
        FROM raw_puzzles r
        JOIN structural_features f ON r.puzzle_id = f.puzzle_id
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    train_mask = (df["split"] == "train").values
    val_mask = (df["split"] == "val").values
    test_mask = (df["split"] == "test").values

    X_all = df[FEATURE_COLS].to_numpy(dtype=np.float32)
    y_all = df["rating"].to_numpy(dtype=np.float32)
    ids_all = df["puzzle_id"].to_numpy()

    data = {
        "X_train": X_all[train_mask],
        "y_train": y_all[train_mask],
        "X_val": X_all[val_mask],
        "y_val": y_all[val_mask],
        "id_val": ids_all[val_mask],
        "X_test": X_all[test_mask],
        "y_test": y_all[test_mask],
        "id_test": ids_all[test_mask],
    }

    del df, X_all, y_all, ids_all
    gc.collect()

    print(
        f"Yükleme tamamlandı ({time.time() - t0:.1f} sn) | "
        f"Train: {len(data['y_train']):,} | Val: {len(data['y_val']):,} | Test: {len(data['y_test']):,}"
    )
    return data


def eval_metrics(y_true, y_pred):
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    return mse, rmse


def train_lightgbm(data):
    print("\n[1/3] LightGBM eğitiliyor...")
    t0 = time.time()
    model = lgb.LGBMRegressor(
        n_estimators=2500,
        learning_rate=0.05,
        num_leaves=127,
        subsample_for_bin=200_000,
        colsample_bytree=0.8,
        subsample=0.8,
        subsample_freq=1,
        reg_alpha=1.0,
        reg_lambda=5.0,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        data["X_train"],
        data["y_train"],
        eval_set=[(data["X_val"], data["y_val"])],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=500),
        ],
    )
    val_pred = model.predict(data["X_val"]).astype(np.float32)
    test_pred = model.predict(data["X_test"]).astype(np.float32)
    print(f"LightGBM tamamlandı ({time.time() - t0:.1f} sn) | En iyi iterasyon: {model.best_iteration_}")
    return model, val_pred, test_pred


def train_xgboost(data):
    print("\n[2/3] XGBoost eğitiliyor...")
    t0 = time.time()
    base_params = dict(
        n_estimators=2500,
        learning_rate=0.05,
        max_depth=8,
        max_bin=256,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=5.0,
        tree_method="hist",
        early_stopping_rounds=50,
        eval_metric="rmse",
        random_state=RANDOM_SEED,
    )
    try:
        model = xgb.XGBRegressor(**base_params, device="cuda")
        model.fit(
            data["X_train"],
            data["y_train"],
            eval_set=[(data["X_val"], data["y_val"])],
            verbose=500,
        )
        print("XGBoost GPU (CUDA) üzerinde eğitildi.")
    except Exception as e:
        print(f"GPU başlatılamadı ({e}), CPU moduna geçiliyor...")
        model = xgb.XGBRegressor(**base_params, device="cpu", n_jobs=-1)
        model.fit(
            data["X_train"],
            data["y_train"],
            eval_set=[(data["X_val"], data["y_val"])],
            verbose=500,
        )

    val_pred = model.predict(data["X_val"]).astype(np.float32)
    test_pred = model.predict(data["X_test"]).astype(np.float32)
    print(f"XGBoost tamamlandı ({time.time() - t0:.1f} sn) | En iyi iterasyon: {model.best_iteration}")
    return model, val_pred, test_pred


def train_catboost(data):
    print("\n[3/3] CatBoost eğitiliyor...")
    t0 = time.time()
    base_params = dict(
        iterations=3000,
        learning_rate=0.06,
        depth=8,
        l2_leaf_reg=5.0,
        loss_function="RMSE",
        eval_metric="RMSE",
        early_stopping_rounds=50,
        random_seed=RANDOM_SEED,
        verbose=500,
    )
    try:
        model = CatBoostRegressor(**base_params, task_type="GPU", border_count=128)
        model.fit(
            data["X_train"],
            data["y_train"],
            eval_set=(data["X_val"], data["y_val"]),
            use_best_model=True,
        )
        print("CatBoost GPU üzerinde eğitildi.")
    except Exception as e:
        print(f"CatBoost GPU başlatılamadı ({e}), CPU moduna geçiliyor...")
        model = CatBoostRegressor(**base_params, task_type="CPU", thread_count=-1)
        model.fit(
            data["X_train"],
            data["y_train"],
            eval_set=(data["X_val"], data["y_val"]),
            use_best_model=True,
        )

    val_pred = model.predict(data["X_val"]).astype(np.float32)
    test_pred = model.predict(data["X_test"]).astype(np.float32)
    print(f"CatBoost tamamlandı ({time.time() - t0:.1f} sn) | En iyi iterasyon: {model.get_best_iteration()}")
    return model, val_pred, test_pred


def optimize_blend_weights(preds_list, y_true):
    P = np.column_stack(preds_list)

    def loss_fn(weights):
        blend = P @ weights
        return mean_squared_error(y_true, blend)

    init_w = np.ones(len(preds_list)) / len(preds_list)
    bounds = [(0.0, 1.0)] * len(preds_list)
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    res = minimize(loss_fn, init_w, method="SLSQP", bounds=bounds, constraints=constraints)
    return res.x


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    data = load_datasets()

    # 0. Naif Ortalama Taban Çizgisi (Dummy Baseline)
    mean_rating = float(np.mean(data["y_train"]))
    base_val_mse, base_val_rmse = eval_metrics(data["y_val"], np.full_like(data["y_val"], mean_rating))
    base_test_mse, base_test_rmse = eval_metrics(data["y_test"], np.full_like(data["y_test"], mean_rating))

    # 1-3. GBDT Modelleri
    lgb_model, val_lgb, test_lgb = train_lightgbm(data)
    xgb_model, val_xgb, test_xgb = train_xgboost(data)
    cat_model, val_cat, test_cat = train_catboost(data)

    # Doğrulama seti üzerinde en iyi topluluk ağırlıklarını bul
    weights = optimize_blend_weights([val_lgb, val_xgb, val_cat], data["y_val"])
    val_ens = weights[0] * val_lgb + weights[1] * val_xgb + weights[2] * val_cat
    test_ens = weights[0] * test_lgb + weights[1] * test_xgb + weights[2] * test_cat

    # Sonuç Tablosu
    results = [
        ("Ortalama Taban (Baseline)", base_val_mse, base_val_rmse, base_test_mse, base_test_rmse),
        ("LightGBM", *eval_metrics(data["y_val"], val_lgb), *eval_metrics(data["y_test"], test_lgb)),
        ("XGBoost", *eval_metrics(data["y_val"], val_xgb), *eval_metrics(data["y_test"], test_xgb)),
        ("CatBoost", *eval_metrics(data["y_val"], val_cat), *eval_metrics(data["y_test"], test_cat)),
        ("3'lü GBDT Topluluğu", *eval_metrics(data["y_val"], val_ens), *eval_metrics(data["y_test"], test_ens)),
    ]

    summary_df = pd.DataFrame(
        results, columns=["Model", "Val_MSE", "Val_RMSE", "Test_MSE", "Test_RMSE"]
    )
    print("\n================ FAZ 3 MODEL KARŞILAŞTIRMASI ================")
    print(
        f"Optimal Topluluk Ağırlıkları -> LightGBM: {weights[0]:.3f} | "
        f"XGBoost: {weights[1]:.3f} | CatBoost: {weights[2]:.3f}\n"
    )
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    # En önemli 10 yapısal öznitelik
    imp_df = pd.DataFrame(
        {"Ozellik": FEATURE_COLS, "Onem_Skoru": lgb_model.feature_importances_}
    ).sort_values("Onem_Skoru", ascending=False)
    print("\n--- EN ÖNEMLİ 10 YAPISAL ÖZNİTELİK (LightGBM) ---")
    print(imp_df.head(10).to_string(index=False))

    # Modelleri ve Faz 4 tahminlerini kaydet
    lgb_model.booster_.save_model(str(MODELS_DIR / "lgb_model.txt"))
    xgb_model.save_model(str(MODELS_DIR / "xgb_model.json"))
    cat_model.save_model(str(MODELS_DIR / "cat_model.cbm"))
    joblib.dump({"weights": weights, "feature_cols": FEATURE_COLS}, MODELS_DIR / "ensemble_meta.joblib")

    np.savez_compressed(
        MODELS_DIR / "stage1_predictions.npz",
        X_val=data["X_val"],
        y_val=data["y_val"],
        id_val=data["id_val"],
        val_lgb=val_lgb,
        val_xgb=val_xgb,
        val_cat=val_cat,
        val_ens=val_ens,
        X_test=data["X_test"],
        y_test=data["y_test"],
        id_test=data["id_test"],
        test_lgb=test_lgb,
        test_xgb=test_xgb,
        test_cat=test_cat,
        test_ens=test_ens,
    )
    print(f"\nModeller ve Faz 4 tahminleri '{MODELS_DIR}/' klasörüne kaydedildi.")


if __name__ == "__main__":
    main()