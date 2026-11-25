import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer

MODEL_DIR = Path("models_v4")
V3_ABLATION = Path("models_v3_full") / "ablation_v3_full.csv"
V31_ABLATION = Path("models_v3_1") / "ablation_v3_1.csv"

TIERS = [
    ("Başlangıç (<1100)", 0, 1100),
    ("Orta (1100–1500)", 1100, 1500),
    ("İleri (1500–1900)", 1500, 1900),
    ("Usta Adayı (1900–2300)", 1900, 2300),
    ("Usta Seviyesi (2300+)", 2300, 4000),
]


def eval_row(stage: str, y_val: np.ndarray, p_val: np.ndarray, y_test: np.ndarray, p_test: np.ndarray) -> dict:
    val_rmse = float(np.sqrt(mean_squared_error(y_val, p_val)))
    test_mse = float(mean_squared_error(y_test, p_test))
    test_rmse = float(np.sqrt(test_mse))
    test_mae = float(mean_absolute_error(y_test, p_test))
    test_r2 = float(r2_score(y_test, p_test))
    return {
        "Asama": stage,
        "Val_RMSE": round(val_rmse, 3),
        "Test_MSE": round(test_mse, 3),
        "Test_RMSE": round(test_rmse, 3),
        "Test_MAE": round(test_mae, 3),
        "Test_R2": round(test_r2, 4),
    }


def build_meta_features(
    p_lgb: np.ndarray, p_xgb: np.ndarray, p_cat: np.ndarray, p_ens: np.ndarray, context: np.ndarray
) -> np.ndarray:
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


def main():
    data_path = MODEL_DIR / "v4_oof_and_test.npz"
    if not data_path.exists():
        raise FileNotFoundError("Önce 'python src/03_train_v4_gbdt.py' çalıştırılmalıdır!")

    with open(MODEL_DIR / "ensemble_weights.json", "r", encoding="utf-8") as f:
        weights_info = json.load(f)

    arr = np.load(data_path)
    y_val = arr["y_val"]
    val_lgb, val_xgb, val_cat, val_ens = arr["val_lgb"], arr["val_xgb"], arr["val_cat"], arr["val_ens"]
    val_context = arr["val_context"]

    y_test = arr["y_test"]
    test_lgb, test_xgb, test_cat, test_ens = arr["test_lgb"], arr["test_xgb"], arr["test_cat"], arr["test_ens"]
    test_context = arr["test_context"]

    X_meta_val = build_meta_features(val_lgb, val_xgb, val_cat, val_ens, val_context)
    X_meta_test = build_meta_features(test_lgb, test_xgb, test_cat, test_ens, test_context)
    res_target_val = y_val - val_ens

    print("[1/3] 5-Katlı Sızıntısız Artık Öğrenme (v4 Residual GBDT) eğitiliyor...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_res_pred = np.zeros(len(y_val), dtype=np.float32)
    test_res_preds = np.zeros((5, len(y_test)), dtype=np.float32)
    res_models = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_meta_val)):
        rm = lgb.LGBMRegressor(
            n_estimators=550,
            learning_rate=0.03,
            max_depth=5,
            num_leaves=25,
            min_child_samples=150,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_alpha=2.0,
            reg_lambda=10.0,
            n_jobs=-1,
            random_state=42 + fold,
            verbose=-1,
        )
        rm.fit(X_meta_val[tr_idx], res_target_val[tr_idx])
        oof_res_pred[va_idx] = rm.predict(X_meta_val[va_idx]).astype(np.float32)
        test_res_preds[fold] = rm.predict(X_meta_test).astype(np.float32)
        res_models.append(rm)

    test_res_mean = np.mean(test_res_preds, axis=0)

    opt = minimize_scalar(
        lambda a: float(np.sqrt(np.mean((y_val - (val_ens + a * oof_res_pred)) ** 2))),
        bounds=(0.0, 1.5),
        method="bounded",
    )
    alpha = float(opt.x)
    print(f"Optimal artık düzeltme katsayısı (alpha): {alpha:.3f}")

    val_res_corr = val_ens + alpha * oof_res_pred
    test_res_corr = test_ens + alpha * test_res_mean

    print("[2/3] 5-Katlı Küplü B-Spline + Ridge Kuyruk Kalibrasyonu eğitiliyor...")
    oof_cal = np.zeros(len(y_val), dtype=np.float32)
    test_cal_folds = np.zeros((5, len(y_test)), dtype=np.float32)
    calibrators = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(val_res_corr)):
        cal_pipe = make_pipeline(
            SplineTransformer(n_knots=12, degree=3, knots="quantile", include_bias=False),
            Ridge(alpha=50.0),
        )
        cal_pipe.fit(val_res_corr[tr_idx].reshape(-1, 1), y_val[tr_idx])
        oof_cal[va_idx] = cal_pipe.predict(val_res_corr[va_idx].reshape(-1, 1)).astype(np.float32)
        test_cal_folds[fold] = cal_pipe.predict(test_res_corr.reshape(-1, 1)).astype(np.float32)
        calibrators.append(cal_pipe)

    val_cal = oof_cal
    test_cal = np.mean(test_cal_folds, axis=0).astype(np.float32)

    print("[3/3] 5-Katlı Heteroscedastic Hata Varyansı (Belirsizlik) Modeli eğitiliyor...")
    val_sq_err = (y_val - val_cal) ** 2
    log_var_target = np.log(val_sq_err + 2500.0).astype(np.float32)

    oof_log_var = np.zeros(len(y_val), dtype=np.float32)
    test_log_var_folds = np.zeros((5, len(y_test)), dtype=np.float32)
    unc_models = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_meta_val)):
        um = lgb.LGBMRegressor(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=5,
            num_leaves=25,
            min_child_samples=150,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_alpha=2.0,
            reg_lambda=10.0,
            n_jobs=-1,
            random_state=142 + fold,
            verbose=-1,
        )
        um.fit(X_meta_val[tr_idx], log_var_target[tr_idx])
        oof_log_var[va_idx] = um.predict(X_meta_val[va_idx]).astype(np.float32)
        test_log_var_folds[fold] = um.predict(X_meta_test).astype(np.float32)
        unc_models.append(um)

    test_pred_log_var = np.mean(test_log_var_folds, axis=0)
    test_pred_sigma = np.sqrt(np.maximum(np.exp(test_pred_log_var) - 2500.0, 6400.0)).astype(np.float32)

    train_mean = float(weights_info["train_mean"])
    rows = [
        eval_row("0. Ortalama Taban (Baseline)", y_val, np.full_like(y_val, train_mean), y_test, np.full_like(y_test, train_mean)),
        eval_row("1. Tekil LightGBM (v4 107 Özellik)", y_val, val_lgb, y_test, test_lgb),
        eval_row("2. Tekil XGBoost (v4 107 Özellik)", y_val, val_xgb, y_test, test_xgb),
        eval_row("3. Tekil CatBoost (v4 107 Özellik)", y_val, val_cat, y_test, test_cat),
        eval_row("4. 3'lü v4 GBDT Topluluğu", y_val, val_ens, y_test, test_ens),
        eval_row("5. + Artık Öğrenme (Residual)", y_val, val_res_corr, y_test, test_res_corr),
        eval_row("6. + B-Spline Kalibrasyon (Nihai v4)", y_val, val_cal, y_test, test_cal),
    ]
    ablation_df = pd.DataFrame(rows)
    ablation_df.to_csv(MODEL_DIR / "ablation_v4.csv", index=False)

    print("\n================ ABLASYON TABLOSU (v4 - 107 ÖZNİTELİK) ================")
    print(ablation_df.to_string(index=False))

    sq_err = (y_test - test_cal) ** 2
    full_mse = float(np.mean(sq_err))
    keep_mask = test_pred_sigma <= np.quantile(test_pred_sigma, 0.90)
    masked_mse = float(np.mean(sq_err[keep_mask]))
    oracle_mse = float(np.mean(sq_err[sq_err <= np.quantile(sq_err, 0.90)]))
    np_ratio = masked_mse / oracle_mse

    print("\n================ FEDCSIS 2025 METRİKLERİ (v4) ================")
    print(f"Tam Test Seti MSE (%100 veri)      : {full_mse:,.2f} (RMSE: {np.sqrt(full_mse):.3f})")
    print(f"En Belirsiz %10 Elendikten Sonra   : {masked_mse:,.2f} (RMSE: {np.sqrt(masked_mse):.3f})")
    print(f"FedCSIS 2025 Maske Oranı (N/P)     : {np_ratio:.3f} (v3: 1.731 -> v3.1: 1.628)")

    print("\n================ ZORLUK KADEMELERİNE GÖRE v4 SAPMA (BIAS) KONTROLÜ ================")
    for label, lo, hi in TIERS:
        m = (y_test >= lo) & (y_test < hi)
        yt, yp = y_test[m], test_cal[m]
        print(
            f"  - {label:23s}: Gerçek={np.mean(yt):6.1f} | Tahmin={np.mean(yp):6.1f} | "
            f"Bias={np.mean(yp - yt):+6.1f} | RMSE={np.sqrt(mean_squared_error(yt, yp)):6.2f} | MAE={mean_absolute_error(yt, yp):6.2f}"
        )

    print("\n================ MİMARİ EVRİM KIYASLAMASI (v1 -> v4) ================")
    print("v1 (Saf Yapısal, 56 özellik)         -> RMSE: 282.710 | MAE: 221.400 | R2: 0.6477")
    print("v2 (+ Ply 1 Maia-2, 70 özellik)      -> RMSE: 261.680 | MAE: 202.910 | R2: 0.6982")
    if V3_ABLATION.exists():
        v3_b = pd.read_csv(V3_ABLATION).iloc[-1]
        print(f"v3 (+ Ply 3 Zincir, 84 özellik)      -> RMSE: {v3_b['Test_RMSE']:.3f} | MAE: {v3_b['Test_MAE']:.3f} | R2: {v3_b['Test_R2']:.4f}")
    if V31_ABLATION.exists():
        v31_b = pd.read_csv(V31_ABLATION).iloc[-1]
        print(f"v3.1 (+ RD Ağırlık & Hetero Unc)     -> RMSE: {v31_b['Test_RMSE']:.3f} | MAE: {v31_b['Test_MAE']:.3f} | R2: {v31_b['Test_R2']:.4f}")
    v4_b = ablation_df.iloc[-1]
    print(f"v4 (+ Blitz/Rapid Tuzak & Ply 5, 107)-> RMSE: {v4_b['Test_RMSE']:.3f} | MAE: {v4_b['Test_MAE']:.3f} | R2: {v4_b['Test_R2']:.4f}")

    joblib.dump(res_models, MODEL_DIR / "residual_models.joblib")
    joblib.dump(calibrators, MODEL_DIR / "calibrator.joblib")
    joblib.dump(unc_models, MODEL_DIR / "uncertainty_models.joblib")
    with open(MODEL_DIR / "residual_meta.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "alpha": alpha,
                "calibrator_type": "BaggedSplineRidge",
                "has_heteroscedastic_unc": True,
                "test_rmse": float(v4_b["Test_RMSE"]),
                "test_mae": float(v4_b["Test_MAE"]),
                "np_ratio": float(np_ratio),
            },
            f,
            indent=2,
        )
    print(f"\nv4 Faz 4 kalibratör ve belirsizlik modelleri '{MODEL_DIR}/' klasörüne kaydedildi.")


if __name__ == "__main__":
    main()