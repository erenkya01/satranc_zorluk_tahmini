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

MODEL_DIR = Path("models_hybrid_maia2")
V1_ABLATION = Path("models_structural_only") / "ablation_structural_only.csv"


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


def build_meta_features(p_lgb: np.ndarray, p_xgb: np.ndarray, p_cat: np.ndarray, p_ens: np.ndarray, context: np.ndarray) -> np.ndarray:
    stack3 = np.column_stack([p_lgb, p_xgb, p_cat])
    pred_std = np.std(stack3, axis=1)
    pred_range = np.max(stack3, axis=1) - np.min(stack3, axis=1)
    diff_lgb_xgb = p_lgb - p_xgb
    diff_lgb_cat = p_lgb - p_cat
    diff_xgb_cat = p_xgb - p_cat
    base_meta = np.column_stack(
        [
            p_lgb,
            p_xgb,
            p_cat,
            p_ens,
            pred_std,
            pred_range,
            diff_lgb_xgb,
            diff_lgb_cat,
            diff_xgb_cat,
        ]
    ).astype(np.float32)
    return np.hstack([base_meta, context.astype(np.float32)])


def main():
    data_path = MODEL_DIR / "hybrid_oof_and_test.npz"
    if not data_path.exists():
        raise FileNotFoundError("Önce 'python src/03_train_hybrid_gbdt.py' çalıştırılmalıdır!")

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

    print("Artık Öğrenme (Residual GBDT) 5-Katlı Sızıntısız Çapraz Doğrulama ile eğitiliyor...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_res_pred = np.zeros(len(y_val), dtype=np.float32)
    test_res_preds = np.zeros((5, len(y_test)), dtype=np.float32)
    res_models = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_meta_val)):
        rm = lgb.LGBMRegressor(
            n_estimators=350,
            learning_rate=0.03,
            max_depth=4,
            num_leaves=15,
            min_child_samples=200,
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

    # Doğrusal Kalibrasyon (Doğrulama seti üzerinde öğrenilir)
    calibrator = Ridge(alpha=100.0)
    calibrator.fit(val_res_corr.reshape(-1, 1), y_val)
    val_cal = calibrator.predict(val_res_corr.reshape(-1, 1)).astype(np.float32)
    test_cal = calibrator.predict(test_res_corr.reshape(-1, 1)).astype(np.float32)

    train_mean = float(weights_info["train_mean"])
    rows = [
        eval_row("0. Ortalama Taban (Baseline)", y_val, np.full_like(y_val, train_mean), y_test, np.full_like(y_test, train_mean)),
        eval_row("1. Tekil LightGBM (70 Hibrit Özellik)", y_val, val_lgb, y_test, test_lgb),
        eval_row("2. Tekil XGBoost (70 Hibrit Özellik)", y_val, val_xgb, y_test, test_xgb),
        eval_row("3. Tekil CatBoost (70 Hibrit Özellik)", y_val, val_cat, y_test, test_cat),
        eval_row("4. 3'lü Hibrit GBDT Topluluğu", y_val, val_ens, y_test, test_ens),
        eval_row("5. + Artık Öğrenme (Residual)", y_val, val_res_corr, y_test, test_res_corr),
        eval_row("6. + Doğrusal Kalibrasyon (Nihai Hibrit)", y_val, val_cal, y_test, test_cal),
    ]
    ablation_df = pd.DataFrame(rows)
    ablation_df.to_csv(MODEL_DIR / "ablation_hybrid_maia2.csv", index=False)

    print("\n================ TEZ ABLASYON TABLOSU (70 HİBRİT ÖZNİTELİK) ================")
    print(ablation_df.to_string(index=False))

    # FedCSIS 2025 Belirsizlik ve Maske Analizi
    sq_err = (y_test - test_cal) ** 2
    full_mse = float(np.mean(sq_err))

    # Model belirsizlik skoru: modeller arası standart sapma + artık düzeltme büyüklüğü
    test_std = np.std(np.column_stack([test_lgb, test_xgb, test_cat]), axis=1)
    uncertainty_score = test_std + 0.35 * np.abs(test_res_mean)
    keep_mask = uncertainty_score <= np.quantile(uncertainty_score, 0.90)
    masked_mse = float(np.mean(sq_err[keep_mask]))

    # Teorik mükemmel maske (%10 en yüksek hatayı eler)
    oracle_thresh = np.quantile(sq_err, 0.90)
    oracle_mse = float(np.mean(sq_err[sq_err <= oracle_thresh]))
    np_ratio = masked_mse / oracle_mse

    print("\n================ BELİRSİZLİK VE MASKE ANALİZİ (HİBRİT TEST SETİ) ================")
    print(f"Tam Test Seti MSE (%100 veri)      : {full_mse:,.2f} (RMSE: {np.sqrt(full_mse):.2f})")
    print(f"En Belirsiz %10 Elendikten Sonra   : {masked_mse:,.2f} (RMSE: {np.sqrt(masked_mse):.2f})")
    print(f"Teorik Mükemmel Maske MSE          : {oracle_mse:,.2f}")
    print(f"FedCSIS 2025 Maske Oranı (N/P)     : {np_ratio:.3f} (Literatür aralığı: 1.43 - 1.71)")

    if V1_ABLATION.exists():
        v1_df = pd.read_csv(V1_ABLATION)
        v1_best = v1_df.iloc[-1]
        v2_best = ablation_df.iloc[-1]
        diff_rmse = v2_best["Test_RMSE"] - v1_best["Test_RMSE"]
        diff_mae = v2_best["Test_MAE"] - v1_best["Test_MAE"]
        print("\n================ v1 (56 YAPISAL) vs. v2 (70 HİBRİT + MAIA-2) KIYASI ================")
        print(f"v1 Nihai Test RMSE : {v1_best['Test_RMSE']:.3f} | Test MAE: {v1_best['Test_MAE']:.3f} | R2: {v1_best['Test_R2']:.4f}")
        print(f"v2 Nihai Test RMSE : {v2_best['Test_RMSE']:.3f} | Test MAE: {v2_best['Test_MAE']:.3f} | R2: {v2_best['Test_R2']:.4f}")
        print(f"Net İyileşme       : {diff_rmse:+.3f} RMSE | {diff_mae:+.3f} MAE")

    joblib.dump(res_models, MODEL_DIR / "residual_models.joblib")
    joblib.dump(calibrator, MODEL_DIR / "calibrator.joblib")
    with open(MODEL_DIR / "residual_meta.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "alpha": alpha,
                "test_rmse": float(ablation_df.iloc[-1]["Test_RMSE"]),
                "test_mae": float(ablation_df.iloc[-1]["Test_MAE"]),
                "np_ratio": float(np_ratio),
            },
            f,
            indent=2,
        )
    print(f"\nHibrit Faz 4 modelleri ve 'ablation_hybrid_maia2.csv' '{MODEL_DIR}/' klasörüne kaydedildi.")


if __name__ == "__main__":
    main()