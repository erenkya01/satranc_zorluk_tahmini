from pathlib import Path
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

MODELS_DIR = Path("models_structural_only")
RANDOM_SEED = 42


def eval_metrics(y_true, y_pred):
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return mse, rmse, mae, r2


def build_meta_features(X, p_lgb, p_xgb, p_cat, p_ens):
    preds_stack = np.column_stack([p_lgb, p_xgb, p_cat])
    pred_std = np.std(preds_stack, axis=1, keepdims=True)
    pred_range = (np.max(preds_stack, axis=1) - np.min(preds_stack, axis=1)).reshape(-1, 1)
    lgb_minus_cat = (p_lgb - p_cat).reshape(-1, 1)
    lgb_minus_xgb = (p_lgb - p_xgb).reshape(-1, 1)

    return np.hstack([
        X,
        preds_stack,
        p_ens.reshape(-1, 1),
        pred_std,
        pred_range,
        lgb_minus_cat,
        lgb_minus_xgb,
    ]).astype(np.float32)


def compute_uncertainty_mask_ratio(y_true, y_pred, uncertainty_score, drop_ratio=0.10):
    """
    FedCSIS 2025 belirsizlik maskesi metriği (N/P oranı):
    En belirsiz %10'luk dilim çıkarıldıktan sonraki MSE / Teorik mükemmel maske MSE.
    """
    n = len(y_true)
    k_drop = int(n * drop_ratio)
    sq_errors = (y_true - y_pred) ** 2

    keep_model_idx = np.argsort(uncertainty_score)[: n - k_drop]
    mse_model_masked = float(np.mean(sq_errors[keep_model_idx]))

    keep_oracle_idx = np.argsort(sq_errors)[: n - k_drop]
    mse_oracle_masked = float(np.mean(sq_errors[keep_oracle_idx]))

    np_ratio = mse_model_masked / max(mse_oracle_masked, 1e-8)
    return mse_model_masked, mse_oracle_masked, np_ratio


def main():
    npz_path = MODELS_DIR / "stage1_predictions.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"'{npz_path}' bulunamadı! Önce güncel 03_train_gbdt.py betiğini çalıştırın."
        )

    print("Faz 3 tahminleri yükleniyor...")
    d = np.load(npz_path)

    X_val, y_val = d["X_val"], d["y_val"].astype(np.float64)
    val_lgb, val_xgb, val_cat, val_ens = (
        d["val_lgb"].astype(np.float64),
        d["val_xgb"].astype(np.float64),
        d["val_cat"].astype(np.float64),
        d["val_ens"].astype(np.float64),
    )

    X_test, y_test = d["X_test"], d["y_test"].astype(np.float64)
    test_lgb, test_xgb, test_cat, test_ens = (
        d["test_lgb"].astype(np.float64),
        d["test_xgb"].astype(np.float64),
        d["test_cat"].astype(np.float64),
        d["test_ens"].astype(np.float64),
    )

    # 1. Artık Öğrenme (Residual Learning) için meta-öznitelikler
    X_meta_val = build_meta_features(X_val, val_lgb, val_xgb, val_cat, val_ens)
    X_meta_test = build_meta_features(X_test, test_lgb, test_xgb, test_cat, test_ens)
    val_residuals = y_val - val_ens

    print("Artık Öğrenme (Residual GBDT) 5-Katlı Sızıntısız Çapraz Doğrulama ile eğitiliyor...")
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    oof_res_pred = np.zeros(len(y_val), dtype=np.float64)
    test_res_pred = np.zeros(len(y_test), dtype=np.float64)
    residual_models = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_meta_val), start=1):
        # Dışarıda kalan va_idx katmanına bakmadan tr_idx içinden erken durdurma ayrımı yap
        sub_cut = int(len(tr_idx) * 0.9)
        es_tr_idx, es_va_idx = tr_idx[:sub_cut], tr_idx[sub_cut:]

        res_model = lgb.LGBMRegressor(
            n_estimators=600,
            learning_rate=0.03,
            num_leaves=31,
            min_child_samples=200,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_alpha=5.0,
            reg_lambda=10.0,
            random_state=RANDOM_SEED + fold,
            n_jobs=-1,
            verbose=-1,
        )
        res_model.fit(
            X_meta_val[es_tr_idx],
            val_residuals[es_tr_idx],
            eval_set=[(X_meta_val[es_va_idx], val_residuals[es_va_idx])],
            callbacks=[lgb.early_stopping(stopping_rounds=40, verbose=False)],
        )
        oof_res_pred[va_idx] = res_model.predict(X_meta_val[va_idx])
        test_res_pred += res_model.predict(X_meta_test) / kf.n_splits
        residual_models.append(res_model)

    # Doğrulama OOF üzerinde optimal artık sönümleme katsayısı (alpha) bul
    alpha_opt = minimize_scalar(
        lambda a: mean_squared_error(y_val, val_ens + a * oof_res_pred),
        bounds=(0.0, 1.5),
        method="bounded",
    )
    alpha = float(alpha_opt.x)
    val_after_res = val_ens + alpha * oof_res_pred
    test_after_res = test_ens + alpha * test_res_pred
    print(f"Optimal artık düzeltme katsayısı (alpha): {alpha:.3f}")

    # 2. Sistematik Post-Hoc Kalibrasyon (OOF üzerinde sızıntısız öğrenme)
    oof_cal_lin = np.zeros_like(val_after_res)
    oof_cal_iso = np.zeros_like(val_after_res)

    for tr_idx, va_idx in kf.split(val_after_res):
        lin = Ridge(alpha=1.0)
        lin.fit(val_after_res[tr_idx].reshape(-1, 1), y_val[tr_idx])
        oof_cal_lin[va_idx] = lin.predict(val_after_res[va_idx].reshape(-1, 1))

        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(val_after_res[tr_idx], y_val[tr_idx])
        oof_cal_iso[va_idx] = iso.predict(val_after_res[va_idx])

    mse_lin = mean_squared_error(y_val, oof_cal_lin)
    mse_iso = mean_squared_error(y_val, oof_cal_iso)

    if mse_iso <= mse_lin:
        cal_name = "İzotonik Kalibrasyon"
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(val_after_res, y_val)
        val_final = oof_cal_iso
        test_final = calibrator.predict(test_after_res)
    else:
        cal_name = "Doğrusal Kalibrasyon"
        calibrator = Ridge(alpha=1.0)
        calibrator.fit(val_after_res.reshape(-1, 1), y_val)
        val_final = oof_cal_lin
        test_final = calibrator.predict(test_after_res.reshape(-1, 1))

    # 3. Belirsizlik Skoru ve Maske Oranı (FedCSIS 2025 N/P Metriği)
    val_model_std = np.std(np.column_stack([val_lgb, val_xgb, val_cat]), axis=1)
    test_model_std = np.std(np.column_stack([test_lgb, test_xgb, test_cat]), axis=1)

    val_uncertainty = val_model_std + 0.25 * np.abs(oof_res_pred)
    test_uncertainty = test_model_std + 0.25 * np.abs(test_res_pred)

    # Medyan mutlak hata tabanlı beklenen hata payı ölçeği (~%50 kapsama)
    abs_err_val = np.abs(y_val - val_final)
    uncertainty_scale = float(np.median(abs_err_val) / max(np.median(val_uncertainty), 1e-6))

    masked_mse, oracle_mse, np_ratio = compute_uncertainty_mask_ratio(
        y_test, test_final, test_uncertainty, drop_ratio=0.10
    )

    # 4. Tam Ablasyon Tablosu (Test seti ana değerlendirme metriğidir)
    ablation_rows = [
        ("0. Ortalama Taban (Baseline)", eval_metrics(y_val, np.full_like(y_val, np.mean(y_val)))[1], *eval_metrics(y_test, np.full_like(y_test, np.mean(y_val)))),
        ("1. Tekil LightGBM (56 Özellik)", eval_metrics(y_val, val_lgb)[1], *eval_metrics(y_test, test_lgb)),
        ("2. Tekil XGBoost (56 Özellik)", eval_metrics(y_val, val_xgb)[1], *eval_metrics(y_test, test_xgb)),
        ("3. Tekil CatBoost (56 Özellik)", eval_metrics(y_val, val_cat)[1], *eval_metrics(y_test, test_cat)),
        ("4. 3'lü GBDT Topluluğu (Ensemble)", eval_metrics(y_val, val_ens)[1], *eval_metrics(y_test, test_ens)),
        ("5. + Artık Öğrenme (Residual)", eval_metrics(y_val, val_after_res)[1], *eval_metrics(y_test, test_after_res)),
        (f"6. + {cal_name} (Nihai)", eval_metrics(y_val, val_final)[1], *eval_metrics(y_test, test_final)),
    ]

    ablation_df = pd.DataFrame(
        ablation_rows, columns=["Asama", "Val_RMSE", "Test_MSE", "Test_RMSE", "Test_MAE", "Test_R2"]
    )
    ablation_df.to_csv(MODELS_DIR / "ablation_structural_only.csv", index=False)

    print("\n================ TEZ ABLASYON TABLOSU (56 YAPISAL ÖZNİTELİK) ================")
    print(ablation_df.to_string(index=False, float_format=lambda x: f"{x:,.3f}"))

    print("\n================ BELİRSİZLİK VE MASKE ANALİZİ (TEST SETİ) ================")
    print(f"Tam Test Seti MSE (%100 veri)      : {mean_squared_error(y_test, test_final):,.2f}")
    print(f"En Belirsiz %10 Elendikten Sonra   : {masked_mse:,.2f}")
    print(f"Teorik Mükemmel Maske MSE          : {oracle_mse:,.2f}")
    print(f"FedCSIS 2025 Maske Oranı (N/P)     : {np_ratio:.3f} (Literatür aralığı: 1.43 - 1.71)")

    joblib.dump(
        {
            "residual_models": residual_models,
            "alpha": alpha,
            "calibrator_type": cal_name,
            "calibrator": calibrator,
            "uncertainty_scale": uncertainty_scale,
        },
        MODELS_DIR / "stage2_artifacts.joblib",
    )
    print(f"\nFaz 4 modelleri ve 'ablation_structural_only.csv' '{MODELS_DIR}/' klasörüne kaydedildi.")


if __name__ == "__main__":
    main()