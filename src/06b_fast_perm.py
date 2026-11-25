import json
import sqlite3
import time
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

ROOT_DIR = Path(__file__).resolve().parent.parent
DB_PATH = ROOT_DIR / "data" / "puzzles.db"
V4_DIR = ROOT_DIR / "models_v4"

t0 = time.time()
with open(V4_DIR / "feature_cols.json", "r", encoding="utf-8") as f:
    meta = json.load(f)

feature_cols = meta["feature_cols"]
sel_sf = ", ".join(f"sf.{c}" for c in meta["struct_cols"])
sel_m1 = ", ".join(f"m1.{c}" for c in meta["maia_p1_cols"])
sel_m3 = ", ".join(f"m3.{c}" for c in meta["maia_p3_cols"])
sel_m4 = ", ".join(f"m4.{c}" for c in meta["maia_v4_cols"])

conn = sqlite3.connect(DB_PATH)
df_test = pd.read_sql_query(
    f"""
    SELECT r.rating, {sel_sf}, {sel_m1}, {sel_m3}, {sel_m4}
    FROM structural_features sf
    INNER JOIN maia2_features m1 ON sf.puzzle_id = m1.puzzle_id
    INNER JOIN maia2_ply3_features m3 ON sf.puzzle_id = m3.puzzle_id
    INNER JOIN maia2_v4_features m4 ON sf.puzzle_id = m4.puzzle_id
    INNER JOIN raw_puzzles r ON sf.puzzle_id = r.puzzle_id
    WHERE LOWER(r.split) = 'test'
    """,
    conn,
)
conn.close()

spec = spec_from_file_location("train_v4", ROOT_DIR / "src" / "03_train_v4_gbdt.py")
mod = module_from_spec(spec)
spec.loader.exec_module(mod)
mod.add_v4_interactions(df_test)

sub = df_test.sample(n=25_000, random_state=42)
X_sub = sub[feature_cols].to_numpy(dtype=np.float32)
y_sub = sub["rating"].to_numpy(dtype=np.float32)

lgb_model = joblib.load(V4_DIR / "lgb_v4.joblib")
lgb_model.set_params(n_jobs=6)

perm = permutation_importance(
    lgb_model,
    X_sub,
    y_sub,
    scoring="neg_root_mean_squared_error",
    n_repeats=2,
    random_state=42,
    n_jobs=1,
)

perm_df = pd.DataFrame({
    "feature": feature_cols,
    "rmse_increase_when_shuffled": np.round(perm.importances_mean, 3),
    "std": np.round(perm.importances_std, 3),
}).sort_values("rmse_increase_when_shuffled", ascending=False)

out_path = ROOT_DIR / "reports" / "tables" / "v4_permutation_importance.csv"
perm_df.to_csv(out_path, index=False)
print(f"Tamamlandi ({time.time() - t0:.1f} sn) -> {out_path}")
print("\nEn Kritik Ilk 15 Oznitelik (Permutation Importance - RMSE Artisi):")
print(perm_df.head(15).to_string(index=False))