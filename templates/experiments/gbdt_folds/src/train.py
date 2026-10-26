"""Huấn luyện 1 fold bằng XGBoost trên GPU được runner gán (CUDA_VISIBLE_DEVICES).

Mỗi tiến trình chỉ thấy đúng 1 GPU nên luôn dùng device="cuda" (tức cuda:0 cục bộ).
Ghi OOF của fold ra outputs/oof_fold{k}.npz để aggregate.py tính CV tổng.
"""
import argparse
import os
import time

import numpy as np

from common import CONFIG, OUT, SMOKE, fold_indices, load_train, score, write_metrics

ap = argparse.ArgumentParser()
ap.add_argument("--fold", type=int, required=True)
ap.add_argument("--seed", type=int, default=CONFIG["seed"])
args = ap.parse_args()

t0 = time.time()
df = load_train()
features = [c for c in df.columns if c not in (CONFIG["target"], CONFIG["id_col"])]


# ===================== FEATURE ENGINEERING — agent sửa phần này =====================
def make_features(frame):
    import pandas as pd
    X = frame[features].copy()
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]) and not isinstance(X[c].dtype, pd.CategoricalDtype):
            X[c] = X[c].astype("category")
    return X
# ====================================================================================


X, y = make_features(df), df[CONFIG["target"]].values
tr, va = fold_indices(df, args.fold)
use_gpu = bool(os.environ.get("CUDA_VISIBLE_DEVICES")) and not SMOKE

try:
    import xgboost as xgb
    params = dict(
        n_estimators=60 if SMOKE else 5000,
        learning_rate=0.05,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        device="cuda" if use_gpu else "cpu",
        enable_categorical=True,
        early_stopping_rounds=None if SMOKE else 200,
        random_state=args.seed,
        n_jobs=2,   # Kaggle chỉ có ~4 nhân CPU dùng chung cho 2 job song song
    )
    if CONFIG["task"] == "regression":
        model = xgb.XGBRegressor(**params)
    else:
        model = xgb.XGBClassifier(**params)
    model.fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[va], y[va])], verbose=500)
    best_iter = getattr(model, "best_iteration", None)
    lib = f"xgboost-{'cuda' if use_gpu else 'cpu'}"
except ImportError:  # môi trường cục bộ không có xgboost
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    Xn = X.apply(lambda s: s.cat.codes if str(s.dtype) == "category" else s)
    est = HistGradientBoostingRegressor if CONFIG["task"] == "regression" else HistGradientBoostingClassifier
    model = est(max_iter=60 if SMOKE else 500, random_state=args.seed).fit(Xn.iloc[tr], y[tr])
    X, best_iter, lib = Xn, None, "sklearn-hgb"

if CONFIG["task"] == "regression":
    pred = model.predict(X.iloc[va])
elif CONFIG["task"] == "binary":
    pred = model.predict_proba(X.iloc[va])[:, 1]
else:
    pred = model.predict_proba(X.iloc[va])

fold_score = score(y[va], pred)
np.savez(OUT / f"oof_fold{args.fold}.npz", idx=va, pred=pred, y=y[va])
write_metrics({"fold": args.fold, "fold_score": fold_score, "cv": fold_score, "best_iter": best_iter,
               "lib": lib, "seconds": round(time.time() - t0, 1), "n_train": len(tr)})
