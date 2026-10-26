"""Tiện ích dùng chung cho template GBDT: tìm dữ liệu, chia fold, metric, ghi kết quả."""
import glob
import json
import os
from pathlib import Path

import numpy as np

# ===================== CẤU HÌNH BÀI TOÁN — agent sửa phần này =====================
CONFIG = {
    "train_file": "train.csv",   # tên file trong dữ liệu cuộc thi
    "target": "target",          # cột nhãn
    "id_col": "id",              # cột id (bỏ khỏi đặc trưng); None nếu không có
    "task": "binary",            # binary | multiclass | regression
    "metric": "auc",             # auc | logloss | rmse | mae | accuracy | f1_macro
    "n_folds": 5,
    "seed": 42,
    "stratify": True,
}
# ===============================================================================

SMOKE = os.environ.get("KRA_SMOKE") == "1"
OUT = Path(os.environ.get("KRA_OUTPUT_DIR", "outputs"))
OUT.mkdir(parents=True, exist_ok=True)


def find_input(name: str) -> str:
    """Tìm file trong /kaggle/input (đường dẫn mount có thể thay đổi theo thời gian)."""
    roots = [os.environ.get("KRA_INPUT_DIR"), "/kaggle/input", "data"]
    for r in filter(None, roots):
        hits = glob.glob(os.path.join(r, "**", name), recursive=True)
        if hits:
            return sorted(hits, key=len)[0]
    raise FileNotFoundError(f"Không tìm thấy {name} trong {roots}")


def load_train():
    import pandas as pd
    df = pd.read_csv(find_input(CONFIG["train_file"]))
    if SMOKE:
        df = df.sample(n=min(len(df), 3000), random_state=0).reset_index(drop=True)
    return df


def fold_indices(df, fold: int):
    from sklearn.model_selection import KFold, StratifiedKFold
    y = df[CONFIG["target"]]
    if CONFIG["stratify"] and CONFIG["task"] != "regression":
        kf = StratifiedKFold(CONFIG["n_folds"], shuffle=True, random_state=CONFIG["seed"])
        splits = list(kf.split(df, y))
    else:
        kf = KFold(CONFIG["n_folds"], shuffle=True, random_state=CONFIG["seed"])
        splits = list(kf.split(df))
    return splits[fold]


def score(y_true, pred) -> float:
    from sklearn import metrics as M
    m = CONFIG["metric"]
    if m == "auc":
        return float(M.roc_auc_score(y_true, pred, multi_class="ovr") if np.ndim(pred) > 1
                     else M.roc_auc_score(y_true, pred))
    if m == "logloss":
        return float(M.log_loss(y_true, pred))
    if m == "rmse":
        return float(np.sqrt(M.mean_squared_error(y_true, pred)))
    if m == "mae":
        return float(M.mean_absolute_error(y_true, pred))
    if m == "accuracy":
        lab = pred.argmax(1) if np.ndim(pred) > 1 else (pred > 0.5).astype(int)
        return float(M.accuracy_score(y_true, lab))
    if m == "f1_macro":
        lab = pred.argmax(1) if np.ndim(pred) > 1 else (pred > 0.5).astype(int)
        return float(M.f1_score(y_true, lab, average="macro"))
    raise ValueError(m)


def write_metrics(d: dict):
    path = os.environ.get("KRA_METRICS_PATH", str(OUT / "metrics_local.json"))
    Path(path).write_text(json.dumps(d, indent=2))
    print("[metrics]", json.dumps(d))
