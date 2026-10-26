"""Gom OOF của mọi fold → CV tổng (OOF score) + độ lệch chuẩn giữa các fold."""
import numpy as np

from common import OUT, score, write_metrics

parts = sorted(OUT.glob("oof_fold*.npz"))
if not parts:
    write_metrics({"cv": None, "error": "không có oof_fold*.npz"})
    raise SystemExit(1)

fold_scores, idx, pred, y = [], [], [], []
for p in parts:
    d = np.load(p)
    fold_scores.append(score(d["y"], d["pred"]))
    idx.append(d["idx"]); pred.append(d["pred"]); y.append(d["y"])

pred_all, y_all = np.concatenate(pred), np.concatenate(y)
np.savez(OUT / "oof_all.npz", idx=np.concatenate(idx), pred=pred_all, y=y_all)
write_metrics({
    "cv": score(y_all, pred_all),
    "cv_std": float(np.std(fold_scores)),
    "fold_scores": fold_scores,
    "n_folds_done": len(parts),
})
