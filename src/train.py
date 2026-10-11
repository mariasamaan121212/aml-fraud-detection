import json
import time
from datetime import datetime, timezone

import lightgbm as lgb
import sklearn
from sklearn.metrics import average_precision_score

from src.data import ROOT, load_split
from src.features import FEATURE_COLUMNS, make_features

PARAMS = dict(
    n_estimators=1500, learning_rate=0.05, num_leaves=15,
    min_child_samples=500, reg_lambda=10.0,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
    random_state=42, n_jobs=-1, verbose=-1,
)
MODELS = ROOT / "models"


def main():
    MODELS.mkdir(exist_ok=True)
    t0 = time.time()

    train = load_split("benchmark", "train")
    y_train = train["is_fraud"].to_numpy()
    X_train = make_features(train)
    del train

    valid = load_split("benchmark", "valid")
    y_valid = valid["is_fraud"].to_numpy()
    X_valid = make_features(valid)
    del valid

    model = lgb.LGBMClassifier(**PARAMS)
    model.fit(X_train, y_train)

    p_valid = model.predict_proba(X_valid)[:, 1]
    pr_auc = average_precision_score(y_valid, p_valid)

    model.booster_.save_model(str(MODELS / "lgbm_v1.txt"))
    meta = {
        "model_file": "lgbm_v1.txt",
        "split_protocol": "benchmark",
        "params": PARAMS,
        "feature_columns": FEATURE_COLUMNS,
        "n_train_rows": int(len(y_train)),
        "n_train_fraud": int(y_train.sum()),
        "valid_pr_auc": round(float(pr_auc), 4),
        "lightgbm_version": lgb.__version__,
        "sklearn_version": sklearn.__version__,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "train_minutes": round((time.time() - t0) / 60, 1),
    }
    (MODELS / "lgbm_v1_meta.json").write_text(json.dumps(meta, indent=2))

    print("valid PR-AUC:", meta["valid_pr_auc"])
    print("minutes:", meta["train_minutes"])


if __name__ == "__main__":
    main()