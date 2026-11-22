"""
Baseline (paper-like) experiment: Decision Tree, Random Forest, XGBoost on ALL genes.
Follows Dalmolin et al., AI 2025, 6(1), 2:  80/20 split + 10-fold CV.

Put this file in:  src/baseline.py
Run from the project root (venv active):
    python src\\baseline.py                       # both gene versions
    python src\\baseline.py --versions allgenes   # only one version
    python src\\baseline.py --no-validation      # CV only, skip the 20% validation scoring

Needs:  pip install xgboost scikit-learn

Protocol (as in the paper)
    1. Fixed stratified split: 80% train / 20% validation (random_state=42).
    2. 10-fold stratified CV inside the 80%  -> mean +/- std per model (the paper's table).
    3. Fit on the whole 80%, score the 20% VALIDATION set once.
       (use --no-validation to skip this and run the CV only)

Output (experiments/paper_baseline/results/):
    split_ids.csv                        sample id + "train"/"validation"
    baseline_<version>_cv.csv            CV mean/std per model
    baseline_<version>_validation.csv    metrics on the 20% validation set
    baseline_<version>_cm_<model>.csv    confusion matrix on the 20% validation set
"""
import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# the depth-3 / 5-leaf tree may never predict some class -> harmless precision warning
warnings.filterwarnings ("ignore", message=".*ill-defined.*")

SEED = 42
TEST_SIZE = 0.2
N_FOLDS = 10
DATA = Path("experiments/paper_baseline/data")   # put the two *_balanced.csv files here
RES = Path("experiments/paper_baseline/results")

SCORING = {
    "accuracy": "accuracy",
    "precision": "precision_macro",
    "recall": "recall_macro",
    "f1": "f1_macro",
}


def make_models() -> dict:
    """Hyperparameters as described in the paper (Section 3.4)."""
    return {
        "DT": DecisionTreeClassifier(max_depth=3, max_leaf_nodes=5,
                                     criterion="entropy", random_state=SEED),
        "RF": RandomForestClassifier(n_estimators=100, min_samples_split=2, max_depth=3,
                                     criterion="log_loss", random_state=SEED, n_jobs=-1),
        "XGB": XGBClassifier(objective="multi:softmax", max_depth=3, tree_method="hist",
                             random_state=SEED, n_jobs=-1),
    }


def load(version: str):
    path = DATA / f"tcga_5cancers_{version}_balanced.csv"
    print(f"\n=== {version}: loading {path} ===")
    df = pd.read_csv(path, index_col=0)
    y_text = df["label"]
    X = df.drop(columns="label").astype("float32")
    enc = LabelEncoder().fit(y_text)
    y = pd.Series(enc.transform(y_text), index=df.index, name="label")
    print(f"samples: {X.shape[0]} | genes: {X.shape[1]} | classes: {list(enc.classes_)}")
    return X, y, enc


def run_version(version: str, do_test: bool):
    RES.mkdir(parents=True, exist_ok=True)
    X, y, enc = load(version)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y)
    print(f"train (80%, used for CV): {X_train.shape[0]} | validation (20%): {X_test.shape[0]}")

    # save the split once so everybody can verify they use the same one
    split_file = RES / "split_ids.csv"
    split = pd.DataFrame({"sample": list(X_train.index) + list(X_test.index),
                          "set": ["train"] * len(X_train) + ["validation"] * len(X_test)})
    if split_file.exists():
        old = pd.read_csv(split_file)
        if set(old.loc[old.set == "validation", "sample"]) != set(X_test.index):
            print("WARNING: split differs from experiments/paper_baseline/results/split_ids.csv - versions use different samples!")
    else:
        split.to_csv(split_file, index=False)

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    cv_rows, test_rows = [], []

    for name, model in make_models().items():
        t0 = time.time()
        scores = cross_validate(model, X_train, y_train, cv=cv, scoring=SCORING, n_jobs=1)
        row = {"model": name}
        for key in SCORING:
            vals = scores[f"test_{key}"]
            row[f"{key}_mean"] = vals.mean()
            row[f"{key}_std"] = vals.std()
        cv_rows.append(row)
        print(f"[{name}] CV  acc={row['accuracy_mean']:.4f}+/-{row['accuracy_std']:.4f} "
              f"f1={row['f1_mean']:.4f}  ({time.time() - t0:.0f}s)")

        if do_test:
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            test_rows.append({
                "model": name,
                "accuracy": accuracy_score(y_test, pred),
                "precision": precision_score(y_test, pred, average="macro"),
                "recall": recall_score(y_test, pred, average="macro"),
                "f1": f1_score(y_test, pred, average="macro"),
            })
            cm = pd.DataFrame(confusion_matrix(y_test, pred), index=enc.classes_, columns=enc.classes_)
            cm.to_csv(RES / f"baseline_{version}_cm_{name}.csv")
            print(f"[{name}] VALIDATION-20% acc={test_rows[-1]['accuracy']:.4f} f1={test_rows[-1]['f1']:.4f}")

    pd.DataFrame(cv_rows).round(4).to_csv(RES / f"baseline_{version}_cv.csv", index=False)
    if do_test:
        pd.DataFrame(test_rows).round(4).to_csv(RES / f"baseline_{version}_validation.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--versions", nargs="+", default=["allgenes", "proteincoding"],
                    choices=["allgenes", "proteincoding"])
    ap.add_argument("--no-validation", action="store_true",
                    help="skip scoring the 20% validation set (CV only)")
    args = ap.parse_args()
    for v in args.versions:
        run_version(v, do_test=not args.no_validation)
    print("\nDone. Results are in experiments/paper_baseline/results/")


if __name__ == "__main__":
    main()