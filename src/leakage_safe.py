"""
Experiment B - leakage-safe pipeline.

Order of operations (everything that LEARNS from the data uses the training set only):
    1. log2(TPM+1)                      (done in prepare_logged.py, per-value -> no leakage)
    2. stratified 80/20 split of ALL 3001 samples (random_state=42), test set untouched
    3. gene subset: allgenes, or proteincoding (external annotation, no leakage)
    4. low-expression filter (mean > 25% quantile), computed on TRAIN only,
       the same gene list is then applied to the test set
    5. undersample the TRAIN set to its smallest class (test keeps its natural distribution)
    6. 10-fold CV inside the train set  -> mean +/- std
    7. final fit on train -> test metrics, ONCE

Put this file in:  src/leakage_safe.py   (next to src/baseline.py, it imports from it)
Run from the project root:
    python src\\prepare_logged.py       # once
    python src\\leakage_safe.py
    python src\\leakage_safe.py --no-undersample   # ablation: only the filter is leakage-safe

Output (experiments/leakage_safe/results/  or  .../results_noundersample/):
    split_ids.csv
    <version>_genes_kept.csv, <version>_cv.csv, <version>_test.csv
    <version>_cm_<model>.csv, <version>_report_<model>.csv (per-class precision/recall/F1)
"""
import argparse
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import LabelEncoder

from baseline import N_FOLDS, SCORING, SEED, TEST_SIZE, make_models

LOGGED = Path("experiments/leakage_safe/data/tcga_5cancers_logged.pkl")
GENE_TYPES = Path("data/raw/gene_types.csv")
QNT_CUT = 0.25


def run(undersample: bool):
    res = Path("experiments/leakage_safe") / ("results" if undersample else "results_noundersample")
    res.mkdir(parents=True, exist_ok=True)

    print(f"Loading {LOGGED} ...")
    data = pd.read_pickle(LOGGED)
    assert data.index.is_unique
    X_all = data.drop(columns="label")
    enc = LabelEncoder().fit(data["label"])
    y = pd.Series(enc.transform(data["label"]), index=data.index)
    print(f"samples: {X_all.shape[0]} | genes: {X_all.shape[1]} | classes: {list(enc.classes_)}")

    # ONE split for every version (depends only on the labels)
    train_idx, test_idx = train_test_split(
        data.index, test_size=TEST_SIZE, random_state=SEED, stratify=y)
    pd.DataFrame({"sample": list(train_idx) + list(test_idx),
                  "set": ["train"] * len(train_idx) + ["test"] * len(test_idx)}
                 ).to_csv(res / "split_ids.csv", index=False)
    y_tr, y_te = y.loc[train_idx], y.loc[test_idx]
    print(f"train: {len(train_idx)} {y_tr.value_counts().sort_index().to_dict()}")
    print(f"test : {len(test_idx)} {y_te.value_counts().sort_index().to_dict()}")

    for version in ["allgenes", "proteincoding"]:
        print(f"\n=== {version} ===")
        cols = list(X_all.columns)
        if version == "proteincoding":
            gt = pd.read_csv(GENE_TYPES, index_col=0)["gene_type"]
            cols = [g for g in cols if gt.get(g) == "protein_coding"]
        print("genes before filter:", len(cols))

        # low-expression filter learned from TRAIN only
        X_tr = X_all.loc[train_idx, cols]
        means = X_tr.mean(axis=0)
        keep = means.index[means > means.quantile(QNT_CUT)]
        X_tr, X_te = X_tr[keep], X_all.loc[test_idx, keep]
        print("genes after train-only filter:", len(keep))
        pd.Series(list(keep), name="gene").to_csv(res / f"{version}_genes_kept.csv", index=False)

        # undersample TRAIN only
        if undersample:
            n = int(y_tr.value_counts().min())
            sel = pd.concat([y_tr[y_tr == c].sample(n=n, random_state=SEED)
                             for c in sorted(y_tr.unique())]).index
            X_fit, y_fit = X_tr.loc[sel], y_tr.loc[sel]
            print(f"undersampled train: {n} per class -> {len(sel)} samples")
        else:
            X_fit, y_fit = X_tr, y_tr

        cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        cv_rows, test_rows = [], []
        for name, model in make_models().items():
            t0 = time.time()
            scores = cross_validate(model, X_fit, y_fit, cv=cv, scoring=SCORING, n_jobs=1)
            row = {"model": name, "n_genes": len(keep)}
            for key in SCORING:
                row[f"{key}_mean"] = scores[f"test_{key}"].mean()
                row[f"{key}_std"] = scores[f"test_{key}"].std()
            cv_rows.append(row)
            print(f"[{name}] CV   acc={row['accuracy_mean']:.4f} f1={row['f1_mean']:.4f} "
                  f"({time.time() - t0:.0f}s)")

            model.fit(X_fit, y_fit)                      # final fit, then test ONCE
            pred = model.predict(X_te)
            test_rows.append({
                "model": name, "n_genes": len(keep),
                "accuracy": accuracy_score(y_te, pred),
                "balanced_accuracy": balanced_accuracy_score(y_te, pred),
                "precision_macro": precision_score(y_te, pred, average="macro", zero_division=0),
                "recall_macro": recall_score(y_te, pred, average="macro", zero_division=0),
                "f1_macro": f1_score(y_te, pred, average="macro", zero_division=0),
            })
            print(f"[{name}] TEST acc={test_rows[-1]['accuracy']:.4f} "
                  f"macroF1={test_rows[-1]['f1_macro']:.4f}")
            pd.DataFrame(confusion_matrix(y_te, pred), index=enc.classes_, columns=enc.classes_
                         ).to_csv(res / f"{version}_cm_{name}.csv")
            pd.DataFrame(classification_report(y_te, pred, target_names=enc.classes_,
                                               output_dict=True, zero_division=0)
                         ).T.round(4).to_csv(res / f"{version}_report_{name}.csv")

        pd.DataFrame(cv_rows).round(4).to_csv(res / f"{version}_cv.csv", index=False)
        pd.DataFrame(test_rows).round(4).to_csv(res / f"{version}_test.csv", index=False)

    print("\nDone. Results in", res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-undersample", action="store_true",
                    help="train on the natural (imbalanced) train set")
    run(undersample=not ap.parse_args().no_undersample)