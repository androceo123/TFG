from __future__ import annotations

import argparse
import joblib
import pandas as pd
import numpy as np

from pandas.api.types import is_scalar
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.metrics import classification_report, f1_score, hamming_loss
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier


DEFAULT_FEATURES = [
    "uri_len","path_depth","query_len","n_query_params","max_param_value_len",
    "uri_pct_non_alnum_ratio","encoded","suspicious_tokens_count","has_suspicious_tokens",
    "uncommon_method","req_content_length","body_len","body_suspicious_tokens_count","body_has_suspicious_tokens","body_encoded",
    "method_GET","method_POST","method_HEAD","method_PUT","method_DELETE","method_PATCH","method_OPTIONS",
    "method_TRACE","method_CONNECT","method_OTHER",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--task", required=True, choices=["multiclass", "multilabel"])
    ap.add_argument("--label-col", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_parquet(args.data) if args.data.lower().endswith(".parquet") else pd.read_csv(args.data)
    X = df[DEFAULT_FEATURES].fillna(0)

    if args.task == "multiclass":
        y = df[args.label_col].astype(str)
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=args.test_size, random_state=args.seed, stratify=y
        )
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ])
        clf.fit(Xtr, ytr)
        pred = clf.predict(Xte)
        print(classification_report(yte, pred, digits=4))

        joblib.dump(clf, args.out)
        print(f"Saved: {args.out}")
        return

    # ---- multilabel ----
    raw = df[args.label_col]

    def normalize(v):
        # 1) None / scalar-NA
        if v is None:
            return []
        if is_scalar(v) and pd.isna(v):
            return []

        # 2) list-like from parquet (often numpy arrays)
        if isinstance(v, np.ndarray):
            v = v.tolist()
        if isinstance(v, (list, tuple, set)):
            return [str(x).strip() for x in v if str(x).strip()]

        # 3) strings: separators or stringified python lists
        s = str(v).strip()
        if not s or s.lower() in {"[]", "nan", "none"}:
            return []

        for sep in ["|", ";", ","]:
            if sep in s:
                return [p.strip() for p in s.split(sep) if p.strip()]

        if s.startswith("[") and s.endswith("]"):
            inner = s.strip()[1:-1].strip()
            if not inner:
                return []
            return [p.strip().strip("'").strip('"') for p in inner.split(",") if p.strip()]

        return [s]

    y_list = raw.map(normalize).tolist()

    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(y_list)

    Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=args.test_size, random_state=args.seed)

    base = LogisticRegression(max_iter=2000)
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("ovr", OneVsRestClassifier(base)),
    ])
    clf.fit(Xtr, Ytr)
    pred = clf.predict(Xte)

    print("F1 micro:", f1_score(Yte, pred, average="micro", zero_division=0))
    print("F1 macro:", f1_score(Yte, pred, average="macro", zero_division=0))
    print("Hamming loss:", hamming_loss(Yte, pred))

    joblib.dump({"pipeline": clf, "mlb": mlb}, args.out)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()