from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd

from pandas.api.types import is_scalar
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.metrics import classification_report, f1_score, hamming_loss
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.inspection import permutation_importance


DEFAULT_FEATURES = [
    "uri_len", "path_depth", "query_len", "n_query_params", "max_param_value_len",
    "uri_pct_non_alnum_ratio", "encoded", "suspicious_tokens_count", "has_suspicious_tokens",
    "uncommon_method", "req_content_length", "body_len", "body_suspicious_tokens_count", "body_has_suspicious_tokens", "body_encoded",
    "method_GET", "method_POST", "method_HEAD", "method_PUT", "method_DELETE", "method_PATCH", "method_OPTIONS",
    "method_TRACE", "method_CONNECT", "method_OTHER",
]


def _read_table(path: str) -> pd.DataFrame:
    p = path.lower()
    if p.endswith(".parquet"):
        return pd.read_parquet(path)
    if p.endswith(".csv"):
        return pd.read_csv(path)
    return pd.read_csv(path)


def _safe_name(s: str) -> str:
    s = str(s)
    s = re.sub(r"[^0-9a-zA-Z]+", "_", s).strip("_")
    return s or "CLASS"


def _derive_fi_paths(model_out: str, tag: str) -> str:
    return f"{model_out}.feature_importance.{tag}.csv"


def _print_top(df_imp: pd.DataFrame, col: str, k: int) -> None:
    if df_imp is None or df_imp.empty:
        return
    k = int(k) if k else 0
    if k <= 0:
        return
    view = df_imp.sort_values(col, ascending=False).head(k)
    print(f"\n=== Top {k} features by {col} ===")
    for _, r in view.iterrows():
        print(f"{r['feature']}: {r[col]:.6g}")


def _coef_importance_multiclass(pipeline: Pipeline, features: List[str]) -> pd.DataFrame:
    lr: LogisticRegression = pipeline.named_steps["lr"]
    coefs = np.asarray(lr.coef_)

    if coefs.ndim == 1:
        coefs = coefs.reshape(1, -1)

    classes = getattr(lr, "classes_", [f"class_{i}" for i in range(coefs.shape[0])])
    classes = [str(c) for c in classes]

    abs_mean = np.mean(np.abs(coefs), axis=0)

    out = pd.DataFrame({
        "feature": features,
        "importance_abs_mean": abs_mean,
    })

    for i, c in enumerate(classes):
        out[f"coef_{_safe_name(c)}"] = coefs[i, :]

    return out.sort_values("importance_abs_mean", ascending=False).reset_index(drop=True)


def _coef_importance_multilabel(pipeline: Pipeline, features: List[str], labels: List[str]) -> pd.DataFrame:
    ovr: OneVsRestClassifier = pipeline.named_steps["ovr"]

    n_labels = len(labels)
    n_feats = len(features)
    coef_mat = np.zeros((n_labels, n_feats), dtype=float)

    ests = getattr(ovr, "estimators_", None) or []
    for i in range(min(n_labels, len(ests))):
        est = ests[i]
        if est is None or not hasattr(est, "coef_"):
            continue
        c = np.asarray(est.coef_).reshape(-1)
        if c.shape[0] == n_feats:
            coef_mat[i, :] = c

    abs_mean = np.mean(np.abs(coef_mat), axis=0)

    out = pd.DataFrame({
        "feature": features,
        "importance_abs_mean": abs_mean,
    })

    for i, lab in enumerate(labels):
        out[f"coef_{_safe_name(lab)}"] = coef_mat[i, :]

    return out.sort_values("importance_abs_mean", ascending=False).reset_index(drop=True)


def _perm_importance(
    estimator: Pipeline,
    X: pd.DataFrame,
    y,
    features: List[str],
    *,
    scoring: str,
    n_repeats: int,
    seed: int,
    n_jobs: int,
) -> pd.DataFrame:
    res = permutation_importance(
        estimator,
        X,
        y,
        scoring=scoring,
        n_repeats=int(n_repeats),
        random_state=int(seed),
        n_jobs=int(n_jobs),
    )
    return (
        pd.DataFrame({
            "feature": features,
            "perm_importance_mean": res.importances_mean,
            "perm_importance_std": res.importances_std,
        })
        .sort_values("perm_importance_mean", ascending=False)
        .reset_index(drop=True)
    )


def _safe_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float,
    seed: int,
    try_stratify: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    if not try_stratify:
        return train_test_split(X, y, test_size=test_size, random_state=seed, shuffle=True)

    try:
        return train_test_split(
            X, y,
            test_size=test_size,
            random_state=seed,
            stratify=y,
        )
    except ValueError as e:
        # Typical: "least populated classes ... only 1 member"
        vc = y.value_counts(dropna=False)
        rare = vc[vc < 2]
        print("\n[WARN] Stratified train/test split failed; falling back to non-stratified split.")
        print(f"[WARN] Reason: {e}")
        if len(rare) > 0:
            print("[WARN] Classes with <2 samples (cannot be stratified):")
            for cls, cnt in rare.items():
                print(f"  - {cls}: {cnt}")
        return train_test_split(X, y, test_size=test_size, random_state=seed, shuffle=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--task", required=True, choices=["multiclass", "multilabel"])
    ap.add_argument("--label-col", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)

    # --- Rare-class handling (multiclass only) ---
    ap.add_argument(
        "--rare-class-policy",
        default="keep",
        choices=["keep", "merge", "drop"],
        help="What to do with classes that have < --rare-class-min-count samples (multiclass only). "
             "keep=do nothing (may disable stratify), merge=map them to OTHER_RARE, drop=remove those rows.",
    )
    ap.add_argument(
        "--rare-class-min-count",
        type=int,
        default=2,
        help="Minimum samples per class required for stratification (multiclass only). Default 2.",
    )

    # --- Feature Importance options ---
    ap.add_argument(
        "--fi-kind",
        default="coef",
        choices=["coef", "permutation", "both", "none"],
        help="Feature importance method. coef is fast for LR; permutation is slower but model-agnostic.",
    )
    ap.add_argument("--fi-topk", type=int, default=25, help="Print top-K features to stdout.")
    ap.add_argument("--fi-n-repeats", type=int, default=5, help="Permutation repeats (if enabled).")
    ap.add_argument("--fi-n-jobs", type=int, default=-1, help="Permutation n_jobs (if enabled).")
    ap.add_argument(
        "--fi-scoring",
        default=None,
        help="Override sklearn scoring for permutation importance. "
             "Defaults: multiclass=f1_weighted, multilabel=f1_micro.",
    )

    args = ap.parse_args()

    df = _read_table(args.data)

    # Ensure all features exist (fail fast if not)
    missing = [c for c in DEFAULT_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    X = df[DEFAULT_FEATURES].fillna(0)

    fi_kind = (args.fi_kind or "none").strip().lower()
    want_coef = fi_kind in {"coef", "both"}
    want_perm = fi_kind in {"permutation", "both"}
    want_any = fi_kind != "none"

    if args.task == "multiclass":
        y = df[args.label_col].astype(str).fillna("UNKNOWN")

        # Optional rare-class policy BEFORE split
        min_cnt = int(args.rare_class_min_count)
        vc = y.value_counts(dropna=False)
        rare_classes = vc[vc < min_cnt].index.tolist()

        if rare_classes:
            print(f"\n[INFO] Found {len(rare_classes)} classes with <{min_cnt} samples.")
            print("[INFO] rare-class-policy =", args.rare_class_policy)
            if args.rare_class_policy == "drop":
                keep_mask = ~y.isin(rare_classes)
                X = X.loc[keep_mask].reset_index(drop=True)
                y = y.loc[keep_mask].reset_index(drop=True)
                print(f"[INFO] Dropped {int((~keep_mask).sum())} rows from rare classes.")
            elif args.rare_class_policy == "merge":
                y = y.where(~y.isin(rare_classes), other="OTHER_RARE")
                print("[INFO] Merged rare classes into label 'OTHER_RARE'.")

        # Decide whether to TRY stratify
        # (still may fail due to rounding constraints; we'll fallback automatically)
        try_stratify = True

        Xtr, Xte, ytr, yte = _safe_train_test_split(
            X, y,
            test_size=args.test_size,
            seed=args.seed,
            try_stratify=try_stratify,
        )

        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ])
        clf.fit(Xtr, ytr)
        pred = clf.predict(Xte)

        print(classification_report(yte, pred, digits=4, zero_division=0))

        # --- Feature importance (multiclass) ---
        if want_any:
            if want_coef:
                df_coef = _coef_importance_multiclass(clf, DEFAULT_FEATURES)
                coef_path = _derive_fi_paths(args.out, "coef")
                df_coef.to_csv(coef_path, index=False)
                print(f"Saved feature importance (coef): {coef_path}")
                _print_top(df_coef, "importance_abs_mean", args.fi_topk)

            if want_perm:
                scoring = args.fi_scoring or "f1_weighted"
                df_perm = _perm_importance(
                    clf, Xte, yte, DEFAULT_FEATURES,
                    scoring=scoring,
                    n_repeats=args.fi_n_repeats,
                    seed=args.seed,
                    n_jobs=args.fi_n_jobs,
                )
                perm_path = _derive_fi_paths(args.out, "perm")
                df_perm.to_csv(perm_path, index=False)
                print(f"Saved feature importance (permutation, scoring={scoring}): {perm_path}")
                _print_top(df_perm, "perm_importance_mean", args.fi_topk)

        joblib.dump(clf, args.out)
        print(f"Saved: {args.out}")
        return

    # ---- multilabel ----
    raw = df[args.label_col]

    def normalize(v):
        if v is None:
            return []
        if is_scalar(v) and pd.isna(v):
            return []
        if isinstance(v, np.ndarray):
            v = v.tolist()
        if isinstance(v, (list, tuple, set)):
            return [str(x).strip() for x in v if str(x).strip()]

        s = str(v).strip()
        if not s or s.lower() in {"[]", "nan", "none"}:
            return []

        for sep in ["|", ";", ","]:
            if sep in s:
                return [p.strip() for p in s.split(sep) if p.strip()]

        if s.startswith("[") and s.endswith("]"):
            inner = s[1:-1].strip()
            if not inner:
                return []
            return [p.strip().strip("'").strip('"') for p in inner.split(",") if p.strip()]

        return [s]

    y_list = raw.map(normalize).tolist()

    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(y_list)

    Xtr, Xte, Ytr, Yte = train_test_split(
        X, Y, test_size=args.test_size, random_state=args.seed, shuffle=True
    )

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

    # --- Feature importance (multilabel) ---
    if want_any:
        labels = [str(x) for x in mlb.classes_]

        if want_coef:
            df_coef = _coef_importance_multilabel(clf, DEFAULT_FEATURES, labels)
            coef_path = _derive_fi_paths(args.out, "coef")
            df_coef.to_csv(coef_path, index=False)
            print(f"Saved feature importance (coef): {coef_path}")
            _print_top(df_coef, "importance_abs_mean", args.fi_topk)

        if want_perm:
            scoring = args.fi_scoring or "f1_micro"
            df_perm = _perm_importance(
                clf, Xte, Yte, DEFAULT_FEATURES,
                scoring=scoring,
                n_repeats=args.fi_n_repeats,
                seed=args.seed,
                n_jobs=args.fi_n_jobs,
            )
            perm_path = _derive_fi_paths(args.out, "perm")
            df_perm.to_csv(perm_path, index=False)
            print(f"Saved feature importance (permutation, scoring={scoring}): {perm_path}")
            _print_top(df_perm, "perm_importance_mean", args.fi_topk)

    joblib.dump({"pipeline": clf, "mlb": mlb}, args.out)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
