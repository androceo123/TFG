from __future__ import annotations

"""
waf_ml.scripts.train_supervised

Supervised baselines for WAF-ML features:
  - Multiclass Logistic Regression
  - Multilabel One-vs-Rest Logistic Regression

Adds optional hyperparameter tuning:
  --tune none|grid|random

Backwards compatible with your current calls (tuning defaults to none).

Compatibility fixes included:
- Some scikit-learn versions do NOT accept LogisticRegression(multi_class=...).
  We only pass multi_class if supported.
- Some older scikit-learn versions may not support solver="lbfgs". We try lbfgs
  first and fall back to liblinear automatically.
- GridSearchCV for multiclass uses StratifiedKFold by default, which fails when
  some classes have fewer samples than n_splits. We fall back to KFold for tuning.
"""

import argparse
import inspect
import re
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, hamming_loss, make_scorer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split, KFold, StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler

try:
    from sklearn.inspection import permutation_importance
except Exception:  # pragma: no cover
    permutation_importance = None


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


def _derive_fi_path(model_out: str, tag: str) -> str:
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
    out = pd.DataFrame({"feature": features, "importance_abs_mean": abs_mean})

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
    out = pd.DataFrame({"feature": features, "importance_abs_mean": abs_mean})

    for i, lab in enumerate(labels):
        out[f"coef_{_safe_name(lab)}"] = coef_mat[i, :]

    return out.sort_values("importance_abs_mean", ascending=False).reset_index(drop=True)


def _perm_importance(
    estimator: Pipeline,
    X: pd.DataFrame,
    y,
    features: List[str],
    *,
    scoring,
    n_repeats: int,
    seed: int,
    n_jobs: int,
) -> pd.DataFrame:
    if permutation_importance is None:
        raise RuntimeError(
            "permutation_importance is not available in your scikit-learn version. "
            "Upgrade scikit-learn or run with --fi-kind coef/none."
        )

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


def _safe_train_test_split_multiclass(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float,
    seed: int,
    try_stratify: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    if not try_stratify:
        return train_test_split(X, y, test_size=test_size, random_state=seed, shuffle=True)

    try:
        return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y, shuffle=True)
    except ValueError as e:
        vc = y.value_counts(dropna=False)
        rare = vc[vc < 2]
        print("\n[WARN] Stratified split failed; falling back to non-stratified split.")
        print(f"[WARN] Reason: {e}")
        if len(rare) > 0:
            print("[WARN] Classes with <2 samples:")
            for cls, cnt in rare.items():
                print(f"  - {cls}: {cnt}")
        return train_test_split(X, y, test_size=test_size, random_state=seed, shuffle=True)


def _make_cv_for_multiclass(y: pd.Series, n_splits: int, seed: int):
    """Prefer StratifiedKFold if feasible; otherwise fall back to KFold."""
    n_splits = int(n_splits)
    if n_splits < 2:
        n_splits = 2
    vc = y.value_counts(dropna=False)
    min_count = int(vc.min()) if len(vc) else 0
    if min_count >= n_splits:
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
    print(f"[WARN] CV: min class count={min_count} < cv={n_splits}. Using non-stratified KFold for tuning.")
    return KFold(n_splits=n_splits, shuffle=True, random_state=int(seed))


def _parse_multilabel_cell(v) -> List[str]:
    """Accepts list-like, or a string representation like "['A','B']" or "A,B"."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return []
    if isinstance(v, list):
        return [str(x) for x in v if str(x).strip()]
    if isinstance(v, tuple):
        return [str(x) for x in v if str(x).strip()]
    s = str(v).strip()
    if s == "":
        return []
    if s.startswith("[") and s.endswith("]"):
        s2 = s[1:-1].strip()
        if not s2:
            return []
        parts = [p.strip().strip("'\"") for p in s2.split(",")]
        return [p for p in parts if p]
    if "," in s:
        return [p.strip() for p in s.split(",") if p.strip()]
    return [s]


def _make_logreg_balanced() -> LogisticRegression:
    """
    Create a LogisticRegression instance as compatible as possible across sklearn versions.
    Tries lbfgs first (faster), then falls back to liblinear (older compatibility).
    """
    sig = inspect.signature(LogisticRegression.__init__)
    base_kwargs = dict(max_iter=2000, class_weight="balanced")

    # Only pass multi_class if supported
    if "multi_class" in sig.parameters:
        base_kwargs["multi_class"] = "auto"

    # Try solvers in order
    last_err: Exception | None = None
    for solver in ("lbfgs", "liblinear"):
        try:
            return LogisticRegression(solver=solver, **base_kwargs)
        except Exception as e:  # TypeError/ValueError depending on sklearn version
            last_err = e
            continue
    assert last_err is not None
    raise last_err


def main() -> None:
    ap = argparse.ArgumentParser(description="Train supervised baselines (multiclass/multilabel) on WAF features")
    ap.add_argument("--data", required=True)
    ap.add_argument("--task", required=True, choices=["multiclass", "multilabel"])
    ap.add_argument("--label-col", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)

    # Rare-class handling (multiclass only)
    ap.add_argument("--rare-class-policy", default="keep", choices=["keep", "merge", "drop"])
    ap.add_argument("--rare-class-min-count", type=int, default=2)

    # Tuning
    ap.add_argument("--tune", default="none", choices=["none", "grid", "random"])
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--tune-n-iter", type=int, default=60)
    ap.add_argument("--tune-n-jobs", type=int, default=-1)
    ap.add_argument("--tune-sample-n", type=int, default=0, help="Optional cap rows for tuning only (0 = full train)")
    ap.add_argument("--tune-results-out", default=None)

    # Feature importance
    ap.add_argument("--fi-kind", default="coef", choices=["coef", "permutation", "both", "none"])
    ap.add_argument("--fi-topk", type=int, default=25)
    ap.add_argument("--fi-n-repeats", type=int, default=5)
    ap.add_argument("--fi-n-jobs", type=int, default=-1)
    ap.add_argument("--fi-scoring", default=None)
    ap.add_argument("--max-eval", type=int, default=0, help="Optional cap rows for permutation FI only (0 = full test split)")

    args = ap.parse_args()

    df = _read_table(args.data)

    missing = [c for c in DEFAULT_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing {len(missing)} features: {missing}")

    X_all = df[DEFAULT_FEATURES].fillna(0)

    fi_kind = (args.fi_kind or "none").strip().lower()
    do_coef = fi_kind in {"coef", "both"}
    do_perm = fi_kind in {"permutation", "both"}

    if args.task == "multiclass":
        y_all = df[args.label_col].astype(str)

        vc = y_all.value_counts(dropna=False)
        rare_classes = set(vc[vc < int(args.rare_class_min_count)].index.tolist())
        if rare_classes:
            if args.rare_class_policy == "drop":
                mask = ~y_all.isin(rare_classes)
                X_all = X_all.loc[mask].reset_index(drop=True)
                y_all = y_all.loc[mask].reset_index(drop=True)
                print(f"[INFO] Dropped {int((~mask).sum())} rows from rare classes (<{args.rare_class_min_count}).")
            elif args.rare_class_policy == "merge":
                y_all = y_all.where(~y_all.isin(rare_classes), other="OTHER_RARE")
                print(f"[INFO] Merged {len(rare_classes)} rare classes into OTHER_RARE.")
            else:
                print(f"[INFO] Keeping {len(rare_classes)} rare classes; stratify may fail.")

        X_train, X_test, y_train, y_test = _safe_train_test_split_multiclass(
            X_all, y_all, test_size=args.test_size, seed=args.seed, try_stratify=True
        )

        pipe = Pipeline([("scaler", StandardScaler()), ("lr", _make_logreg_balanced())])
        estimator: Pipeline = pipe

        if args.tune != "none":
            X_tune, y_tune = X_train, y_train
            if args.tune_sample_n and args.tune_sample_n > 0 and len(X_tune) > args.tune_sample_n:
                idx = np.random.default_rng(args.seed).choice(len(X_tune), size=int(args.tune_sample_n), replace=False)
                X_tune = X_tune.iloc[idx]
                y_tune = y_tune.iloc[idx]
                print(f"[TUNE] Using tune sample n={len(X_tune)}")

            scoring = "f1_weighted"
            cv_obj = _make_cv_for_multiclass(y_tune, args.cv, args.seed)

            if args.tune == "grid":
                param_grid = {"lr__C": [0.01, 0.1, 1.0, 10.0, 100.0]}
                search = GridSearchCV(
                    estimator,
                    param_grid=param_grid,
                    scoring=scoring,
                    cv=cv_obj,
                    n_jobs=int(args.tune_n_jobs),
                    refit=True,
                    verbose=1,
                )
            else:
                param_dist = {"lr__C": np.logspace(-3, 3, 25)}
                search = RandomizedSearchCV(
                    estimator,
                    param_distributions=param_dist,
                    n_iter=int(args.tune_n_iter),
                    scoring=scoring,
                    cv=cv_obj,
                    n_jobs=int(args.tune_n_jobs),
                    random_state=int(args.seed),
                    refit=True,
                    verbose=1,
                )

            search.fit(X_tune, y_tune)
            best_params = dict(search.best_params_)
            print(f"[TUNE] best_params={best_params}")
            print(f"[TUNE] best_score={search.best_score_}")

            if args.tune_results_out:
                pd.DataFrame(search.cv_results_).to_csv(args.tune_results_out, index=False)
                print(f"[TUNE] Saved results: {args.tune_results_out}")

            # Refit on full train split
            estimator = pipe.set_params(**best_params)
            estimator.fit(X_train, y_train)
        else:
            estimator.fit(X_train, y_train)

        y_pred = estimator.predict(X_test)
        print("=== Test evaluation (multiclass) ===")
        print(classification_report(y_test, y_pred, digits=4, zero_division=0))

        joblib.dump(estimator, args.out)
        print(f"Saved model: {args.out}")

        if do_coef:
            df_coef = _coef_importance_multiclass(estimator, DEFAULT_FEATURES)
            out_path = _derive_fi_path(args.out, "coef")
            df_coef.to_csv(out_path, index=False)
            print(f"Saved coef importance: {out_path}")
            _print_top(df_coef, "importance_abs_mean", args.fi_topk)

        if do_perm:
            X_fi, y_fi = X_test, y_test
            if args.max_eval and args.max_eval > 0 and len(X_fi) > args.max_eval:
                idx = np.random.default_rng(args.seed).choice(len(X_fi), size=int(args.max_eval), replace=False)
                X_fi = X_fi.iloc[idx]
                y_fi = y_fi.iloc[idx]

            scoring = args.fi_scoring or "f1_weighted"
            df_perm = _perm_importance(
                estimator,
                X_fi,
                y_fi,
                DEFAULT_FEATURES,
                scoring=scoring,
                n_repeats=int(args.fi_n_repeats),
                seed=int(args.seed),
                n_jobs=int(args.fi_n_jobs),
            )
            out_path = _derive_fi_path(args.out, "perm")
            df_perm.to_csv(out_path, index=False)
            print(f"Saved permutation importance (scoring={scoring}): {out_path}")
            _print_top(df_perm, "perm_importance_mean", args.fi_topk)

        return

    # -------------------------
    # Multilabel (OvR)
    # -------------------------
    y_raw = df[args.label_col].apply(_parse_multilabel_cell)
    mlb = MultiLabelBinarizer()
    Y_all = mlb.fit_transform(y_raw)
    labels = [str(c) for c in mlb.classes_]

    X_train, X_test, Y_train, Y_test = train_test_split(
        X_all, Y_all, test_size=args.test_size, random_state=args.seed, shuffle=True
    )

    base_lr = _make_logreg_balanced()
    pipe = Pipeline([("scaler", StandardScaler()), ("ovr", OneVsRestClassifier(base_lr))])
    estimator: Pipeline = pipe

    if args.tune != "none":
        X_tune, Y_tune = X_train, Y_train
        if args.tune_sample_n and args.tune_sample_n > 0 and len(X_tune) > args.tune_sample_n:
            idx = np.random.default_rng(args.seed).choice(len(X_tune), size=int(args.tune_sample_n), replace=False)
            X_tune = X_tune.iloc[idx]
            Y_tune = Y_tune[idx]
            print(f"[TUNE] Using tune sample n={len(X_tune)}")

        scorer = make_scorer(f1_score, average="micro", zero_division=0)
        cv_obj = KFold(n_splits=max(2, int(args.cv)), shuffle=True, random_state=int(args.seed))

        if args.tune == "grid":
            param_grid = {"ovr__estimator__C": [0.01, 0.1, 1.0, 10.0, 100.0]}
            search = GridSearchCV(
                estimator, param_grid=param_grid, scoring=scorer, cv=cv_obj,
                n_jobs=int(args.tune_n_jobs), refit=True, verbose=1
            )
        else:
            param_dist = {"ovr__estimator__C": np.logspace(-3, 3, 25)}
            search = RandomizedSearchCV(
                estimator, param_distributions=param_dist, n_iter=int(args.tune_n_iter),
                scoring=scorer, cv=cv_obj, n_jobs=int(args.tune_n_jobs),
                random_state=int(args.seed), refit=True, verbose=1
            )

        search.fit(X_tune, Y_tune)
        best_params = dict(search.best_params_)
        print(f"[TUNE] best_params={best_params}")
        print(f"[TUNE] best_score={search.best_score_}")

        if args.tune_results_out:
            pd.DataFrame(search.cv_results_).to_csv(args.tune_results_out, index=False)
            print(f"[TUNE] Saved results: {args.tune_results_out}")

        estimator = pipe.set_params(**best_params)
        estimator.fit(X_train, Y_train)
    else:
        estimator.fit(X_train, Y_train)

    Y_pred = estimator.predict(X_test)
    f1_micro = f1_score(Y_test, Y_pred, average="micro", zero_division=0)
    f1_macro = f1_score(Y_test, Y_pred, average="macro", zero_division=0)
    h_loss = hamming_loss(Y_test, Y_pred)
    print("=== Test evaluation (multilabel) ===")
    print(f"F1 micro:  {f1_micro:.6f}")
    print(f"F1 macro:  {f1_macro:.6f}")
    print(f"Hamming:   {h_loss:.6f}")

    joblib.dump({"model": estimator, "mlb": mlb}, args.out)
    print(f"Saved model+mlb: {args.out}")

    if do_coef:
        df_coef = _coef_importance_multilabel(estimator, DEFAULT_FEATURES, labels)
        out_path = _derive_fi_path(args.out, "coef")
        df_coef.to_csv(out_path, index=False)
        print(f"Saved coef importance: {out_path}")
        _print_top(df_coef, "importance_abs_mean", args.fi_topk)

    if do_perm:
        X_fi, Y_fi = X_test, Y_test
        if args.max_eval and args.max_eval > 0 and len(X_fi) > args.max_eval:
            idx = np.random.default_rng(args.seed).choice(len(X_fi), size=int(args.max_eval), replace=False)
            X_fi = X_fi.iloc[idx]
            Y_fi = Y_fi[idx]

        scorer = make_scorer(f1_score, average="micro", zero_division=0) if args.fi_scoring is None else args.fi_scoring
        df_perm = _perm_importance(
            estimator, X_fi, Y_fi, DEFAULT_FEATURES,
            scoring=scorer, n_repeats=int(args.fi_n_repeats),
            seed=int(args.seed), n_jobs=int(args.fi_n_jobs)
        )
        out_path = _derive_fi_path(args.out, "perm")
        df_perm.to_csv(out_path, index=False)
        print(f"Saved permutation importance: {out_path}")
        _print_top(df_perm, "perm_importance_mean", args.fi_topk)


if __name__ == "__main__":
    main()
