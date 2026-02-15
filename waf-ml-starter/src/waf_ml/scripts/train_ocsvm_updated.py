from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd

from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


DEFAULT_FEATURES = [
    "uri_len", "path_depth", "query_len", "n_query_params", "max_param_value_len",
    "uri_pct_non_alnum_ratio", "encoded", "suspicious_tokens_count", "has_suspicious_tokens",
    "uncommon_method", "req_content_length", "body_len", "body_suspicious_tokens_count", "body_has_suspicious_tokens", "body_encoded",
    # method one-hot
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


def _load_many(paths: List[str]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for p in paths:
        df = _read_table(p)
        df["_source_file"] = Path(p).name
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]


def _ensure_features(df: pd.DataFrame, features: List[str]) -> None:
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing {len(missing)} feature columns: {missing}\n"
            f"Available columns (sample): {list(df.columns)[:50]}"
        )


def _save_table(df: pd.DataFrame, path: str) -> None:
    p = path.lower()
    if p.endswith(".parquet"):
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def _predict(model: Pipeline, df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    X = df[features].fillna(0)
    # OneClassSVM: +1 normal, -1 anomaly
    pred_raw = model.predict(X)
    out = df.copy()
    out["pred_raw"] = pred_raw
    out["pred_anomaly"] = (pred_raw == -1).astype(int)

    # Higher = more normal (for OneClassSVM)
    try:
        out["score"] = model.decision_function(X)
    except Exception:
        pass
    return out


def _eval_if_possible(df_pred: pd.DataFrame, label_col: str) -> None:
    if label_col in df_pred.columns:
        y_true = df_pred[label_col].astype(int)
        y_hat = df_pred["pred_anomaly"].astype(int)
        print(classification_report(y_true, y_hat, digits=4, zero_division=0))
        return

    rate = float(df_pred["pred_anomaly"].mean()) if len(df_pred) else 0.0
    print(f"No '{label_col}' column found; anomaly rate = {rate:.4f} (fraction predicted as anomaly).")


def _fi_output_path(base_model_path: str, tag: str) -> str:
    return f"{base_model_path}.feature_importance.{tag}.csv"


def _print_top_fi(df_fi: pd.DataFrame, col: str, k: int) -> None:
    if df_fi is None or df_fi.empty:
        return
    k = int(k) if k else 0
    if k <= 0:
        return
    view = df_fi.sort_values(col, ascending=False).head(k)
    print(f"\n=== Top {k} features by {col} ===")
    for _, r in view.iterrows():
        print(f"{r['feature']}: {r[col]:.6g}")


def _compute_permutation_fi(
    model: Pipeline,
    X: pd.DataFrame,
    y: Optional[np.ndarray],
    features: List[str],
    *,
    seed: int,
    n_repeats: int,
    n_jobs: int,
    fi_metric: str,
) -> pd.DataFrame:
    """
    Permutation importance for OCSVM.

    - If y is provided (0 normal / 1 anomaly):
        * roc_auc  : uses anomaly_score = -decision_function(X)
        * f1       : threshold-based using predict() -> pred_anomaly
    - If y is None:
        * mean_score: uses mean(decision_function); fallback to (1 - anomaly_rate)
    """
    fi_metric = (fi_metric or "").strip().lower()

    if y is not None:
        y = np.asarray(y).astype(int)

        if fi_metric in {"", "auto", "roc_auc", "auc"}:
            def scorer(estimator: Pipeline, X_, y_) -> float:
                # decision_function higher => more normal; invert to get anomaly score
                try:
                    s = estimator.decision_function(X_)
                    anomaly_score = -np.asarray(s)
                except Exception:
                    # fallback: use pred_anomaly as a crude score
                    pred = estimator.predict(X_)
                    anomaly_score = (pred == -1).astype(float)

                # ROC-AUC needs both classes
                if len(np.unique(y_)) < 2:
                    return 0.0
                return float(roc_auc_score(y_, anomaly_score))

            scoring = scorer
            score_name = "roc_auc_anomaly"

        elif fi_metric in {"f1", "f1_anomaly"}:
            def scorer(estimator: Pipeline, X_, y_) -> float:
                pred = estimator.predict(X_)
                pred_anom = (pred == -1).astype(int)
                return float(f1_score(y_, pred_anom, zero_division=0))

            scoring = scorer
            score_name = "f1_anomaly"

        else:
            raise ValueError(f"Unknown --fi-metric '{fi_metric}'. Use: roc_auc | f1 | mean_score")

    else:
        # Unsupervised / no labels
        if fi_metric not in {"", "auto", "mean_score"}:
            # force safe behavior rather than silently doing something else
            raise ValueError("When labels are missing, --fi-metric must be 'mean_score' (or auto).")

        def scorer(estimator: Pipeline, X_, y_=None) -> float:
            try:
                s = estimator.decision_function(X_)
                return float(np.mean(s))
            except Exception:
                pred = estimator.predict(X_)
                anom = (pred == -1).astype(int)
                return float(1.0 - np.mean(anom))

        scoring = scorer
        score_name = "mean_decision_function"

    res = permutation_importance(
        model,
        X,
        y if y is not None else None,
        scoring=scoring,
        n_repeats=int(n_repeats),
        random_state=int(seed),
        n_jobs=int(n_jobs),
    )

    df_fi = (
        pd.DataFrame({
            "feature": features,
            f"perm_importance_mean_{score_name}": res.importances_mean,
            f"perm_importance_std_{score_name}": res.importances_std,
        })
        .sort_values(f"perm_importance_mean_{score_name}", ascending=False)
        .reset_index(drop=True)
    )
    return df_fi


def main() -> None:
    ap = argparse.ArgumentParser(description="Train/Test One-Class SVM (OCSVM) on WAF features.")
    ap.add_argument(
        "--mode",
        choices=["train", "test"],
        default="train",
        help="train: fit model on normal traffic; test: load model and evaluate on separate data",
    )
    ap.add_argument("--data", nargs="+", required=True, help="One or more CSV/Parquet files with feature columns")
    ap.add_argument("--label-col", default="label_binary", help="0 normal / 1 attack (if present)")
    ap.add_argument("--out", help="(train) Path to save joblib pipeline")
    ap.add_argument("--model", help="(test) Path to load joblib pipeline")
    ap.add_argument("--nu", type=float, default=0.05)
    ap.add_argument("--gamma", default="scale")
    ap.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="(train) Fraction of NORMAL traffic held out for sanity evaluation",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--max-eval",
        type=int,
        default=50000,
        help="Max rows used for evaluation printing / FI (to keep it fast)",
    )
    ap.add_argument(
        "--pred-out",
        default=None,
        help="Optional: save predictions for the (test) data to this CSV/Parquet path",
    )
    ap.add_argument(
        "--eval-data",
        nargs="*",
        default=None,
        help="(train, optional) Extra CSV/Parquet file(s) to include in sanity evaluation (e.g., separate attack file)",
    )

    # --- Feature Importance options ---
    ap.add_argument(
        "--fi-kind",
        default="permutation",
        choices=["permutation", "none"],
        help="OCSVM has no native coefficients; default is permutation importance.",
    )
    ap.add_argument(
        "--fi-metric",
        default="auto",
        choices=["auto", "roc_auc", "f1", "mean_score"],
        help="Scoring used for permutation importance. "
             "auto => roc_auc if labels exist else mean_score.",
    )
    ap.add_argument("--fi-n-repeats", type=int, default=10, help="Permutation repeats (more = stabler).")
    ap.add_argument("--fi-n-jobs", type=int, default=-1, help="Permutation n_jobs.")
    ap.add_argument("--fi-topk", type=int, default=25, help="Print top-K features.")
    ap.add_argument(
        "--fi-out",
        default=None,
        help="Optional: override FI output path. Default: <model_path>.feature_importance.perm.csv",
    )

    args = ap.parse_args()

    mode: str = args.mode
    label_col: str = args.label_col

    if mode == "train" and not args.out:
        ap.error("--out is required in --mode train")
    if mode == "test" and not args.model:
        ap.error("--model is required in --mode test")

    df = _load_many(args.data)
    _ensure_features(df, DEFAULT_FEATURES)

    fi_kind = (args.fi_kind or "").strip().lower()
    do_fi = fi_kind != "none"

    # Resolve FI metric
    fi_metric = (args.fi_metric or "auto").strip().lower()

    if mode == "train":
        if label_col not in df.columns:
            raise ValueError(
                f"Training requires '{label_col}' to identify normal traffic (0). "
                f"Got columns: {list(df.columns)[:50]}"
            )

        df_norm = df[df[label_col] == 0].copy()
        if len(df_norm) == 0:
            raise ValueError(f"No normal rows found where {label_col} == 0. Cannot train OCSVM.")

        # Split ONLY normals so we don't leak eval normals into training
        if args.test_size and 0.0 < args.test_size < 1.0 and len(df_norm) >= 2:
            df_norm_train, df_norm_hold = train_test_split(
                df_norm, test_size=args.test_size, random_state=args.seed, shuffle=True
            )
        else:
            df_norm_train, df_norm_hold = df_norm, df_norm.iloc[0:0].copy()

        X_train = df_norm_train[DEFAULT_FEATURES].fillna(0)

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("ocsvm", OneClassSVM(kernel="rbf", nu=args.nu, gamma=args.gamma)),
        ])
        model.fit(X_train)

        # Build sanity eval set: held-out normals + attacks from input + optional eval-data
        eval_parts: List[pd.DataFrame] = []
        if len(df_norm_hold) > 0:
            eval_parts.append(df_norm_hold)

        df_att = df[df[label_col] == 1].copy()
        if len(df_att) > 0:
            eval_parts.append(df_att)

        if args.eval_data:
            df_extra = _load_many(args.eval_data)
            _ensure_features(df_extra, DEFAULT_FEATURES)
            eval_parts.append(df_extra)

        if eval_parts:
            df_eval = pd.concat(eval_parts, ignore_index=True)

            if args.max_eval and len(df_eval) > args.max_eval:
                df_eval = df_eval.sample(args.max_eval, random_state=args.seed).reset_index(drop=True)

            df_pred = _predict(model, df_eval, DEFAULT_FEATURES)
            print("=== Sanity evaluation (held-out normals + eval data; no overlap with training normals) ===")
            _eval_if_possible(df_pred, label_col)

            # --- Feature importance on eval set ---
            if do_fi:
                X_fi = df_eval[DEFAULT_FEATURES].fillna(0)
                y_fi = df_eval[label_col].astype(int).to_numpy() if label_col in df_eval.columns else None

                metric = fi_metric
                if metric in {"auto", ""}:
                    metric = "roc_auc" if y_fi is not None else "mean_score"

                df_fi = _compute_permutation_fi(
                    model,
                    X_fi,
                    y_fi,
                    DEFAULT_FEATURES,
                    seed=args.seed,
                    n_repeats=args.fi_n_repeats,
                    n_jobs=args.fi_n_jobs,
                    fi_metric=metric,
                )

                out_path = args.fi_out or _fi_output_path(args.out, "perm")
                df_fi.to_csv(out_path, index=False)
                print(f"Saved feature importance (permutation, metric={metric}): {out_path}")

                col = [c for c in df_fi.columns if c.startswith("perm_importance_mean_")][0]
                _print_top_fi(df_fi, col, args.fi_topk)

        else:
            print("No hold-out normals or eval data available for sanity evaluation.")

            # If no eval set, compute FI on a sample of train normals using mean_score
            if do_fi:
                df_fi_base = df_norm_train
                if args.max_eval and len(df_fi_base) > args.max_eval:
                    df_fi_base = df_fi_base.sample(args.max_eval, random_state=args.seed).reset_index(drop=True)

                X_fi = df_fi_base[DEFAULT_FEATURES].fillna(0)

                metric = "mean_score"
                df_fi = _compute_permutation_fi(
                    model,
                    X_fi,
                    y=None,
                    features=DEFAULT_FEATURES,
                    seed=args.seed,
                    n_repeats=args.fi_n_repeats,
                    n_jobs=args.fi_n_jobs,
                    fi_metric=metric,
                )
                out_path = args.fi_out or _fi_output_path(args.out, "perm")
                df_fi.to_csv(out_path, index=False)
                print(f"Saved feature importance (permutation, metric={metric}): {out_path}")

                col = [c for c in df_fi.columns if c.startswith("perm_importance_mean_")][0]
                _print_top_fi(df_fi, col, args.fi_topk)

        joblib.dump(model, args.out)
        print(f"Saved model: {args.out}")
        return

    # ---- mode == test ----
    model: Pipeline = joblib.load(args.model)

    df_eval = df
    if args.max_eval and len(df_eval) > args.max_eval:
        df_eval = df_eval.sample(args.max_eval, random_state=args.seed).reset_index(drop=True)

    df_pred = _predict(model, df_eval, DEFAULT_FEATURES)
    print("=== Test evaluation ===")
    _eval_if_possible(df_pred, label_col)

    # --- Feature importance on test data ---
    if do_fi:
        X_fi = df_eval[DEFAULT_FEATURES].fillna(0)
        y_fi = df_eval[label_col].astype(int).to_numpy() if label_col in df_eval.columns else None

        metric = fi_metric
        if metric in {"auto", ""}:
            metric = "roc_auc" if y_fi is not None else "mean_score"

        df_fi = _compute_permutation_fi(
            model,
            X_fi,
            y_fi,
            DEFAULT_FEATURES,
            seed=args.seed,
            n_repeats=args.fi_n_repeats,
            n_jobs=args.fi_n_jobs,
            fi_metric=metric,
        )

        out_path = args.fi_out or _fi_output_path(args.model, "perm")
        df_fi.to_csv(out_path, index=False)
        print(f"Saved feature importance (permutation, metric={metric}): {out_path}")

        col = [c for c in df_fi.columns if c.startswith("perm_importance_mean_")][0]
        _print_top_fi(df_fi, col, args.fi_topk)

    if args.pred_out:
        _save_table(df_pred, args.pred_out)
        print(f"Saved predictions: {args.pred_out}")


if __name__ == "__main__":
    main()
