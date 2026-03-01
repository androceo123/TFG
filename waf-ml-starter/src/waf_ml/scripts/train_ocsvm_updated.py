from __future__ import annotations

"""waf_ml.scripts.train_ocsvm_updated

OCSVM (RBF) training/testing script with optional hyperparameter tuning.

Backwards compatible with the original usage:

  # Train on normals
  python -m waf_ml.scripts.train_ocsvm_updated --mode train --data normals.parquet --label-col label_binary --out model.joblib --test-size 0

  # Test on mixed normals/anoms
  python -m waf_ml.scripts.train_ocsvm_updated --mode test --model model.joblib --data test.parquet --label-col label_binary

Adds optional tuning:
  --tune grid|random (searches nu/gamma using a validation set built from held-out normals + --eval-data)

Notes:
  - Training ALWAYS fits only on rows where label_col == 0 (normals).
  - For tuning with roc_auc/f1 you should provide anomalies in --eval-data, and labels in eval data.
"""

import argparse
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

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


def _load_many(paths: Sequence[str]) -> pd.DataFrame:
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
    pred_raw = model.predict(X)  # +1 normal, -1 anomaly
    out = df.copy()
    out["pred_raw"] = pred_raw
    out["pred_anomaly"] = (pred_raw == -1).astype(int)
    try:
        out["score"] = model.decision_function(X)  # higher = more normal
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
    """Permutation importance for OCSVM.

    - If y exists (0 normal / 1 anomaly):
        * roc_auc: uses anomaly_score = -decision_function(X)
        * f1: uses predict() -> pred_anomaly
    - If y missing: mean_score uses mean(decision_function).
    """
    fi_metric = (fi_metric or "").strip().lower()

    if y is not None:
        y = np.asarray(y).astype(int)

        if fi_metric in {"", "auto", "roc_auc", "auc"}:
            def scorer(estimator: Pipeline, X_, y_) -> float:
                try:
                    s = estimator.decision_function(X_)
                    anomaly_score = -np.asarray(s)
                except Exception:
                    pred = estimator.predict(X_)
                    anomaly_score = (pred == -1).astype(float)
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
        if fi_metric not in {"", "auto", "mean_score"}:
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


def _parse_csv_list(s: Optional[str]) -> List[str]:
    if not s:
        return []
    return [p.strip() for p in str(s).split(",") if p.strip()]


def _parse_nu_grid(s: Optional[str]) -> List[float]:
    return [float(t) for t in _parse_csv_list(s)]


def _parse_gamma_grid(s: Optional[str]) -> List[object]:
    out: List[object] = []
    for t in _parse_csv_list(s):
        tl = t.lower()
        if tl in {"scale", "auto"}:
            out.append(tl)
        else:
            out.append(float(t))
    return out


def _build_ocsvm(nu: float, gamma: object) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("ocsvm", OneClassSVM(kernel="rbf", nu=float(nu), gamma=gamma)),
    ])


def _score_candidate(model: Pipeline, X_eval: pd.DataFrame, y_eval: Optional[np.ndarray], metric: str) -> float:
    metric = (metric or "auto").strip().lower()

    if y_eval is None:
        # Unsupervised fallback
        try:
            return float(np.mean(model.decision_function(X_eval)))
        except Exception:
            pred = model.predict(X_eval)
            return float(1.0 - np.mean((pred == -1).astype(int)))

    y_eval = np.asarray(y_eval).astype(int)

    if metric in {"auto", "roc_auc", "auc"}:
        if len(np.unique(y_eval)) < 2:
            metric = "f1"
        else:
            s = model.decision_function(X_eval)
            anomaly_score = -np.asarray(s)
            return float(roc_auc_score(y_eval, anomaly_score))

    if metric in {"f1", "f1_anomaly"}:
        pred = model.predict(X_eval)
        pred_anom = (pred == -1).astype(int)
        return float(f1_score(y_eval, pred_anom, zero_division=0))

    if metric in {"mean_score"}:
        try:
            return float(np.mean(model.decision_function(X_eval)))
        except Exception:
            pred = model.predict(X_eval)
            return float(1.0 - np.mean((pred == -1).astype(int)))

    raise ValueError(f"Unknown tuning metric '{metric}'. Use roc_auc | f1 | mean_score | auto")


def _iter_candidates(
    tune: str,
    nu_grid: List[float],
    gamma_grid: List[object],
    n_iter: int,
    seed: int,
) -> Iterable[Tuple[float, object]]:
    tune = (tune or "none").strip().lower()

    if tune == "grid":
        for nu in nu_grid:
            for gamma in gamma_grid:
                yield nu, gamma
        return

    if tune == "random":
        rng = np.random.default_rng(int(seed))
        pairs = [(nu, gamma) for nu in nu_grid for gamma in gamma_grid]
        rng.shuffle(pairs)
        for nu, gamma in pairs[: int(n_iter)]:
            yield float(nu), gamma
        return

    return


def main() -> None:
    ap = argparse.ArgumentParser(description="Train/Test One-Class SVM (OCSVM) on WAF features.")
    ap.add_argument("--mode", choices=["train", "test"], default="train")
    ap.add_argument("--data", nargs="+", required=True, help="One or more CSV/Parquet files with feature columns")
    ap.add_argument("--label-col", default="label_binary", help="0 normal / 1 attack (if present)")
    ap.add_argument("--out", help="(train) Path to save joblib pipeline")
    ap.add_argument("--model", help="(test) Path to load joblib pipeline")

    # Baseline params (used when --tune none)
    ap.add_argument("--nu", type=float, default=0.05)
    ap.add_argument("--gamma", default="scale")

    ap.add_argument("--test-size", type=float, default=0.2, help="(train) Held-out NORMAL fraction for sanity eval/tuning")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--max-train", type=int, default=0, help="Optional cap for FINAL training normals rows (0 = no cap)")
    ap.add_argument(
        "--max-eval",
        type=int,
        default=50000,
        help="Max rows used for evaluation printing / FI (0 = use all; default 50k)",
    )
    ap.add_argument("--pred-out", default=None, help="Optional: save predictions for test data")
    ap.add_argument(
        "--eval-data",
        nargs="*",
        default=None,
        help="(train) Extra CSV/Parquet file(s) to include in sanity evaluation / tuning (e.g. attacks file)",
    )

    # Tuning options
    ap.add_argument("--tune", default="none", choices=["none", "grid", "random"], help="Search nu/gamma")
    ap.add_argument("--tune-metric", default="auto", choices=["auto", "roc_auc", "f1", "mean_score"])
    ap.add_argument("--tune-nu-grid", default="0.001,0.003,0.01,0.03,0.05,0.1,0.2")
    ap.add_argument("--tune-gamma-grid", default="scale,0.0001,0.001,0.01,0.1,1,10")
    ap.add_argument("--tune-n-iter", type=int, default=60)
    ap.add_argument("--tune-max-train", type=int, default=0, help="Optional cap for tuning train normals (0 = no cap)")
    ap.add_argument("--tune-max-eval", type=int, default=0, help="Optional cap for tuning eval rows (0 = no cap)")
    ap.add_argument("--tune-results-out", default=None, help="Optional CSV to save per-candidate tuning scores")

    # Feature importance
    ap.add_argument("--fi-kind", default="permutation", choices=["permutation", "none"])
    ap.add_argument("--fi-metric", default="auto", choices=["auto", "roc_auc", "f1", "mean_score"])
    ap.add_argument("--fi-n-repeats", type=int, default=10)
    ap.add_argument("--fi-n-jobs", type=int, default=-1)
    ap.add_argument("--fi-topk", type=int, default=25)
    ap.add_argument("--fi-out", default=None)

    args = ap.parse_args()

    if args.mode == "train" and not args.out:
        ap.error("--out is required in --mode train")
    if args.mode == "test" and not args.model:
        ap.error("--model is required in --mode test")

    df = _load_many(args.data)
    _ensure_features(df, DEFAULT_FEATURES)

    do_fi = (args.fi_kind or "").strip().lower() != "none"
    fi_metric = (args.fi_metric or "auto").strip().lower()

    if args.mode == "train":
        label_col = args.label_col
        if label_col not in df.columns:
            raise ValueError(f"Training requires '{label_col}' to identify normal traffic (0).")

        df_norm = df[df[label_col] == 0].copy()
        if len(df_norm) == 0:
            raise ValueError(f"No normal rows found where {label_col} == 0. Cannot train OCSVM.")

        # Split only normals for holdout eval
        if args.test_size and 0.0 < args.test_size < 1.0 and len(df_norm) >= 2:
            df_norm_train, df_norm_hold = train_test_split(df_norm, test_size=args.test_size, random_state=args.seed, shuffle=True)
        else:
            df_norm_train, df_norm_hold = df_norm, df_norm.iloc[0:0].copy()

        # Tuning train cap
        df_norm_train_tune = df_norm_train
        if args.tune_max_train and args.tune_max_train > 0 and len(df_norm_train_tune) > args.tune_max_train:
            df_norm_train_tune = df_norm_train_tune.sample(args.tune_max_train, random_state=args.seed).reset_index(drop=True)

        # Build eval set (holdout normals + any anomalies in main data + eval-data)
        eval_parts: List[pd.DataFrame] = []
        if len(df_norm_hold) > 0:
            eval_parts.append(df_norm_hold)
        df_att_in_data = df[df[label_col] == 1].copy()
        if len(df_att_in_data) > 0:
            eval_parts.append(df_att_in_data)
        if args.eval_data:
            df_extra = _load_many(args.eval_data)
            _ensure_features(df_extra, DEFAULT_FEATURES)
            eval_parts.append(df_extra)

        df_eval_full = pd.concat(eval_parts, ignore_index=True) if eval_parts else None
        df_eval_tune = df_eval_full
        if df_eval_tune is not None and args.tune_max_eval and args.tune_max_eval > 0 and len(df_eval_tune) > args.tune_max_eval:
            df_eval_tune = df_eval_tune.sample(args.tune_max_eval, random_state=args.seed).reset_index(drop=True)

        best_nu = float(args.nu)
        best_gamma: object = args.gamma
        best_score = -np.inf

        tune_mode = (args.tune or "none").strip().lower()
        if tune_mode != "none":
            if df_eval_tune is None or len(df_eval_tune) == 0:
                raise ValueError("--tune requires an evaluation set. Provide --eval-data and/or a non-zero --test-size.")

            X_train_tune = df_norm_train_tune[DEFAULT_FEATURES].fillna(0)
            X_eval_tune = df_eval_tune[DEFAULT_FEATURES].fillna(0)
            y_eval_tune = df_eval_tune[label_col].astype(int).to_numpy() if label_col in df_eval_tune.columns else None

            nu_grid = _parse_nu_grid(args.tune_nu_grid)
            gamma_grid = _parse_gamma_grid(args.tune_gamma_grid)
            if not nu_grid or not gamma_grid:
                raise ValueError("Empty tuning grids. Check --tune-nu-grid and --tune-gamma-grid")

            rows = []
            print(f"[TUNE] mode={tune_mode} candidates: nu={len(nu_grid)} gamma={len(gamma_grid)}")
            for i, (nu, gamma) in enumerate(_iter_candidates(tune_mode, nu_grid, gamma_grid, args.tune_n_iter, args.seed), start=1):
                m = _build_ocsvm(nu, gamma)
                m.fit(X_train_tune)
                score = _score_candidate(m, X_eval_tune, y_eval_tune, args.tune_metric)
                rows.append({"nu": float(nu), "gamma": str(gamma), "score": float(score)})
                if score > best_score:
                    best_score = score
                    best_nu, best_gamma = float(nu), gamma
                if i % 10 == 0:
                    print(f"[TUNE] tried {i} candidates; current best score={best_score:.6g} (nu={best_nu}, gamma={best_gamma})")

            print(f"[TUNE] BEST score={best_score:.6g} (nu={best_nu}, gamma={best_gamma})")
            if args.tune_results_out:
                pd.DataFrame(rows).sort_values("score", ascending=False).to_csv(args.tune_results_out, index=False)
                print(f"[TUNE] Saved results: {args.tune_results_out}")

        # Final training on ALL normals (optionally capped)
        df_final_train = df_norm
        if args.max_train and args.max_train > 0 and len(df_final_train) > args.max_train:
            df_final_train = df_final_train.sample(args.max_train, random_state=args.seed).reset_index(drop=True)

        X_train = df_final_train[DEFAULT_FEATURES].fillna(0)
        model = _build_ocsvm(best_nu, best_gamma)
        model.fit(X_train)

        # Sanity evaluation + FI
        if df_eval_full is not None and len(df_eval_full) > 0:
            df_eval = df_eval_full
            if args.max_eval and args.max_eval > 0 and len(df_eval) > args.max_eval:
                df_eval = df_eval.sample(args.max_eval, random_state=args.seed).reset_index(drop=True)

            df_pred = _predict(model, df_eval, DEFAULT_FEATURES)
            print("=== Sanity evaluation (held-out normals + eval data) ===")
            _eval_if_possible(df_pred, label_col)

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

        joblib.dump(model, args.out)
        print(f"Saved model: {args.out}")
        return

    # TEST mode
    model: Pipeline = joblib.load(args.model)
    df_eval = df
    if args.max_eval and args.max_eval > 0 and len(df_eval) > args.max_eval:
        df_eval = df_eval.sample(args.max_eval, random_state=args.seed).reset_index(drop=True)

    df_pred = _predict(model, df_eval, DEFAULT_FEATURES)
    print("=== Test evaluation ===")
    _eval_if_possible(df_pred, args.label_col)

    if (args.fi_kind or "").strip().lower() != "none":
        X_fi = df_eval[DEFAULT_FEATURES].fillna(0)
        y_fi = df_eval[args.label_col].astype(int).to_numpy() if args.label_col in df_eval.columns else None

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
