from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import joblib
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


DEFAULT_FEATURES = [
    "uri_len","path_depth","query_len","n_query_params","max_param_value_len",
    "uri_pct_non_alnum_ratio","encoded","suspicious_tokens_count","has_suspicious_tokens",
    "uncommon_method","req_content_length","body_len","body_suspicious_tokens_count","body_has_suspicious_tokens","body_encoded",
    # method one-hot
    "method_GET","method_POST","method_HEAD","method_PUT","method_DELETE","method_PATCH","method_OPTIONS",
    "method_TRACE","method_CONNECT","method_OTHER",
]


def _read_table(path: str) -> pd.DataFrame:
    p = path.lower()
    if p.endswith(".parquet"):
        return pd.read_parquet(path)
    if p.endswith(".csv"):
        return pd.read_csv(path)
    # Try CSV as a safe default
    return pd.read_csv(path)


def _load_many(paths: List[str]) -> pd.DataFrame:
    frames = []
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
    # ocsvm predicts +1 normal, -1 anomaly
    pred_raw = model.predict(X)
    out = df.copy()
    out["pred_raw"] = pred_raw
    out["pred_anomaly"] = (pred_raw == -1).astype(int)

    # Optional scoring (higher = more normal for OneClassSVM decision_function)
    if hasattr(model, "decision_function"):
        try:
            out["score"] = model.decision_function(X)
        except Exception:
            pass
    return out


def _eval_if_possible(df_pred: pd.DataFrame, label_col: str) -> None:
    if label_col in df_pred.columns:
        y_true = df_pred[label_col].astype(int)
        y_hat = df_pred["pred_anomaly"].astype(int)
        print(classification_report(y_true, y_hat, digits=4))
        return

    # If no labels, at least print anomaly rate
    rate = float(df_pred["pred_anomaly"].mean()) if len(df_pred) else 0.0
    print(f"No '{label_col}' column found; anomaly rate = {rate:.4f} (fraction predicted as anomaly).")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train/Test One-Class SVM (OCSVM) on WAF features.")
    ap.add_argument("--mode", choices=["train", "test"], default="train",
                    help="train: fit model on normal traffic; test: load model and evaluate on separate data")
    ap.add_argument("--data", nargs="+", required=True, help="One or more CSV/Parquet files with feature columns")
    ap.add_argument("--label-col", default="label_binary", help="0 normal / 1 attack")
    ap.add_argument("--out", help="(train) Path to save joblib pipeline")
    ap.add_argument("--model", help="(test) Path to load joblib pipeline")
    ap.add_argument("--nu", type=float, default=0.05)
    ap.add_argument("--gamma", default="scale")
    ap.add_argument("--test-size", type=float, default=0.2,
                    help="(train) Fraction of NORMAL traffic held out for sanity evaluation")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-eval", type=int, default=50000,
                    help="Max rows used for evaluation printing (to keep it fast)")
    ap.add_argument("--pred-out", default=None,
                    help="Optional: save predictions for the (test) data to this CSV/Parquet path")
    ap.add_argument("--eval-data", nargs="*", default=None,
                    help="(train, optional) Extra CSV/Parquet file(s) to include in sanity evaluation (e.g., separate attack file)")
    args = ap.parse_args()

    mode: str = args.mode
    label_col: str = args.label_col

    if mode == "train" and not args.out:
        ap.error("--out is required in --mode train")
    if mode == "test" and not args.model:
        ap.error("--model is required in --mode test")

    df = _load_many(args.data)
    _ensure_features(df, DEFAULT_FEATURES)

    if mode == "train":
        if label_col not in df.columns:
            raise ValueError(
                f"Training requires '{label_col}' to identify normal traffic (0). "
                f"Got columns: {list(df.columns)[:50]}"
            )

        df_norm = df[df[label_col] == 0].copy()
        if len(df_norm) == 0:
            raise ValueError(f"No normal rows found where {label_col} == 0. Cannot train OCSVM.")

        # Split ONLY normals so we never evaluate on the exact same normal rows we fit on
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

        # Sanity eval: held-out normals + attacks present in the input + optional eval-data
        eval_parts = []
        if len(df_norm_hold) > 0:
            eval_parts.append(df_norm_hold)

        # attacks inside training input (if any)
        df_att = df[df[label_col] == 1].copy()
        if len(df_att) > 0:
            eval_parts.append(df_att)

        # extra eval files (e.g., separate attack file)
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
        else:
            print("No hold-out normals or eval data available for sanity evaluation.")

        joblib.dump(model, args.out)
        print(f"Saved model: {args.out}")
        return

    # mode == "test"
    model: Pipeline = joblib.load(args.model)

    df_eval = df
    if args.max_eval and len(df_eval) > args.max_eval:
        df_eval = df_eval.sample(args.max_eval, random_state=args.seed).reset_index(drop=True)

    df_pred = _predict(model, df_eval, DEFAULT_FEATURES)
    print("=== Test evaluation ===")
    _eval_if_possible(df_pred, label_col)

    if args.pred_out:
        _save_table(df_pred, args.pred_out)
        print(f"Saved predictions: {args.pred_out}")


if __name__ == "__main__":
    main()
