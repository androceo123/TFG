from __future__ import annotations

import argparse
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


DEFAULT_FEATURES = [
    "uri_len","path_depth","query_len","n_query_params","max_param_value_len",
    "uri_pct_non_alnum_ratio","encoded","suspicious_tokens_count","has_suspicious_tokens",
    "uncommon_method","req_content_length","body_len","body_suspicious_tokens_count","body_has_suspicious_tokens","body_encoded",
    # method one-hot
    "method_GET","method_POST","method_HEAD","method_PUT","method_DELETE","method_PATCH","method_OPTIONS",
    "method_TRACE","method_CONNECT","method_OTHER",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Parquet/CSV with feature columns")
    ap.add_argument("--label-col", default="label_binary", help="0 normal / 1 attack")
    ap.add_argument("--out", required=True, help="Path to save joblib pipeline")
    ap.add_argument("--nu", type=float, default=0.05)
    ap.add_argument("--gamma", default="scale")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_parquet(args.data) if args.data.lower().endswith(".parquet") else pd.read_csv(args.data)

    # Train OCSVM only on normal traffic (label_binary == 0)
    df_train = df[df[args.label_col] == 0].copy()
    X = df_train[DEFAULT_FEATURES].fillna(0)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("ocsvm", OneClassSVM(kernel="rbf", nu=args.nu, gamma=args.gamma)),
    ])

    model.fit(X)

    # Quick sanity eval on a mixed holdout (if available)
    if args.label_col in df.columns:
        mixed = df.sample(min(len(df), 50000), random_state=args.seed)
        Xm = mixed[DEFAULT_FEATURES].fillna(0)
        ym = mixed[args.label_col].astype(int)

        # ocsvm predicts +1 normal, -1 anomaly
        pred = model.predict(Xm)
        yhat = (pred == -1).astype(int)  # 1 = attack/anomaly
        print(classification_report(ym, yhat, digits=4))

    joblib.dump(model, args.out)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
