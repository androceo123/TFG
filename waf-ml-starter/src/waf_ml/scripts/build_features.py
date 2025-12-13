from __future__ import annotations

import argparse
import pandas as pd
from tqdm import tqdm

from waf_ml.data.srbh_loader import load_srbh_csv, make_labels, SRBHSchema
from waf_ml.features.http_features import extract_http_features


def _to_bytes(v) -> bytes:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return b""
    s = str(v)
    if s in {"", "nan", "None"}:
        return b""
    return s.encode("utf-8", errors="ignore")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to raw SR-BH CSV/TSV")
    ap.add_argument("--output", required=True, help="Path to output parquet/csv (features)")

    # these match your dataset
    ap.add_argument("--method-col", default="request_http_method")
    ap.add_argument("--uri-col", default="request_http_request")
    ap.add_argument("--body-col", default="request_body")

    ap.add_argument("--sep", default="\t", help="Input delimiter (use ',' for CSV)")
    ap.add_argument("--sample-n", type=int, default=0, help="If >0, only process first N rows (debug)")
    ap.add_argument("--out-format", default="parquet", choices=["parquet", "csv"])
    args = ap.parse_args()

    schema = SRBHSchema(
        sep=args.sep,
        method_col=args.method_col,
        uri_col=args.uri_col,
        body_col=args.body_col,
    )

    df = load_srbh_csv(args.input, schema=schema)
    if args.sample_n and args.sample_n > 0:
        df = df.head(args.sample_n).copy()

    # Creates: label_binary, label_multiclass, label_multilabel
    df = make_labels(df, schema=schema)

    # Safer null handling
    df[schema.method_col] = df[schema.method_col].fillna("")
    df[schema.uri_col] = df[schema.uri_col].fillna("")
    df[schema.body_col] = df[schema.body_col].fillna("")

    tqdm.pandas(desc="Extracting features")
    feats = df.progress_apply(
        lambda r: extract_http_features(
            method=str(r[schema.method_col]),
            uri=str(r[schema.uri_col]),
            headers=None,
            body=_to_bytes(r[schema.body_col]),
        ),
        axis=1,
    )

    feat_df = pd.DataFrame(list(feats))
    out = pd.concat([df, feat_df], axis=1)

    if args.out_format == "parquet" or args.output.lower().endswith(".parquet"):
        out.to_parquet(args.output, index=False)
    else:
        out.to_csv(args.output, index=False)

    print(f"Wrote: {args.output}  rows={len(out)}  cols={len(out.columns)}")


if __name__ == "__main__":
    main()
