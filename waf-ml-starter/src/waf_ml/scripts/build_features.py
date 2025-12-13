from __future__ import annotations

import argparse
import pandas as pd
from tqdm import tqdm

from waf_ml.data.srbh_loader import load_srbh_csv, make_labels, SRBHSchema
from waf_ml.features.http_features import extract_http_features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to raw SR-BH CSV")
    ap.add_argument("--output", required=True, help="Path to output parquet/csv (features)")
    ap.add_argument("--method-col", default="method")
    ap.add_argument("--uri-col", default="uri")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--out-format", default="parquet", choices=["parquet", "csv"])
    args = ap.parse_args()

    schema = SRBHSchema(method_col=args.method_col, uri_col=args.uri_col, label_col=args.label_col)
    df = load_srbh_csv(args.input, schema=schema)
    df = make_labels(df, schema=schema)

    tqdm.pandas(desc="Extracting features")
    feats = df.progress_apply(lambda r: extract_http_features(method=str(r[schema.method_col]), uri=str(r[schema.uri_col])), axis=1)
    feat_df = pd.DataFrame(list(feats))

    out = pd.concat([df, feat_df], axis=1)

    if args.out_format == "parquet" or args.output.lower().endswith(".parquet"):
        out.to_parquet(args.output, index=False)
    else:
        out.to_csv(args.output, index=False)

    print(f"Wrote: {args.output}  rows={len(out)}  cols={len(out.columns)}")


if __name__ == "__main__":
    main()
