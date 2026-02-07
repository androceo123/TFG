from __future__ import annotations

import argparse
import json
import pandas as pd
from tqdm import tqdm

from waf_ml.data.csic_loader import load_csic_txt, make_labels
from waf_ml.features.http_features import extract_http_features


def _to_bytes(v) -> bytes:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return b""
    s = str(v)
    if s in {"", "nan", "None"}:
        return b""
    return s.encode("utf-8", errors="ignore")


def _json_to_headers(s: str) -> dict:
    if not s:
        return {}
    try:
        return json.loads(s)
    except Exception:
        return {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to raw CSIC txt")
    ap.add_argument("--output", required=True, help="Path to output parquet/csv (features)")
    ap.add_argument("--label", required=True, choices=["normal", "anomalous"], help="Label for this file")

    ap.add_argument("--sample-n", type=int, default=0, help="If >0, only process first N requests (debug)")
    ap.add_argument("--out-format", default="parquet", choices=["parquet", "csv"])
    ap.add_argument("--keep-absolute-uri", action="store_true", help="Do NOT strip http://host:port from request line")
    ap.add_argument("--use-headers", action="store_true", help="Pass parsed headers to feature extractor")

    args = ap.parse_args()

    df = load_csic_txt(args.input, keep_absolute_uri=args.keep_absolute_uri, sample_n=args.sample_n)
    if df.empty:
        raise SystemExit(f"No requests parsed from: {args.input}")

    df["source_file"] = args.input
    df = make_labels(df, label=args.label)

    # Null-safe
    df["request_http_method"] = df["request_http_method"].fillna("")
    df["request_http_request"] = df["request_http_request"].fillna("")
    df["request_body"] = df["request_body"].fillna("")
    df["request_headers_json"] = df["request_headers_json"].fillna("")

    tqdm.pandas(desc="Extracting features (CSIC)")
    feats = df.progress_apply(
        lambda r: extract_http_features(
            method=str(r["request_http_method"]),
            uri=str(r["request_http_request"]),
            headers=_json_to_headers(str(r["request_headers_json"])) if args.use_headers else None,
            body=_to_bytes(r["request_body"]),
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
