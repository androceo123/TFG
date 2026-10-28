from __future__ import annotations

import argparse
import json
import os
import glob

import pandas as pd
from tqdm import tqdm

from waf_ml.data.torpeda_loader import load_torpeda_xml, make_labels
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


def _collect_inputs(items: list[str]) -> list[str]:
    paths: list[str] = []
    for it in items:
        it = (it or "").strip()
        if not it:
            continue

        if os.path.isdir(it):
            paths.extend(sorted(glob.glob(os.path.join(it, "**", "*.xml"), recursive=True)))
        else:
            # supports globs like data/raw/torpeda/**/*.xml
            paths.extend(sorted(glob.glob(it)))

    # de-dup, keep order
    seen = set()
    out = []
    for p in paths:
        if p not in seen:
            out.append(p)
            seen.add(p)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="One or more XML files, globs, or directories (will load **/*.xml)",
    )
    ap.add_argument("--output", required=True, help="Path to output parquet/csv (features)")

    ap.add_argument("--sample-n", type=int, default=0, help="If >0, only process first N samples PER FILE (debug)")
    ap.add_argument("--out-format", default="parquet", choices=["parquet", "csv"])

    ap.add_argument("--keep-absolute-uri", action="store_true", help="Do NOT strip http://host:port (if present)")
    ap.add_argument("--use-headers", action="store_true", help="Pass parsed headers to feature extractor")

    # Prefix para labels TorpEda:
    #   NORMAL / {PREFIX}-ANOMALOUS / {PREFIX}-{ATTACK_NAME}
    ap.add_argument(
        "--label-prefix",
        default="TORPEDA",
        help="Prefix used to build TorpEda label_multiclass values (e.g., TORPEDA).",
    )

    args = ap.parse_args()

    inputs = _collect_inputs(args.inputs)
    if not inputs:
        raise SystemExit("No XML inputs found. Check your --inputs paths/globs.")

    dfs = []
    for p in tqdm(inputs, desc="Loading XML"):
        df = load_torpeda_xml(p, keep_absolute_uri=args.keep_absolute_uri, sample_n=args.sample_n)
        if df is None or df.empty:
            continue
        df["source_file"] = p
        dfs.append(df)

    if not dfs:
        raise SystemExit("No samples parsed from inputs.")

    df = pd.concat(dfs, ignore_index=True)

    # Create: label_binary, label_multiclass, label_multilabel (TorpEda nativo)
    df = make_labels(df, label_prefix=args.label_prefix)

    # Null-safe
    df["request_http_method"] = df["request_http_method"].fillna("")
    df["request_http_request"] = df["request_http_request"].fillna("")
    df["request_body"] = df["request_body"].fillna("")
    df["request_headers_json"] = df["request_headers_json"].fillna("")

    tqdm.pandas(desc="Extracting features (TorpEda)")
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