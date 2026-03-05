from __future__ import annotations

import argparse
import os
import glob
from typing import List, Tuple

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


def _has_glob(s: str) -> bool:
    return any(ch in s for ch in ["*", "?", "["])


def _collect_inputs(items: List[str]) -> Tuple[List[str], List[str]]:
    """
    Acepta:
      - archivos (path directo)
      - globs (ej: data/raw/harvard/*.csv)
      - directorios (escanea recursivo)
    Soporta: .csv .tsv .csv.gz .tsv.gz (case-insensitive)
    """
    found: List[str] = []
    missing: List[str] = []

    for it in items:
        it = (it or "").strip()
        if not it:
            continue

        if os.path.isdir(it):
            for root, _, files in os.walk(it):
                for fn in files:
                    low = fn.lower()
                    if low.endswith((".csv", ".tsv", ".csv.gz", ".tsv.gz")):
                        found.append(os.path.join(root, fn))
            continue

        # file/glob
        if _has_glob(it):
            matches = sorted(glob.glob(it))
            found.extend(matches)
        else:
            if os.path.exists(it):
                found.append(it)
            else:
                missing.append(it)

    # de-dup, keep order
    seen = set()
    out: List[str] = []
    for p in sorted(found):
        if p not in seen:
            out.append(p)
            seen.add(p)

    return out, missing


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--inputs",
        nargs="*",
        default=None,
        help="Uno o más paths/globs/directorios (lee recursivo .csv/.tsv/.gz).",
    )
    ap.add_argument("--input", default=None, help="(legacy) Path a un único archivo. Preferí --inputs.")
    ap.add_argument("--output", required=True, help="Path al output parquet/csv (final)")

    ap.add_argument("--method-col", default="request_http_method")
    ap.add_argument("--uri-col", default="request_http_request")
    ap.add_argument("--body-col", default="request_body")

    ap.add_argument(
        "--sep",
        default="auto",
        help="Delimitador de entrada: ',' '\\t' ';' '|' o 'auto' (recomendado).",
    )
    ap.add_argument(
        "--sample-n",
        type=int,
        default=0,
        help="Si >0, procesa solo las primeras N filas POR ARCHIVO (debug).",
    )
    ap.add_argument("--out-format", default="parquet", choices=["parquet", "csv"])
    args = ap.parse_args()

    raw_items: List[str] = []
    if args.inputs:
        raw_items.extend(args.inputs)
    if args.input:
        raw_items.append(args.input)

    if not raw_items:
        raise SystemExit("Tenés que pasar --inputs (recomendado) o --input (legacy).")

    inputs, missing = _collect_inputs(raw_items)
    if missing and not inputs:
        raise SystemExit(f"No existen estos paths: {missing}")
    if not inputs:
        raise SystemExit(
            "No se encontraron archivos SR-BH en --inputs/--input. "
            "Busco .csv/.tsv/.csv.gz/.tsv.gz dentro de directorios."
        )

    schema = SRBHSchema(
        sep=str(args.sep),
        method_col=args.method_col,
        uri_col=args.uri_col,
        body_col=args.body_col,
    )

    dfs = []
    for p in tqdm(inputs, desc="Loading SR-BH"):
        df_p = load_srbh_csv(p, schema=schema)
        if args.sample_n and args.sample_n > 0:
            df_p = df_p.head(args.sample_n).copy()
        df_p["source_file"] = p
        dfs.append(df_p)

    df = pd.concat(dfs, ignore_index=True)

    # labels: label_binary, label_multiclass, label_multilabel
    df = make_labels(df, schema=schema)

    # Null-safe
    df[schema.method_col] = df[schema.method_col].fillna("")
    df[schema.uri_col] = df[schema.uri_col].fillna("")
    df[schema.body_col] = df[schema.body_col].fillna("")

    tqdm.pandas(desc="Extracting features (SR-BH)")
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

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    if args.out_format == "parquet" or args.output.lower().endswith(".parquet"):
        out.to_parquet(args.output, index=False)
    else:
        out.to_csv(args.output, index=False)

    print(f"Wrote: {args.output}  rows={len(out)}  cols={len(out.columns)}")


if __name__ == "__main__":
    main()