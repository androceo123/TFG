from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List
import re
import pandas as pd


_LABEL_COL_RE = re.compile(r"^\d{1,3}\s+-\s+.+$")  # e.g. '66 - SQL Injection'
_ID_NAME_RE = re.compile(r"^(\d{1,3})\s+-\s+(.+)$")


@dataclass
class SRBHSchema:
    # Dataset columns (your header is tab-separated / TSV)
    sep: str = "\t"
    method_col: str = "request_http_method"
    uri_col: str = "request_http_request"
    body_col: str = "request_body"

    # Label columns pattern: '000 - Normal', '66 - SQL Injection', etc.
    normal_col: str = "000 - Normal"
    # Optional: provide exact label column names if auto-discovery fails
    label_cols: Optional[List[str]] = None


def load_srbh_csv(path: str, schema: Optional[SRBHSchema] = None) -> pd.DataFrame:
    """Load SR-BH 2020 traffic dataset (usually TSV)."""
    schema = schema or SRBHSchema()
    df = pd.read_csv(path, sep=schema.sep, engine="python")

    missing = [c for c in [schema.method_col, schema.uri_col, schema.body_col] if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing expected columns: {missing}. "
            "Update SRBHSchema in src/waf_ml/data/srbh_loader.py to match your CSV headers."
        )
    return df


def _discover_label_cols(df: pd.DataFrame, schema: SRBHSchema) -> List[str]:
    if schema.label_cols:
        cols = [c for c in schema.label_cols if c in df.columns]
        if not cols:
            raise ValueError("schema.label_cols provided but none were found in the CSV.")
        return cols

    cols = [c for c in df.columns if _LABEL_COL_RE.match(str(c).strip())]
    if not cols:
        raise ValueError("Could not auto-discover label columns like '66 - SQL Injection'. Provide SRBHSchema.label_cols.")
    return cols


def _canonical_label(col_name: str) -> str:
    """Convert '66 - SQL Injection' -> 'CAPEC-66 SQL Injection'"""
    m = _ID_NAME_RE.match(col_name.strip())
    if not m:
        return col_name.strip()
    capec_id, name = m.group(1), m.group(2).strip()
    return f"CAPEC-{int(capec_id)} {name}"


def make_labels(df: pd.DataFrame, schema: Optional[SRBHSchema] = None) -> pd.DataFrame:
    """Create labels for 3 tasks: binary, multiclass, multilabel."""
    schema = schema or SRBHSchema()
    label_cols = _discover_label_cols(df, schema)

    if schema.normal_col not in df.columns:
        raise ValueError(f"Normal label column '{schema.normal_col}' not found. Discovered label cols: {label_cols}")

    attack_cols = [c for c in label_cols if c != schema.normal_col]

    out = df.copy()

    # make labels int(0/1)
    for c in label_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype(int)

    def row_labels(r) -> List[str]:
        labs = [c for c in attack_cols if int(r[c]) == 1]
        return [_canonical_label(c) for c in labs]

    out["label_multilabel"] = out.apply(row_labels, axis=1)

    # binary: normal only if normal_col==1 and no attack labels
    out["label_binary"] = out.apply(
        lambda r: 0 if (int(r[schema.normal_col]) == 1 and all(int(r[c]) == 0 for c in attack_cols)) else 1,
        axis=1,
    )

    # multiclass: choose first label deterministically
    out["label_multiclass"] = out["label_multilabel"].map(lambda labs: labs[0] if len(labs) else "NORMAL")
    return out
