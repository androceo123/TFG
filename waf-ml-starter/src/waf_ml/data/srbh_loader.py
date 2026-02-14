from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
import re
import pandas as pd


_LABEL_COL_RE = re.compile(r"^\d{1,3}\s+-\s+.+$")  # e.g. '66 - SQL Injection'
_ID_NAME_RE = re.compile(r"^(\d{1,3})\s+-\s+(.+)$")
_CANON_CAPEC_RE = re.compile(r"\bCAPEC-(\d{1,3})\b")


# -----------------------------------------------------------------------------
# Severity policy (SRBH)
#
# This is a *policy*, not a scientific "truth".
# Higher number = more severe.
#
# You can override this at runtime via SRBHSchema.severity_map.
# Keys supported:
#   - "CAPEC-66" (recommended)
#   - "66" (also supported)
#   - full canonical label "CAPEC-66 SQL Injection" (also supported)
# -----------------------------------------------------------------------------
DEFAULT_SEVERITY_MAP: Dict[str, int] = {
    # 000 - Normal (handled separately as "NORMAL")

    # Highest severity (typical direct execution / injection)
    "CAPEC-242": 10,  # 242 - Code Injection
    "CAPEC-88": 10,   # 88 - OS Command Injection
    "CAPEC-248": 10,  # 248 - Command Injection

    # Very high severity
    "CAPEC-66": 9,    # 66 - SQL Injection
    "CAPEC-33": 9,    # 33 - HTTP Request Smuggling

    # High / medium-high
    "CAPEC-126": 8,   # 126 - Path Traversal
    "CAPEC-34": 7,    # 34 - HTTP Response Splitting

    # Medium
    "CAPEC-272": 6,   # 272 - Protocol Manipulation
    "CAPEC-274": 6,   # 274 - HTTP Verb Tampering
    "CAPEC-194": 6,   # 194 - Fake the Source of Data
    "CAPEC-153": 6,   # 153 - Input Data Manipulation

    # Lower
    "CAPEC-16": 4,    # 16 - Dictionary-based Password Attack
    "CAPEC-310": 2,   # 310 - Scanning for Vulnerable Software
}


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

    # Multiclass label policy:
    # - "first": keep current behavior (first label in label_multilabel)
    # - "severity": pick the most severe label among label_multilabel
    multiclass_strategy: str = "severity"

    # Optional override severity map (if None, DEFAULT_SEVERITY_MAP is used)
    severity_map: Optional[Dict[str, int]] = None

    # If True, keep the original "first label" multiclass in a separate column for audit
    keep_first_for_audit: bool = True


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


def _capec_id_from_canonical(label: str) -> Optional[str]:
    m = _CANON_CAPEC_RE.search(label or "")
    if not m:
        return None
    return f"CAPEC-{int(m.group(1))}"


def _severity_of(label: str, sev_map: Dict[str, int]) -> int:
    """
    Lookup severity by:
      1) exact canonical label
      2) CAPEC-id key like "CAPEC-66"
      3) numeric id key like "66"
      4) default 0
    """
    if not label:
        return 0

    if label in sev_map:
        return int(sev_map[label])

    capec = _capec_id_from_canonical(label)
    if capec and capec in sev_map:
        return int(sev_map[capec])

    if capec:
        num = capec.replace("CAPEC-", "")
        if num in sev_map:
            return int(sev_map[num])

    return 0


def _choose_primary_by_severity(labels: List[str], sev_map: Dict[str, int]) -> Tuple[str, int]:
    """
    Choose label with max severity; ties broken by original order (stable).
    Returns (label, severity).
    """
    if not labels:
        return ("NORMAL", 0)

    best_label = labels[0]
    best_sev = _severity_of(best_label, sev_map)

    for lab in labels[1:]:
        s = _severity_of(lab, sev_map)
        if s > best_sev:
            best_label, best_sev = lab, s

    return best_label, best_sev


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

    # Keep current "first label" behavior for audit/debug if requested
    if schema.keep_first_for_audit:
        out["label_multiclass_first"] = out["label_multilabel"].map(lambda labs: labs[0] if len(labs) else "NORMAL")

    # multiclass strategy
    strategy = (schema.multiclass_strategy or "first").strip().lower()
    sev_map = schema.severity_map if schema.severity_map is not None else DEFAULT_SEVERITY_MAP

    if strategy == "severity":
        picked = out["label_multilabel"].map(lambda labs: _choose_primary_by_severity(labs, sev_map))
        out["label_multiclass"] = picked.map(lambda t: t[0])
        out["label_primary_severity"] = picked.map(lambda t: t[1])
    else:
        # multiclass: choose first label deterministically (existing behavior)
        out["label_multiclass"] = out["label_multilabel"].map(lambda labs: labs[0] if len(labs) else "NORMAL")

    return out
