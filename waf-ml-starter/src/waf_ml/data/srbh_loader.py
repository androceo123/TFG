from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
import re
import csv
import gzip
import pandas as pd
from pandas.errors import ParserError


_LABEL_COL_RE = re.compile(r"^\d{1,3}\s+-\s+.+$")  # e.g. '66 - SQL Injection'
_ID_NAME_RE = re.compile(r"^(\d{1,3})\s+-\s+(.+)$")
_CANON_CAPEC_RE = re.compile(r"\bCAPEC-(\d{1,3})\b")


DEFAULT_SEVERITY_MAP: Dict[str, int] = {
    "CAPEC-242": 10,  # 242 - Code Injection
    "CAPEC-88": 10,   # 88 - OS Command Injection
    "CAPEC-248": 10,  # 248 - Command Injection
    "CAPEC-66": 9,    # 66 - SQL Injection
    "CAPEC-33": 9,    # 33 - HTTP Request Smuggling
    "CAPEC-126": 8,   # 126 - Path Traversal
    "CAPEC-34": 7,    # 34 - HTTP Response Splitting
    "CAPEC-272": 6,   # 272 - Protocol Manipulation
    "CAPEC-274": 6,   # 274 - HTTP Verb Tampering
    "CAPEC-194": 6,   # 194 - Fake the Source of Data
    "CAPEC-153": 6,   # 153 - Input Data Manipulation
    "CAPEC-16": 4,    # 16 - Dictionary-based Password Attack
    "CAPEC-310": 2,   # 310 - Scanning for Vulnerable Software
}


@dataclass
class SRBHSchema:
    # Delimiter: can be "\t", ",", ";", "|" or "auto"
    sep: str = "auto"

    method_col: str = "request_http_method"
    uri_col: str = "request_http_request"
    body_col: str = "request_body"

    normal_col: str = "000 - Normal"
    label_cols: Optional[List[str]] = None

    multiclass_strategy: str = "severity"
    severity_map: Optional[Dict[str, int]] = None
    keep_first_for_audit: bool = True


def _read_sample_text(path: str, max_chars: int = 65536) -> str:
    if path.lower().endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8", errors="ignore", newline="") as f:
            return f.read(max_chars)
    with open(path, "rt", encoding="utf-8", errors="ignore", newline="") as f:
        return f.read(max_chars)


def _sniff_sep(path: str) -> str:
    sample = _read_sample_text(path)
    if not sample.strip():
        return ","
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", "\t", ";", "|"])
        return dialect.delimiter
    except Exception:
        header = sample.splitlines()[0] if sample.splitlines() else sample
        candidates = [",", "\t", ";", "|"]
        counts = {c: header.count(c) for c in candidates}
        best = max(counts, key=counts.get)
        return best if counts[best] > 0 else ","


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = []
    for c in df.columns:
        s = str(c)
        s = s.lstrip("\ufeff")  # remove UTF-8 BOM
        s = s.strip()
        cols.append(s)
    df.columns = cols
    return df


def load_srbh_csv(path: str, schema: Optional[SRBHSchema] = None) -> pd.DataFrame:
    """Load SR-BH / Harvard traffic dataset (CSV/TSV)."""
    schema = schema or SRBHSchema()

    sep = schema.sep
    if not sep or str(sep).lower() == "auto":
        sep = _sniff_sep(path)

    # 1) Try fast C-engine first (supports low_memory)
    try:
        df = pd.read_csv(
            path,
            sep=sep,
            engine="c",
            compression="infer",
            skipinitialspace=True,
            low_memory=False,
        )
    except UnicodeDecodeError:
        df = pd.read_csv(
            path,
            sep=sep,
            engine="c",
            compression="infer",
            encoding="latin-1",
            skipinitialspace=True,
            low_memory=False,
        )
    except (ParserError, ValueError):
        # 2) Fallback to python-engine for weird/malformed rows (NO low_memory here)
        try:
            df = pd.read_csv(
                path,
                sep=sep,
                engine="python",
                compression="infer",
                skipinitialspace=True,
            )
        except UnicodeDecodeError:
            df = pd.read_csv(
                path,
                sep=sep,
                engine="python",
                compression="infer",
                encoding="latin-1",
                skipinitialspace=True,
            )

    df = _normalize_columns(df)

    missing = [c for c in [schema.method_col, schema.uri_col, schema.body_col] if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing expected columns: {missing}. "
            f"Detected separator='{sep}'. "
            "If your headers differ, set --method-col/--uri-col/--body-col or adjust SRBHSchema."
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
    schema = schema or SRBHSchema()
    label_cols = _discover_label_cols(df, schema)

    if schema.normal_col not in df.columns:
        raise ValueError(f"Normal label column '{schema.normal_col}' not found. Discovered label cols: {label_cols}")

    attack_cols = [c for c in label_cols if c != schema.normal_col]

    out = df.copy()

    for c in label_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype(int)

    def row_labels(r) -> List[str]:
        labs = [c for c in attack_cols if int(r[c]) == 1]
        return [_canonical_label(c) for c in labs]

    out["label_multilabel"] = out.apply(row_labels, axis=1)

    out["label_binary"] = out.apply(
        lambda r: 0 if (int(r[schema.normal_col]) == 1 and all(int(r[c]) == 0 for c in attack_cols)) else 1,
        axis=1,
    )

    if schema.keep_first_for_audit:
        out["label_multiclass_first"] = out["label_multilabel"].map(lambda labs: labs[0] if len(labs) else "NORMAL")

    strategy = (schema.multiclass_strategy or "first").strip().lower()
    sev_map = schema.severity_map if schema.severity_map is not None else DEFAULT_SEVERITY_MAP

    if strategy == "severity":
        picked = out["label_multilabel"].map(lambda labs: _choose_primary_by_severity(labs, sev_map))
        out["label_multiclass"] = picked.map(lambda t: t[0])
        out["label_primary_severity"] = picked.map(lambda t: t[1])
    else:
        out["label_multiclass"] = out["label_multilabel"].map(lambda labs: labs[0] if len(labs) else "NORMAL")

    return out