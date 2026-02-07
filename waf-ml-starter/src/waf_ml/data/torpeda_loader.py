from __future__ import annotations

import json
import re
from urllib.parse import urlsplit
import xml.etree.ElementTree as ET

import pandas as pd


_HTTP_PROTO_RE = re.compile(r"^HTTP/(\d+(?:\.\d+)?)$", re.IGNORECASE)


def _normalize_uri(raw_uri: str, keep_absolute_uri: bool) -> str:
    """Strip absolute URIs (http(s)://host:port/..) to path?query unless keep_absolute_uri=True."""
    raw_uri = (raw_uri or "").strip()
    if keep_absolute_uri:
        return raw_uri

    try:
        parts = urlsplit(raw_uri)
        if parts.scheme and parts.netloc:
            path = parts.path or ""
            query = parts.query or ""
            return f"{path}?{query}" if query else path
    except Exception:
        pass

    return raw_uri


def _combine_path_query(path: str, query: str) -> str:
    p = (path or "").strip()
    q = (query or "").strip()
    if not q:
        return p
    if q.startswith("?"):
        q = q[1:]
    return f"{p}?{q}" if p else f"?{q}"


def _parse_headers_cdata(headers_text: str) -> dict:
    """Parse a CDATA multiline header block into a {name: value} dict."""
    headers: dict = {}
    for line in (headers_text or "").splitlines():
        line = line.strip("\r\n")
        if not line.strip():
            continue
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        headers[k.strip()] = v.strip()
    return headers


def load_torpeda_xml(path: str, *, keep_absolute_uri: bool = False, sample_n: int = 0) -> pd.DataFrame:
    """Load a Torpeda/CSIC-style XML (allNormals*, allAnomalies*, allAttacks*)."""
    raw = open(path, "rb").read()
    text = raw.decode("utf-8", errors="ignore")

    root = ET.fromstring(text)
    dataset_author = root.attrib.get("author", "")
    dataset_name = root.attrib.get("name", "")

    rows = []
    for idx, sample in enumerate(root.findall(".//sample")):
        req = sample.find("request")
        if req is None:
            continue

        sample_id = sample.attrib.get("id", "")
        method = (req.findtext("method") or "").strip().upper()

        protocol = (req.findtext("protocol") or "").strip()
        m = _HTTP_PROTO_RE.match(protocol)
        http_version = m.group(1) if m else (protocol.replace("HTTP/", "").strip() if protocol else "")

        path_text = req.findtext("path") or ""
        query_text = req.findtext("query") or ""
        uri = _combine_path_query(path_text, query_text)
        uri = _normalize_uri(uri, keep_absolute_uri=keep_absolute_uri)

        headers_text = req.findtext("headers") or ""
        headers = _parse_headers_cdata(headers_text)

        body_text = (req.findtext("body") or "").strip()

        lab = sample.find("label")
        label_type = ((lab.findtext("type") if lab is not None else "") or "").strip()
        label_attack = ((lab.findtext("attack") if lab is not None else "") or "").strip()

        rows.append(
            {
                "sample_id": sample_id,
                "request_http_method": method,
                "request_http_request": uri,
                "request_body": body_text,
                "http_version": http_version,
                "request_headers_json": json.dumps(headers, ensure_ascii=False),
                "label_type_raw": label_type,
                "label_attack_raw": label_attack,
                "dataset_author": dataset_author,
                "dataset_name": dataset_name,
            }
        )

        if sample_n and (idx + 1) >= sample_n:
            break

    return pd.DataFrame(rows)


def make_labels(
    df: pd.DataFrame,
    *,
    label_prefix: str = "CSIC-ANOMALOUS",
    label_type_col: str = "label_type_raw",
) -> pd.DataFrame:
    """Create labels in the same schema as your Harvard/CSIC pipeline."""
    if label_type_col not in df.columns:
        raise ValueError(f"Missing '{label_type_col}' column. Columns: {list(df.columns)}")

    out = df.copy()

    def _is_normal(v) -> bool:
        return str(v or "").strip().lower() == "normal"

    is_normal = out[label_type_col].map(_is_normal)

    # Treat 'attack' and 'anomalous' the same (anomaly) for OCSVM
    out["label_binary"] = (~is_normal).astype(int)
    out["label_multiclass"] = is_normal.map(lambda x: "NORMAL" if x else label_prefix)
    out["label_multilabel"] = is_normal.map(lambda x: [] if x else [label_prefix])

    return out
