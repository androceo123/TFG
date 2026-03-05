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


def _norm_attack_name(s: str) -> str:
    # Mantener “lo más nativo posible”, pero colapsar whitespace para evitar clases duplicadas por espacios.
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def make_labels(
    df: pd.DataFrame,
    *,
    label_prefix: str = "TORPEDA",
    label_type_col: str = "label_type_raw",
    label_attack_col: str = "label_attack_raw",
) -> pd.DataFrame:
    """
    Labels (alineado con libro/propuesta):
      - label_binary: 0 normal / 1 no-normal (anomalous o attack)
      - label_multiclass (nativo TorpEda):
          * NORMAL
          * {PREFIX}-ANOMALOUS
          * {PREFIX}-{ATTACK_NAME}   (8 tipos de ataque típicos)
      - label_multilabel: [] para NORMAL, si no-normal => [label_multiclass]
        (no es multilabel real, pero mantiene el mismo “shape” que el pipeline)
    """
    if label_type_col not in df.columns:
        raise ValueError(f"Missing '{label_type_col}' column. Columns: {list(df.columns)}")
    if label_attack_col not in df.columns:
        raise ValueError(f"Missing '{label_attack_col}' column. Columns: {list(df.columns)}")

    prefix = (label_prefix or "TORPEDA").strip()
    out = df.copy()

    typ = out[label_type_col].fillna("").astype(str).str.strip().str.lower()
    atk = out[label_attack_col].fillna("").astype(str).map(_norm_attack_name)

    is_normal = typ.eq("normal")
    is_anom = typ.eq("anomalous")
    is_attack = typ.eq("attack")

    # Binario (para OCSVM): todo lo no-normal = anomalía/ataque
    out["label_binary"] = (~is_normal).astype(int)

    # Multiclase (fiel a TorpEda: normal vs anomalous vs tipo de ataque)
    label_mc = pd.Series(index=out.index, dtype=object)

    label_mc[is_normal] = "NORMAL"
    label_mc[is_anom] = f"{prefix}-ANOMALOUS"

    # Para ataques: usar el nombre nativo de <attack> si existe
    def _attack_label(a: str) -> str:
        a = _norm_attack_name(a)
        return f"{prefix}-{a}" if a else f"{prefix}-ATTACK"

    label_mc[is_attack] = atk[is_attack].map(_attack_label)

    # Fallback: tipos inesperados => prefijo + tipo
    other = ~(is_normal | is_anom | is_attack)
    if other.any():
        def _other_label(t: str) -> str:
            t = (t or "").strip()
            t = re.sub(r"\s+", " ", t)
            return f"{prefix}-{t.upper()}" if t else f"{prefix}-UNKNOWN"

        label_mc[other] = typ[other].map(_other_label)

    out["label_multiclass"] = label_mc

    # Multilabel “compat” (lista)
    out["label_multilabel"] = [
        [] if lab == "NORMAL" else [lab]
        for lab in out["label_multiclass"].astype(str).tolist()
    ]

    return out