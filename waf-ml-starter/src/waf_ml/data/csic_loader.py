from __future__ import annotations

import json
import re
from typing import Dict, Iterator, List, Tuple, Optional

import pandas as pd
from urllib.parse import urlsplit


_REQ_LINE_RE = re.compile(
    r"^(GET|POST|HEAD|PUT|DELETE|PATCH|OPTIONS|TRACE|CONNECT)\s+(.+?)\s+HTTP/([0-9.]+)\s*$",
    re.IGNORECASE,
)


def _normalize_uri(raw_uri: str, keep_absolute_uri: bool) -> str:
    """
    CSIC suele traer URIs absolutas:
      GET http://localhost:8080/tienda1/index.jsp HTTP/1.1

    Para quedar parecido al dataset ya procesado (Harvard/SRBH),
    por defecto se convierte a: /tienda1/index.jsp?...
    """
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


def _parse_headers(lines: List[str], start: int) -> Tuple[Dict[str, str], int]:
    headers: Dict[str, str] = {}
    i = start
    n = len(lines)

    while i < n:
        line = lines[i].rstrip("\r\n")
        if line.strip() == "":
            break

        if ":" in line:
            k, v = line.split(":", 1)
            headers[k.strip()] = v.strip()

        i += 1

    return headers, i


def iter_csic_requests(text: str, *, keep_absolute_uri: bool = False) -> Iterator[dict]:
    """
    Itera requests crudas del CSIC.
    Cada request tiene:
      - request line
      - headers
      - línea en blanco
      - body (opcional)
      - luego otra request line, etc.
    """
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    i = 0
    n = len(lines)

    while i < n:
        # Buscar próxima request line
        m = None
        while i < n:
            m = _REQ_LINE_RE.match(lines[i].strip())
            if m:
                break
            i += 1

        if not m or i >= n:
            break

        method = m.group(1).upper()
        raw_uri = m.group(2).strip()
        http_version = m.group(3).strip()

        uri = _normalize_uri(raw_uri, keep_absolute_uri=keep_absolute_uri)

        # Headers
        i += 1
        headers, i = _parse_headers(lines, i)

        # Consumir separador (línea en blanco) post-headers
        while i < n and lines[i].strip() == "":
            i += 1

        # Body: todo hasta la próxima request line (si existe)
        body_lines: List[str] = []
        while i < n:
            if _REQ_LINE_RE.match(lines[i].strip()):
                break
            body_lines.append(lines[i])
            i += 1

        body_text = "\n".join(body_lines).strip()

        yield {
            "request_http_method": method,
            "request_http_request": uri,
            "request_body": body_text,
            "http_version": http_version,
            "request_headers_json": json.dumps(headers, ensure_ascii=False),
        }

        # Consumir separadores entre requests
        while i < n and lines[i].strip() == "":
            i += 1


def load_csic_txt(path: str, *, keep_absolute_uri: bool = False, sample_n: int = 0) -> pd.DataFrame:
    """
    Carga un .txt de CSIC (normalTrafficTraining/Test o anomalousTrafficTest)
    y devuelve un DataFrame con:
      - request_http_method
      - request_http_request
      - request_body
      - http_version
      - request_headers_json
    """
    text = open(path, "r", encoding="utf-8", errors="ignore").read()
    rows = []
    for idx, req in enumerate(iter_csic_requests(text, keep_absolute_uri=keep_absolute_uri)):
        rows.append(req)
        if sample_n and (idx + 1) >= sample_n:
            break
    return pd.DataFrame(rows)


def make_labels(
    df: pd.DataFrame,
    *,
    label: Optional[str] = None,
    label_prefix: str = "CSIC-ANOMALOUS",
    label_type_col: str = "label_type_raw",
) -> pd.DataFrame:
    """
    Crea labels en el MISMO formato que tu pipeline (K=2 para CSIC 2010):
      - label_binary: 0 normal / 1 ataque/anómalo
      - label_multiclass: "NORMAL" o label_prefix (default "CSIC-ANOMALOUS")
      - label_multilabel: [] o [label_prefix]

    Soporta 2 modos:
      A) (legacy) pasar label="normal"|"anomalous" y lo aplica a todo el DF.
      B) pasar un DF con una columna label_type_col (por fila) para concatenar múltiples archivos.
    """
    out = df.copy()

    if label is not None:
        lab = (label or "").strip().lower()
        if lab not in {"normal", "anomalous"}:
            raise ValueError("label must be 'normal' or 'anomalous'")
        out[label_type_col] = lab

    if label_type_col not in out.columns:
        raise ValueError(
            f"Missing '{label_type_col}'. Provide make_labels(..., label='normal'|'anomalous') "
            f"or add a per-row '{label_type_col}' column before calling make_labels()."
        )

    def _is_normal(v) -> bool:
        return str(v or "").strip().lower() == "normal"

    is_normal = out[label_type_col].map(_is_normal)

    out["label_binary"] = (~is_normal).astype(int)
    out["label_multiclass"] = is_normal.map(lambda x: "NORMAL" if x else label_prefix)
    out["label_multilabel"] = is_normal.map(lambda x: [] if x else [label_prefix])

    return out