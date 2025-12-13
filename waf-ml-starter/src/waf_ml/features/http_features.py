\
from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urlsplit, parse_qsl
from typing import Dict, Any, Optional, Tuple, List


SUSPICIOUS_TOKENS = [
    "../",
    "<script",
    "onerror=",
    "onload=",
    "'",
    '"',
    "--",
    ";",
    " union ",
    " select ",
    "sleep(",
    " or 1=1",
    "benchmark(",
    "information_schema",
]

COMMON_METHODS = {"GET", "POST", "HEAD"}
KNOWN_METHODS = ["GET", "POST", "HEAD", "PUT", "DELETE", "PATCH", "OPTIONS", "TRACE", "CONNECT"]


_pct_enc_re = re.compile(r"%[0-9a-fA-F]{2}")


def _ratio(n: int, d: int) -> float:
    return float(n) / float(d) if d else 0.0


def extract_http_features(
    *,
    method: str,
    uri: str,
    headers: Optional[Dict[str, str]] = None,
    body: Optional[bytes] = None,
) -> Dict[str, Any]:
    \"\"\"Compute the proposal's HTTP-oriented features from (method, uri, headers, body).

    Expected to be used both in offline dataset processing and in online WAF inference.
    \"\"\"
    headers = headers or {}
    method_up = (method or "").upper().strip()

    # --- URI decomposition ---
    parts = urlsplit(uri or "")
    path = parts.path or ""
    query = parts.query or ""

    uri_len = len(uri or "")
    path_depth = path.count("/")  # includes leading slash as 1; that's OK as long as consistent
    query_len = len(query)

    qsl = parse_qsl(query, keep_blank_values=True)
    n_query_params = len(qsl)
    max_param_value_len = max((len(v) for _, v in qsl), default=0)

    # "non-alnum ratio" (approx): how "weird" the URI looks
    non_alnum = sum(1 for ch in (uri or "") if not ch.isalnum())
    uri_pct_non_alnum_ratio = _ratio(non_alnum, uri_len)

    # "%xx" encoding density
    enc_hits = len(_pct_enc_re.findall(uri or ""))
    encoded = _ratio(enc_hits * 3, uri_len)  # each match is 3 chars (%xx)

    # suspicious token counts
    lowered = (uri or "").lower()
    suspicious_tokens_count = 0
    for tok in SUSPICIOUS_TOKENS:
        suspicious_tokens_count += lowered.count(tok.strip().lower())

    has_suspicious_tokens = 1 if suspicious_tokens_count > 0 else 0

    # uncommon method flag + one-hot methods
    uncommon_method = 1 if method_up and method_up not in COMMON_METHODS else 0

    method_onehot = {f"method_{m}": 1 if method_up == m else 0 for m in KNOWN_METHODS}
    method_onehot["method_OTHER"] = 0 if method_up in KNOWN_METHODS else 1

    # Content-Length (0 if not present / invalid)
    cl_raw = headers.get("content-length") or headers.get("Content-Length")
    try:
        req_content_length = int(cl_raw) if cl_raw is not None else 0
        if req_content_length < 0:
            req_content_length = 0
    except (TypeError, ValueError):
        req_content_length = 0

    # optional: body length (not in proposal, but often helpful)
    body_len = len(body) if body is not None else 0

    feats: Dict[str, Any] = {
        "uri_len": uri_len,
        "path_depth": path_depth,
        "query_len": query_len,
        "n_query_params": n_query_params,
        "max_param_value_len": max_param_value_len,
        "uri_pct_non_alnum_ratio": uri_pct_non_alnum_ratio,
        "encoded": encoded,
        "suspicious_tokens_count": suspicious_tokens_count,
        "has_suspicious_tokens": has_suspicious_tokens,
        "uncommon_method": uncommon_method,
        "req_content_length": req_content_length,
        "body_len": body_len,
    }
    feats.update(method_onehot)
    return feats
