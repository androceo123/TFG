from __future__ import annotations

import math
import re
from collections import Counter
from urllib.parse import urlsplit, parse_qsl
from typing import Dict, Any, Optional


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


def _entropy(s: str) -> float:
    """Shannon entropy (bits) of a string. Returns 0.0 for empty or single-char strings."""
    if not s:
        return 0.0
    counts = Counter(s)
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _char_ratios(s: str) -> tuple[float, float]:
    """Returns (pct_digit, pct_alpha) for a string. Remaining fraction is special chars."""
    if not s:
        return 0.0, 0.0
    n = len(s)
    digits = sum(1 for c in s if c.isdigit())
    alpha = sum(1 for c in s if c.isalpha())
    return digits / n, alpha / n


# Bin boundaries for the 5-interval character distribution (Nico/Ralf OCS-WAF 2017).
# Frequencies are sorted descending; positions are grouped into 5 bins:
#   i0: position 0         (most frequent char)
#   i1: positions 1-2
#   i2: positions 3-5
#   i3: positions 6-9
#   i4: positions 10+      (tail)
_CHAR_DIST_BINS = [(0, 1), (1, 3), (3, 6), (6, 10), (10, None)]


def _char_dist_intervals(s: str) -> list[float]:
    """5-interval character frequency distribution (Nico/Ralf Tabla 2.3).

    Computes relative frequencies of each distinct character, sorts them
    descending, groups into 5 bins and sums each bin.  Returns a list of
    5 floats in [0, 1] that sum to 1.0 (or all zeros for empty input).
    Captures distribution *shape*: normal text has a smooth gradual decrease
    whereas attacks (buffer overflow, SQLi, XSS) show irregular profiles.
    """
    if not s:
        return [0.0] * 5
    counts = Counter(s)
    n = len(s)
    freqs = sorted((c / n for c in counts.values()), reverse=True)
    result = []
    for lo, hi in _CHAR_DIST_BINS:
        chunk = freqs[lo:hi] if hi is not None else freqs[lo:]
        result.append(sum(chunk))
    return result


def extract_http_features(
    *,
    method: str,
    uri: str,
    headers: Optional[Dict[str, str]] = None,
    body: Optional[bytes] = None,
) -> Dict[str, Any]:
    """Compute HTTP-oriented features from (method, uri, headers, body)."""
    headers = headers or {}
    method_up = (method or "").upper().strip()

    parts = urlsplit(uri or "")
    path = parts.path or ""
    query = parts.query or ""

    uri_len = len(uri or "")
    path_depth = path.count("/")
    query_len = len(query)

    qsl = parse_qsl(query, keep_blank_values=True)
    n_query_params = len(qsl)
    max_param_value_len = max((len(v) for _, v in qsl), default=0)

    non_alnum = sum(1 for ch in (uri or "") if not ch.isalnum())
    uri_pct_non_alnum_ratio = _ratio(non_alnum, uri_len)

    enc_hits = len(_pct_enc_re.findall(uri or ""))
    encoded = _ratio(enc_hits * 3, uri_len)

    lowered = (uri or "").lower()
    suspicious_tokens_count = 0
    for tok in SUSPICIOUS_TOKENS:
        suspicious_tokens_count += lowered.count(tok.strip().lower())
    has_suspicious_tokens = 1 if suspicious_tokens_count > 0 else 0

    # --- Features de entropia y distribucion de caracteres ---
    # Inspirados en el enfoque de Nico/Ralf (OCS-WAF 2017), que analizaban
    # la distribucion de caracteres y la entropia de cada valor de parametro.
    # Aqui se computan versiones globales compatibles con un modelo unico.

    uri_entropy = _entropy(uri or "")
    query_entropy = _entropy(query)

    # Entropia y distribucion por parametro (application-independent: se agrega
    # estadisticamente sobre todos los valores, sin usar nombres de parametros).
    param_value_entropies = [_entropy(v) for _, v in qsl] if qsl else [0.0]
    max_param_value_entropy = max(param_value_entropies)
    mean_param_value_entropy = sum(param_value_entropies) / len(param_value_entropies)
    std_param_value_entropy = (
        math.sqrt(
            sum((e - mean_param_value_entropy) ** 2 for e in param_value_entropies)
            / len(param_value_entropies)
        )
        if len(param_value_entropies) > 1
        else 0.0
    )

    # 5-interval character distribution del query string completo.
    query_char_dist = _char_dist_intervals(query)

    # 5-interval character distribution del valor de parametro con mayor entropia
    # (el mas anomalo). Application-independent: no usa nombre del parametro.
    if qsl:
        max_entropy_idx = param_value_entropies.index(max_param_value_entropy)
        max_entropy_param_value = qsl[max_entropy_idx][1]
    else:
        max_entropy_param_value = ""
    max_param_char_dist = _char_dist_intervals(max_entropy_param_value)

    query_pct_digit, query_pct_alpha = _char_ratios(query)

    # --- Features de cuerpo (body) ---
    body_suspicious_tokens_count = 0
    body_has_suspicious_tokens = 0
    body_encoded = 0.0
    body_entropy = 0.0
    body_pct_digit = 0.0
    body_pct_alpha = 0.0
    body_char_dist = [0.0] * 5
    max_body_param_value_entropy = 0.0
    mean_body_param_value_entropy = 0.0
    max_body_param_char_dist = [0.0] * 5

    if body:
        try:
            body_text = body.decode("utf-8", errors="ignore").lower()
        except Exception:
            body_text = str(body).lower()

        for tok in SUSPICIOUS_TOKENS:
            body_suspicious_tokens_count += body_text.count(tok.strip().lower())

        body_has_suspicious_tokens = 1 if body_suspicious_tokens_count > 0 else 0
        enc_hits_body = len(_pct_enc_re.findall(body_text))
        body_encoded = _ratio(enc_hits_body * 3, len(body_text) if body_text else 0)
        body_entropy = _entropy(body_text)
        body_pct_digit, body_pct_alpha = _char_ratios(body_text)
        body_char_dist = _char_dist_intervals(body_text)

        # Para el body tambien extraemos entropia y distribucion de cada parametro
        # (los POST de formulario tienen pares param=valor separados por &)
        body_qsl = parse_qsl(body_text, keep_blank_values=True)
        if body_qsl:
            body_param_entropies = [_entropy(v) for _, v in body_qsl]
            max_body_param_value_entropy = max(body_param_entropies)
            mean_body_param_value_entropy = sum(body_param_entropies) / len(body_param_entropies)
            max_body_entropy_idx = body_param_entropies.index(max_body_param_value_entropy)
            max_body_entropy_param_value = body_qsl[max_body_entropy_idx][1]
            max_body_param_char_dist = _char_dist_intervals(max_body_entropy_param_value)
        else:
            max_body_param_value_entropy = 0.0
            mean_body_param_value_entropy = 0.0
            max_body_param_char_dist = [0.0] * 5

    uncommon_method = 1 if method_up and method_up not in COMMON_METHODS else 0

    method_onehot = {f"method_{m}": 1 if method_up == m else 0 for m in KNOWN_METHODS}
    method_onehot["method_OTHER"] = 0 if method_up in KNOWN_METHODS else 1

    cl_raw = headers.get("content-length") or headers.get("Content-Length")
    try:
        req_content_length = int(cl_raw) if cl_raw is not None else 0
        if req_content_length < 0:
            req_content_length = 0
    except (TypeError, ValueError):
        req_content_length = 0

    body_len = len(body) if body is not None else 0
    if req_content_length == 0 and body_len > 0:
        req_content_length = body_len

    feats: Dict[str, Any] = {
        # --- Features estructurales originales (26) ---
        "uri_len": uri_len,
        "path_depth": path_depth,
        "query_len": query_len,
        "n_query_params": n_query_params,
        "max_param_value_len": max_param_value_len,
        "uri_pct_non_alnum_ratio": uri_pct_non_alnum_ratio,
        "encoded": encoded,
        "suspicious_tokens_count": suspicious_tokens_count,
        "has_suspicious_tokens": has_suspicious_tokens,
        "body_suspicious_tokens_count": body_suspicious_tokens_count,
        "body_has_suspicious_tokens": body_has_suspicious_tokens,
        "body_encoded": body_encoded,
        "uncommon_method": uncommon_method,
        "req_content_length": req_content_length,
        "body_len": body_len,
        # --- Features de entropia y distribucion de caracteres (v1, 8) ---
        "uri_entropy": uri_entropy,
        "query_entropy": query_entropy,
        "max_param_value_entropy": max_param_value_entropy,
        "query_pct_digit": query_pct_digit,
        "query_pct_alpha": query_pct_alpha,
        "body_entropy": body_entropy,
        "body_pct_digit": body_pct_digit,
        "body_pct_alpha": body_pct_alpha,
        # --- 5-interval char distribution (v2, inspirado en Nico/Ralf Tabla 2.3) ---
        # query string completo
        "query_char_dist_i0": query_char_dist[0],
        "query_char_dist_i1": query_char_dist[1],
        "query_char_dist_i2": query_char_dist[2],
        "query_char_dist_i3": query_char_dist[3],
        "query_char_dist_i4": query_char_dist[4],
        # valor de parametro query con mayor entropia (el mas anomalo)
        "max_param_char_dist_i0": max_param_char_dist[0],
        "max_param_char_dist_i1": max_param_char_dist[1],
        "max_param_char_dist_i2": max_param_char_dist[2],
        "max_param_char_dist_i3": max_param_char_dist[3],
        "max_param_char_dist_i4": max_param_char_dist[4],
        # body completo
        "body_char_dist_i0": body_char_dist[0],
        "body_char_dist_i1": body_char_dist[1],
        "body_char_dist_i2": body_char_dist[2],
        "body_char_dist_i3": body_char_dist[3],
        "body_char_dist_i4": body_char_dist[4],
        # valor de parametro body con mayor entropia
        "max_body_param_char_dist_i0": max_body_param_char_dist[0],
        "max_body_param_char_dist_i1": max_body_param_char_dist[1],
        "max_body_param_char_dist_i2": max_body_param_char_dist[2],
        "max_body_param_char_dist_i3": max_body_param_char_dist[3],
        "max_body_param_char_dist_i4": max_body_param_char_dist[4],
        # estadisticas de entropia entre parametros (no solo max)
        "mean_param_value_entropy": mean_param_value_entropy,
        "std_param_value_entropy": std_param_value_entropy,
        "max_body_param_value_entropy": max_body_param_value_entropy,
        "mean_body_param_value_entropy": mean_body_param_value_entropy,
    }
    feats.update(method_onehot)
    return feats
