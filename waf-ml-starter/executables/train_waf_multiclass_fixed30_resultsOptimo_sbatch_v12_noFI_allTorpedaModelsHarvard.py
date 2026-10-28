#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WAF-ML multiclass trainer para TorpEda o Harvard/SR-BH con el mismo set FIJO
compacto de 30 features intra-request. Harvard optimizado puede usar familias CAPEC agrupadas, filtrado de clases raras, umbral para NORMAL y candidatos two-stage, sin colapsar a HARVARD-ANOMALOUS. Versión v12: busca el mejor modelo Harvard probando también el menú TorpEda y sin feature importance.

Uso típico:
  --dataset torpeda --inputs data/raw/torpeda
  --dataset harvard --inputs data/raw/harvard

No usa TF-IDF, vocabularios, endpoint/app baselines, host, IP, sesión, histórico
ni fuentes externas. Sólo usa method/URI/headers/body de la request actual.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import glob
import html
import inspect
import json
import math
import os
import re
import subprocess
import sys
import time
import warnings
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import parse_qsl, unquote_plus, urlsplit

import joblib
import numpy as np
import pandas as pd
from pandas.errors import ParserError
from tqdm import tqdm

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.utils.class_weight import compute_sample_weight

try:
    from sklearn.inspection import permutation_importance
except Exception:  # pragma: no cover
    permutation_importance = None

RAW_METHOD_COL = "request_http_method"
RAW_URI_COL = "request_http_request"
RAW_BODY_COL = "request_body"
RAW_HEADERS_JSON_COL = "request_headers_json"
LABEL_COL = "label_multiclass"
DEFAULT_PROJECT_DIR = "/home_data/aroman/TFG/waf-ml-starter"

COMMON_ATTACKS = ["BufferOverflow", "CRLFi", "FormatString", "LDAPi", "SQLi", "SSI", "XPath", "XSS"]

# EXACTAMENTE 30 features fijas, numéricas e intra-request.
FIXED_FEATURES: List[str] = [
    "uri_len",
    "body_len",
    "query_len",
    "path_depth",
    "n_query_params",
    "max_param_value_len",
    "uri_encoded_ratio",
    "body_encoded_ratio",
    "invalid_percent_encoding_count",
    "double_encoding_count",
    "uri_non_alnum_ratio",
    "body_non_alnum_ratio",
    "uri_entropy",
    "body_entropy",
    "quote_count",
    "angle_bracket_count",
    "paren_count",
    "semicolon_count",
    "percent_count",
    "slash_backslash_count",
    "max_repeated_char_run",
    "max_token_len",
    "sql_token_count",
    "xss_token_count",
    "crlf_token_count",
    "ldap_token_count",
    "xpath_token_count",
    "ssi_token_count",
    "format_string_token_count",
    "buffer_overflow_token_count",
]

_HTTP_PROTO_RE = re.compile(r"^HTTP/(\d+(?:\.\d+)?)$", re.IGNORECASE)
_HEX = r"[0-9a-fA-F]"
_PCT_ENC_RE = re.compile(rf"%{_HEX}{{2}}")
_INVALID_PERCENT_RE = re.compile(rf"%(?!{_HEX}{{2}})")
_DOUBLE_ENC_RE = re.compile(rf"%25{_HEX}{{2}}", re.IGNORECASE)
_TOKEN_RE = re.compile(r"[A-Za-z0-9_%\\x\\u./:-]+")

_LABEL_COL_RE = re.compile(r"^\d{1,3}\s+-\s+.+$")  # SR-BH/Harvard, e.g. '66 - SQL Injection'
_ID_NAME_RE = re.compile(r"^(\d{1,3})\s+-\s+(.+)$")
_CANON_CAPEC_RE = re.compile(r"\bCAPEC-(\d{1,3})\b")


def _compile(patterns: Sequence[str]) -> List[re.Pattern[str]]:
    return [re.compile(p, re.IGNORECASE | re.DOTALL) for p in patterns]


PATTERNS: Dict[str, List[re.Pattern[str]]] = {
    "sql": _compile([
        r"\bunion(?:\s+all)?\s+select\b", r"\bselect\b", r"\binsert\b", r"\bupdate\b", r"\bdelete\b",
        r"\bdrop\b", r"\bfrom\b", r"\bwhere\b", r"\binformation_schema\b", r"\bgroup\s+by\b",
        r"\border\s+by\b", r"\bsleep\s*\(", r"\bbenchmark\s*\(", r"\bload_file\s*\(",
        r"\binto\s+outfile\b", r"\bwaitfor\s+delay\b", r"\bxp_cmdshell\b", r"--", r"#", r"/\*", r"\*/",
        r"\bor\s+['\"]?\d+['\"]?\s*=\s*['\"]?\d+['\"]?", r"\band\s+['\"]?\d+['\"]?\s*=\s*['\"]?\d+['\"]?",
        r"\b1\s*=\s*1\b", r"\bconcat\s*\(", r"\bchar\s*\(", r"\bsubstr(?:ing)?\s*\(", r"@@version",
    ]),
    "xss": _compile([
        r"<\s*/?\s*script\b", r"<\s*(?:img|svg|iframe|body|input|meta|object|embed|link|style)\b",
        r"\b(?:onerror|onload|onclick|onmouseover|onfocus|onblur|onreadystatechange)\s*=", r"javascript\s*:",
        r"vbscript\s*:", r"\bdocument\s*\.\s*cookie\b", r"\bdocument\s*\.\s*location\b",
        r"\b(?:alert|prompt|confirm|eval|atob)\s*\(", r"fromcharcode", r"<\s*/?\s*[a-z][a-z0-9:-]*\b",
    ]),
    "crlf": _compile([
        r"%0d", r"%0a", r"%0d%0a", r"%250d", r"%250a", r"\\r", r"\\n", r"\r\n", r"\n", r"\r",
        r"(?:\r|\n|%0d|%0a|%250d|%250a)\s*(?:set-cookie|location|content-length|content-type|refresh|x-forwarded-for|host)\s*:",
    ]),
    "ldap": _compile([
        r"\(\s*[&|!]", r"\(\s*[a-z][a-z0-9_-]*\s*=\s*\*?[^)]*\)",
        r"\b(?:objectclass|samaccountname|distinguishedname|userpassword|uid|cn|sn|mail)\s*=",
        r"\*\)\s*\(", r"\|\s*\(", r"&\s*\(", r"ldap://", r"%28", r"%29",
    ]),
    "xpath": _compile([
        r"\b(?:ancestor|ancestor-or-self|attribute|child|descendant|descendant-or-self|following|following-sibling|parent|preceding|preceding-sibling|self)::",
        r"\b(?:node|text|comment|processing-instruction)\s*\(\s*\)",
        r"\b(?:count|contains|starts-with|string-length|substring|position|last|name|local-name)\s*\(",
        r"//[^\s?&=]+", r"@\s*[a-zA-Z_][\w:-]*", r"\bor\s+['\"]?\d+['\"]?\s*=\s*['\"]?\d+['\"]?",
    ]),
    "ssi": _compile([
        r"<!--\s*#\s*(?:exec|include|echo|config|fsize|flastmod|printenv)",
        r"#\s*(?:exec|include|echo|config|printenv)\b", r"\b(?:cmd|cgi|virtual|file)\s*=", r"\.shtml",
    ]),
    "format_string": _compile([
        r"%(?:[0-9]+\$)?[-+#0 ]*(?:\d+|\*)?(?:\.(?:\d+|\*))?(?:hh|h|ll|l|L|z|j|t)?[diuoxXfFeEgGaAcspn]",
        r"%n", r"%x", r"%p", r"%s", r"%08x", r"%hn", r"%hhn", r"\{\d+\}", r"\{\}",
    ]),
    "buffer_overflow": _compile([
        r"A{32,}", r"B{32,}", r"C{32,}", r"D{32,}", r"0{32,}", r"1{32,}", r"[A-Za-z0-9]{128,}",
        r"%u9090", r"%u0c0c", r"\\x90", r"\x90", r"nop\s+sled", r"shellcode",
    ]),
}

ATTACK_CANON_MAP = {
    "bufferoverflow": "BufferOverflow", "bufferoverflows": "BufferOverflow", "bufferoverflowattack": "BufferOverflow",
    "crlf": "CRLFi", "crlfi": "CRLFi", "crlfinjection": "CRLFi", "crlfattack": "CRLFi", "httpresponsesplitting": "CRLFi", "responsesplitting": "CRLFi",
    "formatstring": "FormatString", "formatstrings": "FormatString", "formatstringattack": "FormatString",
    "ldap": "LDAPi", "ldapi": "LDAPi", "ldapinjection": "LDAPi",
    "sql": "SQLi", "sqli": "SQLi", "sqlinjection": "SQLi", "sqlia": "SQLi",
    "ssi": "SSI", "serversideinclude": "SSI", "serversideincludes": "SSI",
    "xpath": "XPath", "xpathi": "XPath", "xpathinjection": "XPath",
    "xss": "XSS", "crosssitescripting": "XSS", "crosssitescript": "XSS",
}

DEFAULT_SRBH_SEVERITY_MAP: Dict[str, int] = {
    "CAPEC-242": 10,  # Code Injection
    "CAPEC-88": 10,   # OS Command Injection
    "CAPEC-248": 10,  # Command Injection
    "CAPEC-66": 9,    # SQL Injection
    "CAPEC-33": 9,    # HTTP Request Smuggling
    "CAPEC-126": 8,   # Path Traversal
    "CAPEC-34": 7,    # HTTP Response Splitting
    "CAPEC-272": 6,   # Protocol Manipulation
    "CAPEC-274": 6,   # HTTP Verb Tampering
    "CAPEC-194": 6,   # Fake the Source of Data
    "CAPEC-153": 6,   # Input Data Manipulation
    "CAPEC-16": 4,    # Dictionary-based Password Attack
    "CAPEC-310": 2,   # Scanning for Vulnerable Software
}

# Harvard/SR-BH: clases nativas/familias del dataset. Importante: no hay fallback
# genérico HARVARD-ANOMALOUS; cada CAPEC queda como clase a predecir.
HARVARD_CAPEC_FAMILY_MAP: Dict[str, str] = {
    "CAPEC-66": "SQLi",
    "CAPEC-34": "CRLFi",
    "CAPEC-242": "CodeInjection",
    "CAPEC-88": "OSCommandInjection",
    "CAPEC-248": "CommandInjection",
    "CAPEC-126": "PathTraversal",
    "CAPEC-33": "HTTPRequestSmuggling",
    "CAPEC-272": "ProtocolManipulation",
    "CAPEC-274": "HTTPVerbTampering",
    "CAPEC-194": "FakeSourceData",
    "CAPEC-153": "InputDataManipulation",
    "CAPEC-16": "DictionaryPasswordAttack",
    "CAPEC-310": "VulnerableSoftwareScanning",
}

# Harvard optimizado: mantiene clases explícitas, pero agrupa CAPECs que son
# semánticamente cercanos y que con features intra-request suelen confundirse.
# No existe HARVARD-ANOMALOUS: todo queda en una clase interpretable.
HARVARD_OPTIMIZED_FAMILY_MAP: Dict[str, str] = {
    "CAPEC-66": "SQLi",
    "CAPEC-34": "CRLFi",
    "CAPEC-242": "CommandInjection",
    "CAPEC-88": "CommandInjection",
    "CAPEC-248": "CommandInjection",
    "CAPEC-126": "PathTraversal",
    "CAPEC-33": "HTTPProtocolManipulation",
    "CAPEC-272": "HTTPProtocolManipulation",
    "CAPEC-274": "HTTPProtocolManipulation",
    "CAPEC-194": "DataManipulation",
    "CAPEC-153": "DataManipulation",
    "CAPEC-16": "PasswordAttack",
    "CAPEC-310": "VulnerabilityScanning",
}


@dataclass
class SRBHSchema:
    sep: str = "auto"
    method_col: str = RAW_METHOD_COL
    uri_col: str = RAW_URI_COL
    body_col: str = RAW_BODY_COL
    normal_col: str = "000 - Normal"
    label_cols: Optional[List[str]] = None
    multiclass_strategy: str = "severity"
    severity_map: Optional[Dict[str, int]] = None


def _safe_str(v: Any) -> str:
    if v is None:
        return ""
    try:
        if isinstance(v, float) and pd.isna(v):
            return ""
    except Exception:
        pass
    s = str(v)
    return "" if s in {"nan", "None"} else s


def _safe_name(s: str) -> str:
    s = re.sub(r"[^0-9a-zA-Z]+", "_", str(s)).strip("_")
    return s or "CLASS"


def _ratio(n: float, d: float) -> float:
    return float(n) / float(d) if d else 0.0


def _entropy(text: str) -> float:
    if not text:
        return 0.0
    c = Counter(text)
    total = float(len(text))
    return float(-sum((v / total) * math.log2(v / total) for v in c.values()))


def _decode_once(text: str) -> str:
    try:
        return html.unescape(unquote_plus(text))
    except Exception:
        try:
            return unquote_plus(text)
        except Exception:
            return text


def _decode_twice(text: str) -> str:
    return _decode_once(_decode_once(text))


def _count_patterns(text: str, patterns: Sequence[re.Pattern[str]]) -> int:
    return int(sum(len(p.findall(text or "")) for p in patterns))


def _max_repeated_char_run(text: str) -> int:
    if not text:
        return 0
    best = cur = 1
    prev = text[0]
    for ch in text[1:]:
        if ch == prev:
            cur += 1
            best = max(best, cur)
        else:
            prev = ch
            cur = 1
    return int(best)


def _parse_qs(query: str) -> List[Tuple[str, str]]:
    if not query:
        return []
    try:
        return [(str(k), str(v)) for k, v in parse_qsl(query, keep_blank_values=True, strict_parsing=False)]
    except Exception:
        return []


def _looks_form_like(body_text: str, content_type: str = "") -> bool:
    s = _safe_str(body_text).strip()
    if not s:
        return False
    if "application/x-www-form-urlencoded" in content_type.lower():
        return True
    if s.startswith(("{", "[", "<")):
        return False
    return "=" in s and ("&" in s or len(s.split("=", 1)[0]) <= 128)


def _json_loads_dict(v: Any) -> Dict[str, str]:
    s = _safe_str(v).strip()
    if not s:
        return {}
    try:
        obj = json.loads(s)
        if not isinstance(obj, dict):
            return {}
        return {str(k): str(val) for k, val in obj.items()}
    except Exception:
        return {}


def _content_type(headers: Dict[str, str]) -> str:
    for k, v in headers.items():
        if str(k).lower().strip() == "content-type":
            return str(v)
    return ""


def extract_fixed_features(method: str, uri: str, headers: Optional[Dict[str, str]] = None, body: Any = None) -> Dict[str, float]:
    """Devuelve exactamente las 30 features fijas en el orden FIXED_FEATURES."""
    del method  # se evita usar método para no capturar sesgo de dataset/app.

    uri_raw = _safe_str(uri)
    if isinstance(body, bytes):
        body_raw = body.decode("utf-8", errors="ignore")
        body_len = len(body)
    else:
        body_raw = _safe_str(body)
        body_len = len(body_raw.encode("utf-8", errors="ignore"))

    uri_dec1, uri_dec2 = _decode_once(uri_raw), _decode_twice(uri_raw)
    body_dec1, body_dec2 = _decode_once(body_raw), _decode_twice(body_raw)
    raw_combined = f"{uri_raw}\n{body_raw}"
    analysis_text = f"{raw_combined}\n{uri_dec1}\n{body_dec1}\n{uri_dec2}\n{body_dec2}"

    parts = urlsplit(uri_raw)
    path = parts.path or ""
    query = parts.query or (uri_raw.split("?", 1)[1] if "?" in uri_raw else "")
    q_params = _parse_qs(query)
    headers_d = headers or {}
    b_params = _parse_qs(body_raw) if _looks_form_like(body_raw, _content_type(headers_d)) else []
    values = [v for _, v in (q_params + b_params)]

    uri_len = len(uri_raw)
    body_text_len = len(body_raw)
    tokens = _TOKEN_RE.findall(analysis_text)

    feats = {
        "uri_len": float(uri_len),
        "body_len": float(body_len),
        "query_len": float(len(query)),
        "path_depth": float(len([p for p in path.split("/") if p])),
        "n_query_params": float(len(q_params)),
        "max_param_value_len": float(max((len(v) for v in values), default=0)),
        "uri_encoded_ratio": _ratio(len(_PCT_ENC_RE.findall(uri_raw)) * 3, uri_len),
        "body_encoded_ratio": _ratio(len(_PCT_ENC_RE.findall(body_raw)) * 3, body_text_len),
        "invalid_percent_encoding_count": float(len(_INVALID_PERCENT_RE.findall(raw_combined))),
        "double_encoding_count": float(len(_DOUBLE_ENC_RE.findall(raw_combined))),
        "uri_non_alnum_ratio": _ratio(sum(1 for ch in uri_raw if not ch.isalnum()), uri_len),
        "body_non_alnum_ratio": _ratio(sum(1 for ch in body_raw if not ch.isalnum()), body_text_len),
        "uri_entropy": _entropy(uri_raw),
        "body_entropy": _entropy(body_raw),
        "quote_count": float(analysis_text.count("'") + analysis_text.count('"')),
        "angle_bracket_count": float(analysis_text.count("<") + analysis_text.count(">")),
        "paren_count": float(analysis_text.count("(") + analysis_text.count(")")),
        "semicolon_count": float(analysis_text.count(";")),
        "percent_count": float(raw_combined.count("%")),
        "slash_backslash_count": float(analysis_text.count("/") + analysis_text.count("\\")),
        "max_repeated_char_run": float(_max_repeated_char_run(analysis_text)),
        "max_token_len": float(max((len(t) for t in tokens), default=0)),
        "sql_token_count": float(_count_patterns(analysis_text, PATTERNS["sql"])),
        "xss_token_count": float(_count_patterns(analysis_text, PATTERNS["xss"])),
        "crlf_token_count": float(_count_patterns(analysis_text, PATTERNS["crlf"])),
        "ldap_token_count": float(_count_patterns(analysis_text, PATTERNS["ldap"])),
        "xpath_token_count": float(_count_patterns(analysis_text, PATTERNS["xpath"])),
        "ssi_token_count": float(_count_patterns(analysis_text, PATTERNS["ssi"])),
        "format_string_token_count": float(_count_patterns(analysis_text, PATTERNS["format_string"])),
        "buffer_overflow_token_count": float(_count_patterns(analysis_text, PATTERNS["buffer_overflow"])),
    }
    return {f: float(feats[f]) for f in FIXED_FEATURES}


# ----------------------- Labels TorpEda / common ------------------------

def _canonical_attack_name(raw: str) -> str:
    s = re.sub(r"\s+", " ", _safe_str(raw).strip())
    key = re.sub(r"[^a-z0-9]+", "", s.lower())
    if key in ATTACK_CANON_MAP:
        return ATTACK_CANON_MAP[key]
    for needle, canon in [
        ("bufferoverflow", "BufferOverflow"), ("overflow", "BufferOverflow"),
        ("httpresponsesplitting", "CRLFi"), ("responsesplitting", "CRLFi"), ("crlf", "CRLFi"),
        ("formatstring", "FormatString"), ("ldap", "LDAPi"), ("sql", "SQLi"),
        ("serversideinclude", "SSI"), ("ssi", "SSI"), ("xpath", "XPath"),
        ("crosssitescript", "XSS"), ("xss", "XSS"),
    ]:
        if needle in key:
            return canon
    return s if s else "ATTACK"


def _make_prefixed_common_label(raw_label_or_attack: str, prefix: str, *, unknown_as_anomalous: bool = False) -> str:
    lab = _safe_str(raw_label_or_attack).strip()
    pref = _safe_str(prefix).strip() or "DATASET"
    if lab.upper() == "NORMAL":
        return "NORMAL"
    if lab.upper().endswith("-ANOMALOUS") or lab.upper() == "ANOMALOUS":
        return f"{pref}-ANOMALOUS"
    attack_part = lab.split("-", 1)[1] if "-" in lab else lab
    canon = _canonical_attack_name(attack_part)
    if canon in COMMON_ATTACKS:
        return f"{pref}-{canon}"
    return f"{pref}-ANOMALOUS" if unknown_as_anomalous else f"{pref}-{canon}"


def _make_torpeda_label(label_type: str, attack: str, prefix: str) -> str:
    typ = _safe_str(label_type).strip().lower()
    pref = _safe_str(prefix).strip() or "TORPEDA"
    if typ == "normal":
        return "NORMAL"
    if typ == "anomalous":
        return f"{pref}-ANOMALOUS"
    if typ == "attack":
        return f"{pref}-{_canonical_attack_name(attack)}"
    return f"{pref}-{typ.upper()}" if typ else f"{pref}-UNKNOWN"


# ----------------------- Loader TorpEda XML ------------------------

def _normalize_uri(raw_uri: str, keep_absolute_uri: bool) -> str:
    raw_uri = _safe_str(raw_uri).strip()
    if keep_absolute_uri:
        return raw_uri
    try:
        parts = urlsplit(raw_uri)
        if parts.scheme and parts.netloc:
            return f"{parts.path}?{parts.query}" if parts.query else (parts.path or "")
    except Exception:
        pass
    return raw_uri


def _combine_path_query(path: str, query: str) -> str:
    p, q = _safe_str(path).strip(), _safe_str(query).strip()
    if not q:
        return p
    q = q[1:] if q.startswith("?") else q
    return f"{p}?{q}" if p else f"?{q}"


def _parse_headers_cdata(headers_text: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for line in _safe_str(headers_text).splitlines():
        if not line.strip() or ":" not in line:
            continue
        k, v = line.split(":", 1)
        out[k.strip()] = v.strip()
    return out


def load_torpeda_xml(path: str, *, keep_absolute_uri: bool, sample_n: int, label_prefix: str) -> pd.DataFrame:
    text = open(path, "rb").read().decode("utf-8", errors="ignore")
    root = ET.fromstring(text)
    rows: List[Dict[str, Any]] = []
    for idx, sample in enumerate(root.findall(".//sample")):
        req = sample.find("request")
        if req is None:
            continue
        method = (req.findtext("method") or "").strip().upper()
        protocol = (req.findtext("protocol") or "").strip()
        m = _HTTP_PROTO_RE.match(protocol)
        http_version = m.group(1) if m else protocol.replace("HTTP/", "").strip()
        uri = _normalize_uri(_combine_path_query(req.findtext("path") or "", req.findtext("query") or ""), keep_absolute_uri)
        headers = _parse_headers_cdata(req.findtext("headers") or "")
        body_text = (req.findtext("body") or "").strip()
        lab = sample.find("label")
        label_type = ((lab.findtext("type") if lab is not None else "") or "").strip()
        label_attack = ((lab.findtext("attack") if lab is not None else "") or "").strip()
        label_mc = _make_torpeda_label(label_type, label_attack, label_prefix)
        rows.append({
            "sample_id": sample.attrib.get("id", ""),
            RAW_METHOD_COL: method,
            RAW_URI_COL: uri,
            RAW_BODY_COL: body_text,
            RAW_HEADERS_JSON_COL: json.dumps(headers, ensure_ascii=False),
            "http_version": http_version,
            "label_type_raw": label_type,
            "label_attack_raw": label_attack,
            LABEL_COL: label_mc,
            "label_binary": 0 if label_mc == "NORMAL" else 1,
            "source_file": path,
            "dataset_author": root.attrib.get("author", ""),
            "dataset_name": root.attrib.get("name", ""),
        })
        if sample_n and (idx + 1) >= sample_n:
            break
    return pd.DataFrame(rows)


# ----------------------- Loader Harvard/SR-BH CSV/TSV ------------------------

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
        s = str(c).lstrip("\ufeff").strip()
        cols.append(s)
    df.columns = cols
    return df


def load_srbh_csv(path: str, schema: SRBHSchema) -> pd.DataFrame:
    sep = schema.sep
    if not sep or str(sep).lower() == "auto":
        sep = _sniff_sep(path)
    try:
        df = pd.read_csv(path, sep=sep, engine="c", compression="infer", skipinitialspace=True, low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(path, sep=sep, engine="c", compression="infer", encoding="latin-1", skipinitialspace=True, low_memory=False)
    except (ParserError, ValueError):
        try:
            df = pd.read_csv(path, sep=sep, engine="python", compression="infer", skipinitialspace=True)
        except UnicodeDecodeError:
            df = pd.read_csv(path, sep=sep, engine="python", compression="infer", encoding="latin-1", skipinitialspace=True)
    df = _normalize_columns(df)
    missing = [c for c in [schema.method_col, schema.uri_col, schema.body_col] if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing expected columns: {missing}. Detected separator={sep!r}. "
            "Use --method-col/--uri-col/--body-col if headers differ."
        )
    return df


def _discover_srbh_label_cols(df: pd.DataFrame, schema: SRBHSchema) -> List[str]:
    if schema.label_cols:
        cols = [c for c in schema.label_cols if c in df.columns]
        if not cols:
            raise ValueError("--label-cols provided but none were found in CSV.")
        return cols
    cols = [c for c in df.columns if _LABEL_COL_RE.match(str(c).strip())]
    if not cols:
        raise ValueError("Could not auto-discover Harvard/SR-BH label columns like '66 - SQL Injection'.")
    return cols


def _canonical_capec_label(col_name: str) -> str:
    m = _ID_NAME_RE.match(col_name.strip())
    if not m:
        return col_name.strip()
    capec_id, name = m.group(1), m.group(2).strip()
    return f"CAPEC-{int(capec_id)} {name}"


def _capec_id_from_label(label: str) -> Optional[str]:
    m = _CANON_CAPEC_RE.search(label or "")
    if not m:
        return None
    return f"CAPEC-{int(m.group(1))}"


def _severity_of_capec_label(label: str, sev_map: Dict[str, int]) -> int:
    if not label:
        return 0
    if label in sev_map:
        return int(sev_map[label])
    capec = _capec_id_from_label(label)
    if capec and capec in sev_map:
        return int(sev_map[capec])
    if capec:
        num = capec.replace("CAPEC-", "")
        if num in sev_map:
            return int(sev_map[num])
    return 0


def _choose_primary_label(labels: List[str], strategy: str, sev_map: Dict[str, int]) -> Tuple[str, int]:
    if not labels:
        return ("NORMAL", 0)
    if strategy.strip().lower() != "severity":
        return (labels[0], _severity_of_capec_label(labels[0], sev_map))
    best = labels[0]
    best_s = _severity_of_capec_label(best, sev_map)
    for lab in labels[1:]:
        s = _severity_of_capec_label(lab, sev_map)
        if s > best_s:
            best, best_s = lab, s
    return best, best_s


def _camel_safe_attack_name(label: str) -> str:
    """Build a compact stable class suffix from a CAPEC/native Harvard label."""
    lab = _safe_str(label).strip()
    if not lab:
        return "UNKNOWN"
    # Strip optional CAPEC prefix but preserve the semantic name.
    lab = re.sub(r"^CAPEC-\d+\s+", "", lab).strip()
    lab = re.sub(r"^\d{1,3}\s+-\s+", "", lab).strip()
    words = re.findall(r"[A-Za-z0-9]+", lab)
    if not words:
        return "UNKNOWN"
    return "".join(w[:1].upper() + w[1:] for w in words)


def _map_harvard_native_to_family(label: str, prefix: str) -> str:
    """Map Harvard CAPEC/native labels to readable classes, without HARVARD-ANOMALOUS."""
    lab = _safe_str(label).strip()
    pref = _safe_str(prefix).strip() or "HARVARD"
    if not lab or lab == "NORMAL":
        return "NORMAL"
    low = lab.lower()
    capec = _capec_id_from_label(lab) or ""

    if capec in HARVARD_CAPEC_FAMILY_MAP:
        return f"{pref}-{HARVARD_CAPEC_FAMILY_MAP[capec]}"

    # Keyword fallback. Still no generic anomalous class: unknown labels become their own class suffix.
    if "sql injection" in low:
        return f"{pref}-SQLi"
    if "cross site scripting" in low or "cross-site scripting" in low or "xss" in low:
        return f"{pref}-XSS"
    if "ldap" in low:
        return f"{pref}-LDAPi"
    if "xpath" in low:
        return f"{pref}-XPath"
    if "format string" in low:
        return f"{pref}-FormatString"
    if "buffer overflow" in low or "overflow" in low:
        return f"{pref}-BufferOverflow"
    if "server side include" in low or "server-side include" in low or "ssi" in low:
        return f"{pref}-SSI"
    if "response splitting" in low or "crlf" in low:
        return f"{pref}-CRLFi"
    if "path traversal" in low or "file inclusion" in low:
        return f"{pref}-PathTraversal"
    if "request smuggling" in low:
        return f"{pref}-HTTPRequestSmuggling"
    if "command injection" in low:
        return f"{pref}-CommandInjection"
    if "code injection" in low:
        return f"{pref}-CodeInjection"
    if "verb tampering" in low:
        return f"{pref}-HTTPVerbTampering"
    if "protocol manipulation" in low:
        return f"{pref}-ProtocolManipulation"
    if "input data manipulation" in low:
        return f"{pref}-InputDataManipulation"
    if "scanning" in low:
        return f"{pref}-VulnerableSoftwareScanning"

    return f"{pref}-{_camel_safe_attack_name(lab)}"


def _map_harvard_native_to_optimized_family(label: str, prefix: str) -> str:
    """Harvard optimized labels: explicit merged CAPEC families, no HARVARD-ANOMALOUS."""
    lab = _safe_str(label).strip()
    pref = _safe_str(prefix).strip() or "HARVARD"
    if not lab or lab == "NORMAL":
        return "NORMAL"
    low = lab.lower()
    capec = _capec_id_from_label(lab) or ""

    if capec in HARVARD_OPTIMIZED_FAMILY_MAP:
        return f"{pref}-{HARVARD_OPTIMIZED_FAMILY_MAP[capec]}"

    # Robust keyword fallbacks. Unknowns still become their own explicit class.
    if "sql injection" in low:
        return f"{pref}-SQLi"
    if "response splitting" in low or "crlf" in low:
        return f"{pref}-CRLFi"
    if "request smuggling" in low or "protocol manipulation" in low or "verb tampering" in low:
        return f"{pref}-HTTPProtocolManipulation"
    if "command injection" in low or "code injection" in low:
        return f"{pref}-CommandInjection"
    if "path traversal" in low or "file inclusion" in low:
        return f"{pref}-PathTraversal"
    if "fake" in low and "source" in low:
        return f"{pref}-DataManipulation"
    if "input data manipulation" in low:
        return f"{pref}-DataManipulation"
    if "dictionary" in low or "password" in low:
        return f"{pref}-PasswordAttack"
    if "scanning" in low or "vulnerable software" in low:
        return f"{pref}-VulnerabilityScanning"
    if "cross site scripting" in low or "cross-site scripting" in low or "xss" in low:
        return f"{pref}-XSS"
    if "ldap" in low:
        return f"{pref}-LDAPi"
    if "xpath" in low:
        return f"{pref}-XPath"
    if "format string" in low:
        return f"{pref}-FormatString"
    if "buffer overflow" in low or "overflow" in low:
        return f"{pref}-BufferOverflow"
    if "server side include" in low or "server-side include" in low or "ssi" in low:
        return f"{pref}-SSI"

    return f"{pref}-{_camel_safe_attack_name(lab)}"


def _map_harvard_native_to_legacy_common_anomalous(label: str, prefix: str) -> str:
    """Old compatibility mode: maps known common attacks, collapses the rest to HARVARD-ANOMALOUS."""
    lab = _safe_str(label).strip()
    if not lab or lab == "NORMAL":
        return "NORMAL"
    low = lab.lower()
    capec = _capec_id_from_label(lab) or ""
    if capec == "CAPEC-66" or "sql injection" in low:
        return f"{prefix}-SQLi"
    if "cross site scripting" in low or "cross-site scripting" in low or "xss" in low:
        return f"{prefix}-XSS"
    if "ldap" in low:
        return f"{prefix}-LDAPi"
    if "xpath" in low:
        return f"{prefix}-XPath"
    if "format string" in low:
        return f"{prefix}-FormatString"
    if "buffer overflow" in low or "overflow" in low:
        return f"{prefix}-BufferOverflow"
    if "server side include" in low or "server-side include" in low or "ssi" in low:
        return f"{prefix}-SSI"
    if capec == "CAPEC-34" or "response splitting" in low or "crlf" in low:
        return f"{prefix}-CRLFi"
    return f"{prefix}-ANOMALOUS"


def make_srbh_labels(df: pd.DataFrame, schema: SRBHSchema, *, label_prefix: str, harvard_label_mode: str) -> pd.DataFrame:
    label_cols = _discover_srbh_label_cols(df, schema)
    if schema.normal_col not in df.columns:
        raise ValueError(f"Normal label column {schema.normal_col!r} not found. Discovered label cols: {label_cols}")
    attack_cols = [c for c in label_cols if c != schema.normal_col]
    out = df.copy()
    for c in label_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype(int)

    def row_labels(r) -> List[str]:
        labs = [c for c in attack_cols if int(r[c]) == 1]
        return [_canonical_capec_label(c) for c in labs]

    out["label_multilabel"] = out.apply(row_labels, axis=1)
    out["label_binary"] = out.apply(
        lambda r: 0 if (int(r[schema.normal_col]) == 1 and all(int(r[c]) == 0 for c in attack_cols)) else 1,
        axis=1,
    )
    sev_map = schema.severity_map if schema.severity_map is not None else DEFAULT_SRBH_SEVERITY_MAP
    picked = out["label_multilabel"].map(lambda labs: _choose_primary_label(labs, schema.multiclass_strategy, sev_map))
    out["label_multiclass_native"] = picked.map(lambda t: t[0])
    out["label_primary_severity"] = picked.map(lambda t: t[1])
    mode = (harvard_label_mode or "family").strip().lower()
    if mode in {"native", "capec", "srbh"}:
        # Keep the exact Harvard/CAPEC primary class, e.g. "CAPEC-66 SQL Injection".
        # No HARVARD-ANOMALOUS fallback.
        out[LABEL_COL] = out["label_multiclass_native"]
    elif mode in {"family", "mapped", "attack-family", "attack_families"}:
        # Compact readable Harvard classes, e.g. HARVARD-SQLi, HARVARD-PathTraversal,
        # HARVARD-HTTPRequestSmuggling. No HARVARD-ANOMALOUS fallback.
        pref = label_prefix or "HARVARD"
        out[LABEL_COL] = out["label_multiclass_native"].map(lambda x: _map_harvard_native_to_family(str(x), pref))
    elif mode in {"optimized-family", "optimized_family", "optimized", "coarse", "coarse-family", "coarse_family"}:
        # Optimized Harvard classes: keeps explicit classes, but merges CAPEC variants
        # that are difficult to separate with only the shared fixed intra-request features.
        pref = label_prefix or "HARVARD"
        out[LABEL_COL] = out["label_multiclass_native"].map(lambda x: _map_harvard_native_to_optimized_family(str(x), pref))
    elif mode in {"legacy-common-anomalous", "legacy_common_anomalous"}:
        pref = label_prefix or "HARVARD"
        out[LABEL_COL] = out["label_multiclass_native"].map(lambda x: _map_harvard_native_to_legacy_common_anomalous(str(x), pref))
    else:
        raise ValueError(
            f"Unsupported --harvard-label-mode={harvard_label_mode!r}; "
            "use family|optimized-family|native|legacy-common-anomalous"
        )
    return out


# ----------------------- Dataset loading ------------------------

def _has_glob(s: str) -> bool:
    return any(ch in s for ch in ["*", "?", "["])


def _collect_inputs(items: Sequence[str], *, dataset: str) -> List[str]:
    dataset = dataset.lower()
    exts = (".xml",) if dataset == "torpeda" else (".csv", ".tsv", ".csv.gz", ".tsv.gz")
    paths: List[str] = []
    for item in items:
        item = _safe_str(item).strip()
        if not item:
            continue
        if os.path.isdir(item):
            for root, _, files in os.walk(item):
                for fn in files:
                    low = fn.lower()
                    if any(low.endswith(ext) for ext in exts):
                        paths.append(os.path.join(root, fn))
        elif _has_glob(item):
            paths.extend(glob.glob(item, recursive=True))
        elif os.path.exists(item):
            paths.append(item)
        else:
            print(f"[WARN] Input path not found: {item}", flush=True)
    seen = set()
    out: List[str] = []
    for p in sorted(paths):
        if p not in seen:
            out.append(p)
            seen.add(p)
    return out


def _read_table(path: str) -> pd.DataFrame:
    p = path.lower()
    if p.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def load_dataset(args: argparse.Namespace) -> pd.DataFrame:
    dataset = args.dataset.lower()
    label_prefix = args.label_prefix
    if dataset == "torpeda":
        inputs = _collect_inputs(args.inputs or [], dataset="torpeda")
        if not inputs:
            raise SystemExit("No XML inputs found. Check --inputs/TORPEDA_RAW_DIR paths/globs.")
        print(f"[INFO] TorpEda XML files found: {len(inputs)}", flush=True)
        print(f"[INFO] First XML: {inputs[0]}", flush=True)
        if len(inputs) > 1:
            print(f"[INFO] Last XML:  {inputs[-1]}", flush=True)
        frames = []
        for p in tqdm(inputs, desc="Loading TorpEda XML", mininterval=5):
            d = load_torpeda_xml(p, keep_absolute_uri=args.keep_absolute_uri, sample_n=args.sample_n, label_prefix=label_prefix)
            if d is not None and not d.empty:
                frames.append(d)
        if not frames:
            raise SystemExit("No samples parsed from TorpEda XML inputs.")
        df = pd.concat(frames, ignore_index=True)
        print(f"[INFO] TorpEda parsed rows before filtering: {len(df)}", flush=True)
    elif dataset == "harvard":
        inputs = _collect_inputs(args.inputs or [], dataset="harvard")
        if not inputs:
            raise SystemExit("No Harvard/SR-BH CSV/TSV inputs found. Check --inputs/HARVARD_RAW_DIR paths/globs.")
        print(f"[INFO] Harvard/SR-BH files found: {len(inputs)}", flush=True)
        print(f"[INFO] First Harvard file: {inputs[0]}", flush=True)
        if len(inputs) > 1:
            print(f"[INFO] Last Harvard file:  {inputs[-1]}", flush=True)
        label_cols = [x.strip() for x in (args.label_cols or "").split(",") if x.strip()] or None
        schema = SRBHSchema(
            sep=args.sep,
            method_col=args.method_col,
            uri_col=args.uri_col,
            body_col=args.body_col,
            normal_col=args.normal_col,
            label_cols=label_cols,
            multiclass_strategy=args.harvard_multiclass_strategy,
            severity_map=DEFAULT_SRBH_SEVERITY_MAP,
        )
        frames = []
        for p in tqdm(inputs, desc="Loading Harvard/SR-BH", mininterval=5):
            d = load_srbh_csv(p, schema)
            if args.sample_n and args.sample_n > 0:
                d = d.head(args.sample_n).copy()
            d["source_file"] = p
            frames.append(d)
        df = pd.concat(frames, ignore_index=True)
        df = make_srbh_labels(df, schema, label_prefix=label_prefix, harvard_label_mode=args.harvard_label_mode)
        # Normalize raw column names to project standard.
        if schema.method_col != RAW_METHOD_COL:
            df[RAW_METHOD_COL] = df[schema.method_col]
        if schema.uri_col != RAW_URI_COL:
            df[RAW_URI_COL] = df[schema.uri_col]
        if schema.body_col != RAW_BODY_COL:
            df[RAW_BODY_COL] = df[schema.body_col]
        if RAW_HEADERS_JSON_COL not in df.columns:
            df[RAW_HEADERS_JSON_COL] = "{}"
        if "sample_id" not in df.columns:
            df["sample_id"] = np.arange(len(df)).astype(str)
        print(f"[INFO] Harvard/SR-BH rows before filtering: {len(df)}", flush=True)
    elif dataset == "table":
        if not args.table:
            raise SystemExit("--table is required with --dataset table")
        df = _read_table(args.table)
        label_col = args.label_col or LABEL_COL
        if label_col not in df.columns:
            raise SystemExit(f"Missing label column {label_col!r}. Columns: {list(df.columns)}")
        if label_col != LABEL_COL:
            df[LABEL_COL] = df[label_col].astype(str)
        for col in [RAW_METHOD_COL, RAW_URI_COL, RAW_BODY_COL]:
            if col not in df.columns:
                raise SystemExit(f"Missing raw request column {col!r}.")
        if RAW_HEADERS_JSON_COL not in df.columns:
            df[RAW_HEADERS_JSON_COL] = "{}"
        if "sample_id" not in df.columns:
            df["sample_id"] = np.arange(len(df)).astype(str)
        if "source_file" not in df.columns:
            df["source_file"] = args.table
    else:
        raise SystemExit(f"Unsupported dataset={args.dataset!r}")

    for col in [RAW_METHOD_COL, RAW_URI_COL, RAW_BODY_COL, RAW_HEADERS_JSON_COL, LABEL_COL]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("")
    df[LABEL_COL] = df[LABEL_COL].astype(str)

    if args.only_common_labels:
        allowed = {"NORMAL", f"{label_prefix}-ANOMALOUS", *[f"{label_prefix}-{a}" for a in COMMON_ATTACKS]}
        before = len(df)
        df = df[df[LABEL_COL].isin(allowed)].reset_index(drop=True)
        print(f"[INFO] only_common_labels: kept={len(df)} removed={before-len(df)}", flush=True)
        if df.empty:
            raise SystemExit("No rows left after --only-common-labels. Check labels/prefix/harvard-label-mode.")
    return df.reset_index(drop=True)


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    tqdm.pandas(desc="Extracting fixed30 features")
    series = df.progress_apply(
        lambda r: extract_fixed_features(
            method=r.get(RAW_METHOD_COL, ""),
            uri=r.get(RAW_URI_COL, ""),
            headers=_json_loads_dict(r.get(RAW_HEADERS_JSON_COL, "{}")),
            body=r.get(RAW_BODY_COL, ""),
        ),
        axis=1,
    )
    feats = pd.DataFrame(list(series), columns=FIXED_FEATURES).fillna(0).astype(np.float32)
    return pd.concat([df.reset_index(drop=True), feats], axis=1)


# ----------------------- Utilities ------------------------

def _json_default(obj: Any):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def _safe_float(v: Any) -> Optional[float]:
    try:
        x = float(v)
        return None if np.isnan(x) or np.isinf(x) else x
    except Exception:
        return None


def _save_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=_json_default)


def _save_processed_atomic(df: pd.DataFrame, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    tmp = path.with_name(f".{path.name}.tmp_{os.getpid()}{path.suffix or '.tmp'}")
    try:
        if suffix == ".parquet":
            df.to_parquet(tmp, index=False)
        elif suffix == ".csv":
            df.to_csv(tmp, index=False)
        else:
            raise ValueError(f"Unsupported processed output extension: {path}. Use .parquet or .csv")
        os.replace(tmp, path)
        return path
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass


def _can_stratify(y: Sequence[Any], test_size: float) -> bool:
    counts = pd.Series(y).value_counts(dropna=False)
    if counts.empty or counts.min() < 2:
        return False
    n_classes = len(counts)
    n_test = int(math.ceil(len(y) * test_size))
    return n_test >= n_classes and (len(y) - n_test) >= n_classes


def _split_indices(y: Sequence[Any], test_size: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    idx = np.arange(len(y))
    strat = y if _can_stratify(y, test_size) else None
    if strat is None:
        print("[WARN] Split no estratificado: hay clases raras o split chico.")
    tr, te = train_test_split(idx, test_size=test_size, random_state=seed, shuffle=True, stratify=strat)
    return np.asarray(tr), np.asarray(te)


def _has_nvidia_smi() -> bool:
    try:
        return subprocess.run(["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=4).returncode == 0
    except Exception:
        return False


def _gpu_visible_from_env() -> bool:
    env_names = ["CUDA_VISIBLE_DEVICES", "SLURM_JOB_GPUS", "SLURM_STEP_GPUS", "NVIDIA_VISIBLE_DEVICES"]
    bad_values = {"", "-1", "none", "void", "nodevfiles", "no_dev_files", "unset", "null"}
    for name in env_names:
        val = str(os.environ.get(name, "")).strip()
        if val and val.lower() not in bad_values:
            return True
    return False


def _gpu_visible_auto() -> bool:
    return _gpu_visible_from_env() or _has_nvidia_smi()


def _supports_fit_parameter(estimator: Any, param: str) -> bool:
    try:
        return param in inspect.signature(estimator.fit).parameters
    except Exception:
        return False


def _xgb_available() -> bool:
    try:
        import xgboost  # noqa: F401
        return True
    except Exception:
        return False


def _parse_xgb_major(version: str) -> int:
    try:
        return int(str(version).split(".", 1)[0])
    except Exception:
        return 2


def _xgb_candidate(
    name: str,
    *,
    n_classes: int,
    seed: int,
    n_jobs: int,
    gpu: bool,
    depth: int,
    n_estimators: int,
    lr: float,
    subsample: float,
    colsample: float,
    reg_lambda: float,
) -> Tuple[str, Any]:
    import xgboost as xgb
    from xgboost import XGBClassifier

    xgb_major = _parse_xgb_major(getattr(xgb, "__version__", "2"))
    kwargs: Dict[str, Any] = {
        "objective": "multi:softprob",
        "eval_metric": "mlogloss",
        "num_class": int(n_classes),
        "max_depth": int(depth),
        "n_estimators": int(n_estimators),
        "learning_rate": float(lr),
        "subsample": float(subsample),
        "colsample_bytree": float(colsample),
        "reg_lambda": float(reg_lambda),
        "random_state": int(seed),
        "n_jobs": int(n_jobs),
        "tree_method": "hist",
        "verbosity": 0,
    }
    if gpu:
        if xgb_major >= 2:
            kwargs["device"] = "cuda"
        else:
            kwargs["tree_method"] = "gpu_hist"
            kwargs["predictor"] = "gpu_predictor"
    else:
        if xgb_major >= 2:
            kwargs["device"] = "cpu"
    return name, XGBClassifier(**kwargs)


def _parse_csv_list_simple(s: Optional[str]) -> List[str]:
    if not s:
        return []
    return [x.strip() for x in str(s).split(",") if x.strip()]


def _make_sample_weight_vector(y: np.ndarray, *, mode: str, clip: float = 20.0) -> np.ndarray:
    mode_l = (mode or "balanced").strip().lower().replace("-", "_")
    y_arr = np.asarray(y)
    if mode_l in {"", "none", "off", "false", "no"}:
        return np.ones(len(y_arr), dtype=np.float32)
    base = compute_sample_weight(class_weight="balanced", y=y_arr).astype(np.float64)
    if mode_l in {"balanced", "full_balanced"}:
        w = base
    elif mode_l in {"sqrt", "sqrt_balanced", "balanced_sqrt"}:
        w = np.sqrt(base)
    elif mode_l in {"cuberoot", "cbrt", "cbrt_balanced"}:
        w = np.cbrt(base)
    elif mode_l in {"balanced_clipped", "clipped", "clip"}:
        w = np.minimum(base, float(clip))
    elif mode_l in {"sqrt_clipped", "sqrt_balanced_clipped"}:
        w = np.minimum(np.sqrt(base), float(clip))
    else:
        raise ValueError(f"Unsupported sample weight mode: {mode!r}")
    mean = float(np.mean(w)) if len(w) else 1.0
    if mean > 0:
        w = w / mean
    return w.astype(np.float32)


def _parse_threshold_grid(spec: Optional[str]) -> List[Optional[float]]:
    s = str(spec or "").strip().lower()
    if not s or s in {"off", "none", "false", "0"}:
        return []
    vals: List[float] = []
    if ":" in s:
        parts = [p.strip() for p in s.split(":")]
        if len(parts) != 3:
            raise ValueError("--normal-threshold-grid with ':' must be start:stop:step, e.g. 0.02:0.98:0.01")
        start, stop, step = map(float, parts)
        if step <= 0:
            raise ValueError("threshold step must be > 0")
        x = start
        # Include stop with small tolerance.
        while x <= stop + 1e-12:
            vals.append(round(float(x), 6))
            x += step
    else:
        vals = [float(x) for x in _parse_csv_list_simple(s)]
    vals = [v for v in vals if 0.0 <= v <= 1.0]
    # None = original model argmax baseline. Keep it as an option.
    out: List[Optional[float]] = [None]
    out.extend(sorted(set(vals)))
    return out


def _pred_indices_from_proba(proba: np.ndarray, *, normal_idx: Optional[int], normal_threshold: Optional[float]) -> np.ndarray:
    arr = np.asarray(proba, dtype=float)
    if arr.ndim != 2:
        return np.asarray([], dtype=int)
    if normal_idx is None or normal_threshold is None or not (0 <= int(normal_idx) < arr.shape[1]):
        return np.argmax(arr, axis=1).astype(int)
    nidx = int(normal_idx)
    pred = np.argmax(arr, axis=1).astype(int)
    normal_mask = arr[:, nidx] >= float(normal_threshold)
    if arr.shape[1] > 1:
        attack_arr = arr.copy()
        attack_arr[:, nidx] = -np.inf
        attack_pred = np.argmax(attack_arr, axis=1).astype(int)
        pred = np.where(normal_mask, nidx, attack_pred).astype(int)
    else:
        pred = np.full(arr.shape[0], nidx, dtype=int)
    return pred


def _predict_indices_with_threshold(estimator: Any, X: np.ndarray, *, n_classes: int, normal_idx: Optional[int], threshold: Optional[float]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    proba = _align_proba_to_all_classes(_predict_proba_safe(estimator, X), estimator, n_classes)
    if proba is not None and threshold is not None:
        return _pred_indices_from_proba(proba, normal_idx=normal_idx, normal_threshold=threshold), proba
    if proba is not None and isinstance(estimator, NormalThresholdWrapper):
        return estimator.predict(X), proba
    return np.asarray(estimator.predict(X), dtype=int), proba


class NormalThresholdWrapper(BaseEstimator, ClassifierMixin):
    """Post-procesa un modelo probabilístico: si P(NORMAL)>=threshold, predice NORMAL."""

    def __init__(self, estimator: Any, normal_class: int, threshold: float, n_classes: int):
        self.estimator = estimator
        self.normal_class = int(normal_class)
        self.threshold = float(threshold)
        self.n_classes = int(n_classes)
        self.classes_ = np.arange(self.n_classes, dtype=int)

    def fit(self, X, y=None):
        return self

    def predict_proba(self, X):
        return _align_proba_to_all_classes(_predict_proba_safe(self.estimator, X), self.estimator, self.n_classes)

    def predict(self, X):
        proba = self.predict_proba(X)
        if proba is None:
            return np.asarray(self.estimator.predict(X), dtype=int)
        return _pred_indices_from_proba(proba, normal_idx=self.normal_class, normal_threshold=self.threshold)


class TwoStageMulticlassClassifier(BaseEstimator, ClassifierMixin):
    """1) NORMAL vs ataque; 2) clase de ataque. Útil cuando multiclass directo sobrepredice ataques."""

    def __init__(
        self,
        binary_estimator: Any,
        attack_estimator: Any,
        normal_class: int,
        threshold: float = 0.5,
        binary_weight_mode: str = "sqrt_balanced",
        attack_weight_mode: str = "balanced_clipped",
        sample_weight_clip: float = 20.0,
    ):
        self.binary_estimator = binary_estimator
        self.attack_estimator = attack_estimator
        self.normal_class = int(normal_class)
        self.threshold = float(threshold)
        self.binary_weight_mode = binary_weight_mode
        self.attack_weight_mode = attack_weight_mode
        self.sample_weight_clip = float(sample_weight_clip)

    def fit(self, X, y, sample_weight=None):
        y_arr = np.asarray(y, dtype=int)
        self.classes_ = np.arange(int(np.max(y_arr)) + 1, dtype=int)
        self.normal_class_ = int(self.normal_class)
        attack_mask = y_arr != self.normal_class_
        if attack_mask.sum() == 0:
            raise ValueError("TwoStage needs at least one attack sample")
        if (~attack_mask).sum() == 0:
            raise ValueError("TwoStage needs at least one NORMAL sample")

        y_bin = attack_mask.astype(int)
        sw_bin = _make_sample_weight_vector(y_bin, mode=self.binary_weight_mode, clip=self.sample_weight_clip)
        if sample_weight is not None:
            sw_ext = np.asarray(sample_weight, dtype=float)
            if len(sw_ext) == len(sw_bin):
                sw_bin = sw_bin * (sw_ext / max(float(np.mean(sw_ext)), 1e-12))
        self.binary_estimator_ = clone(self.binary_estimator)
        if _supports_fit_parameter(self.binary_estimator_, "sample_weight"):
            self.binary_estimator_.fit(X, y_bin, sample_weight=sw_bin)
        else:
            self.binary_estimator_.fit(X, y_bin)

        attack_global = np.array(sorted(np.unique(y_arr[attack_mask])), dtype=int)
        self.attack_classes_global_ = attack_global
        self.global_to_local_ = {int(g): i for i, g in enumerate(attack_global.tolist())}
        y_attack_local = np.array([self.global_to_local_[int(v)] for v in y_arr[attack_mask]], dtype=int)
        sw_attack = _make_sample_weight_vector(y_attack_local, mode=self.attack_weight_mode, clip=self.sample_weight_clip)
        if sample_weight is not None:
            sw_ext_attack = np.asarray(sample_weight, dtype=float)[attack_mask]
            if len(sw_ext_attack) == len(sw_attack):
                sw_attack = sw_attack * (sw_ext_attack / max(float(np.mean(sw_ext_attack)), 1e-12))
        self.attack_estimator_ = clone(self.attack_estimator)
        if _supports_fit_parameter(self.attack_estimator_, "sample_weight"):
            self.attack_estimator_.fit(X[attack_mask], y_attack_local, sample_weight=sw_attack)
        else:
            self.attack_estimator_.fit(X[attack_mask], y_attack_local)
        return self

    def _binary_attack_proba(self, X) -> np.ndarray:
        p = _predict_proba_safe(self.binary_estimator_, X)
        if p is None:
            return np.asarray(self.binary_estimator_.predict(X), dtype=float)
        p = np.asarray(p, dtype=float)
        cls = _estimator_classes(self.binary_estimator_)
        if cls is not None and len(cls) == p.shape[1]:
            for j, c in enumerate(cls):
                if int(c) == 1:
                    return p[:, j]
        return p[:, -1] if p.shape[1] > 1 else np.zeros(p.shape[0], dtype=float)

    def predict_proba(self, X):
        p_attack = np.clip(self._binary_attack_proba(X), 0.0, 1.0)
        p_cond = _predict_proba_safe(self.attack_estimator_, X)
        if p_cond is None:
            pred_local = np.asarray(self.attack_estimator_.predict(X), dtype=int)
            p_cond = np.zeros((len(pred_local), len(self.attack_classes_global_)), dtype=float)
            p_cond[np.arange(len(pred_local)), pred_local] = 1.0
        p_cond = np.asarray(p_cond, dtype=float)
        # Align conditional attack probabilities to local attack classes.
        if p_cond.shape[1] != len(self.attack_classes_global_):
            local_full = np.zeros((p_cond.shape[0], len(self.attack_classes_global_)), dtype=float)
            cls = _estimator_classes(self.attack_estimator_)
            if cls is not None and len(cls) == p_cond.shape[1]:
                for j, c in enumerate(cls):
                    idx = int(c)
                    if 0 <= idx < local_full.shape[1]:
                        local_full[:, idx] = p_cond[:, j]
                p_cond = local_full
        full = np.zeros((p_cond.shape[0], len(self.classes_)), dtype=float)
        full[:, self.normal_class_] = 1.0 - p_attack
        for local_idx, global_idx in enumerate(self.attack_classes_global_):
            if local_idx < p_cond.shape[1]:
                full[:, int(global_idx)] = p_attack * p_cond[:, local_idx]
        row_sum = full.sum(axis=1)
        bad = row_sum <= 0
        if bad.any():
            full[bad, self.normal_class_] = 1.0
            row_sum = full.sum(axis=1)
        full = full / row_sum[:, None]
        return full

    def predict(self, X):
        return _pred_indices_from_proba(self.predict_proba(X), normal_idx=self.normal_class_, normal_threshold=self.threshold)


def _model_family(name: str) -> str:
    s = str(name).lower()
    s = re.sub(r"__w_[a-z0-9_]+$", "", s)
    if s.startswith("two_stage"):
        m = re.search(r"_d(\d+)", s)
        return f"two_stage_xgboost_d{m.group(1)}" if m else "two_stage"
    if s.startswith("xgb"):
        m = re.search(r"_d(\d+)_", s)
        return f"xgboost_d{m.group(1)}" if m else "xgboost"
    if "hist_gradient" in s:
        return "hist_gradient_boosting"
    if "extra_trees" in s:
        return "extra_trees"
    if "random_forest" in s:
        return "random_forest"
    if "logreg" in s or "logistic" in s:
        return "logistic_regression"
    return s


def _xgb_binary_estimator(*, seed: int, n_jobs: int, gpu: bool, depth: int, n_estimators: int, lr: float, subsample: float, colsample: float, reg_lambda: float) -> Any:
    import xgboost as xgb
    from xgboost import XGBClassifier
    xgb_major = _parse_xgb_major(getattr(xgb, "__version__", "2"))
    kwargs: Dict[str, Any] = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": int(depth),
        "n_estimators": int(n_estimators),
        "learning_rate": float(lr),
        "subsample": float(subsample),
        "colsample_bytree": float(colsample),
        "reg_lambda": float(reg_lambda),
        "random_state": int(seed),
        "n_jobs": int(n_jobs),
        "tree_method": "hist",
        "verbosity": 0,
    }
    if gpu:
        if xgb_major >= 2:
            kwargs["device"] = "cuda"
        else:
            kwargs["tree_method"] = "gpu_hist"
            kwargs["predictor"] = "gpu_predictor"
    else:
        if xgb_major >= 2:
            kwargs["device"] = "cpu"
    return XGBClassifier(**kwargs)


def _xgb_multi_estimator(*, n_classes: int, seed: int, n_jobs: int, gpu: bool, depth: int, n_estimators: int, lr: float, subsample: float, colsample: float, reg_lambda: float) -> Any:
    return _xgb_candidate(
        "internal_xgb_multi",
        n_classes=n_classes,
        seed=seed,
        n_jobs=n_jobs,
        gpu=gpu,
        depth=depth,
        n_estimators=n_estimators,
        lr=lr,
        subsample=subsample,
        colsample=colsample,
        reg_lambda=reg_lambda,
    )[1]


def _attach_weight_mode(est: Any, mode: str) -> Any:
    try:
        setattr(est, "_candidate_weight_mode", str(mode))
    except Exception:
        pass
    return est


def build_candidates(args: argparse.Namespace, n_classes: int, use_gpu: bool, normal_idx: Optional[int] = None) -> List[Tuple[str, Any]]:
    preset = args.preset
    base: List[Tuple[str, Any]] = []
    if _xgb_available():
        if preset == "fast":
            specs = [(4, 120, 0.12, 0.90, 0.90, 1.0), (6, 180, 0.09, 0.90, 0.90, 1.5)]
        elif preset == "max":
            specs = [(3, 600, 0.05, 0.95, 0.95, 0.8), (5, 750, 0.045, 0.90, 0.90, 1.0), (7, 750, 0.04, 0.85, 0.85, 1.5)]
        else:
            specs = [(3, 250, 0.08, 0.95, 0.95, 0.8), (5, 350, 0.065, 0.90, 0.90, 1.0), (7, 450, 0.055, 0.90, 0.85, 1.5)]

        devices: List[Tuple[str, bool]] = []
        if use_gpu:
            devices.append(("gpu", True))
        if (not use_gpu) or bool(getattr(args, "xgb_cpu_fallback", False)):
            devices.append(("cpu", False))

        seen_devices = set()
        for device_label, gpu_flag in devices:
            if device_label in seen_devices:
                continue
            seen_devices.add(device_label)
            for i, (d, n, lr, sub, col, reg) in enumerate(specs, 1):
                base.append(_xgb_candidate(
                    f"xgb_{device_label}_d{d}_{i}",
                    n_classes=n_classes,
                    seed=args.seed,
                    n_jobs=args.n_jobs,
                    gpu=gpu_flag,
                    depth=d,
                    n_estimators=n,
                    lr=lr,
                    subsample=sub,
                    colsample=col,
                    reg_lambda=reg,
                ))

        if bool(getattr(args, "add_two_stage", False)) and normal_idx is not None and 0 <= int(normal_idx) < int(n_classes) and n_classes > 2:
            # Two-stage: binary normal/attack + attack-family multiclass. It is especially useful
            # for Harvard, where direct class balancing can overpredict attack labels for NORMAL.
            two_stage_specs = specs[-2:] if preset in {"strong", "max"} else specs[-1:]
            for device_label, gpu_flag in devices:
                for i, (d, n, lr, sub, col, reg) in enumerate(two_stage_specs, 1):
                    bin_est = _xgb_binary_estimator(
                        seed=args.seed + 101 + i,
                        n_jobs=args.n_jobs,
                        gpu=gpu_flag,
                        depth=max(2, min(int(d), 5)),
                        n_estimators=max(150, int(n * 0.75)),
                        lr=lr,
                        subsample=sub,
                        colsample=col,
                        reg_lambda=reg,
                    )
                    atk_est = _xgb_multi_estimator(
                        n_classes=n_classes - 1,
                        seed=args.seed + 202 + i,
                        n_jobs=args.n_jobs,
                        gpu=gpu_flag,
                        depth=int(d),
                        n_estimators=int(n),
                        lr=lr,
                        subsample=sub,
                        colsample=col,
                        reg_lambda=reg,
                    )
                    ts = TwoStageMulticlassClassifier(
                        binary_estimator=bin_est,
                        attack_estimator=atk_est,
                        normal_class=int(normal_idx),
                        threshold=0.5,
                        binary_weight_mode=getattr(args, "two_stage_binary_weight_mode", "sqrt_balanced"),
                        attack_weight_mode=getattr(args, "two_stage_attack_weight_mode", "balanced_clipped"),
                        sample_weight_clip=float(getattr(args, "sample_weight_clip", 20.0)),
                    )
                    _attach_weight_mode(ts, "none")
                    base.append((f"two_stage_xgb_{device_label}_d{d}_{i}", ts))
    elif args.use_gpu == "on":
        print("[WARN] xgboost no está instalado; no se pudo usar GPU.")

    if not bool(getattr(args, "two_stage_only", False)):
        if preset in {"strong", "max"} and not bool(getattr(args, "skip_histgb", False)):
            base.append(("hist_gradient_boosting", HistGradientBoostingClassifier(
                max_iter=450 if preset == "strong" else 750,
                learning_rate=0.045 if preset == "strong" else 0.035,
                max_leaf_nodes=31 if preset == "strong" else 63,
                l2_regularization=0.05,
                early_stopping=True,
                random_state=args.seed,
            )))
        if not bool(getattr(args, "skip_extra_trees", False)):
            base.append(("extra_trees", ExtraTreesClassifier(
                n_estimators=600 if preset != "max" else 1000,
                max_features="sqrt",
                class_weight="balanced",
                random_state=args.seed,
                n_jobs=args.n_jobs,
            )))
        if not bool(getattr(args, "skip_logreg", False)):
            base.append(("logreg_balanced", Pipeline([
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(solver="saga", penalty="l2", C=1.5, class_weight="balanced", max_iter=3000, n_jobs=args.n_jobs, random_state=args.seed)),
            ])))
        if preset == "max" and not bool(getattr(args, "skip_random_forest", False)):
            base.append(("random_forest_balanced", RandomForestClassifier(
                n_estimators=900,
                max_features="sqrt",
                class_weight="balanced_subsample",
                n_jobs=args.n_jobs,
                random_state=args.seed,
            )))

    # Duplicate direct candidates across weighting modes. TwoStage handles its own internal weights.
    modes = _parse_csv_list_simple(getattr(args, "candidate_weight_modes", "")) or [str(getattr(args, "sample_weight_mode", "balanced"))]
    modes = [m.strip().lower().replace("-", "_") for m in modes if m.strip()]
    out: List[Tuple[str, Any]] = []
    for name, est in base:
        if isinstance(est, TwoStageMulticlassClassifier):
            out.append((name, est))
            continue
        for mode in modes:
            try:
                est2 = clone(est)
            except Exception:
                est2 = est
            _attach_weight_mode(est2, mode)
            suffix = f"__w_{mode}"
            out.append((name + suffix, est2))
    return out


def _fit(estimator: Any, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray) -> None:
    if isinstance(estimator, Pipeline):
        # Current pipeline uses LR as final step.
        estimator.fit(X, y, lr__sample_weight=sample_weight)
    elif _supports_fit_parameter(estimator, "sample_weight"):
        estimator.fit(X, y, sample_weight=sample_weight)
    else:
        estimator.fit(X, y)


def _predict_proba_safe(estimator: Any, X: np.ndarray) -> Optional[np.ndarray]:
    try:
        return np.asarray(estimator.predict_proba(X), dtype=float)
    except Exception:
        return None


def _estimator_classes(estimator: Any) -> Optional[np.ndarray]:
    if hasattr(estimator, "classes_"):
        try:
            return np.asarray(getattr(estimator, "classes_"))
        except Exception:
            return None
    if isinstance(estimator, Pipeline):
        try:
            last = estimator.steps[-1][1]
            if hasattr(last, "classes_"):
                return np.asarray(getattr(last, "classes_"))
        except Exception:
            return None
    return None


def _align_proba_to_all_classes(proba: Optional[np.ndarray], estimator: Any, n_classes: int) -> Optional[np.ndarray]:
    if proba is None:
        return None
    arr = np.asarray(proba, dtype=float)
    if arr.ndim != 2:
        return None
    if arr.shape[1] == n_classes:
        return arr
    est_classes = _estimator_classes(estimator)
    if est_classes is None or len(est_classes) != arr.shape[1]:
        return arr
    full = np.zeros((arr.shape[0], n_classes), dtype=float)
    for j, c in enumerate(est_classes):
        try:
            idx = int(c)
        except Exception:
            continue
        if 0 <= idx < n_classes:
            full[:, idx] = arr[:, j]
    return full


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, classes: Sequence[str], proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
    labels = list(classes)
    d: Dict[str, Any] = {
        "n": int(len(y_true)),
        "labels": labels,
        "accuracy": _safe_float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": _safe_float(balanced_accuracy_score(y_true, y_pred)),
        "f1_micro": _safe_float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "f1_macro": _safe_float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": _safe_float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_macro": _safe_float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": _safe_float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_macro": _safe_float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_weighted": _safe_float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "mcc": _safe_float(matthews_corrcoef(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "classification_report": classification_report(y_true, y_pred, labels=labels, output_dict=True, digits=6, zero_division=0),
    }
    if proba is not None and len(labels) > 2:
        try:
            y_bin = label_binarize(y_true, classes=labels)
            d["roc_auc_ovr_macro"] = _safe_float(roc_auc_score(y_bin, proba, multi_class="ovr", average="macro"))
            d["roc_auc_ovr_weighted"] = _safe_float(roc_auc_score(y_bin, proba, multi_class="ovr", average="weighted"))
            d["pr_auc_macro"] = _safe_float(average_precision_score(y_bin, proba, average="macro"))
            d["pr_auc_weighted"] = _safe_float(average_precision_score(y_bin, proba, average="weighted"))
        except Exception:
            pass
    return d


def _safe_extract_final_estimator(estimator: Any) -> Any:
    if isinstance(estimator, NormalThresholdWrapper):
        return _safe_extract_final_estimator(estimator.estimator)
    if isinstance(estimator, TwoStageMulticlassClassifier):
        # Prefer attack estimator for native multiclass importances; permutation importance
        # is still computed on the full two-stage wrapper.
        return getattr(estimator, "attack_estimator_", estimator)
    if isinstance(estimator, Pipeline):
        return estimator.steps[-1][1]
    return estimator


def _native_importance(estimator: Any, features: Sequence[str], classes: Sequence[str]) -> pd.DataFrame:
    feats = list(features)
    base = _safe_extract_final_estimator(estimator)
    out = pd.DataFrame({"feature": feats})
    try:
        # XGBoost: prefer gain, but also save all common score types.
        if base.__class__.__module__.startswith("xgboost") and hasattr(base, "get_booster"):
            booster = base.get_booster()
            score_types = ["gain", "weight", "cover", "total_gain", "total_cover"]
            scores_by_type = {}
            for typ in score_types:
                scores = booster.get_score(importance_type=typ)
                vals = []
                for i, f in enumerate(feats):
                    vals.append(float(scores.get(f, scores.get(f"f{i}", 0.0))))
                out[f"xgb_{typ}"] = vals
                scores_by_type[typ] = vals
            out["native_importance"] = pd.to_numeric(out["xgb_gain"], errors="coerce").fillna(0.0)
            out["native_importance_method"] = "xgboost_gain"
            return out
        if hasattr(base, "feature_importances_"):
            vals = np.asarray(getattr(base, "feature_importances_"), dtype=float)
            if vals.shape[0] == len(feats):
                out["native_importance"] = vals
                out["native_importance_method"] = "feature_importances_"
                return out
        if hasattr(base, "coef_"):
            coef = np.asarray(getattr(base, "coef_"), dtype=float)
            if coef.ndim == 1:
                coef = coef.reshape(1, -1)
            if coef.shape[1] == len(feats):
                abs_mean = np.mean(np.abs(coef), axis=0)
                out["native_importance"] = abs_mean
                out["native_importance_method"] = "abs_mean_coef"
                for i, cls in enumerate(list(classes)[:coef.shape[0]]):
                    out[f"coef_{_safe_name(cls)}"] = coef[i, :]
                return out
    except Exception as e:
        out["native_importance_error"] = f"{type(e).__name__}: {e}"
    out["native_importance"] = 0.0
    out["native_importance_method"] = "not_available"
    return out


def _scoring_name(name: str) -> str:
    aliases = {
        "f1_macro": "f1_macro",
        "f1_weighted": "f1_weighted",
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
    }
    return aliases.get(str(name).strip().lower(), "f1_macro")


def _permutation_importance_df(
    estimator: Any,
    features: Sequence[str],
    X: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    n_repeats: int,
    n_jobs: int,
    max_rows: int,
    scoring: str,
) -> pd.DataFrame:
    feats = list(features)
    out = pd.DataFrame({"feature": feats})
    if permutation_importance is None or n_repeats <= 0:
        out["permutation_importance_mean"] = 0.0
        out["permutation_importance_std"] = 0.0
        out["permutation_scoring"] = _scoring_name(scoring)
        return out
    Xp, yp = X, y
    if max_rows and max_rows > 0 and len(X) > max_rows:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X), size=int(max_rows), replace=False)
        Xp = X[idx]
        yp = y[idx]
    try:
        res = permutation_importance(
            estimator,
            Xp,
            yp,
            scoring=_scoring_name(scoring),
            n_repeats=int(n_repeats),
            random_state=int(seed),
            n_jobs=int(n_jobs),
        )
        out["permutation_importance_mean"] = res.importances_mean
        out["permutation_importance_std"] = res.importances_std
        out["permutation_scoring"] = _scoring_name(scoring)
        out["permutation_rows"] = int(len(Xp))
    except Exception as e:
        out["permutation_importance_mean"] = 0.0
        out["permutation_importance_std"] = 0.0
        out["permutation_scoring"] = _scoring_name(scoring)
        out["permutation_error"] = f"{type(e).__name__}: {e}"
    return out


def _univariate_eta2(X: np.ndarray, y: np.ndarray, features: Sequence[str]) -> pd.DataFrame:
    vals = []
    y = np.asarray(y)
    for j in range(X.shape[1]):
        x = np.asarray(X[:, j], dtype=float)
        overall = float(np.nanmean(x)) if len(x) else 0.0
        ss_between = 0.0
        ss_total = float(np.nansum((x - overall) ** 2))
        for cls in np.unique(y):
            xc = x[y == cls]
            if len(xc) == 0:
                continue
            ss_between += len(xc) * float((np.nanmean(xc) - overall) ** 2)
        vals.append(float(ss_between / ss_total) if ss_total > 0 else 0.0)
    return pd.DataFrame({"feature": list(features), "univariate_eta2": vals})


def _feature_importance(
    estimator: Any,
    features: Sequence[str],
    classes: Sequence[str],
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    seed: int,
    kind: str,
    n_repeats: int,
    n_jobs: int,
    max_rows: int,
    scoring: str,
) -> pd.DataFrame:
    feats = list(features)
    out = pd.DataFrame({"feature": feats})
    kind = (kind or "both").strip().lower()

    native = _native_importance(estimator, feats, classes) if kind in {"native", "builtin", "auto", "both"} else pd.DataFrame({"feature": feats})
    perm = _permutation_importance_df(
        estimator,
        feats,
        X_test,
        y_test,
        seed=seed,
        n_repeats=n_repeats if kind in {"permutation", "both", "auto"} else 0,
        n_jobs=n_jobs,
        max_rows=max_rows,
        scoring=scoring,
    ) if kind != "none" else pd.DataFrame({"feature": feats})
    uni = _univariate_eta2(X_test, y_test, feats)

    out = out.merge(native, on="feature", how="left").merge(perm, on="feature", how="left").merge(uni, on="feature", how="left")
    for c in ["native_importance", "permutation_importance_mean", "univariate_eta2"]:
        if c not in out.columns:
            out[c] = 0.0
        out[c] = pd.to_numeric(out[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

    perm_arr = out["permutation_importance_mean"].to_numpy(dtype=float)
    native_arr = out["native_importance"].to_numpy(dtype=float)
    uni_arr = out["univariate_eta2"].to_numpy(dtype=float)

    if kind != "native" and np.isfinite(perm_arr).any() and float(np.nansum(np.abs(perm_arr))) > 0.0:
        out["importance"] = out["permutation_importance_mean"]
        out["importance_method"] = "permutation_importance"
    elif np.isfinite(native_arr).any() and float(np.nansum(np.abs(native_arr))) > 0.0:
        out["importance"] = out["native_importance"]
        method = out["native_importance_method"] if "native_importance_method" in out.columns else "native"
        out["importance_method"] = method
    elif np.isfinite(uni_arr).any() and float(np.nansum(np.abs(uni_arr))) > 0.0:
        out["importance"] = out["univariate_eta2"]
        out["importance_method"] = "univariate_eta2_fallback"
    else:
        out["importance"] = 0.0
        out["importance_method"] = "zero_fallback"

    for col in out.columns:
        if col == "feature" or col.endswith("method") or col.endswith("error") or col == "permutation_scoring":
            continue
        out[col] = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["importance"] = pd.to_numeric(out["importance"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    denom = float(np.abs(out["importance"]).sum())
    out["importance_normalized_abs"] = np.abs(out["importance"]) / denom if denom > 0 else 0.0
    out = out.sort_values("importance", ascending=False, kind="mergesort").reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1, dtype=int)
    front = ["feature", "importance", "rank", "importance_normalized_abs", "importance_method"]
    rest = [c for c in out.columns if c not in front]
    return out[front + rest]


RESULT_ARTIFACTS_TO_CLEAN = [
    "model_multiclass_fixed30.joblib",
    "metrics_multiclass_fixed30.json",
    "candidate_scores.csv",
    "candidate_scores_ranked.csv",
    "candidate_manifest.csv",
    "candidate_manifest.json",
    "selected_model_summary.json",
    "selected_model_summary.txt",
    "confusion_matrix_test.csv",
    "classification_report_test.csv",
    "test_predictions.csv",
    "feature_importance.csv",
    "feature_importance_native.csv",
    "feature_importance_permutation.csv",
    "feature_importance_skipped.json",
    "label_distribution.csv",
    "processed_fixed30.parquet",
    "processed_fixed30.csv",
]


def _clean_result_artifacts(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in RESULT_ARTIFACTS_TO_CLEAN:
        p = out_dir / name
        if p.exists():
            if p.is_dir():
                import shutil
                shutil.rmtree(p)
            else:
                p.unlink()
    for p in out_dir.glob("*.tmp"):
        try:
            p.unlink()
        except Exception:
            pass



def _read_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    """Read a JSON file if it exists; return None if missing or invalid."""
    try:
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else None
    except Exception as e:
        print(f"[WARN] Could not read JSON {path}: {type(e).__name__}: {e}", flush=True)
        return None


def _write_candidate_scores(score_rows: Sequence[Dict[str, Any]], out_dir: Path, metric_name: str) -> pd.DataFrame:
    """Save raw and ranked candidate scores; return the ranked DataFrame."""
    raw_path = out_dir / "candidate_scores.csv"
    ranked_path = out_dir / "candidate_scores_ranked.csv"
    df_scores = pd.DataFrame(list(score_rows))
    df_scores.to_csv(raw_path, index=False)
    if df_scores.empty:
        df_scores.to_csv(ranked_path, index=False)
        return df_scores

    ranked = df_scores.copy()
    if "status" not in ranked.columns:
        ranked["status"] = "unknown"
    if metric_name not in ranked.columns:
        ranked[metric_name] = np.nan
    ranked["_candidate_order"] = np.arange(len(ranked), dtype=int)
    ranked["_status_ok"] = ranked["status"].astype(str).eq("ok")
    ranked["_selection_metric_value"] = pd.to_numeric(ranked[metric_name], errors="coerce")
    ranked = ranked.sort_values(
        by=["_status_ok", "_selection_metric_value", "_candidate_order"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1, dtype=int))
    ranked = ranked.drop(columns=[c for c in ["_candidate_order", "_status_ok", "_selection_metric_value"] if c in ranked.columns])
    ranked.to_csv(ranked_path, index=False)

    ok = ranked[ranked["status"].astype(str).eq("ok")]
    if not ok.empty:
        print("\n[LEADERBOARD] Top candidates by validation metric:", flush=True)
        show_cols = [c for c in ["rank", "candidate", "model_family", "weight_mode", "normal_threshold", metric_name, "accuracy", "f1_weighted", "train_seconds"] if c in ok.columns]
        print(ok.head(15)[show_cols].to_string(index=False), flush=True)
    return ranked




def _is_completed_candidate_status(status: Any) -> bool:
    s = str(status or "").strip().lower()
    return s == "ok" or s.startswith("failed:")


def _to_resume_row_dict(row: pd.Series) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in row.to_dict().items():
        if pd.isna(v):
            out[str(k)] = None
        elif isinstance(v, np.generic):
            out[str(k)] = v.item()
        else:
            out[str(k)] = v
    return out


def _optional_float_from_any(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        if isinstance(v, str) and not v.strip():
            return None
        x = float(v)
        if not np.isfinite(x):
            return None
        return x
    except Exception:
        return None


def _load_completed_candidate_scores(out_dir: Path, candidate_names: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    """Load already completed candidate scores so a long all-model run can resume safely."""
    path = out_dir / "candidate_scores.csv"
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"[WARN] Could not read existing candidate_scores.csv for resume: {type(e).__name__}: {e}", flush=True)
        return {}
    if df.empty or "candidate" not in df.columns or "status" not in df.columns:
        return {}
    allowed = set(str(x) for x in candidate_names)
    completed: Dict[str, Dict[str, Any]] = {}
    for _, row in df.iterrows():
        name = str(row.get("candidate", ""))
        status = row.get("status", "")
        if name in allowed and _is_completed_candidate_status(status):
            completed[name] = _to_resume_row_dict(row)
    return completed


def _write_candidate_manifest(
    candidates: Sequence[Tuple[str, Any]],
    out_dir: Path,
    *,
    args: argparse.Namespace,
    use_gpu: bool,
    required_families: Sequence[str],
) -> Dict[str, Any]:
    """Write the exact model menu before training and optionally fail if a required family is missing."""
    rows: List[Dict[str, Any]] = []
    for i, (name, est) in enumerate(candidates, 1):
        rows.append({
            "order": int(i),
            "candidate": str(name),
            "model_family": _model_family(str(name)),
            "weight_mode": str(getattr(est, "_candidate_weight_mode", "")),
            "estimator_class": est.__class__.__name__,
            "estimator_module": est.__class__.__module__,
        })
    manifest_df = pd.DataFrame(rows)
    manifest_csv = out_dir / "candidate_manifest.csv"
    manifest_json = out_dir / "candidate_manifest.json"
    manifest_df.to_csv(manifest_csv, index=False)

    observed = sorted(set(manifest_df["model_family"].astype(str).tolist())) if not manifest_df.empty else []
    required = [str(x).strip() for x in required_families if str(x).strip()]
    missing = sorted(set(required) - set(observed))
    payload = {
        "candidate_count": int(len(rows)),
        "families": observed,
        "required_families": required,
        "missing_required_families": missing,
        "use_gpu_resolved": bool(use_gpu),
        "preset": str(getattr(args, "preset", "")),
        "xgb_cpu_fallback": bool(getattr(args, "xgb_cpu_fallback", False)),
        "add_two_stage": bool(getattr(args, "add_two_stage", False)),
        "skip_histgb": bool(getattr(args, "skip_histgb", False)),
        "skip_extra_trees": bool(getattr(args, "skip_extra_trees", False)),
        "skip_logreg": bool(getattr(args, "skip_logreg", False)),
        "skip_random_forest": bool(getattr(args, "skip_random_forest", False)),
        "candidate_weight_modes": str(getattr(args, "candidate_weight_modes", "")),
        "paths": {"csv": str(manifest_csv), "json": str(manifest_json)},
    }
    _save_json(payload, manifest_json)
    print("\n[MANIFEST] Candidate model menu:", flush=True)
    print(f"[MANIFEST] count={len(rows)} families={', '.join(observed)}", flush=True)
    if required:
        print(f"[MANIFEST] required_families={', '.join(required)}", flush=True)
    if missing:
        raise SystemExit("Missing required model families: " + ", ".join(missing) + f". See {manifest_csv}")
    return payload

def _write_selected_model_summary(
    *,
    out_dir: Path,
    project_dir: str,
    dataset: str,
    classes: Sequence[str],
    best_name: str,
    best_score: float,
    best_threshold: Optional[float],
    best_weight_mode: str,
    metric_name: str,
    ranked_scores: pd.DataFrame,
    test_metrics: Dict[str, Any],
    comparison: Dict[str, Any],
    model_path: Path,
    processed_path: Path,
) -> Dict[str, str]:
    """Write compact best-model summary and Harvard-vs-TorpEda comparison."""
    torpeda_metrics_path = Path(project_dir) / "resultsOptimo" / "torpeda" / "multiclass" / "metrics_multiclass_fixed30.json"
    torpeda_metrics = _read_json_if_exists(torpeda_metrics_path)
    torpeda_ref: Optional[Dict[str, Any]] = None
    matches_torpeda_exact: Optional[bool] = None
    matches_torpeda_family: Optional[bool] = None
    if torpeda_metrics:
        torpeda_model = str(torpeda_metrics.get("selected_model") or "")
        torpeda_family = str(torpeda_metrics.get("selected_model_family") or _model_family(torpeda_model))
        current_family = _model_family(best_name)
        matches_torpeda_exact = bool(torpeda_model and torpeda_model == best_name)
        matches_torpeda_family = bool(torpeda_family and torpeda_family == current_family)
        torpeda_ref = {
            "metrics_path": str(torpeda_metrics_path),
            "selected_model": torpeda_model,
            "selected_model_family": torpeda_family,
            "selection_metric": torpeda_metrics.get("selection_metric"),
            "selection_score": torpeda_metrics.get("selection_score"),
            "test_f1_macro": (torpeda_metrics.get("test_metrics") or {}).get("f1_macro"),
            "test_accuracy": (torpeda_metrics.get("test_metrics") or {}).get("accuracy"),
        }

    top_candidates: List[Dict[str, Any]] = []
    if ranked_scores is not None and not ranked_scores.empty:
        wanted_cols = ["rank", "candidate", "model_family", "weight_mode", "normal_threshold", metric_name, "accuracy", "balanced_accuracy", "f1_macro", "f1_weighted", "mcc", "train_seconds", "status"]
        cols = []
        for c in wanted_cols:
            if c in ranked_scores.columns and c not in cols:
                cols.append(c)
        top_candidates = ranked_scores.head(20)[cols].to_dict(orient="records")

    summary = {
        "dataset": dataset,
        "task": "multiclass",
        "best_model": best_name,
        "best_model_family": _model_family(best_name),
        "selection_metric": metric_name,
        "validation_selection_score": float(best_score),
        "selected_normal_threshold": best_threshold,
        "selected_weight_mode": best_weight_mode,
        "test_f1_macro": test_metrics.get("f1_macro"),
        "test_accuracy": test_metrics.get("accuracy"),
        "test_balanced_accuracy": test_metrics.get("balanced_accuracy"),
        "test_mcc": test_metrics.get("mcc"),
        "classes": list(classes),
        "n_classes": int(len(classes)),
        "feature_importance_skipped": True,
        "top_candidates": top_candidates,
        "torpeda_reference": torpeda_ref,
        "matches_torpeda_exact_model": matches_torpeda_exact,
        "matches_torpeda_model_family": matches_torpeda_family,
        "cross_dataset_comparison": comparison,
        "artifacts": {
            "model": str(model_path),
            "processed_features": str(processed_path),
            "candidate_scores": str(out_dir / "candidate_scores.csv"),
            "candidate_scores_ranked": str(out_dir / "candidate_scores_ranked.csv"),
            "metrics": str(out_dir / "metrics_multiclass_fixed30.json"),
        },
    }
    json_path = out_dir / "selected_model_summary.json"
    txt_path = out_dir / "selected_model_summary.txt"
    _save_json(summary, json_path)

    lines = [
        "BEST_MODEL_SUMMARY",
        f"dataset={dataset}",
        f"best_model={best_name}",
        f"best_model_family={_model_family(best_name)}",
        f"selection_metric={metric_name}",
        f"validation_selection_score={best_score:.6f}",
        f"selected_normal_threshold={best_threshold}",
        f"selected_weight_mode={best_weight_mode}",
        f"test_f1_macro={test_metrics.get('f1_macro')}",
        f"test_accuracy={test_metrics.get('accuracy')}",
        f"test_balanced_accuracy={test_metrics.get('balanced_accuracy')}",
        f"test_mcc={test_metrics.get('mcc')}",
        f"feature_importance_skipped=True",
    ]
    if torpeda_ref is None:
        lines.extend([
            "torpeda_reference_found=False",
            f"torpeda_expected_metrics_path={torpeda_metrics_path}",
            "matches_torpeda_exact_model=unknown",
            "matches_torpeda_model_family=unknown",
        ])
    else:
        lines.extend([
            "torpeda_reference_found=True",
            f"torpeda_selected_model={torpeda_ref.get('selected_model')}",
            f"torpeda_selected_model_family={torpeda_ref.get('selected_model_family')}",
            f"torpeda_test_f1_macro={torpeda_ref.get('test_f1_macro')}",
            f"matches_torpeda_exact_model={matches_torpeda_exact}",
            f"matches_torpeda_model_family={matches_torpeda_family}",
        ])
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\n[SUMMARY] Best Harvard model:", flush=True)
    print("\n".join(lines), flush=True)
    return {"json": str(json_path), "txt": str(txt_path)}

def _write_cross_dataset_comparison(project_dir: str, current_dataset: str) -> Dict[str, Any]:
    root = Path(project_dir) / "resultsOptimo"
    rows: List[Dict[str, Any]] = []
    for dataset_dir in sorted(root.glob("*/multiclass")):
        metrics_path = dataset_dir / "metrics_multiclass_fixed30.json"
        if not metrics_path.exists():
            continue
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                m = json.load(f)
            ds = str(m.get("dataset") or dataset_dir.parent.name)
            selected = str(m.get("selected_model") or "")
            rows.append({
                "dataset": ds,
                "selected_model": selected,
                "selected_model_family": str(m.get("selected_model_family") or _model_family(selected)),
                "selection_metric": m.get("selection_metric"),
                "selection_score": m.get("selection_score"),
                "test_f1_macro": (m.get("test_metrics") or {}).get("f1_macro"),
                "test_accuracy": (m.get("test_metrics") or {}).get("accuracy"),
                "metrics_path": str(metrics_path),
            })
        except Exception as e:
            print(f"[WARN] Could not read comparison metrics from {metrics_path}: {e}", flush=True)
    comparison: Dict[str, Any] = {"current_dataset": current_dataset, "datasets_found": rows}
    if rows:
        models = {r["selected_model"] for r in rows if r.get("selected_model")}
        families = {r["selected_model_family"] for r in rows if r.get("selected_model_family")}
        comparison["all_selected_models_match"] = len(models) == 1
        comparison["all_selected_model_families_match"] = len(families) == 1
        if len(models) == 1:
            comparison["recommended_model"] = next(iter(models))
            comparison["recommendation_reason"] = "all_datasets_selected_same_exact_model"
        elif len(families) == 1:
            comparison["recommended_model_family"] = next(iter(families))
            comparison["recommendation_reason"] = "all_datasets_selected_same_model_family"
        else:
            comparison["recommended_model"] = None
            comparison["recommendation_reason"] = "selected_models_do_not_match_yet_compare_candidate_scores_or_run_for_all_datasets"
    root.mkdir(parents=True, exist_ok=True)
    out_json = root / "multiclass_fixed30_model_comparison.json"
    _save_json(comparison, out_json)
    try:
        pd.DataFrame(rows).to_csv(root / "multiclass_fixed30_model_comparison.csv", index=False)
    except Exception:
        pass
    print(f"[INFO] Cross-dataset comparison written: {out_json}", flush=True)
    return comparison


def _apply_dataset_optimizations(df: pd.DataFrame, args: argparse.Namespace, out_dir: Path) -> pd.DataFrame:
    """Filter/clean labels before saving processed dataset and splitting.

    This is controlled by CLI flags. For Harvard optimized runs it removes classes
    that cannot be reliably stratified/evaluated, while preserving explicit labels.
    """
    out = df.copy()
    reports: Dict[str, Any] = {
        "rows_before": int(len(out)),
        "class_distribution_before": out[LABEL_COL].astype(str).value_counts().to_dict() if LABEL_COL in out.columns else {},
        "operations": [],
    }

    if LABEL_COL not in out.columns:
        raise SystemExit(f"Missing {LABEL_COL!r} before dataset optimization")

    # Optional explicit keep/drop regexes for experiments.
    keep_re = str(getattr(args, "keep_labels_regex", "") or "").strip()
    if keep_re:
        before = len(out)
        mask = out[LABEL_COL].astype(str).str.contains(keep_re, regex=True, na=False)
        out = out[mask].copy()
        reports["operations"].append({"op": "keep_labels_regex", "regex": keep_re, "removed": int(before - len(out))})

    drop_re = str(getattr(args, "drop_labels_regex", "") or "").strip()
    if drop_re:
        before = len(out)
        mask = out[LABEL_COL].astype(str).str.contains(drop_re, regex=True, na=False)
        out = out[~mask].copy()
        reports["operations"].append({"op": "drop_labels_regex", "regex": drop_re, "removed": int(before - len(out))})

    min_count = int(getattr(args, "min_class_count", 0) or 0)
    if min_count > 1:
        vc = out[LABEL_COL].astype(str).value_counts()
        keep = set(vc[vc >= min_count].index.astype(str).tolist())
        dropped = vc[vc < min_count].rename_axis("label").reset_index(name="count")
        if len(dropped) > 0:
            dropped.to_csv(out_dir / "dropped_classes_by_min_count.csv", index=False)
        before = len(out)
        out = out[out[LABEL_COL].astype(str).isin(keep)].copy()
        reports["operations"].append({"op": "min_class_count", "min_class_count": min_count, "removed_rows": int(before - len(out)), "dropped_classes": dropped.to_dict(orient="records")})

    # Optional cap for NORMAL count to speed experiments. Default 0 means keep all.
    max_normal = int(getattr(args, "max_normal_rows", 0) or 0)
    if max_normal > 0:
        normal_mask = out[LABEL_COL].astype(str).eq("NORMAL")
        n_normal = int(normal_mask.sum())
        if n_normal > max_normal:
            normal_df = out[normal_mask].sample(max_normal, random_state=int(args.seed))
            other_df = out[~normal_mask]
            out = pd.concat([normal_df, other_df], ignore_index=True).sample(frac=1.0, random_state=int(args.seed)).reset_index(drop=True)
            reports["operations"].append({"op": "max_normal_rows", "before_normal": n_normal, "after_normal": max_normal, "removed_rows": int(n_normal - max_normal)})

    # Sanity after filters.
    vc_after = out[LABEL_COL].astype(str).value_counts()
    if len(out) == 0:
        raise SystemExit("No rows left after dataset optimizations/filters.")
    if len(vc_after) < 2:
        raise SystemExit("Need at least two classes after dataset optimizations/filters.")
    reports["rows_after"] = int(len(out))
    reports["class_distribution_after"] = vc_after.to_dict()
    _save_json(reports, out_dir / "dataset_optimization_report.json")
    print(f"[INFO] Dataset optimization: rows {reports['rows_before']} -> {reports['rows_after']}", flush=True)
    if reports["operations"]:
        print("[INFO] Dataset optimization operations:", json.dumps(reports["operations"], ensure_ascii=False, indent=2), flush=True)
    return out.reset_index(drop=True)


def _evaluate_with_optional_threshold(
    estimator: Any,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    le: LabelEncoder,
    classes: Sequence[str],
    *,
    normal_idx: Optional[int],
    threshold_grid: Sequence[Optional[float]],
    metric_name: str,
) -> Tuple[Dict[str, Any], Optional[float], np.ndarray, Optional[np.ndarray]]:
    proba = _align_proba_to_all_classes(_predict_proba_safe(estimator, X_eval), estimator, len(classes))
    y_true_labels = le.inverse_transform(y_eval)
    best_m: Optional[Dict[str, Any]] = None
    best_t: Optional[float] = None
    best_pred_i: Optional[np.ndarray] = None
    # Always evaluate baseline argmax/predict first.
    grids = list(threshold_grid) if threshold_grid else [None]
    if None not in grids:
        grids = [None] + grids

    for t in grids:
        if proba is not None and t is not None:
            pred_i = _pred_indices_from_proba(proba, normal_idx=normal_idx, normal_threshold=t)
        elif proba is not None and t is None:
            # Use the estimator's own predict for baseline because some models may not be pure argmax.
            pred_i = np.asarray(estimator.predict(X_eval), dtype=int)
        else:
            pred_i = np.asarray(estimator.predict(X_eval), dtype=int)
        pred_lab = le.inverse_transform(pred_i.astype(int))
        m = _metrics(y_true_labels, pred_lab, classes, proba)
        score = -1.0 if m.get(metric_name) is None else float(m.get(metric_name))
        if best_m is None or score > float(best_m.get(metric_name) or -1.0):
            best_m = m
            best_t = t
            best_pred_i = pred_i
    assert best_m is not None and best_pred_i is not None
    return best_m, best_t, best_pred_i.astype(int), proba


def train_eval(args: argparse.Namespace, df_raw: pd.DataFrame) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if bool(getattr(args, "clean_output_dir", False)):
        print(f"[INFO] Cleaning old artifacts in {out_dir}", flush=True)
        _clean_result_artifacts(out_dir)

    df = build_feature_frame(df_raw)
    df = _apply_dataset_optimizations(df, args, out_dir)

    processed_target = Path(args.processed_out) if args.processed_out else (out_dir / "processed_fixed30.parquet")
    processed_path = _save_processed_atomic(df, processed_target)
    print(f"[INFO] Replaced processed dataset: {processed_path}", flush=True)

    X = df[FIXED_FEATURES].fillna(0).astype(np.float32).to_numpy()
    y_labels = df[LABEL_COL].astype(str).to_numpy()
    le = LabelEncoder()
    y = le.fit_transform(y_labels)
    classes = [str(c) for c in le.classes_]
    if len(classes) < 2:
        raise SystemExit("Need at least 2 classes to train.")

    print(f"\n[INFO] Dataset={args.dataset} Feature count fixed: {len(FIXED_FEATURES)}", flush=True)
    print("[INFO] Features:", ", ".join(FIXED_FEATURES), flush=True)
    dist = pd.Series(y_labels).value_counts().rename_axis("label").reset_index(name="count")
    print("\n[INFO] Label distribution:", flush=True)
    print(dist.to_string(index=False), flush=True)
    dist.to_csv(out_dir / "label_distribution.csv", index=False)

    trainval_idx, test_idx = _split_indices(y_labels, args.test_size, args.seed)
    X_trainval, X_test = X[trainval_idx], X[test_idx]
    y_trainval, y_test = y[trainval_idx], y[test_idx]
    y_trainval_labels = y_labels[trainval_idx]

    val_relative = args.val_size / max(1e-9, (1.0 - args.test_size))
    val_relative = min(max(val_relative, 0.05), 0.50)
    tr_local, val_local = _split_indices(y_trainval_labels, val_relative, args.seed + 1)
    X_train, X_val = X_trainval[tr_local], X_trainval[val_local]
    y_train, y_val = y_trainval[tr_local], y_trainval[val_local]

    normal_idx = classes.index("NORMAL") if "NORMAL" in classes else None
    threshold_grid = _parse_threshold_grid(args.normal_threshold_grid)
    if threshold_grid:
        print(f"[INFO] Normal threshold tuning enabled: {len(threshold_grid)-1} thresholds + baseline", flush=True)
    else:
        print("[INFO] Normal threshold tuning disabled", flush=True)

    use_gpu = args.use_gpu == "on" or (args.use_gpu == "auto" and _gpu_visible_auto())
    print(f"\n[INFO] GPU requested={args.use_gpu}; resolved={use_gpu}", flush=True)
    print(f"[INFO] GPU env: CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')} "
          f"SLURM_JOB_GPUS={os.environ.get('SLURM_JOB_GPUS', 'unset')} "
          f"SLURM_STEP_GPUS={os.environ.get('SLURM_STEP_GPUS', 'unset')} "
          f"NVIDIA_VISIBLE_DEVICES={os.environ.get('NVIDIA_VISIBLE_DEVICES', 'unset')}", flush=True)

    candidates = build_candidates(args, len(classes), use_gpu, normal_idx=normal_idx)
    if not candidates:
        raise SystemExit("No model candidates available.")

    metric_name = args.metric_select
    candidate_names = [name for name, _ in candidates]
    required_families = _parse_csv_list_simple(getattr(args, "require_candidate_families", ""))
    candidate_manifest = _write_candidate_manifest(
        candidates,
        out_dir,
        args=args,
        use_gpu=use_gpu,
        required_families=required_families,
    )

    score_rows: List[Dict[str, Any]] = []
    best_score = -1.0
    best_name = ""
    best_threshold: Optional[float] = None
    best_weight_mode: str = ""
    resume_rows: Dict[str, Dict[str, Any]] = {}

    if bool(getattr(args, "resume_candidate_scores", False)):
        resume_rows = _load_completed_candidate_scores(out_dir, candidate_names)
        if resume_rows:
            # Preserve current candidate order, ignoring candidates that are no longer in this run.
            score_rows = [resume_rows[name] for name in candidate_names if name in resume_rows]
            print(f"[RESUME] Reusing {len(score_rows)} completed candidate rows from {out_dir / 'candidate_scores.csv'}", flush=True)
            for row in score_rows:
                if str(row.get("status", "")).strip().lower() != "ok":
                    continue
                score_val = _optional_float_from_any(row.get(metric_name))
                if score_val is None:
                    continue
                if score_val > best_score:
                    best_score = score_val
                    best_name = str(row.get("candidate") or "")
                    best_threshold = _optional_float_from_any(row.get("normal_threshold"))
                    best_weight_mode = str(row.get("weight_mode") or "")
            if best_name:
                print(f"[RESUME] Current best before continuing: {best_name} {metric_name}={best_score:.6f}", flush=True)

    for name, est in candidates:
        if name in resume_rows:
            row = resume_rows[name]
            print(f"\n[RESUME] {name}: status={row.get('status')} {metric_name}={row.get(metric_name)}", flush=True)
            continue
        print(f"\n[TRAIN] {name}", flush=True)
        t0 = time.perf_counter()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                weight_mode = str(getattr(est, "_candidate_weight_mode", args.sample_weight_mode))
                sw_train = _make_sample_weight_vector(y_train, mode=weight_mode, clip=args.sample_weight_clip)
                _fit(est, X_train, y_train, sw_train)
            val_m, val_threshold, pred_val_i, proba_val = _evaluate_with_optional_threshold(
                est, X_val, y_val, le, classes, normal_idx=normal_idx, threshold_grid=threshold_grid, metric_name=metric_name
            )
            secs = time.perf_counter() - t0
            row = {
                "candidate": name,
                "model_family": _model_family(name),
                "weight_mode": str(getattr(est, "_candidate_weight_mode", args.sample_weight_mode)),
                "normal_threshold": val_threshold,
                "status": "ok",
                "train_seconds": secs,
                **{k: val_m.get(k) for k in ["accuracy", "balanced_accuracy", "f1_macro", "f1_weighted", "precision_macro", "recall_macro", "mcc"]},
            }
            score_rows.append(row)
            score_val = val_m.get(metric_name)
            score = -1.0 if score_val is None else float(score_val)
            print(f"[VAL] {name}: {metric_name}={score:.6f} f1_macro={val_m.get('f1_macro')} acc={val_m.get('accuracy')} threshold={val_threshold} weight={row['weight_mode']} secs={secs:.1f}", flush=True)
            if score > best_score:
                best_score, best_name = score, name
                best_threshold = val_threshold
                best_weight_mode = row["weight_mode"]
            if bool(getattr(args, "checkpoint_candidate_scores", False)):
                _write_candidate_scores(score_rows, out_dir, metric_name)
        except Exception as e:
            secs = time.perf_counter() - t0
            score_rows.append({"candidate": name, "model_family": _model_family(name), "weight_mode": str(getattr(est, "_candidate_weight_mode", args.sample_weight_mode)), "status": f"failed: {type(e).__name__}: {e}", "train_seconds": secs})
            print(f"[WARN] failed {name}: {type(e).__name__}: {e}", flush=True)
            if bool(getattr(args, "checkpoint_candidate_scores", False)):
                _write_candidate_scores(score_rows, out_dir, metric_name)

    ranked_scores = _write_candidate_scores(score_rows, out_dir, metric_name)
    if not best_name:
        raise SystemExit("All candidates failed.")
    print(f"\n[SELECTED] {best_name} by {metric_name}={best_score:.6f} threshold={best_threshold} weight={best_weight_mode}", flush=True)

    selected = None
    for name, est in build_candidates(args, len(classes), use_gpu, normal_idx=normal_idx):
        if name == best_name:
            selected = est
            break
    if selected is None:
        raise SystemExit("Could not rebuild selected model.")

    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sw_trainval = _make_sample_weight_vector(y_trainval, mode=best_weight_mode or args.sample_weight_mode, clip=args.sample_weight_clip)
        _fit(selected, X_trainval, y_trainval, sw_trainval)
    final_secs = time.perf_counter() - t0

    if best_threshold is not None:
        selected_for_predict: Any = NormalThresholdWrapper(selected, normal_class=int(normal_idx), threshold=float(best_threshold), n_classes=len(classes))
    else:
        selected_for_predict = selected

    pred_test_i, proba_test = _predict_indices_with_threshold(
        selected_for_predict, X_test, n_classes=len(classes), normal_idx=normal_idx, threshold=None
    )
    y_true = le.inverse_transform(y_test)
    y_pred = le.inverse_transform(np.asarray(pred_test_i, dtype=int))
    test_m = _metrics(y_true, y_pred, classes, proba_test)

    print("\n=== TEST ===", flush=True)
    print(f"accuracy={test_m.get('accuracy')} balanced_accuracy={test_m.get('balanced_accuracy')} "
          f"f1_macro={test_m.get('f1_macro')} f1_weighted={test_m.get('f1_weighted')} mcc={test_m.get('mcc')}", flush=True)
    print("\n" + classification_report(y_true, y_pred, labels=classes, digits=4, zero_division=0), flush=True)

    cm = pd.DataFrame(confusion_matrix(y_true, y_pred, labels=classes), index=pd.Index(classes, name="true\\pred"), columns=classes)
    cm.to_csv(out_dir / "confusion_matrix_test.csv")
    pd.DataFrame(test_m["classification_report"]).transpose().to_csv(out_dir / "classification_report_test.csv")

    pred_df = pd.DataFrame({
        "row_index": test_idx,
        "sample_id": df.iloc[test_idx].get("sample_id", pd.Series(test_idx)).astype(str).to_numpy(),
        "source_file": df.iloc[test_idx].get("source_file", pd.Series([""] * len(test_idx))).astype(str).to_numpy(),
        "y_true": y_true,
        "y_pred": y_pred,
    })
    if proba_test is not None:
        pred_df["pred_prob_max"] = np.max(proba_test, axis=1)
        for i, cls in enumerate(classes):
            pred_df[f"proba_{_safe_name(cls)}"] = proba_test[:, i]
    pred_df.to_csv(out_dir / "test_predictions.csv", index=False)

    fi_skipped = bool(getattr(args, "skip_feature_importance", False)) or str(args.fi_kind).strip().lower() == "none"
    feature_importance_artifacts: Dict[str, Optional[str]] = {}
    if fi_skipped:
        skip_payload = {
            "skipped": True,
            "reason": "feature importance disabled for best-model search",
            "fi_kind": str(args.fi_kind),
            "fi_n_repeats": int(args.fi_n_repeats),
            "fi_max_rows": int(args.fi_max_rows),
            "message": "No native, permutation, or univariate feature importance was computed in this run.",
        }
        _save_json(skip_payload, out_dir / "feature_importance_skipped.json")
        feature_importance_artifacts = {
            "feature_importance": None,
            "feature_importance_native": None,
            "feature_importance_permutation": None,
            "feature_importance_skipped": str(out_dir / "feature_importance_skipped.json"),
        }
        print("\n[FI] Skipped feature importance completely (--fi-kind none / --skip-feature-importance).", flush=True)
    else:
        fi = _feature_importance(
            selected_for_predict,
            FIXED_FEATURES,
            classes,
            X_test,
            y_test,
            seed=args.seed,
            kind=args.fi_kind,
            n_repeats=args.fi_n_repeats,
            n_jobs=args.fi_n_jobs,
            max_rows=args.fi_max_rows,
            scoring=args.fi_scoring,
        )
        fi.to_csv(out_dir / "feature_importance.csv", index=False)
        native_cols = [c for c in fi.columns if c == "feature" or c.startswith("native_") or c.startswith("xgb_") or c.startswith("coef_")]
        perm_cols = [c for c in fi.columns if c == "feature" or c.startswith("permutation_")]
        if len(native_cols) > 1:
            fi[native_cols].to_csv(out_dir / "feature_importance_native.csv", index=False)
        else:
            pd.DataFrame({"feature": FIXED_FEATURES}).to_csv(out_dir / "feature_importance_native.csv", index=False)
        if len(perm_cols) > 1:
            fi[perm_cols].to_csv(out_dir / "feature_importance_permutation.csv", index=False)
        else:
            pd.DataFrame({"feature": FIXED_FEATURES}).to_csv(out_dir / "feature_importance_permutation.csv", index=False)
        feature_importance_artifacts = {
            "feature_importance": str(out_dir / "feature_importance.csv"),
            "feature_importance_native": str(out_dir / "feature_importance_native.csv"),
            "feature_importance_permutation": str(out_dir / "feature_importance_permutation.csv"),
            "feature_importance_skipped": None,
        }
        print("\n[FI] Top feature_importance:", flush=True)
        print(fi.head(min(30, len(fi))).to_string(index=False), flush=True)

    model_path = out_dir / "model_multiclass_fixed30.joblib"
    joblib.dump({
        "task": "multiclass",
        "dataset": args.dataset,
        "model": selected_for_predict,
        "base_model_before_postprocess": selected,
        "normal_threshold": best_threshold,
        "weight_mode": best_weight_mode,
        "label_encoder": le,
        "features": FIXED_FEATURES,
        "feature_count": len(FIXED_FEATURES),
        "classes": classes,
        "selected_model": best_name,
        "selected_model_family": _model_family(best_name),
        "config": vars(args),
    }, model_path)

    comparison: Dict[str, Any] = {}

    metrics_payload = {
        "task": "multiclass",
        "dataset": args.dataset,
        "dataset_label_mode": args.harvard_label_mode if args.dataset == "harvard" else "torpeda_native_common_names",
        "feature_policy": "fixed_compact_intra_request",
        "feature_count": len(FIXED_FEATURES),
        "features_used": FIXED_FEATURES,
        "classes": classes,
        "label_distribution": dist.to_dict(orient="records"),
        "selected_model": best_name,
        "selected_model_family": _model_family(best_name),
        "selection_metric": metric_name,
        "selection_score": best_score,
        "selected_normal_threshold": best_threshold,
        "selected_weight_mode": best_weight_mode,
        "candidate_scores": score_rows,
        "candidate_manifest": candidate_manifest,
        "train_rows": int(len(X_trainval)),
        "test_rows": int(len(X_test)),
        "test_size": float(args.test_size),
        "val_size": float(args.val_size),
        "seed": int(args.seed),
        "use_gpu_requested": args.use_gpu,
        "use_gpu_resolved": bool(use_gpu),
        "final_train_seconds": float(final_secs),
        "feature_importance_config": {
            "fi_kind": str(args.fi_kind),
            "fi_n_repeats": int(args.fi_n_repeats),
            "fi_max_rows": int(args.fi_max_rows),
            "fi_scoring": str(args.fi_scoring),
            "fi_n_jobs": int(args.fi_n_jobs),
            "skipped": bool(fi_skipped),
        },
        "cross_dataset_comparison_at_train_time": comparison,
        "test_metrics": test_m,
        "artifacts": {
            "model": str(model_path),
            "processed_features": str(processed_path),
            "candidate_scores": str(out_dir / "candidate_scores.csv"),
            "candidate_scores_ranked": str(out_dir / "candidate_scores_ranked.csv"),
            "candidate_manifest": str(out_dir / "candidate_manifest.csv"),
            "candidate_manifest_json": str(out_dir / "candidate_manifest.json"),
            "selected_model_summary": str(out_dir / "selected_model_summary.json"),
            "confusion_matrix": str(out_dir / "confusion_matrix_test.csv"),
            "classification_report": str(out_dir / "classification_report_test.csv"),
            "predictions": str(out_dir / "test_predictions.csv"),
            **feature_importance_artifacts,
        },
    }
    _save_json(metrics_payload, out_dir / "metrics_multiclass_fixed30.json")
    # Re-write comparison after saving current metrics, so it includes this run.
    comparison = _write_cross_dataset_comparison(args.project_dir, args.dataset)
    metrics_payload["cross_dataset_comparison_at_train_time"] = comparison
    summary_paths = _write_selected_model_summary(
        out_dir=out_dir,
        project_dir=args.project_dir,
        dataset=args.dataset,
        classes=classes,
        best_name=best_name,
        best_score=best_score,
        best_threshold=best_threshold,
        best_weight_mode=best_weight_mode,
        metric_name=metric_name,
        ranked_scores=ranked_scores,
        test_metrics=test_m,
        comparison=comparison,
        model_path=model_path,
        processed_path=processed_path,
    )
    metrics_payload["artifacts"]["selected_model_summary"] = summary_paths["json"]
    metrics_payload["artifacts"]["selected_model_summary_txt"] = summary_paths["txt"]
    _save_json(metrics_payload, out_dir / "metrics_multiclass_fixed30.json")

    print("\n[OK] Saved artifacts:", flush=True)
    for p in [
        processed_path,
        model_path,
        out_dir / "metrics_multiclass_fixed30.json",
        out_dir / "candidate_scores.csv",
        out_dir / "candidate_scores_ranked.csv",
        out_dir / "candidate_manifest.csv",
        out_dir / "candidate_manifest.json",
        out_dir / "selected_model_summary.json",
        out_dir / "selected_model_summary.txt",
        out_dir / "confusion_matrix_test.csv",
        out_dir / "classification_report_test.csv",
        out_dir / "test_predictions.csv",
        out_dir / "feature_importance.csv",
        out_dir / "feature_importance_native.csv",
        out_dir / "feature_importance_permutation.csv",
        out_dir / "feature_importance_skipped.json",
        Path(args.project_dir) / "resultsOptimo" / "multiclass_fixed30_model_comparison.json",
        Path(args.project_dir) / "resultsOptimo" / "multiclass_fixed30_model_comparison.csv",
    ]:
        if Path(p).exists():
            print(f"  {p}", flush=True)


# ----------------------- CLI ------------------------

def _default_label_prefix(dataset: str) -> str:
    if dataset.lower() == "torpeda":
        return "TORPEDA"
    if dataset.lower() == "harvard":
        return "HARVARD"
    return "DATASET"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Multiclass fixed30 intra-request trainer for TorpEda or Harvard/SR-BH")
    ap.add_argument("--dataset", default="harvard", choices=["torpeda", "harvard", "table"], help="Dataset to process/train: torpeda or harvard. table is for preloaded tables.")
    ap.add_argument("--project-dir", default=DEFAULT_PROJECT_DIR)
    ap.add_argument("--inputs", nargs="*", default=None, help="Input directory/files/globs. TorpEda: XML. Harvard: CSV/TSV/CSV.GZ/TSV.GZ.")
    ap.add_argument("--table", default=None, help="Existing table path for --dataset table.")
    ap.add_argument("--label-col", default=LABEL_COL)
    ap.add_argument("--label-prefix", default=None, help="Default: TORPEDA for torpeda, HARVARD for harvard.")
    ap.add_argument("--output-dir", default=None, help="Default: PROJECT_DIR/resultsOptimo/{dataset}/multiclass")
    ap.add_argument("--processed-out", default=None, help="Default: PROJECT_DIR/data/processed/{dataset}/{dataset}_features.parquet")
    ap.add_argument("--only-common-labels", action="store_true")
    ap.add_argument("--keep-absolute-uri", action="store_true")
    ap.add_argument("--sample-n", type=int, default=0)

    # Harvard/SR-BH CSV options.
    ap.add_argument("--sep", default="auto", help="Harvard CSV/TSV delimiter: auto, ',', '\\t', ';', '|'.")
    ap.add_argument("--method-col", default=RAW_METHOD_COL)
    ap.add_argument("--uri-col", default=RAW_URI_COL)
    ap.add_argument("--body-col", default=RAW_BODY_COL)
    ap.add_argument("--normal-col", default="000 - Normal")
    ap.add_argument("--label-cols", default="", help="Optional comma-separated Harvard label columns. Default: autodetect columns like '66 - SQL Injection'.")
    ap.add_argument(
        "--harvard-label-mode",
        default="optimized-family",
        choices=["optimized-family", "optimized_family", "optimized", "coarse", "coarse-family", "coarse_family", "family", "mapped", "attack-family", "attack_families", "native", "capec", "srbh", "legacy-common-anomalous", "legacy_common_anomalous"],
        help=(
            "Harvard labels. optimized-family=default recommended for best Harvard results without HARVARD-ANOMALOUS; "
            "family=more granular readable CAPEC families; native/capec keeps CAPEC labels; legacy-common-anomalous reproduces the old collapse mode."
        ),
    )
    ap.add_argument("--harvard-multiclass-strategy", default="severity", choices=["severity", "first"], help="How to choose one multiclass label when Harvard rows have multiple CAPEC labels.")

    # Dataset optimization/filtering.
    ap.add_argument("--min-class-count", type=int, default=0, help="Drop classes with fewer than this many rows before split. Useful for Harvard classes that cannot be stratified.")
    ap.add_argument("--keep-labels-regex", default="", help="Optional regex: keep only labels matching it.")
    ap.add_argument("--drop-labels-regex", default="", help="Optional regex: drop labels matching it.")
    ap.add_argument("--max-normal-rows", type=int, default=0, help="If >0, downsample NORMAL to this many rows after label filtering. Default keeps all NORMAL.")

    # Train/eval options.
    ap.add_argument("--test-size", type=float, default=0.20)
    ap.add_argument("--val-size", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--preset", default="strong", choices=["fast", "strong", "max"])
    ap.add_argument("--use-gpu", default="auto", choices=["auto", "on", "off"])
    ap.add_argument("--metric-select", default="f1_macro", choices=["f1_macro", "f1_weighted", "balanced_accuracy", "accuracy"])
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--xgb-cpu-fallback", action="store_true")
    ap.add_argument("--sample-weight-mode", default="balanced", choices=["none", "balanced", "sqrt_balanced", "balanced_clipped", "sqrt_clipped", "cbrt_balanced"], help="Default sample weighting if --candidate-weight-modes is empty.")
    ap.add_argument("--candidate-weight-modes", default="", help="Comma-separated modes to try per direct model, e.g. none,sqrt_balanced,balanced_clipped.")
    ap.add_argument("--sample-weight-clip", type=float, default=20.0)
    ap.add_argument("--normal-threshold-grid", default="", help="Optional threshold grid for NORMAL override, e.g. 0.02:0.98:0.01. Empty/off disables.")
    ap.add_argument("--add-two-stage", action="store_true", help="Add two-stage normal-vs-attack + attack multiclass XGBoost candidates.")
    ap.add_argument("--two-stage-only", action="store_true", help="Train only two-stage candidates plus no direct models.")
    ap.add_argument("--two-stage-binary-weight-mode", default="sqrt_balanced")
    ap.add_argument("--two-stage-attack-weight-mode", default="balanced_clipped")
    ap.add_argument("--skip-logreg", action="store_true", help="Skip LogisticRegression; useful for Harvard where it is slow and underperforms.")
    ap.add_argument("--skip-histgb", action="store_true")
    ap.add_argument("--skip-extra-trees", action="store_true")
    ap.add_argument("--skip-random-forest", action="store_true")
    ap.add_argument("--require-candidate-families", default="", help="Comma-separated model families that must be present in the candidate menu before training, e.g. xgboost_d3,xgboost_d5,xgboost_d7,hist_gradient_boosting,extra_trees,logistic_regression.")
    ap.add_argument("--checkpoint-candidate-scores", action="store_true", help="Write candidate_scores.csv after every candidate, so long jobs keep partial results.")
    ap.add_argument("--resume-candidate-scores", action="store_true", help="Reuse completed rows from an existing candidate_scores.csv and continue missing candidates.")

    # Feature importance.
    ap.add_argument("--fi-kind", default="both", choices=["native", "builtin", "auto", "permutation", "both", "none"])
    ap.add_argument("--fi-n-repeats", type=int, default=5)
    ap.add_argument("--fi-max-rows", type=int, default=10000)
    ap.add_argument("--fi-scoring", default="f1_macro")
    ap.add_argument("--fi-n-jobs", type=int, default=1)
    ap.add_argument("--skip-feature-importance", action="store_true", help="Skip all FI computation, including native/permutation/univariate, and write only a skipped marker.")
    ap.add_argument("--clean-output-dir", "--replace-output", dest="clean_output_dir", action="store_true")

    args = ap.parse_args(argv)
    if args.label_prefix is None:
        args.label_prefix = _default_label_prefix(args.dataset)
    if args.output_dir is None:
        args.output_dir = f"{args.project_dir}/resultsOptimo/{args.dataset}/multiclass"
    if args.processed_out is None:
        args.processed_out = f"{args.project_dir}/data/processed/{args.dataset}/{args.dataset}_features.parquet"
    if bool(getattr(args, "skip_feature_importance", False)):
        args.fi_kind = "none"
        args.fi_n_repeats = 0
        args.fi_max_rows = 0
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    print("[START] train_waf_multiclass_fixed30_resultsOptimo_sbatch_v12_noFI_allTorpedaModelsHarvard.py", flush=True)
    print("[INFO] Arguments:", json.dumps(vars(args), ensure_ascii=False, indent=2), flush=True)
    if args.use_gpu == "off":
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    if os.environ.get("REQUIRE_XGBOOST", "0") == "1" and not _xgb_available():
        raise SystemExit("[ERROR] REQUIRE_XGBOOST=1 pero xgboost no está instalado en este Python/.venv")
    df = load_dataset(args)
    print(f"[INFO] Dataset ready: rows={len(df)} cols={len(df.columns)}", flush=True)
    train_eval(args, df)


if __name__ == "__main__":
    main(sys.argv[1:])
