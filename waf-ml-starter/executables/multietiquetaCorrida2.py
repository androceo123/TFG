#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entrenamiento DEFINITIVO multietiqueta WAF-ML para Harvard/SR-BH.

- Procesa el dataset crudo Harvard/SR-BH desde CSV/TSV/GZ.
- Conserva las 30 features anteriores y añade 60 señales HTTP request-only (90 total).
- Entrena un clasificador binario por etiqueta de ataque.
- Mantiene XGBoost One-vs-Rest y ajusta hiperparámetros/peso por etiqueta.
- Usa XGBoost CUDA cuando se solicita y está disponible; en c3 puede operar sólo con CPU.
- Separa train, tune, calibration y test bloqueado; optimiza los umbrales sólo en calibration.
- Guarda todos los artefactos bajo multietiquetaCorrida2 por defecto.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import html
import ipaddress
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import warnings
import xml.etree.ElementTree as ET
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import parse_qsl, unquote_plus, urlsplit

# Evita sobre-suscripción extrema de OpenMP/BLAS si se ejecuta directo sin el .sh.
# Debe fijarse antes de importar numpy/scikit-learn para que tenga efecto.
_DEFAULT_THREADS = str(max(1, min(16, os.cpu_count() or 1)))
for _var in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ.setdefault(_var, _DEFAULT_THREADS)

import joblib
import numpy as np
import pandas as pd
from pandas.errors import ParserError
from tqdm import tqdm

from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
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
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_sample_weight

try:  # Opcional: mejor medición de RSS actual si está instalado.
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None

try:  # Unix/Linux: RSS pico y CPU del proceso.
    import resource
except Exception:  # pragma: no cover
    resource = None


RAW_METHOD_COL = "request_http_method"
RAW_URI_COL = "request_http_request"
RAW_PROTOCOL_COL = "request_http_protocol"
RAW_BODY_COL = "request_body"
RAW_HEADERS_JSON_COL = "request_headers_json"
LABEL_COL = "label_multiclass"
DEFAULT_PROJECT_DIR = "/home_data/aroman/TFG/waf-ml-starter"
DEFAULT_RESULTS_DIRNAME = "multietiquetaCorrida2"

COMMON_ATTACKS = ["BufferOverflow", "CRLFi", "FormatString", "LDAPi", "SQLi", "SSI", "XPath", "XSS"]

# Se conservan los mismos 30 nombres y conceptos de la corrida anterior; esta
# revisión corrige la construcción del texto para no inyectar CR/LF ficticios.
BASE_FEATURES_30: List[str] = [
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

# 60 señales nuevas, request-only. No se usan timestamp, IP/puertos ni campos
# response_*: el vector puede calcularse antes de que el servidor responda.
NEW_FEATURES_60: List[str] = [
    # Método + versión HTTP (14)
    "method_is_get",
    "method_is_post",
    "method_is_head",
    "method_is_put",
    "method_is_delete",
    "method_is_patch",
    "method_is_options",
    "method_is_trace",
    "method_is_connect",
    "method_is_other_or_missing",
    "http_version_is_1_0",
    "http_version_is_1_1",
    "http_version_is_2_plus",
    "http_version_is_other_or_missing",
    # Headers de request (20)
    "request_header_present_count",
    "request_header_total_len",
    "request_header_max_value_len",
    "request_header_control_char_count",
    "user_agent_len",
    "user_agent_entropy",
    "user_agent_product_count",
    "referer_present",
    "referer_host_mismatch",
    "host_is_ip_literal",
    "origin_present",
    "origin_host_mismatch",
    "cookie_len",
    "cookie_pair_count",
    "content_type_is_form",
    "content_type_is_json",
    "content_type_is_xml",
    "content_type_is_multipart",
    "accept_item_count",
    "connection_nonstandard",
    # Body + parámetros (18)
    "n_body_params",
    "distinct_param_name_count",
    "duplicate_param_name_count",
    "empty_param_name_count",
    "empty_param_value_count",
    "max_param_name_len",
    "mean_param_value_len",
    "query_body_shared_param_count",
    "abnormal_param_name_count",
    "body_is_json",
    "body_json_valid",
    "json_max_depth",
    "json_key_count",
    "body_is_xml",
    "xml_tag_count",
    "body_is_multipart",
    "multipart_part_count",
    "content_type_body_mismatch",
    # Encoding + normalización (8)
    "plus_encoding_count",
    "backslash_hex_escape_count",
    "backslash_unicode_escape_count",
    "html_entity_count",
    "null_byte_evasion_count",
    "decode_replacement_char_count",
    "decode_once_length_delta_ratio",
    "decode_twice_additional_delta_ratio",
]

FIXED_FEATURES: List[str] = BASE_FEATURES_30 + NEW_FEATURES_60
FEATURE_SCHEMA_VERSION = "sr-bh-request90-v2"
FEATURE_SCHEMA_SHA256 = hashlib.sha256(
    (FEATURE_SCHEMA_VERSION + "\n" + "\n".join(FIXED_FEATURES)).encode("utf-8")
).hexdigest()
REQUEST_FINGERPRINT_VERSION = "request-canonical-blake2b128-v1"
REQUEST_FINGERPRINT_COL = "request_fingerprint"
DUPLICATE_GROUP_SIZE_COL = "duplicate_group_size"
assert len(BASE_FEATURES_30) == 30, len(BASE_FEATURES_30)
assert len(NEW_FEATURES_60) == 60, len(NEW_FEATURES_60)
assert len(FIXED_FEATURES) == 90, len(FIXED_FEATURES)

SRBH_HEADER_COLUMN_MAP: Dict[str, str] = {
    "request_user_agent": "user-agent",
    "request_referer": "referer",
    "request_host": "host",
    "request_origin": "origin",
    "request_cookie": "cookie",
    "request_content_type": "content-type",
    "request_accept": "accept",
    "request_accept_language": "accept-language",
    "request_accept_encoding": "accept-encoding",
    "request_do_not_track": "dnt",
    "request_connection": "connection",
}

CORE_COLUMN_ALIASES: Dict[str, Tuple[str, ...]] = {
    RAW_METHOD_COL: ("http_method", "request_method", "method", "verb"),
    RAW_URI_COL: ("request_target", "request_uri", "request_url", "http_request", "uri", "url", "path_query"),
    RAW_PROTOCOL_COL: ("request_protocol", "http_protocol", "http_version", "request_http_version", "protocol", "version"),
    RAW_BODY_COL: ("http_body", "body", "payload", "request_payload", "post_data"),
    RAW_HEADERS_JSON_COL: ("headers_json", "request_headers", "http_headers", "headers"),
}

HEADER_COLUMN_ALIASES: Dict[str, Tuple[str, ...]] = {
    "request_user_agent": ("user_agent", "http_user_agent", "header_user_agent"),
    "request_referer": ("request_referrer", "referer", "referrer", "http_referer"),
    "request_host": ("host", "http_host"),
    "request_origin": ("origin", "http_origin"),
    "request_cookie": ("cookie", "http_cookie"),
    "request_content_type": ("content_type", "http_content_type"),
    "request_accept": ("accept", "http_accept"),
    "request_accept_language": ("accept_language", "http_accept_language"),
    "request_accept_encoding": ("accept_encoding", "http_accept_encoding"),
    "request_do_not_track": ("do_not_track", "dnt"),
    "request_connection": ("connection", "http_connection"),
}

_HTTP_PROTO_RE = re.compile(r"^HTTP/(\d+(?:\.\d+)?)$", re.IGNORECASE)
_REQUEST_LINE_RE = re.compile(
    r"^\s*([A-Za-z][A-Za-z0-9_.-]{0,31})\s+(\S+)\s+((?:HTTP/)?\d+(?:\.\d+)?|h2c?|h3)\s*$",
    re.IGNORECASE,
)
_HEX = r"[0-9a-fA-F]"
_PCT_ENC_RE = re.compile(rf"%{_HEX}{{2}}")
_INVALID_PERCENT_RE = re.compile(rf"%(?!{_HEX}{{2}})")
_DOUBLE_ENC_RE = re.compile(rf"%25{_HEX}{{2}}", re.IGNORECASE)
_TOKEN_RE = re.compile(r"[A-Za-z0-9_%\\x\\u./:-]+")
_UA_PRODUCT_RE = re.compile(r"(?<![A-Za-z0-9_.-])[A-Za-z][A-Za-z0-9_.-]{0,63}/[^\s();]{1,64}")
_ABNORMAL_PARAM_NAME_RE = re.compile(r"[^A-Za-z0-9_.\-\[\]]")
_XML_TAG_RE = re.compile(r"<\s*/?\s*[A-Za-z_][A-Za-z0-9_.:-]*(?:\s|/?>)")
_BACKSLASH_HEX_RE = re.compile(r"\\x[0-9a-fA-F]{2}")
_BACKSLASH_UNICODE_RE = re.compile(r"\\u(?:[0-9a-fA-F]{4}|\{[0-9a-fA-F]{1,8}\})")
_HTML_ENTITY_RE = re.compile(r"&(?:[A-Za-z][A-Za-z0-9]{1,31}|#[0-9]{1,8}|#x[0-9a-fA-F]{1,8});")
_NULL_EVASION_RE = re.compile(
    r"(?:\x00|%00|\\x00|\\u(?:0000|\{0+\})|&#0+;|&#x0+;)",
    re.IGNORECASE,
)
_LABEL_COL_RE = re.compile(r"^\d{1,3}\s+-\s+.+$")
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
        r"%(?:[0-9]+\$)?[-+#0 ]*(?:\d+|\*)?(?:\.(?:\d+|\*)?)?(?:hh|h|ll|l|L|z|j|t)?[diuoxXfFeEgGaAcspn]",
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
    "CAPEC-242": 10, "CAPEC-88": 10, "CAPEC-248": 10,
    "CAPEC-66": 9, "CAPEC-33": 9, "CAPEC-126": 8,
    "CAPEC-34": 7, "CAPEC-272": 6, "CAPEC-274": 6,
    "CAPEC-194": 6, "CAPEC-153": 6, "CAPEC-16": 4, "CAPEC-310": 2,
}

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


@dataclass
class SRBHSchema:
    sep: str = "auto"
    method_col: str = RAW_METHOD_COL
    uri_col: str = RAW_URI_COL
    protocol_col: str = RAW_PROTOCOL_COL
    body_col: str = RAW_BODY_COL
    normal_col: str = "000 - Normal"
    label_cols: Optional[List[str]] = None
    multiclass_strategy: str = "severity"
    severity_map: Optional[Dict[str, int]] = None


# ---------------------------------------------------------------------------
# Utilidades generales
# ---------------------------------------------------------------------------

def _safe_str(v: Any) -> str:
    if v is None:
        return ""
    try:
        missing = pd.isna(v)
        if isinstance(missing, (bool, np.bool_)) and bool(missing):
            return ""
    except Exception:
        pass
    s = str(v)
    return "" if s.strip().lower() in {"nan", "none", "<na>", "nat"} else s


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
        return [
            (str(k), str(v))
            for k, v in parse_qsl(
                query,
                keep_blank_values=True,
                strict_parsing=False,
                max_num_fields=4096,
            )
        ]
    except ValueError:
        # Un payload con miles de campos no debe anular todas las señales. Se
        # conserva un prefijo acotado y el resto queda reflejado en longitudes.
        limited = "&".join(str(query).split("&", 4096)[:4096])
        try:
            return [(str(k), str(v)) for k, v in parse_qsl(limited, keep_blank_values=True, strict_parsing=False)]
        except Exception:
            return []
    except TypeError:  # Python antiguo sin max_num_fields.
        try:
            limited = "&".join(str(query).split("&", 4096)[:4096])
            return [(str(k), str(v)) for k, v in parse_qsl(limited, keep_blank_values=True, strict_parsing=False)]
        except Exception:
            return []
    except Exception:
        return []


def _canonical_header_name(name: Any) -> str:
    s = _safe_str(name).strip().lower().replace("_", "-")
    return re.sub(r"\s+", "-", s).strip("-")


def _flatten_header_value(value: Any) -> str:
    if isinstance(value, (list, tuple, set)):
        return ", ".join(_safe_str(x) for x in value if _safe_str(x).strip())
    # No quitar controles en los extremos: son una señal de evasión. Los
    # consumidores semánticos aplican strip cuando corresponde.
    return _safe_str(value)


def _parse_headers_any(v: Any) -> Dict[str, str]:
    """Acepta JSON dict/list o bloque raw ``Nombre: valor``."""
    if isinstance(v, dict):
        obj: Any = v
    elif isinstance(v, (list, tuple)):
        obj = v
    else:
        s = _safe_str(v).strip()
        if not s or s in {"{}", "[]"}:
            return {}
        try:
            obj = json.loads(s)
        except Exception:
            obj = None
            out: Dict[str, str] = {}
            for line in s.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
                if ":" not in line:
                    continue
                key, value = line.split(":", 1)
                key_n = _canonical_header_name(key)
                if key_n:
                    out[key_n] = _flatten_header_value(value)
            return out

    out: Dict[str, str] = {}
    if isinstance(obj, dict):
        items = list(obj.items())
    elif isinstance(obj, (list, tuple)):
        items: List[Tuple[Any, Any]] = []
        for item in obj:
            if isinstance(item, dict):
                items.extend(item.items())
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                items.append((item[0], item[1]))
    else:
        return {}
    for key, value in items:
        key_n = _canonical_header_name(key)
        if key_n:
            out[key_n] = _flatten_header_value(value)
    return out


def _json_loads_dict(v: Any) -> Dict[str, str]:
    return _parse_headers_any(v)


def _merge_request_headers(headers_raw: Any, individual_values: Sequence[Any]) -> Dict[str, str]:
    headers = _parse_headers_any(headers_raw)
    for (column, header_name), value in zip(SRBH_HEADER_COLUMN_MAP.items(), individual_values):
        del column
        value_s = _safe_str(value)
        if value_s.strip():
            # La columna explícita del dataset prevalece sobre un JSON genérico.
            headers[header_name] = value_s
    return headers


def _content_type(headers: Dict[str, str]) -> str:
    for k, v in headers.items():
        if _canonical_header_name(k) == "content-type":
            return str(v)
    return ""


def _looks_form_like(body_text: str, content_type: str = "") -> bool:
    s = _safe_str(body_text).strip()
    if not s:
        return False
    content_type_l = _safe_str(content_type).lower()
    # Un tipo explícito JSON/XML/multipart prima sobre una coincidencia casual
    # de '=' dentro del payload.
    if (
        "application/json" in content_type_l
        or re.search(r"[+/]json(?:\s*;|\s*$)", content_type_l)
        or "application/xml" in content_type_l
        or "text/xml" in content_type_l
        or re.search(r"[+/]xml(?:\s*;|\s*$)", content_type_l)
        or "multipart/" in content_type_l
    ):
        return False
    if s.startswith(("{", "[", "<")):
        return False
    if "application/x-www-form-urlencoded" in content_type_l:
        return True
    return "=" in s and ("&" in s or len(s.split("=", 1)[0]) <= 128)


def _save_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=_json_default)


def _atomic_json(data: Dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp_{os.getpid()}")
    try:
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2, default=_json_default)
        os.replace(tmp, path)
        return path
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass


def _json_default(obj: Any):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        x = float(obj)
        return None if math.isnan(x) or math.isinf(x) else x
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def _safe_float(v: Any) -> Optional[float]:
    try:
        x = float(v)
        return None if np.isnan(x) or np.isinf(x) else x
    except Exception:
        return None


def _safe_div(num: float, den: float) -> Optional[float]:
    if den == 0:
        return None
    return float(num) / float(den)


def _now_iso() -> str:
    import datetime as _dt
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _get_peak_rss_mb() -> Optional[float]:
    if resource is None:
        return None
    try:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux: KB. macOS/BSD: bytes. En el cluster Linux normalmente es KB.
        if sys.platform == "darwin":
            return float(rss) / (1024.0 * 1024.0)
        return float(rss) / 1024.0
    except Exception:
        return None


def _get_current_rss_mb() -> Optional[float]:
    if psutil is None:
        return None
    try:
        return float(psutil.Process(os.getpid()).memory_info().rss) / (1024.0 * 1024.0)
    except Exception:
        return None


def _gpu_visible_from_env() -> bool:
    env_names = ["CUDA_VISIBLE_DEVICES", "SLURM_JOB_GPUS", "SLURM_STEP_GPUS", "NVIDIA_VISIBLE_DEVICES"]
    bad_values = {"", "-1", "none", "void", "nodevfiles", "no_dev_files", "unset", "null"}
    for name in env_names:
        val = str(os.environ.get(name, "")).strip()
        if val and val.lower() not in bad_values:
            return True
    return False


def _has_nvidia_smi() -> bool:
    try:
        return subprocess.run(["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=4).returncode == 0
    except Exception:
        return False


def _gpu_visible_auto() -> bool:
    return _gpu_visible_from_env() or _has_nvidia_smi()


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _save_processed(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    tmp = path.with_name(f".{path.name}.tmp_{os.getpid()}{suffix or '.tmp'}")
    try:
        if suffix == ".parquet":
            try:
                df.to_parquet(tmp, index=False)
            except Exception as e:
                fallback = path.with_suffix(".csv")
                print(f"[WARN] No pude escribir parquet ({type(e).__name__}: {e}); guardo CSV en {fallback}", flush=True)
                fallback_tmp = fallback.with_name(f".{fallback.name}.tmp_{os.getpid()}.csv")
                try:
                    df.to_csv(fallback_tmp, index=False)
                    os.replace(fallback_tmp, fallback)
                    return fallback
                finally:
                    try:
                        if fallback_tmp.exists():
                            fallback_tmp.unlink()
                    except Exception:
                        pass
        elif suffix == ".csv":
            df.to_csv(tmp, index=False)
        else:
            raise ValueError(f"Extensión no soportada para processed: {path}; usar .parquet o .csv")
        os.replace(tmp, path)
        return path
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass


def _load_processed(path: Path) -> pd.DataFrame:
    """Carga un processed generado por _save_processed, sea parquet o CSV fallback."""
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Extensión no soportada para processed/resume: {path}")


def _atomic_joblib_dump(obj: Any, path: Path, *, compress: int = 3) -> Path:
    """Escritura atómica para checkpoints: evita dejar .joblib corrupto si el job cae justo al guardar."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp_{os.getpid()}")
    try:
        joblib.dump(obj, tmp, compress=int(compress))
        os.replace(tmp, path)
        return path
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass


def _atomic_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp_{os.getpid()}.csv")
    try:
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)
        return path
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass


def _resume_enabled(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "resume", True))


def _checkpoint_root(args: argparse.Namespace, out_dir: Path) -> Path:
    custom = str(getattr(args, "checkpoint_dir", "") or "").strip()
    return Path(custom) if custom else (out_dir / "checkpoints")


def _current_input_signature(args: argparse.Namespace) -> str:
    cached_input_signature = getattr(args, "_input_signature_cache", None)
    if cached_input_signature is None:
        try:
            current_inputs = _collect_inputs(args.inputs or args.harvard_inputs or [], dataset="harvard")
            cached_input_signature = _input_signature(current_inputs)
        except Exception:
            cached_input_signature = "unavailable"
        setattr(args, "_input_signature_cache", cached_input_signature)
    return str(cached_input_signature)


def _preprocess_signature(args: argparse.Namespace) -> str:
    keys = [
        "sample_n", "seed", "sep", "method_col", "uri_col", "protocol_col", "body_col",
        "normal_col", "label_cols", "label_mode", "min_positive_count",
        "max_negative_positive_ratio", "max_normal_rows", "keep_labels_regex", "drop_labels_regex",
        "drop_invalid_rows", "deduplicate_requests",
    ]
    payload = {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "request_fingerprint_version": REQUEST_FINGERPRINT_VERSION,
        "input_signature": _current_input_signature(args),
        "args": {key: getattr(args, key, None) for key in keys},
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _training_signature(args: argparse.Namespace) -> str:
    keys = [
        "seed", "model_backend", "use_gpu", "search_level", "candidate_weight_modes",
        "selection_metric", "xgb_preset", "xgb_early_stopping_rounds",
        "xgb_per_label_tuning", "xgb_tuning_trials", "xgb_weak_label_trials",
        "xgb_weak_label_regex", "xgb_label_selection_metric", "scale_pos_weight_cap",
        "threshold_objective", "threshold_min", "threshold_max", "threshold_min_recall",
        "min_attack_score", "test_size", "valid_size", "calibration_share",
        "min_calibration_positives", "group_split_by_request", "refit_full_after_thresholds",
        "fi_n_repeats", "fi_max_rows", "fi_scoring", "operational_max_rows",
    ]
    payload = {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "preprocess_signature": _preprocess_signature(args),
        "args": {key: getattr(args, key, None) for key in keys},
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _calibration_signature(args: argparse.Namespace) -> str:
    """Firma separada: cambiar targets no invalida estimadores/checkpoints."""
    keys = [
        "joint_threshold_optimization", "target_label_f1", "target_normal",
        "target_normal_metric", "joint_threshold_max_passes", "min_calibration_positives",
        "threshold_objective", "threshold_min", "threshold_max", "threshold_min_recall",
        "min_attack_score",
    ]
    payload = {
        "training_signature": _training_signature(args),
        "args": {key: getattr(args, key, None) for key in keys},
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _input_signature(paths: Sequence[str]) -> str:
    manifest: List[Dict[str, Any]] = []
    for raw_path in sorted(map(str, paths)):
        path = Path(raw_path)
        try:
            stat = path.stat()
            manifest.append({"path": str(path.resolve()), "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)})
        except OSError:
            manifest.append({"path": str(path), "missing": True})
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode("utf-8")).hexdigest()


def _candidate_id(backend: str, weight_mode: str) -> str:
    return f"{_safe_name(backend)}__w_{_safe_name(weight_mode)}"


def _candidate_checkpoint_path(args: argparse.Namespace, out_dir: Path, backend: str, weight_mode: str) -> Path:
    return _checkpoint_root(args, out_dir) / "candidates" / f"{_candidate_id(backend, weight_mode)}.joblib"


def _candidate_to_summary(candidate: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "backend": candidate.get("backend"),
        "weight_mode": candidate.get("weight_mode"),
        "fit_status": candidate.get("fit_status", "ok"),
        "selection_metric": candidate.get("selection_metric"),
        "selection_score": candidate.get("selection_score"),
        "gpu_used_any": candidate.get("gpu_used_any"),
        "validation_metrics": candidate.get("validation_metrics") or {},
        "train_select_seconds": candidate.get("train_select_seconds"),
        **({"fit_error": candidate.get("fit_error")} if candidate.get("fit_error") else {}),
        **({"resumed_from_checkpoint": True} if candidate.get("resumed_from_checkpoint") else {}),
    }


def _write_candidate_progress(out_dir: Path, summaries: Sequence[Dict[str, Any]]) -> None:
    """Escribe progreso incremental legible mientras corre la búsqueda."""
    if not out_dir:
        return
    rows = list(summaries)
    if not rows:
        return
    _atomic_csv(pd.DataFrame(rows), out_dir / "candidate_validation_summary.partial.csv")
    _save_json({"created_at": _now_iso(), "candidates": rows}, out_dir / "candidate_validation_summary.partial.json")


def _load_candidate_checkpoint(
    args: argparse.Namespace,
    out_dir: Path,
    backend: str,
    weight_mode: str,
    labels: Sequence[str],
) -> Optional[Dict[str, Any]]:
    if not _resume_enabled(args):
        return None
    path = _candidate_checkpoint_path(args, out_dir, backend, weight_mode)
    if not path.exists():
        return None
    try:
        payload = joblib.load(path)
        candidate = payload.get("candidate") if isinstance(payload, dict) else None
        if not isinstance(candidate, dict):
            raise ValueError("checkpoint sin candidate dict")
        if str(candidate.get("backend")) != str(backend) or str(candidate.get("weight_mode")) != str(weight_mode):
            raise ValueError("checkpoint no coincide con backend/weight_mode")
        if list(payload.get("labels") or []) != list(labels):
            raise ValueError("checkpoint no coincide con target_labels")
        if list(payload.get("features") or []) != list(FIXED_FEATURES) or payload.get("feature_schema_sha256") != FEATURE_SCHEMA_SHA256:
            raise ValueError("checkpoint no coincide con request90/feature schema")
        if payload.get("training_signature") != _training_signature(args):
            raise ValueError("checkpoint no coincide con configuración de tuning/split")
        if "records" not in candidate or not candidate.get("records"):
            raise ValueError("checkpoint sin records entrenados")
        candidate = dict(candidate)
        candidate["resumed_from_checkpoint"] = True
        candidate["checkpoint_path"] = str(path)
        return candidate
    except Exception as e:
        print(f"[WARN] No pude reutilizar checkpoint {path}: {type(e).__name__}: {e}. Reentreno candidato.", flush=True)
        return None


def _save_candidate_checkpoint(
    args: argparse.Namespace,
    out_dir: Path,
    candidate: Dict[str, Any],
    labels: Sequence[str],
) -> Optional[Path]:
    if not _resume_enabled(args):
        return None
    path = _candidate_checkpoint_path(args, out_dir, str(candidate["backend"]), str(candidate["weight_mode"]))
    payload = {
        "kind": "multilabel_candidate_checkpoint",
        "created_at": _now_iso(),
        "script": Path(__file__).name,
        "feature_count": len(FIXED_FEATURES),
        "features": FIXED_FEATURES,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "request_fingerprint_version": REQUEST_FINGERPRINT_VERSION,
        "training_signature": _training_signature(args),
        "labels": list(labels),
        "selection_metric": str(getattr(args, "selection_metric", "")),
        "candidate": candidate,
    }
    try:
        _atomic_joblib_dump(payload, path, compress=int(getattr(args, "checkpoint_compress", 3)))
        return path
    except Exception as e:
        print(f"[WARN] No pude guardar checkpoint de candidato {candidate.get('backend')}/{candidate.get('weight_mode')}: {type(e).__name__}: {e}", flush=True)
        return None


def _final_refit_checkpoint_path(args: argparse.Namespace, out_dir: Path, backend: str, weight_mode: str) -> Path:
    return _checkpoint_root(args, out_dir) / "final" / f"final_refit__{_candidate_id(backend, weight_mode)}.joblib"


def _label_record_checkpoint_path(args: argparse.Namespace, out_dir: Path, phase: str, backend: str, weight_mode: str, label_index: int, label: str) -> Path:
    return (
        _checkpoint_root(args, out_dir)
        / "label_records"
        / _safe_name(phase)
        / _candidate_id(backend, weight_mode)
        / f"{int(label_index):04d}__{_safe_name(label)}.joblib"
    )


def _load_label_record_checkpoint(
    args: argparse.Namespace,
    out_dir: Optional[Path],
    phase: str,
    backend: str,
    weight_mode: str,
    label_index: int,
    label: str,
) -> Optional[Dict[str, Any]]:
    if out_dir is None or not _resume_enabled(args) or not phase:
        return None
    path = _label_record_checkpoint_path(args, out_dir, phase, backend, weight_mode, label_index, label)
    if not path.exists():
        return None
    try:
        payload = joblib.load(path)
        rec = payload.get("record") if isinstance(payload, dict) else None
        if not isinstance(rec, dict):
            raise ValueError("checkpoint label sin record")
        if int(payload.get("label_index", -1)) != int(label_index) or str(payload.get("label")) != str(label):
            raise ValueError("checkpoint label no coincide con label/index")
        if str(payload.get("backend")) != str(backend) or str(payload.get("weight_mode")) != str(weight_mode):
            raise ValueError("checkpoint label no coincide con backend/weight_mode")
        if list(payload.get("features") or []) != list(FIXED_FEATURES) or payload.get("feature_schema_sha256") != FEATURE_SCHEMA_SHA256:
            raise ValueError("checkpoint label no coincide con request90/feature schema")
        if payload.get("training_signature") != _training_signature(args):
            raise ValueError("checkpoint label no coincide con configuración de tuning/split")
        rec = dict(rec)
        rec["resumed_from_label_checkpoint"] = True
        rec["label_checkpoint_path"] = str(path)
        return rec
    except Exception as e:
        print(f"[WARN] No pude reutilizar checkpoint de etiqueta {path}: {type(e).__name__}: {e}. Reentreno esa etiqueta.", flush=True)
        return None


def _save_label_record_checkpoint(
    args: argparse.Namespace,
    out_dir: Optional[Path],
    phase: str,
    backend: str,
    weight_mode: str,
    label_index: int,
    label: str,
    record: Dict[str, Any],
) -> Optional[Path]:
    if out_dir is None or not _resume_enabled(args) or not phase:
        return None
    path = _label_record_checkpoint_path(args, out_dir, phase, backend, weight_mode, label_index, label)
    payload = {
        "kind": "multilabel_label_record_checkpoint",
        "created_at": _now_iso(),
        "script": Path(__file__).name,
        "phase": phase,
        "backend": backend,
        "weight_mode": weight_mode,
        "label_index": int(label_index),
        "label": str(label),
        "feature_count": len(FIXED_FEATURES),
        "features": FIXED_FEATURES,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "request_fingerprint_version": REQUEST_FINGERPRINT_VERSION,
        "training_signature": _training_signature(args),
        "record": record,
    }
    try:
        return _atomic_joblib_dump(payload, path, compress=int(getattr(args, "checkpoint_compress", 3)))
    except Exception as e:
        print(f"[WARN] No pude guardar checkpoint de etiqueta {backend}/{weight_mode}/{label}: {type(e).__name__}: {e}", flush=True)
        return None


def _load_final_refit_checkpoint(
    args: argparse.Namespace,
    out_dir: Path,
    best: Dict[str, Any],
    labels: Sequence[str],
) -> Optional[Dict[str, Any]]:
    if not _resume_enabled(args):
        return None
    path = _final_refit_checkpoint_path(args, out_dir, str(best["backend"]), str(best["weight_mode"]))
    if not path.exists():
        return None
    try:
        payload = joblib.load(path)
        if list(payload.get("labels") or []) != list(labels):
            raise ValueError("checkpoint final no coincide con target_labels")
        if list(payload.get("features") or []) != list(FIXED_FEATURES) or payload.get("feature_schema_sha256") != FEATURE_SCHEMA_SHA256:
            raise ValueError("checkpoint final no coincide con request90/feature schema")
        if payload.get("training_signature") != _training_signature(args):
            raise ValueError("checkpoint final no coincide con configuración de tuning/split")
        records = payload.get("records")
        if not records:
            raise ValueError("checkpoint final sin records")
        restored = dict(best)
        restored["records"] = records
        restored["final_refit_seconds"] = float(payload.get("final_refit_seconds") or 0.0)
        restored["gpu_used_any"] = any(bool(r.get("gpu_used")) for r in records)
        restored["resumed_final_refit_from_checkpoint"] = True
        restored["final_refit_checkpoint_path"] = str(path)
        print(f"[RESUME] Reutilizo refit final desde checkpoint: {path}", flush=True)
        return restored
    except Exception as e:
        print(f"[WARN] No pude reutilizar checkpoint final {path}: {type(e).__name__}: {e}. Reentreno refit final.", flush=True)
        return None


def _save_final_refit_checkpoint(
    args: argparse.Namespace,
    out_dir: Path,
    best: Dict[str, Any],
    labels: Sequence[str],
) -> Optional[Path]:
    if not _resume_enabled(args):
        return None
    path = _final_refit_checkpoint_path(args, out_dir, str(best["backend"]), str(best["weight_mode"]))
    payload = {
        "kind": "multilabel_final_refit_checkpoint",
        "created_at": _now_iso(),
        "script": Path(__file__).name,
        "feature_count": len(FIXED_FEATURES),
        "features": FIXED_FEATURES,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "request_fingerprint_version": REQUEST_FINGERPRINT_VERSION,
        "training_signature": _training_signature(args),
        "labels": list(labels),
        "backend": best.get("backend"),
        "weight_mode": best.get("weight_mode"),
        "records": best.get("records"),
        "final_refit_seconds": best.get("final_refit_seconds"),
    }
    try:
        _atomic_joblib_dump(payload, path, compress=int(getattr(args, "checkpoint_compress", 3)))
        return path
    except Exception as e:
        print(f"[WARN] No pude guardar checkpoint final: {type(e).__name__}: {e}", flush=True)
        return None


def _try_resume_processed(out_dir: Path, args: argparse.Namespace) -> Optional[Tuple[pd.DataFrame, Path, List[str], List[Dict[str, Any]], List[str], Dict[str, Any], float, float]]:
    """
    Reutiliza processed_request90_multilabel + run_config si existen.
    Devuelve df_features, processed_path, target_labels, target_meta, inputs, split_info_previo, load_seconds, feature_extraction_seconds.
    """
    if not _resume_enabled(args) or bool(getattr(args, "force_reprocess", False)):
        return None
    config_path = out_dir / "run_config.json"
    if not config_path.exists():
        return None
    try:
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        if (
            cfg.get("feature_schema_version") != FEATURE_SCHEMA_VERSION
            or cfg.get("feature_schema_sha256") != FEATURE_SCHEMA_SHA256
            or list(cfg.get("features") or []) != list(FIXED_FEATURES)
        ):
            raise ValueError("run_config pertenece a otro feature schema; request90 requiere reprocesar")
        current_inputs = _collect_inputs(args.inputs or args.harvard_inputs or [], dataset="harvard")
        if cfg.get("input_signature") != _input_signature(current_inputs):
            raise ValueError("cambiaron los archivos de entrada; no reutilizo processed")
        if cfg.get("preprocess_signature") != _preprocess_signature(args):
            raise ValueError("cambió sample_n/esquema/filtros de preprocesado; no reutilizo processed")
        processed_raw = cfg.get("processed_path") or str(out_dir / "processed_request90_multilabel.parquet")
        processed_path = Path(processed_raw)
        if not processed_path.exists():
            alt = processed_path.with_suffix(".csv")
            if alt.exists():
                processed_path = alt
            else:
                return None
        target_labels = list(cfg.get("target_labels") or [])
        if not target_labels:
            return None
        df_features = _load_processed(processed_path)
        expected_rows = int(cfg.get("processed_rows", -1))
        if expected_rows < 0 or len(df_features) != expected_rows:
            raise ValueError(f"processed truncado/distinto: rows={len(df_features)} expected={expected_rows}")
        target_cols = [_target_col_name(lab) for lab in target_labels]
        required_processed_cols = [
            *FIXED_FEATURES,
            *target_cols,
            REQUEST_FINGERPRINT_COL,
            DUPLICATE_GROUP_SIZE_COL,
        ]
        missing = [c for c in required_processed_cols if c not in df_features.columns]
        if missing:
            raise ValueError(f"processed incompleto; faltan columnas: {missing[:10]}")
        print(f"[RESUME] Reutilizo processed request90 existente: {processed_path}", flush=True)
        print(f"[RESUME] Salteo carga cruda + extracción de features. Filas={len(df_features)} labels={len(target_labels)}", flush=True)
        return (
            df_features,
            processed_path,
            target_labels,
            list(cfg.get("target_meta") or []),
            list(cfg.get("inputs") or []),
            dict(cfg.get("split_info") or {}),
            float(cfg.get("load_seconds") or 0.0),
            float(cfg.get("feature_extraction_seconds") or 0.0),
        )
    except Exception as e:
        print(f"[WARN] No pude resumir processed previo: {type(e).__name__}: {e}. Reproceso desde crudo.", flush=True)
        return None


# ---------------------------------------------------------------------------
# Feature extraction request90
# ---------------------------------------------------------------------------

def _request_line_fields(method: Any, uri: Any, protocol: Any) -> Tuple[str, str, str]:
    method_s = _safe_str(method).strip()
    uri_s = _safe_str(uri).strip()
    protocol_s = _safe_str(protocol).strip()
    match = _REQUEST_LINE_RE.match(uri_s)
    if match:
        if not method_s:
            method_s = match.group(1)
        uri_s = match.group(2)
        if not protocol_s:
            protocol_s = match.group(3)
    return method_s, uri_s, protocol_s


def _http_version_bucket(protocol: str) -> str:
    p = _safe_str(protocol).strip().lower()
    if p in {"h2", "h2c", "http/2", "http/2.0", "2", "2.0", "h3", "http/3", "http/3.0", "3", "3.0"}:
        return "2_plus"
    match = re.fullmatch(r"(?:http/)?(\d+)(?:\.(\d+))?", p)
    if not match:
        return "other"
    major = int(match.group(1))
    minor = int(match.group(2) or 0)
    if major == 1 and minor == 0:
        return "1_0"
    if major == 1 and minor == 1:
        return "1_1"
    if major >= 2:
        return "2_plus"
    return "other"


def _host_from_header(value: Any) -> str:
    raw = _safe_str(value).strip()
    if not raw:
        return ""
    try:
        parsed = urlsplit(raw if "://" in raw else "//" + raw)
        return (parsed.hostname or "").rstrip(".").lower()
    except Exception:
        return ""


def _absolute_url_host(value: Any) -> str:
    raw = _safe_str(value).strip()
    if not raw or raw.lower() == "null":
        return ""
    try:
        parsed = urlsplit(raw)
        if not parsed.scheme or not parsed.netloc:
            return ""
        return (parsed.hostname or "").rstrip(".").lower()
    except Exception:
        return ""


def _host_is_ip(value: Any) -> bool:
    host = _host_from_header(value)
    if not host:
        return False
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def _json_structure_stats(obj: Any, max_nodes: int = 100000) -> Tuple[int, int]:
    """Devuelve (profundidad máxima, total de claves) con recorrido acotado."""
    stack: List[Tuple[Any, int]] = [(obj, 1)]
    max_depth = 0
    key_count = 0
    visited = 0
    while stack and visited < max_nodes:
        current, depth = stack.pop()
        visited += 1
        max_depth = max(max_depth, depth)
        if isinstance(current, dict):
            key_count += len(current)
            stack.extend((v, depth + 1) for v in current.values())
        elif isinstance(current, (list, tuple)):
            stack.extend((v, depth + 1) for v in current)
    return int(max_depth), int(key_count)


def _multipart_stats(body: str, content_type: str) -> Tuple[bool, int, str]:
    body_s = _safe_str(body)
    ct = _safe_str(content_type)
    ct_multipart = "multipart/" in ct.lower()
    match = re.search(r"boundary\s*=\s*(?:\"([^\"]{1,200})\"|([^;\s]{1,200}))", ct, re.IGNORECASE)
    boundary = (match.group(1) or match.group(2)).strip() if match else ""
    normalized = body_s[:1048576].replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.split("\n")
    if not boundary:
        first_line = lines[0].strip() if lines else ""
        if first_line.startswith("--") and 2 < len(first_line) <= 202:
            candidate = first_line[2:]
            opener = "--" + candidate
            closer = opener + "--"
            stripped_lines = [line.strip() for line in lines]
            opener_count = sum(line == opener for line in stripped_lines)
            has_closer = closer in stripped_lines
            has_disposition = "content-disposition:" in normalized[:262144].lower()
            # Sin Content-Type, una sola línea '--foo' no basta para declarar
            # multipart: exigimos estructura repetida/cierre o una cabecera de parte.
            if (has_disposition and opener_count >= 1) or opener_count >= 2 or has_closer:
                boundary = candidate
    part_count = 0
    marker_found = False
    if boundary:
        opener = "--" + boundary
        closer = opener + "--"
        for line in lines:
            stripped = line.strip()
            if stripped in {opener, closer}:
                marker_found = True
            if stripped == opener:
                part_count += 1
    lexical = bool(marker_found and (ct_multipart or part_count > 0))
    return lexical, int(part_count), boundary


def _request_header_control_chars(headers: Dict[str, str]) -> int:
    return int(sum(1 for value in headers.values() for ch in value if (ord(ch) < 32 and ch != "\t") or ord(ch) == 127))


def extract_fixed_features(
    method: str,
    uri: str,
    headers: Optional[Dict[str, str]] = None,
    body: Any = None,
    protocol: str = "",
) -> Dict[str, float]:
    """Devuelve exactamente 90 features request-only en el orden FIXED_FEATURES."""
    method_raw, uri_raw, protocol_raw = _request_line_fields(method, uri, protocol)
    if isinstance(body, bytes):
        body_raw = body.decode("utf-8", errors="ignore")
        body_len = len(body)
    else:
        body_raw = _safe_str(body)
        body_len = len(body_raw.encode("utf-8", errors="ignore"))

    uri_dec1, uri_dec2 = _decode_once(uri_raw), _decode_twice(uri_raw)
    body_dec1, body_dec2 = _decode_once(body_raw), _decode_twice(body_raw)
    raw_combined = f"{uri_raw} {body_raw}"
    # Separadores neutros: no se inyectan CR/LF artificiales en crlf_token_count.
    analysis_text = " ".join([uri_raw, body_raw, uri_dec1, body_dec1, uri_dec2, body_dec2])

    try:
        parts = urlsplit(uri_raw)
        path = parts.path or ""
        query = parts.query or (uri_raw.split("?", 1)[1] if "?" in uri_raw else "")
    except ValueError:
        # Autoridades IPv6 u otras URI malformadas no deben abortar un job de
        # horas. Conservamos las señales léxicas y separamos query manualmente.
        path, sep, query = uri_raw.partition("?")
        if not sep:
            query = ""
    q_params = _parse_qs(query)
    headers_d = {_canonical_header_name(k): _flatten_header_value(v) for k, v in (headers or {}).items() if _canonical_header_name(k)}
    content_type = _content_type(headers_d).strip()
    content_type_l = content_type.lower()
    b_params = _parse_qs(body_raw) if _looks_form_like(body_raw, content_type) else []
    values = [v for _, v in (q_params + b_params)]
    all_params = q_params + b_params
    param_names = [k for k, _ in all_params]
    param_name_counts = Counter(param_names)

    method_u = method_raw.upper()
    known_methods = ["GET", "POST", "HEAD", "PUT", "DELETE", "PATCH", "OPTIONS", "TRACE", "CONNECT"]
    version_bucket = _http_version_bucket(protocol_raw)

    header_values = [str(v).strip() for v in headers_d.values() if str(v).strip()]
    user_agent = headers_d.get("user-agent", "")
    referer = headers_d.get("referer", "")
    host = headers_d.get("host", "")
    origin = headers_d.get("origin", "")
    cookie = headers_d.get("cookie", "")
    accept = headers_d.get("accept", "")
    connection = headers_d.get("connection", "")
    host_name = _host_from_header(host)
    referer_host = _absolute_url_host(referer)
    origin_host = _absolute_url_host(origin)

    ct_form = "application/x-www-form-urlencoded" in content_type_l
    ct_json = "application/json" in content_type_l or bool(re.search(r"[+/]json(?:\s*;|\s*$)", content_type_l))
    ct_xml = (
        "application/xml" in content_type_l
        or "text/xml" in content_type_l
        or bool(re.search(r"[+/]xml(?:\s*;|\s*$)", content_type_l))
    )
    ct_multipart = "multipart/" in content_type_l

    stripped_body = body_raw.lstrip()
    body_json_lexical = stripped_body.startswith(("{", "["))
    json_candidate = bool(body_json_lexical or ct_json)
    json_valid = False
    json_depth = 0
    json_keys = 0
    if json_candidate and body_raw and len(body_raw.encode("utf-8", errors="ignore")) <= 1048576:
        try:
            parsed_json = json.loads(body_raw)
            json_valid = True
            json_depth, json_keys = _json_structure_stats(parsed_json)
        except (ValueError, TypeError, RecursionError, MemoryError):
            pass

    body_xml_lexical = stripped_body.startswith("<")
    xml_tag_count = len(_XML_TAG_RE.findall(body_raw[:1048576])) if body_xml_lexical or ct_xml else 0
    multipart_lexical, multipart_parts, multipart_boundary = _multipart_stats(body_raw, content_type)
    body_multipart = bool(ct_multipart or multipart_lexical)

    mismatch = False
    if body_raw:
        if ct_json and not json_valid:
            mismatch = True
        elif ct_xml and not body_xml_lexical:
            mismatch = True
        elif ct_multipart and (not multipart_boundary or multipart_parts <= 0):
            mismatch = True
        elif ct_form and (body_json_lexical or body_xml_lexical or multipart_lexical):
            mismatch = True

    decoded_den = max(1, len(uri_raw) + len(body_raw))
    replacement_count = max(0, uri_dec1.count("\ufffd") - uri_raw.count("\ufffd")) + max(
        0, body_dec1.count("\ufffd") - body_raw.count("\ufffd")
    )

    uri_len = len(uri_raw)
    body_text_len = len(body_raw)
    tokens = _TOKEN_RE.findall(analysis_text)

    feats: Dict[str, float] = {
        # Base 30
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
        # Método + versión HTTP (14)
        **{f"method_is_{name.lower()}": float(method_u == name) for name in known_methods},
        "method_is_other_or_missing": float(method_u not in known_methods),
        "http_version_is_1_0": float(version_bucket == "1_0"),
        "http_version_is_1_1": float(version_bucket == "1_1"),
        "http_version_is_2_plus": float(version_bucket == "2_plus"),
        "http_version_is_other_or_missing": float(version_bucket == "other"),
        # Headers (20)
        "request_header_present_count": float(len(header_values)),
        "request_header_total_len": float(sum(len(v) for v in header_values)),
        "request_header_max_value_len": float(max((len(v) for v in header_values), default=0)),
        "request_header_control_char_count": float(_request_header_control_chars(headers_d)),
        "user_agent_len": float(len(user_agent)),
        "user_agent_entropy": float(_entropy(user_agent)),
        "user_agent_product_count": float(len(_UA_PRODUCT_RE.findall(user_agent))),
        "referer_present": float(bool(referer)),
        "referer_host_mismatch": float(bool(referer_host and host_name and referer_host != host_name)),
        "host_is_ip_literal": float(_host_is_ip(host)),
        "origin_present": float(bool(origin.strip()) and origin.strip().lower() != "null"),
        "origin_host_mismatch": float(bool(origin_host and host_name and origin_host != host_name)),
        "cookie_len": float(len(cookie)),
        "cookie_pair_count": float(len([x for x in cookie.split(";") if x.strip()])) if cookie else 0.0,
        "content_type_is_form": float(ct_form),
        "content_type_is_json": float(ct_json),
        "content_type_is_xml": float(ct_xml),
        "content_type_is_multipart": float(ct_multipart),
        "accept_item_count": float(len([x for x in accept.split(",") if x.strip()])) if accept else 0.0,
        "connection_nonstandard": float(
            bool(connection)
            and any(token.strip().lower() not in {"keep-alive", "close", "upgrade"} for token in connection.split(",") if token.strip())
        ),
        # Body + parámetros (18)
        "n_body_params": float(len(b_params)),
        "distinct_param_name_count": float(len(set(param_names))),
        "duplicate_param_name_count": float(sum(1 for count in param_name_counts.values() if count > 1)),
        "empty_param_name_count": float(sum(1 for name in param_names if name == "")),
        "empty_param_value_count": float(sum(1 for _, value in all_params if value == "")),
        "max_param_name_len": float(max((len(name) for name in param_names), default=0)),
        "mean_param_value_len": float(sum(len(value) for _, value in all_params) / len(all_params)) if all_params else 0.0,
        "query_body_shared_param_count": float(len({k for k, _ in q_params} & {k for k, _ in b_params})),
        "abnormal_param_name_count": float(sum(1 for name in param_names if name and _ABNORMAL_PARAM_NAME_RE.search(name))),
        "body_is_json": float(body_json_lexical),
        "body_json_valid": float(json_valid),
        "json_max_depth": float(json_depth),
        "json_key_count": float(json_keys),
        "body_is_xml": float(body_xml_lexical),
        "xml_tag_count": float(xml_tag_count),
        "body_is_multipart": float(body_multipart),
        "multipart_part_count": float(multipart_parts),
        "content_type_body_mismatch": float(mismatch),
        # Encoding + normalización (8)
        "plus_encoding_count": float(query.count("+") + body_raw.count("+")),
        "backslash_hex_escape_count": float(len(_BACKSLASH_HEX_RE.findall(raw_combined))),
        "backslash_unicode_escape_count": float(len(_BACKSLASH_UNICODE_RE.findall(raw_combined))),
        "html_entity_count": float(len(_HTML_ENTITY_RE.findall(raw_combined))),
        "null_byte_evasion_count": float(len(_NULL_EVASION_RE.findall(raw_combined))),
        "decode_replacement_char_count": float(replacement_count),
        "decode_once_length_delta_ratio": float(
            (abs(len(uri_raw) - len(uri_dec1)) + abs(len(body_raw) - len(body_dec1))) / decoded_den
        ),
        "decode_twice_additional_delta_ratio": float(
            (abs(len(uri_dec1) - len(uri_dec2)) + abs(len(body_dec1) - len(body_dec2))) / decoded_den
        ),
    }
    missing = [feature for feature in FIXED_FEATURES if feature not in feats]
    if missing:
        raise AssertionError(f"Extractor request90 incompleto: {missing}")
    return {feature: float(feats[feature]) if math.isfinite(float(feats[feature])) else 0.0 for feature in FIXED_FEATURES}


def _extract_features_from_tuple(row_tuple: Tuple[Any, ...]) -> List[float]:
    expected = 5 + len(SRBH_HEADER_COLUMN_MAP)
    if len(row_tuple) != expected:
        raise ValueError(f"Tuple request90 inválida: len={len(row_tuple)} expected={expected}")
    method, uri, protocol, headers_json, body, *individual_headers = row_tuple
    headers = _merge_request_headers(headers_json, individual_headers)
    d = extract_fixed_features(method, uri, headers, body, protocol)
    return [float(d[f]) for f in FIXED_FEATURES]


def _feature_input_columns() -> List[str]:
    return [
        RAW_METHOD_COL,
        RAW_URI_COL,
        RAW_PROTOCOL_COL,
        RAW_HEADERS_JSON_COL,
        RAW_BODY_COL,
        *SRBH_HEADER_COLUMN_MAP.keys(),
    ]


def _feature_input_tuples(df: pd.DataFrame) -> Iterable[Tuple[Any, ...]]:
    input_columns = _feature_input_columns()
    work = df
    for column in input_columns:
        if column not in work.columns:
            work[column] = ""
    return work[input_columns].itertuples(index=False, name=None)


def _canonical_protocol_for_fingerprint(protocol: Any) -> str:
    raw = _safe_str(protocol).strip()
    low = raw.lower()
    if low in {"h2", "h2c", "http/2", "http/2.0", "2", "2.0"}:
        return "HTTP/2"
    if low in {"h3", "http/3", "http/3.0", "3", "3.0"}:
        return "HTTP/3"
    match = re.fullmatch(r"(?:http/)?(\d+)(?:\.(\d+))?", low)
    if match:
        return f"HTTP/{int(match.group(1))}.{int(match.group(2) or 0)}"
    return raw.upper()


def _fingerprint_update(hasher: Any, value: Any) -> None:
    payload = _safe_str(value).encode("utf-8", errors="surrogatepass")
    hasher.update(len(payload).to_bytes(8, byteorder="big", signed=False))
    hasher.update(payload)


def _request_fingerprint_from_tuple(row_tuple: Tuple[Any, ...]) -> str:
    """Hash estable de la request; excluye respuesta, timestamp, IP y puertos."""
    expected = 5 + len(SRBH_HEADER_COLUMN_MAP)
    if len(row_tuple) != expected:
        raise ValueError(f"Tuple fingerprint inválida: len={len(row_tuple)} expected={expected}")
    method, uri, protocol, headers_json, body, *individual_headers = row_tuple
    method_s, uri_s, protocol_s = _request_line_fields(method, uri, protocol)
    headers = _merge_request_headers(headers_json, individual_headers)

    # BLAKE2b-128 hace despreciable la probabilidad de colisión incluso con
    # ~1 millón de filas. Los campos se codifican con longitud para evitar
    # ambigüedades por separadores presentes en el payload.
    hasher = hashlib.blake2b(digest_size=16, person=b"srbh-reqfp-v1")
    _fingerprint_update(hasher, REQUEST_FINGERPRINT_VERSION)
    _fingerprint_update(hasher, method_s.upper())
    _fingerprint_update(hasher, uri_s)
    _fingerprint_update(hasher, _canonical_protocol_for_fingerprint(protocol_s))
    for name in sorted(headers):
        _fingerprint_update(hasher, _canonical_header_name(name))
        # HTTP permite OWS alrededor del valor; CR/LF/NUL no se eliminan porque
        # distinguen intentos de manipulación de protocolo.
        _fingerprint_update(hasher, _flatten_header_value(headers[name]).strip(" \t"))
    _fingerprint_update(hasher, body)
    return hasher.hexdigest()


def build_request_fingerprints(
    df: pd.DataFrame,
    *,
    workers: int = 1,
    chunksize: int = 256,
) -> Tuple[pd.Series, float]:
    """Calcula fingerprints antes de extraer features para poder deduplicar."""
    rows = _feature_input_tuples(df)
    t0 = time.perf_counter()
    if workers and workers > 1 and len(df) > 500:
        with ProcessPoolExecutor(max_workers=int(workers)) as ex:
            values = list(tqdm(
                ex.map(_request_fingerprint_from_tuple, rows, chunksize=max(1, int(chunksize))),
                total=len(df),
                desc="Fingerprinting HTTP requests",
                mininterval=5,
            ))
    else:
        values = [
            _request_fingerprint_from_tuple(row)
            for row in tqdm(rows, total=len(df), desc="Fingerprinting HTTP requests", mininterval=5)
        ]
    seconds = time.perf_counter() - t0
    fingerprints = pd.Series(values, index=df.index, name=REQUEST_FINGERPRINT_COL, dtype="string")
    if len(fingerprints) != len(df) or fingerprints.isna().any() or not fingerprints.str.len().eq(32).all():
        raise RuntimeError("Fingerprint request-only incompleto o inválido")
    return fingerprints, float(seconds)


def build_feature_frame(df: pd.DataFrame, *, workers: int = 1, chunksize: int = 256) -> Tuple[pd.DataFrame, float]:
    # itertuples es lazy: evita materializar 907k tuples con 16 strings antes
    # de arrancar el ProcessPool.
    rows = _feature_input_tuples(df)
    t0 = time.perf_counter()
    if workers and workers > 1 and len(df) > 500:
        with ProcessPoolExecutor(max_workers=int(workers)) as ex:
            vals = list(tqdm(ex.map(_extract_features_from_tuple, rows, chunksize=max(1, int(chunksize))), total=len(df), desc="Extracting request90 features", mininterval=5))
    else:
        vals = [_extract_features_from_tuple(r) for r in tqdm(rows, total=len(df), desc="Extracting request90 features", mininterval=5)]
    secs = time.perf_counter() - t0
    feats = pd.DataFrame(vals, columns=FIXED_FEATURES).fillna(0).astype(np.float32)
    if feats.shape[1] != 90 or not np.isfinite(feats.to_numpy(copy=False)).all():
        raise RuntimeError(f"Matriz request90 inválida: shape={feats.shape} finite={np.isfinite(feats.to_numpy(copy=False)).all()}")
    return pd.concat([df.reset_index(drop=True), feats], axis=1), secs


# ---------------------------------------------------------------------------
# Labels y loaders TorpEda
# ---------------------------------------------------------------------------

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
            RAW_PROTOCOL_COL: protocol or http_version,
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


# ---------------------------------------------------------------------------
# Labels y loaders Harvard/SR-BH
# ---------------------------------------------------------------------------

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
    df.columns = [str(c).lstrip("\ufeff").strip() for c in df.columns]
    return df


def _normalized_column_key(name: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", _safe_str(name).strip().lower()).strip("_")


def _coalesce_alias_column(df: pd.DataFrame, canonical: str, aliases: Sequence[str]) -> None:
    lookup: Dict[str, str] = {}
    for column in df.columns:
        lookup.setdefault(_normalized_column_key(column), str(column))
    candidates: List[str] = []
    for name in (canonical, *aliases):
        actual = lookup.get(_normalized_column_key(name))
        if actual and actual not in candidates:
            candidates.append(actual)
    if not candidates:
        if canonical not in df.columns:
            df[canonical] = ""
        return
    result = df[candidates[0]]
    for column in candidates[1:]:
        present = result.notna() & result.astype(str).str.strip().ne("") & result.astype(str).str.lower().ne("nan")
        result = result.where(present, df[column])
    df[canonical] = result


def _ensure_request_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Crea las columnas canónicas sin tocar response_* ni metadatos de red."""
    for canonical, aliases in CORE_COLUMN_ALIASES.items():
        _coalesce_alias_column(df, canonical, aliases)
    for canonical, aliases in HEADER_COLUMN_ALIASES.items():
        _coalesce_alias_column(df, canonical, aliases)
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
    original_keys = {_normalized_column_key(column) for column in df.columns}
    # Si se usan los nombres por defecto, permite aliases habituales. Si el
    # usuario pasó un nombre explícito, se respeta y se valida literalmente.
    if schema.method_col == RAW_METHOD_COL or schema.uri_col == RAW_URI_COL or schema.body_col == RAW_BODY_COL:
        df = _ensure_request_feature_columns(df)
    missing: List[str] = []
    for required in [schema.method_col, schema.uri_col, schema.body_col]:
        if required in {RAW_METHOD_COL, RAW_URI_COL, RAW_BODY_COL}:
            candidates = (required, *CORE_COLUMN_ALIASES.get(required, ()))
            if not any(_normalized_column_key(name) in original_keys for name in candidates):
                missing.append(required)
        elif required not in df.columns:
            missing.append(required)
    if missing:
        raise ValueError(
            f"Missing expected Harvard/SR-BH columns: {missing}. Detected sep={sep!r}. "
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
    lab = _safe_str(label).strip()
    if not lab:
        return "UNKNOWN"
    lab = re.sub(r"^CAPEC-\d+\s+", "", lab).strip()
    lab = re.sub(r"^\d{1,3}\s+-\s+", "", lab).strip()
    words = re.findall(r"[A-Za-z0-9]+", lab)
    if not words:
        return "UNKNOWN"
    return "".join(w[:1].upper() + w[1:] for w in words)


def _map_harvard_native_to_family(label: str, prefix: str) -> str:
    lab = _safe_str(label).strip()
    pref = _safe_str(prefix).strip() or "HARVARD"
    if not lab or lab == "NORMAL":
        return "NORMAL"
    low = lab.lower()
    capec = _capec_id_from_label(lab) or ""
    if capec in HARVARD_CAPEC_FAMILY_MAP:
        return f"{pref}-{HARVARD_CAPEC_FAMILY_MAP[capec]}"
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
    lab = _safe_str(label).strip()
    pref = _safe_str(prefix).strip() or "HARVARD"
    if not lab or lab == "NORMAL":
        return "NORMAL"
    low = lab.lower()
    capec = _capec_id_from_label(lab) or ""
    if capec in HARVARD_OPTIMIZED_FAMILY_MAP:
        return f"{pref}-{HARVARD_OPTIMIZED_FAMILY_MAP[capec]}"
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
    mode = (harvard_label_mode or "optimized-family").strip().lower()
    pref = label_prefix or "HARVARD"
    if mode in {"native", "capec", "srbh"}:
        out[LABEL_COL] = out["label_multiclass_native"]
    elif mode in {"family", "mapped", "attack-family", "attack_families"}:
        out[LABEL_COL] = out["label_multiclass_native"].map(lambda x: _map_harvard_native_to_family(str(x), pref))
    elif mode in {"optimized-family", "optimized_family", "optimized", "coarse", "coarse-family", "coarse_family"}:
        out[LABEL_COL] = out["label_multiclass_native"].map(lambda x: _map_harvard_native_to_optimized_family(str(x), pref))
    elif mode in {"legacy-common-anomalous", "legacy_common_anomalous"}:
        out[LABEL_COL] = out["label_multiclass_native"].map(lambda x: _map_harvard_native_to_legacy_common_anomalous(str(x), pref))
    else:
        raise ValueError(f"Unsupported --harvard-label-mode={harvard_label_mode!r}")
    return out


# ---------------------------------------------------------------------------
# Dataset loading y optimización previa al split
# ---------------------------------------------------------------------------

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
            import glob
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


def load_dataset_for_run(args: argparse.Namespace, dataset: str) -> Tuple[pd.DataFrame, List[str]]:
    label_prefix = "TORPEDA" if dataset == "torpeda" else "HARVARD"
    if dataset == "torpeda":
        inputs = _collect_inputs(args.inputs or args.torpeda_inputs or [], dataset="torpeda")
        if not inputs:
            raise SystemExit("No XML inputs found for TorpEda. Check --inputs/--torpeda-inputs/TORPEDA_RAW_DIR.")
        frames = []
        print(f"[INFO] TorpEda XML files: {len(inputs)}", flush=True)
        for p in tqdm(inputs, desc="Loading TorpEda XML", mininterval=5):
            d = load_torpeda_xml(p, keep_absolute_uri=args.keep_absolute_uri, sample_n=args.sample_n, label_prefix=label_prefix)
            if d is not None and not d.empty:
                frames.append(d)
        if not frames:
            raise SystemExit("No samples parsed from TorpEda XML inputs.")
        df = pd.concat(frames, ignore_index=True)
    elif dataset == "harvard":
        inputs = _collect_inputs(args.inputs or args.harvard_inputs or [], dataset="harvard")
        if not inputs:
            raise SystemExit("No Harvard/SR-BH CSV/TSV inputs found. Check --inputs/--harvard-inputs/HARVARD_RAW_DIR.")
        label_cols = [x.strip() for x in (args.label_cols or "").split(",") if x.strip()] or None
        schema = SRBHSchema(
            sep=args.sep,
            method_col=args.method_col,
            uri_col=args.uri_col,
            protocol_col=args.protocol_col,
            body_col=args.body_col,
            normal_col=args.normal_col,
            label_cols=label_cols,
            multiclass_strategy=args.harvard_multiclass_strategy,
            severity_map=DEFAULT_SRBH_SEVERITY_MAP,
        )
        frames = []
        print(f"[INFO] Harvard/SR-BH files: {len(inputs)}", flush=True)
        for p in tqdm(inputs, desc="Loading Harvard/SR-BH", mininterval=5):
            d = load_srbh_csv(p, schema)
            if args.sample_n and args.sample_n > 0:
                d = d.head(args.sample_n).copy()
            d["source_file"] = p
            frames.append(d)
        df = pd.concat(frames, ignore_index=True)
        df = make_srbh_labels(df, schema, label_prefix=label_prefix, harvard_label_mode=args.harvard_label_mode)
        if schema.method_col != RAW_METHOD_COL:
            df[RAW_METHOD_COL] = df[schema.method_col]
        if schema.uri_col != RAW_URI_COL:
            df[RAW_URI_COL] = df[schema.uri_col]
        if schema.protocol_col != RAW_PROTOCOL_COL and schema.protocol_col in df.columns:
            df[RAW_PROTOCOL_COL] = df[schema.protocol_col]
        if schema.body_col != RAW_BODY_COL:
            df[RAW_BODY_COL] = df[schema.body_col]
        df = _ensure_request_feature_columns(df)
        if "sample_id" not in df.columns:
            df["sample_id"] = np.arange(len(df)).astype(str)
    else:
        raise SystemExit(f"Unsupported dataset={dataset!r}")

    for col in [RAW_METHOD_COL, RAW_URI_COL, RAW_PROTOCOL_COL, RAW_BODY_COL, RAW_HEADERS_JSON_COL, *SRBH_HEADER_COLUMN_MAP.keys(), LABEL_COL]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("")
    df[LABEL_COL] = df[LABEL_COL].astype(str)
    if "sample_id" not in df.columns:
        df["sample_id"] = np.arange(len(df)).astype(str)
    if "source_file" not in df.columns:
        df["source_file"] = ""
    return df.reset_index(drop=True), inputs


def apply_label_filters(df: pd.DataFrame, args: argparse.Namespace, dataset: str, out_dir: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    out = df.copy()
    report: Dict[str, Any] = {
        "dataset": dataset,
        "rows_before": int(len(out)),
        "class_distribution_before": out[LABEL_COL].astype(str).value_counts().to_dict(),
        "operations": [],
    }

    if dataset == "torpeda" and args.torpeda_only_common_labels:
        allowed = {"NORMAL", "TORPEDA-ANOMALOUS", *[f"TORPEDA-{a}" for a in COMMON_ATTACKS]}
        before = len(out)
        out = out[out[LABEL_COL].isin(allowed)].copy()
        report["operations"].append({"op": "torpeda_only_common_labels", "removed_rows": int(before - len(out)), "allowed_labels": sorted(allowed)})

    keep_re = str(args.keep_labels_regex or "").strip()
    if keep_re:
        before = len(out)
        out = out[out[LABEL_COL].astype(str).str.contains(keep_re, regex=True, na=False)].copy()
        report["operations"].append({"op": "keep_labels_regex", "regex": keep_re, "removed_rows": int(before - len(out))})

    drop_re = str(args.drop_labels_regex or "").strip()
    if drop_re:
        before = len(out)
        out = out[~out[LABEL_COL].astype(str).str.contains(drop_re, regex=True, na=False)].copy()
        report["operations"].append({"op": "drop_labels_regex", "regex": drop_re, "removed_rows": int(before - len(out))})

    min_count = int(args.harvard_min_class_count if dataset == "harvard" else args.torpeda_min_class_count)
    if min_count > 1:
        vc = out[LABEL_COL].astype(str).value_counts()
        keep = set(vc[vc >= min_count].index.astype(str).tolist())
        dropped = vc[vc < min_count].rename_axis("label").reset_index(name="count")
        if not dropped.empty:
            dropped.to_csv(out_dir / "dropped_classes_by_min_count.csv", index=False)
        before = len(out)
        out = out[out[LABEL_COL].astype(str).isin(keep)].copy()
        report["operations"].append({"op": "min_class_count", "min_class_count": int(min_count), "removed_rows": int(before - len(out)), "dropped_classes": dropped.to_dict(orient="records")})

    max_normal = int(args.max_normal_rows or 0)
    if max_normal > 0 and "NORMAL" in set(out[LABEL_COL]):
        normal = out[out[LABEL_COL] == "NORMAL"]
        other = out[out[LABEL_COL] != "NORMAL"]
        if len(normal) > max_normal:
            normal = normal.sample(n=max_normal, random_state=args.seed)
            before = len(out)
            out = pd.concat([normal, other], ignore_index=True).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
            report["operations"].append({"op": "max_normal_rows", "max_normal_rows": int(max_normal), "removed_rows": int(before - len(out))})

    out = out.reset_index(drop=True)
    if out.empty:
        raise SystemExit(f"No rows left for {dataset} after label filters.")
    if out[LABEL_COL].nunique() < 2:
        raise SystemExit(f"Need at least 2 classes for {dataset}; got {out[LABEL_COL].unique().tolist()}")

    report["rows_after"] = int(len(out))
    report["class_distribution_after"] = out[LABEL_COL].astype(str).value_counts().to_dict()
    _save_json(report, out_dir / "dataset_filtering_report.json")
    return out, report



# ---------------------------------------------------------------------------
# DEFINITIVO multietiqueta Harvard/SR-BH
# ---------------------------------------------------------------------------

from sklearn.metrics import (
    hamming_loss,
    jaccard_score,
    multilabel_confusion_matrix,
)

TARGET_PREFIX = "target__"
DEFAULT_MULTILABEL_MODEL_NAME = "xgboost_ovr_request90"


def _dataset_out_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    return Path(args.definitivo_dir) / "multietiqueta" / "harvard"


def _clean_output_dir(out_dir: Path) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


def _normalize_bool_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.astype(int)
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").fillna(0).astype(int).clip(0, 1)
    return s.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y", "si", "sí", "attack", "malicious"}).astype(int)


def _target_col_name(label: str) -> str:
    return TARGET_PREFIX + _safe_name(label)


def _label_from_harvard_col(col: str, label_prefix: str, mode: str) -> str:
    canon = _canonical_capec_label(str(col))
    mode_l = (mode or "native").strip().lower()
    pref = label_prefix or "HARVARD"
    if mode_l in {"native", "capec", "srbh"}:
        return canon
    if mode_l in {"family", "mapped", "attack-family", "attack_families"}:
        return _map_harvard_native_to_family(canon, pref)
    if mode_l in {"optimized-family", "optimized_family", "optimized", "coarse", "coarse-family", "coarse_family"}:
        return _map_harvard_native_to_optimized_family(canon, pref)
    raise ValueError(f"Unsupported --label-mode={mode!r}")


def load_harvard_multilabel_raw(args: argparse.Namespace) -> Tuple[pd.DataFrame, List[str], SRBHSchema, List[str]]:
    inputs = _collect_inputs(args.inputs or args.harvard_inputs or [], dataset="harvard")
    if not inputs:
        raise SystemExit("No encontré CSV/TSV/GZ de Harvard/SR-BH. Revisá --harvard-inputs o HARVARD_RAW_DIR.")

    label_cols = [x.strip() for x in (args.label_cols or "").split(",") if x.strip()] or None
    schema = SRBHSchema(
        sep=args.sep,
        method_col=args.method_col,
        uri_col=args.uri_col,
        protocol_col=args.protocol_col,
        body_col=args.body_col,
        normal_col=args.normal_col,
        label_cols=label_cols,
        multiclass_strategy="severity",
        severity_map=DEFAULT_SRBH_SEVERITY_MAP,
    )
    frames: List[pd.DataFrame] = []
    print(f"[INFO] Harvard/SR-BH files: {len(inputs)}", flush=True)
    for p in tqdm(inputs, desc="Loading Harvard/SR-BH", mininterval=5):
        d = load_srbh_csv(p, schema)
        if args.sample_n and args.sample_n > 0:
            d = d.head(args.sample_n).copy()
        d["source_file"] = p
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df = _normalize_columns(df)
    discovered_label_cols = _discover_srbh_label_cols(df, schema)

    for c in [schema.method_col, schema.uri_col, schema.body_col]:
        if c not in df.columns:
            raise SystemExit(f"Falta columna requerida en Harvard/SR-BH: {c!r}")
    if schema.method_col != RAW_METHOD_COL:
        df[RAW_METHOD_COL] = df[schema.method_col]
    if schema.uri_col != RAW_URI_COL:
        df[RAW_URI_COL] = df[schema.uri_col]
    if schema.protocol_col != RAW_PROTOCOL_COL and schema.protocol_col in df.columns:
        df[RAW_PROTOCOL_COL] = df[schema.protocol_col]
    if schema.body_col != RAW_BODY_COL:
        df[RAW_BODY_COL] = df[schema.body_col]
    df = _ensure_request_feature_columns(df)
    if "sample_id" not in df.columns:
        df["sample_id"] = np.arange(len(df)).astype(str)
    for col in [RAW_METHOD_COL, RAW_URI_COL, RAW_PROTOCOL_COL, RAW_BODY_COL, RAW_HEADERS_JSON_COL, *SRBH_HEADER_COLUMN_MAP.keys(), "sample_id", "source_file"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("")
    return df.reset_index(drop=True), inputs, schema, discovered_label_cols


def audit_and_deduplicate_requests(
    work: pd.DataFrame,
    *,
    schema: SRBHSchema,
    attack_cols: Sequence[str],
    target_cols_by_label: Dict[str, str],
    args: argparse.Namespace,
    out_dir: Path,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Audita requests idénticas y, si se solicita, une sus labels con OR."""
    rows_before = int(len(work))
    fingerprints, fingerprint_seconds = build_request_fingerprints(
        work,
        workers=int(getattr(args, "feature_workers", 1)),
        chunksize=int(getattr(args, "feature_chunksize", 256)),
    )
    work[REQUEST_FINGERPRINT_COL] = fingerprints.to_numpy()

    target_cols = [column for column in target_cols_by_label.values() if column in work.columns]
    union_cols = list(dict.fromkeys([schema.normal_col, *attack_cols, *target_cols]))
    grouped = work.groupby(REQUEST_FINGERPRINT_COL, sort=False, dropna=False)
    group_sizes = grouped.size().astype(np.int64)
    label_max = grouped[union_cols].max().astype(np.int8)
    label_min = grouped[union_cols].min().astype(np.int8)
    disagreements = label_max.ne(label_min)
    disagreement_any = disagreements.any(axis=1)

    duplicate_mask = group_sizes.gt(1)
    duplicate_groups = int(duplicate_mask.sum())
    unique_requests = int(len(group_sizes))
    duplicate_rows = int(rows_before - unique_requests)
    attack_union = label_max[list(attack_cols)].max(axis=1).astype(bool)
    normal_union = label_max[schema.normal_col].astype(bool)
    normal_attack_union = normal_union & attack_union

    duplicate_index = group_sizes.index[duplicate_mask]
    duplicate_table = pd.DataFrame({
        REQUEST_FINGERPRINT_COL: duplicate_index.astype(str),
        "group_size": group_sizes.loc[duplicate_index].to_numpy(dtype=np.int64),
        "label_disagreement": disagreement_any.loc[duplicate_index].to_numpy(dtype=bool),
        "normal_present_in_group": normal_union.loc[duplicate_index].to_numpy(dtype=bool),
        "attack_present_in_group": attack_union.loc[duplicate_index].to_numpy(dtype=bool),
        "normal_attack_conflict_after_or": normal_attack_union.loc[duplicate_index].to_numpy(dtype=bool),
    })
    if len(duplicate_table):
        target_matrix = label_max.loc[duplicate_index, target_cols].to_numpy(dtype=np.int8, copy=False)
        labels = list(target_cols_by_label.keys())
        duplicate_table["target_labels_union_json"] = [
            json.dumps([labels[j] for j, value in enumerate(row) if int(value) == 1], ensure_ascii=False)
            for row in target_matrix
        ]
    else:
        duplicate_table["target_labels_union_json"] = pd.Series(dtype=str)
    _atomic_csv(duplicate_table, out_dir / "request_duplicate_groups.csv")

    deduplicate = bool(getattr(args, "deduplicate_requests", True))
    if deduplicate:
        first_rows = ~work[REQUEST_FINGERPRINT_COL].duplicated(keep="first")
        deduplicated = work.loc[first_rows].copy()
        order = deduplicated[REQUEST_FINGERPRINT_COL].astype(str).tolist()
        aggregated = label_max.reindex(order)
        if aggregated.isna().any().any():
            raise RuntimeError("No pude alinear labels agregados con fingerprints representativos")
        for column in union_cols:
            deduplicated[column] = aggregated[column].to_numpy(dtype=np.int8)
        deduplicated[DUPLICATE_GROUP_SIZE_COL] = group_sizes.reindex(order).to_numpy(dtype=np.int64)
        # En una unión OR, cualquier etiqueta de ataque domina a NORMAL. Se
        # conserva el conflicto en la auditoría, no como target contradictorio.
        has_attack = deduplicated[list(attack_cols)].max(axis=1).astype(bool)
        deduplicated[schema.normal_col] = (~has_attack).astype(np.int8)
        result = deduplicated.reset_index(drop=True)
    else:
        work[DUPLICATE_GROUP_SIZE_COL] = work[REQUEST_FINGERPRINT_COL].map(group_sizes).astype(np.int64)
        result = work.reset_index(drop=True)

    target_positive_rows_before = {
        label: int(work[column].sum()) for label, column in target_cols_by_label.items()
    }
    target_positive_groups_after_or = {
        label: int(label_max[column].sum()) for label, column in target_cols_by_label.items()
    }
    target_conflict_groups = {
        label: int(disagreements[column].sum()) for label, column in target_cols_by_label.items()
    }
    audit: Dict[str, Any] = {
        "fingerprint_version": REQUEST_FINGERPRINT_VERSION,
        "fingerprint_digest_bits": 128,
        "fingerprint_request_only": True,
        "fingerprint_excluded_fields": ["timestamp", "src_ip", "src_port", "dst_ip", "dst_port", "response_*"],
        "fingerprint_seconds": float(fingerprint_seconds),
        "rows_before_deduplication": rows_before,
        "unique_request_fingerprints": unique_requests,
        "duplicate_groups": duplicate_groups,
        "duplicate_rows_beyond_first": duplicate_rows,
        "max_duplicate_group_size": int(group_sizes.max()) if len(group_sizes) else 0,
        "groups_with_any_label_disagreement": int(disagreement_any.sum()),
        "groups_with_normal_attack_conflict_after_or": int(normal_attack_union.sum()),
        "deduplicate_requests": deduplicate,
        "aggregation": "OR labels; first request row retained as representative" if deduplicate else "audit only; rows retained",
        "rows_after_deduplication": int(len(result)),
        "rows_removed_by_deduplication": int(rows_before - len(result)),
        "target_positive_rows_before": target_positive_rows_before,
        "target_positive_groups_after_or": target_positive_groups_after_or,
        "target_conflict_groups": target_conflict_groups,
        "estimated_blake2b128_collision_probability_upper_bound": float(
            rows_before * max(0, rows_before - 1) / (2.0 * (2.0 ** 128))
        ),
        "duplicate_groups_csv": str(out_dir / "request_duplicate_groups.csv"),
    }
    _save_json(audit, out_dir / "request_deduplication_audit.json")
    print(
        "[DEDUP] "
        f"rows={rows_before} unique={unique_requests} duplicate_groups={duplicate_groups} "
        f"label_conflicts={audit['groups_with_any_label_disagreement']} "
        f"normal_attack_conflicts={audit['groups_with_normal_attack_conflict_after_or']} "
        f"after={len(result)} enabled={deduplicate} fingerprint_s={fingerprint_seconds:.1f}",
        flush=True,
    )
    return result, audit


def make_multilabel_targets(
    df: pd.DataFrame,
    schema: SRBHSchema,
    label_cols: Sequence[str],
    args: argparse.Namespace,
    out_dir: Path,
) -> Tuple[pd.DataFrame, np.ndarray, List[str], List[Dict[str, Any]], Dict[str, Any]]:
    if schema.normal_col not in df.columns:
        raise SystemExit(f"No está la columna normal {schema.normal_col!r}. Ajustá --normal-col.")
    all_label_cols = [c for c in label_cols if c in df.columns]
    attack_cols = [c for c in all_label_cols if c != schema.normal_col]
    if not attack_cols:
        raise SystemExit("No encontré columnas de ataque para multietiqueta. Revisá --label-cols/--normal-col.")

    work = df.copy()
    for c in all_label_cols:
        work[c] = _normalize_bool_series(work[c])

    rows_before_cleaning = int(len(work))
    attack_any_native = work[attack_cols].max(axis=1).astype(int)
    normal_native = work[schema.normal_col].astype(int)
    empty_request = (
        work[RAW_METHOD_COL].map(_safe_str).str.strip().eq("")
        & work[RAW_URI_COL].map(_safe_str).str.strip().eq("")
        & work[RAW_BODY_COL].map(_safe_str).str.strip().eq("")
    )
    unlabeled = (normal_native == 0) & (attack_any_native == 0)
    normal_attack_conflict = (normal_native == 1) & (attack_any_native == 1)
    row_audit = {
        "empty_request_rows": int(empty_request.sum()),
        "unlabeled_rows": int(unlabeled.sum()),
        "normal_attack_conflict_rows": int(normal_attack_conflict.sum()),
    }
    pre_target_operations: List[Dict[str, Any]] = []
    if bool(getattr(args, "drop_invalid_rows", True)):
        invalid = empty_request | unlabeled
        if bool(invalid.any()):
            work = work.loc[~invalid].reset_index(drop=True)
            pre_target_operations.append({
                "op": "drop_invalid_rows",
                "removed_rows": int(invalid.sum()),
                "empty_request_rows": int(empty_request.sum()),
                "unlabeled_rows": int(unlabeled.sum()),
            })
    rows_after_invalid_filter = int(len(work))

    label_mode = (args.label_mode or "native").strip().lower()
    grouped: Dict[str, List[str]] = {}
    for c in attack_cols:
        label = _label_from_harvard_col(c, "HARVARD", label_mode)
        grouped.setdefault(label, []).append(c)

    target_labels_all = sorted(grouped.keys())
    target_cols_all: Dict[str, str] = {}
    for lab in target_labels_all:
        tcol = _target_col_name(lab)
        target_cols_all[lab] = tcol
        cols = grouped[lab]
        work[tcol] = work[cols].max(axis=1).astype(np.int8)

    work, deduplication_audit = audit_and_deduplicate_requests(
        work,
        schema=schema,
        attack_cols=attack_cols,
        target_cols_by_label=target_cols_all,
        args=args,
        out_dir=out_dir,
    )
    if bool(getattr(args, "deduplicate_requests", True)):
        pre_target_operations.append({
            "op": "deduplicate_requests_or_labels",
            "fingerprint_version": REQUEST_FINGERPRINT_VERSION,
            "removed_rows": int(deduplication_audit["rows_removed_by_deduplication"]),
            "duplicate_groups": int(deduplication_audit["duplicate_groups"]),
            "label_disagreement_groups": int(deduplication_audit["groups_with_any_label_disagreement"]),
            "normal_attack_conflict_groups": int(deduplication_audit["groups_with_normal_attack_conflict_after_or"]),
        })

    # Construye la lista de etiquetas nativas por fila, útil para auditoría y predicciones.
    native_labels_per_row: List[List[str]] = []
    for _, row in work[attack_cols].iterrows():
        labs = [_canonical_capec_label(c) for c in attack_cols if int(row[c]) == 1]
        native_labels_per_row.append(labs)
    work["labels_native_json"] = [json.dumps(x, ensure_ascii=False) for x in native_labels_per_row]

    target_counts = []
    keep_labels: List[str] = []
    dropped_labels: List[Dict[str, Any]] = []
    n_rows = len(work)
    min_pos = int(args.min_positive_count or 0)
    max_neg_ratio = float(args.max_negative_positive_ratio or 0.0)
    for lab in target_labels_all:
        tcol = target_cols_all[lab]
        pos = int(work[tcol].sum())
        neg = int(n_rows - pos)
        prevalence = float(pos / n_rows) if n_rows else 0.0
        rec = {
            "label": lab,
            "target_col": tcol,
            "source_label_cols": grouped[lab],
            "positive_count": pos,
            "negative_count": neg,
            "prevalence": prevalence,
        }
        target_counts.append(rec)
        drop_reason = None
        if pos < min_pos:
            drop_reason = f"positive_count<{min_pos}"
        elif pos == 0 or neg == 0:
            drop_reason = "constant_target"
        elif max_neg_ratio > 0 and pos > 0 and (neg / max(pos, 1)) > max_neg_ratio:
            # Filtro opcional para etiquetas extremadamente raras si se busca estabilidad.
            drop_reason = f"neg_pos_ratio>{max_neg_ratio}"
        if drop_reason:
            dropped = dict(rec)
            dropped["drop_reason"] = drop_reason
            dropped_labels.append(dropped)
        else:
            keep_labels.append(lab)

    pd.DataFrame(target_counts).sort_values(["positive_count", "label"], ascending=[False, True]).to_csv(out_dir / "label_distribution_multilabel_all.csv", index=False)
    if dropped_labels:
        pd.DataFrame(dropped_labels).to_csv(out_dir / "dropped_multilabel_targets.csv", index=False)

    if not keep_labels:
        raise SystemExit("Todas las etiquetas multietiqueta fueron descartadas. Bajá --min-positive-count o revisá el dataset.")

    target_cols = [_target_col_name(lab) for lab in keep_labels]
    Y = work[target_cols].fillna(0).astype(np.int8).to_numpy()
    work["label_count"] = Y.sum(axis=1).astype(int)
    work["label_binary"] = (work["label_count"] > 0).astype(np.int8)
    work["labels_target_json"] = [json.dumps([keep_labels[j] for j, v in enumerate(row) if int(v) == 1], ensure_ascii=False) for row in Y]

    # Filtros de filas: normalmente se conserva todo, salvo downsample normal explícito.
    report: Dict[str, Any] = {
        "task": "multilabel",
        "dataset": "harvard",
        "rows_before": rows_before_cleaning,
        "rows_after_invalid_filter": rows_after_invalid_filter,
        "rows_after_deduplication": int(deduplication_audit["rows_after_deduplication"]),
        "row_audit": row_audit,
        "request_deduplication": deduplication_audit,
        "label_mode": label_mode,
        "all_target_count": int(len(target_labels_all)),
        "kept_target_count": int(len(keep_labels)),
        "dropped_target_count": int(len(dropped_labels)),
        "min_positive_count": min_pos,
        "operations": list(pre_target_operations),
    }

    max_normal = int(args.max_normal_rows or 0)
    if max_normal > 0:
        Y_now = work[target_cols].fillna(0).astype(np.int8).to_numpy()
        normal_mask = Y_now.sum(axis=1) == 0
        normal_idx = np.flatnonzero(normal_mask)
        attack_idx = np.flatnonzero(~normal_mask)
        if len(normal_idx) > max_normal:
            rng = np.random.default_rng(int(args.seed))
            keep_normal = rng.choice(normal_idx, size=max_normal, replace=False)
            keep_idx = np.concatenate([keep_normal, attack_idx])
            rng.shuffle(keep_idx)
            before = len(work)
            work = work.iloc[keep_idx].reset_index(drop=True)
            Y = work[target_cols].fillna(0).astype(np.int8).to_numpy()
            work["label_count"] = Y.sum(axis=1).astype(int)
            work["label_binary"] = (work["label_count"] > 0).astype(np.int8)
            work["labels_target_json"] = [json.dumps([keep_labels[j] for j, v in enumerate(row) if int(v) == 1], ensure_ascii=False) for row in Y]
            report["operations"].append({"op": "max_normal_rows", "max_normal_rows": max_normal, "removed_rows": int(before - len(work))})

    # Regex opcional sobre etiquetas target agregadas por fila.
    keep_re = str(args.keep_labels_regex or "").strip()
    drop_re = str(args.drop_labels_regex or "").strip()
    if keep_re:
        before = len(work)
        mask = work["labels_target_json"].str.contains(keep_re, regex=True, na=False)
        mask |= (work["label_count"] == 0)  # por defecto se conservan normales para contraste.
        work = work[mask].reset_index(drop=True)
        Y = work[target_cols].fillna(0).astype(np.int8).to_numpy()
        report["operations"].append({"op": "keep_labels_regex", "regex": keep_re, "removed_rows": int(before - len(work))})
    if drop_re:
        before = len(work)
        mask = ~work["labels_target_json"].str.contains(drop_re, regex=True, na=False)
        work = work[mask].reset_index(drop=True)
        Y = work[target_cols].fillna(0).astype(np.int8).to_numpy()
        report["operations"].append({"op": "drop_labels_regex", "regex": drop_re, "removed_rows": int(before - len(work))})

    target_meta = [r for r in target_counts if r["label"] in set(keep_labels)]
    # Reordenar meta como keep_labels.
    meta_by_label = {r["label"]: r for r in target_meta}
    target_meta = [meta_by_label[lab] for lab in keep_labels]
    pd.DataFrame(target_meta).to_csv(out_dir / "label_distribution_multilabel_kept.csv", index=False)

    report["rows_after"] = int(len(work))
    report["normal_rows_after"] = int((Y.sum(axis=1) == 0).sum())
    report["attack_rows_after"] = int((Y.sum(axis=1) > 0).sum())
    report["kept_labels"] = keep_labels
    _save_json(report, out_dir / "dataset_filtering_report_multilabel.json")
    return work, Y, keep_labels, target_meta, report


def _signature_from_row(row: np.ndarray, labels: Sequence[str]) -> str:
    idx = np.flatnonzero(row > 0)
    if len(idx) == 0:
        return "NORMAL"
    if len(idx) <= 3:
        return "|".join(labels[i] for i in idx)
    return "MULTI_" + str(len(idx)) + "__" + "|".join(labels[i] for i in idx[:3])


def _valid_stratify(keys: Sequence[str], test_size: float) -> bool:
    counts = pd.Series(keys).value_counts(dropna=False)
    if counts.empty or counts.min() < 2:
        return False
    n_classes = len(counts)
    n = len(keys)
    n_test = int(math.ceil(n * test_size))
    return n_test >= n_classes and (n - n_test) >= n_classes


def choose_multilabel_stratify_key(Y: np.ndarray, labels: Sequence[str], test_size: float) -> Tuple[Optional[np.ndarray], str]:
    signatures = np.array([_signature_from_row(row, labels) for row in Y], dtype=object)
    if _valid_stratify(signatures, test_size):
        return signatures, "labelset_signature"
    primary = []
    for row in Y:
        idx = np.flatnonzero(row > 0)
        primary.append("NORMAL" if len(idx) == 0 else labels[int(idx[0])])
    primary_arr = np.array(primary, dtype=object)
    if _valid_stratify(primary_arr, test_size):
        return primary_arr, "primary_label"
    binary = np.where(Y.sum(axis=1) > 0, "ATTACK", "NORMAL")
    if _valid_stratify(binary, test_size):
        return binary, "binary_attack_normal"
    return None, "none"


def _split_indices_once(
    indices: np.ndarray,
    Y: np.ndarray,
    labels: Sequence[str],
    test_size: float,
    seed: int,
    groups: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Divide filas una sola vez; si hay grupos, nunca los cruza entre lados."""
    indices = np.asarray(indices, dtype=np.int64)
    if groups is None:
        strat, strat_name = choose_multilabel_stratify_key(Y[indices], labels, test_size)
        left_rel, right_rel = train_test_split(
            np.arange(len(indices)),
            test_size=test_size,
            random_state=seed,
            shuffle=True,
            stratify=strat,
        )
        return indices[left_rel], indices[right_rel], strat_name

    group_values = np.asarray(groups, dtype=object)[indices].astype(str)
    unique_groups, group_inverse = np.unique(group_values, return_inverse=True)
    if len(unique_groups) == len(indices):
        # Camino rápido: después del dedup cada request suele ser un grupo único.
        strat, strat_name = choose_multilabel_stratify_key(Y[indices], labels, test_size)
        left_rel, right_rel = train_test_split(
            np.arange(len(indices)),
            test_size=test_size,
            random_state=seed,
            shuffle=True,
            stratify=strat,
        )
        return indices[left_rel], indices[right_rel], f"{strat_name}_group_unique"

    group_y = np.zeros((len(unique_groups), Y.shape[1]), dtype=np.int8)
    np.maximum.at(group_y, group_inverse, Y[indices].astype(np.int8))
    strat, strat_name = choose_multilabel_stratify_key(group_y, labels, test_size)
    left_groups, right_groups = train_test_split(
        np.arange(len(unique_groups)),
        test_size=test_size,
        random_state=seed,
        shuffle=True,
        stratify=strat,
    )
    right_mask = np.isin(group_inverse, right_groups)
    left_idx = indices[~right_mask]
    right_idx = indices[right_mask]
    return left_idx, right_idx, f"{strat_name}_grouped"


def split_train_tune_calibration_test(
    Y: np.ndarray,
    labels: Sequence[str],
    args: argparse.Namespace,
    groups: Optional[Sequence[Any]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Crea cuatro particiones disjuntas: train, tune, calibration y test bloqueado.

    ``valid_size`` conserva su semántica histórica: es la fracción del bloque
    posterior a test reservada para tune+calibration. ``calibration_share``
    decide qué fracción de ese holdout se usa exclusivamente para umbrales.
    Con los defaults: 64 % train, 8 % tune, 8 % calibration y 20 % test.
    """
    test_size = float(args.test_size)
    valid_size = float(args.valid_size)
    calibration_share = float(getattr(args, "calibration_share", 0.50))
    for name, value in [
        ("test_size", test_size),
        ("valid_size", valid_size),
        ("calibration_share", calibration_share),
    ]:
        if not 0.0 < value < 1.0:
            raise ValueError(f"--{name.replace('_', '-')} debe estar entre 0 y 1; recibido={value}")
    if Y.shape[0] < 8:
        raise ValueError("Se requieren al menos 8 filas para separar train/tune/calibration/test.")

    groups_arr = None if groups is None else np.asarray(groups, dtype=object)
    if groups_arr is not None and len(groups_arr) != Y.shape[0]:
        raise ValueError("groups debe tener una entrada por fila de Y")

    all_idx = np.arange(Y.shape[0])
    development_idx, test_idx, test_stratify = _split_indices_once(
        all_idx, Y, labels, test_size, int(args.seed), groups_arr
    )
    train_idx, tune_calibration_idx, holdout_stratify = _split_indices_once(
        development_idx, Y, labels, valid_size, int(args.seed) + 1, groups_arr
    )
    tune_idx, calibration_idx, calibration_stratify = _split_indices_once(
        tune_calibration_idx, Y, labels, calibration_share, int(args.seed) + 2, groups_arr
    )

    info: Dict[str, Any] = {
        "test_stratify": test_stratify,
        "tune_calibration_stratify": holdout_stratify,
        "calibration_stratify": calibration_stratify,
        "group_split": bool(groups_arr is not None),
        "train_rows": int(len(train_idx)),
        "tune_rows": int(len(tune_idx)),
        "calibration_rows": int(len(calibration_idx)),
        "test_rows": int(len(test_idx)),
        "effective_train_fraction": float(len(train_idx) / len(Y)),
        "effective_tune_fraction": float(len(tune_idx) / len(Y)),
        "effective_calibration_fraction": float(len(calibration_idx) / len(Y)),
        "effective_test_fraction": float(len(test_idx) / len(Y)),
    }
    if groups_arr is not None:
        split_groups = {
            name: set(map(str, groups_arr[split_idx]))
            for name, split_idx in [
                ("train", train_idx),
                ("tune", tune_idx),
                ("calibration", calibration_idx),
                ("test", test_idx),
            ]
        }
        info["group_counts"] = {name: int(len(values)) for name, values in split_groups.items()}
        info["group_overlap_count"] = int(sum(
            len(split_groups[a] & split_groups[b])
            for i, a in enumerate(split_groups)
            for b in list(split_groups)[i + 1:]
        ))
        if info["group_overlap_count"] != 0:
            raise RuntimeError("El split por request produjo grupos solapados; se aborta para evitar leakage.")

    minimum_calibration_positives = max(1, int(getattr(args, "min_calibration_positives", 3)))
    for split_name, split_idx in [
        ("train", train_idx),
        ("tune", tune_idx),
        ("calibration", calibration_idx),
        ("test", test_idx),
    ]:
        pos = Y[split_idx].sum(axis=0).astype(int)
        missing = [labels[i] for i, count in enumerate(pos) if count == 0]
        if missing:
            print(
                f"[WARN] Split {split_name} sin positivos para {len(missing)} etiquetas: "
                f"{missing[:5]}{'...' if len(missing) > 5 else ''}",
                flush=True,
            )
        info[f"{split_name}_positive_counts"] = {labels[i]: int(pos[i]) for i in range(len(labels))}
        info[f"{split_name}_negative_counts"] = {
            labels[i]: int(len(split_idx) - pos[i]) for i in range(len(labels))
        }
    sparse_calibration = [
        labels[i]
        for i, count in enumerate(Y[calibration_idx].sum(axis=0).astype(int))
        if count < minimum_calibration_positives
    ]
    info["calibration_labels_below_min_positives"] = sparse_calibration
    if sparse_calibration:
        print(
            f"[WARN] Calibration tiene <{minimum_calibration_positives} positivos para "
            f"{len(sparse_calibration)} etiquetas; usarán fallback explícito de umbral tune.",
            flush=True,
        )
    return train_idx, tune_idx, calibration_idx, test_idx, info


def _xgboost_available() -> Tuple[bool, Optional[str]]:
    try:
        import xgboost as xgb  # type: ignore
        return True, str(getattr(xgb, "__version__", "unknown"))
    except Exception:
        return False, None


def _module_available(module_name: str) -> Tuple[bool, Optional[str]]:
    try:
        import importlib
        mod = importlib.import_module(module_name)
        return True, str(getattr(mod, "__version__", "unknown"))
    except Exception:
        return False, None


def _lightgbm_available() -> Tuple[bool, Optional[str]]:
    return _module_available("lightgbm")


def _catboost_available() -> Tuple[bool, Optional[str]]:
    return _module_available("catboost")


def _canonical_backend_name(name: str) -> str:
    b = (name or "").strip().lower().replace("-", "_")
    aliases = {
        "xgb": "xgboost",
        "xgboost_gpu": "xgboost",
        "hist_gradient_boosting": "histgb",
        "histgradientboosting": "histgb",
        "sklearn_histgb": "histgb",
        "sklearn": "histgb",
        "extra_trees": "extra_trees",
        "extratrees": "extra_trees",
        "extra_tree": "extra_trees",
        "et": "extra_trees",
        "random_forest": "random_forest",
        "randomforest": "random_forest",
        "rf": "random_forest",
        "forest": "random_forest",
        "logistic": "logistic_regression",
        "logreg": "logistic_regression",
        "logistic_regression": "logistic_regression",
        "lr": "logistic_regression",
        "linear_svc": "linear_svc",
        "linearsvc": "linear_svc",
        "linear_svm": "linear_svc",
        "svm_linear": "linear_svc",
        "svm": "linear_svc",
        "sgd": "sgd_logloss",
        "sgd_log": "sgd_logloss",
        "sgd_logistic": "sgd_logloss",
        "sgd_logloss": "sgd_logloss",
        "lightgbm": "lightgbm",
        "lgbm": "lightgbm",
        "lgb": "lightgbm",
        "catboost": "catboost",
        "cat": "catboost",
    }
    return aliases.get(b, b)


def _backend_available(backend: str) -> Tuple[bool, Optional[str]]:
    backend = _canonical_backend_name(backend)
    if backend == "xgboost":
        return _xgboost_available()
    if backend == "lightgbm":
        return _lightgbm_available()
    if backend == "catboost":
        return _catboost_available()
    if backend in {"histgb", "extra_trees", "random_forest", "logistic_regression", "linear_svc", "sgd_logloss"}:
        return True, "sklearn"
    return False, None


def describe_backend(backend: str) -> str:
    backend = _canonical_backend_name(backend)
    return {
        "xgboost": "xgboost.XGBClassifier por etiqueta; usa CUDA si hay GPU visible",
        "lightgbm": "lightgbm.LGBMClassifier por etiqueta; intenta GPU si LightGBM está compilado con GPU",
        "catboost": "catboost.CatBoostClassifier por etiqueta; intenta task_type=GPU si hay GPU visible",
        "histgb": "sklearn.ensemble.HistGradientBoostingClassifier por etiqueta; CPU/OpenMP",
        "extra_trees": "sklearn.ensemble.ExtraTreesClassifier por etiqueta; CPU paralelo",
        "random_forest": "sklearn.ensemble.RandomForestClassifier por etiqueta; CPU paralelo",
        "logistic_regression": "Pipeline(StandardScaler, LogisticRegression) por etiqueta; CPU",
        "linear_svc": "Pipeline(StandardScaler, LinearSVC) por etiqueta; scores por decision_function sigmoide; CPU",
        "sgd_logloss": "Pipeline(StandardScaler, SGDClassifier log_loss) por etiqueta; CPU",
    }.get(backend, backend)


def backend_gpu_note(backend: str) -> str:
    backend = _canonical_backend_name(backend)
    if backend == "xgboost":
        return "XGBoost intenta CUDA cuando hay GPU visible; si CUDA falla hace fallback por etiqueta a CPU."
    if backend == "lightgbm":
        return "LightGBM intenta device_type=gpu cuando hay GPU visible; si falla se reintenta CPU por etiqueta."
    if backend == "catboost":
        return "CatBoost intenta task_type=GPU cuando hay GPU visible; si falla se reintenta CPU por etiqueta."
    return "Este backend sklearn no usa CUDA; se acelera con CPU/OpenMP y n_jobs cuando aplica."


def _parse_version_major(version: Optional[str]) -> int:
    if not version:
        return 0
    m = re.match(r"^(\d+)", str(version))
    return int(m.group(1)) if m else 0


def _want_gpu(args: argparse.Namespace) -> bool:
    return str(args.use_gpu).lower() in {"auto", "on", "true", "1", "yes"}


def _compute_scale_pos_weight(y: np.ndarray, mode: str, cap: float) -> float:
    pos = float(np.sum(y == 1))
    neg = float(np.sum(y == 0))
    if pos <= 0:
        return 1.0
    ratio = max(1.0, neg / pos)
    mode_l = (mode or "sqrt").strip().lower().replace("-", "_")
    if mode_l in {"none", "off", "false", "no", "1"}:
        w = 1.0
    elif mode_l in {"sqrt", "sqrt_balanced", "balanced_sqrt"}:
        w = math.sqrt(ratio)
    elif mode_l in {"balanced", "full_balanced"}:
        w = ratio
    elif mode_l in {"cuberoot", "cbrt", "cbrt_balanced"}:
        w = ratio ** (1.0 / 3.0)
    else:
        raise ValueError(f"scale-pos-weight mode no soportado: {mode!r}")
    if cap and cap > 0:
        w = min(float(w), float(cap))
    return float(max(1.0, w))


def _binary_sample_weight(y: np.ndarray, mode: str, cap: float) -> Optional[np.ndarray]:
    mode_l = (mode or "sqrt").strip().lower().replace("-", "_")
    if mode_l in {"none", "off", "false", "no"}:
        return None
    y = np.asarray(y).astype(int)
    pos = float((y == 1).sum())
    neg = float((y == 0).sum())
    if pos <= 0 or neg <= 0:
        return None
    pos_w = _compute_scale_pos_weight(y, mode_l, cap)
    w = np.where(y == 1, pos_w, 1.0).astype(np.float32)
    mean = float(w.mean()) if len(w) else 1.0
    if mean > 0:
        w /= mean
    return w


def _xgb_params_for_preset(args: argparse.Namespace, y_binary: np.ndarray, weight_mode: str, device: str, xgb_version: Optional[str]) -> Dict[str, Any]:
    preset = str(args.xgb_preset or "strong").strip().lower()
    if preset == "fast":
        params = dict(n_estimators=700, learning_rate=0.055, max_depth=4, min_child_weight=1.0, subsample=0.9, colsample_bytree=0.9, reg_alpha=0.0, reg_lambda=1.0, max_delta_step=0.0, max_bin=256)
    elif preset == "max":
        params = dict(n_estimators=2600, learning_rate=0.022, max_depth=6, min_child_weight=1.0, subsample=0.95, colsample_bytree=0.95, reg_alpha=0.0, reg_lambda=1.0, max_delta_step=1.0, max_bin=512)
    else:
        params = dict(n_estimators=1500, learning_rate=0.035, max_depth=5, min_child_weight=1.0, subsample=0.9, colsample_bytree=0.9, reg_alpha=0.0, reg_lambda=1.0, max_delta_step=1.0, max_bin=256)
    params.update(
        objective="binary:logistic",
        eval_metric="aucpr",
        random_state=int(args.seed),
        n_jobs=int(args.n_jobs),
        scale_pos_weight=_compute_scale_pos_weight(y_binary, weight_mode, float(args.scale_pos_weight_cap)),
        verbosity=0,
    )
    if int(args.xgb_early_stopping_rounds or 0) > 0:
        params["early_stopping_rounds"] = int(args.xgb_early_stopping_rounds)
    major = _parse_version_major(xgb_version)
    if device == "cuda":
        if major >= 2:
            params.update(tree_method="hist", device="cuda")
        else:
            params.update(tree_method="gpu_hist", predictor="gpu_predictor")
    else:
        params.update(tree_method="hist")
        if major >= 2:
            params.update(device="cpu")
    return params


def _best_binary_threshold_exact(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    objective: str = "f1",
    min_threshold: float = 0.001,
    max_threshold: float = 0.999,
    min_recall: float = 0.0,
) -> Dict[str, float]:
    """Escanea todos los cortes distintos; no depende de una grilla arbitraria."""
    yt = np.asarray(y_true, dtype=np.int8).ravel()
    sc = np.nan_to_num(np.asarray(scores, dtype=float).ravel(), nan=0.0, posinf=1.0, neginf=0.0)
    if len(yt) == 0 or int(yt.sum()) == 0 or len(np.unique(yt)) < 2:
        return {"threshold": 0.5, "precision": 0.0, "recall": 0.0, "f1": 0.0, "score": 0.0}
    order = np.argsort(-sc, kind="mergesort")
    sorted_scores = sc[order]
    sorted_y = yt[order]
    low = float(min(min_threshold, max_threshold))
    high = float(max(min_threshold, max_threshold))
    distinct = np.unique(sorted_scores[(sorted_scores >= low) & (sorted_scores <= high)])
    thresholds = np.unique(np.r_[distinct, low, high]).astype(float)
    # sorted_scores está descendente; searchsorted sobre el negativo devuelve
    # exactamente cuántos scores cumplen score >= threshold.
    predicted_i = np.searchsorted(-sorted_scores, -thresholds, side="right")
    cumulative_tp = np.cumsum(sorted_y, dtype=np.int64)
    tp = np.where(predicted_i > 0, cumulative_tp[np.maximum(predicted_i - 1, 0)], 0).astype(float)
    predicted = predicted_i.astype(float)
    fp = predicted - tp
    positives = float(yt.sum())
    negatives = float(len(yt) - positives)
    fn = positives - tp
    tn = negatives - fp
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2.0 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) > 0)
    specificity = np.divide(tn, tn + fp, out=np.zeros_like(tn), where=(tn + fp) > 0)

    allowed = np.ones_like(thresholds, dtype=bool)
    if min_recall > 0:
        allowed &= recall >= float(min_recall)
    if not np.any(allowed):
        allowed = np.ones_like(thresholds, dtype=bool)

    objective_l = (objective or "f1").lower()
    if objective_l in {"f2", "fbeta2"}:
        metric = np.divide(5.0 * precision * recall, 4.0 * precision + recall, out=np.zeros_like(f1), where=(4.0 * precision + recall) > 0)
    elif objective_l == "precision":
        metric = precision
    elif objective_l == "recall":
        metric = recall
    elif objective_l in {"youden", "balanced_accuracy"}:
        metric = 0.5 * (recall + specificity)
    else:
        metric = f1

    candidates = np.flatnonzero(allowed)
    best_idx = max(
        candidates.tolist(),
        key=lambda i: (float(metric[i]), float(f1[i]), float(precision[i]), float(thresholds[i])),
    )
    return {
        "threshold": float(thresholds[best_idx]),
        "precision": float(precision[best_idx]),
        "recall": float(recall[best_idx]),
        "f1": float(f1[best_idx]),
        "score": float(metric[best_idx]),
    }


def _log_uniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(math.exp(rng.uniform(math.log(low), math.log(high))))


def _xgb_scale_candidates(y: np.ndarray, cap: float) -> List[Tuple[str, float]]:
    pos = float(np.sum(np.asarray(y) == 1))
    neg = float(np.sum(np.asarray(y) == 0))
    ratio = max(1.0, neg / max(1.0, pos))
    raw = [
        ("none", 1.0),
        ("cuberoot", ratio ** (1.0 / 3.0)),
        ("sqrt", math.sqrt(ratio)),
        ("quarter_balanced", 0.25 * ratio),
        ("half_balanced", 0.50 * ratio),
        ("balanced", ratio),
    ]
    out: List[Tuple[str, float]] = []
    seen: set = set()
    for name, value in raw:
        clipped = min(float(value), float(cap)) if cap and cap > 0 else float(value)
        key = round(max(1.0, clipped), 8)
        if key not in seen:
            seen.add(key)
            out.append((name, float(key)))
    return out


def _xgb_trial_specs(args: argparse.Namespace, y: np.ndarray, label: str, label_index: int) -> List[Dict[str, Any]]:
    weak = bool(re.search(str(args.xgb_weak_label_regex), str(label), re.IGNORECASE)) if str(args.xgb_weak_label_regex).strip() else False
    requested = int(args.xgb_weak_label_trials if weak else args.xgb_tuning_trials)
    requested = max(1, requested)
    rng = np.random.default_rng(int(args.seed) + 10007 * (int(label_index) + 1))
    scales = _xgb_scale_candidates(y, float(args.scale_pos_weight_cap))
    specs: List[Dict[str, Any]] = []
    # Primero cubre explícitamente todo el rango de pesos con el preset base.
    for weight_name, scale in scales:
        specs.append({
            "trial_kind": "weight_baseline",
            "weight_name": weight_name,
            "scale_pos_weight": scale,
            "overrides": {},
        })
        if len(specs) >= requested:
            return specs
    while len(specs) < requested:
        weight_name, scale = scales[int(rng.integers(0, len(scales)))]
        overrides = {
            "max_depth": int(rng.integers(3, 9)),
            "min_child_weight": _log_uniform(rng, 0.75, 12.0),
            "learning_rate": _log_uniform(rng, 0.018, 0.085),
            "subsample": float(rng.uniform(0.75, 1.0)),
            "colsample_bytree": float(rng.uniform(0.65, 1.0)),
            "gamma": 0.0 if rng.random() < 0.35 else _log_uniform(rng, 0.005, 1.0),
            "reg_alpha": 0.0 if rng.random() < 0.35 else _log_uniform(rng, 0.001, 2.0),
            "reg_lambda": _log_uniform(rng, 1.0, 12.0),
            "max_delta_step": float(rng.choice([0.0, 1.0, 3.0, 5.0])),
            "max_bin": int(rng.choice([256, 384, 512])),
        }
        specs.append({
            "trial_kind": "random",
            "weight_name": weight_name,
            "scale_pos_weight": scale,
            "overrides": overrides,
        })
    return specs


def _fit_xgb_candidate(
    args: argparse.Namespace,
    xgb: Any,
    xgb_version: str,
    y_train: np.ndarray,
    X_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    spec: Dict[str, Any],
    seed_offset: int,
    prefer_gpu: bool,
    label: str,
) -> Tuple[Any, Dict[str, Any], bool]:
    device_order = ["cuda", "cpu"] if (prefer_gpu and _want_gpu(args)) else ["cpu"]
    last_error: Optional[Exception] = None
    xgb_major = _parse_version_major(xgb_version)
    for device in device_order:
        params = _xgb_params_for_preset(args, y_train, "none", device, xgb_version)
        params.update(dict(spec.get("overrides") or {}))
        params["scale_pos_weight"] = float(spec["scale_pos_weight"])
        params["random_state"] = int(args.seed) + int(seed_offset)
        try:
            constructor_params = dict(params)
            fit_kwargs: Dict[str, Any] = {"eval_set": [(X_valid, y_valid)], "verbose": False}
            if 0 < xgb_major < 2 and "early_stopping_rounds" in constructor_params:
                fit_kwargs["early_stopping_rounds"] = int(constructor_params.pop("early_stopping_rounds"))
            model = xgb.XGBClassifier(**constructor_params)
            model.fit(X_train, y_train, **fit_kwargs)
            return model, constructor_params, bool(device == "cuda")
        except TypeError as error:
            # XGBoost 1.x recibe early_stopping_rounds en fit, 2.x/3.x en ctor.
            last_error = error
            if "early_stopping_rounds" in params:
                params_legacy = dict(params)
                early = int(params_legacy.pop("early_stopping_rounds"))
                try:
                    model = xgb.XGBClassifier(**params_legacy)
                    model.fit(
                        X_train,
                        y_train,
                        eval_set=[(X_valid, y_valid)],
                        early_stopping_rounds=early,
                        verbose=False,
                    )
                    return model, params_legacy, bool(device == "cuda")
                except Exception as legacy_error:
                    last_error = legacy_error
        except Exception as error:
            last_error = error
        if device == "cuda":
            print(f"[WARN] XGBoost CUDA falló para {label}; pruebo CPU. Error: {last_error}", flush=True)
            continue
        break
    raise RuntimeError(f"No pude entrenar XGBoost para {label}: {last_error}")


def _xgb_trial_progress_paths(
    args: argparse.Namespace,
    out_dir: Optional[Path],
    checkpoint_phase: str,
    label_index: int,
    label: str,
) -> Tuple[Optional[Path], Optional[Path]]:
    if out_dir is None or not checkpoint_phase:
        return None, None
    root = (
        _checkpoint_root(args, out_dir)
        / "xgb_trial_progress"
        / _safe_name(checkpoint_phase)
        / f"{int(label_index):04d}__{_safe_name(label)}"
    )
    return root / "progress.json", root / "best_estimator.joblib"


def _xgb_specs_signature(specs: Sequence[Dict[str, Any]]) -> str:
    return hashlib.sha256(json.dumps(list(specs), sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _load_xgb_trial_progress(
    args: argparse.Namespace,
    out_dir: Optional[Path],
    checkpoint_phase: str,
    label: str,
    label_index: int,
    specs: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Any, Dict[str, Any], Optional[Tuple[float, ...]], Dict[str, Any]]:
    empty = ([], None, {}, None, {})
    if not _resume_enabled(args):
        return empty
    progress_path, best_path = _xgb_trial_progress_paths(args, out_dir, checkpoint_phase, label_index, label)
    if progress_path is None or best_path is None or not progress_path.exists():
        return empty
    try:
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        expected = {
            "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
            "training_signature": _training_signature(args),
            "specs_signature": _xgb_specs_signature(specs),
            "label": str(label),
            "label_index": int(label_index),
        }
        if any(progress.get(key) != value for key, value in expected.items()):
            raise ValueError("firma de progreso no coincide")
        rows = list(progress.get("rows") or [])
        successful = [row for row in rows if row.get("status") == "ok"]
        if not successful:
            return rows, None, {}, None, {}
        if not best_path.exists():
            raise ValueError("hay trials exitosos pero falta best_estimator.joblib")
        payload = joblib.load(best_path)
        if any(payload.get(key) != value for key, value in expected.items()):
            raise ValueError("firma del mejor estimator no coincide")
        estimator = payload.get("estimator")
        if estimator is None:
            raise ValueError("checkpoint best sin estimator")
        key_raw = payload.get("best_key")
        best_key = tuple(float(x) for x in key_raw) if key_raw is not None else None
        print(f"[RESUME] {label}: {len(rows)}/{len(specs)} trials XGBoost ya registrados.", flush=True)
        return rows, estimator, dict(payload.get("params") or {}), best_key, dict(payload.get("best_row") or {})
    except Exception as error:
        print(f"[WARN] No reutilizo progreso de trials para {label}: {type(error).__name__}: {error}", flush=True)
        return empty


def _save_xgb_trial_progress(
    args: argparse.Namespace,
    out_dir: Optional[Path],
    checkpoint_phase: str,
    label: str,
    label_index: int,
    specs: Sequence[Dict[str, Any]],
    rows: Sequence[Dict[str, Any]],
    best_model: Any,
    best_params: Dict[str, Any],
    best_key: Optional[Tuple[float, ...]],
    best_row: Dict[str, Any],
    *,
    save_best_estimator: bool,
) -> None:
    if not _resume_enabled(args):
        return
    progress_path, best_path = _xgb_trial_progress_paths(args, out_dir, checkpoint_phase, label_index, label)
    if progress_path is None or best_path is None:
        return
    common = {
        "kind": "xgb_per_label_trial_progress",
        "created_at": _now_iso(),
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "training_signature": _training_signature(args),
        "specs_signature": _xgb_specs_signature(specs),
        "label": str(label),
        "label_index": int(label_index),
    }
    # El estimator se publica primero. Si Slurm corta entre ambas escrituras,
    # el JSON viejo sólo provoca repetir como máximo el último trial.
    if save_best_estimator and best_model is not None and best_key is not None:
        _atomic_joblib_dump({
            **common,
            "estimator": best_model,
            "params": best_params,
            "best_key": list(best_key),
            "best_row": best_row,
        }, best_path, compress=int(getattr(args, "checkpoint_compress", 3)))
    _atomic_json({**common, "rows": list(rows), "best_trial_index": best_row.get("trial_index")}, progress_path)


def _tune_xgb_one_label(
    args: argparse.Namespace,
    label: str,
    label_index: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    prefer_gpu: bool,
    out_dir: Optional[Path] = None,
    checkpoint_phase: str = "",
) -> Dict[str, Any]:
    try:
        import xgboost as xgb  # type: ignore
    except Exception as error:
        raise RuntimeError(f"xgboost no disponible: {error}")
    xgb_version = str(getattr(xgb, "__version__", "unknown"))
    specs = _xgb_trial_specs(args, y_train, label, label_index)
    rows, best_model, best_params, best_key, best_row = _load_xgb_trial_progress(
        args, out_dir, checkpoint_phase, label, label_index, specs
    )
    completed_trials = {int(row["trial_index"]) for row in rows if row.get("trial_index") is not None}
    gpu_any = False
    t_total = time.perf_counter()
    cpu_total = time.process_time()
    metric_name = str(args.xgb_label_selection_metric).lower()
    for trial_index, spec in enumerate(specs):
        if trial_index in completed_trials:
            continue
        t0 = time.perf_counter()
        try:
            model, params, gpu_used = _fit_xgb_candidate(
                args,
                xgb,
                xgb_version,
                y_train,
                X_train,
                X_valid,
                y_valid,
                spec,
                seed_offset=1000 * (label_index + 1) + trial_index,
                prefer_gpu=prefer_gpu,
                label=label,
            )
            valid_scores = np.asarray(model.predict_proba(X_valid))[:, 1]
            ap = float(average_precision_score(y_valid, valid_scores))
            threshold_stats = _best_binary_threshold_exact(
                y_valid,
                valid_scores,
                objective="f1",
                min_threshold=float(args.threshold_min),
                max_threshold=float(args.threshold_max),
                min_recall=0.0,
            )
            if metric_name == "f1":
                key = (float(threshold_stats["f1"]), ap)
            elif metric_name == "blend":
                blend = 0.5 * (ap + float(threshold_stats["f1"]))
                key = (blend, ap, float(threshold_stats["f1"]))
            else:
                key = (ap, float(threshold_stats["f1"]))
            row = {
                "label": label,
                "label_index": int(label_index),
                "trial_index": int(trial_index),
                "trial_kind": spec["trial_kind"],
                "weight_name": spec["weight_name"],
                "scale_pos_weight": float(spec["scale_pos_weight"]),
                "selection_metric": metric_name,
                "pr_auc": ap,
                "best_valid_f1": float(threshold_stats["f1"]),
                "best_valid_precision": float(threshold_stats["precision"]),
                "best_valid_recall": float(threshold_stats["recall"]),
                "best_valid_threshold": float(threshold_stats["threshold"]),
                "best_iteration": int(getattr(model, "best_iteration", -1) if getattr(model, "best_iteration", None) is not None else -1),
                "fit_seconds": float(time.perf_counter() - t0),
                "gpu_used": bool(gpu_used),
                "status": "ok",
                **{f"param_{name}": value for name, value in params.items() if name in {
                    "n_estimators", "learning_rate", "max_depth", "min_child_weight", "subsample",
                    "colsample_bytree", "gamma", "reg_alpha", "reg_lambda", "max_delta_step", "max_bin",
                }},
            }
            rows.append(row)
            gpu_any = gpu_any or bool(gpu_used)
            is_new_best = best_key is None or key > best_key
            if is_new_best:
                best_key = key
                best_model = model
                best_params = params
                best_row = row
            else:
                del model
            _save_xgb_trial_progress(
                args,
                out_dir,
                checkpoint_phase,
                label,
                label_index,
                specs,
                rows,
                best_model,
                best_params,
                best_key,
                best_row,
                save_best_estimator=is_new_best,
            )
            print(
                f"[XGB-TUNE] {label} trial={trial_index + 1}/{len(specs)} "
                f"AP={ap:.6f} F1*={threshold_stats['f1']:.6f} weight={spec['weight_name']} "
                f"best={'yes' if best_row is row else 'no'}",
                flush=True,
            )
        except Exception as error:
            rows.append({
                "label": label,
                "label_index": int(label_index),
                "trial_index": int(trial_index),
                "trial_kind": spec.get("trial_kind"),
                "weight_name": spec.get("weight_name"),
                "scale_pos_weight": spec.get("scale_pos_weight"),
                "fit_seconds": float(time.perf_counter() - t0),
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
            })
            _save_xgb_trial_progress(
                args,
                out_dir,
                checkpoint_phase,
                label,
                label_index,
                specs,
                rows,
                best_model,
                best_params,
                best_key,
                best_row,
                save_best_estimator=False,
            )
            print(f"[WARN] Trial XGBoost falló para {label}: {type(error).__name__}: {error}", flush=True)
    if best_model is None:
        raise RuntimeError(f"Todos los trials XGBoost fallaron para {label}")
    return {
        "label": label,
        "label_index": int(label_index),
        "backend": "xgboost",
        "weight_mode": str(best_row.get("weight_name", "per_label")),
        "estimator": best_model,
        "constant_proba": None,
        "fit_seconds": float(time.perf_counter() - t_total),
        "fit_cpu_seconds": float(time.process_time() - cpu_total),
        "gpu_used": bool(best_row.get("gpu_used", gpu_any)),
        "fit_error": None,
        "params": best_params,
        "xgboost_version": xgb_version,
        "selected_tuning_trial": best_row,
        "tuning_trials": rows,
    }


def _histgb_params_for_preset(args: argparse.Namespace) -> Dict[str, Any]:
    preset = str(args.histgb_preset or "strong").strip().lower()
    if preset == "fast":
        return dict(max_iter=220, learning_rate=0.07, max_leaf_nodes=31, l2_regularization=0.05, early_stopping=True, random_state=int(args.seed))
    if preset == "max":
        return dict(max_iter=750, learning_rate=0.035, max_leaf_nodes=63, l2_regularization=0.05, early_stopping=True, random_state=int(args.seed))
    return dict(max_iter=450, learning_rate=0.045, max_leaf_nodes=31, l2_regularization=0.05, early_stopping=True, random_state=int(args.seed))


def _forest_params_for_preset(args: argparse.Namespace, backend: str) -> Dict[str, Any]:
    preset = str(getattr(args, "forest_preset", "strong") or "strong").strip().lower()
    if preset == "fast":
        n_estimators, min_samples_leaf, max_features = 350, 1, "sqrt"
    elif preset == "max":
        n_estimators, min_samples_leaf, max_features = 1400, 1, "sqrt"
    else:
        n_estimators, min_samples_leaf, max_features = 800, 1, "sqrt"
    params: Dict[str, Any] = dict(
        n_estimators=n_estimators,
        max_features=max_features,
        min_samples_leaf=min_samples_leaf,
        random_state=int(args.seed),
        n_jobs=int(args.n_jobs),
        class_weight=None,
    )
    if backend == "random_forest":
        params.update(bootstrap=True, max_samples=None)
    else:
        params.update(bootstrap=False)
    return params


def _logistic_params_for_preset(args: argparse.Namespace) -> Dict[str, Any]:
    preset = str(getattr(args, "linear_preset", "strong") or "strong").strip().lower()
    max_iter = 500 if preset == "fast" else (3000 if preset == "max" else 1500)
    c_value = 1.5 if preset == "max" else 1.0
    return dict(
        C=c_value,
        penalty="l2",
        solver="saga",
        max_iter=max_iter,
        tol=1e-4 if preset != "max" else 5e-5,
        n_jobs=int(args.n_jobs),
        random_state=int(args.seed),
        class_weight=None,
    )


def _linear_svc_params_for_preset(args: argparse.Namespace) -> Dict[str, Any]:
    preset = str(getattr(args, "linear_preset", "strong") or "strong").strip().lower()
    max_iter = 3000 if preset == "fast" else (12000 if preset == "max" else 7000)
    c_value = 1.0 if preset != "max" else 1.5
    return dict(C=c_value, max_iter=max_iter, tol=1e-4 if preset != "max" else 5e-5, class_weight=None, dual=False, random_state=int(args.seed))


def _sgd_params_for_preset(args: argparse.Namespace) -> Dict[str, Any]:
    preset = str(getattr(args, "linear_preset", "strong") or "strong").strip().lower()
    max_iter = 800 if preset == "fast" else (3500 if preset == "max" else 1800)
    return dict(
        loss="log_loss",
        penalty="elasticnet" if preset == "max" else "l2",
        alpha=1e-5 if preset == "max" else 1e-4,
        l1_ratio=0.10,
        max_iter=max_iter,
        tol=1e-4 if preset != "max" else 5e-5,
        random_state=int(args.seed),
        n_jobs=int(args.n_jobs),
        class_weight=None,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
    )


def _lgbm_params_for_preset(args: argparse.Namespace, y_binary: np.ndarray, weight_mode: str, device: str) -> Dict[str, Any]:
    preset = str(getattr(args, "lgbm_preset", "strong") or "strong").strip().lower()
    if preset == "fast":
        params = dict(n_estimators=550, learning_rate=0.055, num_leaves=31, max_depth=-1, min_child_samples=20, subsample=0.90, colsample_bytree=0.90, reg_alpha=0.0, reg_lambda=1.0)
    elif preset == "max":
        params = dict(n_estimators=1800, learning_rate=0.022, num_leaves=63, max_depth=-1, min_child_samples=10, subsample=0.95, colsample_bytree=0.95, reg_alpha=0.0, reg_lambda=1.0)
    else:
        params = dict(n_estimators=950, learning_rate=0.035, num_leaves=47, max_depth=-1, min_child_samples=15, subsample=0.90, colsample_bytree=0.90, reg_alpha=0.0, reg_lambda=1.0)
    params.update(
        objective="binary",
        random_state=int(args.seed),
        n_jobs=int(args.n_jobs),
        scale_pos_weight=_compute_scale_pos_weight(y_binary, weight_mode, float(args.scale_pos_weight_cap)),
        verbosity=-1,
    )
    if device == "cuda":
        params.update(device_type="gpu")
    else:
        params.update(device_type="cpu")
    return params


def _catboost_params_for_preset(args: argparse.Namespace, y_binary: np.ndarray, weight_mode: str, device: str) -> Dict[str, Any]:
    preset = str(getattr(args, "catboost_preset", "strong") or "strong").strip().lower()
    if preset == "fast":
        params = dict(iterations=550, learning_rate=0.055, depth=6, l2_leaf_reg=3.0)
    elif preset == "max":
        params = dict(iterations=1800, learning_rate=0.022, depth=8, l2_leaf_reg=3.0)
    else:
        params = dict(iterations=950, learning_rate=0.035, depth=7, l2_leaf_reg=3.0)
    params.update(
        loss_function="Logloss",
        eval_metric="Logloss",
        random_seed=int(args.seed),
        thread_count=int(args.n_jobs),
        verbose=False,
        allow_writing_files=False,
        scale_pos_weight=_compute_scale_pos_weight(y_binary, weight_mode, float(args.scale_pos_weight_cap)),
    )
    if int(getattr(args, "catboost_early_stopping_rounds", 0) or 0) > 0:
        params.update(od_type="Iter", od_wait=int(args.catboost_early_stopping_rounds))
    if device == "cuda":
        params.update(task_type="GPU", devices="0")
    else:
        params.update(task_type="CPU")
    return params


def train_one_label_estimator(
    args: argparse.Namespace,
    backend: str,
    weight_mode: str,
    label: str,
    label_index: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: Optional[np.ndarray],
    y_valid: Optional[np.ndarray],
    prefer_gpu: bool,
    out_dir: Optional[Path] = None,
    checkpoint_phase: str = "",
) -> Dict[str, Any]:
    y_train = np.asarray(y_train).astype(int)
    unique = np.unique(y_train)
    rec: Dict[str, Any] = {
        "label": label,
        "label_index": int(label_index),
        "backend": backend,
        "weight_mode": weight_mode,
        "estimator": None,
        "constant_proba": None,
        "fit_seconds": 0.0,
        "fit_cpu_seconds": 0.0,
        "gpu_used": False,
        "fit_error": None,
        "params": {},
    }
    if len(unique) < 2:
        p = float(unique[0]) if len(unique) else 0.0
        rec["backend"] = "constant"
        rec["constant_proba"] = p
        return rec

    t0 = time.perf_counter()
    cpu0 = time.process_time()
    backend = _canonical_backend_name(backend)
    if backend == "xgboost":
        if (
            bool(getattr(args, "xgb_per_label_tuning", True))
            and X_valid is not None
            and y_valid is not None
            and len(np.unique(y_valid)) >= 2
        ):
            return _tune_xgb_one_label(
                args=args,
                label=label,
                label_index=label_index,
                X_train=X_train,
                y_train=y_train,
                X_valid=X_valid,
                y_valid=np.asarray(y_valid).astype(int),
                prefer_gpu=prefer_gpu,
                out_dir=out_dir,
                checkpoint_phase=checkpoint_phase,
            )
        try:
            import xgboost as xgb  # type: ignore
        except Exception as e:
            raise RuntimeError(f"xgboost no disponible: {e}")
        xgb_version = str(getattr(xgb, "__version__", "unknown"))
        device_order = ["cuda", "cpu"] if (prefer_gpu and _want_gpu(args)) else ["cpu"]
        has_usable_valid = bool(
            X_valid is not None
            and y_valid is not None
            and len(np.unique(np.asarray(y_valid).astype(int))) >= 2
        )
        last_error: Optional[Exception] = None
        for device in device_order:
            params = _xgb_params_for_preset(args, y_train, weight_mode, device, xgb_version)
            # XGBoost exige eval_set cuando early stopping está activo. Las
            # etiquetas raras pueden quedar con una sola clase en tune.
            if not has_usable_valid:
                params.pop("early_stopping_rounds", None)
            try:
                model = xgb.XGBClassifier(**params)
                fit_kwargs: Dict[str, Any] = {"verbose": False}
                if has_usable_valid:
                    fit_kwargs["eval_set"] = [(X_valid, y_valid)]
                model.fit(X_train, y_train, **fit_kwargs)
                rec.update({
                    "estimator": model,
                    "backend": "xgboost",
                    "gpu_used": bool(device == "cuda"),
                    "params": params,
                    "xgboost_version": xgb_version,
                })
                break
            except TypeError as e:
                # Compatibilidad con XGBoost más viejo si no acepta early_stopping_rounds en constructor.
                last_error = e
                if "early_stopping_rounds" in params:
                    params2 = dict(params)
                    params2.pop("early_stopping_rounds", None)
                    try:
                        model = xgb.XGBClassifier(**params2)
                        fit_kwargs = {"verbose": False}
                        if has_usable_valid:
                            fit_kwargs["eval_set"] = [(X_valid, y_valid)]
                        model.fit(X_train, y_train, **fit_kwargs)
                        rec.update({"estimator": model, "backend": "xgboost", "gpu_used": bool(device == "cuda"), "params": params2, "xgboost_version": xgb_version})
                        break
                    except Exception as e2:
                        last_error = e2
                if device == "cuda":
                    print(f"[WARN] XGBoost CUDA falló para {label}; pruebo CPU. Error: {last_error}", flush=True)
                    continue
                raise
            except Exception as e:
                last_error = e
                if device == "cuda":
                    print(f"[WARN] XGBoost CUDA falló para {label}; pruebo CPU. Error: {e}", flush=True)
                    continue
                raise
        if rec["estimator"] is None:
            raise RuntimeError(f"No pude entrenar XGBoost para {label}: {last_error}")

    elif backend == "lightgbm":
        try:
            import lightgbm as lgb  # type: ignore
        except Exception as e:
            raise RuntimeError(f"lightgbm no disponible: {e}")
        lgbm_version = str(getattr(lgb, "__version__", "unknown"))
        device_order = ["cuda", "cpu"] if (prefer_gpu and _want_gpu(args)) else ["cpu"]
        last_error: Optional[Exception] = None
        for device in device_order:
            params = _lgbm_params_for_preset(args, y_train, weight_mode, device)
            try:
                model = lgb.LGBMClassifier(**params)
                fit_kwargs: Dict[str, Any] = {}
                callbacks = []
                if int(getattr(args, "lgbm_early_stopping_rounds", 0) or 0) > 0 and X_valid is not None and y_valid is not None and len(np.unique(y_valid)) >= 2:
                    fit_kwargs["eval_set"] = [(X_valid, y_valid)]
                    fit_kwargs["eval_metric"] = "binary_logloss"
                    try:
                        callbacks.append(lgb.early_stopping(int(args.lgbm_early_stopping_rounds), verbose=False))
                    except Exception:
                        callbacks = []
                if callbacks:
                    fit_kwargs["callbacks"] = callbacks
                model.fit(X_train, y_train, **fit_kwargs)
                rec.update({"estimator": model, "backend": "lightgbm", "gpu_used": bool(device == "cuda"), "params": params, "lightgbm_version": lgbm_version})
                break
            except Exception as e:
                last_error = e
                if device == "cuda":
                    print(f"[WARN] LightGBM GPU falló para {label}; pruebo CPU. Error: {e}", flush=True)
                    continue
                raise
        if rec["estimator"] is None:
            raise RuntimeError(f"No pude entrenar LightGBM para {label}: {last_error}")

    elif backend == "catboost":
        try:
            from catboost import CatBoostClassifier  # type: ignore
            import catboost as _catboost  # type: ignore
        except Exception as e:
            raise RuntimeError(f"catboost no disponible: {e}")
        catboost_version = str(getattr(_catboost, "__version__", "unknown"))
        device_order = ["cuda", "cpu"] if (prefer_gpu and _want_gpu(args)) else ["cpu"]
        last_error: Optional[Exception] = None
        for device in device_order:
            params = _catboost_params_for_preset(args, y_train, weight_mode, device)
            try:
                model = CatBoostClassifier(**params)
                fit_kwargs: Dict[str, Any] = {}
                if X_valid is not None and y_valid is not None and len(np.unique(y_valid)) >= 2:
                    fit_kwargs["eval_set"] = (X_valid, y_valid)
                    fit_kwargs["use_best_model"] = True
                model.fit(X_train, y_train, **fit_kwargs)
                rec.update({"estimator": model, "backend": "catboost", "gpu_used": bool(device == "cuda"), "params": params, "catboost_version": catboost_version})
                break
            except Exception as e:
                last_error = e
                if device == "cuda":
                    print(f"[WARN] CatBoost GPU falló para {label}; pruebo CPU. Error: {e}", flush=True)
                    continue
                raise
        if rec["estimator"] is None:
            raise RuntimeError(f"No pude entrenar CatBoost para {label}: {last_error}")

    elif backend == "histgb":
        params = _histgb_params_for_preset(args)
        model = HistGradientBoostingClassifier(**params)
        if len(X_train) < max(50, 3 * 2):
            try:
                model.set_params(early_stopping=False)
            except Exception:
                pass
        sw = _binary_sample_weight(y_train, weight_mode, float(args.scale_pos_weight_cap))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if sw is None:
                model.fit(X_train, y_train)
            else:
                model.fit(X_train, y_train, sample_weight=sw)
        rec.update({"estimator": model, "backend": "histgb", "gpu_used": False, "params": params})

    elif backend in {"extra_trees", "random_forest"}:
        params = _forest_params_for_preset(args, backend)
        model = ExtraTreesClassifier(**params) if backend == "extra_trees" else RandomForestClassifier(**params)
        sw = _binary_sample_weight(y_train, weight_mode, float(args.scale_pos_weight_cap))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if sw is None:
                model.fit(X_train, y_train)
            else:
                model.fit(X_train, y_train, sample_weight=sw)
        rec.update({"estimator": model, "backend": backend, "gpu_used": False, "params": params})

    elif backend == "logistic_regression":
        params = _logistic_params_for_preset(args)
        model = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(**params))])
        sw = _binary_sample_weight(y_train, weight_mode, float(args.scale_pos_weight_cap))
        fit_kwargs = {"clf__sample_weight": sw} if sw is not None else {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train, y_train, **fit_kwargs)
        rec.update({"estimator": model, "backend": "logistic_regression", "gpu_used": False, "params": params})

    elif backend == "linear_svc":
        params = _linear_svc_params_for_preset(args)
        model = Pipeline([("scaler", StandardScaler()), ("clf", LinearSVC(**params))])
        sw = _binary_sample_weight(y_train, weight_mode, float(args.scale_pos_weight_cap))
        fit_kwargs = {"clf__sample_weight": sw} if sw is not None else {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train, y_train, **fit_kwargs)
        rec.update({"estimator": model, "backend": "linear_svc", "gpu_used": False, "params": params})

    elif backend == "sgd_logloss":
        params = _sgd_params_for_preset(args)
        try:
            clf = SGDClassifier(**params)
        except TypeError:
            params = dict(params)
            params["loss"] = "log"
            clf = SGDClassifier(**params)
        model = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        sw = _binary_sample_weight(y_train, weight_mode, float(args.scale_pos_weight_cap))
        fit_kwargs = {"clf__sample_weight": sw} if sw is not None else {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                model.fit(X_train, y_train, **fit_kwargs)
            except ValueError as e:
                if "log_loss" in str(e):
                    params = dict(params)
                    params["loss"] = "log"
                    model = Pipeline([("scaler", StandardScaler()), ("clf", SGDClassifier(**params))])
                    model.fit(X_train, y_train, **fit_kwargs)
                else:
                    raise
        rec.update({"estimator": model, "backend": "sgd_logloss", "gpu_used": False, "params": params})

    else:
        raise ValueError(f"Backend no soportado: {backend}")

    rec["fit_seconds"] = float(time.perf_counter() - t0)
    rec["fit_cpu_seconds"] = float(time.process_time() - cpu0)
    return rec


def train_ovr_records(
    args: argparse.Namespace,
    backend: str,
    weight_mode: str,
    labels: Sequence[str],
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_valid: Optional[np.ndarray],
    Y_valid: Optional[np.ndarray],
    prefer_gpu: bool,
    desc: str,
    out_dir: Optional[Path] = None,
    checkpoint_phase: str = "",
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    resumed_labels = 0
    for j, lab in enumerate(tqdm(labels, desc=desc, mininterval=5)):
        rec = _load_label_record_checkpoint(args, out_dir, checkpoint_phase, backend, weight_mode, j, lab)
        if rec is not None:
            records.append(rec)
            resumed_labels += 1
            continue
        rec = train_one_label_estimator(
            args=args,
            backend=backend,
            weight_mode=weight_mode,
            label=lab,
            label_index=j,
            X_train=X_train,
            y_train=Y_train[:, j],
            X_valid=X_valid,
            y_valid=None if Y_valid is None else Y_valid[:, j],
            prefer_gpu=prefer_gpu,
            out_dir=out_dir,
            checkpoint_phase=checkpoint_phase,
        )
        _save_label_record_checkpoint(args, out_dir, checkpoint_phase, backend, weight_mode, j, lab, rec)
        records.append(rec)
    if resumed_labels:
        print(f"[RESUME] {desc}: reutilicé {resumed_labels}/{len(labels)} modelos binarios por etiqueta.", flush=True)
    return records


def _write_xgb_tuning_records(records: Sequence[Dict[str, Any]], out_dir: Optional[Path]) -> None:
    if out_dir is None:
        return
    trial_rows: List[Dict[str, Any]] = []
    selected_rows: List[Dict[str, Any]] = []
    for record in records:
        selected = dict(record.get("selected_tuning_trial") or {})
        selected_index = selected.get("trial_index")
        if selected:
            selected_rows.append(selected)
        for row0 in record.get("tuning_trials") or []:
            row = dict(row0)
            row["selected"] = bool(row.get("trial_index") == selected_index and row.get("label") == record.get("label"))
            trial_rows.append(row)
    if trial_rows:
        _atomic_csv(pd.DataFrame(trial_rows), out_dir / "xgb_tuning_trials_by_label.csv")
    if selected_rows:
        _atomic_csv(pd.DataFrame(selected_rows), out_dir / "xgb_best_params_by_label.csv")


def predict_score_matrix(records: Sequence[Dict[str, Any]], X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    scores = np.zeros((X.shape[0], len(records)), dtype=np.float32)
    for j, rec in enumerate(records):
        if rec.get("backend") == "constant" or rec.get("estimator") is None:
            scores[:, j] = float(rec.get("constant_proba") or 0.0)
            continue
        est = rec["estimator"]
        try:
            p = est.predict_proba(X)
            if isinstance(p, list):
                # Algunas APIs multilabel devuelven lista; no debería pasar aquí.
                p = p[0]
            p_arr = np.asarray(p)
            scores[:, j] = p_arr[:, 1] if p_arr.ndim == 2 and p_arr.shape[1] > 1 else p_arr.ravel()
        except Exception:
            # Último recurso: decisión/predicción dura.
            try:
                if hasattr(est, "decision_function"):
                    d = np.asarray(est.decision_function(X)).ravel()
                    scores[:, j] = 1.0 / (1.0 + np.exp(-d))
                else:
                    scores[:, j] = np.asarray(est.predict(X)).astype(float).ravel()
            except Exception as e:
                raise RuntimeError(f"No pude predecir scores para label={rec.get('label')}: {e}")
    return scores


def _fbeta_from_pr(precision: float, recall: float, beta: float) -> float:
    if precision <= 0.0 and recall <= 0.0:
        return 0.0
    b2 = beta * beta
    den = b2 * precision + recall
    if den <= 0:
        return 0.0
    return (1 + b2) * precision * recall / den


def optimize_thresholds(
    Y_true: np.ndarray,
    scores: np.ndarray,
    labels: Sequence[str],
    objective: str = "f1",
    grid_size: int = 199,
    min_threshold: float = 0.01,
    max_threshold: float = 0.99,
    min_recall: float = 0.0,
) -> Tuple[np.ndarray, pd.DataFrame]:
    objective_l = (objective or "f1").lower()
    thresholds = np.full(scores.shape[1], 0.5, dtype=np.float32)
    rows: List[Dict[str, Any]] = []
    del grid_size  # compatibilidad CLI; el barrido ahora usa todos los scores distintos.
    for j, lab in enumerate(labels):
        yt = Y_true[:, j].astype(int)
        sc = scores[:, j].astype(float)
        support = int(yt.sum())
        if support <= 0 or len(np.unique(yt)) < 2:
            rows.append({"label": lab, "threshold": 0.5, "objective": objective_l, "valid_support": support, "precision": None, "recall": None, "f1": None, "score": None})
            continue
        best = _best_binary_threshold_exact(
            yt,
            sc,
            objective=objective_l,
            min_threshold=float(min_threshold),
            max_threshold=float(max_threshold),
            min_recall=float(min_recall),
        )
        thresholds[j] = float(best["threshold"])
        rows.append({
            "label": lab,
            "threshold": float(best["threshold"]),
            "objective": objective_l,
            "search": "all_distinct_scores",
            "valid_support": support,
            "precision": float(best["precision"]),
            "recall": float(best["recall"]),
            "f1": float(best["f1"]),
            "score": float(best["score"]),
        })
    return thresholds, pd.DataFrame(rows)


def _threshold_curve_f1_exact(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    min_threshold: float,
    max_threshold: float,
    extra_threshold: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    """Curva PR/F1 exacta para todos los cortes observados, en orden ascendente.

    Además de las métricas devuelve el orden de los scores y la posición de
    cada corte. Así el optimizador conjunto puede recalcular NORMAL en O(n)
    por coordenada, sin construir una matriz filas x umbrales.
    """
    yt = np.asarray(y_true, dtype=np.int8).ravel()
    sc = np.nan_to_num(
        np.asarray(scores, dtype=np.float32).ravel(),
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
    if len(yt) != len(sc):
        raise ValueError("y_true y scores deben tener la misma longitud")

    low = float(min(min_threshold, max_threshold))
    high = float(max(min_threshold, max_threshold))
    observed = sc[(sc >= low) & (sc <= high)]
    extra = []
    if extra_threshold is not None and np.isfinite(float(extra_threshold)):
        extra.append(float(extra_threshold))
    thresholds = np.unique(np.r_[observed, low, high, extra]).astype(np.float32)

    order = np.argsort(sc, kind="mergesort").astype(np.int32, copy=False)
    sorted_scores = sc[order]
    positions = np.searchsorted(sorted_scores, thresholds, side="left").astype(np.int32, copy=False)
    prefix_positive = np.r_[0, np.cumsum(yt[order], dtype=np.int64)]
    positives = float(yt.sum())
    predicted = (len(yt) - positions).astype(np.float64)
    tp = positives - prefix_positive[positions].astype(np.float64)
    fp = predicted - tp
    fn = positives - tp
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(tp),
        where=(precision + recall) > 0,
    )
    return {
        "thresholds": thresholds,
        "precision": precision.astype(np.float32),
        "recall": recall.astype(np.float32),
        "f1": f1.astype(np.float32),
        "order": order,
        "positions": positions,
    }


def _binary_prf_at_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> Tuple[float, float, float]:
    yt = np.asarray(y_true, dtype=np.int8).ravel()
    pred = np.asarray(scores, dtype=float).ravel() >= float(threshold)
    tp = int(np.count_nonzero((yt == 1) & pred))
    fp = int(np.count_nonzero((yt == 0) & pred))
    fn = int(np.count_nonzero((yt == 1) & ~pred))
    precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
    f1 = float(2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


def _normal_metrics_from_any_attack(
    true_any_attack: np.ndarray,
    pred_any_attack: np.ndarray,
) -> Dict[str, float]:
    true_attack = np.asarray(true_any_attack, dtype=bool).ravel()
    pred_attack = np.asarray(pred_any_attack, dtype=bool).ravel()
    normal = ~true_attack
    tn = int(np.count_nonzero(normal & ~pred_attack))
    fp = int(np.count_nonzero(normal & pred_attack))
    fn = int(np.count_nonzero(true_attack & ~pred_attack))
    precision = float(tn / (tn + fn)) if (tn + fn) else 0.0
    recall = float(tn / (tn + fp)) if (tn + fp) else 0.0
    f1 = float(2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "normal_support": int(np.count_nonzero(normal)),
    }


def _joint_state_key(
    normal_value: float,
    macro_f1: float,
    worst_f1: float,
    normal_target: float,
) -> Tuple[float, float, float, float]:
    """Orden target-first: al llegar a NORMAL, recupera F1 de etiquetas."""
    if float(normal_value) >= float(normal_target) - 1e-12:
        return (1.0, float(macro_f1), float(worst_f1), float(normal_value))
    return (0.0, float(normal_value), float(macro_f1), float(worst_f1))


def _strictly_better_tuple(
    candidate: Sequence[float],
    current: Sequence[float],
    eps: float = 1e-12,
) -> bool:
    for new_value, old_value in zip(candidate, current):
        if float(new_value) > float(old_value) + eps:
            return True
        if float(new_value) < float(old_value) - eps:
            return False
    return False


def _keep_best_indices(
    indices: np.ndarray,
    values: np.ndarray,
    *,
    maximize: bool = True,
    eps: float = 1e-12,
) -> np.ndarray:
    """Filtro vectorizado y determinista para desempates lexicográficos."""
    if len(indices) <= 1:
        return indices
    selected_values = np.asarray(values, dtype=float)[indices]
    finite = np.isfinite(selected_values)
    if not np.any(finite):
        return indices[:1]
    indices = indices[finite]
    selected_values = selected_values[finite]
    best = float(np.max(selected_values) if maximize else np.min(selected_values))
    if maximize:
        return indices[selected_values >= best - eps]
    return indices[selected_values <= best + eps]


def optimize_thresholds_joint_targets(
    Y_true: np.ndarray,
    scores: np.ndarray,
    labels: Sequence[str],
    initial_thresholds: np.ndarray,
    *,
    label_f1_target: float = 0.85,
    normal_target: float = 0.99,
    normal_metric: str = "recall",
    min_threshold: float = 0.001,
    max_threshold: float = 0.999,
    min_recall: float = 0.0,
    max_passes: int = 6,
    fixed_labels: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Optimiza conjuntamente thresholds usando exclusivamente calibration.

    La búsqueda es un coordinate descent determinista sobre todos los scores
    distintos. Para cada etiqueta sólo permite cortes con F1 >= target cuando
    ese target es factible en calibration; si no lo es, conserva el máximo F1
    alcanzable. NORMAL se optimiza sobre la unión de predicciones de ataque.
    El método es target-first y no consulta test ni entrena estimadores.
    """
    Y = np.asarray(Y_true, dtype=np.int8)
    S = np.nan_to_num(np.asarray(scores, dtype=np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    thresholds_before = np.asarray(initial_thresholds, dtype=np.float32).copy()
    if Y.ndim != 2 or S.shape != Y.shape:
        raise ValueError("Y_true y scores deben ser matrices 2D con la misma forma")
    if Y.shape[1] != len(labels) or thresholds_before.shape != (len(labels),):
        raise ValueError("labels/initial_thresholds no coinciden con las columnas de Y_true")
    normal_metric_l = str(normal_metric or "recall").strip().lower()
    if normal_metric_l not in {"recall", "f1"}:
        raise ValueError("normal_metric debe ser 'recall' o 'f1'")
    if not 0.0 <= float(label_f1_target) <= 1.0 or not 0.0 <= float(normal_target) <= 1.0:
        raise ValueError("Los targets deben estar entre 0 y 1")

    fixed = np.zeros(len(labels), dtype=bool) if fixed_labels is None else np.asarray(fixed_labels, dtype=bool)
    if fixed.shape != (len(labels),):
        raise ValueError("fixed_labels debe tener una entrada por etiqueta")

    curves: List[Optional[Dict[str, Any]]] = []
    thresholds = thresholds_before.copy()
    label_f1 = np.zeros(len(labels), dtype=np.float64)
    feasibility: List[Optional[bool]] = []
    initialization_moves: List[Dict[str, Any]] = []

    # Genera y comprime la curva de cada etiqueta una sola vez. Se retienen
    # sólo los cortes permitidos y el orden de filas necesario para NORMAL.
    for j, label in enumerate(labels):
        p0, r0, f0 = _binary_prf_at_threshold(Y[:, j], S[:, j], float(thresholds[j]))
        if fixed[j] or int(Y[:, j].sum()) == 0 or len(np.unique(Y[:, j])) < 2:
            label_f1[j] = f0
            feasibility.append(None if fixed[j] else False)
            curves.append(None)
            continue

        full = _threshold_curve_f1_exact(
            Y[:, j],
            S[:, j],
            min_threshold=float(min_threshold),
            max_threshold=float(max_threshold),
            extra_threshold=float(thresholds[j]),
        )
        base_allowed = np.ones(len(full["thresholds"]), dtype=bool)
        if float(min_recall) > 0:
            base_allowed &= full["recall"] >= float(min_recall) - 1e-12
            if not np.any(base_allowed):
                base_allowed[:] = True
        reaches_target = base_allowed & (full["f1"] >= float(label_f1_target) - 1e-12)
        target_feasible = bool(np.any(reaches_target))
        feasibility.append(target_feasible)
        if target_feasible:
            allowed = reaches_target
        else:
            best_f1 = float(np.max(full["f1"][base_allowed]))
            allowed = base_allowed & (full["f1"] >= best_f1 - 1e-12)
        allowed_idx = np.flatnonzero(allowed)

        # Para NORMAL-recall el corte más alto que todavía satisface F1>=target
        # domina en falsos positivos normales. Si el target no es factible (o
        # NORMAL se optimiza por F1), se parte del máximo F1 de la etiqueta.
        if target_feasible and normal_metric_l == "recall":
            start = _keep_best_indices(allowed_idx, full["thresholds"], maximize=True)
        else:
            start = _keep_best_indices(allowed_idx, full["f1"], maximize=True)
            distance = np.abs(full["thresholds"].astype(float) - float(thresholds_before[j]))
            start = _keep_best_indices(start, distance, maximize=False)
            start = _keep_best_indices(start, full["precision"], maximize=True)
            start = _keep_best_indices(start, full["thresholds"], maximize=True)
        start_idx = int(start[0])
        new_threshold = float(full["thresholds"][start_idx])
        if abs(new_threshold - float(thresholds[j])) > 1e-12:
            initialization_moves.append({
                "phase": "label_f1_target_initialization",
                "label": str(label),
                "threshold_before": float(thresholds[j]),
                "threshold_after": new_threshold,
                "label_f1_before": float(f0),
                "label_f1_after": float(full["f1"][start_idx]),
            })
        thresholds[j] = new_threshold
        label_f1[j] = float(full["f1"][start_idx])
        curves.append({
            "thresholds": full["thresholds"][allowed_idx],
            "precision": full["precision"][allowed_idx],
            "recall": full["recall"][allowed_idx],
            "f1": full["f1"][allowed_idx],
            "positions": full["positions"][allowed_idx],
            "order": full["order"],
            "target_feasible": target_feasible,
        })

    pred = S >= thresholds.reshape(1, -1)
    coverage = pred.sum(axis=1, dtype=np.int32)
    true_any_attack = Y.sum(axis=1) > 0
    normal_before = _normal_metrics_from_any_attack(
        true_any_attack,
        (S >= thresholds_before.reshape(1, -1)).any(axis=1),
    )
    normal_initial = _normal_metrics_from_any_attack(true_any_attack, coverage > 0)
    history: List[Dict[str, Any]] = list(initialization_moves)
    accepted_coordinate_moves = 0
    completed_passes = 0

    for pass_index in range(max(0, int(max_passes))):
        completed_passes = pass_index + 1
        changed = False
        order_labels = range(len(labels)) if pass_index % 2 == 0 else range(len(labels) - 1, -1, -1)
        for j in order_labels:
            curve = curves[j]
            if curve is None or len(curve["thresholds"]) == 0:
                continue
            old_pred = pred[:, j]
            other_any = (coverage - old_pred.astype(np.int32)) > 0
            exposed = ~other_any
            row_order = curve["order"]
            normal_exposed_sorted = ((~true_any_attack) & exposed)[row_order]
            attack_exposed_sorted = (true_any_attack & exposed)[row_order]
            normal_prefix = np.r_[0, np.cumsum(normal_exposed_sorted, dtype=np.int64)]
            attack_prefix = np.r_[0, np.cumsum(attack_exposed_sorted, dtype=np.int64)]
            positions = curve["positions"]
            normal_exposed_total = int(normal_prefix[-1])
            normal_predicted_by_j = normal_exposed_total - normal_prefix[positions]
            fixed_normal_fp = int(np.count_nonzero((~true_any_attack) & other_any))
            tn = int(np.count_nonzero(~true_any_attack)) - fixed_normal_fp - normal_predicted_by_j
            fp = fixed_normal_fp + normal_predicted_by_j
            fn = attack_prefix[positions]
            normal_precision = np.divide(
                tn,
                tn + fn,
                out=np.zeros(len(tn), dtype=float),
                where=(tn + fn) > 0,
            )
            normal_recall = np.divide(
                tn,
                tn + fp,
                out=np.zeros(len(tn), dtype=float),
                where=(tn + fp) > 0,
            )
            normal_f1 = np.divide(
                2.0 * normal_precision * normal_recall,
                normal_precision + normal_recall,
                out=np.zeros(len(tn), dtype=float),
                where=(normal_precision + normal_recall) > 0,
            )
            normal_values = normal_recall if normal_metric_l == "recall" else normal_f1
            macro_values = (float(label_f1.sum()) - float(label_f1[j]) + curve["f1"]) / max(1, len(labels))
            if len(labels) > 1:
                other_worst = float(np.min(np.delete(label_f1, j)))
                worst_values = np.minimum(other_worst, curve["f1"])
            else:
                worst_values = curve["f1"].astype(float)

            candidate_idx = np.arange(len(curve["thresholds"]), dtype=np.int64)
            meets_normal = normal_values >= float(normal_target) - 1e-12
            if np.any(meets_normal):
                candidate_idx = candidate_idx[meets_normal]
                candidate_idx = _keep_best_indices(candidate_idx, macro_values, maximize=True)
                candidate_idx = _keep_best_indices(candidate_idx, worst_values, maximize=True)
                candidate_idx = _keep_best_indices(candidate_idx, normal_values, maximize=True)
            else:
                candidate_idx = _keep_best_indices(candidate_idx, normal_values, maximize=True)
                candidate_idx = _keep_best_indices(candidate_idx, macro_values, maximize=True)
                candidate_idx = _keep_best_indices(candidate_idx, worst_values, maximize=True)
            candidate_idx = _keep_best_indices(candidate_idx, curve["f1"], maximize=True)
            distance = np.abs(curve["thresholds"].astype(float) - float(thresholds_before[j]))
            candidate_idx = _keep_best_indices(candidate_idx, distance, maximize=False)
            candidate_idx = _keep_best_indices(candidate_idx, curve["precision"], maximize=True)
            candidate_idx = _keep_best_indices(candidate_idx, curve["thresholds"], maximize=True)
            best_idx = int(candidate_idx[0])

            current_normal = _normal_metrics_from_any_attack(true_any_attack, coverage > 0)
            current_normal_value = float(current_normal[normal_metric_l])
            current_key = _joint_state_key(
                current_normal_value,
                float(np.mean(label_f1)) if len(label_f1) else 0.0,
                float(np.min(label_f1)) if len(label_f1) else 0.0,
                float(normal_target),
            )
            candidate_key = _joint_state_key(
                float(normal_values[best_idx]),
                float(macro_values[best_idx]),
                float(worst_values[best_idx]),
                float(normal_target),
            )
            if not _strictly_better_tuple(candidate_key, current_key):
                continue

            threshold_before = float(thresholds[j])
            label_f1_before = float(label_f1[j])
            new_threshold = float(curve["thresholds"][best_idx])
            new_pred = S[:, j] >= new_threshold
            coverage += new_pred.astype(np.int32) - old_pred.astype(np.int32)
            pred[:, j] = new_pred
            thresholds[j] = new_threshold
            label_f1[j] = float(curve["f1"][best_idx])
            history.append({
                "phase": "joint_coordinate_descent",
                "pass": int(pass_index + 1),
                "label": str(labels[j]),
                "threshold_before": threshold_before,
                "threshold_after": new_threshold,
                "label_f1_before": label_f1_before,
                "label_f1_after": float(label_f1[j]),
                f"normal_{normal_metric_l}_before": current_normal_value,
                f"normal_{normal_metric_l}_after": float(normal_values[best_idx]),
                "normal_precision_after": float(normal_precision[best_idx]),
                "normal_recall_after": float(normal_recall[best_idx]),
                "normal_f1_after": float(normal_f1[best_idx]),
            })
            accepted_coordinate_moves += 1
            changed = True
        if not changed:
            break

    normal_final = _normal_metrics_from_any_attack(true_any_attack, coverage > 0)
    per_label_rows: List[Dict[str, Any]] = []
    final_label_f1 = np.zeros(len(labels), dtype=float)
    for j, label in enumerate(labels):
        p_before, r_before, f_before = _binary_prf_at_threshold(
            Y[:, j], S[:, j], float(thresholds_before[j])
        )
        p_final, r_final, f_final = _binary_prf_at_threshold(
            Y[:, j], S[:, j], float(thresholds[j])
        )
        final_label_f1[j] = f_final
        per_label_rows.append({
            "label": str(label),
            "optimization_eligible": bool(not fixed[j] and curves[j] is not None),
            "fixed_by_calibration_fallback": bool(fixed[j]),
            "target_feasible_on_calibration": feasibility[j],
            "support": int(Y[:, j].sum()),
            "target_metric": "f1",
            "target": float(label_f1_target),
            "target_met": bool(f_final >= float(label_f1_target) - 1e-12),
            "threshold_before_joint": float(thresholds_before[j]),
            "threshold_joint": float(thresholds[j]),
            "precision_before_joint": p_before,
            "recall_before_joint": r_before,
            "f1_before_joint": f_before,
            "precision_joint": p_final,
            "recall_joint": r_final,
            "f1_joint": f_final,
        })

    normal_value_before = float(normal_before[normal_metric_l])
    normal_value_final = float(normal_final[normal_metric_l])
    report: Dict[str, Any] = {
        "enabled": True,
        "data_role": "calibration_only",
        "search": "deterministic_coordinate_descent_all_distinct_scores",
        "n_rows": int(len(Y)),
        "n_labels": int(len(labels)),
        "label_f1_target": float(label_f1_target),
        "normal_metric": normal_metric_l,
        "normal_target": float(normal_target),
        "max_passes": int(max_passes),
        "completed_passes": int(completed_passes),
        "accepted_coordinate_moves": int(accepted_coordinate_moves),
        "fallback_fixed_labels": [str(labels[j]) for j in range(len(labels)) if fixed[j]],
        "labels_target_feasible": [
            str(labels[j]) for j, value in enumerate(feasibility) if value is True
        ],
        "labels_target_infeasible": [
            str(labels[j]) for j, value in enumerate(feasibility) if value is False
        ],
        "before_joint": {
            "normal": normal_before,
            "normal_target_value": normal_value_before,
            "normal_target_met": bool(normal_value_before >= float(normal_target) - 1e-12),
            "label_targets_met": int(sum(row["f1_before_joint"] >= float(label_f1_target) - 1e-12 for row in per_label_rows)),
        },
        "after_f1_initialization": {"normal": normal_initial},
        "final": {
            "normal": normal_final,
            "normal_target_value": normal_value_final,
            "normal_target_met": bool(normal_value_final >= float(normal_target) - 1e-12),
            "label_targets_met": int(np.count_nonzero(final_label_f1 >= float(label_f1_target) - 1e-12)),
            "label_count": int(len(labels)),
            "all_label_targets_met": bool(np.all(final_label_f1 >= float(label_f1_target) - 1e-12)),
            "all_targets_met": bool(
                np.all(final_label_f1 >= float(label_f1_target) - 1e-12)
                and normal_value_final >= float(normal_target) - 1e-12
            ),
            "macro_label_f1": float(np.mean(final_label_f1)) if len(final_label_f1) else 0.0,
            "worst_label_f1": float(np.min(final_label_f1)) if len(final_label_f1) else 0.0,
        },
        "per_label": per_label_rows,
        "moves": history,
        "note": (
            "Heurística target-first; si reporta targets no alcanzados, los scores de calibration "
            "no ofrecieron una solución en el óptimo local sin violar las restricciones F1."
        ),
    }
    return thresholds.astype(np.float32), report


def _ensure_at_least_one_label(scores: np.ndarray, pred: np.ndarray, min_attack_score: float) -> np.ndarray:
    # Opcional: cuando el modelo cree que hay ataque pero todos los umbrales son altos,
    # asigna la etiqueta con mayor score. Por defecto queda desactivado con min_attack_score<=0.
    if min_attack_score <= 0:
        return pred
    out = pred.copy()
    max_scores = scores.max(axis=1)
    empty = out.sum(axis=1) == 0
    needs = empty & (max_scores >= float(min_attack_score))
    if np.any(needs):
        best = np.argmax(scores[needs], axis=1)
        rows = np.flatnonzero(needs)
        out[rows, best] = 1
    return out


def compute_multilabel_metrics(
    Y_true: np.ndarray,
    scores: np.ndarray,
    thresholds: np.ndarray,
    labels: Sequence[str],
    min_attack_score: float = 0.0,
) -> Tuple[Dict[str, Any], pd.DataFrame, np.ndarray]:
    Y_true = np.asarray(Y_true).astype(int)
    scores = np.asarray(scores, dtype=np.float32)
    Y_pred = (scores >= thresholds.reshape(1, -1)).astype(int)
    Y_pred = _ensure_at_least_one_label(scores, Y_pred, min_attack_score=min_attack_score)

    out: Dict[str, Any] = {
        "exact_match_accuracy": _safe_float(accuracy_score(Y_true, Y_pred)),
        "hamming_loss": _safe_float(hamming_loss(Y_true, Y_pred)),
        "label_density_true": _safe_float(float(Y_true.sum()) / float(np.prod(Y_true.shape))) if Y_true.size else None,
        "label_density_pred": _safe_float(float(Y_pred.sum()) / float(np.prod(Y_pred.shape))) if Y_pred.size else None,
        "avg_true_labels_per_row": _safe_float(np.mean(Y_true.sum(axis=1))) if len(Y_true) else None,
        "avg_pred_labels_per_row": _safe_float(np.mean(Y_pred.sum(axis=1))) if len(Y_pred) else None,
    }
    for avg in ["micro", "macro", "weighted", "samples"]:
        try:
            out[f"precision_{avg}"] = _safe_float(precision_score(Y_true, Y_pred, average=avg, zero_division=0))
            out[f"recall_{avg}"] = _safe_float(recall_score(Y_true, Y_pred, average=avg, zero_division=0))
            out[f"f1_{avg}"] = _safe_float(f1_score(Y_true, Y_pred, average=avg, zero_division=0))
            out[f"jaccard_{avg}"] = _safe_float(jaccard_score(Y_true, Y_pred, average=avg, zero_division=0))
        except Exception:
            pass
    try:
        if len(np.unique(Y_true.ravel())) == 2:
            out["roc_auc_micro"] = _safe_float(roc_auc_score(Y_true.ravel(), scores.ravel()))
            out["pr_auc_micro"] = _safe_float(average_precision_score(Y_true.ravel(), scores.ravel()))
    except Exception:
        pass
    for avg in ["macro", "weighted"]:
        try:
            out[f"roc_auc_{avg}"] = _safe_float(roc_auc_score(Y_true, scores, average=avg))
        except Exception:
            out[f"roc_auc_{avg}"] = None
        try:
            out[f"pr_auc_{avg}"] = _safe_float(average_precision_score(Y_true, scores, average=avg))
        except Exception:
            out[f"pr_auc_{avg}"] = None

    mcm = multilabel_confusion_matrix(Y_true, Y_pred, labels=list(range(len(labels))))
    rows: List[Dict[str, Any]] = []
    for j, lab in enumerate(labels):
        tn, fp, fn, tp = [int(x) for x in mcm[j].ravel()]
        support = int(Y_true[:, j].sum())
        pred_support = int(Y_pred[:, j].sum())
        prec = _safe_div(tp, tp + fp)
        rec = _safe_div(tp, tp + fn)
        spec = _safe_div(tn, tn + fp)
        fpr = _safe_div(fp, fp + tn)
        fnr = _safe_div(fn, fn + tp)
        f1 = None if prec is None or rec is None or (prec + rec) == 0 else 2.0 * prec * rec / (prec + rec)
        roc = None
        pr = None
        try:
            if len(np.unique(Y_true[:, j])) == 2:
                roc = _safe_float(roc_auc_score(Y_true[:, j], scores[:, j]))
        except Exception:
            pass
        try:
            if support > 0:
                pr = _safe_float(average_precision_score(Y_true[:, j], scores[:, j]))
        except Exception:
            pass
        rows.append({
            "label": lab,
            "label_index": int(j),
            "threshold": float(thresholds[j]),
            "support": support,
            "pred_support": pred_support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": prec,
            "recall_tpr": rec,
            "specificity_tnr": spec,
            "fpr": fpr,
            "fnr": fnr,
            "f1": f1,
            "roc_auc": roc,
            "pr_auc": pr,
            "prevalence": _safe_div(support, len(Y_true)),
            "pred_prevalence": _safe_div(pred_support, len(Y_true)),
        })
    per_label = pd.DataFrame(rows)
    out["per_label_macro_f1_manual"] = _safe_float(per_label["f1"].dropna().mean()) if not per_label.empty else None
    out["per_label_macro_roc_auc_manual"] = _safe_float(per_label["roc_auc"].dropna().mean()) if not per_label.empty else None
    out["per_label_macro_pr_auc_manual"] = _safe_float(per_label["pr_auc"].dropna().mean()) if not per_label.empty else None

    yt_bin = (Y_true.sum(axis=1) > 0).astype(int)
    yp_bin = (Y_pred.sum(axis=1) > 0).astype(int)
    cm = confusion_matrix(yt_bin, yp_bin, labels=[0, 1])
    tn, fp, fn, tp = [int(x) for x in cm.ravel()]
    bin_scores = scores.max(axis=1) if scores.size else np.zeros(len(Y_true))
    binary_metrics = {
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "accuracy": _safe_float(accuracy_score(yt_bin, yp_bin)),
        "precision": _safe_float(precision_score(yt_bin, yp_bin, zero_division=0)),
        "recall_tpr": _safe_float(recall_score(yt_bin, yp_bin, zero_division=0)),
        "specificity_tnr": _safe_div(tn, tn + fp),
        "fpr": _safe_div(fp, fp + tn),
        "fnr": _safe_div(fn, fn + tp),
        "f1": _safe_float(f1_score(yt_bin, yp_bin, zero_division=0)),
        "mcc": _safe_float(matthews_corrcoef(yt_bin, yp_bin)),
    }
    normal_precision = _safe_div(tn, tn + fn)
    normal_recall = _safe_div(tn, tn + fp)
    normal_f1 = (
        None
        if normal_precision is None or normal_recall is None or (normal_precision + normal_recall) == 0
        else 2.0 * normal_precision * normal_recall / (normal_precision + normal_recall)
    )
    binary_metrics.update({
        "normal_support": int((yt_bin == 0).sum()),
        "normal_pred_support": int((yp_bin == 0).sum()),
        "normal_precision": normal_precision,
        "normal_recall": normal_recall,
        "normal_f1": normal_f1,
        "normal_recall_target_99_met": bool(normal_recall is not None and normal_recall >= 0.99),
        "normal_f1_target_99_met": bool(normal_f1 is not None and normal_f1 >= 0.99),
    })
    try:
        if len(np.unique(yt_bin)) == 2:
            binary_metrics["roc_auc"] = _safe_float(roc_auc_score(yt_bin, bin_scores))
            binary_metrics["pr_auc"] = _safe_float(average_precision_score(yt_bin, bin_scores))
    except Exception:
        pass
    out["binary_normal_vs_attack"] = binary_metrics
    return out, per_label, Y_pred


def evaluate_candidate_on_validation(
    records: Sequence[Dict[str, Any]],
    X_valid: np.ndarray,
    Y_valid: np.ndarray,
    labels: Sequence[str],
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], pd.DataFrame, np.ndarray, np.ndarray]:
    scores = predict_score_matrix(records, X_valid)
    thresholds, th_df = optimize_thresholds(
        Y_valid,
        scores,
        labels,
        objective=str(args.threshold_objective),
        grid_size=int(args.threshold_grid_size),
        min_threshold=float(args.threshold_min),
        max_threshold=float(args.threshold_max),
        min_recall=float(args.threshold_min_recall),
    )
    metrics, per_label, pred = compute_multilabel_metrics(Y_valid, scores, thresholds, labels, min_attack_score=float(args.min_attack_score))
    return metrics, th_df, thresholds, scores


def _threshold_objective_value(
    precision: Optional[float],
    recall: Optional[float],
    specificity: Optional[float],
    objective: str,
) -> Optional[float]:
    if precision is None or recall is None:
        return None
    p = float(precision)
    r = float(recall)
    objective_l = str(objective or "f1").lower()
    if objective_l in {"f2", "fbeta2"}:
        return float(_fbeta_from_pr(p, r, beta=2.0))
    if objective_l == "precision":
        return p
    if objective_l == "recall":
        return r
    if objective_l in {"youden", "balanced_accuracy"}:
        return None if specificity is None else float(0.5 * (r + float(specificity)))
    return float(_fbeta_from_pr(p, r, beta=1.0))


def calibrate_selected_candidate(
    best: Dict[str, Any],
    X_calibration: np.ndarray,
    Y_calibration: np.ndarray,
    labels: Sequence[str],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Fija umbrales del candidato elegido usando sólo calibration.

    Train/tune ya determinaron estimadores e hiperparámetros. Esta función no
    entrena ni selecciona modelos y no toca test. Si calibration no contiene
    suficientes positivos o carece de una clase para una etiqueta rara,
    conserva explícitamente su umbral de tune y lo registra como fallback.
    """
    if len(X_calibration) == 0 or len(Y_calibration) == 0:
        raise ValueError("El split calibration está vacío; no se pueden calibrar umbrales.")
    if Y_calibration.ndim != 2 or Y_calibration.shape[1] != len(labels):
        raise ValueError("Y_calibration no coincide con la cantidad de etiquetas.")

    result = dict(best)
    tune_thresholds = np.asarray(best.get("thresholds", np.full(len(labels), 0.5)), dtype=np.float32)
    if tune_thresholds.shape != (len(labels),):
        tune_thresholds = np.full(len(labels), 0.5, dtype=np.float32)
    result["tuning_thresholds"] = tune_thresholds.copy()
    result["tuning_thresholds_df"] = best.get("thresholds_df")
    result["tuning_metrics"] = dict(best.get("tuning_metrics") or best.get("validation_metrics") or {})

    # La matriz de scores se crea una sola vez y proviene exclusivamente del
    # split calibration; el test permanece bloqueado.
    calibration_scores = predict_score_matrix(best["records"], X_calibration)
    thresholds, threshold_df = optimize_thresholds(
        Y_calibration,
        calibration_scores,
        labels,
        objective=str(args.threshold_objective),
        grid_size=int(args.threshold_grid_size),
        min_threshold=float(args.threshold_min),
        max_threshold=float(args.threshold_max),
        min_recall=float(args.threshold_min_recall),
    )

    min_positives = max(1, int(getattr(args, "min_calibration_positives", 3)))
    fallback_by_label: Dict[str, Dict[str, Any]] = {}
    positives = Y_calibration.sum(axis=0).astype(int)
    negatives = (len(Y_calibration) - positives).astype(int)
    for j, label in enumerate(labels):
        reason = None
        if int(positives[j]) < min_positives:
            reason = f"positives_below_{min_positives}"
        elif int(negatives[j]) < 1:
            reason = "no_negative_examples"
        if reason is None:
            continue
        fallback_threshold = float(tune_thresholds[j]) if np.isfinite(tune_thresholds[j]) else 0.5
        thresholds[j] = fallback_threshold
        fallback_by_label[str(label)] = {
            "label": str(label),
            "reason": reason,
            "calibration_positives": int(positives[j]),
            "calibration_negatives": int(negatives[j]),
            "threshold": fallback_threshold,
            "threshold_source": "tune_fallback_rare_label",
        }

    fixed_joint_labels = np.asarray(
        [str(label) in fallback_by_label for label in labels],
        dtype=bool,
    )
    if bool(getattr(args, "joint_threshold_optimization", True)):
        thresholds, joint_threshold_report = optimize_thresholds_joint_targets(
            Y_calibration,
            calibration_scores,
            labels,
            thresholds,
            label_f1_target=float(getattr(args, "target_label_f1", 0.85)),
            normal_target=float(getattr(args, "target_normal", 0.99)),
            normal_metric=str(getattr(args, "target_normal_metric", "recall")),
            min_threshold=float(args.threshold_min),
            max_threshold=float(args.threshold_max),
            min_recall=float(args.threshold_min_recall),
            max_passes=int(getattr(args, "joint_threshold_max_passes", 6)),
            fixed_labels=fixed_joint_labels,
        )
    else:
        normal_metric_l = str(getattr(args, "target_normal_metric", "recall"))
        current_any = (calibration_scores >= thresholds.reshape(1, -1)).any(axis=1)
        current_normal = _normal_metrics_from_any_attack(
            Y_calibration.sum(axis=1) > 0,
            current_any,
        )
        joint_threshold_report = {
            "enabled": False,
            "data_role": "calibration_only",
            "label_f1_target": float(getattr(args, "target_label_f1", 0.85)),
            "normal_metric": normal_metric_l,
            "normal_target": float(getattr(args, "target_normal", 0.99)),
            "final": {
                "normal": current_normal,
                "normal_target_value": float(current_normal[normal_metric_l]),
                "normal_target_met": bool(
                    current_normal[normal_metric_l] >= float(getattr(args, "target_normal", 0.99))
                ),
            },
            "per_label": [],
            "moves": [],
            "note": "Desactivado por --no-joint-threshold-optimization.",
        }

    calibration_metrics, calibration_per_label, _ = compute_multilabel_metrics(
        Y_calibration,
        calibration_scores,
        thresholds,
        labels,
        min_attack_score=float(args.min_attack_score),
    )
    per_label_lookup = {
        str(row["label"]): row
        for row in calibration_per_label.to_dict(orient="records")
    }
    threshold_rows: List[Dict[str, Any]] = []
    original_rows = {
        str(row.get("label")): row
        for row in threshold_df.to_dict(orient="records")
    }
    joint_rows_lookup = {
        str(row.get("label")): row
        for row in joint_threshold_report.get("per_label", [])
    }
    for j, label in enumerate(labels):
        label_s = str(label)
        row = dict(original_rows.get(label_s) or {})
        per_label_row = per_label_lookup.get(label_s) or {}
        fallback = fallback_by_label.get(label_s)
        joint_row = joint_rows_lookup.get(label_s) or {}
        joint_changed = abs(float(thresholds[j]) - float(row.get("threshold", thresholds[j]))) > 1e-12
        row.update({
            "label": label_s,
            "threshold": float(thresholds[j]),
            "objective": str(args.threshold_objective),
            "calibration_support": int(positives[j]),
            "calibration_negative_count": int(negatives[j]),
            "threshold_source": (
                fallback["threshold_source"]
                if fallback is not None
                else ("calibration_joint_targets" if joint_changed else "calibration_independent")
            ),
            "fallback_reason": None if fallback is None else fallback["reason"],
            "joint_optimization_eligible": joint_row.get("optimization_eligible"),
            "label_f1_target": float(getattr(args, "target_label_f1", 0.85)),
            "label_f1_target_feasible_on_calibration": joint_row.get("target_feasible_on_calibration"),
            "label_f1_target_met": bool(
                (per_label_row.get("f1") or 0.0) >= float(getattr(args, "target_label_f1", 0.85))
            ),
            "precision": per_label_row.get("precision"),
            "recall": per_label_row.get("recall_tpr"),
            "f1": per_label_row.get("f1"),
            "score": _threshold_objective_value(
                per_label_row.get("precision"),
                per_label_row.get("recall_tpr"),
                per_label_row.get("specificity_tnr"),
                str(args.threshold_objective),
            ),
        })
        threshold_rows.append(row)
    threshold_df = pd.DataFrame(threshold_rows)

    joint_report_rows = list(joint_threshold_report.get("per_label", []))
    normal_joint = (joint_threshold_report.get("final") or {}).get("normal") or {}
    joint_report_rows.append({
        "label": "NORMAL",
        "optimization_eligible": True,
        "target_metric": f"normal_{getattr(args, 'target_normal_metric', 'recall')}",
        "target": float(getattr(args, "target_normal", 0.99)),
        "target_met": bool((joint_threshold_report.get("final") or {}).get("normal_target_met")),
        "precision_joint": normal_joint.get("precision"),
        "recall_joint": normal_joint.get("recall"),
        "f1_joint": normal_joint.get("f1"),
        "support": normal_joint.get("normal_support"),
    })
    joint_threshold_report_df = pd.DataFrame(joint_report_rows)

    fallback_rows = list(fallback_by_label.values())
    if fallback_rows:
        print(
            f"[WARN] {len(fallback_rows)} etiquetas sin calibration suficiente conservaron "
            "el umbral de tune; revisar calibration_fallback_labels.json.",
            flush=True,
        )
    print(
        f"[CAL] rows={len(Y_calibration)} f1_macro={calibration_metrics.get('f1_macro')} "
        f"f1_micro={calibration_metrics.get('f1_micro')} fallbacks={len(fallback_rows)}",
        flush=True,
    )
    result["thresholds"] = np.asarray(thresholds, dtype=np.float32)
    result["thresholds_df"] = threshold_df
    result["calibration_metrics"] = calibration_metrics
    result["calibration_per_label"] = calibration_per_label
    result["calibration_fallback_labels"] = fallback_rows
    result["joint_threshold_report"] = joint_threshold_report
    result["joint_threshold_report_df"] = joint_threshold_report_df
    result["calibration_rows"] = int(len(Y_calibration))
    result["calibration_score_shape"] = [int(x) for x in calibration_scores.shape]
    return result


def _metric_value_for_selection(metrics: Dict[str, Any], name: str) -> float:
    val = metrics.get(name)
    if val is None:
        # aliases comunes
        aliases = {
            "f1_macro": "f1_macro",
            "f1_micro": "f1_micro",
            "jaccard_macro": "jaccard_macro",
            "pr_auc_macro": "pr_auc_macro",
            "roc_auc_macro": "roc_auc_macro",
            "exact_match_accuracy": "exact_match_accuracy",
        }
        val = metrics.get(aliases.get(name, name))
    try:
        x = float(val)
        if math.isnan(x) or math.isinf(x):
            return -1.0
        return x
    except Exception:
        return -1.0


def determine_candidate_backends(args: argparse.Namespace) -> List[str]:
    backend_spec = (args.model_backend or "auto").strip().lower()
    xgb_ok, xgb_ver = _xgboost_available()
    lgbm_ok, lgbm_ver = _lightgbm_available()
    cat_ok, cat_ver = _catboost_available()

    fast_list = ["xgboost", "histgb", "extra_trees", "logistic_regression"]
    standard_list = ["xgboost", "histgb", "extra_trees", "random_forest", "logistic_regression", "sgd_logloss"]
    max_list = ["xgboost", "lightgbm", "catboost", "histgb", "extra_trees", "random_forest", "logistic_regression", "linear_svc", "sgd_logloss"]

    level = str(args.search_level or "standard").strip().lower()
    if backend_spec in {"auto", "best"}:
        requested = standard_list if level != "max" else max_list
    elif backend_spec in {"compare", "all", "full", "multifamily", "multi", "families"}:
        requested = max_list if level == "max" else standard_list
    else:
        requested = [_canonical_backend_name(x) for x in re.split(r"[,;\s]+", backend_spec) if x.strip()]

    if level in {"none", "off", "0"} and requested:
        requested = [requested[0]]

    available: List[str] = []
    skipped: List[Dict[str, Any]] = []
    for b0 in requested:
        b = _canonical_backend_name(b0)
        ok, ver = _backend_available(b)
        if ok:
            if b not in available:
                available.append(b)
        else:
            skipped.append({"backend": b, "reason": "dependency_not_available", "version": ver})

    if skipped:
        print(f"[WARN] Backends omitidos por dependencias no disponibles: {skipped}", flush=True)
    if not available:
        raise SystemExit(f"No hay ningún backend disponible para --model-backend={args.model_backend!r}")

    print(
        "[INFO] Backend availability: "
        f"xgboost={xgb_ok}({xgb_ver}) lightgbm={lgbm_ok}({lgbm_ver}) catboost={cat_ok}({cat_ver}); "
        f"selected={available}",
        flush=True,
    )
    return available


def candidate_weight_modes(args: argparse.Namespace) -> List[str]:
    modes = [_safe_name(m.strip().lower().replace("-", "_")) for m in str(args.candidate_weight_modes or "sqrt").split(",") if m.strip()]
    # _safe_name convierte cuberoot a cuberoot, pero también cambiaría vacío; dejamos aliases aceptados.
    modes = [m for m in modes if m]
    if not modes:
        modes = ["sqrt"]
    # Preserva orden y evita duplicados.
    out: List[str] = []
    for m in modes:
        if m not in out:
            out.append(m)
    if str(args.search_level).lower() in {"none", "off", "0"}:
        return [out[0]]
    return out


def _candidate_sort_key(candidate: Dict[str, Any]) -> Tuple[float, float, float, float, float, float, int, int, float]:
    metrics = candidate.get("validation_metrics") or {}
    family_priority = {
        "xgboost": 90,
        "lightgbm": 85,
        "catboost": 80,
        "histgb": 70,
        "extra_trees": 65,
        "random_forest": 60,
        "logistic_regression": 40,
        "linear_svc": 35,
        "sgd_logloss": 30,
    }.get(str(candidate.get("backend")), 0)
    return (
        float(candidate.get("selection_score", -1.0)),
        _metric_value_for_selection(metrics, "f1_macro"),
        _metric_value_for_selection(metrics, "f1_micro"),
        _metric_value_for_selection(metrics, "pr_auc_macro"),
        _metric_value_for_selection(metrics, "roc_auc_macro"),
        _metric_value_for_selection(metrics, "exact_match_accuracy"),
        1 if candidate.get("gpu_used_any") else 0,
        int(family_priority),
        -float(candidate.get("train_select_seconds", 0.0)),
    )


def train_select_candidate(
    args: argparse.Namespace,
    labels: Sequence[str],
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_valid: np.ndarray,
    Y_valid: np.ndarray,
    out_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    backends = determine_candidate_backends(args)
    modes = candidate_weight_modes(args)
    prefer_gpu = _gpu_visible_auto() or str(args.use_gpu).lower() == "on"
    candidates: List[Dict[str, Any]] = []
    summary_all: List[Dict[str, Any]] = []
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        if _resume_enabled(args):
            print(f"[INFO] Resume/checkpoints activos en: {_checkpoint_root(args, out_dir)}", flush=True)
            _checkpoint_root(args, out_dir).mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Candidate backends={backends} weight_modes={modes}", flush=True)
    for backend in backends:
        for weight_mode in modes:
            # Reanudar por candidato completo: si existe checkpoint OK, no se vuelve a entrenar ese candidato.
            if out_dir is not None:
                restored = _load_candidate_checkpoint(args, out_dir, backend, weight_mode, labels)
                if restored is not None:
                    _write_xgb_tuning_records(restored.get("records") or [], out_dir)
                    candidates.append(restored)
                    summary_all.append(_candidate_to_summary(restored))
                    _write_candidate_progress(out_dir, summary_all)
                    print(
                        f"[RESUME] Candidato ya entrenado: backend={backend} weight_mode={weight_mode} "
                        f"{restored.get('selection_metric')}={restored.get('selection_score')} checkpoint={restored.get('checkpoint_path')}",
                        flush=True,
                    )
                    continue

            print(f"[INFO] Entrenando candidato backend={backend} weight_mode={weight_mode}", flush=True)
            t0 = time.perf_counter()
            try:
                records = train_ovr_records(
                    args=args,
                    backend=backend,
                    weight_mode=weight_mode,
                    labels=labels,
                    X_train=X_train,
                    Y_train=Y_train,
                    X_valid=X_valid,
                    Y_valid=Y_valid,
                    prefer_gpu=prefer_gpu,
                    desc=f"Training {backend}/{weight_mode}",
                    out_dir=out_dir,
                    checkpoint_phase="candidate_selection",
                )
                _write_xgb_tuning_records(records, out_dir)
                val_metrics, th_df, thresholds, val_scores = evaluate_candidate_on_validation(records, X_valid, Y_valid, labels, args)
                score = _metric_value_for_selection(val_metrics, str(args.selection_metric))
                gpu_used_any = any(bool(r.get("gpu_used")) for r in records)
                elapsed = float(time.perf_counter() - t0)
                candidate = {
                    "backend": backend,
                    "weight_mode": weight_mode,
                    "records": records,
                    "thresholds": thresholds,
                    "thresholds_df": th_df,
                    "validation_metrics": val_metrics,
                    "selection_metric": str(args.selection_metric),
                    "selection_score": score,
                    "gpu_used_any": gpu_used_any,
                    "train_select_seconds": elapsed,
                    "fit_status": "ok",
                }
                if out_dir is not None:
                    ckpt = _save_candidate_checkpoint(args, out_dir, candidate, labels)
                    if ckpt is not None:
                        candidate["checkpoint_path"] = str(ckpt)
                candidates.append(candidate)
                summary_all.append(_candidate_to_summary(candidate))
                if out_dir is not None:
                    _write_candidate_progress(out_dir, summary_all)
                print(
                    f"[VAL] backend={backend} weight_mode={weight_mode} "
                    f"{args.selection_metric}={score} f1_macro={val_metrics.get('f1_macro')} "
                    f"f1_micro={val_metrics.get('f1_micro')} exact_match={val_metrics.get('exact_match_accuracy')} "
                    f"gpu_used={gpu_used_any}",
                    flush=True,
                )
            except KeyboardInterrupt:
                raise
            except Exception as e:
                elapsed = float(time.perf_counter() - t0)
                err = f"{type(e).__name__}: {e}"
                failed = {
                    "backend": backend,
                    "weight_mode": weight_mode,
                    "fit_status": "failed",
                    "selection_metric": str(args.selection_metric),
                    "selection_score": -1.0,
                    "gpu_used_any": False,
                    "validation_metrics": {},
                    "train_select_seconds": elapsed,
                    "fit_error": err,
                }
                summary_all.append(failed)
                if out_dir is not None:
                    _write_candidate_progress(out_dir, summary_all)
                print(f"[WARN] Candidato falló y se omite: backend={backend} weight_mode={weight_mode} error={err}", flush=True)
                continue

    if not candidates:
        raise SystemExit("Todos los candidatos fallaron; revisá candidate_validation_summary.partial.json y logs.")

    candidates_sorted = sorted(candidates, key=_candidate_sort_key, reverse=True)
    best = candidates_sorted[0]
    ok_summaries = [s for s in summary_all if s.get("fit_status") == "ok"]
    failed_summaries = [s for s in summary_all if s.get("fit_status") != "ok"]
    ok_summaries = sorted(ok_summaries, key=_candidate_sort_key, reverse=True)
    best["candidate_summary"] = ok_summaries + failed_summaries
    best["candidate_failures"] = failed_summaries
    print(
        f"[BEST] backend={best['backend']} weight_mode={best['weight_mode']} "
        f"{best['selection_metric']}={best['selection_score']} gpu_used={best.get('gpu_used_any')}",
        flush=True,
    )
    return best


def refit_final_if_requested(
    args: argparse.Namespace,
    best: Dict[str, Any],
    labels: Sequence[str],
    X_train_valid: np.ndarray,
    Y_train_valid: np.ndarray,
    X_valid: np.ndarray,
    Y_valid: np.ndarray,
    out_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    if not args.refit_full_after_thresholds:
        return best
    if out_dir is not None:
        restored = _load_final_refit_checkpoint(args, out_dir, best, labels)
        if restored is not None:
            return restored
    print(f"[INFO] Refit final con train+valid backend={best['backend']} weight_mode={best['weight_mode']}", flush=True)
    prefer_gpu = _gpu_visible_auto() or str(args.use_gpu).lower() == "on"
    t0 = time.perf_counter()
    records = train_ovr_records(
        args=args,
        backend=best["backend"],
        weight_mode=best["weight_mode"],
        labels=labels,
        X_train=X_train_valid,
        Y_train=Y_train_valid,
        X_valid=X_valid,
        Y_valid=Y_valid,
        prefer_gpu=prefer_gpu,
        desc=f"Final refit {best['backend']}/{best['weight_mode']}",
        out_dir=out_dir,
        checkpoint_phase="final_refit",
    )
    best = dict(best)
    best["records"] = records
    best["final_refit_seconds"] = float(time.perf_counter() - t0)
    best["gpu_used_any"] = any(bool(r.get("gpu_used")) for r in records)
    if out_dir is not None:
        ckpt = _save_final_refit_checkpoint(args, out_dir, best, labels)
        if ckpt is not None:
            best["final_refit_checkpoint_path"] = str(ckpt)
    return best


def compute_feature_importance_multilabel(
    records: Sequence[Dict[str, Any]],
    X: np.ndarray,
    Y: np.ndarray,
    thresholds: np.ndarray,
    labels: Sequence[str],
    args: argparse.Namespace,
    out_dir: Path,
) -> Dict[str, str]:
    paths: Dict[str, str] = {}

    # Importancia nativa de XGBoost por gain, ponderada por soporte de cada etiqueta.
    xgb_rows: List[Dict[str, Any]] = []
    for j, rec in enumerate(records):
        est = rec.get("estimator")
        if rec.get("backend") != "xgboost" or est is None:
            continue
        try:
            booster = est.get_booster()
            gain = booster.get_score(importance_type="gain")
            weight = booster.get_score(importance_type="weight")
            cover = booster.get_score(importance_type="cover")
            support = int(Y[:, j].sum()) if Y is not None and Y.size else 1
            for i, feat in enumerate(FIXED_FEATURES):
                keys = [feat, f"f{i}"]
                xgb_rows.append({
                    "label": labels[j],
                    "feature": feat,
                    "gain": float(sum(float(gain.get(k, 0.0)) for k in keys)),
                    "weight": float(sum(float(weight.get(k, 0.0)) for k in keys)),
                    "cover": float(sum(float(cover.get(k, 0.0)) for k in keys)),
                    "label_support": support,
                })
        except Exception:
            continue
    if xgb_rows:
        xgb_df = pd.DataFrame(xgb_rows)
        xgb_df.to_csv(out_dir / "feature_importance_xgb_by_label.csv", index=False)
        agg_rows: List[Dict[str, Any]] = []
        for feat, g in xgb_df.groupby("feature", sort=False):
            agg_rows.append({
                "feature": feat,
                "gain_weighted_by_support": float(np.average(g["gain"], weights=np.maximum(g["label_support"], 1))),
                "gain_mean": float(g["gain"].mean()),
                "weight_sum": float(g["weight"].sum()),
                "cover_mean": float(g["cover"].mean()),
            })
        agg = pd.DataFrame(agg_rows).sort_values("gain_weighted_by_support", ascending=False)
        agg.to_csv(out_dir / "feature_importance_xgb_gain_aggregated.csv", index=False)
        paths["feature_importance_xgb_by_label"] = str(out_dir / "feature_importance_xgb_by_label.csv")
        paths["feature_importance_xgb_gain_aggregated"] = str(out_dir / "feature_importance_xgb_gain_aggregated.csv")

    if int(args.fi_n_repeats or 0) <= 0:
        return paths
    if X.shape[0] == 0:
        return paths
    n = X.shape[0]
    sampled = False
    if int(args.fi_max_rows or 0) > 0 and n > int(args.fi_max_rows):
        rng = np.random.default_rng(int(args.seed))
        idx = rng.choice(np.arange(n), size=int(args.fi_max_rows), replace=False)
        X_eval = X[idx].copy()
        Y_eval = Y[idx].copy()
        sampled = True
    else:
        X_eval = X.copy()
        Y_eval = Y.copy()
    base_scores = predict_score_matrix(records, X_eval)
    base_metrics, _, _ = compute_multilabel_metrics(Y_eval, base_scores, thresholds, labels, min_attack_score=float(args.min_attack_score))
    base_score = _metric_value_for_selection(base_metrics, str(args.fi_scoring))
    rng = np.random.default_rng(int(args.seed) + 123)
    rows = []
    for feat_idx, feat in enumerate(tqdm(FIXED_FEATURES, desc="Permutation feature importance", mininterval=5)):
        vals = []
        for _ in range(int(args.fi_n_repeats)):
            Xp = X_eval.copy()
            rng.shuffle(Xp[:, feat_idx])
            sc = predict_score_matrix(records, Xp)
            met, _, _ = compute_multilabel_metrics(Y_eval, sc, thresholds, labels, min_attack_score=float(args.min_attack_score))
            vals.append(base_score - _metric_value_for_selection(met, str(args.fi_scoring)))
        rows.append({
            "feature": feat,
            "importance_mean": float(np.mean(vals)),
            "importance_std": float(np.std(vals)),
            "base_score": float(base_score),
            "scoring": str(args.fi_scoring),
            "n_repeats": int(args.fi_n_repeats),
            "rows_used": int(len(X_eval)),
            "sampled": bool(sampled),
        })
    fi_df = pd.DataFrame(rows).sort_values("importance_mean", ascending=False)
    fi_df.to_csv(out_dir / "feature_importance.csv", index=False)
    paths["feature_importance_permutation"] = str(out_dir / "feature_importance.csv")
    return paths


def _percentile(vals: Sequence[float], p: float) -> Optional[float]:
    if not vals:
        return None
    return float(np.percentile(np.asarray(vals, dtype=float), p))


def measure_operational_metrics_multilabel(
    records: Sequence[Dict[str, Any]],
    df_test_raw: pd.DataFrame,
    X_test: np.ndarray,
    thresholds: np.ndarray,
    labels: Sequence[str],
    args: argparse.Namespace,
    out_dir: Path,
) -> Dict[str, Any]:
    n_total = len(df_test_raw)
    if n_total == 0:
        return {"available": False, "reason": "empty test"}
    if int(args.operational_max_rows or 0) > 0 and n_total > int(args.operational_max_rows):
        df_meas = df_test_raw.sample(n=int(args.operational_max_rows), random_state=int(args.seed)).reset_index(drop=True)
        # X_test debe seguir el mismo índice relativo del test; para batch inference no importa si usamos muestra raw extraída.
        sampled = True
    else:
        df_meas = df_test_raw.reset_index(drop=True)
        sampled = False
    n = len(df_meas)

    # Batch inference sobre features ya extraídas del test completo/muestra si se puede.
    if sampled:
        tuples_for_batch = list(_feature_input_tuples(df_meas))
        X_batch = np.asarray([_extract_features_from_tuple(t) for t in tuples_for_batch], dtype=np.float32)
    else:
        X_batch = X_test.astype(np.float32, copy=False)

    cpu0 = time.process_time()
    t0 = time.perf_counter()
    _ = predict_score_matrix(records, X_batch)
    batch_predict_wall = time.perf_counter() - t0
    batch_predict_cpu = time.process_time() - cpu0

    tuples = list(_feature_input_tuples(df_meas))
    cpu0 = time.process_time()
    t0 = time.perf_counter()
    feat_vals = [_extract_features_from_tuple(t) for t in tuples]
    X_full = np.asarray(feat_vals, dtype=np.float32)
    scores = predict_score_matrix(records, X_full)
    _ = (scores >= thresholds.reshape(1, -1)).astype(np.int8)
    batch_pipeline_wall = time.perf_counter() - t0
    batch_pipeline_cpu = time.process_time() - cpu0

    lat_ms: List[float] = []
    cpu0 = time.process_time()
    t0 = time.perf_counter()
    for tup in tqdm(tuples, total=len(tuples), desc="Operational latency raw request -> multilabel prediction", mininterval=5):
        a = time.perf_counter()
        x = np.asarray(_extract_features_from_tuple(tup), dtype=np.float32).reshape(1, -1)
        sc = predict_score_matrix(records, x)
        _ = (sc >= thresholds.reshape(1, -1)).astype(np.int8)
        b = time.perf_counter()
        lat_ms.append((b - a) * 1000.0)
    online_pipeline_wall = time.perf_counter() - t0
    online_pipeline_cpu = time.process_time() - cpu0

    cpu_count = os.cpu_count() or 1
    metrics = {
        "available": True,
        "measured_rows": int(n),
        "test_rows_total": int(n_total),
        "sampled": bool(sampled),
        "operational_max_rows": int(args.operational_max_rows),
        "latency_ms_pipeline_online_p50": _percentile(lat_ms, 50),
        "latency_ms_pipeline_online_p95": _percentile(lat_ms, 95),
        "latency_ms_pipeline_online_p99": _percentile(lat_ms, 99),
        "latency_ms_pipeline_online_mean": float(np.mean(lat_ms)) if lat_ms else None,
        "throughput_req_s_pipeline_online": _safe_div(n, online_pipeline_wall),
        "throughput_req_s_pipeline_batch": _safe_div(n, batch_pipeline_wall),
        "throughput_req_s_inference_batch": _safe_div(n, batch_predict_wall),
        "t_apply_pipeline_online_seconds": float(online_pipeline_wall),
        "t_apply_pipeline_batch_seconds": float(batch_pipeline_wall),
        "t_apply_inference_batch_seconds": float(batch_predict_wall),
        "cpu_time_pipeline_online_seconds": float(online_pipeline_cpu),
        "cpu_time_pipeline_batch_seconds": float(batch_pipeline_cpu),
        "cpu_time_inference_batch_seconds": float(batch_predict_cpu),
        "cpu_utilization_pipeline_online_percent_of_all_cores": _safe_div(online_pipeline_cpu, online_pipeline_wall * cpu_count) * 100.0 if online_pipeline_wall > 0 else None,
        "peak_rss_mb": _get_peak_rss_mb(),
        "current_rss_mb": _get_current_rss_mb(),
    }
    pd.DataFrame({"latency_ms_pipeline_online": lat_ms}).to_csv(out_dir / "operational_latency_online_ms.csv", index=False)
    pd.DataFrame([metrics]).to_csv(out_dir / "operational_metrics.csv", index=False)
    _save_json(metrics, out_dir / "operational_metrics.json")
    return metrics


def flatten_metrics_for_comparison(m: Dict[str, Any]) -> Dict[str, Any]:
    tm = m.get("test_metrics") or {}
    bm = tm.get("binary_normal_vs_attack") or {}
    om = m.get("operational_metrics") or {}
    timing = m.get("timing") or {}
    return {
        "dataset": m.get("dataset"),
        "task": m.get("task"),
        "selected_model": m.get("selected_model"),
        "selected_model_family": m.get("selected_model_family"),
        "backend": m.get("backend"),
        "weight_mode": m.get("weight_mode"),
        "gpu_used": m.get("gpu_used"),
        "feature_count": m.get("feature_count"),
        "n_labels": m.get("n_labels"),
        "train_rows": m.get("train_rows"),
        "valid_rows": m.get("valid_rows"),
        "test_rows": m.get("test_rows"),
        "exact_match_accuracy": tm.get("exact_match_accuracy"),
        "hamming_loss": tm.get("hamming_loss"),
        "precision_micro": tm.get("precision_micro"),
        "precision_macro": tm.get("precision_macro"),
        "precision_weighted": tm.get("precision_weighted"),
        "recall_micro": tm.get("recall_micro"),
        "recall_macro": tm.get("recall_macro"),
        "recall_weighted": tm.get("recall_weighted"),
        "f1_micro": tm.get("f1_micro"),
        "f1_macro": tm.get("f1_macro"),
        "f1_weighted": tm.get("f1_weighted"),
        "f1_samples": tm.get("f1_samples"),
        "jaccard_micro": tm.get("jaccard_micro"),
        "jaccard_macro": tm.get("jaccard_macro"),
        "roc_auc_micro": tm.get("roc_auc_micro"),
        "roc_auc_macro": tm.get("roc_auc_macro"),
        "pr_auc_micro": tm.get("pr_auc_micro"),
        "pr_auc_macro": tm.get("pr_auc_macro"),
        "binary_accuracy": bm.get("accuracy"),
        "binary_precision": bm.get("precision"),
        "binary_recall_tpr": bm.get("recall_tpr"),
        "binary_specificity_tnr": bm.get("specificity_tnr"),
        "binary_fpr": bm.get("fpr"),
        "binary_fnr": bm.get("fnr"),
        "binary_f1": bm.get("f1"),
        "binary_mcc": bm.get("mcc"),
        "normal_precision": bm.get("normal_precision"),
        "normal_recall": bm.get("normal_recall"),
        "normal_f1": bm.get("normal_f1"),
        "t_train_select_seconds": timing.get("train_select_seconds"),
        "t_final_refit_seconds": timing.get("final_refit_seconds"),
        "feature_extraction_seconds": timing.get("feature_extraction_seconds"),
        "latency_ms_p50_pipeline_online": om.get("latency_ms_pipeline_online_p50"),
        "latency_ms_p95_pipeline_online": om.get("latency_ms_pipeline_online_p95"),
        "throughput_req_s_pipeline_online": om.get("throughput_req_s_pipeline_online"),
        "throughput_req_s_inference_batch": om.get("throughput_req_s_inference_batch"),
        "model_size_mb": (m.get("resource_usage") or {}).get("model_size_mb"),
        "metrics_path": (m.get("artifacts") or {}).get("metrics"),
    }


def make_summary_text(m: Dict[str, Any]) -> str:
    tm = m.get("test_metrics") or {}
    bm = tm.get("binary_normal_vs_attack") or {}
    om = m.get("operational_metrics") or {}
    lines = [
        "BEST_MODEL_DEFINITIVO_MULTIETIQUETA_HARVARD",
        f"dataset={m.get('dataset')}",
        f"task={m.get('task')}",
        f"selected_model={m.get('selected_model')}",
        f"backend={m.get('backend')}",
        f"weight_mode={m.get('weight_mode')}",
        f"gpu_used={m.get('gpu_used')}",
        f"feature_count={m.get('feature_count')}",
        f"n_labels={m.get('n_labels')}",
        f"train_rows={m.get('train_rows')}",
        f"valid_rows={m.get('valid_rows')}",
        f"test_rows={m.get('test_rows')}",
        f"exact_match_accuracy={tm.get('exact_match_accuracy')}",
        f"hamming_loss={tm.get('hamming_loss')}",
        f"precision_micro={tm.get('precision_micro')}",
        f"precision_macro={tm.get('precision_macro')}",
        f"recall_micro={tm.get('recall_micro')}",
        f"recall_macro={tm.get('recall_macro')}",
        f"f1_micro={tm.get('f1_micro')}",
        f"f1_macro={tm.get('f1_macro')}",
        f"f1_weighted={tm.get('f1_weighted')}",
        f"jaccard_macro={tm.get('jaccard_macro')}",
        f"roc_auc_macro={tm.get('roc_auc_macro')}",
        f"pr_auc_macro={tm.get('pr_auc_macro')}",
        f"normal_precision={bm.get('normal_precision')}",
        f"normal_recall={bm.get('normal_recall')}",
        f"normal_f1={bm.get('normal_f1')}",
        f"binary_recall_tpr={bm.get('recall_tpr')}",
        f"binary_specificity_tnr={bm.get('specificity_tnr')}",
        f"binary_fpr={bm.get('fpr')}",
        f"binary_fnr={bm.get('fnr')}",
        f"latency_ms_p50_pipeline_online={om.get('latency_ms_pipeline_online_p50')}",
        f"latency_ms_p95_pipeline_online={om.get('latency_ms_pipeline_online_p95')}",
        f"throughput_req_s_pipeline_online={om.get('throughput_req_s_pipeline_online')}",
    ]
    return "\n".join(lines) + "\n"


def write_aggregate_outputs(args: argparse.Namespace, payload: Dict[str, Any]) -> None:
    root = Path(args.definitivo_dir) / "multietiqueta"
    root.mkdir(parents=True, exist_ok=True)
    row = flatten_metrics_for_comparison(payload)
    pd.DataFrame([row]).to_csv(root / "model_comparison_multilabel_definitivo.csv", index=False)
    _save_json({
        "created_at": _now_iso(),
        "task": "multilabel",
        "dataset": "harvard",
        "model_family": payload.get("selected_model_family"),
        "feature_count": len(FIXED_FEATURES),
        "features": FIXED_FEATURES,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "rows": [row],
        "artifacts": {
            "comparison_csv": str(root / "model_comparison_multilabel_definitivo.csv"),
            "comparison_json": str(root / "model_comparison_multilabel_definitivo.json"),
        },
    }, root / "model_comparison_multilabel_definitivo.json")
    _save_json({
        "created_at": _now_iso(),
        "root": str(Path(args.definitivo_dir)),
        "generated_task": "multietiqueta",
        "structure": {"multietiqueta": {"harvard": payload.get("artifacts") or {}}},
    }, Path(args.definitivo_dir) / "manifest_definitivo_multilabel.json")


def dependency_check() -> Dict[str, Any]:
    import importlib.util
    required = ["numpy", "pandas", "sklearn", "joblib", "tqdm"]
    missing = [p for p in required if importlib.util.find_spec(p) is None]
    if missing:
        raise SystemExit("Faltan dependencias base: " + ", ".join(missing))
    xgb_ok, xgb_ver = _xgboost_available()
    lgbm_ok, lgbm_ver = _lightgbm_available()
    cat_ok, cat_ver = _catboost_available()
    parquet = importlib.util.find_spec("pyarrow") is not None or importlib.util.find_spec("fastparquet") is not None
    if not xgb_ok:
        print("[WARN] xgboost no está instalado: no se probará la familia GPU XGBoost salvo que lo instales o cambies REQUIRE_XGBOOST.", flush=True)
    if not lgbm_ok:
        print("[INFO] lightgbm no está instalado: se omite esa familia opcional.", flush=True)
    if not cat_ok:
        print("[INFO] catboost no está instalado: se omite esa familia opcional.", flush=True)
    if not parquet:
        print("[WARN] pyarrow/fastparquet no detectado: processed se guardará como CSV fallback.", flush=True)
    return {
        "python": sys.executable,
        "xgboost_available": bool(xgb_ok),
        "xgboost_version": xgb_ver,
        "lightgbm_available": bool(lgbm_ok),
        "lightgbm_version": lgbm_ver,
        "catboost_available": bool(cat_ok),
        "catboost_version": cat_ver,
        "parquet_available": bool(parquet),
        "gpu_visible": _gpu_visible_auto(),
    }


def _try_return_completed_run(out_dir: Path, args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    """Si la corrida ya terminó completa, evita recomputar al relanzar sbatch."""
    if (
        not _resume_enabled(args)
        or bool(getattr(args, "force_reprocess", False))
        or not bool(getattr(args, "resume_skip_complete", True))
    ):
        return None
    metrics_path = out_dir / "metrics_multilabel_xgb_request90.json"
    model_path = out_dir / "model_multilabel_xgb_request90.joblib"
    summary_path = out_dir / "selected_model_summary.txt"
    if not (metrics_path.exists() and model_path.exists() and summary_path.exists()):
        return None
    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        if payload.get("feature_schema_sha256") != FEATURE_SCHEMA_SHA256:
            raise ValueError("artefacto completo pertenece a otro feature schema")
        if payload.get("training_signature") != _training_signature(args):
            raise ValueError("artefacto completo pertenece a otra configuración")
        if payload.get("calibration_signature") != _calibration_signature(args):
            raise ValueError("artefacto completo pertenece a otra calibración/targets")
        current_inputs = _collect_inputs(args.inputs or args.harvard_inputs or [], dataset="harvard")
        if payload.get("input_signature") != _input_signature(current_inputs):
            raise ValueError("cambiaron los inputs desde la corrida completa")
        print(f"[RESUME] Corrida completa detectada. No recomputo: {metrics_path}", flush=True)
        print(f"[RESUME] Modelo existente: {model_path}", flush=True)
        return payload
    except Exception as e:
        print(f"[WARN] No pude leer métricas completas existentes ({type(e).__name__}: {e}); continúo normalmente.", flush=True)
        return None


def train_eval_harvard_multilabel(args: argparse.Namespace) -> Dict[str, Any]:
    out_dir = _dataset_out_dir(args)
    if args.clean_output_dir and _resume_enabled(args):
        print("[WARN] --clean-output-dir borra checkpoints; desactivo limpieza porque --resume está activo.", flush=True)
        args.clean_output_dir = False
    if args.clean_output_dir:
        _clean_output_dir(out_dir)
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)

    completed = _try_return_completed_run(out_dir, args)
    if completed is not None:
        return completed

    print("================================================================", flush=True)
    print(f"[START] Harvard multietiqueta XGBoost request90 @ {_now_iso()}", flush=True)
    print(f"[OUT] {out_dir}", flush=True)
    print("================================================================", flush=True)

    gpu_visible = _gpu_visible_auto()
    if str(args.use_gpu).lower() == "on" and not gpu_visible:
        print("[WARN] USE_GPU=on pero no detecto GPU visible. Los backends con GPU intentarán CUDA/GPU y harán fallback CPU si falla.", flush=True)
    elif gpu_visible:
        print("[INFO] GPU visible: se intentarán backends GPU disponibles dentro de la búsqueda multifamilia.", flush=True)
    else:
        print("[INFO] GPU no visible: se usará CPU/fallback.", flush=True)

    t_run0 = time.perf_counter()
    resumed_processed = _try_resume_processed(out_dir, args)
    if resumed_processed is not None:
        (
            df_features,
            processed_path,
            target_labels,
            target_meta,
            inputs,
            _previous_split_info,
            load_seconds,
            feature_extraction_seconds,
        ) = resumed_processed
    else:
        df_raw, inputs, schema, label_cols = load_harvard_multilabel_raw(args)
        load_seconds = time.perf_counter() - t_run0
        print(f"[INFO] Rows loaded: {len(df_raw)}", flush=True)
        print(f"[INFO] Label columns discovered: {len(label_cols)}", flush=True)

        df_targets, Y, target_labels, target_meta, filtering_report = make_multilabel_targets(df_raw, schema, label_cols, args, out_dir)
        print(f"[INFO] Rows after filters: {len(df_targets)} | targets kept: {len(target_labels)}", flush=True)

        df_features, feature_extraction_seconds = build_feature_frame(df_targets, workers=int(args.feature_workers), chunksize=int(args.feature_chunksize))
        # Incluye target columns en processed para reproducibilidad y para poder reanudar si Slurm corta por timeout.
        processed_path = _save_processed(df_features, out_dir / "processed_request90_multilabel.parquet")

    X = df_features[FIXED_FEATURES].fillna(0).astype(np.float32).to_numpy()
    feature_stats = pd.DataFrame({
        "feature": FIXED_FEATURES,
        "nonzero_rows": [int(np.count_nonzero(X[:, i])) for i in range(X.shape[1])],
        "nonzero_ratio": [float(np.count_nonzero(X[:, i]) / max(1, len(X))) for i in range(X.shape[1])],
        "mean": [float(np.mean(X[:, i])) for i in range(X.shape[1])],
        "std": [float(np.std(X[:, i])) for i in range(X.shape[1])],
        "min": [float(np.min(X[:, i])) for i in range(X.shape[1])],
        "max": [float(np.max(X[:, i])) for i in range(X.shape[1])],
    })
    feature_stats.to_csv(out_dir / "feature_coverage_request90.csv", index=False)
    target_cols = [_target_col_name(lab) for lab in target_labels]
    Y = df_features[target_cols].fillna(0).astype(np.int8).to_numpy()

    split_groups: Optional[np.ndarray] = None
    if bool(getattr(args, "group_split_by_request", False)):
        if "request_fingerprint" in df_features.columns:
            split_groups = df_features["request_fingerprint"].fillna("").astype(str).to_numpy()
        else:
            print(
                "[WARN] --group-split-by-request activo pero processed no contiene request_fingerprint; "
                "se aplica split por fila.",
                flush=True,
            )
    train_idx, tune_idx, calibration_idx, test_idx, split_info = split_train_tune_calibration_test(
        Y, target_labels, args, groups=split_groups
    )
    X_train, Y_train = X[train_idx], Y[train_idx]
    X_tune, Y_tune = X[tune_idx], Y[tune_idx]
    X_calibration, Y_calibration = X[calibration_idx], Y[calibration_idx]
    X_test, Y_test = X[test_idx], Y[test_idx]
    X_train_tune = X[np.concatenate([train_idx, tune_idx])]
    Y_train_tune = Y[np.concatenate([train_idx, tune_idx])]
    df_test_raw = df_features.iloc[test_idx].copy()

    _save_json({
        "created_at": _now_iso(),
        "task": "multilabel",
        "dataset": "harvard",
        "script": Path(__file__).name,
        "inputs": inputs,
        "input_signature": _input_signature(inputs),
        "preprocess_signature": _preprocess_signature(args),
        "request_fingerprint_version": REQUEST_FINGERPRINT_VERSION,
        "features": FIXED_FEATURES,
        "feature_count": len(FIXED_FEATURES),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "training_signature": _training_signature(args),
        "calibration_signature": _calibration_signature(args),
        "target_labels": target_labels,
        "target_meta": target_meta,
        "args": vars(args),
        "split_info": split_info,
        "environment": {
            "python": sys.version,
            "executable": sys.executable,
            "cwd": os.getcwd(),
            "cpu_count": os.cpu_count(),
            "gpu_visible": bool(gpu_visible),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "SLURM_JOB_ID": os.environ.get("SLURM_JOB_ID"),
            "SLURM_JOB_GPUS": os.environ.get("SLURM_JOB_GPUS"),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
        },
        "load_seconds": float(load_seconds),
        "feature_extraction_seconds": float(feature_extraction_seconds),
        "processed_path": str(processed_path),
        "processed_rows": int(len(df_features)),
    }, out_dir / "run_config.json")

    t_select0 = time.perf_counter()
    best = train_select_candidate(args, target_labels, X_train, Y_train, X_tune, Y_tune, out_dir=out_dir)
    train_select_seconds = time.perf_counter() - t_select0
    pd.DataFrame(best["candidate_summary"]).to_csv(out_dir / "candidate_validation_summary.csv", index=False)
    _save_json({"candidates": best["candidate_summary"]}, out_dir / "candidate_validation_summary.json")
    pd.DataFrame(best["candidate_summary"]).to_csv(out_dir / "candidate_tuning_summary.csv", index=False)
    _save_json({"candidates": best["candidate_summary"]}, out_dir / "candidate_tuning_summary.json")

    # Default OFF. Si se habilita para un backend compatible, calibration sigue
    # completamente fuera del refit y calibra los estimadores finales.
    best = refit_final_if_requested(
        args, best, target_labels, X_train_tune, Y_train_tune, X_tune, Y_tune, out_dir=out_dir
    )
    tune_scores = predict_score_matrix(best["records"], X_tune)
    tune_thresholds, tune_thresholds_df = optimize_thresholds(
        Y_tune,
        tune_scores,
        target_labels,
        objective=str(args.threshold_objective),
        grid_size=int(args.threshold_grid_size),
        min_threshold=float(args.threshold_min),
        max_threshold=float(args.threshold_max),
        min_recall=float(args.threshold_min_recall),
    )
    tuning_metrics, tuning_per_label, _ = compute_multilabel_metrics(
        Y_tune,
        tune_scores,
        tune_thresholds,
        target_labels,
        min_attack_score=float(args.min_attack_score),
    )
    tune_thresholds_df = tune_thresholds_df.copy()
    tune_thresholds_df["threshold_source"] = "tune"
    best["thresholds"] = tune_thresholds
    best["thresholds_df"] = tune_thresholds_df
    best["tuning_metrics"] = tuning_metrics
    best["tuning_per_label"] = tuning_per_label
    tune_thresholds_df.to_csv(out_dir / "thresholds_tuning.csv", index=False)
    # Nombre legado conservado para consumidores existentes; contiene tune, no calibration.
    tune_thresholds_df.to_csv(out_dir / "thresholds_validation.csv", index=False)
    tuning_per_label.to_csv(out_dir / "tuning_per_label_metrics.csv", index=False)
    _save_json(tuning_metrics, out_dir / "tuning_metrics_multilabel.json")

    best = calibrate_selected_candidate(best, X_calibration, Y_calibration, target_labels, args)
    best["thresholds_df"].to_csv(out_dir / "thresholds_calibration.csv", index=False)
    best["calibration_per_label"].to_csv(out_dir / "calibration_per_label_metrics.csv", index=False)
    _save_json(best["calibration_metrics"], out_dir / "calibration_metrics_multilabel.json")
    _save_json(
        best.get("joint_threshold_report") or {},
        out_dir / "threshold_joint_optimization_calibration.json",
    )
    best["joint_threshold_report_df"].to_csv(
        out_dir / "threshold_joint_optimization_calibration.csv",
        index=False,
    )
    pd.DataFrame((best.get("joint_threshold_report") or {}).get("moves") or []).to_csv(
        out_dir / "threshold_joint_moves_calibration.csv",
        index=False,
    )
    _save_json(
        {"labels": best.get("calibration_fallback_labels") or []},
        out_dir / "calibration_fallback_labels.json",
    )
    records = best["records"]
    thresholds = np.asarray(best["thresholds"], dtype=np.float32)

    t0 = time.perf_counter()
    cpu0 = time.process_time()
    test_scores = predict_score_matrix(records, X_test)
    inference_batch_seconds = time.perf_counter() - t0
    inference_batch_cpu_seconds = time.process_time() - cpu0

    test_metrics, per_label_df, Y_pred = compute_multilabel_metrics(Y_test, test_scores, thresholds, target_labels, min_attack_score=float(args.min_attack_score))
    per_label_df.to_csv(out_dir / "per_label_metrics.csv", index=False)
    _save_json(test_metrics, out_dir / "test_metrics_multilabel.json")
    pd.DataFrame([test_metrics.get("binary_normal_vs_attack") or {}]).to_csv(out_dir / "binary_normal_vs_attack_metrics.csv", index=False)
    _save_json(test_metrics.get("binary_normal_vs_attack") or {}, out_dir / "binary_normal_vs_attack_metrics.json")
    target_rows = per_label_df[["label", "support", "precision", "recall_tpr", "f1", "pr_auc"]].copy()
    target_rows["metric"] = "per_label_f1"
    target_rows["target"] = float(args.target_label_f1)
    target_rows["target_met"] = target_rows["f1"].fillna(-1.0) >= float(args.target_label_f1)
    normal_metrics = test_metrics.get("binary_normal_vs_attack") or {}
    normal_metric_name = str(args.target_normal_metric)
    normal_metric_value = normal_metrics.get(f"normal_{normal_metric_name}")
    normal_row = pd.DataFrame([{
        "label": "NORMAL",
        "support": normal_metrics.get("normal_support"),
        "precision": normal_metrics.get("normal_precision"),
        "recall_tpr": normal_metrics.get("normal_recall"),
        "f1": normal_metrics.get("normal_f1"),
        "pr_auc": None,
        "metric": f"normal_{normal_metric_name}",
        "target": float(args.target_normal),
        "target_met": bool((normal_metric_value or 0.0) >= float(args.target_normal)),
    }])
    target_attainment = pd.concat([target_rows, normal_row], ignore_index=True)
    target_attainment.to_csv(out_dir / "target_attainment.csv", index=False)
    # Nombre legado que describe los defaults; se conserva para consumidores previos.
    target_attainment.to_csv(out_dir / "target_attainment_f1_85_normal_99.csv", index=False)

    pred_rows = []
    for r, idx0 in enumerate(test_idx):
        true_labs = [target_labels[j] for j, v in enumerate(Y_test[r]) if int(v) == 1]
        pred_labs = [target_labels[j] for j, v in enumerate(Y_pred[r]) if int(v) == 1]
        row: Dict[str, Any] = {
            "row_index": int(idx0),
            "sample_id": str(df_features.iloc[idx0].get("sample_id", idx0)),
            "source_file": str(df_features.iloc[idx0].get("source_file", "")),
            REQUEST_FINGERPRINT_COL: str(df_features.iloc[idx0].get(REQUEST_FINGERPRINT_COL, "")),
            DUPLICATE_GROUP_SIZE_COL: int(df_features.iloc[idx0].get(DUPLICATE_GROUP_SIZE_COL, 1)),
            "y_true_labels_json": json.dumps(true_labs, ensure_ascii=False),
            "y_pred_labels_json": json.dumps(pred_labs, ensure_ascii=False),
            "n_true_labels": int(len(true_labs)),
            "n_pred_labels": int(len(pred_labs)),
            "score_max": float(test_scores[r].max()) if test_scores.shape[1] else 0.0,
        }
        for j, lab in enumerate(target_labels):
            safe = _safe_name(lab)
            row[f"true_{safe}"] = int(Y_test[r, j])
            row[f"pred_{safe}"] = int(Y_pred[r, j])
            row[f"score_{safe}"] = float(test_scores[r, j])
        pred_rows.append(row)
    pd.DataFrame(pred_rows).to_csv(out_dir / "test_predictions.csv", index=False)

    effective_weight_mode = (
        "per_label_tuned"
        if str(best.get("backend")) == "xgboost" and bool(args.xgb_per_label_tuning)
        else str(best.get("weight_mode"))
    )
    selected_model = f"{best['backend']}_ovr__w_{effective_weight_mode}"
    model_path = out_dir / "model_multilabel_xgb_request90.joblib"
    # No guardo candidatos perdedores para no inflar el .joblib.
    records_for_dump = records
    _atomic_joblib_dump({
        "task": "multilabel",
        "dataset": "harvard",
        "model_family": "one_vs_rest_binary_classifiers",
        "selected_backend": best["backend"],
        "selected_model": selected_model,
        "records": records_for_dump,
        "thresholds": thresholds,
        "threshold_objective": str(args.threshold_objective),
        "thresholds_calibrated_on": "calibration",
        "calibration_fallback_labels": best.get("calibration_fallback_labels") or [],
        "joint_threshold_optimization": best.get("joint_threshold_report") or {},
        "features": FIXED_FEATURES,
        "feature_count": len(FIXED_FEATURES),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "training_signature": _training_signature(args),
        "calibration_signature": _calibration_signature(args),
        "input_signature": _input_signature(inputs),
        "preprocess_signature": _preprocess_signature(args),
        "request_fingerprint_version": REQUEST_FINGERPRINT_VERSION,
        "target_labels": target_labels,
        "target_meta": target_meta,
        "target_columns": target_cols,
        "label_mode": str(args.label_mode),
        "config": vars(args),
        "predict_note": "Para inferencia: extraer FIXED_FEATURES con el mismo código y aplicar predict_score_matrix(records, X) >= thresholds.",
    }, model_path, compress=int(getattr(args, "checkpoint_compress", 3)))
    model_size_bytes = int(model_path.stat().st_size) if model_path.exists() else 0

    # La importancia es diagnóstica para futuras iteraciones: se calcula sobre
    # calibration, nunca sobre el test bloqueado.
    fi_paths = compute_feature_importance_multilabel(
        records, X_calibration, Y_calibration, thresholds, target_labels, args, out_dir
    )
    operational_metrics = measure_operational_metrics_multilabel(records, df_test_raw, X_test, thresholds, target_labels, args, out_dir)
    operational_metrics["model_size_bytes"] = model_size_bytes
    operational_metrics["model_size_mb"] = model_size_bytes / (1024.0 * 1024.0) if model_size_bytes else 0.0
    _save_json(operational_metrics, out_dir / "operational_metrics.json")

    run_total_seconds = time.perf_counter() - t_run0
    metrics_payload: Dict[str, Any] = {
        "created_at": _now_iso(),
        "task": "multilabel",
        "dataset": "harvard",
        "dataset_label_mode": str(args.label_mode),
        "feature_policy": "request_only_structural_90",
        "feature_count": len(FIXED_FEATURES),
        "features_used": FIXED_FEATURES,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "training_signature": _training_signature(args),
        "calibration_signature": _calibration_signature(args),
        "input_signature": _input_signature(inputs),
        "preprocess_signature": _preprocess_signature(args),
        "request_fingerprint_version": REQUEST_FINGERPRINT_VERSION,
        "selected_model": selected_model,
        "selected_model_family": "one_vs_rest_binary_classifiers",
        "backend": best["backend"],
        "weight_mode": effective_weight_mode,
        "model_backend": describe_backend(best["backend"]),
        "gpu_requested": args.use_gpu,
        "gpu_visible": bool(gpu_visible),
        "gpu_used": bool(best.get("gpu_used_any")),
        "gpu_note": backend_gpu_note(best["backend"]),
        "labels": target_labels,
        "n_labels": int(len(target_labels)),
        "target_meta": target_meta,
        "train_rows": int(len(train_idx)),
        "tune_rows": int(len(tune_idx)),
        "calibration_rows": int(len(calibration_idx)),
        "valid_rows": int(len(tune_idx) + len(calibration_idx)),
        "test_rows": int(len(test_idx)),
        "test_size": float(args.test_size),
        "valid_size_inside_train": float(args.valid_size),
        "calibration_share_of_valid": float(args.calibration_share),
        "split_info": split_info,
        "seed": int(args.seed),
        "thresholds": {target_labels[i]: float(thresholds[i]) for i in range(len(target_labels))},
        "threshold_objective": str(args.threshold_objective),
        "thresholds_calibrated_on": "calibration",
        "tuning_metrics": best.get("tuning_metrics") or {},
        "calibration_metrics": best.get("calibration_metrics") or {},
        "calibration_fallback_labels": best.get("calibration_fallback_labels") or [],
        "target_config": {
            "label_metric": "f1",
            "label_target": float(args.target_label_f1),
            "normal_metric": str(args.target_normal_metric),
            "normal_target": float(args.target_normal),
        },
        "joint_threshold_optimization": best.get("joint_threshold_report") or {},
        "candidate_selection": {
            "split": "tune",
            "selection_metric": str(args.selection_metric),
            "selection_score": float(best["selection_score"]),
            "candidate_summary": best["candidate_summary"],
        },
        "timing": {
            "load_seconds": float(load_seconds),
            "feature_extraction_seconds": float(feature_extraction_seconds),
            "train_select_seconds": float(train_select_seconds),
            "final_refit_seconds": float(best.get("final_refit_seconds", 0.0)),
            "t_apply_inference_batch_seconds": float(inference_batch_seconds),
            "t_apply_inference_batch_cpu_seconds": float(inference_batch_cpu_seconds),
            "run_total_seconds": float(run_total_seconds),
        },
        "resource_usage": {
            "peak_rss_mb": _get_peak_rss_mb(),
            "current_rss_mb": _get_current_rss_mb(),
            "model_size_bytes": model_size_bytes,
            "model_size_mb": model_size_bytes / (1024.0 * 1024.0) if model_size_bytes else 0.0,
        },
        "test_metrics": test_metrics,
        "operational_metrics": operational_metrics,
        "feature_importance_config": {
            "fi_n_repeats": int(args.fi_n_repeats),
            "fi_max_rows": int(args.fi_max_rows),
            "fi_scoring": str(args.fi_scoring),
        },
        "artifacts": {
            "processed_features": str(processed_path),
            "model": str(model_path),
            "metrics": str(out_dir / "metrics_multilabel_xgb_request90.json"),
            "metrics_flat": str(out_dir / "metrics_flat.csv"),
            "test_metrics": str(out_dir / "test_metrics_multilabel.json"),
            "per_label_metrics": str(out_dir / "per_label_metrics.csv"),
            "target_attainment": str(out_dir / "target_attainment.csv"),
            "target_attainment_legacy": str(out_dir / "target_attainment_f1_85_normal_99.csv"),
            "feature_coverage": str(out_dir / "feature_coverage_request90.csv"),
            "request_deduplication_audit": str(out_dir / "request_deduplication_audit.json"),
            "request_duplicate_groups": str(out_dir / "request_duplicate_groups.csv"),
            "xgb_tuning_trials": str(out_dir / "xgb_tuning_trials_by_label.csv"),
            "xgb_best_params_by_label": str(out_dir / "xgb_best_params_by_label.csv"),
            "candidate_tuning_summary": str(out_dir / "candidate_tuning_summary.csv"),
            "tuning_metrics": str(out_dir / "tuning_metrics_multilabel.json"),
            "tuning_per_label_metrics": str(out_dir / "tuning_per_label_metrics.csv"),
            "thresholds_tuning": str(out_dir / "thresholds_tuning.csv"),
            "calibration_metrics": str(out_dir / "calibration_metrics_multilabel.json"),
            "calibration_per_label_metrics": str(out_dir / "calibration_per_label_metrics.csv"),
            "calibration_fallback_labels": str(out_dir / "calibration_fallback_labels.json"),
            "joint_threshold_report_json": str(out_dir / "threshold_joint_optimization_calibration.json"),
            "joint_threshold_report_csv": str(out_dir / "threshold_joint_optimization_calibration.csv"),
            "joint_threshold_moves": str(out_dir / "threshold_joint_moves_calibration.csv"),
            "thresholds": str(out_dir / "thresholds_calibration.csv"),
            "predictions": str(out_dir / "test_predictions.csv"),
            "operational_metrics": str(out_dir / "operational_metrics.json"),
            **fi_paths,
        },
    }
    _save_json(metrics_payload, out_dir / "metrics_multilabel_xgb_request90.json")
    flat = flatten_metrics_for_comparison(metrics_payload)
    pd.DataFrame([flat]).to_csv(out_dir / "metrics_flat.csv", index=False)
    _write_text(out_dir / "selected_model_summary.txt", make_summary_text(metrics_payload))
    write_aggregate_outputs(args, metrics_payload)

    print("\n=== TEST MULTILABEL ===", flush=True)
    print(
        f"model={selected_model} exact_match={test_metrics.get('exact_match_accuracy')} "
        f"hamming_loss={test_metrics.get('hamming_loss')} f1_macro={test_metrics.get('f1_macro')} "
        f"f1_micro={test_metrics.get('f1_micro')} pr_auc_macro={test_metrics.get('pr_auc_macro')} gpu_used={metrics_payload.get('gpu_used')}",
        flush=True,
    )
    print("\n[OK] Saved artifacts:", flush=True)
    for p in [
        processed_path,
        model_path,
        out_dir / "metrics_multilabel_xgb_request90.json",
        out_dir / "metrics_flat.csv",
        out_dir / "selected_model_summary.txt",
        out_dir / "per_label_metrics.csv",
        out_dir / "thresholds_tuning.csv",
        out_dir / "thresholds_calibration.csv",
        out_dir / "tuning_metrics_multilabel.json",
        out_dir / "calibration_metrics_multilabel.json",
        out_dir / "threshold_joint_optimization_calibration.json",
        out_dir / "threshold_joint_optimization_calibration.csv",
        out_dir / "target_attainment.csv",
        out_dir / "test_predictions.csv",
        out_dir / "feature_importance.csv",
        out_dir / "operational_metrics.json",
        Path(args.definitivo_dir) / "multietiqueta" / "model_comparison_multilabel_definitivo.csv",
    ]:
        if Path(p).exists():
            print(f"  {p}", flush=True)
    return metrics_payload


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Harvard/SR-BH multietiqueta: XGBoost OVR con request90 y tuning por etiqueta")
    ap.add_argument("--project-dir", default=DEFAULT_PROJECT_DIR)
    ap.add_argument("--definitivo-dir", default=None, help="Default: PROJECT_DIR/multietiquetaCorrida2")
    ap.add_argument("--inputs", nargs="*", default=None, help="Inputs Harvard/SR-BH CSV/TSV/GZ. Si no se pasa, usa --harvard-inputs.")
    ap.add_argument("--harvard-inputs", nargs="*", default=None)
    ap.add_argument("--output-dir", default=None, help="Sobrescribe multietiquetaCorrida2/multietiqueta/harvard.")

    # Dataset/preproceso Harvard.
    ap.add_argument("--sample-n", type=int, default=0, help="Head por archivo para pruebas. 0=todo.")
    ap.add_argument("--sep", default="auto")
    ap.add_argument("--method-col", default=RAW_METHOD_COL)
    ap.add_argument("--uri-col", default=RAW_URI_COL)
    ap.add_argument("--protocol-col", default=RAW_PROTOCOL_COL)
    ap.add_argument("--body-col", default=RAW_BODY_COL)
    ap.add_argument("--normal-col", default="000 - Normal")
    ap.add_argument("--label-cols", default="", help="CSV de columnas label. Vacío=autodetecta '^NNN - Nombre'.")
    ap.add_argument("--label-mode", default="native", choices=["native", "capec", "srbh", "optimized-family", "optimized_family", "optimized", "coarse", "coarse-family", "coarse_family", "family", "mapped", "attack-family", "attack_families"], help="native conserva CAPEC; optimized-family agrupa como el multiclase.")
    ap.add_argument("--min-positive-count", type=int, default=20, help="Descarta etiquetas con menos positivos. 20 es estable para split/validación.")
    ap.add_argument("--max-negative-positive-ratio", type=float, default=0.0, help="0=sin filtro por rareza extrema.")
    ap.add_argument("--max-normal-rows", type=int, default=0, help="0=no downsample de normales.")
    ap.add_argument("--drop-invalid-rows", action=argparse.BooleanOptionalAction, default=True, help="Descarta requests completamente vacíos y filas sin NORMAL ni ataque; no modifica el CSV crudo.")
    ap.add_argument(
        "--deduplicate-requests",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Agrupa requests canónicamente idénticas antes del split y une sus etiquetas con OR.",
    )
    ap.add_argument("--keep-labels-regex", default="")
    ap.add_argument("--drop-labels-regex", default="")

    # Split y selección.
    ap.add_argument("--test-size", type=float, default=0.20)
    ap.add_argument(
        "--valid-size",
        type=float,
        default=0.20,
        help="Fracción del bloque posterior a test reservada para tune+calibration.",
    )
    ap.add_argument(
        "--calibration-share",
        type=float,
        default=0.50,
        help="Fracción del holdout tune+calibration usada sólo para calibrar umbrales (default: 50%%).",
    )
    ap.add_argument(
        "--min-calibration-positives",
        type=int,
        default=3,
        help="Mínimo de positivos por etiqueta en calibration; por debajo conserva umbral tune y lo reporta.",
    )
    ap.add_argument(
        "--group-split-by-request",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Impide que un request_fingerprint aparezca en más de un split.",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--selection-metric", default="f1_macro", choices=["f1_macro", "f1_micro", "f1_weighted", "jaccard_macro", "jaccard_micro", "pr_auc_macro", "roc_auc_macro", "exact_match_accuracy"])
    ap.add_argument("--search-level", default="none", choices=["none", "standard", "max"], help="Se conserva por compatibilidad; el tuning XGB real ocurre por etiqueta.")
    ap.add_argument("--candidate-weight-modes", default="none", help="Compatibilidad del candidato global; los pesos XGB se buscan por etiqueta.")
    ap.add_argument(
        "--refit-full-after-thresholds",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Default OFF: evita reutilizar umbrales de un modelo distinto. Activar sólo con una calibración externa/OOF.",
    )

    # Modelo / familias candidatas.
    ap.add_argument(
        "--model-backend",
        default="xgboost",
        help=(
            "auto|all|compare|full o lista separada por comas. Backends: "
            "xgboost,lightgbm,catboost,histgb,extra_trees,random_forest,logistic_regression,linear_svc,sgd_logloss"
        ),
    )
    ap.add_argument("--use-gpu", default="auto", choices=["auto", "on", "off"], help="auto/on intenta GPU para XGBoost, LightGBM y CatBoost cuando hay GPU visible; fallback CPU si falla.")
    ap.add_argument("--xgb-preset", default="max", choices=["fast", "strong", "max"])
    ap.add_argument("--xgb-early-stopping-rounds", type=int, default=100)
    ap.add_argument("--xgb-per-label-tuning", action=argparse.BooleanOptionalAction, default=True, help="Busca hiperparámetros y scale_pos_weight independientemente por etiqueta.")
    ap.add_argument("--xgb-tuning-trials", type=int, default=12, help="Trials para cada etiqueta regular; incluye primero seis pesos base.")
    ap.add_argument("--xgb-weak-label-trials", type=int, default=28, help="Trials para etiquetas que coinciden con --xgb-weak-label-regex.")
    ap.add_argument("--xgb-weak-label-regex", default=r"CAPEC-(153|272|274)\b")
    ap.add_argument("--xgb-label-selection-metric", default="blend", choices=["aucpr", "f1", "blend"], help="Cómo elegir hiperparámetros dentro de cada etiqueta; blend equilibra PR-AUC y F1 en tune.")
    ap.add_argument("--lgbm-preset", default="strong", choices=["fast", "strong", "max"])
    ap.add_argument("--lgbm-early-stopping-rounds", type=int, default=60)
    ap.add_argument("--catboost-preset", default="strong", choices=["fast", "strong", "max"])
    ap.add_argument("--catboost-early-stopping-rounds", type=int, default=60)
    ap.add_argument("--histgb-preset", default="strong", choices=["fast", "strong", "max"])
    ap.add_argument("--forest-preset", default="strong", choices=["fast", "strong", "max"])
    ap.add_argument("--linear-preset", default="strong", choices=["fast", "strong", "max"])
    ap.add_argument("--scale-pos-weight-cap", type=float, default=100.0)
    ap.add_argument("--n-jobs", type=int, default=max(1, min(8, os.cpu_count() or 1)))

    # Umbrales.
    ap.add_argument("--threshold-objective", default="f1", choices=["f1", "f2", "fbeta2", "precision", "recall", "youden", "balanced_accuracy"])
    ap.add_argument("--threshold-grid-size", type=int, default=0, help="Obsoleto: se evalúan todos los scores distintos.")
    ap.add_argument("--threshold-min", type=float, default=0.001)
    ap.add_argument("--threshold-max", type=float, default=0.999)
    ap.add_argument("--threshold-min-recall", type=float, default=0.0)
    ap.add_argument(
        "--joint-threshold-optimization",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Ajusta conjuntamente los cortes en calibration para los targets por etiqueta y NORMAL.",
    )
    ap.add_argument(
        "--target-label-f1",
        type=float,
        default=0.85,
        help="F1 objetivo por etiqueta usado por la calibración conjunta.",
    )
    ap.add_argument(
        "--target-normal",
        type=float,
        default=0.99,
        help="Objetivo de NORMAL en calibration y en el reporte final.",
    )
    ap.add_argument(
        "--target-normal-metric",
        default="recall",
        choices=["recall", "f1"],
        help="Métrica objetivo de NORMAL; recall equivale a filas normales clasificadas exactamente sin ataques.",
    )
    ap.add_argument(
        "--joint-threshold-max-passes",
        type=int,
        default=6,
        help="Pasadas deterministas máximas del coordinate descent conjunto sobre calibration.",
    )
    ap.add_argument("--min-attack-score", type=float, default=0.0, help="0=desactivado; >0 fuerza una etiqueta si max score supera este valor.")

    # Paralelismo/operativas/FI.
    ap.add_argument("--feature-workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    ap.add_argument("--feature-chunksize", type=int, default=256)
    ap.add_argument("--operational-max-rows", type=int, default=0)
    ap.add_argument("--fi-n-repeats", type=int, default=3)
    ap.add_argument("--fi-max-rows", type=int, default=10000)
    ap.add_argument("--fi-scoring", default="f1_macro", choices=["f1_macro", "f1_micro", "f1_weighted", "jaccard_macro", "jaccard_micro", "exact_match_accuracy", "pr_auc_macro", "roc_auc_macro"])

    # Resume / checkpoints. Default ON: reusa processed + candidatos terminados si el job fue cortado por timeout.
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True, help="Reanuda desde processed/checkpoints existentes. Usá --no-resume para forzar entrenamiento limpio sin borrar outputs.")
    ap.add_argument("--checkpoint-dir", default="", help="Directorio de checkpoints. Default: OUTPUT_DIR/checkpoints.")
    ap.add_argument("--checkpoint-compress", type=int, default=3, help="Compresión joblib para checkpoints/modelo parcial; 0 sin compresión, 3 balanceado.")
    ap.add_argument("--force-reprocess", action="store_true", help="Ignora processed_request90 previo aunque --resume esté activo; no borra checkpoints compatibles.")
    ap.add_argument("--resume-skip-complete", action=argparse.BooleanOptionalAction, default=True, help="Si ya existen métricas finales, modelo y resumen, termina sin recomputar.")
    ap.add_argument("--clean-output-dir", action="store_true", help="Borra OUTPUT_DIR al empezar. OJO: elimina checkpoints y anula el beneficio de --resume para esa corrida.")
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args(argv)

    if args.definitivo_dir is None:
        args.definitivo_dir = str(Path(args.project_dir) / DEFAULT_RESULTS_DIRNAME)
    if not args.harvard_inputs:
        args.harvard_inputs = [os.environ.get("HARVARD_RAW_DIR", str(Path(args.project_dir) / "data" / "raw" / "harvard"))]
    for name in ["test_size", "valid_size", "calibration_share"]:
        value = float(getattr(args, name))
        if not 0.0 < value < 1.0:
            raise SystemExit(f"--{name.replace('_', '-')} debe estar entre 0 y 1; recibido={value}")
    if int(args.min_calibration_positives) < 1:
        raise SystemExit("--min-calibration-positives debe ser >= 1")
    for name in ["target_label_f1", "target_normal"]:
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            raise SystemExit(f"--{name.replace('_', '-')} debe estar entre 0 y 1; recibido={value}")
    if int(args.joint_threshold_max_passes) < 0:
        raise SystemExit("--joint-threshold-max-passes debe ser >= 0")
    if bool(args.joint_threshold_optimization) and float(args.min_attack_score) > 0:
        raise SystemExit(
            "La calibración conjunta requiere --min-attack-score=0 para que las restricciones F1 "
            "coincidan exactamente con las predicciones multilabel."
        )
    if bool(args.refit_full_after_thresholds) and bool(args.xgb_per_label_tuning):
        raise SystemExit(
            "--refit-full-after-thresholds no es compatible con los umbrales calibrados del tuning por etiqueta. "
            "Usá --no-refit-full-after-thresholds (default) para una evaluación válida."
        )
    if _canonical_backend_name(str(args.model_backend)) != "xgboost" and bool(args.xgb_per_label_tuning):
        print("[WARN] --xgb-per-label-tuning sólo aplica a XGBoost.", flush=True)
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    dep = dependency_check()
    if _canonical_backend_name(str(args.model_backend)) == "xgboost" and not dep.get("xgboost_available"):
        raise SystemExit("[ERROR] --model-backend=xgboost requiere una instalación importable de xgboost.")
    Path(args.definitivo_dir).mkdir(parents=True, exist_ok=True)
    _save_json({"created_at": _now_iso(), "dependency_check": dep, "args": vars(args)}, Path(args.definitivo_dir) / "last_invocation_multilabel.json")
    if args.check_only:
        print("[OK] CHECK_ONLY: dependencias y argumentos OK. No entreno.", flush=True)
        return 0
    train_eval_harvard_multilabel(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
