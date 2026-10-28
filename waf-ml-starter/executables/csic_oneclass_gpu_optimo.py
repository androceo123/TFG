#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline optimizado para CSIC 2010 raw HTTP -> features WAF -> detector one-class.

Objetivo:
  1) Leer normalTrafficTraining.txt, normalTrafficTest.txt y anomalousTrafficTest.txt
     desde /home_data/aroman/TFG/waf-ml-starter/data/raw/csic/.
  2) Borrar/recrear el CSIC procesado si se pide --force-reprocess.
  3) Extraer features pensadas para los ataques TORPEDA-*:
     ANOMALOUS, BufferOverflow, CRLFi, FormatString, LDAPi, SQLi, SSI, XPath, XSS.
  4) Entrenar SOLO con trafico normal un detector one-class rapido.
     - Prioridad: RAPIDS cuML IsolationForest en GPU.
     - Fallback: sklearn IsolationForest CPU multihilo.
     - Opcional: autoencoder PyTorch CUDA si esta disponible.
  5) Tunear hiperparametros y umbral usando un split de validacion etiquetado,
     manteniendo el entrenamiento one-class: el modelo se ajusta solo con normales.
  6) Evaluar en test holdout y en el test oficial completo.
  7) Calcular feature importance por permutacion y guardar resultados.

Uso tipico:
  python csic_oneclass_gpu_optimo.py --force-reprocess

Resultados:
  processed dataset: data/processed/csic_oneclass_gpu_optimo/csic_oneclass_features.parquet/csv
  results:           resultsOptimo/csic/oneclass_gpu_optimo/
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import inspect
import itertools
import json
import math
import os
import random
import re
import shutil
import sys
import time
import traceback
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import parse_qsl, unquote_plus, urlsplit

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import joblib
except Exception as exc:  # pragma: no cover
    raise SystemExit("[ERROR] Falta joblib. Instala: pip install joblib") from exc

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        matthews_corrcoef,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "[ERROR] Faltan dependencias sklearn/numpy/pandas. "
        "Instala en el venv: pip install numpy pandas scikit-learn joblib pyarrow matplotlib"
    ) from exc


# ---------------------------------------------------------------------------
# Regex precompiladas
# ---------------------------------------------------------------------------
REQ_START_RE = re.compile(
    r"^(GET|POST|PUT|DELETE|HEAD|OPTIONS|PATCH|TRACE|CONNECT)\s+", re.IGNORECASE
)
PERCENT_ENC_RE = re.compile(r"%(?:[0-9a-fA-F]{2})")
BAD_PERCENT_RE = re.compile(r"%(?![0-9a-fA-F]{2})")
HTML_ENTITY_RE = re.compile(r"&(lt|gt|amp|quot|apos|#x?[0-9a-fA-F]+);?", re.IGNORECASE)
UNICODE_ESCAPE_RE = re.compile(r"\\u[0-9a-fA-F]{4}")
HEX_ESCAPE_RE = re.compile(r"\\x[0-9a-fA-F]{2}")
TOKEN_RE = re.compile(r"[A-Za-z0-9_%./\\:-]+")
ALNUM_RUN_RE = re.compile(r"[A-Za-z0-9]{8,}")

SQL_KEYWORDS_RE = re.compile(
    r"\b(select|union|insert|update|delete|drop|where|from|sleep|benchmark|pg_sleep|"
    r"waitfor|delay|having|information_schema|table_name|load_file|outfile|substr|"
    r"substring|concat|cast|char|declare|exec|execute|xp_cmdshell|or|and)\b",
    re.IGNORECASE,
)
SQL_UNION_SELECT_RE = re.compile(
    r"\bunion(?:\s|/\*.*?\*/|%20|\+|%09|%0a|%0d)+(?:all\s+)?select\b",
    re.IGNORECASE | re.DOTALL,
)
SQL_TAUTOLOGY_RE = re.compile(
    r"(?:\b(?:or|and)\b\s+['\"]?[\w.]+['\"]?\s*=\s*['\"]?[\w.]+['\"]?)|"
    r"(?:\b1\s*=\s*1\b)|(?:'[^']*'\s*=\s*'[^']*')|(?:\"[^\"]*\"\s*=\s*\"[^\"]*\")",
    re.IGNORECASE,
)
SQL_COMMENT_RE = re.compile(r"(--|#|/\*|\*/|%2d%2d|%23)", re.IGNORECASE)
SQL_SLEEP_RE = re.compile(r"\b(sleep|benchmark|pg_sleep|waitfor\s+delay)\b", re.IGNORECASE)
SQL_INFO_SCHEMA_RE = re.compile(r"\b(information_schema|sysobjects|syscolumns|sqlite_master)\b", re.IGNORECASE)

XSS_TAG_RE = re.compile(
    r"<\s*/?\s*(script|img|svg|iframe|body|object|embed|style|a|input|video|audio|math|meta)",
    re.IGNORECASE,
)
XSS_EVENT_RE = re.compile(r"\bon[a-z]{3,30}\s*=", re.IGNORECASE)
XSS_JS_SCHEME_RE = re.compile(r"\b(javascript|vbscript|data)\s*:", re.IGNORECASE)
XSS_FUNC_RE = re.compile(r"\b(alert|confirm|prompt|eval|atob|fromcharcode)\s*\(", re.IGNORECASE)
XSS_COOKIE_RE = re.compile(r"\bdocument\s*\.\s*(cookie|domain|location)\b", re.IGNORECASE)

SSI_DIRECTIVE_RE = re.compile(r"<!--\s*#\s*(include|exec|echo|config|fsize|flastmod|printenv)", re.IGNORECASE)
SSI_EXEC_RE = re.compile(r"\b(exec|cmd|command)\s*=|/bin/(?:sh|bash|cat|ls)", re.IGNORECASE)
SSI_INCLUDE_RE = re.compile(r"\b(include|virtual|file)\s*=", re.IGNORECASE)

CRLF_ENC_RE = re.compile(r"(%0d|%0a|%0D|%0A|\\r|\\n|%250d|%250a)")
HEADER_WORD_RE = re.compile(
    r"\b(content-length|set-cookie|location|transfer-encoding|host|referer|user-agent)\s*:",
    re.IGNORECASE,
)

# Se remueven percent-encodings validos antes de buscar format strings para no confundir %2f con %f.
FMT_TOKEN_RE = re.compile(
    r"%(?:\d+\$)?[+#0\- ]*(?:\d+|\*)?(?:\.\d+)?[hlLzjt]*[diuoxXfFeEgGaAcspn]"
)
FMT_WRITE_RE = re.compile(r"%(?:\d+\$)?[+#0\- ]*(?:\d+|\*)?(?:\.\d+)?[hlLzjt]*n")
FMT_HEX_PTR_RE = re.compile(r"%(?:\d+\$)?[+#0\- ]*(?:\d+|\*)?(?:\.\d+)?[hlLzjt]*[xXp]")

LDAP_DN_RE = re.compile(r"\b(cn|uid|ou|dc|sn|mail|objectClass)\s*=", re.IGNORECASE)
LDAP_FILTER_RE = re.compile(r"\([|&!]?\s*(?:cn|uid|ou|dc|sn|mail|objectClass)?\s*[~<>]?=", re.IGNORECASE)
LDAP_ESC_RE = re.compile(r"\\[0-9a-fA-F]{2}|%00|\\00", re.IGNORECASE)

XPATH_FUNC_RE = re.compile(
    r"\b(count|string|substring|contains|starts-with|name|text|node|position|last|normalize-space)\s*\(",
    re.IGNORECASE,
)
XPATH_AXIS_RE = re.compile(
    r"\b(ancestor|descendant|following|preceding|parent|child|self|attribute)::", re.IGNORECASE
)
XPATH_TAUTOLOGY_RE = re.compile(
    r"(?:\bor\b|\band\b)\s+['\"]?\w+['\"]?\s*=\s*['\"]?\w+['\"]?", re.IGNORECASE
)

TRAVERSAL_RE = re.compile(r"(\.\./|\.\.\\|%2e%2e%2f|%2e%2e/|%252e%252e%252f)", re.IGNORECASE)
SHELL_KEYWORD_RE = re.compile(
    r"\b(cat|ls|wget|curl|bash|sh|cmd|powershell|nc|netcat|chmod|chown|whoami|id|uname)\b",
    re.IGNORECASE,
)

ATTACK_FAMILIES = [
    "TORPEDA-ANOMALOUS",
    "TORPEDA-BufferOverflow",
    "TORPEDA-CRLFi",
    "TORPEDA-FormatString",
    "TORPEDA-LDAPi",
    "TORPEDA-SQLi",
    "TORPEDA-SSI",
    "TORPEDA-XPath",
    "TORPEDA-XSS",
]

METADATA_COLS = [
    "source_file",
    "source_index",
    "source_split",
    "method",
    "protocol",
    "target",
    "path_preview",
    "query_preview",
    "body_preview",
    "raw_sha1",
    "label_binary",
    "label",
    "attack_family_heuristic",
]


# ---------------------------------------------------------------------------
# Utilidades generales
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    print(msg, flush=True)


def now_s() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj


def save_json(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(to_jsonable(data), indent=2, ensure_ascii=False), encoding="utf-8")


def preview_text(s: str, max_len: int = 240) -> str:
    s = (s or "").replace("\r", "\\r").replace("\n", "\\n")
    if len(s) > max_len:
        return s[:max_len] + "..."
    return s


def parse_int_or_auto(x: str) -> Any:
    x = str(x).strip()
    if x.lower() == "auto":
        return "auto"
    return int(float(x))


def parse_grid(value: str, cast: Any = float) -> List[Any]:
    out: List[Any] = []
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        out.append(cast(item))
    return out


def safe_ratio(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def shannon_entropy(text: str) -> float:
    if not text:
        return 0.0
    # Para no penalizar demasiado tiempo en cuerpos enormes, cap suave.
    if len(text) > 20000:
        text = text[:20000]
    counts = Counter(text)
    n = float(len(text))
    return float(-sum((c / n) * math.log2(c / n) for c in counts.values()))


def max_same_char_run(text: str) -> int:
    if not text:
        return 0
    best = 1
    cur = 1
    prev = text[0]
    for ch in text[1:]:
        if ch == prev:
            cur += 1
            if cur > best:
                best = cur
        else:
            prev = ch
            cur = 1
    return int(best)


def max_alnum_run(text: str) -> int:
    runs = ALNUM_RUN_RE.findall(text or "")
    return int(max((len(x) for x in runs), default=0))


def count_re(pattern: re.Pattern[str], text: str) -> int:
    if not text:
        return 0
    return int(len(pattern.findall(text)))


def decode_repeated(text: str, rounds: int = 2) -> str:
    out = text or ""
    for _ in range(max(0, rounds)):
        try:
            new = unquote_plus(out)
        except Exception:
            break
        if new == out:
            break
        out = new
    return out


def safe_parse_qsl(text: str) -> List[Tuple[str, str]]:
    if not text:
        return []
    try:
        return parse_qsl(
            text,
            keep_blank_values=True,
            strict_parsing=False,
            encoding="latin-1",
            errors="replace",
            max_num_fields=20000,
        )
    except TypeError:
        try:
            return parse_qsl(
                text,
                keep_blank_values=True,
                strict_parsing=False,
                encoding="latin-1",
                errors="replace",
            )
        except Exception:
            pass
    except Exception:
        pass

    # Fallback manual para payloads malformados.
    pairs: List[Tuple[str, str]] = []
    for part in re.split(r"[&;]", text):
        if not part:
            continue
        if "=" in part:
            k, v = part.split("=", 1)
        else:
            k, v = part, ""
        pairs.append((decode_repeated(k, 1), decode_repeated(v, 1)))
    return pairs


def add_text_stats(features: Dict[str, Any], prefix: str, text: str) -> None:
    text = text or ""
    n = len(text)
    alpha = sum(ch.isalpha() for ch in text)
    digit = sum(ch.isdigit() for ch in text)
    space = sum(ch.isspace() for ch in text)
    non_ascii = sum(ord(ch) > 127 for ch in text)
    control = sum((ord(ch) < 32 and ch not in "\t\n\r") or ord(ch) == 127 for ch in text)
    punct = sum((not ch.isalnum()) and (not ch.isspace()) for ch in text)

    features[f"{prefix}_len"] = n
    features[f"{prefix}_entropy"] = shannon_entropy(text)
    features[f"{prefix}_alpha_count"] = alpha
    features[f"{prefix}_digit_count"] = digit
    features[f"{prefix}_space_count"] = space
    features[f"{prefix}_punct_count"] = punct
    features[f"{prefix}_non_ascii_count"] = non_ascii
    features[f"{prefix}_control_count"] = control
    features[f"{prefix}_alpha_ratio"] = safe_ratio(alpha, n)
    features[f"{prefix}_digit_ratio"] = safe_ratio(digit, n)
    features[f"{prefix}_space_ratio"] = safe_ratio(space, n)
    features[f"{prefix}_punct_ratio"] = safe_ratio(punct, n)
    features[f"{prefix}_non_ascii_ratio"] = safe_ratio(non_ascii, n)
    features[f"{prefix}_max_same_char_run"] = max_same_char_run(text)
    features[f"{prefix}_max_alnum_run"] = max_alnum_run(text)


def count_char_features(features: Dict[str, Any], prefix: str, text: str) -> None:
    chars = {
        "single_quote": "'",
        "double_quote": '"',
        "backtick": "`",
        "semicolon": ";",
        "colon": ":",
        "comma": ",",
        "dot": ".",
        "slash": "/",
        "backslash": "\\",
        "pipe": "|",
        "ampersand": "&",
        "equals": "=",
        "question": "?",
        "lt": "<",
        "gt": ">",
        "paren_open": "(",
        "paren_close": ")",
        "bracket_open": "[",
        "bracket_close": "]",
        "brace_open": "{",
        "brace_close": "}",
        "dash": "-",
        "underscore": "_",
        "asterisk": "*",
        "hash": "#",
        "percent": "%",
        "plus": "+",
        "dollar": "$",
    }
    for name, ch in chars.items():
        features[f"{prefix}_{name}_count"] = int((text or "").count(ch))


# ---------------------------------------------------------------------------
# Parser CSIC raw HTTP
# ---------------------------------------------------------------------------

def read_text_lossy(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("utf-8", "latin-1", "iso-8859-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def iter_http_blocks(path: Path) -> Iterable[str]:
    text = read_text_lossy(path)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    block: List[str] = []

    for line in lines:
        is_start = bool(REQ_START_RE.match(line.strip()))
        if is_start and block:
            candidate = "\n".join(block).strip("\n")
            if candidate:
                yield candidate
            block = [line]
        else:
            if not block and not line.strip():
                continue
            block.append(line)

    candidate = "\n".join(block).strip("\n")
    if candidate:
        yield candidate


def parse_http_block(block: str) -> Dict[str, Any]:
    lines = block.splitlines()
    request_line = lines[0].strip() if lines else ""
    # Captura robusta: algunos payloads anomalos pueden traer espacios sin encodear
    # dentro del target; por eso tomamos el ultimo token HTTP/x.y como protocolo.
    m_req = re.match(r"^(\S+)\s+(.+?)\s+(HTTP/\d(?:\.\d)?)\s*$", request_line, flags=re.IGNORECASE)
    if m_req:
        method = m_req.group(1).upper()
        target = m_req.group(2)
        protocol = m_req.group(3)
    else:
        parts = request_line.split()
        method = parts[0].upper() if len(parts) >= 1 else ""
        target = parts[1] if len(parts) >= 2 else ""
        protocol = parts[2] if len(parts) >= 3 else ""

    headers: Dict[str, str] = {}
    body_lines: List[str] = []
    in_body = False
    for line in lines[1:]:
        if not in_body and line.strip() == "":
            in_body = True
            continue
        if not in_body and ":" in line:
            key, val = line.split(":", 1)
            headers[key.strip().lower()] = val.strip()
        else:
            in_body = True
            body_lines.append(line)
    body = "\n".join(body_lines).strip("\n")

    try:
        split = urlsplit(target)
        path = split.path or ""
        query = split.query or ""
    except Exception:
        path = target
        query = ""

    # En targets malformados puede venir una query sin que urlsplit la detecte.
    if not query and "?" in target:
        path, query = target.split("?", 1)

    return {
        "request_line": request_line,
        "method": method,
        "target": target,
        "protocol": protocol,
        "headers": headers,
        "body": body,
        "path": path,
        "query": query,
        "raw_block": block,
    }


def file_label_info(path: Path) -> Tuple[int, str, str]:
    name = path.name.lower()
    if "anomal" in name or "attack" in name:
        return 1, "TORPEDA-ANOMALOUS", "official_test"
    if "training" in name or "train" in name:
        return 0, "TORPEDA-NORMAL", "train_normal"
    return 0, "TORPEDA-NORMAL", "official_test"


def extract_features(parsed: Dict[str, Any], source_file: str, source_index: int, label_binary: int, base_label: str) -> Dict[str, Any]:
    method = parsed["method"]
    target = parsed["target"]
    path = parsed["path"]
    query = parsed["query"]
    body = parsed["body"]
    headers = parsed["headers"]
    raw_block = parsed["raw_block"]

    content_type = headers.get("content-type", "")
    content_length_raw = headers.get("content-length", "")
    try:
        content_length_declared = int(re.sub(r"[^0-9]", "", content_length_raw) or 0)
    except Exception:
        content_length_declared = 0

    # Superficie de ataque: evita contar separadores reales de headers como CRLF.
    attack_surface_raw = " ".join([path, query, body, headers.get("cookie", "")])
    attack_surface_dec1 = decode_repeated(attack_surface_raw, 1)
    attack_surface_dec2 = decode_repeated(attack_surface_raw, 2)
    attack_surface_low = attack_surface_dec2.lower()

    target_dec = decode_repeated(target, 2)
    query_dec = decode_repeated(query, 2)
    body_dec = decode_repeated(body, 2)
    path_dec = decode_repeated(path, 2)

    query_params = safe_parse_qsl(query)
    body_params: List[Tuple[str, str]] = []
    # CSIC suele traer formularios URL-encoded; si el body tiene pares, se extraen igual aunque falte content-type.
    if body and ("=" in body or "application/x-www-form-urlencoded" in content_type.lower()):
        body_params = safe_parse_qsl(body)
    params = query_params + body_params
    param_names = [str(k) for k, _ in params]
    param_values = [str(v) for _, v in params]
    param_values_text_raw = " ".join(param_values)
    param_values_text_dec = decode_repeated(param_values_text_raw, 2)

    token_list = TOKEN_RE.findall(attack_surface_dec2)
    token_lengths = [len(t) for t in token_list] if token_list else [0]
    value_lengths = [len(v) for v in param_values] if param_values else [0]
    name_lengths = [len(k) for k in param_names] if param_names else [0]
    duplicate_names = len(param_names) - len(set(param_names))

    features: Dict[str, Any] = {
        "source_file": source_file,
        "source_index": source_index,
        "source_split": "",
        "method": method,
        "protocol": parsed.get("protocol", ""),
        "target": preview_text(target, 500),
        "path_preview": preview_text(path_dec, 300),
        "query_preview": preview_text(query_dec, 300),
        "body_preview": preview_text(body_dec, 300),
        "raw_sha1": hashlib.sha1(raw_block.encode("utf-8", errors="ignore")).hexdigest(),
        "label_binary": int(label_binary),
        "label": base_label,
    }

    # Metodo / estructura HTTP.
    features.update(
        {
            "is_get": int(method == "GET"),
            "is_post": int(method == "POST"),
            "is_head": int(method == "HEAD"),
            "is_other_method": int(method not in {"GET", "POST", "HEAD"}),
            "request_line_len": len(parsed.get("request_line", "")),
            "target_len": len(target),
            "path_len": len(path),
            "query_len": len(query),
            "body_len": len(body),
            "raw_block_len": len(raw_block),
            "header_count": len(headers),
            "headers_total_len": sum(len(k) + len(v) for k, v in headers.items()),
            "has_cookie": int("cookie" in headers),
            "cookie_len": len(headers.get("cookie", "")),
            "user_agent_len": len(headers.get("user-agent", "")),
            "host_len": len(headers.get("host", "")),
            "content_type_len": len(content_type),
            "content_length_declared": content_length_declared,
            "content_length_delta_abs": abs(content_length_declared - len(body)) if content_length_declared else 0,
            "has_query": int(bool(query)),
            "has_body": int(bool(body)),
            "query_param_count": len(query_params),
            "body_param_count": len(body_params),
            "param_count": len(params),
            "unique_param_name_count": len(set(param_names)),
            "duplicate_param_name_count": duplicate_names,
            "empty_param_value_count": sum(1 for v in param_values if v == ""),
            "mean_param_name_len": float(np.mean(name_lengths)),
            "max_param_name_len": int(np.max(name_lengths)),
            "mean_param_value_len": float(np.mean(value_lengths)),
            "max_param_value_len": int(np.max(value_lengths)),
            "sum_param_value_len": int(np.sum(value_lengths)),
            "std_param_value_len": float(np.std(value_lengths)),
            "token_count": len(token_list),
            "mean_token_len": float(np.mean(token_lengths)),
            "max_token_len": int(np.max(token_lengths)),
            "std_token_len": float(np.std(token_lengths)),
        }
    )

    # Estadisticas de texto. Estas suelen ser muy fuertes para one-class porque los ataques rompen
    # distribuciones de longitud, entropia y puntuacion del trafico normal.
    add_text_stats(features, "target_dec", target_dec)
    add_text_stats(features, "path_dec", path_dec)
    add_text_stats(features, "query_dec", query_dec)
    add_text_stats(features, "body_dec", body_dec)
    add_text_stats(features, "surface_dec", attack_surface_dec2)
    add_text_stats(features, "param_values_dec", param_values_text_dec)
    count_char_features(features, "surface", attack_surface_dec2)

    # Encoding / ofuscacion.
    double_decoded = decode_repeated(attack_surface_raw, 1)
    features.update(
        {
            "percent_count_raw": attack_surface_raw.count("%"),
            "percent_encoded_count": count_re(PERCENT_ENC_RE, attack_surface_raw),
            "bad_percent_count": count_re(BAD_PERCENT_RE, attack_surface_raw),
            "double_encoded_count": count_re(PERCENT_ENC_RE, double_decoded),
            "plus_count_raw": attack_surface_raw.count("+"),
            "encoded_crlf_count": count_re(CRLF_ENC_RE, attack_surface_raw),
            "encoded_null_count": len(re.findall(r"%00|%2500|\\0", attack_surface_raw, flags=re.IGNORECASE)),
            "encoded_slash_backslash_count": len(
                re.findall(r"%2f|%5c|%252f|%255c", attack_surface_raw, flags=re.IGNORECASE)
            ),
            "encoded_dot_count": len(re.findall(r"%2e|%252e", attack_surface_raw, flags=re.IGNORECASE)),
            "html_entity_count": count_re(HTML_ENTITY_RE, attack_surface_raw + " " + attack_surface_dec2),
            "unicode_escape_count": count_re(UNICODE_ESCAPE_RE, attack_surface_raw),
            "hex_escape_count": count_re(HEX_ESCAPE_RE, attack_surface_raw),
            "raw_backslash_count": attack_surface_raw.count("\\"),
        }
    )

    # SQL injection.
    features.update(
        {
            "sql_keyword_count": count_re(SQL_KEYWORDS_RE, attack_surface_dec2),
            "sql_union_select_count": count_re(SQL_UNION_SELECT_RE, attack_surface_dec2 + " " + attack_surface_raw),
            "sql_tautology_count": count_re(SQL_TAUTOLOGY_RE, attack_surface_dec2),
            "sql_comment_count": count_re(SQL_COMMENT_RE, attack_surface_dec2 + " " + attack_surface_raw),
            "sql_sleep_benchmark_count": count_re(SQL_SLEEP_RE, attack_surface_dec2),
            "sql_info_schema_count": count_re(SQL_INFO_SCHEMA_RE, attack_surface_dec2),
            "sql_stacked_query_score": int(";" in attack_surface_dec2)
            * count_re(SQL_KEYWORDS_RE, attack_surface_dec2),
            "sql_quote_count": attack_surface_dec2.count("'") + attack_surface_dec2.count('"'),
        }
    )

    # XSS.
    features.update(
        {
            "xss_tag_count": count_re(XSS_TAG_RE, attack_surface_dec2),
            "xss_event_handler_count": count_re(XSS_EVENT_RE, attack_surface_dec2),
            "xss_js_scheme_count": count_re(XSS_JS_SCHEME_RE, attack_surface_dec2),
            "xss_func_count": count_re(XSS_FUNC_RE, attack_surface_dec2),
            "xss_document_cookie_count": count_re(XSS_COOKIE_RE, attack_surface_dec2),
            "xss_angle_pair_count": min(attack_surface_dec2.count("<"), attack_surface_dec2.count(">")),
        }
    )

    # CRLF / response splitting.
    features.update(
        {
            "crlf_encoded_count": count_re(CRLF_ENC_RE, attack_surface_raw),
            "crlf_literal_count": attack_surface_raw.count("\\r") + attack_surface_raw.count("\\n"),
            "crlf_header_word_count": count_re(HEADER_WORD_RE, attack_surface_dec2),
        }
    )

    # Format string.
    fmt_source = PERCENT_ENC_RE.sub(" ", attack_surface_raw) + " " + attack_surface_dec2
    features.update(
        {
            "fmt_token_count": count_re(FMT_TOKEN_RE, fmt_source),
            "fmt_write_count": count_re(FMT_WRITE_RE, fmt_source),
            "fmt_hex_ptr_count": count_re(FMT_HEX_PTR_RE, fmt_source),
        }
    )

    # LDAP injection.
    features.update(
        {
            "ldap_dn_keyword_count": count_re(LDAP_DN_RE, attack_surface_dec2),
            "ldap_filter_count": count_re(LDAP_FILTER_RE, attack_surface_dec2),
            "ldap_operator_count": attack_surface_dec2.count("(|")
            + attack_surface_dec2.count("(&")
            + attack_surface_dec2.count("(!"),
            "ldap_wildcard_count": attack_surface_dec2.count("*"),
            "ldap_escape_count": count_re(LDAP_ESC_RE, attack_surface_raw + " " + attack_surface_dec2),
        }
    )

    # XPath injection.
    features.update(
        {
            "xpath_func_count": count_re(XPATH_FUNC_RE, attack_surface_dec2),
            "xpath_axis_count": count_re(XPATH_AXIS_RE, attack_surface_dec2),
            "xpath_predicate_count": min(attack_surface_dec2.count("["), attack_surface_dec2.count("]")),
            "xpath_at_count": attack_surface_dec2.count("@"),
            "xpath_double_slash_count": path_dec.count("//") + query_dec.count("//") + body_dec.count("//"),
            "xpath_tautology_count": count_re(XPATH_TAUTOLOGY_RE, attack_surface_dec2),
        }
    )

    # SSI.
    features.update(
        {
            "ssi_directive_count": count_re(SSI_DIRECTIVE_RE, attack_surface_dec2),
            "ssi_exec_count": count_re(SSI_EXEC_RE, attack_surface_dec2),
            "ssi_include_count": count_re(SSI_INCLUDE_RE, attack_surface_dec2),
        }
    )

    # Buffer overflow / payloads largos.
    longest_value = max(value_lengths) if value_lengths else 0
    max_same = max_same_char_run(attack_surface_dec2)
    features.update(
        {
            "long_value_over_64_count": sum(1 for v in value_lengths if v >= 64),
            "long_value_over_128_count": sum(1 for v in value_lengths if v >= 128),
            "long_value_over_256_count": sum(1 for v in value_lengths if v >= 256),
            "longest_value_len": int(longest_value),
            "target_over_512": int(len(target) >= 512),
            "body_over_1024": int(len(body) >= 1024),
            "max_same_char_run_surface": int(max_same),
            "same_char_run_over_16": int(max_same >= 16),
            "same_char_run_over_32": int(max_same >= 32),
            "max_alnum_run_surface": max_alnum_run(attack_surface_dec2),
        }
    )

    # Extras utiles para WAF aunque no esten en la lista principal.
    shell_meta = sum(attack_surface_dec2.count(ch) for ch in [";", "|", "&", "`", "$", ">", "<"])
    features.update(
        {
            "traversal_count": count_re(TRAVERSAL_RE, attack_surface_raw + " " + attack_surface_dec2),
            "shell_meta_count": int(shell_meta),
            "shell_keyword_count": count_re(SHELL_KEYWORD_RE, attack_surface_dec2),
        }
    )

    # Scores heurísticos por familia. Son features y tambien sirven para reportar recall por ataque.
    scores = {
        "TORPEDA-SQLi": (
            2 * features["sql_keyword_count"]
            + 6 * features["sql_union_select_count"]
            + 5 * features["sql_tautology_count"]
            + 2 * features["sql_comment_count"]
            + 6 * features["sql_sleep_benchmark_count"]
            + 5 * features["sql_info_schema_count"]
            + 2 * features["sql_stacked_query_score"]
        ),
        "TORPEDA-XSS": (
            8 * features["xss_tag_count"]
            + 5 * features["xss_event_handler_count"]
            + 6 * features["xss_js_scheme_count"]
            + 5 * features["xss_func_count"]
            + 5 * features["xss_document_cookie_count"]
            + features["xss_angle_pair_count"]
        ),
        "TORPEDA-CRLFi": (
            8 * features["crlf_encoded_count"]
            + 6 * features["crlf_literal_count"]
            + 4 * features["crlf_header_word_count"]
        ),
        "TORPEDA-FormatString": (
            7 * features["fmt_write_count"]
            + 4 * features["fmt_hex_ptr_count"]
            + 3 * features["fmt_token_count"]
        ),
        "TORPEDA-LDAPi": (
            4 * features["ldap_filter_count"]
            + 3 * features["ldap_dn_keyword_count"]
            + 2 * features["ldap_operator_count"]
            + features["ldap_wildcard_count"]
            + 2 * features["ldap_escape_count"]
        ),
        "TORPEDA-XPath": (
            5 * features["xpath_func_count"]
            + 4 * features["xpath_axis_count"]
            + 2 * features["xpath_predicate_count"]
            + 2 * features["xpath_at_count"]
            + 2 * features["xpath_double_slash_count"]
            + 3 * features["xpath_tautology_count"]
        ),
        "TORPEDA-SSI": (
            8 * features["ssi_directive_count"]
            + 4 * features["ssi_exec_count"]
            + 3 * features["ssi_include_count"]
        ),
        "TORPEDA-BufferOverflow": (
            3 * features["long_value_over_64_count"]
            + 5 * features["long_value_over_128_count"]
            + 8 * features["long_value_over_256_count"]
            + 3 * features["target_over_512"]
            + 3 * features["body_over_1024"]
            + 3 * features["same_char_run_over_16"]
            + 6 * features["same_char_run_over_32"]
            + max(0, features["max_token_len"] - 120) / 25.0
        ),
    }
    for family, value in scores.items():
        key = "heur_score_" + family.replace("TORPEDA-", "").lower()
        features[key] = float(value)
    features["heur_score_total"] = float(sum(scores.values()))
    features["attack_marker_total"] = float(
        features["heur_score_total"]
        + features["traversal_count"] * 2
        + features["shell_keyword_count"] * 3
        + features["shell_meta_count"]
    )

    if label_binary == 0:
        attack_family = "TORPEDA-NORMAL"
    else:
        # Prioridad para evitar que XPath/LDAP ganen por caracteres muy comunes.
        priority = [
            "TORPEDA-CRLFi",
            "TORPEDA-XSS",
            "TORPEDA-SQLi",
            "TORPEDA-SSI",
            "TORPEDA-FormatString",
            "TORPEDA-BufferOverflow",
            "TORPEDA-LDAPi",
            "TORPEDA-XPath",
        ]
        best = max(priority, key=lambda fam: (scores.get(fam, 0), -priority.index(fam)))
        if scores.get(best, 0) <= 0:
            attack_family = "TORPEDA-ANOMALOUS"
        else:
            attack_family = best
    features["attack_family_heuristic"] = attack_family

    # La etiqueta visible queda con la familia inferida cuando es anomalo.
    if label_binary == 1:
        features["label"] = attack_family
    return features


def discover_raw_files(raw_dir: Path, train_file: Optional[Path], normal_test_file: Optional[Path], anomalous_test_file: Optional[Path]) -> List[Path]:
    candidates: List[Path] = []
    for p in [train_file, normal_test_file, anomalous_test_file]:
        if p is not None:
            candidates.append(Path(p))
    if not candidates:
        candidates = [
            raw_dir / "normalTrafficTraining.txt",
            raw_dir / "normalTrafficTest.txt",
            raw_dir / "anomalousTrafficTest.txt",
        ]
    existing: List[Path] = []
    seen = set()
    for p in candidates:
        if p.exists() and p.is_file() and str(p.resolve()) not in seen:
            existing.append(p.resolve())
            seen.add(str(p.resolve()))
    if not existing:
        existing = sorted(raw_dir.glob("*.txt"))
    if not existing:
        raise FileNotFoundError(f"No encontre .txt de CSIC en {raw_dir}")
    return existing


def process_raw_csic(args: argparse.Namespace) -> Tuple[pd.DataFrame, Path]:
    raw_dir = Path(args.raw_dir)
    processed_dir = Path(args.processed_dir)
    if args.force_reprocess and processed_dir.exists():
        log(f"[PROCESS] Borrando CSIC procesado existente: {processed_dir}")
        shutil.rmtree(processed_dir)
    ensure_dir(processed_dir)

    dataset_path_parquet = processed_dir / "csic_oneclass_features.parquet"
    dataset_path_csv = processed_dir / "csic_oneclass_features.csv"

    if (not args.force_reprocess) and dataset_path_parquet.exists():
        log(f"[PROCESS] Usando dataset procesado existente: {dataset_path_parquet}")
        return pd.read_parquet(dataset_path_parquet), dataset_path_parquet
    if (not args.force_reprocess) and dataset_path_csv.exists():
        log(f"[PROCESS] Usando dataset procesado existente: {dataset_path_csv}")
        return pd.read_csv(dataset_path_csv), dataset_path_csv

    files = discover_raw_files(
        raw_dir,
        Path(args.train_file) if args.train_file else None,
        Path(args.normal_test_file) if args.normal_test_file else None,
        Path(args.anomalous_test_file) if args.anomalous_test_file else None,
    )
    log("[PROCESS] Archivos raw encontrados:")
    for f in files:
        log(f"          - {f}")

    rows: List[Dict[str, Any]] = []
    for file_path in files:
        label_binary, base_label, source_split = file_label_info(file_path)
        t0 = time.perf_counter()
        n_file = 0
        for idx, block in enumerate(iter_http_blocks(file_path)):
            if not block.strip():
                continue
            try:
                parsed = parse_http_block(block)
                if not parsed.get("method"):
                    continue
                feats = extract_features(parsed, file_path.name, idx, label_binary, base_label)
                feats["source_split"] = source_split
                rows.append(feats)
                n_file += 1
                if n_file % 10000 == 0:
                    log(f"[PROCESS] {file_path.name}: {n_file} requests...")
            except Exception as exc:
                log(f"[WARN] No pude procesar bloque {idx} de {file_path.name}: {exc!r}")
        log(f"[PROCESS] {file_path.name}: {n_file} requests en {time.perf_counter() - t0:.2f}s")

    if not rows:
        raise RuntimeError("No se extrajo ningun request. Revisa el formato del dataset raw CSIC.")

    df = pd.DataFrame(rows)
    # Orden estable de columnas: metadata primero, luego features.
    metadata = [c for c in METADATA_COLS if c in df.columns]
    others = [c for c in df.columns if c not in metadata]
    df = df[metadata + sorted(others)]

    # Guardado preferente en parquet; fallback CSV si falta pyarrow/fastparquet.
    saved_path = dataset_path_parquet
    try:
        df.to_parquet(dataset_path_parquet, index=False)
    except Exception as exc:
        log(f"[WARN] No pude guardar parquet ({exc!r}). Guardo CSV.")
        df.to_csv(dataset_path_csv, index=False)
        saved_path = dataset_path_csv

    # Manifest.
    label_counts = df["label_binary"].value_counts().to_dict()
    fam_counts = df["attack_family_heuristic"].value_counts().to_dict()
    save_json(
        processed_dir / "manifest.json",
        {
            "created_at": now_s(),
            "raw_dir": str(raw_dir),
            "files": [str(x) for x in files],
            "rows": int(len(df)),
            "columns": int(df.shape[1]),
            "label_binary_counts": label_counts,
            "attack_family_heuristic_counts": fam_counts,
            "saved_path": str(saved_path),
        },
    )
    log(f"[PROCESS] Dataset procesado: {saved_path} filas={len(df)} cols={df.shape[1]}")
    return df, saved_path


# ---------------------------------------------------------------------------
# Preprocesamiento numerico
# ---------------------------------------------------------------------------

class NumericPreprocessor:
    def __init__(self, clip_quantile_low: float = 0.001, clip_quantile_high: float = 0.999):
        self.clip_quantile_low = clip_quantile_low
        self.clip_quantile_high = clip_quantile_high
        self.input_feature_names_: List[str] = []
        self.feature_names_: List[str] = []
        self.low_: Optional[np.ndarray] = None
        self.high_: Optional[np.ndarray] = None
        self.scaler_: Optional[StandardScaler] = None
        self.dropped_constant_: List[str] = []

    def _to_matrix(self, df: pd.DataFrame, cols: Sequence[str]) -> np.ndarray:
        if not cols:
            raise ValueError("No hay columnas numericas para modelar.")
        sub = df.loc[:, list(cols)].apply(pd.to_numeric, errors="coerce")
        arr = sub.to_numpy(dtype=np.float32, copy=True)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return arr

    def fit(self, df: pd.DataFrame, feature_cols: Sequence[str]) -> "NumericPreprocessor":
        self.input_feature_names_ = list(feature_cols)
        X = self._to_matrix(df, self.input_feature_names_)
        variances = np.var(X, axis=0)
        keep_mask = variances > 1e-12
        self.feature_names_ = [c for c, keep in zip(self.input_feature_names_, keep_mask) if bool(keep)]
        self.dropped_constant_ = [c for c, keep in zip(self.input_feature_names_, keep_mask) if not bool(keep)]
        X = X[:, keep_mask]
        if X.shape[1] == 0:
            raise ValueError("Todas las features quedaron constantes en el set normal de entrenamiento.")
        self.low_ = np.quantile(X, self.clip_quantile_low, axis=0).astype(np.float32)
        self.high_ = np.quantile(X, self.clip_quantile_high, axis=0).astype(np.float32)
        same = self.high_ <= self.low_
        self.high_[same] = self.low_[same] + 1.0
        X = np.clip(X, self.low_, self.high_)
        self.scaler_ = StandardScaler()
        self.scaler_.fit(X)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self.low_ is None or self.high_ is None or self.scaler_ is None:
            raise RuntimeError("Preprocessor no ajustado.")
        X = self._to_matrix(df, self.feature_names_)
        X = np.clip(X, self.low_, self.high_)
        X = self.scaler_.transform(X).astype(np.float32, copy=False)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return X


def get_numeric_feature_columns(df: pd.DataFrame) -> List[str]:
    exclude = set(METADATA_COLS) | {
        "label_binary",
        "label",
        "attack_family_heuristic",
        "source_index",  # metadata, no feature
    }
    cols: List[str] = []
    for col in df.columns:
        if col in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return sorted(cols)


# ---------------------------------------------------------------------------
# Detectores one-class
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class DetectorResult:
    backend: str
    params: Dict[str, Any]
    detector: Any
    threshold: float
    val_metrics: Dict[str, Any]
    fit_sec: float
    score_sec: float


class SklearnIFDetector:
    backend = "sklearn_iforest"

    def __init__(self, params: Dict[str, Any], n_jobs: int = -1):
        params = dict(params)
        params.setdefault("random_state", 42)
        params.setdefault("contamination", "auto")
        params.setdefault("n_jobs", n_jobs)
        self.params = params
        self.model = IsolationForest(**params)

    def fit(self, X: np.ndarray) -> "SklearnIFDetector":
        self.model.fit(X)
        return self

    def score(self, X: np.ndarray, batch_size: int = 262144) -> np.ndarray:
        # score_samples: mas alto = mas normal. Lo invertimos.
        if len(X) <= batch_size:
            return (-self.model.score_samples(X)).astype(np.float64)
        chunks = []
        for i in range(0, len(X), batch_size):
            chunks.append(-self.model.score_samples(X[i : i + batch_size]))
        return np.concatenate(chunks).astype(np.float64)


def try_import_cuml() -> Tuple[bool, Optional[Any], Optional[Any], str]:
    try:
        from cuml.ensemble import IsolationForest as CumlIsolationForest  # type: ignore
        try:
            import cupy as cp  # type: ignore
        except Exception:
            cp = None
        return True, CumlIsolationForest, cp, "OK"
    except Exception as exc:
        return False, None, None, repr(exc)


def gpu_array_to_numpy(x: Any) -> np.ndarray:
    # cupy ndarray
    try:
        import cupy as cp  # type: ignore
        if isinstance(x, cp.ndarray):
            return cp.asnumpy(x)
    except Exception:
        pass
    # cudf/cupy-like
    if hasattr(x, "to_numpy"):
        try:
            return np.asarray(x.to_numpy())
        except Exception:
            pass
    if hasattr(x, "values_host"):
        try:
            return np.asarray(x.values_host)
        except Exception:
            pass
    return np.asarray(x)


class CumlIFDetector:
    backend = "cuml_iforest_gpu"

    def __init__(self, params: Dict[str, Any], n_jobs: int = -1):
        ok, CumlIsolationForest, cp, msg = try_import_cuml()
        if not ok or CumlIsolationForest is None:
            raise ImportError(f"cuML IsolationForest no disponible: {msg}")
        self.CumlIsolationForest = CumlIsolationForest
        self.cp = cp
        self.params = dict(params)
        self.params.setdefault("random_state", 42)
        self.params.setdefault("contamination", "auto")
        self.model = self._make_model(self.params)

    def _make_model(self, params: Dict[str, Any]) -> Any:
        # cuML cambia argumentos entre versiones. Probamos de mayor a menor.
        attempts = []
        base = dict(params)
        base.pop("n_jobs", None)
        base.pop("max_features", None)  # muchas versiones cuML no lo soportan.
        attempts.append(base)
        no_contam = dict(base)
        no_contam.pop("contamination", None)
        attempts.append(no_contam)
        minimal = {
            k: v
            for k, v in base.items()
            if k in {"n_estimators", "max_samples", "random_state", "max_depth", "bootstrap", "verbose"}
        }
        attempts.append(minimal)
        last_exc: Optional[BaseException] = None
        for kwargs in attempts:
            try:
                return self.CumlIsolationForest(**kwargs)
            except Exception as exc:
                last_exc = exc
        raise RuntimeError(f"No pude instanciar cuML IsolationForest con params={params}: {last_exc!r}")

    def _to_gpu(self, X: np.ndarray) -> Any:
        if self.cp is not None:
            return self.cp.asarray(X, dtype=self.cp.float32)
        return X.astype(np.float32, copy=False)

    def fit(self, X: np.ndarray) -> "CumlIFDetector":
        self.model.fit(self._to_gpu(X))
        return self

    def score(self, X: np.ndarray, batch_size: int = 262144) -> np.ndarray:
        scores: List[np.ndarray] = []
        for i in range(0, len(X), batch_size):
            Xg = self._to_gpu(X[i : i + batch_size])
            if hasattr(self.model, "score_samples"):
                raw = self.model.score_samples(Xg)
            elif hasattr(self.model, "decision_function"):
                raw = self.model.decision_function(Xg)
            else:
                raise RuntimeError("El modelo cuML no expone score_samples ni decision_function.")
            arr = gpu_array_to_numpy(raw).reshape(-1)
            scores.append(-arr.astype(np.float64))
        try:
            if self.cp is not None:
                self.cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass
        return np.concatenate(scores) if scores else np.array([], dtype=np.float64)


class TorchAutoencoderDetector:
    backend = "torch_autoencoder_gpu"

    def __init__(self, params: Dict[str, Any], n_jobs: int = -1):
        try:
            import torch  # type: ignore
            import torch.nn as nn  # type: ignore
        except Exception as exc:
            raise ImportError(f"PyTorch no disponible: {exc!r}") from exc
        if not torch.cuda.is_available() and not params.get("allow_cpu", False):
            raise RuntimeError("PyTorch esta instalado pero CUDA no esta disponible.")
        self.torch = torch
        self.nn = nn
        self.params = dict(params)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[Any] = None
        self.n_features: Optional[int] = None

    def _build(self, n_features: int) -> Any:
        nn = self.nn
        latent = int(self.params.get("latent_dim", max(4, min(32, n_features // 4))))
        hidden1 = int(self.params.get("hidden1", max(16, min(256, n_features * 2))))
        hidden2 = int(self.params.get("hidden2", max(8, min(128, n_features))))
        latent = max(2, min(latent, hidden2))
        return nn.Sequential(
            nn.Linear(n_features, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, latent),
            nn.ReLU(),
            nn.Linear(latent, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, n_features),
        ).to(self.device)

    def fit(self, X: np.ndarray) -> "TorchAutoencoderDetector":
        torch = self.torch
        self.n_features = int(X.shape[1])
        self.model = self._build(self.n_features)
        self.model.train()
        epochs = int(self.params.get("epochs", 25))
        batch_size = int(self.params.get("batch_size", 4096))
        lr = float(self.params.get("lr", 1e-3))
        weight_decay = float(self.params.get("weight_decay", 1e-5))
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = self.nn.MSELoss()
        X_np = X.astype(np.float32, copy=False)
        n = len(X_np)
        rng = np.random.default_rng(int(self.params.get("random_state", 42)))
        best_loss = float("inf")
        patience = int(self.params.get("patience", 5))
        stale = 0
        for epoch in range(epochs):
            order = rng.permutation(n)
            total = 0.0
            seen = 0
            for start in range(0, n, batch_size):
                idx = order[start : start + batch_size]
                xb = torch.as_tensor(X_np[idx], dtype=torch.float32, device=self.device)
                opt.zero_grad(set_to_none=True)
                recon = self.model(xb)
                loss = loss_fn(recon, xb)
                loss.backward()
                opt.step()
                total += float(loss.detach().cpu()) * len(idx)
                seen += len(idx)
            avg = total / max(seen, 1)
            if avg < best_loss * 0.999:
                best_loss = avg
                stale = 0
            else:
                stale += 1
            if stale >= patience:
                break
        return self

    def score(self, X: np.ndarray, batch_size: int = 262144) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Autoencoder no entrenado.")
        torch = self.torch
        self.model.eval()
        scores: List[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(X), batch_size):
                xb = torch.as_tensor(X[start : start + batch_size], dtype=torch.float32, device=self.device)
                recon = self.model(xb)
                err = torch.mean((recon - xb) ** 2, dim=1)
                scores.append(err.detach().cpu().numpy().astype(np.float64))
        return np.concatenate(scores) if scores else np.array([], dtype=np.float64)

    def save_torch(self, path: Path) -> None:
        if self.model is None:
            raise RuntimeError("Autoencoder no entrenado.")
        ensure_dir(path.parent)
        self.torch.save(
            {
                "state_dict": self.model.state_dict(),
                "params": self.params,
                "n_features": self.n_features,
            },
            path,
        )


# ---------------------------------------------------------------------------
# Metricas, threshold y tuning
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> Dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=np.float64)
    y_pred = (scores >= threshold).astype(int)
    labels = [0, 1]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0
    out: Dict[str, Any] = {
        "n": int(len(y_true)),
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "accuracy": float(accuracy_score(y_true, y_pred)) if len(y_true) else 0.0,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else None,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_true)) > 1 else 0.0,
        "fpr": safe_ratio(fp, fp + tn),
        "fnr": safe_ratio(fn, fn + tp),
        "predicted_anomaly_rate": float(np.mean(y_pred)) if len(y_pred) else 0.0,
        "score_mean": float(np.mean(scores)) if len(scores) else 0.0,
        "score_std": float(np.std(scores)) if len(scores) else 0.0,
    }
    if len(np.unique(y_true)) > 1:
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, scores))
        except Exception:
            out["roc_auc"] = None
        try:
            out["pr_auc"] = float(average_precision_score(y_true, scores))
        except Exception:
            out["pr_auc"] = None
    else:
        out["roc_auc"] = None
        out["pr_auc"] = None
    return out


def metric_value(metrics: Dict[str, Any], metric: str) -> float:
    v = metrics.get(metric)
    if v is None:
        return -float("inf")
    return float(v)


def find_best_threshold(y_true: np.ndarray, scores: np.ndarray, metric: str = "mcc", n_grid: int = 1200) -> Tuple[float, Dict[str, Any]]:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=np.float64)
    if len(scores) == 0:
        return 0.0, compute_metrics(y_true, scores, 0.0)
    finite_scores = scores[np.isfinite(scores)]
    if len(finite_scores) == 0:
        return 0.0, compute_metrics(y_true, scores, 0.0)

    if len(finite_scores) <= n_grid:
        candidates = np.unique(finite_scores)
    else:
        qs = np.linspace(0.001, 0.999, n_grid)
        candidates = np.unique(np.quantile(finite_scores, qs))
    # Agregar thresholds de percentiles altos de normales para cuidar FPR.
    normal_scores = finite_scores[y_true[np.isfinite(scores)] == 0] if len(y_true) == len(scores) else finite_scores
    if len(normal_scores):
        extra_q = np.array([0.50, 0.75, 0.85, 0.90, 0.92, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999])
        candidates = np.unique(np.concatenate([candidates, np.quantile(normal_scores, extra_q)]))
    candidates = np.unique(np.concatenate([candidates, [float(np.min(finite_scores)) - 1e-9, float(np.max(finite_scores)) + 1e-9]]))

    best_thr = float(candidates[0])
    best_metrics = compute_metrics(y_true, scores, best_thr)
    best_tuple = (
        metric_value(best_metrics, metric),
        best_metrics.get("f1", 0.0),
        best_metrics.get("balanced_accuracy") or 0.0,
        -(best_metrics.get("fpr") or 0.0),
    )
    for thr in candidates:
        m = compute_metrics(y_true, scores, float(thr))
        cur_tuple = (
            metric_value(m, metric),
            m.get("f1", 0.0),
            m.get("balanced_accuracy") or 0.0,
            -(m.get("fpr") or 0.0),
        )
        if cur_tuple > best_tuple:
            best_tuple = cur_tuple
            best_thr = float(thr)
            best_metrics = m
    best_metrics["threshold_metric"] = metric
    return best_thr, best_metrics


def make_if_configs(args: argparse.Namespace, n_train: int, backend: str) -> List[Dict[str, Any]]:
    n_estimators_grid = parse_grid(args.n_estimators_grid, int)
    max_samples_grid = parse_grid(args.max_samples_grid, parse_int_or_auto)
    max_features_grid = parse_grid(args.max_features_grid, float)
    configs = []
    for n_estimators, max_samples, max_features in itertools.product(
        n_estimators_grid, max_samples_grid, max_features_grid
    ):
        if isinstance(max_samples, int):
            ms = max(2, min(max_samples, n_train))
        elif max_samples == "auto":
            ms = "auto" if backend == "sklearn_iforest" else max(2, min(256, n_train))
        else:
            ms = max_samples
        cfg = {
            "n_estimators": int(n_estimators),
            "max_samples": ms,
            "contamination": "auto",
            "random_state": int(args.seed),
            "bootstrap": False,
        }
        if backend == "sklearn_iforest":
            cfg["max_features"] = float(max_features)
        configs.append(cfg)

    rng = random.Random(args.seed)
    rng.shuffle(configs)
    if args.tune_trials and args.tune_trials > 0:
        configs = configs[: int(args.tune_trials)]
    return configs


def make_ae_configs(args: argparse.Namespace, n_features: int) -> List[Dict[str, Any]]:
    # Pocos configs para que el pipeline siga siendo rapido.
    base = {
        "epochs": int(args.ae_epochs),
        "batch_size": int(args.ae_batch_size),
        "lr": float(args.ae_lr),
        "weight_decay": 1e-5,
        "patience": 4,
        "random_state": int(args.seed),
    }
    latent_options = sorted(set([max(4, min(16, n_features // 6)), max(8, min(32, n_features // 4))]))
    configs = []
    for latent in latent_options:
        cfg = dict(base)
        cfg["latent_dim"] = int(latent)
        cfg["hidden1"] = int(max(32, min(256, n_features * 2)))
        cfg["hidden2"] = int(max(16, min(128, n_features)))
        configs.append(cfg)
    return configs[: max(1, int(args.ae_trials))]


def sample_df(df: pd.DataFrame, max_rows: int, seed: int, stratify_col: Optional[str] = None) -> pd.DataFrame:
    if max_rows <= 0 or len(df) <= max_rows:
        return df.copy()
    if stratify_col and stratify_col in df.columns and df[stratify_col].nunique() > 1:
        parts = []
        # Mantiene anomalias en la muestra cuando son minoria.
        for _, g in df.groupby(stratify_col):
            n = max(1, int(round(max_rows * len(g) / len(df))))
            n = min(n, len(g))
            parts.append(g.sample(n=n, random_state=seed))
        out = pd.concat(parts).sample(frac=1.0, random_state=seed)
        if len(out) > max_rows:
            out = out.sample(n=max_rows, random_state=seed)
        return out
    return df.sample(n=max_rows, random_state=seed)


def fit_and_score_candidate(
    backend: str,
    params: Dict[str, Any],
    X_train: np.ndarray,
    y_val: np.ndarray,
    X_val: np.ndarray,
    args: argparse.Namespace,
) -> Optional[DetectorResult]:
    try:
        if backend == "cuml_iforest_gpu":
            detector = CumlIFDetector(params=params, n_jobs=args.n_jobs)
        elif backend == "sklearn_iforest":
            detector = SklearnIFDetector(params=params, n_jobs=args.n_jobs)
        elif backend == "torch_autoencoder_gpu":
            detector = TorchAutoencoderDetector(params=params, n_jobs=args.n_jobs)
        else:
            raise ValueError(f"Backend no soportado: {backend}")
        t0 = time.perf_counter()
        detector.fit(X_train)
        fit_sec = time.perf_counter() - t0
        t1 = time.perf_counter()
        val_scores = detector.score(X_val, batch_size=int(args.score_batch_size))
        score_sec = time.perf_counter() - t1
        threshold, val_metrics = find_best_threshold(y_val, val_scores, metric=args.threshold_metric)
        val_metrics.update({"backend": backend, "params": params, "fit_sec": fit_sec, "score_sec": score_sec})
        return DetectorResult(backend, params, detector, threshold, val_metrics, fit_sec, score_sec)
    except Exception as exc:
        log(f"[TUNE][WARN] Fallo candidato {backend} params={params}: {exc!r}")
        if args.debug:
            traceback.print_exc()
        return None


def tune_model(
    X_train_full: np.ndarray,
    y_val: np.ndarray,
    X_val: np.ndarray,
    args: argparse.Namespace,
    results_dir: Path,
) -> DetectorResult:
    n_train = X_train_full.shape[0]
    train_idx = np.arange(n_train)
    if args.tune_max_train_rows > 0 and n_train > args.tune_max_train_rows:
        rng = np.random.default_rng(args.seed)
        train_idx = rng.choice(train_idx, size=args.tune_max_train_rows, replace=False)
    X_train_tune = X_train_full[train_idx]

    candidates: List[Tuple[str, Dict[str, Any]]] = []
    cuml_ok, _, _, cuml_msg = try_import_cuml()

    want_gpu = args.gpu in {"auto", "force"}
    if args.model in {"auto", "gpu_iforest", "cuml_iforest"} and want_gpu:
        if cuml_ok:
            for cfg in make_if_configs(args, len(X_train_tune), backend="cuml_iforest_gpu"):
                candidates.append(("cuml_iforest_gpu", cfg))
        elif args.gpu == "force" or args.model in {"gpu_iforest", "cuml_iforest"}:
            raise RuntimeError(f"Pediste GPU/cuML pero no esta disponible: {cuml_msg}")
        else:
            log(f"[GPU][WARN] cuML no disponible; fallback CPU/sklearn. Detalle: {cuml_msg}")

    if args.model in {"auto", "autoencoder", "torch_autoencoder"} and args.enable_autoencoder and want_gpu:
        try:
            import torch  # type: ignore
            if torch.cuda.is_available():
                for cfg in make_ae_configs(args, X_train_full.shape[1]):
                    candidates.append(("torch_autoencoder_gpu", cfg))
                log(f"[GPU] PyTorch CUDA disponible: {torch.cuda.get_device_name(0)}")
            else:
                log("[GPU][WARN] PyTorch instalado pero CUDA no disponible; no uso autoencoder GPU.")
        except Exception as exc:
            log(f"[GPU][WARN] PyTorch CUDA no disponible para autoencoder: {exc!r}")

    if args.model in {"auto", "iforest", "sklearn_iforest"}:
        # Si hay cuML, sklearn queda como respaldo barato solo si se pide comparar CPU o no hay GPU.
        if (not candidates) or args.compare_cpu:
            for cfg in make_if_configs(args, len(X_train_tune), backend="sklearn_iforest"):
                candidates.append(("sklearn_iforest", cfg))

    if not candidates:
        # Ultimo fallback.
        for cfg in make_if_configs(args, len(X_train_tune), backend="sklearn_iforest"):
            candidates.append(("sklearn_iforest", cfg))

    log(f"[TUNE] Candidatos a entrenar: {len(candidates)}")
    results: List[DetectorResult] = []
    rows: List[Dict[str, Any]] = []
    for i, (backend, params) in enumerate(candidates, start=1):
        log(f"[TUNE] {i}/{len(candidates)} backend={backend} params={params}")
        res = fit_and_score_candidate(backend, params, X_train_tune, y_val, X_val, args)
        if res is None:
            continue
        results.append(res)
        rows.append(dict(res.val_metrics))
        log(
            f"[TUNE] -> {args.threshold_metric}={metric_value(res.val_metrics, args.threshold_metric):.5f} "
            f"f1={res.val_metrics.get('f1'):.5f} recall={res.val_metrics.get('recall'):.5f} "
            f"fpr={res.val_metrics.get('fpr'):.5f} fit={res.fit_sec:.2f}s score={res.score_sec:.2f}s"
        )
        pd.DataFrame(rows).to_csv(results_dir / "search.csv", index=False)

    if not results:
        raise RuntimeError("Todos los candidatos fallaron; revisa dependencias GPU/CPU y datos.")

    def sort_key(r: DetectorResult) -> Tuple[float, float, float, float]:
        m = r.val_metrics
        return (
            metric_value(m, args.threshold_metric),
            m.get("f1", 0.0),
            m.get("pr_auc") or 0.0,
            -float(r.fit_sec + r.score_sec),
        )

    best = max(results, key=sort_key)
    log(f"[TUNE] Mejor backend={best.backend} params={best.params} threshold={best.threshold:.8f}")
    save_json(results_dir / "best_validation_metrics.json", best.val_metrics)
    return best


def refit_best_model(best: DetectorResult, X_train_full: np.ndarray, args: argparse.Namespace) -> Any:
    log(f"[FIT] Reentrenando mejor modelo en todo el normalTrafficTraining: {best.backend}")
    if best.backend == "cuml_iforest_gpu":
        detector = CumlIFDetector(params=best.params, n_jobs=args.n_jobs)
    elif best.backend == "sklearn_iforest":
        detector = SklearnIFDetector(params=best.params, n_jobs=args.n_jobs)
    elif best.backend == "torch_autoencoder_gpu":
        detector = TorchAutoencoderDetector(params=best.params, n_jobs=args.n_jobs)
    else:
        raise ValueError(best.backend)
    t0 = time.perf_counter()
    detector.fit(X_train_full)
    log(f"[FIT] Modelo final entrenado en {time.perf_counter() - t0:.2f}s")
    return detector


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def save_predictions(
    df_part: pd.DataFrame,
    scores: np.ndarray,
    threshold: float,
    path: Path,
) -> pd.DataFrame:
    cols = [c for c in METADATA_COLS if c in df_part.columns]
    pred = df_part.loc[:, cols].copy()
    pred["score_anomaly"] = scores
    pred["threshold"] = threshold
    pred["pred_label_binary"] = (scores >= threshold).astype(int)
    pred["pred_label"] = np.where(pred["pred_label_binary"].values == 1, "PRED-ANOMALOUS", "PRED-NORMAL")
    pred.to_csv(path, index=False)
    return pred


def family_report(pred: pd.DataFrame, path: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if "label_binary" not in pred.columns:
        return pd.DataFrame()
    normal = pred[pred["label_binary"] == 0]
    if len(normal):
        rows.append(
            {
                "family": "TORPEDA-NORMAL",
                "n": int(len(normal)),
                "detected_as_anomaly": int(normal["pred_label_binary"].sum()),
                "rate": float(normal["pred_label_binary"].mean()),
                "meaning": "false_positive_rate",
            }
        )
    anomalies = pred[pred["label_binary"] == 1]
    if len(anomalies):
        group_col = "attack_family_heuristic" if "attack_family_heuristic" in anomalies.columns else "label"
        for family, g in anomalies.groupby(group_col):
            rows.append(
                {
                    "family": family,
                    "n": int(len(g)),
                    "detected_as_anomaly": int(g["pred_label_binary"].sum()),
                    "rate": float(g["pred_label_binary"].mean()),
                    "meaning": "recall_by_heuristic_family",
                }
            )
    out = pd.DataFrame(rows).sort_values(["meaning", "family"])
    out.to_csv(path, index=False)
    return out


def permutation_importance(
    detector: Any,
    preprocessor: NumericPreprocessor,
    df_eval: pd.DataFrame,
    threshold: float,
    args: argparse.Namespace,
    results_dir: Path,
) -> pd.DataFrame:
    if args.skip_feature_importance:
        log("[FI] Saltando feature importance por --skip-feature-importance")
        return pd.DataFrame()

    df_imp = sample_df(df_eval, args.fi_max_rows, args.seed, stratify_col="label_binary")
    y = df_imp["label_binary"].to_numpy(dtype=int)
    X = preprocessor.transform(df_imp)
    base_scores = detector.score(X, batch_size=int(args.score_batch_size))
    base_metrics = compute_metrics(y, base_scores, threshold)
    baseline = metric_value(base_metrics, args.fi_metric)
    log(f"[FI] Baseline {args.fi_metric}={baseline:.6f} con n={len(df_imp)} rows y {X.shape[1]} features")

    rng = np.random.default_rng(args.seed)
    rows: List[Dict[str, Any]] = []
    repeats = max(1, int(args.permutation_repeats))
    for j, name in enumerate(preprocessor.feature_names_):
        vals = []
        for _ in range(repeats):
            Xp = X.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            scores = detector.score(Xp, batch_size=int(args.score_batch_size))
            m = compute_metrics(y, scores, threshold)
            vals.append(metric_value(m, args.fi_metric))
        perm_mean = float(np.mean(vals))
        perm_std = float(np.std(vals))
        rows.append(
            {
                "feature": name,
                "importance_drop": float(baseline - perm_mean),
                "baseline_metric": float(baseline),
                "permuted_metric_mean": perm_mean,
                "permuted_metric_std": perm_std,
                "metric": args.fi_metric,
                "repeats": repeats,
            }
        )
        if (j + 1) % 25 == 0:
            log(f"[FI] {j + 1}/{len(preprocessor.feature_names_)} features...")

    fi = pd.DataFrame(rows).sort_values("importance_drop", ascending=False).reset_index(drop=True)
    fi.insert(0, "rank", np.arange(1, len(fi) + 1))
    fi.to_csv(results_dir / "feature_importance_permutation.csv", index=False)
    save_json(results_dir / "feature_importance_baseline_metrics.json", base_metrics)

    try:
        import matplotlib.pyplot as plt

        top = fi.head(int(args.fi_top_k)).iloc[::-1]
        plt.figure(figsize=(12, max(6, 0.30 * len(top))))
        plt.barh(top["feature"], top["importance_drop"])
        plt.xlabel(f"Caida de {args.fi_metric} al permutar")
        plt.title("Feature importance por permutacion")
        plt.tight_layout()
        plt.savefig(results_dir / "feature_importance_top.png", dpi=160)
        plt.close()
    except Exception as exc:
        log(f"[FI][WARN] No pude guardar grafico de feature importance: {exc!r}")

    log(f"[FI] Guardado: {results_dir / 'feature_importance_permutation.csv'}")
    return fi


def save_model_artifacts(
    detector: Any,
    preprocessor: NumericPreprocessor,
    threshold: float,
    best: DetectorResult,
    args: argparse.Namespace,
    results_dir: Path,
) -> None:
    meta = {
        "backend": best.backend,
        "params": best.params,
        "threshold": threshold,
        "feature_names": preprocessor.feature_names_,
        "input_feature_names": preprocessor.input_feature_names_,
        "dropped_constant_features": preprocessor.dropped_constant_,
        "created_at": now_s(),
    }
    save_json(results_dir / "model_meta.json", meta)

    if isinstance(detector, TorchAutoencoderDetector):
        torch_path = results_dir / "torch_autoencoder.pt"
        detector.save_torch(torch_path)
        artifact = {
            "backend": best.backend,
            "threshold": threshold,
            "preprocessor": preprocessor,
            "torch_model_path": str(torch_path),
            "model_meta": meta,
        }
    else:
        artifact = {
            "backend": best.backend,
            "threshold": threshold,
            "preprocessor": preprocessor,
            "model": getattr(detector, "model", detector),
            "model_meta": meta,
        }

    try:
        joblib.dump(artifact, results_dir / "model.joblib")
        log(f"[SAVE] Modelo guardado: {results_dir / 'model.joblib'}")
    except Exception as exc:
        log(f"[SAVE][WARN] No pude serializar model.joblib: {exc!r}")
        try:
            joblib.dump({"preprocessor": preprocessor, "model_meta": meta}, results_dir / "preprocessor_and_meta.joblib")
        except Exception as exc2:
            log(f"[SAVE][WARN] Tampoco pude guardar preprocessor_and_meta.joblib: {exc2!r}")


def write_summary(
    results_dir: Path,
    dataset_path: Path,
    processed_dir: Path,
    best: DetectorResult,
    test_metrics: Dict[str, Any],
    official_metrics: Dict[str, Any],
    fi: pd.DataFrame,
) -> None:
    top_features = []
    if fi is not None and not fi.empty:
        top_features = fi.head(15)[["feature", "importance_drop"]].to_dict(orient="records")
    lines = [
        "# CSIC one-class WAF - resumen",
        "",
        f"Fecha: {now_s()}",
        f"Dataset procesado: `{dataset_path}`",
        f"Directorio procesado: `{processed_dir}`",
        f"Directorio resultados: `{results_dir}`",
        "",
        "## Modelo elegido",
        f"Backend: `{best.backend}`",
        f"Params: `{best.params}`",
        f"Threshold: `{best.threshold}`",
        "",
        "## Test holdout",
        f"MCC: `{test_metrics.get('mcc')}`  F1: `{test_metrics.get('f1')}`  Recall: `{test_metrics.get('recall')}`  FPR: `{test_metrics.get('fpr')}`  ROC-AUC: `{test_metrics.get('roc_auc')}`  PR-AUC: `{test_metrics.get('pr_auc')}`",
        "",
        "## Test oficial completo",
        f"MCC: `{official_metrics.get('mcc')}`  F1: `{official_metrics.get('f1')}`  Recall: `{official_metrics.get('recall')}`  FPR: `{official_metrics.get('fpr')}`  ROC-AUC: `{official_metrics.get('roc_auc')}`  PR-AUC: `{official_metrics.get('pr_auc')}`",
        "",
        "## Top features por permutacion",
    ]
    if top_features:
        for item in top_features:
            lines.append(f"- {item['feature']}: {item['importance_drop']:.6f}")
    else:
        lines.append("- No calculado.")
    (results_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Split de datos
# ---------------------------------------------------------------------------

def build_splits(df: pd.DataFrame, args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if "label_binary" not in df.columns:
        raise ValueError("El dataframe no tiene label_binary.")
    train_norm = df[(df["label_binary"] == 0) & (df.get("source_split", "") == "train_normal")].copy()
    if train_norm.empty:
        train_norm = df[df["label_binary"] == 0].copy()
    if train_norm.empty:
        raise ValueError("No hay trafico normal para entrenar one-class.")

    official_eval = df[df.get("source_split", "") == "official_test"].copy()
    if official_eval.empty or official_eval["label_binary"].nunique() < 2:
        # Fallback si no se detectaron nombres oficiales.
        official_eval = df.copy()

    if official_eval["label_binary"].nunique() < 2:
        raise ValueError("Para tunear/evaluar hacen falta normales y anomalos etiquetados.")

    test_size = float(args.final_test_size)
    if test_size <= 0.0 or len(official_eval) < 10:
        val_df = official_eval.copy()
        test_df = official_eval.copy()
    else:
        val_df, test_df = train_test_split(
            official_eval,
            test_size=test_size,
            random_state=int(args.seed),
            stratify=official_eval["label_binary"],
        )
    return train_norm.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True), official_eval.reset_index(drop=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    default_project = Path("/home_data/aroman/TFG/waf-ml-starter")
    p = argparse.ArgumentParser(
        description="CSIC raw -> features -> one-class GPU/CPU optimized anomaly detector",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--raw-dir", default=str(default_project / "data/raw/csic"), help="Directorio raw CSIC")
    p.add_argument("--train-file", default=None, help="normalTrafficTraining.txt")
    p.add_argument("--normal-test-file", default=None, help="normalTrafficTest.txt")
    p.add_argument("--anomalous-test-file", default=None, help="anomalousTrafficTest.txt")
    p.add_argument(
        "--processed-dir",
        default=str(default_project / "data/processed/csic_oneclass_gpu_optimo"),
        help="Directorio que se borra/recrea para el dataset procesado",
    )
    p.add_argument(
        "--results-dir",
        default=str(default_project / "resultsOptimo/csic/oneclass_gpu_optimo"),
        help="Directorio de resultados",
    )
    p.add_argument("--force-reprocess", action="store_true", help="Borra y recrea processed-dir antes de procesar")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--gpu", choices=["auto", "force", "off"], default="auto")
    p.add_argument(
        "--model",
        choices=["auto", "gpu_iforest", "cuml_iforest", "iforest", "sklearn_iforest", "autoencoder", "torch_autoencoder"],
        default="auto",
        help="Modelo one-class. auto prioriza cuML IF GPU; fallback sklearn IF.",
    )
    p.add_argument("--compare-cpu", action="store_true", help="Tambien compara sklearn IF aunque haya GPU")
    p.add_argument("--enable-autoencoder", action="store_true", help="Prueba autoencoder PyTorch CUDA como candidato adicional")
    p.add_argument("--ae-trials", type=int, default=2)
    p.add_argument("--ae-epochs", type=int, default=25)
    p.add_argument("--ae-batch-size", type=int, default=4096)
    p.add_argument("--ae-lr", type=float, default=1e-3)
    p.add_argument("--tune-trials", type=int, default=18, help="Maximo de configs IF a probar")
    p.add_argument("--tune-max-train-rows", type=int, default=80000, help="Muestra normal max para tuning; 0=todas")
    p.add_argument("--final-max-train-rows", type=int, default=0, help="Muestra normal max para fit final; 0=todas")
    p.add_argument("--n-estimators-grid", default="128,256,512")
    p.add_argument("--max-samples-grid", default="256,512,1024,2048")
    p.add_argument("--max-features-grid", default="0.7,0.9,1.0")
    p.add_argument("--threshold-metric", choices=["mcc", "f1", "balanced_accuracy"], default="mcc")
    p.add_argument("--final-test-size", type=float, default=0.50, help="Fraccion del test oficial reservada como test holdout final")
    p.add_argument("--score-batch-size", type=int, default=262144)
    p.add_argument("--skip-feature-importance", action="store_true")
    p.add_argument("--fi-max-rows", type=int, default=15000)
    p.add_argument("--fi-metric", choices=["mcc", "f1", "balanced_accuracy"], default="mcc")
    p.add_argument("--permutation-repeats", type=int, default=2)
    p.add_argument("--fi-top-k", type=int, default=35)
    p.add_argument("--debug", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    set_all_seeds(args.seed)

    # Ajuste de threads si Slurm no lo hizo.
    if args.n_jobs and args.n_jobs > 0:
        os.environ.setdefault("OMP_NUM_THREADS", str(args.n_jobs))
        os.environ.setdefault("MKL_NUM_THREADS", str(args.n_jobs))
        os.environ.setdefault("OPENBLAS_NUM_THREADS", str(args.n_jobs))
        os.environ.setdefault("NUMEXPR_NUM_THREADS", str(args.n_jobs))

    results_dir = Path(args.results_dir)
    processed_dir = Path(args.processed_dir)
    ensure_dir(results_dir)

    log(f"[START] {now_s()}")
    log(f"[ARGS] {json.dumps(to_jsonable(vars(args)), ensure_ascii=False)}")

    cuml_ok, _, _, cuml_msg = try_import_cuml()
    if args.gpu != "off":
        if cuml_ok:
            log("[GPU] RAPIDS cuML IsolationForest disponible: se intentara usar GPU.")
        else:
            log(f"[GPU][WARN] cuML no disponible: {cuml_msg}")
            log("[GPU][WARN] Si el cluster tiene RAPIDS, activa un entorno con cuml/cupy para GPU real.")
    if args.gpu == "force" and not cuml_ok and not args.enable_autoencoder:
        raise RuntimeError("--gpu force solicitado, pero cuML no esta disponible y autoencoder no esta habilitado.")

    t_global = time.perf_counter()
    df, dataset_path = process_raw_csic(args)
    manifest_path = processed_dir / "manifest.json"
    if manifest_path.exists():
        save_json(results_dir / "dataset_manifest_copy.json", json.loads(manifest_path.read_text(encoding="utf-8")))
    else:
        save_json(
            results_dir / "dataset_manifest_copy.json",
            {"created_at": now_s(), "rows": int(len(df)), "columns": int(df.shape[1]), "dataset_path": str(dataset_path)},
        )

    log("[DATA] Conteo label_binary:")
    log(str(df["label_binary"].value_counts().sort_index()))
    log("[DATA] Conteo familias heuristicas:")
    log(str(df["attack_family_heuristic"].value_counts().head(20)))

    train_norm, val_df, test_df, official_eval = build_splits(df, args)
    log(f"[SPLIT] train_norm={len(train_norm)} val={len(val_df)} test_holdout={len(test_df)} official_eval={len(official_eval)}")
    log(f"[SPLIT] val labels={val_df['label_binary'].value_counts().to_dict()} test labels={test_df['label_binary'].value_counts().to_dict()}")

    if args.final_max_train_rows > 0:
        train_norm_fit = sample_df(train_norm, args.final_max_train_rows, args.seed, stratify_col=None)
    else:
        train_norm_fit = train_norm

    feature_cols = get_numeric_feature_columns(df)
    log(f"[FEATURES] Features numericas candidatas: {len(feature_cols)}")
    save_json(results_dir / "feature_columns_input.json", feature_cols)

    pre = NumericPreprocessor()
    pre.fit(train_norm_fit, feature_cols)
    save_json(
        results_dir / "feature_columns_used.json",
        {
            "used": pre.feature_names_,
            "dropped_constant": pre.dropped_constant_,
            "n_used": len(pre.feature_names_),
            "n_dropped_constant": len(pre.dropped_constant_),
        },
    )
    log(f"[FEATURES] Features usadas tras quitar constantes: {len(pre.feature_names_)}")

    X_train = pre.transform(train_norm_fit)
    X_val = pre.transform(val_df)
    y_val = val_df["label_binary"].to_numpy(dtype=int)

    best = tune_model(X_train, y_val, X_val, args, results_dir)
    detector = refit_best_model(best, X_train, args)
    threshold = best.threshold

    # Evaluacion holdout.
    X_test = pre.transform(test_df)
    y_test = test_df["label_binary"].to_numpy(dtype=int)
    t0 = time.perf_counter()
    test_scores = detector.score(X_test, batch_size=int(args.score_batch_size))
    score_test_sec = time.perf_counter() - t0
    test_metrics = compute_metrics(y_test, test_scores, threshold)
    test_metrics.update({"backend": best.backend, "score_sec": score_test_sec, "split": "holdout_test"})
    save_json(results_dir / "metrics_test_holdout.json", test_metrics)
    pred_test = save_predictions(test_df, test_scores, threshold, results_dir / "predictions_test_holdout.csv")
    family_report(pred_test, results_dir / "attack_family_report_test_holdout.csv")
    log(
        f"[EVAL:test] MCC={test_metrics.get('mcc'):.5f} F1={test_metrics.get('f1'):.5f} "
        f"Recall={test_metrics.get('recall'):.5f} FPR={test_metrics.get('fpr'):.5f} "
        f"ROC-AUC={test_metrics.get('roc_auc')} PR-AUC={test_metrics.get('pr_auc')}"
    )

    # Evaluacion test oficial completo con el mismo threshold elegido en validation.
    X_off = pre.transform(official_eval)
    y_off = official_eval["label_binary"].to_numpy(dtype=int)
    t0 = time.perf_counter()
    off_scores = detector.score(X_off, batch_size=int(args.score_batch_size))
    score_off_sec = time.perf_counter() - t0
    off_metrics = compute_metrics(y_off, off_scores, threshold)
    off_metrics.update({"backend": best.backend, "score_sec": score_off_sec, "split": "official_full"})
    save_json(results_dir / "metrics_official_full.json", off_metrics)
    pred_off = save_predictions(official_eval, off_scores, threshold, results_dir / "predictions_official_full.csv")
    family_report(pred_off, results_dir / "attack_family_report_official_full.csv")
    log(
        f"[EVAL:official] MCC={off_metrics.get('mcc'):.5f} F1={off_metrics.get('f1'):.5f} "
        f"Recall={off_metrics.get('recall'):.5f} FPR={off_metrics.get('fpr'):.5f} "
        f"ROC-AUC={off_metrics.get('roc_auc')} PR-AUC={off_metrics.get('pr_auc')}"
    )

    # Feature importance sobre holdout final.
    fi = permutation_importance(detector, pre, test_df, threshold, args, results_dir)

    save_model_artifacts(detector, pre, threshold, best, args, results_dir)
    save_json(
        results_dir / "run_info.json",
        {
            "created_at": now_s(),
            "elapsed_sec": time.perf_counter() - t_global,
            "dataset_path": str(dataset_path),
            "processed_dir": str(processed_dir),
            "results_dir": str(results_dir),
            "best_backend": best.backend,
            "best_params": best.params,
            "threshold": threshold,
            "test_metrics": test_metrics,
            "official_metrics": off_metrics,
        },
    )
    write_summary(results_dir, dataset_path, processed_dir, best, test_metrics, off_metrics, fi)

    log(f"[DONE] Resultados en: {results_dir}")
    log("       summary.md, metrics_test_holdout.json, metrics_official_full.json, search.csv,")
    log("       feature_importance_permutation.csv, predictions_*.csv, model.joblib/model_meta.json")
    log(f"[END] elapsed={time.perf_counter() - t_global:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
