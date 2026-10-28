#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entrenamiento DEFINITIVO multietiqueta WAF-ML para Harvard/SR-BH.

- Procesa el dataset crudo Harvard/SR-BH desde CSV/TSV/GZ.
- Usa exactamente el vector fijo de 30 features numéricas intra-request del pipeline multiclase.
- Entrena un clasificador binario por etiqueta de ataque.
- Compara varias familias de modelos multietiqueta One-vs-Rest y elige la mejor en validación.
- Usa GPU cuando corresponde: XGBoost CUDA; LightGBM/CatBoost GPU si están instalados y disponibles.
- Optimiza umbrales por etiqueta con validación interna para maximizar F1 u otra métrica elegida.
- Guarda todos los artefactos bajo resultsOptimo/multietiqueta/harvard por defecto.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import html
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
RAW_BODY_COL = "request_body"
RAW_HEADERS_JSON_COL = "request_headers_json"
LABEL_COL = "label_multiclass"
DEFAULT_PROJECT_DIR = "/home_data/aroman/TFG/waf-ml-starter"
DEFAULT_RESULTS_DIRNAME = "resultsOptimo"

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


def _looks_form_like(body_text: str, content_type: str = "") -> bool:
    s = _safe_str(body_text).strip()
    if not s:
        return False
    if "application/x-www-form-urlencoded" in content_type.lower():
        return True
    if s.startswith(("{", "[", "<")):
        return False
    return "=" in s and ("&" in s or len(s.split("=", 1)[0]) <= 128)


def _save_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=_json_default)


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
                df.to_csv(fallback, index=False)
                return fallback
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
    Reutiliza processed_fixed30_multilabel + run_config si existen.
    Devuelve df_features, processed_path, target_labels, target_meta, inputs, split_info_previo, load_seconds, feature_extraction_seconds.
    """
    if not _resume_enabled(args) or bool(getattr(args, "force_reprocess", False)):
        return None
    config_path = out_dir / "run_config.json"
    if not config_path.exists():
        return None
    try:
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        processed_raw = cfg.get("processed_path") or str(out_dir / "processed_fixed30_multilabel.parquet")
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
        target_cols = [_target_col_name(lab) for lab in target_labels]
        missing = [c for c in FIXED_FEATURES + target_cols if c not in df_features.columns]
        if missing:
            raise ValueError(f"processed incompleto; faltan columnas: {missing[:10]}")
        print(f"[RESUME] Reutilizo processed fixed30 existente: {processed_path}", flush=True)
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
# Feature extraction fija
# ---------------------------------------------------------------------------

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


def _extract_features_from_tuple(row_tuple: Tuple[str, str, str, str]) -> List[float]:
    method, uri, headers_json, body = row_tuple
    d = extract_fixed_features(method, uri, _json_loads_dict(headers_json), body)
    return [float(d[f]) for f in FIXED_FEATURES]


def build_feature_frame(df: pd.DataFrame, *, workers: int = 1, chunksize: int = 256) -> Tuple[pd.DataFrame, float]:
    rows = list(zip(
        df[RAW_METHOD_COL].fillna("").astype(str).tolist(),
        df[RAW_URI_COL].fillna("").astype(str).tolist(),
        df[RAW_HEADERS_JSON_COL].fillna("{}").astype(str).tolist(),
        df[RAW_BODY_COL].fillna("").astype(str).tolist(),
    ))
    t0 = time.perf_counter()
    if workers and workers > 1 and len(rows) > 500:
        with ProcessPoolExecutor(max_workers=int(workers)) as ex:
            vals = list(tqdm(ex.map(_extract_features_from_tuple, rows, chunksize=max(1, int(chunksize))), total=len(rows), desc="Extracting fixed30 features", mininterval=5))
    else:
        vals = [_extract_features_from_tuple(r) for r in tqdm(rows, total=len(rows), desc="Extracting fixed30 features", mininterval=5)]
    secs = time.perf_counter() - t0
    feats = pd.DataFrame(vals, columns=FIXED_FEATURES).fillna(0).astype(np.float32)
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
        if schema.body_col != RAW_BODY_COL:
            df[RAW_BODY_COL] = df[schema.body_col]
        if RAW_HEADERS_JSON_COL not in df.columns:
            df[RAW_HEADERS_JSON_COL] = "{}"
        if "sample_id" not in df.columns:
            df["sample_id"] = np.arange(len(df)).astype(str)
    else:
        raise SystemExit(f"Unsupported dataset={dataset!r}")

    for col in [RAW_METHOD_COL, RAW_URI_COL, RAW_BODY_COL, RAW_HEADERS_JSON_COL, LABEL_COL]:
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
DEFAULT_MULTILABEL_MODEL_NAME = "multifamily_ovr_fixed30"


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
    if schema.body_col != RAW_BODY_COL:
        df[RAW_BODY_COL] = df[schema.body_col]
    if RAW_HEADERS_JSON_COL not in df.columns:
        df[RAW_HEADERS_JSON_COL] = "{}"
    if "sample_id" not in df.columns:
        df["sample_id"] = np.arange(len(df)).astype(str)
    for col in [RAW_METHOD_COL, RAW_URI_COL, RAW_BODY_COL, RAW_HEADERS_JSON_COL, "sample_id", "source_file"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("")
    return df.reset_index(drop=True), inputs, schema, discovered_label_cols


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
        "rows_before": int(len(work)),
        "label_mode": label_mode,
        "all_target_count": int(len(target_labels_all)),
        "kept_target_count": int(len(keep_labels)),
        "dropped_target_count": int(len(dropped_labels)),
        "min_positive_count": min_pos,
        "operations": [],
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


def split_train_valid_test(Y: np.ndarray, labels: Sequence[str], args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    idx = np.arange(Y.shape[0])
    strat, strat_name = choose_multilabel_stratify_key(Y, labels, float(args.test_size))
    train_valid_idx, test_idx = train_test_split(
        idx,
        test_size=float(args.test_size),
        random_state=int(args.seed),
        shuffle=True,
        stratify=strat,
    )
    Y_train_valid = Y[train_valid_idx]
    strat2, strat2_name = choose_multilabel_stratify_key(Y_train_valid, labels, float(args.valid_size))
    train_rel, valid_rel = train_test_split(
        np.arange(len(train_valid_idx)),
        test_size=float(args.valid_size),
        random_state=int(args.seed) + 1,
        shuffle=True,
        stratify=strat2,
    )
    train_idx = train_valid_idx[train_rel]
    valid_idx = train_valid_idx[valid_rel]
    info = {
        "test_stratify": strat_name,
        "valid_stratify": strat2_name,
        "train_rows": int(len(train_idx)),
        "valid_rows": int(len(valid_idx)),
        "test_rows": int(len(test_idx)),
    }
    for split_name, split_idx in [("train", train_idx), ("valid", valid_idx), ("test", test_idx)]:
        pos = Y[split_idx].sum(axis=0).astype(int)
        missing = [labels[i] for i, c in enumerate(pos) if c == 0]
        if missing:
            print(f"[WARN] Split {split_name} sin positivos para {len(missing)} etiquetas: {missing[:5]}{'...' if len(missing)>5 else ''}", flush=True)
        info[f"{split_name}_positive_counts"] = {labels[i]: int(pos[i]) for i in range(len(labels))}
    return train_idx, valid_idx, test_idx, info


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
        params = dict(n_estimators=450, learning_rate=0.055, max_depth=4, min_child_weight=1.0, subsample=0.9, colsample_bytree=0.9, reg_alpha=0.0, reg_lambda=1.0, max_bin=256)
    elif preset == "max":
        params = dict(n_estimators=1800, learning_rate=0.022, max_depth=6, min_child_weight=1.0, subsample=0.95, colsample_bytree=0.95, reg_alpha=0.0, reg_lambda=1.0, max_bin=512)
    else:
        params = dict(n_estimators=900, learning_rate=0.035, max_depth=5, min_child_weight=1.0, subsample=0.9, colsample_bytree=0.9, reg_alpha=0.0, reg_lambda=1.0, max_bin=256)
    params.update(
        objective="binary:logistic",
        eval_metric="logloss",
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
        try:
            import xgboost as xgb  # type: ignore
        except Exception as e:
            raise RuntimeError(f"xgboost no disponible: {e}")
        xgb_version = str(getattr(xgb, "__version__", "unknown"))
        device_order = ["cuda", "cpu"] if (prefer_gpu and _want_gpu(args)) else ["cpu"]
        last_error: Optional[Exception] = None
        for device in device_order:
            params = _xgb_params_for_preset(args, y_train, weight_mode, device, xgb_version)
            try:
                model = xgb.XGBClassifier(**params)
                fit_kwargs: Dict[str, Any] = {"verbose": False}
                if X_valid is not None and y_valid is not None and len(np.unique(y_valid)) >= 2:
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
                        if X_valid is not None and y_valid is not None and len(np.unique(y_valid)) >= 2:
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
        )
        _save_label_record_checkpoint(args, out_dir, checkpoint_phase, backend, weight_mode, j, lab, rec)
        records.append(rec)
    if resumed_labels:
        print(f"[RESUME] {desc}: reutilicé {resumed_labels}/{len(labels)} modelos binarios por etiqueta.", flush=True)
    return records


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
    grid = np.linspace(float(min_threshold), float(max_threshold), int(max(3, grid_size)))
    for j, lab in enumerate(labels):
        yt = Y_true[:, j].astype(int)
        sc = scores[:, j].astype(float)
        support = int(yt.sum())
        if support <= 0 or len(np.unique(yt)) < 2:
            rows.append({"label": lab, "threshold": 0.5, "objective": objective_l, "valid_support": support, "precision": None, "recall": None, "f1": None, "score": None})
            continue
        best = (-1.0, 0.5, 0.0, 0.0, 0.0)
        for t in grid:
            yp = (sc >= t).astype(int)
            tp = int(((yt == 1) & (yp == 1)).sum())
            fp = int(((yt == 0) & (yp == 1)).sum())
            fn = int(((yt == 1) & (yp == 0)).sum())
            prec = float(tp / (tp + fp)) if (tp + fp) else 0.0
            rec = float(tp / (tp + fn)) if (tp + fn) else 0.0
            f1 = _fbeta_from_pr(prec, rec, 1.0)
            if min_recall > 0 and rec < min_recall:
                metric = -1.0
            elif objective_l in {"f2", "fbeta2"}:
                metric = _fbeta_from_pr(prec, rec, 2.0)
            elif objective_l in {"precision"}:
                metric = prec
            elif objective_l in {"recall"}:
                metric = rec
            elif objective_l in {"youden", "balanced_accuracy"}:
                tn = int(((yt == 0) & (yp == 0)).sum())
                spec = float(tn / (tn + fp)) if (tn + fp) else 0.0
                metric = 0.5 * (rec + spec)
            else:
                metric = f1
            # Tie-break: F1 real, luego umbral más alto para bajar falsos positivos.
            key = (metric, f1, float(t))
            if key > (best[0], best[3], best[1]):
                best = (metric, float(t), prec, rec, f1)
        thresholds[j] = best[1]
        rows.append({"label": lab, "threshold": best[1], "objective": objective_l, "valid_support": support, "precision": best[2], "recall": best[3], "f1": best[4], "score": best[0]})
    return thresholds, pd.DataFrame(rows)


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
        tuples_for_batch = list(zip(
            df_meas[RAW_METHOD_COL].fillna("").astype(str).tolist(),
            df_meas[RAW_URI_COL].fillna("").astype(str).tolist(),
            df_meas[RAW_HEADERS_JSON_COL].fillna("{}").astype(str).tolist(),
            df_meas[RAW_BODY_COL].fillna("").astype(str).tolist(),
        ))
        X_batch = np.asarray([_extract_features_from_tuple(t) for t in tuples_for_batch], dtype=np.float32)
    else:
        X_batch = X_test.astype(np.float32, copy=False)

    cpu0 = time.process_time()
    t0 = time.perf_counter()
    _ = predict_score_matrix(records, X_batch)
    batch_predict_wall = time.perf_counter() - t0
    batch_predict_cpu = time.process_time() - cpu0

    tuples = list(zip(
        df_meas[RAW_METHOD_COL].fillna("").astype(str).tolist(),
        df_meas[RAW_URI_COL].fillna("").astype(str).tolist(),
        df_meas[RAW_HEADERS_JSON_COL].fillna("{}").astype(str).tolist(),
        df_meas[RAW_BODY_COL].fillna("").astype(str).tolist(),
    ))
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
    if not _resume_enabled(args) or not bool(getattr(args, "resume_skip_complete", True)):
        return None
    metrics_path = out_dir / "metrics_multilabel_multifamily_fixed30.json"
    model_path = out_dir / "model_multilabel_multifamily_fixed30.joblib"
    summary_path = out_dir / "selected_model_summary.txt"
    if not (metrics_path.exists() and model_path.exists() and summary_path.exists()):
        return None
    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
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
    print(f"[START] Harvard multietiqueta definitivo fixed30 @ {_now_iso()}", flush=True)
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
        processed_path = _save_processed(df_features, out_dir / "processed_fixed30_multilabel.parquet")

    X = df_features[FIXED_FEATURES].fillna(0).astype(np.float32).to_numpy()
    target_cols = [_target_col_name(lab) for lab in target_labels]
    Y = df_features[target_cols].fillna(0).astype(np.int8).to_numpy()

    train_idx, valid_idx, test_idx, split_info = split_train_valid_test(Y, target_labels, args)
    X_train, Y_train = X[train_idx], Y[train_idx]
    X_valid, Y_valid = X[valid_idx], Y[valid_idx]
    X_test, Y_test = X[test_idx], Y[test_idx]
    X_train_valid = X[np.concatenate([train_idx, valid_idx])]
    Y_train_valid = Y[np.concatenate([train_idx, valid_idx])]
    df_test_raw = df_features.iloc[test_idx].copy()

    _save_json({
        "created_at": _now_iso(),
        "task": "multilabel",
        "dataset": "harvard",
        "script": Path(__file__).name,
        "inputs": inputs,
        "features": FIXED_FEATURES,
        "feature_count": len(FIXED_FEATURES),
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
    }, out_dir / "run_config.json")

    t_select0 = time.perf_counter()
    best = train_select_candidate(args, target_labels, X_train, Y_train, X_valid, Y_valid, out_dir=out_dir)
    train_select_seconds = time.perf_counter() - t_select0
    pd.DataFrame(best["candidate_summary"]).to_csv(out_dir / "candidate_validation_summary.csv", index=False)
    _save_json({"candidates": best["candidate_summary"]}, out_dir / "candidate_validation_summary.json")
    best["thresholds_df"].to_csv(out_dir / "thresholds_validation.csv", index=False)

    best = refit_final_if_requested(args, best, target_labels, X_train_valid, Y_train_valid, X_valid, Y_valid, out_dir=out_dir)
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

    pred_rows = []
    for r, idx0 in enumerate(test_idx):
        true_labs = [target_labels[j] for j, v in enumerate(Y_test[r]) if int(v) == 1]
        pred_labs = [target_labels[j] for j, v in enumerate(Y_pred[r]) if int(v) == 1]
        row: Dict[str, Any] = {
            "row_index": int(idx0),
            "sample_id": str(df_features.iloc[idx0].get("sample_id", idx0)),
            "source_file": str(df_features.iloc[idx0].get("source_file", "")),
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

    model_path = out_dir / "model_multilabel_multifamily_fixed30.joblib"
    # No guardo candidatos perdedores para no inflar el .joblib.
    records_for_dump = records
    _atomic_joblib_dump({
        "task": "multilabel",
        "dataset": "harvard",
        "model_family": "one_vs_rest_binary_classifiers",
        "selected_backend": best["backend"],
        "selected_model": f"{best['backend']}_ovr__w_{best['weight_mode']}",
        "records": records_for_dump,
        "thresholds": thresholds,
        "threshold_objective": str(args.threshold_objective),
        "features": FIXED_FEATURES,
        "feature_count": len(FIXED_FEATURES),
        "target_labels": target_labels,
        "target_meta": target_meta,
        "target_columns": target_cols,
        "label_mode": str(args.label_mode),
        "config": vars(args),
        "predict_note": "Para inferencia: extraer FIXED_FEATURES con el mismo código y aplicar predict_score_matrix(records, X) >= thresholds.",
    }, model_path, compress=int(getattr(args, "checkpoint_compress", 3)))
    model_size_bytes = int(model_path.stat().st_size) if model_path.exists() else 0

    fi_paths = compute_feature_importance_multilabel(records, X_test, Y_test, thresholds, target_labels, args, out_dir)
    operational_metrics = measure_operational_metrics_multilabel(records, df_test_raw, X_test, thresholds, target_labels, args, out_dir)
    operational_metrics["model_size_bytes"] = model_size_bytes
    operational_metrics["model_size_mb"] = model_size_bytes / (1024.0 * 1024.0) if model_size_bytes else 0.0
    _save_json(operational_metrics, out_dir / "operational_metrics.json")

    run_total_seconds = time.perf_counter() - t_run0
    selected_model = f"{best['backend']}_ovr__w_{best['weight_mode']}"
    metrics_payload: Dict[str, Any] = {
        "created_at": _now_iso(),
        "task": "multilabel",
        "dataset": "harvard",
        "dataset_label_mode": str(args.label_mode),
        "feature_policy": "fixed_compact_intra_request",
        "feature_count": len(FIXED_FEATURES),
        "features_used": FIXED_FEATURES,
        "selected_model": selected_model,
        "selected_model_family": "one_vs_rest_binary_classifiers",
        "backend": best["backend"],
        "weight_mode": best["weight_mode"],
        "model_backend": describe_backend(best["backend"]),
        "gpu_requested": args.use_gpu,
        "gpu_visible": bool(gpu_visible),
        "gpu_used": bool(best.get("gpu_used_any")),
        "gpu_note": backend_gpu_note(best["backend"]),
        "labels": target_labels,
        "n_labels": int(len(target_labels)),
        "target_meta": target_meta,
        "train_rows": int(len(train_idx)),
        "valid_rows": int(len(valid_idx)),
        "test_rows": int(len(test_idx)),
        "test_size": float(args.test_size),
        "valid_size_inside_train": float(args.valid_size),
        "split_info": split_info,
        "seed": int(args.seed),
        "thresholds": {target_labels[i]: float(thresholds[i]) for i in range(len(target_labels))},
        "threshold_objective": str(args.threshold_objective),
        "candidate_selection": {
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
            "metrics": str(out_dir / "metrics_multilabel_multifamily_fixed30.json"),
            "metrics_flat": str(out_dir / "metrics_flat.csv"),
            "test_metrics": str(out_dir / "test_metrics_multilabel.json"),
            "per_label_metrics": str(out_dir / "per_label_metrics.csv"),
            "thresholds": str(out_dir / "thresholds_validation.csv"),
            "predictions": str(out_dir / "test_predictions.csv"),
            "operational_metrics": str(out_dir / "operational_metrics.json"),
            **fi_paths,
        },
    }
    _save_json(metrics_payload, out_dir / "metrics_multilabel_multifamily_fixed30.json")
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
        out_dir / "metrics_multilabel_multifamily_fixed30.json",
        out_dir / "metrics_flat.csv",
        out_dir / "selected_model_summary.txt",
        out_dir / "per_label_metrics.csv",
        out_dir / "thresholds_validation.csv",
        out_dir / "test_predictions.csv",
        out_dir / "feature_importance.csv",
        out_dir / "operational_metrics.json",
        Path(args.definitivo_dir) / "multietiqueta" / "model_comparison_multilabel_definitivo.csv",
    ]:
        if Path(p).exists():
            print(f"  {p}", flush=True)
    return metrics_payload


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="DEFINITIVO multietiqueta Harvard/SR-BH: búsqueda multifamilia OVR fixed30")
    ap.add_argument("--project-dir", default=DEFAULT_PROJECT_DIR)
    ap.add_argument("--definitivo-dir", default=None, help="Default: PROJECT_DIR/resultsOptimo")
    ap.add_argument("--inputs", nargs="*", default=None, help="Inputs Harvard/SR-BH CSV/TSV/GZ. Si no se pasa, usa --harvard-inputs.")
    ap.add_argument("--harvard-inputs", nargs="*", default=None)
    ap.add_argument("--output-dir", default=None, help="Sobrescribe resultsOptimo/multietiqueta/harvard.")

    # Dataset/preproceso Harvard.
    ap.add_argument("--sample-n", type=int, default=0, help="Head por archivo para pruebas. 0=todo.")
    ap.add_argument("--sep", default="auto")
    ap.add_argument("--method-col", default=RAW_METHOD_COL)
    ap.add_argument("--uri-col", default=RAW_URI_COL)
    ap.add_argument("--body-col", default=RAW_BODY_COL)
    ap.add_argument("--normal-col", default="000 - Normal")
    ap.add_argument("--label-cols", default="", help="CSV de columnas label. Vacío=autodetecta '^NNN - Nombre'.")
    ap.add_argument("--label-mode", default="native", choices=["native", "capec", "srbh", "optimized-family", "optimized_family", "optimized", "coarse", "coarse-family", "coarse_family", "family", "mapped", "attack-family", "attack_families"], help="native conserva CAPEC; optimized-family agrupa como el multiclase.")
    ap.add_argument("--min-positive-count", type=int, default=20, help="Descarta etiquetas con menos positivos. 20 es estable para split/validación.")
    ap.add_argument("--max-negative-positive-ratio", type=float, default=0.0, help="0=sin filtro por rareza extrema.")
    ap.add_argument("--max-normal-rows", type=int, default=0, help="0=no downsample de normales.")
    ap.add_argument("--keep-labels-regex", default="")
    ap.add_argument("--drop-labels-regex", default="")

    # Split y selección.
    ap.add_argument("--test-size", type=float, default=0.20)
    ap.add_argument("--valid-size", type=float, default=0.20, help="Fracción del train_valid para umbrales/selección.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--selection-metric", default="f1_macro", choices=["f1_macro", "f1_micro", "f1_weighted", "jaccard_macro", "jaccard_micro", "pr_auc_macro", "roc_auc_macro", "exact_match_accuracy"])
    ap.add_argument("--search-level", default="standard", choices=["none", "standard", "max"], help="none=un modo; standard/max usan candidate-weight-modes.")
    ap.add_argument("--candidate-weight-modes", default="sqrt,balanced", help="Modos separados por coma: none,sqrt,cuberoot,balanced. default evalúa sqrt y balanced.")
    ap.add_argument("--refit-full-after-thresholds", action=argparse.BooleanOptionalAction, default=True)

    # Modelo / familias candidatas.
    ap.add_argument(
        "--model-backend",
        default="auto",
        help=(
            "auto|all|compare|full o lista separada por comas. Backends: "
            "xgboost,lightgbm,catboost,histgb,extra_trees,random_forest,logistic_regression,linear_svc,sgd_logloss"
        ),
    )
    ap.add_argument("--use-gpu", default="auto", choices=["auto", "on", "off"], help="auto/on intenta GPU para XGBoost, LightGBM y CatBoost cuando hay GPU visible; fallback CPU si falla.")
    ap.add_argument("--xgb-preset", default="strong", choices=["fast", "strong", "max"])
    ap.add_argument("--xgb-early-stopping-rounds", type=int, default=60)
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
    ap.add_argument("--threshold-grid-size", type=int, default=199)
    ap.add_argument("--threshold-min", type=float, default=0.01)
    ap.add_argument("--threshold-max", type=float, default=0.99)
    ap.add_argument("--threshold-min-recall", type=float, default=0.0)
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
    ap.add_argument("--force-reprocess", action="store_true", help="Ignora processed_fixed30 previo aunque --resume esté activo; no borra checkpoints de candidatos.")
    ap.add_argument("--resume-skip-complete", action=argparse.BooleanOptionalAction, default=True, help="Si ya existen métricas finales, modelo y resumen, termina sin recomputar.")
    ap.add_argument("--clean-output-dir", action="store_true", help="Borra OUTPUT_DIR al empezar. OJO: elimina checkpoints y anula el beneficio de --resume para esa corrida.")
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args(argv)

    if args.definitivo_dir is None:
        args.definitivo_dir = str(Path(args.project_dir) / DEFAULT_RESULTS_DIRNAME)
    if not args.harvard_inputs:
        args.harvard_inputs = [os.environ.get("HARVARD_RAW_DIR", str(Path(args.project_dir) / "data" / "raw" / "harvard"))]
    if str(args.search_level).lower() == "max" and str(args.candidate_weight_modes) == "sqrt,balanced":
        args.candidate_weight_modes = "sqrt,cuberoot,balanced"
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    dep = dependency_check()
    Path(args.definitivo_dir).mkdir(parents=True, exist_ok=True)
    _save_json({"created_at": _now_iso(), "dependency_check": dep, "args": vars(args)}, Path(args.definitivo_dir) / "last_invocation_multilabel.json")
    if args.check_only:
        print("[OK] CHECK_ONLY: dependencias y argumentos OK. No entreno.", flush=True)
        return 0
    train_eval_harvard_multilabel(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
