#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
11claseResults multiclase WAF-ML para TorpEda y Harvard/SR-BH.

- Usa HistGradientBoostingClassifier como modelo multiclase principal.
- Usa una selección podada de 58 features intra-request tomada de la corrida3: se
  conservan sólo las variables con importancia por clase positiva en Torpeda o Harvard.
- Las features usan sólo método HTTP, URI, query string, headers y body de la misma request;
  no usan IP, sesión, usuario, reputación, endpoint histórico, tiempo ni contexto del aplicativo.
- Guarda TODO bajo 11claseResults/multiclase/{torpeda,harvard}.
- En Harvard/SR-BH, por defecto conserva el top 14 de combinaciones CAPEC más frecuentes
  y aplica una taxonomía operativa: CAPEC-272/CAPEC-274 -> HTTPProtocolManipulation
  y CAPEC-153/CAPEC-194 -> DataManipulation. El resto de combinaciones se ignora antes de extraer features.
- Calcula métricas predictivas, métricas operativas/performance, feature importance global,
  feature importance por clase y feature importance binaria NORMAL vs ataque/anómalo.

Nota sobre GPU:
HistGradientBoostingClassifier de scikit-learn no tiene backend CUDA. Este script detecta GPU
para registrarlo en los metadatos, pero acelera este modelo usando CPU/OpenMP/BLAS y paralelismo
en extracción de features/permutation importance cuando corresponde.
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

from sklearn.ensemble import HistGradientBoostingClassifier
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
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
DEFAULT_RESULT_DIR_NAME = "11claseResults"

COMMON_ATTACKS = ["BufferOverflow", "CRLFi", "FormatString", "LDAPi", "SQLi", "SSI", "XPath", "XSS"]

# Set candidato original de 84 features numéricas intra-request.
# De estas, el modelo final de este archivo usa sólo las 58 seleccionadas por feature importance.
BASE_30_FEATURES: List[str] = [
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

EXTENDED_54_FEATURES: List[str] = [
    # Método HTTP / verb tampering.
    "method_is_get",
    "method_is_post",
    "method_is_put",
    "method_is_delete",
    "method_is_patch",
    "method_is_options",
    "method_is_head",
    "method_is_trace",
    "method_is_connect",
    "method_is_other",
    "method_is_safe",
    "method_body_inconsistency",

    # Headers y protocolo HTTP.
    "header_count",
    "headers_total_len",
    "host_present",
    "user_agent_present",
    "content_type_present",
    "content_length_present",
    "transfer_encoding_present",
    "transfer_encoding_chunked",
    "content_length_transfer_encoding_both",
    "content_length_mismatch_abs",
    "content_length_mismatch_flag",
    "method_override_header_present",
    "header_control_char_count",
    "header_crlf_injection_count",

    # Parámetros/body/input manipulation.
    "n_body_params",
    "n_total_params",
    "empty_param_value_count",
    "max_param_name_len",
    "param_name_entropy",
    "sensitive_param_name_count",
    "json_body_flag",
    "json_key_count",
    "json_max_depth",
    "xml_body_flag",
    "xml_tag_count",
    "multipart_body_flag",

    # XPath/XML específico.
    "xpath_axis_count",
    "xpath_function_count",
    "xpath_predicate_count",
    "xpath_attribute_ref_count",
    "xpath_double_slash_count",
    "xpath_pipe_union_count",
    "xml_entity_count",

    # Scanning/path traversal genérico intra-request.
    "sensitive_path_token_count",
    "hidden_file_path_count",
    "backup_config_token_count",
    "common_scanner_ua_token_count",
    "suspicious_file_extension_count",
    "path_dot_segment_count",
    "path_repeated_slash_count",
    "path_encoded_slash_count",
    "path_null_byte_count",
]

ALL_84_CANDIDATE_FEATURES: List[str] = BASE_30_FEATURES + EXTENDED_54_FEATURES

# Selección derivada de la feature importance por clase de la corrida3 con 84 features.
# Criterio conservador usado: se conserva una feature si tuvo permutation_importance_mean > 0
# para al menos una clase, en Torpeda o Harvard. Se elimina si su máximo por clase fue <= 0
# en las 24 clases evaluadas (10 Torpeda + 14 Harvard) y también quedó en 0 en importancia global/binaria.
FIXED_FEATURES: List[str] = [
    "uri_len",
    "body_len",
    "query_len",
    "path_depth",
    "n_query_params",
    "max_param_value_len",
    "uri_encoded_ratio",
    "body_encoded_ratio",
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
    "method_is_get",
    "method_is_post",
    "method_is_options",
    "method_is_head",
    "method_is_trace",
    "method_is_other",
    "method_is_safe",
    "method_body_inconsistency",
    "header_count",
    "headers_total_len",
    "n_body_params",
    "n_total_params",
    "empty_param_value_count",
    "max_param_name_len",
    "param_name_entropy",
    "sensitive_param_name_count",
    "xpath_predicate_count",
    "xpath_attribute_ref_count",
    "xpath_double_slash_count",
    "xpath_pipe_union_count",
    "sensitive_path_token_count",
    "hidden_file_path_count",
    "backup_config_token_count",
    "common_scanner_ua_token_count",
    "suspicious_file_extension_count",
    "path_dot_segment_count",
    "path_repeated_slash_count",
    "path_encoded_slash_count",
    "path_null_byte_count",
]

FI_DROPPED_FEATURES: List[str] = [
    "invalid_percent_encoding_count",
    "method_is_put",
    "method_is_delete",
    "method_is_patch",
    "method_is_connect",
    "host_present",
    "user_agent_present",
    "content_type_present",
    "content_length_present",
    "transfer_encoding_present",
    "transfer_encoding_chunked",
    "content_length_transfer_encoding_both",
    "content_length_mismatch_abs",
    "content_length_mismatch_flag",
    "method_override_header_present",
    "header_control_char_count",
    "header_crlf_injection_count",
    "json_body_flag",
    "json_key_count",
    "json_max_depth",
    "xml_body_flag",
    "xml_tag_count",
    "multipart_body_flag",
    "xpath_axis_count",
    "xpath_function_count",
    "xml_entity_count",
]

FEATURE_POLICY_NAME = "useful58_intra_request"
FEATURE_SET_VERSION = "useful58_operational_groups_v1"

FEATURE_GROUP_BY_NAME: Dict[str, str] = {**{f: "baseline30" for f in BASE_30_FEATURES}}
FEATURE_GROUP_BY_NAME.update({
    **{f: "http_method" for f in EXTENDED_54_FEATURES[0:12]},
    **{f: "http_headers_protocol" for f in EXTENDED_54_FEATURES[12:26]},
    **{f: "body_params_structure" for f in EXTENDED_54_FEATURES[26:38]},
    **{f: "xpath_xml" for f in EXTENDED_54_FEATURES[38:45]},
    **{f: "scanning_paths" for f in EXTENDED_54_FEATURES[45:54]},
})

_HTTP_PROTO_RE = re.compile(r"^HTTP/(\d+(?:\.\d+)?)$", re.IGNORECASE)
_HEX = r"[0-9a-fA-F]"
_PCT_ENC_RE = re.compile(rf"%{_HEX}{{2}}")
_INVALID_PERCENT_RE = re.compile(rf"%(?!{_HEX}{{2}})")
_DOUBLE_ENC_RE = re.compile(rf"%25{_HEX}{{2}}", re.IGNORECASE)
_TOKEN_RE = re.compile(r"[A-Za-z0-9_%\\x\\u./:-]+")
_LABEL_COL_RE = re.compile(r"^\d{1,3}\s+-\s+.+$")
_ID_NAME_RE = re.compile(r"^(\d{1,3})\s+-\s+(.+)$")
_CANON_CAPEC_RE = re.compile(r"\bCAPEC-(\d{1,3})\b")
_XPATH_AXIS_RE = re.compile(r"\b(?:ancestor|ancestor-or-self|attribute|child|descendant|descendant-or-self|following|following-sibling|namespace|parent|preceding|preceding-sibling|self)::", re.IGNORECASE)
_XPATH_FUNCTION_RE = re.compile(r"\b(?:count|contains|starts-with|string-length|substring|position|last|name|local-name|normalize-space|not|boolean|number|string|concat|translate)\s*\(", re.IGNORECASE)
_XPATH_ATTR_RE = re.compile(r"@\s*[A-Za-z_][\w:-]*")
_XML_TAG_RE = re.compile(r"<\s*/?\s*[A-Za-z_][\w:.-]*\b")
_XML_ENTITY_RE = re.compile(r"&(?:[A-Za-z][A-Za-z0-9]+|#[0-9]+|#x[0-9A-Fa-f]+);")
_ABSOLUTE_HTTP_URI_RE = re.compile(r"^https?://", re.IGNORECASE)

SENSITIVE_PARAM_NAMES = {
    "id", "uid", "userid", "user_id", "user", "username", "account", "account_id", "role", "admin",
    "is_admin", "privilege", "permission", "price", "amount", "total", "balance", "discount", "file",
    "filename", "path", "folder", "dir", "directory", "url", "uri", "redirect", "return", "next", "target",
    "cmd", "command", "exec", "query", "q", "search", "token", "auth", "password", "passwd", "pass",
}
SENSITIVE_PATH_TOKENS = {
    ".env", ".git", ".svn", "wp-admin", "wp-login", "xmlrpc", "phpmyadmin", "pma", "admin",
    "administrator", "manager", "server-status", "cgi-bin", "setup", "install", "debug", "test", "dev",
    "backup", "backups", "config", "configuration", "conf", "db", "database", "dump", "sql", "swagger",
    "actuator", "console", "jmx-console", "hudson", "jenkins", "solr", "elasticsearch",
}
BACKUP_CONFIG_TOKENS = {
    "backup", "bak", "old", "orig", "copy", "save", "tmp", "temp", "config", "conf", "ini", "yaml", "yml",
    "properties", "env", "dump", "db", "database", "sql", "tar", "gz", "zip", "rar", "7z", "log",
}
SCANNER_UA_TOKENS = {
    "sqlmap", "nikto", "nmap", "masscan", "acunetix", "nessus", "openvas", "burp", "zap", "wpscan",
    "dirbuster", "gobuster", "ffuf", "feroxbuster", "dirsearch", "crawler", "scanner", "python-requests",
}
SUSPICIOUS_EXTENSIONS = {
    ".bak", ".old", ".orig", ".save", ".tmp", ".temp", ".swp", ".sql", ".dump", ".db", ".sqlite",
    ".zip", ".tar", ".gz", ".tgz", ".rar", ".7z", ".ini", ".conf", ".config", ".env", ".log",
}
METHOD_OVERRIDE_HEADER_NAMES = {
    "x-http-method-override", "x-method-override", "x-http-method", "x-original-http-method",
    "x-method", "x-http-verb-override",
}



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

# Top 14 de combinaciones exactas CAPEC más frecuentes en Harvard/SR-BH.
# Incluye NORMAL y dos combinaciones multilabel explícitas. Las filas que no
# coinciden exactamente con una de estas combinaciones se ignoran sólo para Harvard.
HARVARD_TOP14_COMBO_CLASSES: List[Dict[str, Any]] = [
    {
        "rank": 1,
        "capec_ids": (),
        "label": "NORMAL",
        "display": "Sin ataque / Normal",
        "expected_count": 525195,
    },
    {
        "rank": 2,
        "capec_ids": ("CAPEC-66",),
        "label": "HARVARD-CAPEC66_SQLInjection",
        "display": "66 - SQL Injection",
        "expected_count": 248093,
    },
    {
        "rank": 3,
        "capec_ids": ("CAPEC-194",),
        "label": "HARVARD-CAPEC194_FakeSourceData",
        "display": "194 - Fake the Source of Data",
        "expected_count": 55983,
    },
    {
        "rank": 4,
        "capec_ids": ("CAPEC-34",),
        "label": "HARVARD-CAPEC34_HTTPResponseSplitting",
        "display": "34 - HTTP Response Splitting",
        "expected_count": 19134,
    },
    {
        "rank": 5,
        "capec_ids": ("CAPEC-126",),
        "label": "HARVARD-CAPEC126_PathTraversal",
        "display": "126 - Path Traversal",
        "expected_count": 17595,
    },
    {
        "rank": 6,
        "capec_ids": ("CAPEC-242",),
        "label": "HARVARD-CAPEC242_CodeInjection",
        "display": "242 - Code Injection",
        "expected_count": 13793,
    },
    {
        "rank": 7,
        "capec_ids": ("CAPEC-272",),
        "label": "HARVARD-CAPEC272_ProtocolManipulation",
        "display": "272 - Protocol Manipulation",
        "expected_count": 7019,
    },
    {
        "rank": 8,
        "capec_ids": ("CAPEC-274",),
        "label": "HARVARD-CAPEC274_HTTPVerbTampering",
        "display": "274 - HTTP Verb Tampering",
        "expected_count": 4055,
    },
    {
        "rank": 9,
        "capec_ids": ("CAPEC-88",),
        "label": "HARVARD-CAPEC88_OSCommandInjection",
        "display": "88 - OS Command Injection",
        "expected_count": 3074,
    },
    {
        "rank": 10,
        "capec_ids": ("CAPEC-88", "CAPEC-126"),
        "label": "HARVARD-CAPEC88_OSCommandInjection__CAPEC126_PathTraversal",
        "display": "88 - OS Command Injection + 126 - Path Traversal",
        "expected_count": 2464,
    },
    {
        "rank": 11,
        "capec_ids": ("CAPEC-310",),
        "label": "HARVARD-CAPEC310_VulnerabilityScanning",
        "display": "310 - Scanning for Vulnerable Software",
        "expected_count": 2413,
    },
    {
        "rank": 12,
        "capec_ids": ("CAPEC-153",),
        "label": "HARVARD-CAPEC153_InputDataManipulation",
        "display": "153 - Input Data Manipulation",
        "expected_count": 1387,
    },
    {
        "rank": 13,
        "capec_ids": ("CAPEC-272", "CAPEC-274"),
        "label": "HARVARD-CAPEC272_ProtocolManipulation__CAPEC274_HTTPVerbTampering",
        "display": "272 - Protocol Manipulation + 274 - HTTP Verb Tampering",
        "expected_count": 1370,
    },
    {
        "rank": 14,
        "capec_ids": ("CAPEC-16",),
        "label": "HARVARD-CAPEC16_DictionaryPasswordAttack",
        "display": "16 - Dictionary-based Password Attack",
        "expected_count": 836,
    },
]

HARVARD_IGNORED_TOP14_LABEL = "HARVARD-IGNORED_NOT_TOP14_COMBO"
HARVARD_TOP14_COMBO_KEY_TO_SPEC: Dict[str, Dict[str, Any]] = {
    ("+".join(spec["capec_ids"]) if spec["capec_ids"] else "NORMAL"): spec
    for spec in HARVARD_TOP14_COMBO_CLASSES
}
HARVARD_TOP14_ALLOWED_LABELS = {str(spec["label"]) for spec in HARVARD_TOP14_COMBO_CLASSES}



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


# ---------------------------------------------------------------------------
# Feature extraction optimizada intra-request
# ---------------------------------------------------------------------------

def _normalize_method(method: str) -> str:
    m = re.sub(r"[^A-Za-z]", "", _safe_str(method).strip().upper())
    return m or "OTHER"


def _header_lookup(headers: Dict[str, str], name: str) -> str:
    target = name.lower().strip()
    for k, v in (headers or {}).items():
        if str(k).lower().strip() == target:
            return _safe_str(v)
    return ""


def _header_present(headers: Dict[str, str], name: str) -> bool:
    return bool(_header_lookup(headers, name)) or any(str(k).lower().strip() == name.lower().strip() for k in (headers or {}).keys())


def _headers_text(headers: Dict[str, str]) -> str:
    if not headers:
        return ""
    return "\n".join(f"{_safe_str(k)}: {_safe_str(v)}" for k, v in headers.items())


def _parse_int_header(value: str) -> Optional[int]:
    s = _safe_str(value).strip()
    if not s:
        return None
    try:
        # Si llega una lista textual de valores, quedarse con el primero numérico.
        m = re.search(r"-?\d+", s)
        if not m:
            return None
        return int(m.group(0))
    except Exception:
        return None


def _json_shape_stats(value: str) -> Tuple[int, int, int]:
    s = _safe_str(value).strip()
    if not s:
        return 0, 0, 0
    try:
        obj = json.loads(s)
    except Exception:
        return 0, 0, 0

    def walk(o: Any, depth: int = 1) -> Tuple[int, int]:
        if isinstance(o, dict):
            if not o:
                return 0, depth
            keys = len(o)
            max_depth = depth
            for v in o.values():
                k2, d2 = walk(v, depth + 1)
                keys += k2
                max_depth = max(max_depth, d2)
            return keys, max_depth
        if isinstance(o, list):
            if not o:
                return 0, depth
            keys = 0
            max_depth = depth
            for v in o:
                k2, d2 = walk(v, depth + 1)
                keys += k2
                max_depth = max(max_depth, d2)
            return keys, max_depth
        return 0, depth

    key_count, max_depth = walk(obj, 1)
    return 1, int(key_count), int(max_depth)


def _token_count_from_set(text: str, vocabulary: set[str]) -> int:
    s = _safe_str(text).lower()
    if not s:
        return 0
    return int(sum(1 for tok in vocabulary if tok and tok in s))


def _path_segments(path_text: str) -> List[str]:
    return [p for p in re.split(r"/+", _safe_str(path_text)) if p != ""]


def feature_definitions_frame() -> pd.DataFrame:
    rows = []
    for i, f in enumerate(FIXED_FEATURES, start=1):
        rows.append({
            "feature_index": i,
            "feature": f,
            "group": FEATURE_GROUP_BY_NAME.get(f, "unknown"),
            "feature_set": FEATURE_SET_VERSION,
            "intra_request_only": True,
        })
    return pd.DataFrame(rows)


def extract_fixed_features(method: str, uri: str, headers: Optional[Dict[str, str]] = None, body: Any = None) -> Dict[str, float]:
    """Devuelve las 58 features útiles en el orden FIXED_FEATURES."""

    method_norm = _normalize_method(method)
    uri_raw = _safe_str(uri)
    if isinstance(body, bytes):
        body_raw = body.decode("utf-8", errors="ignore")
        body_len = len(body)
    else:
        body_raw = _safe_str(body)
        body_len = len(body_raw.encode("utf-8", errors="ignore"))

    headers_d = headers or {}
    headers_txt = _headers_text(headers_d)
    uri_dec1, uri_dec2 = _decode_once(uri_raw), _decode_twice(uri_raw)
    body_dec1, body_dec2 = _decode_once(body_raw), _decode_twice(body_raw)
    headers_dec1, headers_dec2 = _decode_once(headers_txt), _decode_twice(headers_txt)
    raw_combined = f"{uri_raw}\n{body_raw}\n{headers_txt}"
    analysis_text = f"{raw_combined}\n{uri_dec1}\n{body_dec1}\n{headers_dec1}\n{uri_dec2}\n{body_dec2}\n{headers_dec2}"

    parts = urlsplit(uri_raw)
    path = parts.path or ""
    query = parts.query or (uri_raw.split("?", 1)[1] if "?" in uri_raw else "")
    path_dec = _decode_twice(path)
    query_dec = _decode_twice(query)
    q_params = _parse_qs(query)
    content_type = _content_type(headers_d)
    b_params = _parse_qs(body_raw) if _looks_form_like(body_raw, content_type) else []
    all_params = q_params + b_params
    values = [v for _, v in all_params]
    names = [k for k, _ in all_params]

    uri_len = len(uri_raw)
    body_text_len = len(body_raw)
    tokens = _TOKEN_RE.findall(analysis_text)

    # Método HTTP.
    known_methods = {"GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD", "TRACE", "CONNECT"}
    safe_methods = {"GET", "HEAD", "OPTIONS", "TRACE"}
    has_body = body_len > 0
    method_body_inconsistency = int((method_norm in safe_methods and has_body) or (method_norm in {"POST", "PUT", "PATCH"} and _header_present(headers_d, "content-length") and not has_body))

    # Headers/protocolo.
    header_names_l = {str(k).lower().strip() for k in headers_d.keys()}
    content_length_value = _header_lookup(headers_d, "content-length")
    content_length_int = _parse_int_header(content_length_value)
    transfer_encoding_value = _header_lookup(headers_d, "transfer-encoding")
    content_length_present = bool(content_length_value) or "content-length" in header_names_l
    transfer_encoding_present = bool(transfer_encoding_value) or "transfer-encoding" in header_names_l
    content_length_mismatch_abs = 0
    content_length_mismatch_flag = 0
    if content_length_present:
        if content_length_int is None or content_length_int < 0:
            content_length_mismatch_flag = 1
        else:
            content_length_mismatch_abs = abs(int(content_length_int) - int(body_len))
            content_length_mismatch_flag = int(content_length_mismatch_abs > 0)
    method_override_present = int(any(h in header_names_l for h in METHOD_OVERRIDE_HEADER_NAMES) or any(k.lower() in {"_method", "method", "http_method", "httpmethod"} for k in names))
    header_control_chars = sum(1 for ch in headers_txt if ord(ch) < 32 and ch not in "\t\n\r")

    # Body y estructura.
    json_flag, json_key_count, json_max_depth = _json_shape_stats(body_raw)
    body_lstrip = body_raw.lstrip()
    xml_body_flag = int("xml" in content_type.lower() or body_lstrip.startswith("<"))
    xml_tag_count = len(_XML_TAG_RE.findall(body_raw)) if xml_body_flag or "<" in body_raw else 0
    multipart_body_flag = int("multipart/form-data" in content_type.lower())
    param_names_joined = "&".join(names)
    sensitive_param_name_count = sum(1 for n in names if re.sub(r"[^a-z0-9_]+", "", n.lower()) in SENSITIVE_PARAM_NAMES)

    # XPath/XML.
    xpath_axis_count = len(_XPATH_AXIS_RE.findall(analysis_text))
    xpath_function_count = len(_XPATH_FUNCTION_RE.findall(analysis_text))
    xpath_predicate_count = analysis_text.count("[") + analysis_text.count("]")
    xpath_attribute_ref_count = len(_XPATH_ATTR_RE.findall(analysis_text))
    xpath_double_slash_count = analysis_text.count("//")
    xpath_pipe_union_count = analysis_text.count("|")
    xml_entity_count = len(_XML_ENTITY_RE.findall(analysis_text))

    # Scanning/rutas.
    path_l = path_dec.lower()
    uri_l = _decode_twice(uri_raw).lower()
    user_agent = _header_lookup(headers_d, "user-agent").lower()
    segments = _path_segments(path_dec)
    _, ext = os.path.splitext(path_l.split("?", 1)[0])

    feats = {
        # Baseline 30.
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

        # Método.
        "method_is_get": float(method_norm == "GET"),
        "method_is_post": float(method_norm == "POST"),
        "method_is_put": float(method_norm == "PUT"),
        "method_is_delete": float(method_norm == "DELETE"),
        "method_is_patch": float(method_norm == "PATCH"),
        "method_is_options": float(method_norm == "OPTIONS"),
        "method_is_head": float(method_norm == "HEAD"),
        "method_is_trace": float(method_norm == "TRACE"),
        "method_is_connect": float(method_norm == "CONNECT"),
        "method_is_other": float(method_norm not in known_methods),
        "method_is_safe": float(method_norm in safe_methods),
        "method_body_inconsistency": float(method_body_inconsistency),

        # Headers/protocolo.
        "header_count": float(len(headers_d)),
        "headers_total_len": float(len(headers_txt)),
        "host_present": float(_header_present(headers_d, "host")),
        "user_agent_present": float(_header_present(headers_d, "user-agent")),
        "content_type_present": float(_header_present(headers_d, "content-type")),
        "content_length_present": float(content_length_present),
        "transfer_encoding_present": float(transfer_encoding_present),
        "transfer_encoding_chunked": float("chunked" in transfer_encoding_value.lower()),
        "content_length_transfer_encoding_both": float(content_length_present and transfer_encoding_present),
        "content_length_mismatch_abs": float(content_length_mismatch_abs),
        "content_length_mismatch_flag": float(content_length_mismatch_flag),
        "method_override_header_present": float(method_override_present),
        "header_control_char_count": float(header_control_chars),
        "header_crlf_injection_count": float(_count_patterns(headers_txt + "\n" + headers_dec1 + "\n" + headers_dec2, PATTERNS["crlf"])),

        # Body/parámetros.
        "n_body_params": float(len(b_params)),
        "n_total_params": float(len(all_params)),
        "empty_param_value_count": float(sum(1 for _, v in all_params if v == "")),
        "max_param_name_len": float(max((len(k) for k in names), default=0)),
        "param_name_entropy": float(_entropy(param_names_joined)),
        "sensitive_param_name_count": float(sensitive_param_name_count),
        "json_body_flag": float(json_flag),
        "json_key_count": float(json_key_count),
        "json_max_depth": float(json_max_depth),
        "xml_body_flag": float(xml_body_flag),
        "xml_tag_count": float(xml_tag_count),
        "multipart_body_flag": float(multipart_body_flag),

        # XPath/XML.
        "xpath_axis_count": float(xpath_axis_count),
        "xpath_function_count": float(xpath_function_count),
        "xpath_predicate_count": float(xpath_predicate_count),
        "xpath_attribute_ref_count": float(xpath_attribute_ref_count),
        "xpath_double_slash_count": float(xpath_double_slash_count),
        "xpath_pipe_union_count": float(xpath_pipe_union_count),
        "xml_entity_count": float(xml_entity_count),

        # Scanning/rutas.
        "sensitive_path_token_count": float(_token_count_from_set(uri_l, SENSITIVE_PATH_TOKENS)),
        "hidden_file_path_count": float(sum(1 for seg in segments if seg.startswith(".") and len(seg) > 1)),
        "backup_config_token_count": float(_token_count_from_set(uri_l, BACKUP_CONFIG_TOKENS)),
        "common_scanner_ua_token_count": float(_token_count_from_set(user_agent, SCANNER_UA_TOKENS)),
        "suspicious_file_extension_count": float(1 if ext in SUSPICIOUS_EXTENSIONS else 0),
        "path_dot_segment_count": float(sum(1 for seg in segments if seg in {".", ".."}) + path_dec.count("../") + path_dec.count("..\\")),
        "path_repeated_slash_count": float(path_dec.count("//")),
        "path_encoded_slash_count": float(len(re.findall(r"%(?:2f|5c)", uri_raw, flags=re.IGNORECASE))),
        "path_null_byte_count": float(uri_raw.lower().count("%00") + body_raw.lower().count("%00") + analysis_text.count("\x00")),
    }
    return {f: float(feats.get(f, 0.0)) for f in FIXED_FEATURES}


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
    desc = f"Extracting {FEATURE_POLICY_NAME} features ({len(FIXED_FEATURES)})"
    if workers and workers > 1 and len(rows) > 500:
        with ProcessPoolExecutor(max_workers=int(workers)) as ex:
            vals = list(tqdm(ex.map(_extract_features_from_tuple, rows, chunksize=max(1, int(chunksize))), total=len(rows), desc=desc, mininterval=5))
    else:
        vals = [_extract_features_from_tuple(r) for r in tqdm(rows, total=len(rows), desc=desc, mininterval=5)]
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


def _capec_sort_key(capec_id: str) -> Tuple[int, str]:
    m = _CANON_CAPEC_RE.search(capec_id or "")
    if not m:
        return (10**9, capec_id or "")
    return (int(m.group(1)), capec_id or "")


def _capec_ids_from_labels(labels: Sequence[str]) -> Tuple[str, ...]:
    ids: List[str] = []
    seen = set()
    for lab in labels or []:
        capec = _capec_id_from_label(str(lab))
        if capec and capec not in seen:
            ids.append(capec)
            seen.add(capec)
    return tuple(sorted(ids, key=_capec_sort_key))


def _harvard_combo_key_from_labels(labels: Sequence[str]) -> str:
    ids = _capec_ids_from_labels(labels)
    return "NORMAL" if not ids else "+".join(ids)


def _harvard_combo_ids_text_from_key(combo_key: str) -> str:
    key = _safe_str(combo_key).strip()
    if not key or key == "NORMAL":
        return "0"
    return " + ".join(part.replace("CAPEC-", "") for part in key.split("+") if part)


def _harvard_combo_display_from_labels(labels: Sequence[str]) -> str:
    ids = _capec_ids_from_labels(labels)
    if not ids:
        return "Sin ataque / Normal"
    by_id: Dict[str, str] = {}
    for lab in labels or []:
        capec = _capec_id_from_label(str(lab))
        if capec:
            pretty = re.sub(r"^CAPEC-\d+\s+", "", str(lab)).strip() or capec
            by_id[capec] = pretty
    parts = []
    for capec in ids:
        capec_num = capec.replace("CAPEC-", "")
        parts.append(f"{capec_num} - {by_id.get(capec, capec)}")
    return " + ".join(parts)


def _is_harvard_top14_combo_mode(mode: str) -> bool:
    return (mode or "").strip().lower().replace("_", "-") in {
        "top14-combo",
        "top14-combos",
        "top14-combination",
        "top14-combinations",
        "exact-top14-combo",
        "exact-top14-combos",
    }


def _harvard_top14_label_from_combo_key(combo_key: str) -> str:
    spec = HARVARD_TOP14_COMBO_KEY_TO_SPEC.get(_safe_str(combo_key).strip())
    if spec is None:
        return HARVARD_IGNORED_TOP14_LABEL
    return str(spec["label"])


def _harvard_top14_rank_from_combo_key(combo_key: str) -> int:
    spec = HARVARD_TOP14_COMBO_KEY_TO_SPEC.get(_safe_str(combo_key).strip())
    return int(spec["rank"]) if spec is not None else 0


def _harvard_top14_combo_classes_df() -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for spec in HARVARD_TOP14_COMBO_CLASSES:
        capec_ids = tuple(spec["capec_ids"])
        combo_key = "+".join(capec_ids) if capec_ids else "NORMAL"
        rows.append({
            "rank": int(spec["rank"]),
            "label_multiclass": str(spec["label"]),
            "combo_key": combo_key,
            "capec_ids": _harvard_combo_ids_text_from_key(combo_key),
            "display": str(spec["display"]),
            "expected_count_full_dataset": int(spec["expected_count"]),
        })
    return pd.DataFrame(rows)


def _is_harvard_operational_group_mode(mode: str) -> bool:
    """Modo Harvard operativo derivado del top 14 exacto.

    Primero filtra a las mismas 14 combinaciones CAPEC más frecuentes; después
    agrupa las clases ambiguas que venían limitando el F1 por clase:
      * CAPEC-272, CAPEC-274 y CAPEC-272+274 -> HTTPProtocolManipulation.
      * CAPEC-153 y CAPEC-194 -> DataManipulation.

    El resultado final queda en 11 clases operativas para Harvard. Se mantienen
    Torpeda y las 58 features intra-request sin cambios.
    """
    return (mode or "").strip().lower().replace("_", "-") in {
        "top14-operational",
        "top14-operational-groups",
        "top14-operational-group",
        "operational-top14",
        "operational-groups",
        "operational-group",
        "top11-operational",
        "top11-operational-groups",
    }


def _harvard_operational_group_label_from_combo_key(combo_key: str) -> str:
    key = _safe_str(combo_key).strip() or "NORMAL"
    if key not in HARVARD_TOP14_COMBO_KEY_TO_SPEC:
        return HARVARD_IGNORED_TOP14_LABEL
    if key == "NORMAL":
        return "NORMAL"
    if key in {"CAPEC-272", "CAPEC-274", "CAPEC-272+CAPEC-274"}:
        return "HARVARD-HTTPProtocolManipulation"
    if key in {"CAPEC-153", "CAPEC-194"}:
        return "HARVARD-DataManipulation"
    # El resto conserva una etiqueta operativa cercana a la combinación exacta original.
    return str(HARVARD_TOP14_COMBO_KEY_TO_SPEC[key]["label"])


def _harvard_operational_group_classes_df() -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for spec in HARVARD_TOP14_COMBO_CLASSES:
        capec_ids = tuple(spec["capec_ids"])
        combo_key = "+".join(capec_ids) if capec_ids else "NORMAL"
        final_label = _harvard_operational_group_label_from_combo_key(combo_key)
        if final_label == "HARVARD-HTTPProtocolManipulation":
            rule = "CAPEC-272/CAPEC-274 agrupados como manipulación operativa de protocolo HTTP"
        elif final_label == "HARVARD-DataManipulation":
            rule = "CAPEC-153/CAPEC-194 agrupados como manipulación operativa de datos"
        else:
            rule = "conserva la clase/combinación del top 14"
        rows.append({
            "top14_rank": int(spec["rank"]),
            "combo_key": combo_key,
            "capec_ids": _harvard_combo_ids_text_from_key(combo_key),
            "top14_original_label": str(spec["label"]),
            "top14_display": str(spec["display"]),
            "operational_label": final_label,
            "grouping_rule": rule,
            "expected_count_full_dataset": int(spec["expected_count"]),
        })
    return pd.DataFrame(rows)


def _harvard_operational_final_classes_df() -> pd.DataFrame:
    mapping = _harvard_operational_group_classes_df()
    grouped = (
        mapping.groupby("operational_label", dropna=False)
        .agg(
            expected_count_full_dataset=("expected_count_full_dataset", "sum"),
            top14_combo_keys=("combo_key", lambda x: " | ".join(map(str, x))),
            top14_display=("top14_display", lambda x: " | ".join(map(str, x))),
        )
        .reset_index()
        .rename(columns={"operational_label": "label_multiclass"})
    )
    grouped = grouped.sort_values("expected_count_full_dataset", ascending=False).reset_index(drop=True)
    grouped.insert(0, "rank_operational", np.arange(1, len(grouped) + 1))
    return grouped


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
    out["label_combo_key"] = out["label_multilabel"].map(_harvard_combo_key_from_labels)
    out["label_combo_capec_ids"] = out["label_combo_key"].map(_harvard_combo_ids_text_from_key)
    out["label_combo_native"] = out["label_multilabel"].map(_harvard_combo_display_from_labels)
    out["label_top14_combo_rank"] = out["label_combo_key"].map(_harvard_top14_rank_from_combo_key).astype(int)
    out["label_binary"] = out.apply(
        lambda r: 0 if (int(r[schema.normal_col]) == 1 and all(int(r[c]) == 0 for c in attack_cols)) else 1,
        axis=1,
    )
    sev_map = schema.severity_map if schema.severity_map is not None else DEFAULT_SRBH_SEVERITY_MAP
    picked = out["label_multilabel"].map(lambda labs: _choose_primary_label(labs, schema.multiclass_strategy, sev_map))
    out["label_multiclass_native"] = picked.map(lambda t: t[0])
    out["label_primary_severity"] = picked.map(lambda t: t[1])
    mode = (harvard_label_mode or "top14-combo").strip().lower()
    pref = label_prefix or "HARVARD"
    if _is_harvard_operational_group_mode(mode):
        # Primero se conserva el mismo universo top 14; luego se agrupan clases ambiguas
        # en una taxonomía operativa más defendible para multiclase. Las combinaciones
        # fuera del top 14 reciben una etiqueta interna y se eliminan en apply_label_filters.
        out[LABEL_COL] = out["label_combo_key"].map(_harvard_operational_group_label_from_combo_key)
    elif _is_harvard_top14_combo_mode(mode):
        # La clase final de Harvard pasa a ser la combinación exacta CAPEC.
        # Las combinaciones que no están en el top 14 reciben una etiqueta interna
        # y se eliminan en apply_label_filters antes de extraer features.
        out[LABEL_COL] = out["label_combo_key"].map(_harvard_top14_label_from_combo_key)
    elif mode in {"native", "capec", "srbh"}:
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

    if dataset == "harvard" and (
        _is_harvard_top14_combo_mode(getattr(args, "harvard_label_mode", ""))
        or _is_harvard_operational_group_mode(getattr(args, "harvard_label_mode", ""))
    ):
        mode = str(getattr(args, "harvard_label_mode", "") or "").strip().lower()
        if _is_harvard_operational_group_mode(mode):
            mapping_df = _harvard_operational_group_classes_df()
            final_df = _harvard_operational_final_classes_df()
            mapping_df.to_csv(out_dir / "harvard_top14_to_operational_groups_mapping.csv", index=False)
            final_df.to_csv(out_dir / "harvard_operational_group_classes_allowed.csv", index=False)
            allowed = set(final_df["label_multiclass"].astype(str).tolist())
            allowed_combo_keys = mapping_df["combo_key"].astype(str).tolist()
            allowed_classes_path = str(out_dir / "harvard_operational_group_classes_allowed.csv")
            operation_name = "harvard_top14_operational_group_filter"
        else:
            allowed = set(HARVARD_TOP14_ALLOWED_LABELS)
            allowed_df = _harvard_top14_combo_classes_df()
            allowed_df.to_csv(out_dir / "harvard_top14_combo_classes_allowed.csv", index=False)
            allowed_combo_keys = allowed_df["combo_key"].astype(str).tolist()
            allowed_classes_path = str(out_dir / "harvard_top14_combo_classes_allowed.csv")
            operation_name = "harvard_top14_combo_filter"

        before = len(out)
        mask_allowed = out[LABEL_COL].astype(str).isin(allowed)
        dropped_rows = int(before - int(mask_allowed.sum()))
        dropped_summary = pd.DataFrame()
        if dropped_rows > 0:
            if {"label_combo_key", "label_combo_native"}.issubset(out.columns):
                dropped_summary = (
                    out.loc[~mask_allowed, ["label_combo_key", "label_combo_capec_ids", "label_combo_native"]]
                    .groupby(["label_combo_key", "label_combo_capec_ids", "label_combo_native"], dropna=False)
                    .size()
                    .reset_index(name="count")
                    .sort_values("count", ascending=False)
                )
            else:
                dropped_summary = (
                    out.loc[~mask_allowed, LABEL_COL].astype(str).value_counts()
                    .rename_axis("label")
                    .reset_index(name="count")
                )
            dropped_summary.to_csv(out_dir / "harvard_ignored_non_top14_combinations.csv", index=False)
        out = out.loc[mask_allowed].copy()
        report["operations"].append({
            "op": operation_name,
            "removed_rows": dropped_rows,
            "allowed_labels": sorted(allowed),
            "allowed_combo_keys": allowed_combo_keys,
            "ignored_combination_count": int(len(dropped_summary)) if dropped_rows > 0 else 0,
            "allowed_classes_path": allowed_classes_path,
            "operational_mapping_path": str(out_dir / "harvard_top14_to_operational_groups_mapping.csv") if _is_harvard_operational_group_mode(mode) else None,
            "ignored_combinations_path": str(out_dir / "harvard_ignored_non_top14_combinations.csv") if dropped_rows > 0 else None,
        })

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
# Modelo fijado: HistGradientBoostingClassifier
# ---------------------------------------------------------------------------

def make_histgb_model(args: argparse.Namespace) -> HistGradientBoostingClassifier:
    preset = str(args.histgb_preset or "strong").lower()
    if preset == "fast":
        return HistGradientBoostingClassifier(
            max_iter=220,
            learning_rate=0.07,
            max_leaf_nodes=31,
            l2_regularization=0.05,
            early_stopping=True,
            random_state=args.seed,
        )
    if preset == "max":
        return HistGradientBoostingClassifier(
            max_iter=750,
            learning_rate=0.035,
            max_leaf_nodes=63,
            l2_regularization=0.05,
            early_stopping=True,
            random_state=args.seed,
        )
    # Parámetros heredados del preset strong de la búsqueda previa.
    return HistGradientBoostingClassifier(
        max_iter=450,
        learning_rate=0.045,
        max_leaf_nodes=31,
        l2_regularization=0.05,
        early_stopping=True,
        random_state=args.seed,
    )


def default_weight_mode_for_dataset(dataset: str) -> str:
    # Desde el CSV de comparación: TorpEda hist_gradient_boosting; Harvard hist_gradient_boosting__w_sqrt_balanced.
    return "sqrt_balanced" if dataset == "harvard" else "none"


def make_sample_weight_vector(y: np.ndarray, *, mode: str, clip: float = 20.0) -> Optional[np.ndarray]:
    mode_l = (mode or "none").strip().lower().replace("-", "_")
    y_arr = np.asarray(y)
    if mode_l in {"", "none", "off", "false", "no"}:
        return None
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


def can_stratify(y: Sequence[Any], test_size: float) -> bool:
    counts = pd.Series(y).value_counts(dropna=False)
    if counts.empty or counts.min() < 2:
        return False
    n_classes = len(counts)
    n_test = int(math.ceil(len(y) * test_size))
    return n_test >= n_classes and (len(y) - n_test) >= n_classes


def split_indices(y: Sequence[Any], test_size: float, seed: int) -> Tuple[np.ndarray, np.ndarray, bool]:
    idx = np.arange(len(y))
    strat_ok = can_stratify(y, test_size)
    strat = y if strat_ok else None
    if not strat_ok:
        print("[WARN] Split no estratificado: hay clases raras o split demasiado chico.", flush=True)
    tr, te = train_test_split(idx, test_size=test_size, random_state=seed, shuffle=True, stratify=strat)
    return np.asarray(tr), np.asarray(te), bool(strat_ok)


# ---------------------------------------------------------------------------
# Métricas predictivas
# ---------------------------------------------------------------------------

def _per_class_ovr_metrics(
    y_true_idx: np.ndarray,
    y_pred_idx: np.ndarray,
    classes: Sequence[str],
    proba: Optional[np.ndarray],
) -> pd.DataFrame:
    n_classes = len(classes)
    rows: List[Dict[str, Any]] = []
    for k, cls in enumerate(classes):
        yt = (y_true_idx == k).astype(int)
        yp = (y_pred_idx == k).astype(int)
        tp = int(np.sum((yt == 1) & (yp == 1)))
        fp = int(np.sum((yt == 0) & (yp == 1)))
        fn = int(np.sum((yt == 1) & (yp == 0)))
        tn = int(np.sum((yt == 0) & (yp == 0)))
        support = int(np.sum(yt == 1))
        pred_support = int(np.sum(yp == 1))
        prec = _safe_div(tp, tp + fp)
        rec = _safe_div(tp, tp + fn)
        spec = _safe_div(tn, tn + fp)
        fpr = _safe_div(fp, fp + tn)
        fnr = _safe_div(fn, fn + tp)
        f1 = None if prec is None or rec is None or (prec + rec) == 0 else 2.0 * prec * rec / (prec + rec)
        roc = None
        pr = None
        if proba is not None and proba.shape[1] == n_classes:
            scores = proba[:, k]
            try:
                if len(np.unique(yt)) == 2:
                    roc = _safe_float(roc_auc_score(yt, scores))
            except Exception:
                roc = None
            try:
                if support > 0:
                    pr = _safe_float(average_precision_score(yt, scores))
            except Exception:
                pr = None
        rows.append({
            "class": str(cls),
            "class_index": int(k),
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
            "roc_auc_ovr": roc,
            "pr_auc_ovr": pr,
        })
    return pd.DataFrame(rows)


def _weighted_mean(values: Sequence[Optional[float]], weights: Sequence[int]) -> Optional[float]:
    vals = []
    wts = []
    for v, w in zip(values, weights):
        if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
            continue
        vals.append(float(v))
        wts.append(float(w))
    if not vals:
        return None
    if sum(wts) <= 0:
        return float(np.mean(vals))
    return float(np.average(vals, weights=wts))


def compute_multiclass_metrics(
    y_true_idx: np.ndarray,
    y_pred_idx: np.ndarray,
    classes: Sequence[str],
    proba: Optional[np.ndarray],
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    labels_idx = list(range(len(classes)))
    y_true_labels = np.asarray([classes[i] for i in y_true_idx])
    y_pred_labels = np.asarray([classes[i] for i in y_pred_idx])
    per_class = _per_class_ovr_metrics(y_true_idx, y_pred_idx, classes, proba)
    out: Dict[str, Any] = {
        "accuracy": _safe_float(accuracy_score(y_true_idx, y_pred_idx)),
        "balanced_accuracy": _safe_float(balanced_accuracy_score(y_true_idx, y_pred_idx)),
        "precision_micro": _safe_float(precision_score(y_true_idx, y_pred_idx, average="micro", zero_division=0)),
        "precision_macro": _safe_float(precision_score(y_true_idx, y_pred_idx, average="macro", zero_division=0)),
        "precision_weighted": _safe_float(precision_score(y_true_idx, y_pred_idx, average="weighted", zero_division=0)),
        "recall_micro": _safe_float(recall_score(y_true_idx, y_pred_idx, average="micro", zero_division=0)),
        "recall_macro": _safe_float(recall_score(y_true_idx, y_pred_idx, average="macro", zero_division=0)),
        "recall_weighted": _safe_float(recall_score(y_true_idx, y_pred_idx, average="weighted", zero_division=0)),
        "f1_micro": _safe_float(f1_score(y_true_idx, y_pred_idx, average="micro", zero_division=0)),
        "f1_macro": _safe_float(f1_score(y_true_idx, y_pred_idx, average="macro", zero_division=0)),
        "f1_weighted": _safe_float(f1_score(y_true_idx, y_pred_idx, average="weighted", zero_division=0)),
        "mcc": _safe_float(matthews_corrcoef(y_true_idx, y_pred_idx)),
        "confusion_matrix": confusion_matrix(y_true_idx, y_pred_idx, labels=labels_idx).tolist(),
        "classification_report": classification_report(y_true_labels, y_pred_labels, labels=list(classes), output_dict=True, digits=6, zero_division=0),
    }
    # AUC/PR-AUC OVR por clase + agregados macro/weighted/micro.
    if proba is not None and proba.ndim == 2 and proba.shape[1] == len(classes):
        roc_vals = per_class["roc_auc_ovr"].tolist()
        pr_vals = per_class["pr_auc_ovr"].tolist()
        supports = per_class["support"].astype(int).tolist()
        valid_roc = [float(v) for v in roc_vals if pd.notna(v)]
        valid_pr = [float(v) for v in pr_vals if pd.notna(v)]
        out["roc_auc_ovr_macro"] = float(np.mean(valid_roc)) if valid_roc else None
        out["roc_auc_ovr_weighted"] = _weighted_mean([_safe_float(v) for v in roc_vals], supports)
        out["pr_auc_macro"] = float(np.mean(valid_pr)) if valid_pr else None
        out["pr_auc_weighted"] = _weighted_mean([_safe_float(v) for v in pr_vals], supports)
        try:
            y_bin = np.zeros((len(y_true_idx), len(classes)), dtype=int)
            y_bin[np.arange(len(y_true_idx)), y_true_idx] = 1
            if len(np.unique(y_bin.ravel())) == 2:
                out["roc_auc_ovr_micro"] = _safe_float(roc_auc_score(y_bin.ravel(), proba.ravel()))
                out["pr_auc_micro"] = _safe_float(average_precision_score(y_bin.ravel(), proba.ravel()))
        except Exception:
            pass
    return out, per_class


def compute_binary_normal_attack_metrics(
    y_true_idx: np.ndarray,
    y_pred_idx: np.ndarray,
    classes: Sequence[str],
    proba: Optional[np.ndarray],
    normal_label: str = "NORMAL",
) -> Dict[str, Any]:
    if normal_label not in set(classes):
        return {"available": False, "reason": f"normal_label {normal_label!r} not present"}
    normal_idx = list(classes).index(normal_label)
    yt = (y_true_idx != normal_idx).astype(int)  # 1 = ataque/anómalo, 0 = normal
    yp = (y_pred_idx != normal_idx).astype(int)
    cm = confusion_matrix(yt, yp, labels=[0, 1])
    tn, fp, fn, tp = [int(x) for x in cm.ravel()]
    out: Dict[str, Any] = {
        "available": True,
        "positive_class": "attack_or_anomalous",
        "negative_class": normal_label,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "accuracy": _safe_float(accuracy_score(yt, yp)),
        "balanced_accuracy": _safe_float(balanced_accuracy_score(yt, yp)),
        "precision": _safe_float(precision_score(yt, yp, zero_division=0)),
        "recall_tpr": _safe_float(recall_score(yt, yp, zero_division=0)),
        "f1": _safe_float(f1_score(yt, yp, zero_division=0)),
        "specificity_tnr": _safe_div(tn, tn + fp),
        "fpr": _safe_div(fp, fp + tn),
        "fnr": _safe_div(fn, fn + tp),
        "mcc": _safe_float(matthews_corrcoef(yt, yp)),
        "confusion_matrix_labels_0normal_1attack": cm.tolist(),
    }
    if proba is not None and proba.ndim == 2 and proba.shape[1] == len(classes):
        try:
            attack_score = 1.0 - proba[:, normal_idx]
            if len(np.unique(yt)) == 2:
                out["roc_auc"] = _safe_float(roc_auc_score(yt, attack_score))
                out["pr_auc"] = _safe_float(average_precision_score(yt, attack_score))
        except Exception:
            pass
    return out


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

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


def _fi_sample_indices(y: np.ndarray, max_rows: int, seed: int, min_rows_per_class: int = 200) -> Tuple[np.ndarray, bool, Dict[str, int]]:
    y = np.asarray(y)
    n = len(y)
    if max_rows is None or int(max_rows) <= 0 or n <= int(max_rows):
        idx = np.arange(n)
        return idx, False, {str(k): int(v) for k, v in pd.Series(y[idx]).value_counts().sort_index().to_dict().items()}

    max_rows = int(max_rows)
    rng = np.random.RandomState(seed)
    all_idx = np.arange(n)
    classes = np.unique(y)
    selected_mask = np.zeros(n, dtype=bool)
    selected: List[int] = []

    # Garantiza presencia de clases minoritarias en FI por clase.
    per_class_cap = max(1, max_rows // max(1, len(classes)))
    min_pc = max(1, min(int(min_rows_per_class or 1), per_class_cap))
    for cls in classes:
        cls_idx = all_idx[y == cls]
        if len(cls_idx) == 0:
            continue
        take = min(len(cls_idx), min_pc)
        chosen = rng.choice(cls_idx, size=take, replace=False)
        selected.extend(int(i) for i in chosen)
        selected_mask[chosen] = True

    remaining = max_rows - len(selected)
    if remaining > 0:
        rest = all_idx[~selected_mask]
        if len(rest) > 0:
            chosen = rng.choice(rest, size=min(remaining, len(rest)), replace=False)
            selected.extend(int(i) for i in chosen)

    idx = np.asarray(selected, dtype=int)
    rng.shuffle(idx)
    counts = {str(k): int(v) for k, v in pd.Series(y[idx]).value_counts().sort_index().to_dict().items()}
    return idx, True, counts


def _permutation_importance_with_callable(
    model: Any,
    X_fi: np.ndarray,
    y_fi: np.ndarray,
    features: Sequence[str],
    scorer: Any,
    *,
    n_repeats: int,
    random_state: int,
    n_jobs: int,
) -> Tuple[pd.DataFrame, Optional[str]]:
    out = pd.DataFrame({"feature": list(features)})
    err = None
    try:
        r = permutation_importance(
            model,
            X_fi,
            y_fi,
            scoring=scorer,
            n_repeats=int(n_repeats),
            random_state=int(random_state),
            n_jobs=int(n_jobs),
        )
        out["permutation_importance_mean"] = r.importances_mean
        out["permutation_importance_std"] = r.importances_std
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        out["permutation_importance_mean"] = 0.0
        out["permutation_importance_std"] = 0.0
        out["permutation_error"] = err
    return out, err


def compute_feature_importance(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    features: Sequence[str],
    args: argparse.Namespace,
    out_dir: Path,
    classes: Optional[Sequence[str]] = None,
) -> Dict[str, str]:
    idx, sampled, sample_counts = _fi_sample_indices(
        y_test,
        int(args.fi_max_rows),
        int(args.seed),
        int(getattr(args, "fi_min_rows_per_class", 200)),
    )
    X_fi = X_test[idx]
    y_fi = y_test[idx]

    perm_df = pd.DataFrame({"feature": list(features)})
    perm_error = None
    if args.fi_n_repeats and args.fi_n_repeats > 0:
        try:
            r = permutation_importance(
                model,
                X_fi,
                y_fi,
                scoring=args.fi_scoring,
                n_repeats=int(args.fi_n_repeats),
                random_state=args.seed,
                n_jobs=int(args.fi_n_jobs),
            )
            perm_df["permutation_importance_mean"] = r.importances_mean
            perm_df["permutation_importance_std"] = r.importances_std
        except Exception as e:
            perm_error = f"{type(e).__name__}: {e}"
            perm_df["permutation_importance_mean"] = 0.0
            perm_df["permutation_importance_std"] = 0.0
            perm_df["permutation_error"] = perm_error
    else:
        perm_df["permutation_importance_mean"] = 0.0
        perm_df["permutation_importance_std"] = 0.0
        perm_error = "disabled_by_fi_n_repeats"
    perm_df["permutation_scoring"] = args.fi_scoring
    perm_df["fi_rows"] = int(len(X_fi))
    perm_df["fi_sampled"] = bool(sampled)
    perm_df["feature_group"] = [FEATURE_GROUP_BY_NAME.get(f, "unknown") for f in features]
    perm_df.to_csv(out_dir / "feature_importance_permutation.csv", index=False)

    uni_df = _univariate_eta2(X_test, y_test, features)
    uni_df["feature_group"] = [FEATURE_GROUP_BY_NAME.get(f, "unknown") for f in features]
    uni_df.to_csv(out_dir / "feature_importance_univariate_eta2.csv", index=False)

    out = perm_df.merge(uni_df, on=["feature", "feature_group"], how="outer")
    for c in ["permutation_importance_mean", "permutation_importance_std", "univariate_eta2"]:
        if c in out.columns:
            out[c] = out[c].fillna(0.0)
    if float(np.abs(out["permutation_importance_mean"]).sum()) > 0:
        out["importance"] = out["permutation_importance_mean"]
        out["importance_method"] = "permutation_importance"
    else:
        out["importance"] = out["univariate_eta2"]
        out["importance_method"] = "univariate_eta2_fallback"
    out = out.sort_values(["importance", "univariate_eta2"], ascending=False).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    out.to_csv(out_dir / "feature_importance.csv", index=False)

    paths: Dict[str, str] = {
        "feature_importance": str(out_dir / "feature_importance.csv"),
        "feature_importance_permutation": str(out_dir / "feature_importance_permutation.csv"),
        "feature_importance_univariate_eta2": str(out_dir / "feature_importance_univariate_eta2.csv"),
    }

    # Feature importance por clase: one-vs-rest F1 para cada clase del clasificador multiclase.
    per_class_error: Dict[str, Optional[str]] = {}
    if bool(getattr(args, "fi_per_class", True)) and classes is not None and int(getattr(args, "fi_per_class_n_repeats", 0)) > 0:
        class_rows: List[pd.DataFrame] = []
        for class_idx, class_name in enumerate(classes):
            support_in_fi = int(np.sum(y_fi == class_idx))
            support_total = int(np.sum(y_test == class_idx))

            def scorer(estimator: Any, X: np.ndarray, y: np.ndarray, class_idx: int = class_idx) -> float:
                pred = estimator.predict(X)
                return float(f1_score((y == class_idx).astype(int), (pred == class_idx).astype(int), zero_division=0))

            class_df, err = _permutation_importance_with_callable(
                model,
                X_fi,
                y_fi,
                features,
                scorer,
                n_repeats=int(args.fi_per_class_n_repeats),
                random_state=int(args.seed) + 1000 + int(class_idx),
                n_jobs=int(args.fi_per_class_n_jobs),
            )
            per_class_error[str(class_name)] = err
            class_df.insert(0, "class", str(class_name))
            class_df.insert(1, "class_index", int(class_idx))
            class_df["support_total_test"] = support_total
            class_df["support_in_fi_sample"] = support_in_fi
            class_df["fi_rows"] = int(len(X_fi))
            class_df["scoring"] = "f1_one_vs_rest_for_class"
            class_df["feature_group"] = [FEATURE_GROUP_BY_NAME.get(f, "unknown") for f in features]
            class_df = class_df.sort_values("permutation_importance_mean", ascending=False).reset_index(drop=True)
            class_df.insert(2, "rank_in_class", np.arange(1, len(class_df) + 1))
            class_rows.append(class_df)
        if class_rows:
            per_class_df = pd.concat(class_rows, ignore_index=True)
            per_class_df.to_csv(out_dir / "feature_importance_per_class.csv", index=False)
            top20 = per_class_df[per_class_df["rank_in_class"] <= 20].copy()
            top20.to_csv(out_dir / "feature_importance_per_class_top20.csv", index=False)
            paths["feature_importance_per_class"] = str(out_dir / "feature_importance_per_class.csv")
            paths["feature_importance_per_class_top20"] = str(out_dir / "feature_importance_per_class_top20.csv")

    # Feature importance binaria derivada: NORMAL vs ataque/anómalo, manteniendo modelo multiclase.
    binary_errors: Dict[str, Optional[str]] = {}
    if bool(getattr(args, "fi_binary_normal_attack", True)) and classes is not None and "NORMAL" in set(classes) and int(getattr(args, "fi_binary_n_repeats", 0)) > 0:
        normal_idx = list(classes).index("NORMAL")
        bin_rows: List[pd.DataFrame] = []
        scorers = {
            "binary_attack_f1": lambda estimator, X, y: float(f1_score((y != normal_idx).astype(int), (estimator.predict(X) != normal_idx).astype(int), zero_division=0)),
            "binary_normal_f1": lambda estimator, X, y: float(f1_score((y == normal_idx).astype(int), (estimator.predict(X) == normal_idx).astype(int), zero_division=0)),
        }
        for metric_name, scorer in scorers.items():
            bdf, err = _permutation_importance_with_callable(
                model,
                X_fi,
                y_fi,
                features,
                scorer,
                n_repeats=int(args.fi_binary_n_repeats),
                random_state=int(args.seed) + 7000 + len(bin_rows),
                n_jobs=int(args.fi_per_class_n_jobs),
            )
            binary_errors[metric_name] = err
            bdf.insert(0, "binary_metric", metric_name)
            bdf["feature_group"] = [FEATURE_GROUP_BY_NAME.get(f, "unknown") for f in features]
            bdf["fi_rows"] = int(len(X_fi))
            bdf = bdf.sort_values("permutation_importance_mean", ascending=False).reset_index(drop=True)
            bdf.insert(1, "rank", np.arange(1, len(bdf) + 1))
            bin_rows.append(bdf)
        if bin_rows:
            binary_df = pd.concat(bin_rows, ignore_index=True)
            binary_df.to_csv(out_dir / "feature_importance_binary_normal_vs_attack.csv", index=False)
            paths["feature_importance_binary_normal_vs_attack"] = str(out_dir / "feature_importance_binary_normal_vs_attack.csv")

    feature_definitions_frame().to_csv(out_dir / "features_used.csv", index=False)
    paths["features_used"] = str(out_dir / "features_used.csv")

    _save_json({
        "created_at": _now_iso(),
        "method": "permutation_importance + univariate_eta2",
        "global_permutation_scoring": args.fi_scoring,
        "global_n_repeats": int(args.fi_n_repeats),
        "per_class_enabled": bool(getattr(args, "fi_per_class", True)),
        "per_class_n_repeats": int(getattr(args, "fi_per_class_n_repeats", 0)),
        "binary_normal_attack_enabled": bool(getattr(args, "fi_binary_normal_attack", True)),
        "binary_n_repeats": int(getattr(args, "fi_binary_n_repeats", 0)),
        "max_rows": int(args.fi_max_rows),
        "fi_rows_effective": int(len(X_fi)),
        "sampled": bool(sampled),
        "min_rows_per_class_requested": int(getattr(args, "fi_min_rows_per_class", 200)),
        "sample_class_counts": sample_counts,
        "n_features": int(len(features)),
        "feature_policy": FEATURE_POLICY_NAME,
        "feature_set_version": FEATURE_SET_VERSION,
        "permutation_error": perm_error,
        "per_class_errors": per_class_error,
        "binary_errors": binary_errors,
        "artifacts": paths,
    }, out_dir / "feature_importance_metadata.json")
    paths["feature_importance_metadata"] = str(out_dir / "feature_importance_metadata.json")
    return paths


# ---------------------------------------------------------------------------
# Métricas operativas/performance
# ---------------------------------------------------------------------------

def _percentile(vals: Sequence[float], q: float) -> Optional[float]:
    if not vals:
        return None
    return float(np.percentile(np.asarray(vals, dtype=float), q))


def measure_operational_metrics(
    model: Any,
    df_test_raw: pd.DataFrame,
    X_test: np.ndarray,
    args: argparse.Namespace,
    out_dir: Path,
) -> Dict[str, Any]:
    n_total = int(len(df_test_raw))
    if n_total == 0:
        return {"available": False, "reason": "empty test set"}
    rng = np.random.RandomState(args.seed)
    if args.operational_max_rows and args.operational_max_rows > 0 and n_total > args.operational_max_rows:
        local_idx = np.sort(rng.choice(np.arange(n_total), size=int(args.operational_max_rows), replace=False))
        df_meas = df_test_raw.iloc[local_idx].reset_index(drop=True)
        X_batch = X_test[local_idx]
        sampled = True
    else:
        df_meas = df_test_raw.reset_index(drop=True)
        X_batch = X_test
        sampled = False
    n = int(len(df_meas))

    # Batch inference-only throughput.
    cpu0 = time.process_time()
    t0 = time.perf_counter()
    _ = model.predict(X_batch)
    batch_predict_wall = time.perf_counter() - t0
    batch_predict_cpu = time.process_time() - cpu0

    # Batch full apply: features ya extraídas arriba no representan pipeline raw->predict, entonces se mide aparte.
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
    _ = model.predict(X_full)
    batch_pipeline_wall = time.perf_counter() - t0
    batch_pipeline_cpu = time.process_time() - cpu0

    # Latencia request-by-request: feature extraction + inferencia para simular uso WAF online.
    lat_ms: List[float] = []
    cpu0 = time.process_time()
    t0 = time.perf_counter()
    for tup in tqdm(tuples, total=len(tuples), desc="Operational latency raw request -> prediction", mininterval=5):
        a = time.perf_counter()
        x = np.asarray(_extract_features_from_tuple(tup), dtype=np.float32).reshape(1, -1)
        _ = model.predict(x)
        b = time.perf_counter()
        lat_ms.append((b - a) * 1000.0)
    online_pipeline_wall = time.perf_counter() - t0
    online_pipeline_cpu = time.process_time() - cpu0

    cpu_count = os.cpu_count() or 1
    metrics = {
        "available": True,
        "measured_rows": n,
        "test_rows_total": n_total,
        "sampled": bool(sampled),
        "operational_max_rows": int(args.operational_max_rows),
        "latency_ms_pipeline_online_p50": _percentile(lat_ms, 50),
        "latency_ms_pipeline_online_p95": _percentile(lat_ms, 95),
        "latency_ms_pipeline_online_p99": _percentile(lat_ms, 99),
        "latency_ms_pipeline_online_mean": float(np.mean(lat_ms)) if lat_ms else None,
        "latency_ms_pipeline_online_min": float(np.min(lat_ms)) if lat_ms else None,
        "latency_ms_pipeline_online_max": float(np.max(lat_ms)) if lat_ms else None,
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
        "cpu_utilization_pipeline_batch_percent_of_all_cores": _safe_div(batch_pipeline_cpu, batch_pipeline_wall * cpu_count) * 100.0 if batch_pipeline_wall > 0 else None,
        "peak_rss_mb": _get_peak_rss_mb(),
        "current_rss_mb": _get_current_rss_mb(),
    }
    pd.DataFrame({"latency_ms_pipeline_online": lat_ms}).to_csv(out_dir / "operational_latency_online_ms.csv", index=False)
    pd.DataFrame([metrics]).to_csv(out_dir / "operational_metrics.csv", index=False)
    _save_json(metrics, out_dir / "operational_metrics.json")
    return metrics


# ---------------------------------------------------------------------------
# Entrenamiento/evaluación por dataset
# ---------------------------------------------------------------------------

def _dataset_out_dir(args: argparse.Namespace, dataset: str) -> Path:
    if args.output_dir and args.dataset != "both":
        return Path(args.output_dir)
    return Path(args.definitivo_dir) / "multiclase" / dataset


def _clean_output_dir(out_dir: Path) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


def _write_run_config(args: argparse.Namespace, dataset: str, inputs: Sequence[str], out_dir: Path, extra: Dict[str, Any]) -> None:
    payload = {
        "created_at": _now_iso(),
        "dataset": dataset,
        "task": "multiclass",
        "model_family": "hist_gradient_boosting",
        "script": Path(__file__).name,
        "inputs": list(inputs),
        "features": FIXED_FEATURES,
        "feature_count": len(FIXED_FEATURES),
        "feature_policy": FEATURE_POLICY_NAME,
        "feature_set_version": FEATURE_SET_VERSION,
        "all_candidate_feature_count": len(ALL_84_CANDIDATE_FEATURES),
        "dropped_feature_count": len(FI_DROPPED_FEATURES),
        "dropped_features": FI_DROPPED_FEATURES,
        "feature_selection_rule": "keep if per-class permutation_importance_mean > 0 in at least one class across Torpeda or Harvard from corrida3; drop otherwise",
        "args": vars(args),
        "environment": {
            "python": sys.version,
            "executable": sys.executable,
            "cwd": os.getcwd(),
            "cpu_count": os.cpu_count(),
            "gpu_visible": _gpu_visible_auto(),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "SLURM_JOB_ID": os.environ.get("SLURM_JOB_ID"),
            "SLURM_JOB_GPUS": os.environ.get("SLURM_JOB_GPUS"),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
        },
        **extra,
    }
    _save_json(payload, out_dir / "run_config.json")


def train_eval_one_dataset(args: argparse.Namespace, dataset: str) -> Dict[str, Any]:
    out_dir = _dataset_out_dir(args, dataset)
    if args.clean_output_dir:
        _clean_output_dir(out_dir)
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)

    print("================================================================", flush=True)
    print(f"[START] {dataset} 11claseResults multiclase HistGradientBoosting @ {_now_iso()}", flush=True)
    print(f"[OUT] {out_dir}", flush=True)
    print("================================================================", flush=True)

    gpu_visible = _gpu_visible_auto()
    if args.use_gpu in {"on", "auto"}:
        msg = "visible" if gpu_visible else "no visible"
        print(f"[INFO] GPU {msg}; HistGradientBoostingClassifier es CPU/OpenMP, no CUDA. Se usará CPU.", flush=True)

    t_run0 = time.perf_counter()
    df_raw, inputs = load_dataset_for_run(args, dataset)
    load_seconds = time.perf_counter() - t_run0
    print(f"[INFO] Rows loaded before filters: {len(df_raw)}", flush=True)

    df_filtered, filtering_report = apply_label_filters(df_raw, args, dataset, out_dir)
    print(f"[INFO] Rows after filters: {len(df_filtered)}", flush=True)

    df, feature_extraction_seconds = build_feature_frame(df_filtered, workers=int(args.feature_workers), chunksize=int(args.feature_chunksize))
    processed_path = _save_processed(df, out_dir / "processed_useful58_operational.parquet")

    label_dist = df[LABEL_COL].astype(str).value_counts().rename_axis("label").reset_index(name="count")
    label_dist.to_csv(out_dir / "label_distribution.csv", index=False)
    _write_run_config(args, dataset, inputs, out_dir, {
        "load_seconds": float(load_seconds),
        "feature_extraction_seconds": float(feature_extraction_seconds),
        "processed_path": str(processed_path),
        "filtering_report_path": str(out_dir / "dataset_filtering_report.json"),
    })

    X = df[FIXED_FEATURES].fillna(0).astype(np.float32).to_numpy()
    y_labels = df[LABEL_COL].astype(str).to_numpy()
    le = LabelEncoder()
    y = le.fit_transform(y_labels)
    classes = [str(c) for c in le.classes_]
    if len(classes) < 2:
        raise SystemExit(f"Need at least 2 classes to train {dataset}.")

    train_idx, test_idx, stratified = split_indices(y_labels, float(args.test_size), int(args.seed))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    df_test_raw = df.iloc[test_idx].copy()

    weight_mode = str(args.sample_weight_mode or default_weight_mode_for_dataset(dataset))
    sw = make_sample_weight_vector(y_train, mode=weight_mode, clip=float(args.sample_weight_clip))
    model = make_histgb_model(args)
    # En datasets muy chicos de prueba, la validación interna de early_stopping puede
    # quedar con menos filas que clases. En datasets reales se mantiene activada.
    if len(X_train) < max(50, 3 * len(classes)):
        try:
            model.set_params(early_stopping=False)
        except Exception:
            pass

    t0 = time.perf_counter()
    cpu0 = time.process_time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if sw is None:
            model.fit(X_train, y_train)
        else:
            model.fit(X_train, y_train, sample_weight=sw)
    train_seconds = time.perf_counter() - t0
    train_cpu_seconds = time.process_time() - cpu0

    t0 = time.perf_counter()
    cpu0 = time.process_time()
    y_pred = model.predict(X_test).astype(int)
    proba = None
    try:
        proba = model.predict_proba(X_test)
    except Exception:
        proba = None
    inference_batch_seconds = time.perf_counter() - t0
    inference_batch_cpu_seconds = time.process_time() - cpu0

    metrics, per_class_df = compute_multiclass_metrics(y_test, y_pred, classes, proba)
    binary_metrics = compute_binary_normal_attack_metrics(y_test, y_pred, classes, proba, normal_label="NORMAL")

    # Artefactos predictivos.
    cm = pd.DataFrame(metrics["confusion_matrix"], index=pd.Index(classes, name="true\\pred"), columns=classes)
    cm.to_csv(out_dir / "confusion_matrix_test.csv")
    cm_norm = cm.div(cm.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    cm_norm.to_csv(out_dir / "confusion_matrix_test_normalized_true.csv")
    pd.DataFrame(metrics["classification_report"]).transpose().to_csv(out_dir / "classification_report_test.csv")
    _save_json(metrics["classification_report"], out_dir / "classification_report_test.json")
    per_class_df.to_csv(out_dir / "per_class_ovr_metrics.csv", index=False)
    _save_json(binary_metrics, out_dir / "binary_normal_vs_attack_metrics.json")
    pd.DataFrame([binary_metrics]).to_csv(out_dir / "binary_normal_vs_attack_metrics.csv", index=False)

    pred_df = pd.DataFrame({
        "row_index": test_idx,
        "sample_id": df.iloc[test_idx].get("sample_id", pd.Series(test_idx)).astype(str).to_numpy(),
        "source_file": df.iloc[test_idx].get("source_file", pd.Series([""] * len(test_idx))).astype(str).to_numpy(),
        "y_true": [classes[i] for i in y_test],
        "y_pred": [classes[i] for i in y_pred],
    })
    if proba is not None:
        pred_df["pred_prob_max"] = np.max(proba, axis=1)
        for i, cls in enumerate(classes):
            pred_df[f"proba_{_safe_name(cls)}"] = proba[:, i]
    pred_df.to_csv(out_dir / "test_predictions.csv", index=False)

    model_path = out_dir / "model_multiclass_histgb_useful58_operational.joblib"
    joblib.dump({
        "task": "multiclass",
        "dataset": dataset,
        "model": model,
        "label_encoder": le,
        "features": FIXED_FEATURES,
        "feature_count": len(FIXED_FEATURES),
        "feature_policy": FEATURE_POLICY_NAME,
        "feature_set_version": FEATURE_SET_VERSION,
        "classes": classes,
        "selected_model": "hist_gradient_boosting" if weight_mode in {"none", "off", ""} else f"hist_gradient_boosting__w_{weight_mode}",
        "selected_model_family": "hist_gradient_boosting",
        "weight_mode": weight_mode,
        "config": vars(args),
    }, model_path)
    model_size_bytes = int(model_path.stat().st_size) if model_path.exists() else 0

    fi_paths = compute_feature_importance(model, X_test, y_test, FIXED_FEATURES, args, out_dir, classes)
    operational_metrics = measure_operational_metrics(model, df_test_raw, X_test, args, out_dir)
    operational_metrics["model_size_bytes"] = model_size_bytes
    operational_metrics["model_size_mb"] = model_size_bytes / (1024.0 * 1024.0) if model_size_bytes else 0.0
    _save_json(operational_metrics, out_dir / "operational_metrics.json")
    pd.DataFrame([operational_metrics]).to_csv(out_dir / "operational_metrics.csv", index=False)

    final_train_build_seconds = float(train_seconds)
    run_total_seconds = time.perf_counter() - t_run0
    selected_model = "hist_gradient_boosting" if weight_mode in {"none", "off", ""} else f"hist_gradient_boosting__w_{weight_mode}"
    metrics_payload: Dict[str, Any] = {
        "created_at": _now_iso(),
        "task": "multiclass",
        "dataset": dataset,
        "dataset_label_mode": args.harvard_label_mode if dataset == "harvard" else "torpeda_native_common_names",
        "feature_policy": FEATURE_POLICY_NAME,
        "feature_count": len(FIXED_FEATURES),
        "features_used": FIXED_FEATURES,
        "feature_set_version": FEATURE_SET_VERSION,
        "selected_model": selected_model,
        "selected_model_family": "hist_gradient_boosting",
        "model_backend": "sklearn.ensemble.HistGradientBoostingClassifier",
        "gpu_requested": args.use_gpu,
        "gpu_visible": bool(gpu_visible),
        "gpu_used": False,
        "gpu_note": "HistGradientBoostingClassifier de scikit-learn no tiene backend CUDA; se acelera con CPU/OpenMP.",
        "weight_mode": weight_mode,
        "sample_weight_clip": float(args.sample_weight_clip),
        "classes": classes,
        "n_classes": int(len(classes)),
        "label_distribution": label_dist.to_dict(orient="records"),
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "test_size": float(args.test_size),
        "split_stratified": bool(stratified),
        "seed": int(args.seed),
        "timing": {
            "load_seconds": float(load_seconds),
            "feature_extraction_seconds": float(feature_extraction_seconds),
            "t_build_train_seconds": final_train_build_seconds,
            "t_build_train_cpu_seconds": float(train_cpu_seconds),
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
        "test_metrics": metrics,
        "binary_normal_vs_attack_metrics": binary_metrics,
        "operational_metrics": operational_metrics,
        "feature_importance_config": {
            "fi_n_repeats": int(args.fi_n_repeats),
            "fi_max_rows": int(args.fi_max_rows),
            "fi_scoring": str(args.fi_scoring),
            "fi_n_jobs": int(args.fi_n_jobs),
            "fi_min_rows_per_class": int(args.fi_min_rows_per_class),
            "fi_per_class": bool(args.fi_per_class),
            "fi_per_class_n_repeats": int(args.fi_per_class_n_repeats),
            "fi_per_class_n_jobs": int(args.fi_per_class_n_jobs),
            "fi_binary_normal_attack": bool(args.fi_binary_normal_attack),
            "fi_binary_n_repeats": int(args.fi_binary_n_repeats),
        },
        "artifacts": {
            "processed_features": str(processed_path),
            "model": str(model_path),
            "metrics": str(out_dir / "metrics_multiclass_histgb_useful58_operational.json"),
            "metrics_flat": str(out_dir / "metrics_flat.csv"),
            "confusion_matrix": str(out_dir / "confusion_matrix_test.csv"),
            "confusion_matrix_normalized_true": str(out_dir / "confusion_matrix_test_normalized_true.csv"),
            "classification_report_csv": str(out_dir / "classification_report_test.csv"),
            "classification_report_json": str(out_dir / "classification_report_test.json"),
            "per_class_ovr_metrics": str(out_dir / "per_class_ovr_metrics.csv"),
            "binary_normal_vs_attack_metrics": str(out_dir / "binary_normal_vs_attack_metrics.json"),
            "predictions": str(out_dir / "test_predictions.csv"),
            "operational_metrics": str(out_dir / "operational_metrics.json"),
            **fi_paths,
        },
    }
    _save_json(metrics_payload, out_dir / "metrics_multiclass_histgb_useful58_operational.json")

    flat = flatten_metrics_for_comparison(metrics_payload)
    pd.DataFrame([flat]).to_csv(out_dir / "metrics_flat.csv", index=False)
    summary_txt = make_summary_text(metrics_payload)
    _write_text(out_dir / "selected_model_summary.txt", summary_txt)

    print("\n=== TEST ===", flush=True)
    print(
        f"dataset={dataset} model={selected_model} acc={metrics.get('accuracy')} "
        f"balanced_acc={metrics.get('balanced_accuracy')} f1_macro={metrics.get('f1_macro')} "
        f"f1_weighted={metrics.get('f1_weighted')} mcc={metrics.get('mcc')}",
        flush=True,
    )
    print("\n[OK] Saved artifacts:", flush=True)
    for p in [
        processed_path,
        model_path,
        out_dir / "metrics_multiclass_histgb_useful58_operational.json",
        out_dir / "metrics_flat.csv",
        out_dir / "selected_model_summary.txt",
        out_dir / "confusion_matrix_test.csv",
        out_dir / "classification_report_test.csv",
        out_dir / "per_class_ovr_metrics.csv",
        out_dir / "test_predictions.csv",
        out_dir / "feature_importance.csv",
        out_dir / "operational_metrics.json",
    ]:
        if Path(p).exists():
            print(f"  {p}", flush=True)
    return metrics_payload


def flatten_metrics_for_comparison(m: Dict[str, Any]) -> Dict[str, Any]:
    tm = m.get("test_metrics") or {}
    bm = m.get("binary_normal_vs_attack_metrics") or {}
    om = m.get("operational_metrics") or {}
    timing = m.get("timing") or {}
    res = m.get("resource_usage") or {}
    return {
        "dataset": m.get("dataset"),
        "task": m.get("task"),
        "selected_model": m.get("selected_model"),
        "selected_model_family": m.get("selected_model_family"),
        "weight_mode": m.get("weight_mode"),
        "train_rows": m.get("train_rows"),
        "test_rows": m.get("test_rows"),
        "n_classes": m.get("n_classes"),
        "accuracy": tm.get("accuracy"),
        "balanced_accuracy": tm.get("balanced_accuracy"),
        "precision_micro": tm.get("precision_micro"),
        "precision_macro": tm.get("precision_macro"),
        "precision_weighted": tm.get("precision_weighted"),
        "recall_micro": tm.get("recall_micro"),
        "recall_macro": tm.get("recall_macro"),
        "recall_weighted": tm.get("recall_weighted"),
        "f1_micro": tm.get("f1_micro"),
        "f1_macro": tm.get("f1_macro"),
        "f1_weighted": tm.get("f1_weighted"),
        "mcc": tm.get("mcc"),
        "roc_auc_ovr_micro": tm.get("roc_auc_ovr_micro"),
        "roc_auc_ovr_macro": tm.get("roc_auc_ovr_macro"),
        "roc_auc_ovr_weighted": tm.get("roc_auc_ovr_weighted"),
        "pr_auc_micro": tm.get("pr_auc_micro"),
        "pr_auc_macro": tm.get("pr_auc_macro"),
        "pr_auc_weighted": tm.get("pr_auc_weighted"),
        "binary_accuracy": bm.get("accuracy"),
        "binary_precision": bm.get("precision"),
        "binary_recall_tpr": bm.get("recall_tpr"),
        "binary_specificity_tnr": bm.get("specificity_tnr"),
        "binary_fpr": bm.get("fpr"),
        "binary_fnr": bm.get("fnr"),
        "binary_f1": bm.get("f1"),
        "binary_mcc": bm.get("mcc"),
        "binary_roc_auc": bm.get("roc_auc"),
        "binary_pr_auc": bm.get("pr_auc"),
        "t_build_train_seconds": timing.get("t_build_train_seconds"),
        "t_apply_inference_batch_seconds": timing.get("t_apply_inference_batch_seconds"),
        "feature_extraction_seconds": timing.get("feature_extraction_seconds"),
        "latency_ms_p50_pipeline_online": om.get("latency_ms_pipeline_online_p50"),
        "latency_ms_p95_pipeline_online": om.get("latency_ms_pipeline_online_p95"),
        "latency_ms_p99_pipeline_online": om.get("latency_ms_pipeline_online_p99"),
        "throughput_req_s_pipeline_online": om.get("throughput_req_s_pipeline_online"),
        "throughput_req_s_pipeline_batch": om.get("throughput_req_s_pipeline_batch"),
        "throughput_req_s_inference_batch": om.get("throughput_req_s_inference_batch"),
        "cpu_time_pipeline_online_seconds": om.get("cpu_time_pipeline_online_seconds"),
        "cpu_utilization_pipeline_online_percent_of_all_cores": om.get("cpu_utilization_pipeline_online_percent_of_all_cores"),
        "peak_rss_mb": res.get("peak_rss_mb") or om.get("peak_rss_mb"),
        "current_rss_mb": res.get("current_rss_mb") or om.get("current_rss_mb"),
        "model_size_mb": res.get("model_size_mb") or om.get("model_size_mb"),
        "metrics_path": (m.get("artifacts") or {}).get("metrics"),
    }


def make_summary_text(m: Dict[str, Any]) -> str:
    tm = m.get("test_metrics") or {}
    bm = m.get("binary_normal_vs_attack_metrics") or {}
    om = m.get("operational_metrics") or {}
    lines = [
        "BEST_MODEL_12FEATURESRESULTS_MULTICLASE",
        f"dataset={m.get('dataset')}",
        f"task={m.get('task')}",
        f"selected_model={m.get('selected_model')}",
        f"selected_model_family={m.get('selected_model_family')}",
        f"weight_mode={m.get('weight_mode')}",
        f"feature_count={m.get('feature_count')}",
        f"train_rows={m.get('train_rows')}",
        f"test_rows={m.get('test_rows')}",
        f"accuracy={tm.get('accuracy')}",
        f"balanced_accuracy={tm.get('balanced_accuracy')}",
        f"precision_micro={tm.get('precision_micro')}",
        f"precision_macro={tm.get('precision_macro')}",
        f"precision_weighted={tm.get('precision_weighted')}",
        f"recall_micro={tm.get('recall_micro')}",
        f"recall_macro={tm.get('recall_macro')}",
        f"recall_weighted={tm.get('recall_weighted')}",
        f"f1_micro={tm.get('f1_micro')}",
        f"f1_macro={tm.get('f1_macro')}",
        f"f1_weighted={tm.get('f1_weighted')}",
        f"mcc={tm.get('mcc')}",
        f"roc_auc_ovr_macro={tm.get('roc_auc_ovr_macro')}",
        f"pr_auc_macro={tm.get('pr_auc_macro')}",
        f"binary_recall_tpr={bm.get('recall_tpr')}",
        f"binary_specificity_tnr={bm.get('specificity_tnr')}",
        f"binary_fpr={bm.get('fpr')}",
        f"binary_fnr={bm.get('fnr')}",
        f"latency_ms_p50_pipeline_online={om.get('latency_ms_pipeline_online_p50')}",
        f"latency_ms_p95_pipeline_online={om.get('latency_ms_pipeline_online_p95')}",
        f"latency_ms_p99_pipeline_online={om.get('latency_ms_pipeline_online_p99')}",
        f"throughput_req_s_pipeline_online={om.get('throughput_req_s_pipeline_online')}",
        f"model_size_mb={om.get('model_size_mb')}",
        f"gpu_used={m.get('gpu_used')}",
        f"gpu_note={m.get('gpu_note')}",
    ]
    return "\n".join(lines) + "\n"


def write_aggregate_outputs(args: argparse.Namespace, payloads: Sequence[Dict[str, Any]]) -> None:
    root = Path(args.definitivo_dir) / "multiclase"
    root.mkdir(parents=True, exist_ok=True)
    rows = [flatten_metrics_for_comparison(p) for p in payloads]
    df = pd.DataFrame(rows)
    df.to_csv(root / "model_comparison_multiclass_11claseResults.csv", index=False)
    _save_json({
        "created_at": _now_iso(),
        "task": "multiclass",
        "model_family": "hist_gradient_boosting",
        "feature_count": len(FIXED_FEATURES),
        "features": FIXED_FEATURES,
        "datasets": [p.get("dataset") for p in payloads],
        "rows": rows,
        "artifacts": {
            "comparison_csv": str(root / "model_comparison_multiclass_11claseResults.csv"),
            "comparison_json": str(root / "model_comparison_multiclass_11claseResults.json"),
        },
    }, root / "model_comparison_multiclass_11claseResults.json")
    _save_json({
        "created_at": _now_iso(),
        "root": str(Path(args.definitivo_dir)),
        "generated_task": "multiclase",
        "pending_task": "multietiqueta",
        "structure": {
            "multiclase": {
                p.get("dataset"): (p.get("artifacts") or {}) for p in payloads
            },
        },
    }, Path(args.definitivo_dir) / "manifest_11claseResults.json")
    print(f"[OK] Aggregate comparison: {root / 'model_comparison_multiclass_11claseResults.csv'}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="11claseResults multiclase: HistGradientBoosting con 58 features intra-request útiles; Harvard top 14 agrupado en 11 clases operativas por defecto")
    ap.add_argument("--dataset", default="both", choices=["torpeda", "harvard", "both"], help="Dataset a correr. Default: both.")
    ap.add_argument("--project-dir", default=DEFAULT_PROJECT_DIR)
    ap.add_argument("--definitivo-dir", "--result-dir", dest="definitivo_dir", default=None, help="Default: PROJECT_DIR/11claseResults")
    ap.add_argument("--inputs", nargs="*", default=None, help="Inputs para un dataset único. TorpEda: XML. Harvard: CSV/TSV/GZ.")
    ap.add_argument("--torpeda-inputs", nargs="*", default=None, help="Inputs TorpEda cuando --dataset both.")
    ap.add_argument("--harvard-inputs", nargs="*", default=None, help="Inputs Harvard/SR-BH cuando --dataset both.")
    ap.add_argument("--output-dir", default=None, help="Solo para --dataset torpeda/harvard: sobreescribe 11claseResults/multiclase/{dataset}.")

    # Dataset/preproceso.
    ap.add_argument("--sample-n", type=int, default=0, help="Muestreo head por archivo para pruebas rápidas. 0=todo.")
    ap.add_argument("--keep-absolute-uri", action="store_true")
    ap.add_argument("--torpeda-only-common-labels", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--torpeda-min-class-count", type=int, default=0)
    ap.add_argument("--harvard-min-class-count", type=int, default=20)
    ap.add_argument("--max-normal-rows", type=int, default=0, help="0=no downsample.")
    ap.add_argument("--keep-labels-regex", default="")
    ap.add_argument("--drop-labels-regex", default="")

    # Harvard/SR-BH CSV.
    ap.add_argument("--sep", default="auto")
    ap.add_argument("--method-col", default=RAW_METHOD_COL)
    ap.add_argument("--uri-col", default=RAW_URI_COL)
    ap.add_argument("--body-col", default=RAW_BODY_COL)
    ap.add_argument("--normal-col", default="000 - Normal")
    ap.add_argument("--label-cols", default="")
    ap.add_argument(
        "--harvard-label-mode",
        default="top11-operational-groups",
        choices=["top14-operational-groups", "top14-operational", "operational-top14", "operational-groups", "operational-group", "top11-operational", "top11-operational-groups", "top14-combo", "top14-combos", "top14-combination", "top14-combinations", "exact-top14-combo", "exact-top14-combos", "optimized-family", "optimized_family", "optimized", "coarse", "coarse-family", "coarse_family", "family", "mapped", "attack-family", "attack_families", "native", "capec", "srbh", "legacy-common-anomalous", "legacy_common_anomalous"],
    )
    ap.add_argument("--harvard-multiclass-strategy", default="severity", choices=["severity", "first"])

    # Train/test.
    ap.add_argument("--test-size", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--histgb-preset", default="strong", choices=["fast", "strong", "max"])
    ap.add_argument("--sample-weight-mode", default="", help="Default vacío: TorpEda=none, Harvard=sqrt_balanced.")
    ap.add_argument("--sample-weight-clip", type=float, default=20.0)
    ap.add_argument("--use-gpu", default="auto", choices=["auto", "on", "off"], help="Se registra pero HistGB de sklearn no usa CUDA.")

    # Paralelismo/operativas.
    ap.add_argument("--feature-workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    ap.add_argument("--feature-chunksize", type=int, default=256)
    ap.add_argument("--operational-max-rows", type=int, default=0, help="0=todo el test; >0 muestrea para latencia online.")

    # Feature importance.
    ap.add_argument("--fi-n-repeats", type=int, default=5)
    ap.add_argument("--fi-max-rows", type=int, default=10000, help="0=todo el test para FI.")
    ap.add_argument("--fi-scoring", default="f1_macro")
    ap.add_argument("--fi-n-jobs", type=int, default=1)
    ap.add_argument("--fi-min-rows-per-class", type=int, default=200, help="Mínimo de filas por clase dentro del muestreo usado para feature importance.")
    ap.add_argument("--fi-per-class", action=argparse.BooleanOptionalAction, default=True, help="Calcula permutation importance one-vs-rest por clase.")
    ap.add_argument("--fi-per-class-n-repeats", type=int, default=3)
    ap.add_argument("--fi-per-class-n-jobs", type=int, default=1)
    ap.add_argument("--fi-binary-normal-attack", action=argparse.BooleanOptionalAction, default=True, help="Calcula FI binaria derivada NORMAL vs ataque/anómalo.")
    ap.add_argument("--fi-binary-n-repeats", type=int, default=3)

    ap.add_argument("--clean-output-dir", action="store_true")
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args(argv)

    if args.definitivo_dir is None:
        args.definitivo_dir = str(Path(args.project_dir) / DEFAULT_RESULT_DIR_NAME)
    if args.dataset == "both" and args.output_dir:
        raise SystemExit("--output-dir solo se usa con --dataset torpeda o --dataset harvard; con both se usa 11claseResults/multiclase/{dataset}.")
    if not args.torpeda_inputs:
        args.torpeda_inputs = [os.environ.get("TORPEDA_RAW_DIR", str(Path(args.project_dir) / "data" / "raw" / "torpeda"))]
    if not args.harvard_inputs:
        args.harvard_inputs = [os.environ.get("HARVARD_RAW_DIR", str(Path(args.project_dir) / "data" / "raw" / "harvard"))]
    return args


def dependency_check() -> Dict[str, Any]:
    import importlib.util
    required = ["numpy", "pandas", "sklearn", "joblib", "tqdm"]
    missing = [p for p in required if importlib.util.find_spec(p) is None]
    parquet = importlib.util.find_spec("pyarrow") is not None or importlib.util.find_spec("fastparquet") is not None
    info = {"missing": missing, "parquet_available": bool(parquet), "python": sys.executable}
    if missing:
        raise SystemExit("Faltan dependencias base: " + ", ".join(missing))
    if not parquet:
        print("[WARN] pyarrow/fastparquet no detectado: processed se guardará como CSV fallback.", flush=True)
    return info


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    dep = dependency_check()
    Path(args.definitivo_dir).mkdir(parents=True, exist_ok=True)
    _save_json({"created_at": _now_iso(), "dependency_check": dep, "args": vars(args)}, Path(args.definitivo_dir) / "last_invocation.json")

    if args.check_only:
        print("[OK] CHECK_ONLY: dependencias y argumentos OK. No entreno.", flush=True)
        return 0

    datasets = ["torpeda", "harvard"] if args.dataset == "both" else [args.dataset]
    payloads: List[Dict[str, Any]] = []
    for ds in datasets:
        # Default del weight mode por dataset, si no lo fijó el usuario.
        original_weight_mode = args.sample_weight_mode
        if not original_weight_mode:
            args.sample_weight_mode = default_weight_mode_for_dataset(ds)
        payload = train_eval_one_dataset(args, ds)
        payloads.append(payload)
        args.sample_weight_mode = original_weight_mode
    write_aggregate_outputs(args, payloads)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
