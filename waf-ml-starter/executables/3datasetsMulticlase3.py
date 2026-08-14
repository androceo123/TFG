#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline multiclase unificado para TorpEda, Harvard/SR-BH y DS-Augmented-v2.

- Ejecuta el mismo vector compacto de 73 features, el mismo HistGradientBoosting y
  la misma selección de pesos dentro de outer-train en los tres datasets.
- Sólo cambian el loader, la taxonomía y el esquema esperado de cada dataset.
- En Egipcios conserva exactamente NORMAL + ocho ataques y separa CAPEC-88 de
  CAPEC-248; TorpEda y Harvard conservan sus taxonomías reducidas validadas.
- Usa HistGradientBoostingClassifier y un vector compacto de features densas.
- Poda las señales sin importancia positiva en la corrida previa y conserva como
  excepción la señal que sólo cobra sentido al separar CAPEC-88 de CAPEC-248.
- Mantiene equivalentes compactos de WAMM: longitud, word count, ratios/conteos de
  caracteres, entropía, profundidad, caracteres únicos y patrones por familia.
- No usa los 2.000 TF-IDF/n-grams de WAMM ni aprende dominios o endpoints.
- Las features usan sólo la request HTTP actual: método, URI/query, headers y body.
- Genera métricas globales, por clase, operativas y feature importance por clase.

Nota sobre GPU:
HistGradientBoostingClassifier de scikit-learn no tiene backend CUDA. Este script detecta GPU
para registrarlo en los metadatos, pero acelera este modelo usando CPU/OpenMP/BLAS y paralelismo
en extracción de features/permutation importance cuando corresponde.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import html
import json
import math
import os
import re
import shutil
import signal
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
    log_loss,
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
DEFAULT_RESULT_DIR_NAME = "3datasetsMulticlase3"
ARTIFACT_TAG = "compact73_unified_inner_weight_selection"
PIPELINE_RUN_VERSION = "three_datasets_compact73_inner_weight_selection_v1"
MODEL_FILENAME = f"model_multiclass_histgb_{ARTIFACT_TAG}.joblib"
METRICS_FILENAME = f"metrics_multiclass_histgb_{ARTIFACT_TAG}.json"
PROCESSED_PARQUET_FILENAME = f"processed_{ARTIFACT_TAG}.parquet"
PROCESSED_CSV_FILENAME = f"processed_{ARTIFACT_TAG}.csv"

# Dataset Egipcios/DS_Augmented_v2_csv: CSV con una request HTTP completa en col1
# y una etiqueta textual en col2, como en combined_data.csv de la captura.
EGIPCIOS_DEFAULT_REQUEST_COL = "col1"
EGIPCIOS_DEFAULT_LABEL_COL = "col2"
EGIPCIOS_DEFAULT_NORMAL_REGEX = r"^\s*0+\s*-\s*normal\s*$|\bnormal\b"
_HTTP_HEADER_NAME_RE = re.compile(r"^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$")

EGIPCIOS_PAPER_LABEL_SPECS: Dict[int, Tuple[str, str]] = {
    0: ("Normal", "NORMAL"),
    66: ("SQL Injection", "EGIPCIOS-SQLi"),
    88: ("OS Command Injection", "EGIPCIOS-OSCommandInjection"),
    126: ("Path Traversal", "EGIPCIOS-PathTraversal"),
    79: ("Cross-Site Scripting", "EGIPCIOS-XSS"),
    918: ("SSRF", "EGIPCIOS-SSRF"),
    248: ("Command Injection", "EGIPCIOS-CommandInjection"),
    1336: ("SSTI", "EGIPCIOS-SSTI"),
    94: ("Code Injection", "EGIPCIOS-CodeInjection"),
}
EGIPCIOS_EXPECTED_RAW_COUNTS: Dict[str, int] = {
    "000 - Normal": 493515,
    "66 - SQL Injection": 146086,
    "88 - OS Command Injection": 36114,
    "126 - Path Traversal": 17718,
    "79 - Cross-Site Scripting": 2894,
    "918 - SSRF": 2645,
    "248 - Command Injection": 2597,
    "1336 - SSTI": 1897,
    "94 - Code Injection": 1199,
}
EGIPCIOS_EXPECTED_CLASSES = tuple(sorted(spec[1] for spec in EGIPCIOS_PAPER_LABEL_SPECS.values()))
EGIPCIOS_EXPECTED_TOTAL_ROWS = int(sum(EGIPCIOS_EXPECTED_RAW_COUNTS.values()))
EGIPCIOS_EXPECTED_TEST_COUNTS: Dict[str, int] = {
    "NORMAL": 98703,
    "EGIPCIOS-SQLi": 29217,
    "EGIPCIOS-OSCommandInjection": 7223,
    "EGIPCIOS-PathTraversal": 3544,
    "EGIPCIOS-XSS": 579,
    "EGIPCIOS-SSRF": 529,
    "EGIPCIOS-CommandInjection": 519,
    "EGIPCIOS-SSTI": 379,
    "EGIPCIOS-CodeInjection": 240,
}
WAMM_PAPER_ACCURACY = 0.9959
WAMM_PAPER_F1_MACRO = 0.8607
SELECTION_REFERENCE_ACCURACY = 0.9959
SELECTION_REFERENCE_F1_MACRO = 0.8607
MODEL_WEIGHT_POWERS = (0.20, 0.33, 0.40, 0.50)
WAMM_PAPER_URL = "https://arxiv.org/abs/2512.23610"
WAMM_PATTERN_REFERENCE_URL = "https://gist.github.com/youssfqassim/04d20552a2a16b54f1572d0a8ec6fa11"
WAMM_PAPER_BLOCK_RATE: Dict[str, float] = {
    "EGIPCIOS-SQLi": 0.9996,
    "EGIPCIOS-OSCommandInjection": 0.9997,
    "EGIPCIOS-PathTraversal": 0.9800,
    "EGIPCIOS-XSS": 0.9948,
    "EGIPCIOS-SSRF": 0.9981,
    "EGIPCIOS-CommandInjection": 1.0000,
    "EGIPCIOS-SSTI": 1.0000,
    "EGIPCIOS-CodeInjection": 0.9625,
}

TORPEDA_EXPECTED_FULL_COUNTS: Dict[str, int] = {
    "NORMAL": 8363,
    "TORPEDA-ANOMALOUS": 16459,
    "TORPEDA-BufferOverflow": 412,
    "TORPEDA-CRLFi": 327,
    "TORPEDA-FormatString": 41,
    "TORPEDA-LDAPi": 74,
    "TORPEDA-SQLi": 43013,
    "TORPEDA-SSI": 451,
    "TORPEDA-XPath": 175,
    "TORPEDA-XSS": 4818,
}
TORPEDA_EXPECTED_TEST_COUNTS: Dict[str, int] = {
    "NORMAL": 1673,
    "TORPEDA-ANOMALOUS": 3292,
    "TORPEDA-BufferOverflow": 82,
    "TORPEDA-CRLFi": 65,
    "TORPEDA-FormatString": 8,
    "TORPEDA-LDAPi": 15,
    "TORPEDA-SQLi": 8603,
    "TORPEDA-SSI": 90,
    "TORPEDA-XPath": 35,
    "TORPEDA-XSS": 964,
}

HARVARD_EXPECTED_FULL_COUNTS: Dict[str, int] = {
    "NORMAL": 525195,
    "HARVARD-CAPEC66_SQLInjection": 248093,
    "HARVARD-RequestManipulation": 69814,
    "HARVARD-CAPEC34_HTTPResponseSplitting": 19134,
    "HARVARD-CAPEC126_PathTraversal": 17595,
    "HARVARD-CAPEC242_CodeInjection": 13793,
    "HARVARD-CAPEC88_OSCommandInjection": 3074,
    "HARVARD-CAPEC88_OSCommandInjection__CAPEC126_PathTraversal": 2464,
    "HARVARD-CAPEC310_VulnerabilityScanning": 2413,
    "HARVARD-CAPEC16_DictionaryPasswordAttack": 836,
}
HARVARD_EXPECTED_TEST_COUNTS: Dict[str, int] = {
    "NORMAL": 105039,
    "HARVARD-CAPEC66_SQLInjection": 49619,
    "HARVARD-RequestManipulation": 13963,
    "HARVARD-CAPEC34_HTTPResponseSplitting": 3827,
    "HARVARD-CAPEC126_PathTraversal": 3519,
    "HARVARD-CAPEC242_CodeInjection": 2759,
    "HARVARD-CAPEC88_OSCommandInjection": 615,
    "HARVARD-CAPEC88_OSCommandInjection__CAPEC126_PathTraversal": 493,
    "HARVARD-CAPEC310_VulnerabilityScanning": 482,
    "HARVARD-CAPEC16_DictionaryPasswordAttack": 167,
}

DATASET_SCHEMA_PROFILES: Dict[str, Dict[str, Any]] = {
    "torpeda": {
        "label_schema_version": "torpeda_native_common10_v1",
        "dataset_label_mode": "torpeda_native_common_names",
        "expected_full_counts": TORPEDA_EXPECTED_FULL_COUNTS,
        "expected_test_counts": TORPEDA_EXPECTED_TEST_COUNTS,
    },
    "harvard": {
        "label_schema_version": "harvard_top10_request_groups_v1",
        "dataset_label_mode": "top10-request-groups",
        "expected_full_counts": HARVARD_EXPECTED_FULL_COUNTS,
        "expected_test_counts": HARVARD_EXPECTED_TEST_COUNTS,
    },
    "egipcios": {
        "label_schema_version": "egipcios_paper_native9_v1",
        "dataset_label_mode": "egipcios_paper_native9",
        "expected_full_counts": {
            canonical: EGIPCIOS_EXPECTED_RAW_COUNTS[
                next(raw for raw in EGIPCIOS_EXPECTED_RAW_COUNTS if int(raw.split("-", 1)[0].strip()) == class_id)
            ]
            for class_id, (_, canonical) in EGIPCIOS_PAPER_LABEL_SPECS.items()
        },
        "expected_test_counts": EGIPCIOS_EXPECTED_TEST_COUNTS,
    },
}

COMMON_ATTACKS = ["BufferOverflow", "CRLFi", "FormatString", "LDAPi", "SQLi", "SSI", "XPath", "XSS"]

# Set candidato original de 84 features numéricas intra-request.
# De estas se conservan 58; se suman 14 features manuales previas y 21 contextuales, para un total de 93.
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

# Selección base derivada de la feature importance por clase de la corrida3 con 84 features.
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

# Bloque manual incorporado en la corrida anterior. Se mantiene para conservar
# las mejoras ya observadas y facilitar una comparación directa de la nueva corrida.
PAPER_GAP_FEATURES: List[str] = [
    # Estadísticas de la request HTTP completa.
    "full_request_len",
    "full_request_entropy",
    "full_request_special_char_ratio",
    "full_request_digit_ratio",
    "full_request_unique_char_count",
    "full_request_unique_char_ratio",

    # Patrones generales de familias de ataque.
    "ssti_delimiter_count",
    "ssti_keyword_count",
    "ssrf_scheme_count",
    "ssrf_internal_target_count",
    "shell_operator_count",
    "command_keyword_count",
    "code_exec_function_count",
    "xss_dom_event_count",
]

# Señales v3: todas se calculan únicamente con el mensaje HTTP actual. No usan
# hostname esperado, endpoint conocido, frecuencia histórica, sesión, IP cliente,
# usuario, tiempo, reputación ni información externa al request.
CONTEXTUAL_HTTP_FEATURES: List[str] = [
    # Obfuscación y anomalía local: evitan diluir un payload corto en requests grandes.
    "decode_reduction_ratio",
    "max_uri_atom_special_ratio",
    "max_header_atom_special_ratio",
    "max_body_atom_special_ratio",
    "max_atom_entropy",

    # SSTI estructural, incluyendo expresiones completas y probes polyglot.
    "ssti_complete_expression_count",
    "ssti_arithmetic_expression_count",
    "ssti_object_chain_expression_count",
    "ssti_polyglot_count",
    "ssti_first_decode_level",

    # XSS estructural, separado de SSTI y de simples caracteres especiales.
    "xss_dangerous_tag_count",
    "xss_tag_event_pair_count",
    "xss_script_block_count",
    "xss_js_execution_count",
    "xss_first_decode_level",

    # Márgenes explícitos SSTI frente a XSS para la frontera más difícil.
    "ssti_without_xss_structure_count",
    "xss_without_ssti_structure_count",

    # SSRF contextual: URL embebida en target/valor y referencias XML externas.
    "ssrf_url_in_request_target_count",
    "ssrf_url_in_value_count",
    "ssrf_xml_external_reference_count",

    # Command injection: operador y comando en una misma vecindad local.
    "shell_operator_command_pair_count",
]

FIXED_FEATURES += PAPER_GAP_FEATURES + CONTEXTUAL_HTTP_FEATURES
LEGACY_93_FEATURES: List[str] = list(FIXED_FEATURES)

# En la corrida anterior estas 25 variables no tuvieron importancia de permutación
# positiva ni global ni para ninguna clase de Egipcios. Se podan; la antigua
# shell_operator_command_pair_count se conserva deliberadamente porque CAPEC-88 y
# CAPEC-248 estaban fusionadas y, por tanto, aquella evaluación no podía medir su
# utilidad para separar ambas clases.
EGIPCIOS_PRIOR_DROPPED_FEATURES: List[str] = [
    "common_scanner_ua_token_count",
    "empty_param_value_count",
    "method_is_get",
    "method_is_head",
    "method_is_options",
    "method_is_other",
    "method_is_trace",
    "n_body_params",
    "n_query_params",
    "param_name_entropy",
    "sensitive_param_name_count",
    "ssrf_url_in_value_count",
    "ssti_arithmetic_expression_count",
    "ssti_keyword_count",
    "ssti_object_chain_expression_count",
    "ssti_polyglot_count",
    "ssti_without_xss_structure_count",
    "suspicious_file_extension_count",
    "xss_dangerous_tag_count",
    "xss_dom_event_count",
    "xss_first_decode_level",
    "xss_js_execution_count",
    "xss_script_block_count",
    "xss_tag_event_pair_count",
    "xss_without_ssti_structure_count",
]

# Cinco señales compactas: una estadística publicada por WAMM que faltaba de forma
# explícita y cuatro separadores interpretables de command injection. No se añaden
# TF-IDF ni miles de n-grams.
EGIPCIOS_COMPACT_ADDED_FEATURES: List[str] = [
    "full_request_word_count",
    "unix_command_count",
    "windows_command_count",
    "shell_wrapper_count",
    "command_substitution_count",
]

FIXED_FEATURES = [
    f for f in LEGACY_93_FEATURES if f not in set(EGIPCIOS_PRIOR_DROPPED_FEATURES)
] + EGIPCIOS_COMPACT_ADDED_FEATURES

LEGACY_CANDIDATE_DROPPED_FEATURES: List[str] = [
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

ALL_CANDIDATE_FEATURES: List[str] = list(dict.fromkeys(
    ALL_84_CANDIDATE_FEATURES
    + PAPER_GAP_FEATURES
    + CONTEXTUAL_HTTP_FEATURES
    + EGIPCIOS_COMPACT_ADDED_FEATURES
))
FI_DROPPED_FEATURES: List[str] = list(dict.fromkeys(
    LEGACY_CANDIDATE_DROPPED_FEATURES + EGIPCIOS_PRIOR_DROPPED_FEATURES
))

FEATURE_POLICY_NAME = "compact73_cross_dataset"
FEATURE_SET_VERSION = "compact73_cross_dataset_v1"
LABEL_SCHEMA_VERSION = "dataset_specific_native_schemas_v1"

FEATURE_GROUP_BY_NAME: Dict[str, str] = {**{f: "baseline30" for f in BASE_30_FEATURES}}
FEATURE_GROUP_BY_NAME.update({
    **{f: "http_method" for f in EXTENDED_54_FEATURES[0:12]},
    **{f: "http_headers_protocol" for f in EXTENDED_54_FEATURES[12:26]},
    **{f: "body_params_structure" for f in EXTENDED_54_FEATURES[26:38]},
    **{f: "xpath_xml" for f in EXTENDED_54_FEATURES[38:45]},
    **{f: "scanning_paths" for f in EXTENDED_54_FEATURES[45:54]},
})
FEATURE_GROUP_BY_NAME.update({
    **{f: "wamm_global_request_stats" for f in PAPER_GAP_FEATURES[0:6]},
    **{f: "ssti" for f in PAPER_GAP_FEATURES[6:8]},
    **{f: "ssrf" for f in PAPER_GAP_FEATURES[8:10]},
    **{f: "command_code_injection" for f in PAPER_GAP_FEATURES[10:13]},
    **{f: "xss_specific" for f in PAPER_GAP_FEATURES[13:14]},
})
FEATURE_GROUP_BY_NAME.update({
    **{f: "local_request_statistics" for f in CONTEXTUAL_HTTP_FEATURES[0:5]},
    **{f: "ssti_contextual" for f in CONTEXTUAL_HTTP_FEATURES[5:10]},
    **{f: "xss_contextual" for f in CONTEXTUAL_HTTP_FEATURES[10:15]},
    **{f: "ssti_xss_contrast" for f in CONTEXTUAL_HTTP_FEATURES[15:17]},
    **{f: "ssrf_contextual" for f in CONTEXTUAL_HTTP_FEATURES[17:20]},
    **{f: "command_contextual" for f in CONTEXTUAL_HTTP_FEATURES[20:21]},
})
FEATURE_GROUP_BY_NAME.update({
    "full_request_word_count": "wamm_global_request_stats",
    "unix_command_count": "capec88_248_separator",
    "windows_command_count": "capec88_248_separator",
    "shell_wrapper_count": "capec88_248_separator",
    "command_substitution_count": "capec88_248_separator",
})

WAMM_EQUIVALENT_FEATURES = {
    "full_request_len", "full_request_word_count", "full_request_entropy",
    "full_request_special_char_ratio", "full_request_digit_ratio",
    "full_request_unique_char_count", "full_request_unique_char_ratio",
    "path_depth", "quote_count", "angle_bracket_count", "semicolon_count",
    "percent_count", "uri_encoded_ratio", "body_encoded_ratio",
    "sql_token_count", "xss_token_count", "ssti_delimiter_count",
    "ssrf_scheme_count", "ssrf_internal_target_count", "path_dot_segment_count",
    "path_encoded_slash_count", "shell_operator_count", "command_keyword_count",
    "code_exec_function_count", "shell_operator_command_pair_count",
    "unix_command_count", "windows_command_count", "shell_wrapper_count",
    "command_substitution_count",
}

if len(LEGACY_93_FEATURES) != 93 or len(FIXED_FEATURES) != 73:
    raise RuntimeError(
        f"Inventario de features inconsistente: legacy={len(LEGACY_93_FEATURES)}, "
        f"selected={len(FIXED_FEATURES)}; esperado 93/73."
    )
if len(set(FIXED_FEATURES)) != len(FIXED_FEATURES):
    raise RuntimeError("FIXED_FEATURES contiene duplicados.")
if set(FIXED_FEATURES) | set(FI_DROPPED_FEATURES) != set(ALL_CANDIDATE_FEATURES):
    raise RuntimeError("El manifiesto selected/dropped no cubre todos los candidatos.")

_HTTP_PROTO_RE = re.compile(r"^HTTP/(\d+(?:\.\d+)?)$", re.IGNORECASE)
_HEX = r"[0-9a-fA-F]"
_PCT_ENC_RE = re.compile(rf"%{_HEX}{{2}}")
_INVALID_PERCENT_RE = re.compile(rf"%(?!{_HEX}{{2}})")
_DOUBLE_ENC_RE = re.compile(rf"%25{_HEX}{{2}}", re.IGNORECASE)
_TOKEN_RE = re.compile(r"[A-Za-z0-9_%\\x\\u./:-]+")
_LABEL_COL_RE = re.compile(r"^\d{1,4}\s+-\s+.+$")
_ID_NAME_RE = re.compile(r"^(\d{1,4})\s+-\s+(.+)$")
_CANON_CAPEC_RE = re.compile(r"\bCAPEC-(\d{1,4})\b")
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
        r"\(\s*[&|!]", r"\(\s*[a-z][a-z0-9_-]*\s*=\s*\*?[^()\r\n]*\)",
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
        r"%(?:[0-9]{1,12}\$)?[-+#0 ]{0,16}(?:(?:\d{1,20}|\*)?(?:\.(?:\d{1,20}|\*))?)?(?:hh|ll|[hlLzjt])?[diuoxXfFeEgGaAcspn]",
        r"%n", r"%x", r"%p", r"%s", r"%08x", r"%hn", r"%hhn", r"\{\d+\}", r"\{\}",
    ]),
    "buffer_overflow": _compile([
        r"A{32,}", r"B{32,}", r"C{32,}", r"D{32,}", r"0{32,}", r"1{32,}", r"[A-Za-z0-9]{128,}",
        r"%u9090", r"%u0c0c", r"\\x90", r"\x90", r"nop\s+sled", r"shellcode",
    ]),
    "ssti_delimiter": _compile([
        r"\{\{", r"\}\}", r"\{%", r"%\}", r"\$\{", r"#\{", r"<%=", r"<%[-=]?",
    ]),
    "ssti_keyword": _compile([
        r"\b(?:jinja2?|twig|freemarker|velocity|smarty|mako|handlebars|mustache)\b",
        r"__(?:class|mro|subclasses|globals|builtins|base|init)__",
        r"\b(?:config|request|self|cycler|joiner|namespace|lipsum)\s*(?:\.|\[)",
        r"\bget_flashed_messages\b", r"\bapplication\s*\.\s*__globals__\b",
    ]),
    "ssrf_scheme": _compile([
        r"\b(?:https?|file|gopher|ftp|dict|ldap|sftp|tftp|jar|netdoc|phar|expect)://",
    ]),
    "ssrf_internal_target": _compile([
        r"\b(?:localhost|localhost\.localdomain|metadata\.google\.internal|metadata\.azure\.com|instance-data)\b",
        r"\b(?:127(?:\.\d{1,3}){3}|0\.0\.0\.0|169\.254\.(?:169\.254|170\.2)|10(?:\.\d{1,3}){3}|192\.168(?:\.\d{1,3}){2}|172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2})\b",
        r"(?:\[?::1\]?|0:0:0:0:0:0:0:1)", r"\b(?:2130706433|0x7f000001|017700000001)\b",
        r"\b(?:latest/meta-data|computeMetadata/v1|metadata/instance)\b",
    ]),
    "shell_operator": _compile([
        r"&&", r"\|\|", r"(?<!\|)\|(?!\|)", r"`[^`]{0,512}`", r"\$\(", r"\$\{IFS[^{}$\r\n]{0,512}\}",
        r";\s*(?=(?:/[^\s;]+|[A-Za-z][A-Za-z0-9_.-]*)\b)",
    ]),
    "command_keyword": _compile([
        r"\b(?:cmd(?:\.exe)?|powershell(?:\.exe)?|pwsh|bash|zsh|ksh|csh|netcat|ncat|nc|curl|wget|whoami|uname|chmod|chown|ifconfig|ipconfig|netstat|tasklist|certutil|bitsadmin|nslookup)\b",
        r"(?:^|[\s;&|`$()])(?:/bin/|/usr/bin/|/usr/sbin/)?(?:sh|cat|id|ps|ping|dig|env|printenv)(?=$|[\s;&|`$()])",
    ]),
    # Separadores compactos CAPEC-88 / CAPEC-248, inspirados en los patrones
    # RCE y Command_Injection publicados con WAMM. Se mantienen genéricos y no
    # contienen dominios, endpoints ni tokens específicos del dataset.
    "unix_command": _compile([
        r"(?:^|[\s;&|`$()])(?:/bin/|/usr/bin/|/usr/sbin/)?(?:sh|bash|dash|zsh|ksh|csh|cat|id|whoami|uname|pwd|ls|ps|env|printenv|chmod|chown|curl|wget|nc|netcat|ncat|ping|nslookup|dig)(?=$|[\s;&|`$()])",
    ]),
    "windows_command": _compile([
        r"\b(?:cmd(?:\.exe)?|powershell(?:\.exe)?|pwsh(?:\.exe)?|whoami(?:\.exe)?|ipconfig(?:\.exe)?|tasklist(?:\.exe)?|certutil(?:\.exe)?|bitsadmin(?:\.exe)?|netstat(?:\.exe)?|nslookup(?:\.exe)?)\b",
    ]),
    "shell_wrapper": _compile([
        r"(?:^|[\s;&|`$()])(?:/bin/|/usr/bin/)?(?:sh|bash|dash|zsh|ksh|csh)\s+-[a-z]{0,8}c\b",
        r"\bcmd(?:\.exe)?\s*/[ck]\b",
        r"\b(?:powershell|pwsh)(?:\.exe)?\b[^\r\n]{0,128}\s-(?:c|command|enc|encodedcommand)\b",
        r"\b(?:os\.(?:system|popen)|subprocess\.(?:run|call|popen|check_output)|child_process\.(?:exec|spawn))\s*\(",
    ]),
    "command_substitution": _compile([
        r"\$\([^()\r\n]{0,512}\)",
        r"`[^`\r\n]{1,512}`",
        r"\$\{IFS[^{}$\r\n]{0,128}\}",
    ]),
    "code_exec_function": _compile([
        r"\b(?:eval|exec|system|popen|passthru|shell_exec|proc_open|assert)\s*\(",
        r"\b(?:os\.system|os\.popen|subprocess\.(?:run|call|Popen|check_output)|child_process\.(?:exec|spawn))\s*\(",
        r"Runtime\s*\.\s*getRuntime\s*\(\s*\)\s*\.\s*exec\s*\(", r"\bnew\s+ProcessBuilder\s*\(",
    ]),
    "xss_dom_event": _compile([
        r"\bon(?:error|load|click|mouseover|focus|blur|input|change|submit|animationstart|begin|toggle|pointerover)\s*=",
        r"\b(?:javascript|vbscript)\s*:", r"data\s*:\s*text/html",
        r"\b(?:document|window)\s*\.\s*(?:cookie|location|write|writeln|domain)\b",
        r"\b(?:innerHTML|outerHTML|srcdoc|fromCharCode)\b", r"\b(?:alert|prompt|confirm)\s*\(",
    ]),

    # Patrones contextuales v3. Todos están acotados para evitar backtracking
    # patológico sobre cookies, bodies o headers extensos.
    "ssti_complete_expression": _compile([
        r"\{\{[^{}\r\n]{0,512}\}\}",
        r"\{%[^{}\r\n]{0,512}%\}",
        r"\$\{[^{}\r\n]{0,512}\}",
        r"#\{[^{}\r\n]{0,512}\}",
        r"<%=[^<\r\n]{0,512}%>",
    ]),
    "ssti_arithmetic_expression": _compile([
        r"\{\{\s*[-+]?\d+(?:\.\d+)?\s*[*+\-/]\s*[-+]?\d+(?:\.\d+)?\s*\}\}",
        r"\$\{\s*[-+]?\d+(?:\.\d+)?\s*[*+\-/]\s*[-+]?\d+(?:\.\d+)?\s*\}",
        r"#\{\s*[-+]?\d+(?:\.\d+)?\s*[*+\-/]\s*[-+]?\d+(?:\.\d+)?\s*\}",
        r"<%=\s*[-+]?\d+(?:\.\d+)?\s*[*+\-/]\s*[-+]?\d+(?:\.\d+)?\s*%>",
    ]),
    "ssti_object_chain_expression": _compile([
        r"(?:\{\{|\{%|\$\{|#\{|<%=)[^{}\r\n]{0,512}(?:__(?:class|mro|subclasses|globals|builtins|base|init)__|class\.forname|getclass\s*\(|getruntime\s*\(|runtime\.getruntime|processbuilder|\b(?:config|request|self|cycler|joiner|namespace|lipsum)\s*(?:\.|\[))[^{}\r\n]{0,128}(?:\}\}|%\}|\}|%>)",
    ]),
    "ssti_polyglot": _compile([
        # Antes, el separador también podía consumir comillas/backticks/barras y
        # cada repetición admitía muchas particiones equivalentes. Eso provocaba
        # backtracking exponencial. Estas variantes son deterministas y acotadas.
        r"(?:['\"`\\][^'\"`\\{$#\r\n]{0,8}){3,32}(?:\{\{|\{%|\$\{|#\{)",
        r"(?:\{\{|\{%|\$\{|#\{)(?:[^'\"`\\\r\n]{0,8}['\"`\\]){3,32}",
        r"(?:`z['\"]z[\"']|z['\"]z[\"'])[^\r\n]{0,32}(?:\{\{|\{%|\$\{|#\{)",
    ]),
    "xss_dangerous_tag": _compile([
        r"<\s*/?\s*(?:script|img|svg|iframe|body|input|meta|object|embed|link|style|video|audio|math|form|details|marquee|base)\b",
    ]),
    "xss_tag_event_pair": _compile([
        r"<[^>\r\n]{0,512}\bon[a-z]{3,32}\s*=",
    ]),
    "xss_script_block": _compile([
        r"<\s*script\b[^>\r\n]{0,256}>", r"<\s*/\s*script\s*>",
    ]),
    "xss_js_execution": _compile([
        r"\b(?:alert|prompt|confirm|eval|settimeout|setinterval|atob|btoa|fromcharcode)\s*\(",
        r"\b(?:document\.write|window\.open)\s*\(",
    ]),
    "ssrf_xml_external_reference": _compile([
        r"(?:xsi\s*:\s*schemaLocation|schemaLocation|xi\s*:\s*include|xinclude|xlink\s*:\s*href|href|src)\s*=\s*['\"]?\s*(?:https?|ftp|file|gopher|dict|ldap)://",
    ]),
    "shell_operator_command_pair": _compile([
        r"(?:&&|\|\||(?<!\|)\|(?!\|)|;|`|\$\()\s*(?:/bin/|/usr/bin/|/usr/sbin/)?(?:sh|bash|dash|zsh|ksh|cat|id|whoami|uname|pwd|ls|chmod|chown|curl|wget|nc|netcat|ncat|ping|nslookup|dig|cmd(?:\.exe)?|powershell(?:\.exe)?|pwsh)\b",
    ]),
}

SSTI_CONTEXT_PATTERNS: List[re.Pattern[str]] = (
    PATTERNS["ssti_complete_expression"]
    + PATTERNS["ssti_arithmetic_expression"]
    + PATTERNS["ssti_object_chain_expression"]
    + PATTERNS["ssti_polyglot"]
)
XSS_CONTEXT_PATTERNS: List[re.Pattern[str]] = (
    PATTERNS["xss_dangerous_tag"]
    + PATTERNS["xss_tag_event_pair"]
    + PATTERNS["xss_script_block"]
    + PATTERNS["xss_js_execution"]
)

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
# Incluye NORMAL y dos combinaciones multiclase explícitas. Las filas que no
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
    # finditer evita construir listas potencialmente enormes y conserva el
    # mismo conteo de coincidencias no solapadas que findall.
    s = text or ""
    return int(sum(1 for p in patterns for _ in p.finditer(s)))


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
    tmp = path.with_name(f".{path.name}.tmp_{os.getpid()}")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=_json_default)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
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


def _json_shape_stats(value: str, max_nodes: int = 100000) -> Tuple[int, int, int]:
    s = _safe_str(value).strip()
    if not s:
        return 0, 0, 0
    try:
        obj = json.loads(s)
    except Exception:
        return 0, 0, 0

    # Recorrido iterativo: un JSON artificialmente profundo no debe desbordar
    # la pila ni detener un worker. max_nodes es sólo un fusible de seguridad.
    key_count = 0
    max_depth = 1
    visited = 0
    stack: List[Tuple[Any, int]] = [(obj, 1)]
    while stack and visited < max(1, int(max_nodes)):
        current, depth = stack.pop()
        visited += 1
        max_depth = max(max_depth, depth)
        if isinstance(current, dict):
            key_count += len(current)
            stack.extend((child, depth + 1) for child in current.values())
        elif isinstance(current, list):
            stack.extend((child, depth + 1) for child in current)
    return 1, int(key_count), int(max_depth)


def _token_count_from_set(text: str, vocabulary: set[str]) -> int:
    s = _safe_str(text).lower()
    if not s:
        return 0
    return int(sum(1 for tok in vocabulary if tok and tok in s))


def _path_segments(path_text: str) -> List[str]:
    return [p for p in re.split(r"/+", _safe_str(path_text)) if p != ""]


_LOCAL_ATOM_RE = re.compile(r"[^\s&;,]{4,256}")


def _decode_levels(text: str) -> Tuple[str, str, str]:
    """Raw, una y dos decodificaciones; siempre devuelve tres niveles."""
    raw = _safe_str(text)
    once = _decode_once(raw)
    twice = _decode_once(once)
    return raw, once, twice


def _max_pattern_count_levels(levels: Sequence[str], patterns: Sequence[re.Pattern[str]]) -> int:
    """Máximo conteo entre niveles, sin triplicar el mismo payload al decodificar."""
    return int(max((_count_patterns(t, patterns) for t in dict.fromkeys(levels)), default=0))


def _first_pattern_level(levels: Sequence[str], patterns: Sequence[re.Pattern[str]]) -> int:
    """0=ausente, 1=raw, 2=aparece tras una decodificación, 3=tras dos."""
    previous = None
    for idx, text in enumerate(levels, start=1):
        if idx > 1 and text == previous:
            previous = text
            continue
        if _count_patterns(text, patterns) > 0:
            return int(idx)
        previous = text
    return 0


def _max_atom_statistics(text: str, max_atoms: int = 512) -> Tuple[float, float]:
    """Máxima densidad de símbolos y entropía en átomos locales del componente.

    Se limita el tamaño y número de átomos para mantener coste lineal y estable en
    mensajes HTTP grandes. No se utiliza vocabulario del aplicativo.
    """
    max_special = 0.0
    max_entropy = 0.0
    seen = 0
    raw, _, decoded = _decode_levels(text)
    for variant in dict.fromkeys((raw, decoded)):
        for match in _LOCAL_ATOM_RE.finditer(variant):
            atom = match.group(0)
            if not atom:
                continue
            n = len(atom)
            special = sum(1 for ch in atom if not ch.isalnum() and not ch.isspace())
            ratio = _ratio(special, n)
            if ratio > max_special:
                max_special = float(ratio)
            # Entropía local útil sobre payloads compactos; se evita calcularla
            # para tokens largos y totalmente alfanuméricos.
            if special > 0 or n <= 64:
                ent = _entropy(atom)
                if ent > max_entropy:
                    max_entropy = float(ent)
            seen += 1
            if seen >= int(max_atoms):
                break
        if seen >= int(max_atoms):
            break
    return float(max_special), float(max_entropy)


def _embedded_url_count_in_target(levels: Sequence[str]) -> int:
    """Cuenta esquemas URL embebidos dentro del request-target.

    Una absolute-form URI al inicio (uso proxy) no se considera embebida por sí sola;
    sí cuentan URLs posteriores dentro del path/query.
    """
    best = 0
    for text in dict.fromkeys(levels):
        count = 0
        for pattern in PATTERNS["ssrf_scheme"]:
            for match in pattern.finditer(text or ""):
                if match.start() > 0:
                    count += 1
        best = max(best, count)
    return int(best)


def _url_count_in_values(values: Sequence[str]) -> int:
    total = 0
    for value in list(values)[:256]:
        total += _max_pattern_count_levels(_decode_levels(value), PATTERNS["ssrf_scheme"])
    return int(min(total, 1024))


def feature_definitions_frame() -> pd.DataFrame:
    rows = []
    for i, f in enumerate(FIXED_FEATURES, start=1):
        if f == "full_request_word_count":
            origin = "paper_equivalent_added"
        elif f in set(EGIPCIOS_COMPACT_ADDED_FEATURES):
            origin = "capec88_248_separator_added"
        elif f == "shell_operator_command_pair_count":
            origin = "capec88_248_separator_retained"
        else:
            origin = "prior_run_positive_fi"
        rows.append({
            "feature_index": i,
            "feature": f,
            "group": FEATURE_GROUP_BY_NAME.get(f, "unknown"),
            "feature_set": FEATURE_SET_VERSION,
            "intra_request_only": True,
            "selection_origin": origin,
            "wamm_equivalent": f in WAMM_EQUIVALENT_FEATURES,
        })
    return pd.DataFrame(rows)


def feature_selection_manifest_frame() -> pd.DataFrame:
    selected = set(FIXED_FEATURES)
    prior_dropped = set(EGIPCIOS_PRIOR_DROPPED_FEATURES)
    legacy_dropped = set(LEGACY_CANDIDATE_DROPPED_FEATURES)
    added = set(EGIPCIOS_COMPACT_ADDED_FEATURES)
    rows: List[Dict[str, Any]] = []
    for feature in ALL_CANDIDATE_FEATURES:
        if feature in selected:
            if feature == "full_request_word_count":
                origin = "paper_equivalent_added"
                reason = "Estadística explícita publicada por WAMM que faltaba en el vector anterior."
            elif feature in added:
                origin = "capec88_248_separator_added"
                reason = "Señal interpretable añadida para separar CAPEC-88 de CAPEC-248."
            elif feature == "shell_operator_command_pair_count":
                origin = "capec88_248_separator_retained"
                reason = "Excepción: la corrida previa fusionaba las dos clases y no podía medir esta frontera."
            else:
                origin = "prior_run_positive_fi"
                reason = "Importancia de permutación positiva global o para alguna clase en la corrida previa."
            status = "selected"
        elif feature in prior_dropped:
            origin = "prior_run_nonpositive_fi"
            reason = "Sin importancia de permutación positiva global ni por clase en Egipcios previo."
            status = "dropped"
        elif feature in legacy_dropped:
            origin = "earlier_cross_dataset_nonpositive_fi"
            reason = "Descartada en la selección histórica por importancia no positiva."
            status = "dropped"
        else:
            origin = "unclassified"
            reason = "Candidato no seleccionado."
            status = "dropped"
        rows.append({
            "feature": feature,
            "status": status,
            "selected": feature in selected,
            "group": FEATURE_GROUP_BY_NAME.get(feature, "unknown"),
            "selection_origin": origin,
            "reason": reason,
            "wamm_equivalent": feature in WAMM_EQUIVALENT_FEATURES,
            "feature_set": FEATURE_SET_VERSION,
        })
    out = pd.DataFrame(rows)
    if int(out["selected"].sum()) != len(FIXED_FEATURES) or len(out) != len(ALL_CANDIDATE_FEATURES):
        raise RuntimeError("feature_selection_manifest.csv no coincide con el inventario configurado.")
    return out


def extract_fixed_features(method: str, uri: str, headers: Optional[Dict[str, str]] = None, body: Any = None) -> Dict[str, float]:
    """Devuelve las 73 features compactas intra-request en el orden fijado."""

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
    full_request_raw = f"{method_norm} {uri_raw}\n{headers_txt}\n\n{body_raw}"
    request_variants = [
        full_request_raw,
        f"{method_norm} {uri_dec1}\n{headers_dec1}\n\n{body_dec1}",
        f"{method_norm} {uri_dec2}\n{headers_dec2}\n\n{body_dec2}",
    ]
    paper_analysis_text = "\n".join(dict.fromkeys(request_variants))

    try:
        parts = urlsplit(uri_raw)
        path = parts.path or ""
        query = parts.query or (uri_raw.split("?", 1)[1] if "?" in uri_raw else "")
    except (ValueError, UnicodeError):
        # Una URI inválida no debe abortar toda la corrida; se conserva el texto
        # y se separa query de forma literal como fallback reproducible.
        path, marker, query = uri_raw.partition("?")
        if not marker:
            query = ""
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
    full_request_len = len(full_request_raw)
    full_request_unique_char_count = len(set(full_request_raw)) if full_request_raw else 0
    full_request_special_char_count = sum(1 for ch in full_request_raw if not ch.isalnum() and not ch.isspace())
    full_request_digit_count = sum(1 for ch in full_request_raw if ch.isdigit())
    full_request_word_count = len(re.findall(r"[A-Za-z0-9_]+", full_request_raw))

    # Señales contextuales v3. Se usa el máximo entre raw/decoded para no
    # triplicar coincidencias y se conserva el primer nivel donde aparece el patrón.
    full_levels = _decode_levels(full_request_raw)
    uri_levels = _decode_levels(uri_raw)
    body_levels = _decode_levels(body_raw)
    headers_levels = _decode_levels(headers_txt)

    final_decoded_len = len(full_levels[-1])
    decode_reduction_ratio = _ratio(max(0, full_request_len - final_decoded_len), full_request_len)

    max_uri_atom_special_ratio, uri_atom_entropy = _max_atom_statistics(uri_raw, max_atoms=256)
    max_header_atom_special_ratio, header_atom_entropy = _max_atom_statistics(headers_txt, max_atoms=512)
    max_body_atom_special_ratio, body_atom_local_entropy = _max_atom_statistics(body_raw, max_atoms=512)
    max_atom_entropy = max(uri_atom_entropy, header_atom_entropy, body_atom_local_entropy)

    ssti_complete_expression_count = _max_pattern_count_levels(full_levels, PATTERNS["ssti_complete_expression"])
    ssti_arithmetic_expression_count = _max_pattern_count_levels(full_levels, PATTERNS["ssti_arithmetic_expression"])
    ssti_object_chain_expression_count = _max_pattern_count_levels(full_levels, PATTERNS["ssti_object_chain_expression"])
    ssti_polyglot_count = _max_pattern_count_levels(full_levels, PATTERNS["ssti_polyglot"])
    ssti_first_decode_level = _first_pattern_level(full_levels, SSTI_CONTEXT_PATTERNS)

    xss_dangerous_tag_count = _max_pattern_count_levels(full_levels, PATTERNS["xss_dangerous_tag"])
    xss_tag_event_pair_count = _max_pattern_count_levels(full_levels, PATTERNS["xss_tag_event_pair"])
    xss_script_block_count = _max_pattern_count_levels(full_levels, PATTERNS["xss_script_block"])
    xss_js_execution_count = _max_pattern_count_levels(full_levels, PATTERNS["xss_js_execution"])
    xss_first_decode_level = _first_pattern_level(full_levels, XSS_CONTEXT_PATTERNS)

    ssti_structure_score = (
        ssti_complete_expression_count
        + 2 * ssti_arithmetic_expression_count
        + 2 * ssti_object_chain_expression_count
        + 2 * ssti_polyglot_count
    )
    xss_structure_score = (
        xss_dangerous_tag_count
        + 2 * xss_tag_event_pair_count
        + 2 * xss_script_block_count
        + xss_js_execution_count
    )
    ssti_without_xss_structure_count = max(0, ssti_structure_score - xss_structure_score)
    xss_without_ssti_structure_count = max(0, xss_structure_score - ssti_structure_score)

    ssrf_url_in_request_target_count = _embedded_url_count_in_target(uri_levels)
    # Query/form values son genéricos; cuando el body no es form se analiza como un
    # único valor para cubrir JSON, XML y texto sin conocer su esquema de negocio.
    ssrf_values = list(values)
    if body_raw and not b_params:
        ssrf_values.append(body_raw)
    ssrf_url_in_value_count = _url_count_in_values(ssrf_values)
    ssrf_xml_external_reference_count = _max_pattern_count_levels(full_levels, PATTERNS["ssrf_xml_external_reference"])
    shell_operator_command_pair_count = _max_pattern_count_levels(full_levels, PATTERNS["shell_operator_command_pair"])
    unix_command_count = _max_pattern_count_levels(full_levels, PATTERNS["unix_command"])
    windows_command_count = _max_pattern_count_levels(full_levels, PATTERNS["windows_command"])
    shell_wrapper_count = _max_pattern_count_levels(full_levels, PATTERNS["shell_wrapper"])
    command_substitution_count = _max_pattern_count_levels(full_levels, PATTERNS["command_substitution"])

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

        # Estadísticas globales de la request completa y patrones faltantes del paper.
        "full_request_len": float(full_request_len),
        "full_request_entropy": float(_entropy(full_request_raw)),
        "full_request_special_char_ratio": _ratio(full_request_special_char_count, full_request_len),
        "full_request_digit_ratio": _ratio(full_request_digit_count, full_request_len),
        "full_request_unique_char_count": float(full_request_unique_char_count),
        "full_request_unique_char_ratio": _ratio(full_request_unique_char_count, full_request_len),
        "full_request_word_count": float(full_request_word_count),
        "ssti_delimiter_count": float(_count_patterns(paper_analysis_text, PATTERNS["ssti_delimiter"])),
        "ssti_keyword_count": float(_count_patterns(paper_analysis_text, PATTERNS["ssti_keyword"])),
        "ssrf_scheme_count": float(_count_patterns(paper_analysis_text, PATTERNS["ssrf_scheme"])),
        "ssrf_internal_target_count": float(_count_patterns(paper_analysis_text, PATTERNS["ssrf_internal_target"])),
        "shell_operator_count": float(_count_patterns(paper_analysis_text, PATTERNS["shell_operator"])),
        "command_keyword_count": float(_count_patterns(paper_analysis_text, PATTERNS["command_keyword"])),
        "code_exec_function_count": float(_count_patterns(paper_analysis_text, PATTERNS["code_exec_function"])),
        "xss_dom_event_count": float(_count_patterns(paper_analysis_text, PATTERNS["xss_dom_event"])),

        # Contexto y estructura v3: mismas reglas para cualquier mensaje HTTP.
        "decode_reduction_ratio": float(decode_reduction_ratio),
        "max_uri_atom_special_ratio": float(max_uri_atom_special_ratio),
        "max_header_atom_special_ratio": float(max_header_atom_special_ratio),
        "max_body_atom_special_ratio": float(max_body_atom_special_ratio),
        "max_atom_entropy": float(max_atom_entropy),
        "ssti_complete_expression_count": float(ssti_complete_expression_count),
        "ssti_arithmetic_expression_count": float(ssti_arithmetic_expression_count),
        "ssti_object_chain_expression_count": float(ssti_object_chain_expression_count),
        "ssti_polyglot_count": float(ssti_polyglot_count),
        "ssti_first_decode_level": float(ssti_first_decode_level),
        "xss_dangerous_tag_count": float(xss_dangerous_tag_count),
        "xss_tag_event_pair_count": float(xss_tag_event_pair_count),
        "xss_script_block_count": float(xss_script_block_count),
        "xss_js_execution_count": float(xss_js_execution_count),
        "xss_first_decode_level": float(xss_first_decode_level),
        "ssti_without_xss_structure_count": float(ssti_without_xss_structure_count),
        "xss_without_ssti_structure_count": float(xss_without_ssti_structure_count),
        "ssrf_url_in_request_target_count": float(ssrf_url_in_request_target_count),
        "ssrf_url_in_value_count": float(ssrf_url_in_value_count),
        "ssrf_xml_external_reference_count": float(ssrf_xml_external_reference_count),
        "shell_operator_command_pair_count": float(shell_operator_command_pair_count),
        "unix_command_count": float(unix_command_count),
        "windows_command_count": float(windows_command_count),
        "shell_wrapper_count": float(shell_wrapper_count),
        "command_substitution_count": float(command_substitution_count),

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


class _FeatureRowDeadline(BaseException):
    """Interrupción interna que no debe ser absorbida por except Exception."""


def _feature_row_alarm_handler(signum: int, frame: Any) -> None:
    del signum, frame
    raise _FeatureRowDeadline()


def _extract_features_from_tuple(row_tuple: Tuple[str, str, str, str]) -> List[float]:
    method, uri, headers_json, body = row_tuple
    d = extract_fixed_features(method, uri, _json_loads_dict(headers_json), body)
    return [float(d[f]) for f in FIXED_FEATURES]


def _extract_features_with_deadline(
    task: Tuple[int, Tuple[str, str, str, str], float],
) -> List[float]:
    row_index, row_tuple, timeout_seconds = task
    timeout = float(timeout_seconds or 0.0)
    can_alarm = timeout > 0 and hasattr(signal, "setitimer") and hasattr(signal, "SIGALRM")
    previous_handler: Any = None
    try:
        if can_alarm:
            previous_handler = signal.signal(signal.SIGALRM, _feature_row_alarm_handler)
            signal.setitimer(signal.ITIMER_REAL, timeout)
        return _extract_features_from_tuple(row_tuple)
    except _FeatureRowDeadline:
        raise RuntimeError(
            f"Feature extraction timed out at zero-based row {row_index} "
            f"after {timeout:g}s; checkpoint preserved."
        ) from None
    finally:
        if can_alarm:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            if previous_handler is not None:
                signal.signal(signal.SIGALRM, previous_handler)


def _feature_checkpoint_signature(
    df: pd.DataFrame,
    rows: Optional[Iterable[Tuple[str, str, str, str]]] = None,
) -> Dict[str, Any]:
    n = int(len(df))
    sample_positions = sorted(set(i for i in [0, n // 2, n - 1] if 0 <= i < n))
    sample_ids = df.get("sample_id", pd.Series(np.arange(n), index=df.index)).astype(str)
    labels = df[LABEL_COL].astype(str)
    digest = hashlib.sha256()
    digest.update(FEATURE_SET_VERSION.encode("utf-8"))
    for feature in FIXED_FEATURES:
        raw = feature.encode("utf-8")
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    if rows is not None:
        for row in rows:
            for value in row:
                raw = _safe_str(value).encode("utf-8", errors="surrogatepass")
                digest.update(len(raw).to_bytes(8, "big"))
                digest.update(raw)
    return {
        "rows": n,
        "feature_set_version": FEATURE_SET_VERSION,
        "features": list(FIXED_FEATURES),
        "sample_ids": [sample_ids.iloc[i] for i in sample_positions],
        "sample_positions": sample_positions,
        "input_sha256": digest.hexdigest(),
        "label_distribution": {
            str(k): int(v) for k, v in labels.value_counts(dropna=False).sort_index().items()
        },
    }


def build_feature_frame(
    df: pd.DataFrame,
    *,
    workers: int = 1,
    chunksize: int = 256,
    checkpoint_dir: Optional[Path] = None,
    checkpoint_rows: int = 10000,
    resume: bool = False,
    row_timeout_seconds: float = 30.0,
) -> Tuple[pd.DataFrame, float, Dict[str, Any]]:
    rows = list(zip(
        df[RAW_METHOD_COL].fillna("").astype(str).tolist(),
        df[RAW_URI_COL].fillna("").astype(str).tolist(),
        df[RAW_HEADERS_JSON_COL].fillna("{}").astype(str).tolist(),
        df[RAW_BODY_COL].fillna("").astype(str).tolist(),
    ))
    t0 = time.perf_counter()
    desc = f"Extracting {FEATURE_POLICY_NAME} features ({len(FIXED_FEATURES)})"
    n_rows = len(rows)
    signature = _feature_checkpoint_signature(df, rows)
    resumed_rows = 0
    resumed_elapsed_seconds = 0.0
    matrix_path: Optional[Path] = None
    state_path: Optional[Path] = None
    matrix: Any

    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        tag = _safe_name(FEATURE_SET_VERSION).lower()
        matrix_path = checkpoint_dir / f".feature_checkpoint_{tag}.npy"
        state_path = checkpoint_dir / f"feature_checkpoint_{tag}.json"
        state: Dict[str, Any] = {}
        if resume and matrix_path.exists() and state_path.exists():
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
                candidate = np.load(matrix_path, mmap_mode="r+")
                if (
                    state.get("signature") == signature
                    and tuple(candidate.shape) == (n_rows, len(FIXED_FEATURES))
                    and candidate.dtype == np.dtype(np.float32)
                ):
                    matrix = candidate
                    resumed_rows = max(0, min(n_rows, int(state.get("completed_rows", 0))))
                    resumed_elapsed_seconds = max(0.0, float(state.get("elapsed_seconds", 0.0) or 0.0))
                    print(
                        f"[RESUME] Feature checkpoint válido: {resumed_rows}/{n_rows} filas completas.",
                        flush=True,
                    )
                else:
                    del candidate
                    state = {}
            except Exception as e:
                print(f"[WARN] Ignoro feature checkpoint inválido: {type(e).__name__}: {e}", flush=True)
                state = {}
        if not state:
            for stale in [matrix_path, state_path]:
                try:
                    stale.unlink(missing_ok=True)
                except Exception:
                    pass
            matrix = np.lib.format.open_memmap(
                matrix_path,
                mode="w+",
                dtype=np.float32,
                shape=(n_rows, len(FIXED_FEATURES)),
            )
            _save_json(
                {
                    "created_at": _now_iso(),
                    "status": "in_progress",
                    "completed_rows": 0,
                    "elapsed_seconds": 0.0,
                    "signature": signature,
                },
                state_path,
            )
    else:
        matrix = np.empty((n_rows, len(FIXED_FEATURES)), dtype=np.float32)

    batch_rows = max(1, int(checkpoint_rows))
    executor: Optional[ProcessPoolExecutor] = None
    extraction_failed = False
    if workers and workers > 1 and n_rows > 500:
        executor = ProcessPoolExecutor(max_workers=max(1, int(workers)))
    try:
        with tqdm(total=n_rows, initial=resumed_rows, desc=desc, mininterval=5) as progress:
            for start in range(resumed_rows, n_rows, batch_rows):
                end = min(n_rows, start + batch_rows)
                current = [
                    (idx, rows[idx], float(row_timeout_seconds))
                    for idx in range(start, end)
                ]
                if executor is not None:
                    values = list(
                        executor.map(
                            _extract_features_with_deadline,
                            current,
                            chunksize=max(1, int(chunksize)),
                        )
                    )
                else:
                    values = [_extract_features_with_deadline(task) for task in current]
                matrix[start:end, :] = np.asarray(values, dtype=np.float32)
                if hasattr(matrix, "flush"):
                    matrix.flush()
                if state_path is not None:
                    _save_json(
                        {
                            "updated_at": _now_iso(),
                            "status": "in_progress" if end < n_rows else "features_complete",
                            "completed_rows": int(end),
                            "elapsed_seconds": float(resumed_elapsed_seconds + (time.perf_counter() - t0)),
                            "signature": signature,
                        },
                        state_path,
                    )
                progress.update(end - start)
    except BaseException:
        extraction_failed = True
        print(
            f"[WARN] Extracción interrumpida; el checkpoint conserva {state_path or 'el progreso disponible'}.",
            flush=True,
        )
        raise
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=extraction_failed)

    secs = resumed_elapsed_seconds + (time.perf_counter() - t0)
    # Copiar antes de cerrar/eliminar el memmap; con 704k x 93 son ~250 MiB.
    values_array = np.asarray(matrix, dtype=np.float32).copy()
    del matrix
    feats = pd.DataFrame(values_array, columns=FIXED_FEATURES).fillna(0).astype(np.float32)
    report = {
        "feature_set_version": FEATURE_SET_VERSION,
        "input_signature": signature,
        "rows": n_rows,
        "resumed_rows": int(resumed_rows),
        "resumed_elapsed_seconds": float(resumed_elapsed_seconds),
        "checkpoint_rows": int(batch_rows),
        "row_timeout_seconds": float(row_timeout_seconds),
        "checkpoint_matrix": str(matrix_path) if matrix_path else None,
        "checkpoint_state": str(state_path) if state_path else None,
    }
    return pd.concat([df.reset_index(drop=True), feats], axis=1), secs, report


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
# Labels y loaders Egipcios / DS_Augmented_v2_csv
# ---------------------------------------------------------------------------

def _read_csv_table_flexible(path: str, sep: str = "auto") -> Tuple[pd.DataFrame, str]:
    """Lee CSV/TSV/GZ tolerando celdas multilínea y codificación latin-1 como fallback."""
    sep_eff = sep
    if not sep_eff or str(sep_eff).lower() == "auto":
        sep_eff = _sniff_sep(path)
    attempts = [
        {"engine": "c", "encoding": "utf-8", "low_memory": False},
        {"engine": "c", "encoding": "latin-1", "low_memory": False},
        {"engine": "python", "encoding": "utf-8"},
        {"engine": "python", "encoding": "latin-1"},
    ]
    last_err: Optional[Exception] = None
    for kwargs in attempts:
        try:
            df = pd.read_csv(
                path,
                sep=sep_eff,
                compression="infer",
                skipinitialspace=True,
                **kwargs,
            )
            return _normalize_columns(df), str(sep_eff)
        except Exception as e:
            last_err = e
    raise ValueError(f"No pude leer CSV/TSV {path!r} con sep={sep_eff!r}: {type(last_err).__name__}: {last_err}")


def _pick_column(df: pd.DataFrame, preferred: str, fallback_idx: int, role: str) -> str:
    cols = [str(c).strip() for c in df.columns]
    df.columns = cols
    preferred_s = _safe_str(preferred).strip()
    if preferred_s in df.columns:
        return preferred_s
    lower_map = {str(c).strip().lower(): str(c).strip() for c in df.columns}
    if preferred_s.lower() in lower_map:
        return lower_map[preferred_s.lower()]
    if len(cols) > fallback_idx:
        chosen = cols[fallback_idx]
        print(f"[WARN] Columna {role}={preferred_s!r} no existe; uso columna #{fallback_idx + 1}: {chosen!r}", flush=True)
        return chosen
    raise ValueError(f"No encontré columna para {role}. Esperaba {preferred_s!r}; columnas disponibles: {cols}")


def _parse_raw_http_request(raw: Any) -> Dict[str, Any]:
    """Parsea una request HTTP completa desde una celda tipo col1."""
    text = _safe_str(raw).replace("\ufeff", "")
    # Soporta tanto saltos reales de línea como literales exportados como \n.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if "\n" not in text and ("\\n" in text or "\\r" in text):
        text = text.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\r", "\n")
    lines = text.split("\n")
    start = 0
    while start < len(lines) and not lines[start].strip():
        start += 1
    first_line = lines[start].strip() if start < len(lines) else ""

    method = ""
    uri = ""
    http_version = ""
    m = re.match(r"^([A-Za-z]+)\s+(\S+)(?:\s+HTTP/(\d+(?:\.\d+)?))?", first_line)
    if m:
        method = _normalize_method(m.group(1))
        uri = m.group(2)
        http_version = m.group(3) or ""
    else:
        parts = first_line.split()
        if parts:
            method = _normalize_method(parts[0])
        if len(parts) >= 2:
            uri = parts[1]
        elif first_line:
            uri = first_line
        if len(parts) >= 3 and parts[2].upper().startswith("HTTP/"):
            http_version = parts[2].split("/", 1)[1]

    headers: Dict[str, str] = {}
    body_lines: List[str] = []
    in_headers = True
    for line in lines[start + 1:]:
        if in_headers:
            if line.strip() == "":
                in_headers = False
                continue
            if ":" in line:
                k, v = line.split(":", 1)
                k = k.strip()
                if k and _HTTP_HEADER_NAME_RE.match(k):
                    headers[k] = v.strip()
                    continue
            # Si aparece una línea no compatible con header, la tratamos como inicio del body.
            in_headers = False
            body_lines.append(line)
        else:
            body_lines.append(line)
    body = "\n".join(body_lines).strip("\n")
    return {
        "request_line": first_line,
        "method": method,
        "uri": uri,
        "headers": headers,
        "body": body,
        "http_version": http_version,
    }


def _egipcios_label_name_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", _safe_str(value).casefold())


def _normalize_egipcios_label(
    raw: Any,
    *,
    label_prefix: str = "EGIPCIOS",
    normal_regex: str = EGIPCIOS_DEFAULT_NORMAL_REGEX,
) -> str:
    """Mapea las nueve clases del paper por ID; nunca aplica un fallback familiar."""
    del label_prefix, normal_regex  # compatibilidad de firma; el esquema es deliberadamente fijo.
    raw_text = _safe_str(raw).strip()
    match = re.fullmatch(r"\s*(\d{1,4})\s*[-_:]\s*(.*?)\s*", raw_text)
    if not match:
        raise ValueError(f"Etiqueta Egipcios fuera del esquema WAMM: {raw_text!r}")
    class_id = int(match.group(1))
    spec = EGIPCIOS_PAPER_LABEL_SPECS.get(class_id)
    if spec is None:
        raise ValueError(f"ID Egipcios inesperado: {class_id}: {raw_text!r}")
    expected_name, canonical = spec
    observed_name = match.group(2)
    if _egipcios_label_name_key(observed_name) != _egipcios_label_name_key(expected_name):
        raise ValueError(
            f"Descriptor contradictorio para ID {class_id}: {observed_name!r}; "
            f"esperado {expected_name!r}."
        )
    return canonical


def load_egipcios_csv(
    path: str,
    *,
    sep: str,
    request_col: str,
    label_col: str,
    sample_n: int,
    label_prefix: str,
    normal_regex: str,
) -> pd.DataFrame:
    df, sep_eff = _read_csv_table_flexible(path, sep=sep)
    request_c = _pick_column(df, request_col or EGIPCIOS_DEFAULT_REQUEST_COL, 0, "request_col")
    label_c = _pick_column(df, label_col or EGIPCIOS_DEFAULT_LABEL_COL, 1, "label_col")
    if sample_n and sample_n > 0:
        df = df.head(int(sample_n)).copy()

    rows: List[Dict[str, Any]] = []
    for local_i, row in enumerate(df.itertuples(index=False), start=0):
        # itertuples no conserva nombres con espacios de forma directa; usar iloc por robustez.
        raw_req = df.iloc[local_i][request_c]
        raw_label = df.iloc[local_i][label_c]
        parsed = _parse_raw_http_request(raw_req)
        try:
            label_mc = _normalize_egipcios_label(raw_label, label_prefix=label_prefix, normal_regex=normal_regex)
        except ValueError as exc:
            raise ValueError(f"{path}: fila de datos {local_i + 2}: {exc}") from exc
        rows.append({
            "sample_id": f"{Path(path).name}:{local_i}",
            "raw_request": _safe_str(raw_req),
            "request_line": parsed.get("request_line", ""),
            RAW_METHOD_COL: parsed.get("method", ""),
            RAW_URI_COL: parsed.get("uri", ""),
            RAW_BODY_COL: parsed.get("body", ""),
            RAW_HEADERS_JSON_COL: json.dumps(parsed.get("headers", {}) or {}, ensure_ascii=False),
            "http_version": parsed.get("http_version", ""),
            "label_raw": _safe_str(raw_label),
            LABEL_COL: label_mc,
            "label_binary": 0 if label_mc == "NORMAL" else 1,
            "source_file": path,
            "detected_sep": sep_eff,
        })
    return pd.DataFrame(rows)


def _validate_egipcios_paper_dataset(
    df: pd.DataFrame,
    *,
    require_exact_distribution: bool,
) -> Dict[str, Any]:
    raw_counts = {str(k): int(v) for k, v in df["label_raw"].astype(str).value_counts().items()}
    canonical_counts = {str(k): int(v) for k, v in df[LABEL_COL].astype(str).value_counts().items()}
    observed_classes = tuple(sorted(canonical_counts))
    if observed_classes != EGIPCIOS_EXPECTED_CLASSES:
        raise ValueError(
            "Taxonomía Egipcios inválida. "
            f"observada={list(observed_classes)!r}; esperada={list(EGIPCIOS_EXPECTED_CLASSES)!r}."
        )
    if require_exact_distribution:
        if len(df) != EGIPCIOS_EXPECTED_TOTAL_ROWS:
            raise ValueError(
                f"DS-Augmented-v2 debe tener {EGIPCIOS_EXPECTED_TOTAL_ROWS} filas; "
                f"se cargaron {len(df)}."
            )
        if raw_counts != EGIPCIOS_EXPECTED_RAW_COUNTS:
            raise ValueError(
                "Distribución raw distinta de DS-Augmented-v2 publicada/validada. "
                f"observada={raw_counts!r}; esperada={EGIPCIOS_EXPECTED_RAW_COUNTS!r}."
            )
    return {
        "status": "valid",
        "label_schema_version": "egipcios_paper_native9_v1",
        "methodology_sources": {
            "wamm_paper": WAMM_PAPER_URL,
            "wamm_pattern_reference": WAMM_PATTERN_REFERENCE_URL,
        },
        "strict_exact_distribution": bool(require_exact_distribution),
        "rows": int(len(df)),
        "n_classes": int(len(observed_classes)),
        "classes": list(observed_classes),
        "raw_label_counts": raw_counts,
        "canonical_label_counts": canonical_counts,
        "expected_total_rows": EGIPCIOS_EXPECTED_TOTAL_ROWS,
        "expected_raw_label_counts": EGIPCIOS_EXPECTED_RAW_COUNTS,
        "capec88_and_capec248_separate": (
            "EGIPCIOS-OSCommandInjection" in canonical_counts
            and "EGIPCIOS-CommandInjection" in canonical_counts
        ),
    }


def _validate_dataset_schema(
    df: pd.DataFrame,
    dataset: str,
    *,
    require_exact_distribution: bool,
) -> Dict[str, Any]:
    profile = _dataset_profile(dataset)
    observed_counts = {
        str(key): int(value)
        for key, value in df[LABEL_COL].astype(str).value_counts().items()
    }
    expected_counts = dict(profile["expected_full_counts"])
    observed_classes = tuple(sorted(observed_counts))
    expected_classes = tuple(profile["expected_classes"])
    if observed_classes != expected_classes:
        raise ValueError(
            f"Taxonomía {dataset} inválida: observada={observed_classes!r}; "
            f"esperada={expected_classes!r}."
        )
    if require_exact_distribution and observed_counts != expected_counts:
        raise ValueError(
            f"Distribución {dataset} distinta de la validada: "
            f"observada={observed_counts!r}; esperada={expected_counts!r}."
        )
    payload = {
        "status": "valid",
        "dataset": dataset,
        "label_schema_version": profile["label_schema_version"],
        "strict_exact_distribution": bool(require_exact_distribution),
        "rows": int(len(df)),
        "n_classes": int(len(observed_classes)),
        "classes": list(observed_classes),
        "canonical_label_counts": observed_counts,
        "expected_canonical_label_counts": expected_counts,
        "expected_total_rows": int(sum(expected_counts.values())),
    }
    if dataset == "egipcios":
        payload["capec88_and_capec248_separate"] = bool(
            "EGIPCIOS-OSCommandInjection" in observed_counts
            and "EGIPCIOS-CommandInjection" in observed_counts
        )
    return payload


def _dataset_label_mapping_frame(dataset: str) -> pd.DataFrame:
    profile = _dataset_profile(dataset)
    rows = [
        {
            "dataset": dataset,
            "canonical_label": label,
            "expected_full_count": int(count),
            "label_schema_version": profile["label_schema_version"],
        }
        for label, count in profile["expected_full_counts"].items()
    ]
    if dataset == "egipcios":
        by_label = {
            canonical: (class_id, descriptor)
            for class_id, (descriptor, canonical) in EGIPCIOS_PAPER_LABEL_SPECS.items()
        }
        for row in rows:
            row["capec_id"], row["raw_descriptor"] = by_label[row["canonical_label"]]
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
    """Modo Harvard operativo de 10 clases derivado del top 14 exacto.

    Primero filtra a las mismas 14 combinaciones CAPEC más frecuentes; después
    agrupa las clases que, para un WAF, representan manipulación directa de la request:
      * CAPEC-272, CAPEC-274 y CAPEC-272+274 -> RequestManipulation.
      * CAPEC-153 y CAPEC-194 -> RequestManipulation.

    El resultado final queda en 10 clases operativas para Harvard. Se mantienen
    Torpeda y la base de 58 features intra-request sin cambios; las 14 nuevas se aplican a los tres datasets.
    """
    return (mode or "").strip().lower().replace("_", "-") in {
        "top10-request-groups",
        "top10-request-group",
        "request-groups",
        "request-group",
        "top14-request-groups",
        "top14-request-group",
        "top10-operational",
        "top10-operational-groups",
        "top10-operational-group",
    }


def _harvard_operational_group_label_from_combo_key(combo_key: str) -> str:
    key = _safe_str(combo_key).strip() or "NORMAL"
    if key not in HARVARD_TOP14_COMBO_KEY_TO_SPEC:
        return HARVARD_IGNORED_TOP14_LABEL
    if key == "NORMAL":
        return "NORMAL"
    if key in {"CAPEC-272", "CAPEC-274", "CAPEC-272+CAPEC-274", "CAPEC-153", "CAPEC-194"}:
        return "HARVARD-RequestManipulation"
    # El resto conserva una etiqueta operativa cercana a la combinación exacta original.
    return str(HARVARD_TOP14_COMBO_KEY_TO_SPEC[key]["label"])


def _harvard_operational_group_classes_df() -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for spec in HARVARD_TOP14_COMBO_CLASSES:
        capec_ids = tuple(spec["capec_ids"])
        combo_key = "+".join(capec_ids) if capec_ids else "NORMAL"
        final_label = _harvard_operational_group_label_from_combo_key(combo_key)
        if final_label == "HARVARD-RequestManipulation":
            if combo_key in {"CAPEC-272", "CAPEC-274", "CAPEC-272+CAPEC-274"}:
                rule = "CAPEC-272/CAPEC-274 agrupados junto con manipulación de datos como RequestManipulation"
            else:
                rule = "CAPEC-153/CAPEC-194 agrupados junto con manipulación HTTP como RequestManipulation"
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

    out["label_multiclase"] = out.apply(row_labels, axis=1)
    out["label_combo_key"] = out["label_multiclase"].map(_harvard_combo_key_from_labels)
    out["label_combo_capec_ids"] = out["label_combo_key"].map(_harvard_combo_ids_text_from_key)
    out["label_combo_native"] = out["label_multiclase"].map(_harvard_combo_display_from_labels)
    out["label_top14_combo_rank"] = out["label_combo_key"].map(_harvard_top14_rank_from_combo_key).astype(int)
    out["label_binary"] = out.apply(
        lambda r: 0 if (int(r[schema.normal_col]) == 1 and all(int(r[c]) == 0 for c in attack_cols)) else 1,
        axis=1,
    )
    sev_map = schema.severity_map if schema.severity_map is not None else DEFAULT_SRBH_SEVERITY_MAP
    picked = out["label_multiclase"].map(lambda labs: _choose_primary_label(labs, schema.multiclass_strategy, sev_map))
    out["label_multiclass_native"] = picked.map(lambda t: t[0])
    out["label_primary_severity"] = picked.map(lambda t: t[1])
    mode = (harvard_label_mode or "top14-combo").strip().lower()
    pref = label_prefix or "HARVARD"
    if _is_harvard_operational_group_mode(mode):
        # Primero se conserva el mismo universo top 14; luego se agrupan clases ambiguas
        # en una taxonomía operativa de 10 clases más defendible para multiclase. Las combinaciones
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
    label_prefix = {"torpeda": "TORPEDA", "harvard": "HARVARD", "egipcios": "EGIPCIOS"}.get(dataset, dataset.upper())
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
    elif dataset == "egipcios":
        inputs = _collect_inputs(args.inputs or args.egipcios_inputs or [], dataset="egipcios")
        if not inputs:
            raise SystemExit("No Egipcios CSV/TSV inputs found. Check --inputs/--egipcios-inputs/EGIPCIOS_RAW_DIR.")
        frames = []
        print(f"[INFO] Egipcios CSV/TSV files: {len(inputs)}", flush=True)
        for p in tqdm(inputs, desc="Loading Egipcios raw HTTP CSV", mininterval=5):
            d = load_egipcios_csv(
                p,
                sep=args.egipcios_sep,
                request_col=args.egipcios_request_col,
                label_col=args.egipcios_label_col,
                sample_n=args.sample_n,
                label_prefix=label_prefix,
                normal_regex=args.egipcios_normal_regex,
            )
            if d is not None and not d.empty:
                frames.append(d)
        if not frames:
            raise SystemExit("No samples parsed from Egipcios CSV inputs.")
        df = pd.concat(frames, ignore_index=True)
        label_validation = _validate_egipcios_paper_dataset(
            df,
            require_exact_distribution=bool(args.egipcios_strict_paper_dataset) and int(args.sample_n or 0) == 0,
        )
        df.attrs["egipcios_label_validation"] = label_validation
        print(
            "[OK] Taxonomía Egipcios: 9 clases nativas; "
            "CAPEC-88 y CAPEC-248 permanecen separadas.",
            flush=True,
        )
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
            mapping_df.to_csv(out_dir / "harvard_top14_to_request_groups_mapping.csv", index=False)
            final_df.to_csv(out_dir / "harvard_request_group_classes_allowed.csv", index=False)
            allowed = set(final_df["label_multiclass"].astype(str).tolist())
            allowed_combo_keys = mapping_df["combo_key"].astype(str).tolist()
            allowed_classes_path = str(out_dir / "harvard_request_group_classes_allowed.csv")
            operation_name = "harvard_top14_request_group_filter"
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
            "request_group_mapping_path": str(out_dir / "harvard_top14_to_request_groups_mapping.csv") if _is_harvard_operational_group_mode(mode) else None,
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

    if dataset == "harvard":
        min_count = int(args.harvard_min_class_count)
    elif dataset == "egipcios":
        min_count = int(args.egipcios_min_class_count)
    else:
        min_count = int(args.torpeda_min_class_count)
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
            max_iter=900,
            learning_rate=0.035,
            max_leaf_nodes=63,
            min_samples_leaf=12,
            l2_regularization=0.08,
            early_stopping=True,
            validation_fraction=0.10,
            n_iter_no_change=25,
            tol=1e-7,
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
    # TorpEda conserva el default original sin pesos; Harvard y Egipcios suelen estar desbalanceados.
    return "sqrt_balanced" if dataset in {"harvard", "egipcios"} else "none"


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
    elif re.fullmatch(r"power_balanced_0(?:_\d+|\.\d+)", mode_l):
        power_text = mode_l.removeprefix("power_balanced_").replace("_", ".")
        power = float(power_text)
        if not 0.0 < power <= 1.0:
            raise ValueError(f"Power fuera de rango: {power}")
        w = np.power(base, power)
        w = np.minimum(w, float(clip))
    else:
        raise ValueError(f"Unsupported sample weight mode: {mode!r}")
    mean = float(np.mean(w)) if len(w) else 1.0
    if mean > 0:
        w = w / mean
    return w.astype(np.float32)


def _histgb_model_from_params(args: argparse.Namespace, params: Dict[str, Any]) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=int(params["max_iter"]),
        learning_rate=float(params["learning_rate"]),
        max_leaf_nodes=int(params["max_leaf_nodes"]),
        min_samples_leaf=int(params["min_samples_leaf"]),
        l2_regularization=float(params["l2_regularization"]),
        early_stopping=True,
        validation_fraction=0.10,
        n_iter_no_change=int(params["n_iter_no_change"]),
        tol=float(params["tol"]),
        random_state=int(args.seed),
    )


def _model_candidates() -> List[Dict[str, Any]]:
    common = {
        "max_iter": 900,
        "learning_rate": 0.035,
        "max_leaf_nodes": 63,
        "min_samples_leaf": 12,
        "l2_regularization": 0.08,
        "n_iter_no_change": 25,
        "tol": 1e-7,
    }
    candidates = [
        {"name": "accuracy_guard_p020", "weight_power": 0.20, "params": dict(common)},
        {"name": "accuracy_macro_balance_p033", "weight_power": 0.33, "params": dict(common)},
        {"name": "macro_guard_p040", "weight_power": 0.40, "params": dict(common)},
        {"name": "sqrt_balanced_p050", "weight_power": 0.50, "params": dict(common)},
    ]
    if tuple(float(row["weight_power"]) for row in candidates) != MODEL_WEIGHT_POWERS:
        raise RuntimeError("Los candidatos no coinciden con la versión del pipeline unificado.")
    return candidates


def _cap_stratified_indices(y: np.ndarray, max_rows: int, seed: int) -> np.ndarray:
    idx = np.arange(len(y), dtype=int)
    if int(max_rows or 0) <= 0 or len(idx) <= int(max_rows):
        return idx
    selected, _ = train_test_split(
        idx,
        train_size=int(max_rows),
        random_state=int(seed),
        shuffle=True,
        stratify=np.asarray(y),
    )
    return np.asarray(selected, dtype=int)


def select_model_on_inner_validation(
    args: argparse.Namespace,
    X_train: np.ndarray,
    y_train: np.ndarray,
    out_dir: Path,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], float]:
    """Selecciona pesos con el mismo procedimiento, sólo dentro de outer-train."""
    inner_idx = np.arange(len(y_train), dtype=int)
    fit_idx, val_idx = train_test_split(
        inner_idx,
        test_size=float(args.model_selection_validation_size),
        random_state=int(args.seed) + 1701,
        shuffle=True,
        stratify=y_train,
    )
    cap_local = _cap_stratified_indices(
        y_train[fit_idx],
        int(args.model_selection_max_train_rows),
        int(args.seed) + 1702,
    )
    fit_idx = fit_idx[cap_local]
    rows: List[Dict[str, Any]] = []
    started = time.perf_counter()
    for candidate in _model_candidates():
        power = float(candidate["weight_power"])
        weight_mode = f"power_balanced_{power:.2f}".replace(".", "_")
        sample_weight = make_sample_weight_vector(
            y_train[fit_idx], mode=weight_mode, clip=float(args.sample_weight_clip)
        )
        model = _histgb_model_from_params(args, candidate["params"])
        fit0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train[fit_idx], y_train[fit_idx], sample_weight=sample_weight)
        pred = model.predict(X_train[val_idx]).astype(int)
        accuracy = float(accuracy_score(y_train[val_idx], pred))
        f1_macro_value = float(f1_score(y_train[val_idx], pred, average="macro", zero_division=0))
        # Misma regla bifocal en los tres datasets: evita ganar accuracy a costa
        # de destruir F1 macro, sin consultar jamás el outer-test.
        target_ratio = min(
            accuracy / SELECTION_REFERENCE_ACCURACY,
            f1_macro_value / SELECTION_REFERENCE_F1_MACRO,
        )
        rows.append({
            "candidate": candidate["name"],
            "weight_power": power,
            "weight_mode": weight_mode,
            "selection_score_min_target_ratio": target_ratio,
            "validation_accuracy": accuracy,
            "validation_f1_macro": f1_macro_value,
            "validation_rows": int(len(val_idx)),
            "selection_train_rows": int(len(fit_idx)),
            "fit_seconds": float(time.perf_counter() - fit0),
            "n_iter": int(getattr(model, "n_iter_", 0) or 0),
            "params": candidate["params"],
        })
        print(
            f"[SELECT] {candidate['name']}: acc={accuracy:.6f} "
            f"f1_macro={f1_macro_value:.6f} score={target_ratio:.6f}",
            flush=True,
        )
    rows.sort(
        key=lambda row: (
            float(row["selection_score_min_target_ratio"]),
            float(row["validation_f1_macro"]),
            float(row["validation_accuracy"]),
        ),
        reverse=True,
    )
    winner = rows[0]
    elapsed = time.perf_counter() - started
    _save_json({
        "selection_scope": "outer_train_only",
        "outer_test_touched": False,
        "objective": "max min(accuracy/0.9959, f1_macro/0.8607), shared across datasets",
        "winner": winner,
        "candidates_ranked": rows,
        "selection_seconds": elapsed,
    }, out_dir / "model_selection_inner_validation.json")
    pd.DataFrame([{k: v for k, v in row.items() if k != "params"} for row in rows]).to_csv(
        out_dir / "model_selection_inner_validation.csv", index=False
    )
    return winner, rows, float(elapsed)


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
# Curvas de pérdida / MSE / RMSE por época de boosting (TUTOR_LOSS_CURVES_FINAL)
# ---------------------------------------------------------------------------

def _lc_sample_indices(y: np.ndarray, max_rows: int, seed: int, min_rows_per_class: int = 200) -> Tuple[np.ndarray, bool, Dict[str, int]]:
    """Muestreo estratificado para curvas. max_rows=0 usa todo el split."""
    y = np.asarray(y)
    n = int(len(y))
    max_rows = int(max_rows or 0)
    if max_rows <= 0 or n <= max_rows:
        idx = np.arange(n, dtype=int)
        return idx, False, {str(k): int(v) for k, v in pd.Series(y[idx]).value_counts().sort_index().to_dict().items()}
    rng = np.random.RandomState(int(seed))
    all_idx = np.arange(n, dtype=int)
    selected: List[int] = []
    selected_mask = np.zeros(n, dtype=bool)
    classes_local = np.unique(y)
    per_class_cap = max(1, max_rows // max(1, len(classes_local)))
    min_pc = max(1, min(int(min_rows_per_class or 1), per_class_cap))
    for cls in classes_local:
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
    return idx, True, {str(k): int(v) for k, v in pd.Series(y[idx]).value_counts().sort_index().to_dict().items()}

def _lc_metrics_from_proba(y_true: np.ndarray, proba: np.ndarray, n_classes: int) -> Dict[str, Optional[float]]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(proba, dtype=np.float64)
    if p.ndim != 2 or p.shape[0] != len(y) or p.shape[1] != int(n_classes) or len(y) == 0:
        return {"log_loss": None, "mse_onehot": None, "rmse_onehot": None, "accuracy": None, "prediction_error_rate": None}
    p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    p = np.clip(p, 1e-15, 1.0)
    row_sum = p.sum(axis=1, keepdims=True)
    p = np.divide(p, row_sum, out=np.full_like(p, 1.0 / int(n_classes)), where=row_sum > 0)
    try:
        ll = _safe_float(log_loss(y, p, labels=list(range(int(n_classes)))))
    except Exception:
        ll = None
    true_p = p[np.arange(len(y)), y]
    mse = float((np.sum(p * p) - 2.0 * np.sum(true_p) + float(len(y))) / (float(len(y)) * float(n_classes)))
    mse = max(0.0, mse)
    pred = np.argmax(p, axis=1).astype(int)
    acc = float(np.mean(pred == y))
    return {
        "log_loss": ll,
        "mse_onehot": mse,
        "rmse_onehot": float(math.sqrt(mse)),
        "accuracy": acc,
        "prediction_error_rate": float(1.0 - acc),
    }

def _lc_internal_scores(model: Any, dataset: str) -> pd.DataFrame:
    try:
        train_score = list(np.asarray(getattr(model, "train_score_", []), dtype=float).reshape(-1))
    except Exception:
        train_score = []
    try:
        validation_score = list(np.asarray(getattr(model, "validation_score_", []), dtype=float).reshape(-1))
    except Exception:
        validation_score = []
    n = max(len(train_score), len(validation_score))
    rows: List[Dict[str, Any]] = []
    for i in range(n):
        tr = train_score[i] if i < len(train_score) else None
        va = validation_score[i] if i < len(validation_score) else None
        rows.append({
            "dataset": dataset,
            "epoch_internal": int(i),
            "boosting_iteration_internal": int(i),
            "train_score_raw": _safe_float(tr),
            "validation_score_raw": _safe_float(va),
            "train_loss_if_negative_score": -float(tr) if tr is not None and np.isfinite(float(tr)) else None,
            "validation_loss_if_negative_score": -float(va) if va is not None and np.isfinite(float(va)) else None,
        })
    return pd.DataFrame(rows)

def _lc_plot(df: pd.DataFrame, out_path: Path, y_cols: Sequence[Tuple[str, str]], title: str, ylabel: str) -> Optional[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:
        _write_text(out_path.with_suffix(out_path.suffix + ".plot_error.txt"), f"No se pudo importar matplotlib: {type(e).__name__}: {e}\n")
        return None
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)
    plotted = False
    for col, label in y_cols:
        if col not in df.columns:
            continue
        g = df[["epoch", col]].copy()
        g[col] = pd.to_numeric(g[col], errors="coerce")
        g = g.dropna().sort_values("epoch")
        if g.empty:
            continue
        ax.plot(g["epoch"].to_numpy(), g[col].to_numpy(), label=label, linewidth=1.8)
        plotted = True
    if not plotted:
        plt.close(fig)
        return None
    ax.set_title(title)
    ax.set_xlabel("Época / iteración de boosting")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return str(out_path)

def _lc_best(df: pd.DataFrame, col: str, minimize: bool = True) -> Dict[str, Optional[float]]:
    if df.empty or col not in df.columns or not df[col].notna().any():
        return {"epoch": None, "value": None}
    idx = df[col].idxmin() if minimize else df[col].idxmax()
    row = df.loc[idx]
    return {"epoch": int(row["epoch"]), "value": _safe_float(row[col])}

def compute_and_save_loss_curves(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    classes: Sequence[str],
    args: argparse.Namespace,
    out_dir: Path,
    dataset: str,
) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """Guarda curva de pérdida, MSE, RMSE, accuracy y error por época/iteración."""
    paths: Dict[str, str] = {}
    enabled = bool(getattr(args, "loss_curve", True))
    summary: Dict[str, Any] = {
        "available": False,
        "enabled": enabled,
        "dataset": dataset,
        "created_at": _now_iso(),
        "epoch_definition": "epoch = boosting iteration for sklearn HistGradientBoostingClassifier",
        "loss_definition": "multiclass log-loss from staged_predict_proba",
        "mse_rmse_definition": "MSE/RMSE between predicted class probabilities and the one-hot true class vector",
    }
    if not enabled:
        summary["reason"] = "disabled_by_no_loss_curve"
        _save_json(summary, out_dir / "loss_curve_metadata.json")
        paths["loss_curve_metadata"] = str(out_dir / "loss_curve_metadata.json")
        return paths, summary
    if not hasattr(model, "staged_predict_proba"):
        summary["reason"] = "model_has_no_staged_predict_proba"
        _save_json(summary, out_dir / "loss_curve_metadata.json")
        paths["loss_curve_metadata"] = str(out_dir / "loss_curve_metadata.json")
        return paths, summary

    t0 = time.perf_counter()
    max_rows = int(getattr(args, "loss_curve_max_rows", 50000))
    min_pc = int(getattr(args, "loss_curve_min_rows_per_class", 200))
    every_n = max(1, int(getattr(args, "loss_curve_every_n_iter", 1)))
    n_classes = int(len(classes))
    train_idx, train_sampled, train_counts = _lc_sample_indices(y_train, max_rows, int(args.seed) + 3100, min_pc)
    test_idx, test_sampled, test_counts = _lc_sample_indices(y_test, max_rows, int(args.seed) + 3200, min_pc)
    Xtr_c, ytr_c = np.asarray(X_train)[train_idx], np.asarray(y_train)[train_idx]
    Xte_c, yte_c = np.asarray(X_test)[test_idx], np.asarray(y_test)[test_idx]
    print(f"[INFO] Calculando curva de pérdida/MSE/RMSE para {dataset}: train={len(ytr_c)}, test={len(yte_c)}, every_n={every_n}", flush=True)

    rows_by_epoch: Dict[int, Dict[str, Any]] = {}
    def add(epoch: int, prefix: str, metrics: Dict[str, Optional[float]], rows_used: int, sampled: bool) -> None:
        row = rows_by_epoch.setdefault(int(epoch), {"dataset": dataset, "epoch": int(epoch), "boosting_iteration": int(epoch)})
        for k, v in metrics.items():
            row[f"{prefix}_{k}"] = v
        row[f"{prefix}_rows_used"] = int(rows_used)
        row[f"{prefix}_sampled"] = bool(sampled)

    n_iter_hint = int(getattr(model, "n_iter_", 0) or 0) or None
    try:
        for epoch, (p_tr, p_te) in enumerate(tqdm(zip(model.staged_predict_proba(Xtr_c), model.staged_predict_proba(Xte_c)), total=n_iter_hint, desc=f"Loss curve HistGB {dataset}", mininterval=5), start=1):
            if (epoch % every_n != 0) and (n_iter_hint is None or epoch != n_iter_hint):
                continue
            add(epoch, "train", _lc_metrics_from_proba(ytr_c, p_tr, n_classes), len(ytr_c), train_sampled)
            add(epoch, "test", _lc_metrics_from_proba(yte_c, p_te, n_classes), len(yte_c), test_sampled)
    except Exception as e:
        summary["staged_error"] = f"{type(e).__name__}: {e}"

    if not rows_by_epoch:
        summary["reason"] = "no_curve_rows"
        _save_json(summary, out_dir / "loss_curve_metadata.json")
        paths["loss_curve_metadata"] = str(out_dir / "loss_curve_metadata.json")
        return paths, summary

    curve_df = pd.DataFrame([rows_by_epoch[k] for k in sorted(rows_by_epoch)]).sort_values("epoch").reset_index(drop=True)
    internal_df = _lc_internal_scores(model, dataset)
    if not internal_df.empty:
        internal_csv = out_dir / "loss_curve_histgb_internal.csv"
        internal_df.to_csv(internal_csv, index=False)
        paths["loss_curve_internal_csv"] = str(internal_csv)
        mapping_train = dict(zip(internal_df["epoch_internal"].astype(int), pd.to_numeric(internal_df["train_loss_if_negative_score"], errors="coerce")))
        mapping_val = dict(zip(internal_df["epoch_internal"].astype(int), pd.to_numeric(internal_df["validation_loss_if_negative_score"], errors="coerce")))
        curve_df["internal_train_loss"] = curve_df["epoch"].map(mapping_train)
        curve_df["internal_validation_loss"] = curve_df["epoch"].map(mapping_val)

    by_epoch_csv = out_dir / "loss_curve_by_epoch.csv"
    histgb_csv = out_dir / "loss_curve_histgb.csv"
    error_csv = out_dir / "error_curve_mse_rmse_by_epoch.csv"
    curve_df.to_csv(by_epoch_csv, index=False)
    curve_df.to_csv(histgb_csv, index=False)
    curve_df.to_csv(error_csv, index=False)
    paths["loss_curve_by_epoch_csv"] = str(by_epoch_csv)
    paths["loss_curve_csv"] = str(histgb_csv)
    paths["error_curve_mse_rmse_by_epoch_csv"] = str(error_csv)

    for key, fname, cols, title, ylabel in [
        ("loss_curve_log_loss_png", "loss_curve_log_loss.png", [("train_log_loss", "train"), ("test_log_loss", "test/validación"), ("internal_train_loss", "train interno"), ("internal_validation_loss", "validación interna")], f"{dataset}: curva de pérdida por época", "Log-loss"),
        ("loss_curve_mse_png", "loss_curve_mse.png", [("train_mse_onehot", "train"), ("test_mse_onehot", "test/validación")], f"{dataset}: MSE por época", "MSE one-hot"),
        ("loss_curve_rmse_png", "loss_curve_rmse.png", [("train_rmse_onehot", "train"), ("test_rmse_onehot", "test/validación")], f"{dataset}: RMSE por época", "RMSE one-hot"),
        ("loss_curve_accuracy_png", "loss_curve_accuracy.png", [("train_accuracy", "train"), ("test_accuracy", "test/validación")], f"{dataset}: accuracy por época", "Accuracy"),
        ("prediction_error_curve_png", "prediction_error_curve_histgb.png", [("train_prediction_error_rate", "train"), ("test_prediction_error_rate", "test/validación")], f"{dataset}: error de predicción por época", "Error rate"),
    ]:
        png = _lc_plot(curve_df, out_dir / fname, cols, title, ylabel)
        if png:
            paths[key] = png
    if "loss_curve_log_loss_png" in paths:
        try:
            shutil.copyfile(paths["loss_curve_log_loss_png"], out_dir / "loss_curve_histgb.png")
            paths["loss_curve_png"] = str(out_dir / "loss_curve_histgb.png")
        except Exception:
            pass

    final = curve_df.tail(1).iloc[0].to_dict()
    best_log = _lc_best(curve_df, "test_log_loss", True)
    best_mse = _lc_best(curve_df, "test_mse_onehot", True)
    best_rmse = _lc_best(curve_df, "test_rmse_onehot", True)
    summary.update({
        "available": True,
        "n_epochs_recorded": int(curve_df["epoch"].max()),
        "n_curve_rows": int(len(curve_df)),
        "n_classes": n_classes,
        "loss_curve_max_rows": max_rows,
        "loss_curve_min_rows_per_class": min_pc,
        "loss_curve_every_n_iter": every_n,
        "train_rows_for_curve": int(len(ytr_c)),
        "test_rows_for_curve": int(len(yte_c)),
        "train_sampled": bool(train_sampled),
        "test_sampled": bool(test_sampled),
        "train_curve_class_counts": train_counts,
        "test_curve_class_counts": test_counts,
        "best_epoch_test_log_loss": best_log["epoch"],
        "best_test_log_loss": best_log["value"],
        "best_epoch_test_mse": best_mse["epoch"],
        "best_test_mse": best_mse["value"],
        "best_epoch_test_rmse": best_rmse["epoch"],
        "best_test_rmse": best_rmse["value"],
        "final_test_log_loss": _safe_float(final.get("test_log_loss")),
        "final_test_mse": _safe_float(final.get("test_mse_onehot")),
        "final_test_rmse": _safe_float(final.get("test_rmse_onehot")),
        "final_test_error_rate": _safe_float(final.get("test_prediction_error_rate")),
        "final_epoch_metrics": {str(k): (_safe_float(v) if isinstance(v, (int, float, np.integer, np.floating)) else v) for k, v in final.items()},
        "runtime_seconds": float(time.perf_counter() - t0),
        "artifacts": paths,
    })
    report_path = out_dir / "loss_curve_report.md"
    _write_text(report_path, "\n".join([
        "# Curvas de pérdida, MSE y RMSE por época",
        "",
        "Como el modelo es HistGradientBoostingClassifier, cada época se reporta como una iteración de boosting.",
        f"Dataset: {dataset}",
        f"Iteraciones registradas: {summary.get('n_epochs_recorded')}",
        f"Filas train usadas para curvas: {summary.get('train_rows_for_curve')}",
        f"Filas test usadas para curvas: {summary.get('test_rows_for_curve')}",
        f"Mejor test log-loss: epoch={summary.get('best_epoch_test_log_loss')}, value={summary.get('best_test_log_loss')}",
        f"Mejor test MSE: epoch={summary.get('best_epoch_test_mse')}, value={summary.get('best_test_mse')}",
        f"Mejor test RMSE: epoch={summary.get('best_epoch_test_rmse')}, value={summary.get('best_test_rmse')}",
        "",
        "Archivos principales: loss_curve_by_epoch.csv, loss_curve_histgb.csv, error_curve_mse_rmse_by_epoch.csv, loss_curve_log_loss.png, loss_curve_histgb.png, loss_curve_mse.png, loss_curve_rmse.png, prediction_error_curve_histgb.png.",
    ]) + "\n")
    paths["loss_curve_report"] = str(report_path)
    metadata = out_dir / "loss_curve_metadata.json"
    _save_json(summary, metadata)
    paths["loss_curve_metadata"] = str(metadata)
    summary["artifacts"] = paths
    return paths, summary

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
    feature_selection_manifest_frame().to_csv(out_dir / "feature_selection_manifest.csv", index=False)
    paths["feature_selection_manifest"] = str(out_dir / "feature_selection_manifest.csv")

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
        "pipeline_run_version": PIPELINE_RUN_VERSION,
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
    row_timeout = float(getattr(args, "feature_row_timeout_seconds", 30.0))
    feat_vals = [
        _extract_features_with_deadline((i, tup, row_timeout))
        for i, tup in enumerate(tuples)
    ]
    X_full = np.asarray(feat_vals, dtype=np.float32)
    _ = model.predict(X_full)
    batch_pipeline_wall = time.perf_counter() - t0
    batch_pipeline_cpu = time.process_time() - cpu0

    # Latencia request-by-request: feature extraction + inferencia para simular uso WAF online.
    lat_ms: List[float] = []
    cpu0 = time.process_time()
    t0 = time.perf_counter()
    for local_i, tup in enumerate(tqdm(tuples, total=len(tuples), desc="Operational latency raw request -> prediction", mininterval=5)):
        a = time.perf_counter()
        x = np.asarray(
            _extract_features_with_deadline((local_i, tup, row_timeout)),
            dtype=np.float32,
        ).reshape(1, -1)
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


def _load_json_file(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _input_file_fingerprints(paths: Sequence[str]) -> List[Dict[str, Any]]:
    fingerprints: List[Dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Input inexistente: {path}")
        stat_before = path.stat()
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        stat = path.stat()
        if (
            int(stat.st_size) != int(stat_before.st_size)
            or int(stat.st_mtime_ns) != int(stat_before.st_mtime_ns)
        ):
            raise RuntimeError(f"El input cambió mientras se calculaba su fingerprint: {path}")
        fingerprints.append({
            "path": str(path),
            "size_bytes": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
            "sha256": digest.hexdigest(),
        })
    return fingerprints


_METHODOLOGY_SIGNATURE_EXCLUDED_ARGS = {
    "dataset",
    "resume",
    "skip_complete",
    "clean_output_dir",
    "check_only",
    "project_dir",
    "definitivo_dir",
    "output_dir",
    "inputs",
    "torpeda_inputs",
    "harvard_inputs",
    "egipcios_inputs",
}


def _dataset_profile(dataset: str) -> Dict[str, Any]:
    try:
        profile = DATASET_SCHEMA_PROFILES[str(dataset).lower()]
    except KeyError as exc:
        raise ValueError(f"Dataset sin perfil: {dataset!r}") from exc
    return {
        **profile,
        "expected_full_counts": dict(profile["expected_full_counts"]),
        "expected_test_counts": dict(profile["expected_test_counts"]),
        "expected_classes": tuple(sorted(profile["expected_full_counts"])),
    }


def _dataset_label_schema_version(dataset: str) -> str:
    return str(_dataset_profile(dataset)["label_schema_version"])


def _methodology_configuration(args: argparse.Namespace, dataset: str) -> Dict[str, Any]:
    """Configuración que puede cambiar métricas, modelo o artefactos evaluados."""
    arg_payload = {
        str(key): value
        for key, value in sorted(vars(args).items())
        if key not in _METHODOLOGY_SIGNATURE_EXCLUDED_ARGS
    }
    return {
        "pipeline_run_version": PIPELINE_RUN_VERSION,
        "feature_set_version": FEATURE_SET_VERSION,
        "label_schema_version": _dataset_label_schema_version(dataset),
        "dataset_schema_profile": _dataset_profile(dataset),
        "features": list(FIXED_FEATURES),
        "model_candidates": _model_candidates(),
        "args": arg_payload,
        "thread_environment": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS"),
        },
    }


def _methodology_configuration_sha256(configuration: Dict[str, Any]) -> str:
    encoded = json.dumps(
        configuration,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _dataset_required_artifacts(
    out_dir: Path,
    *,
    dataset: str,
    require_loss_curve: bool = True,
    require_fi_per_class: bool = True,
) -> List[Path]:
    paths = [
        out_dir / MODEL_FILENAME,
        out_dir / METRICS_FILENAME,
        out_dir / "operational_metrics.json",
        out_dir / "operational_metrics.csv",
        out_dir / "run_status.json",
        out_dir / "features_used.csv",
        out_dir / "feature_selection_manifest.csv",
        out_dir / "label_schema_validation.json",
        out_dir / "label_mapping.csv",
        out_dir / "model_selection_inner_validation.json",
        out_dir / "model_selection_inner_validation.csv",
        out_dir / "run_config.json",
        out_dir / "classification_report_test.csv",
        out_dir / "classification_report_test.json",
        out_dir / "confusion_matrix_test.csv",
        out_dir / "per_class_ovr_metrics.csv",
        out_dir / "test_predictions.csv",
        out_dir / "metrics_flat.csv",
        out_dir / "feature_importance.csv",
        out_dir / "feature_importance_metadata.json",
        out_dir / "selected_model_summary.txt",
    ]
    if dataset == "egipcios":
        paths.extend([
            out_dir / "benchmark_vs_wamm.json",
            out_dir / "benchmark_vs_wamm.csv",
            out_dir / "benchmark_wamm_per_class_block_rate.csv",
        ])
    if require_fi_per_class:
        paths.append(out_dir / "feature_importance_per_class.csv")
    if require_loss_curve:
        paths.extend([
            out_dir / "loss_curve_histgb.csv",
            out_dir / "loss_curve_metadata.json",
        ])
    return paths


def _analysis_artifacts_valid(
    out_dir: Path,
    dataset: str,
    classes: Sequence[str],
    features: Sequence[str],
    *,
    expected_test_rows: int,
    require_loss_curve: bool = True,
) -> bool:
    """Valida contenido, no sólo existencia, de los artefactos del informe."""
    try:
        class_order = [str(value) for value in classes]
        expected_classes = set(class_order)
        expected_features = {str(value) for value in features}
        n_classes = len(expected_classes)
        n_features = len(expected_features)
        if len(class_order) != n_classes or len(features) != n_features:
            return False

        metrics_payload = _load_json_file(out_dir / METRICS_FILENAME)
        fi_meta = _load_json_file(out_dir / "feature_importance_metadata.json")
        operational = _load_json_file(out_dir / "operational_metrics.json")
        global_fi = pd.read_csv(out_dir / "feature_importance.csv")
        per_class_fi = pd.read_csv(out_dir / "feature_importance_per_class.csv")
        per_class_metrics = pd.read_csv(out_dir / "per_class_ovr_metrics.csv")
        predictions = pd.read_csv(
            out_dir / "test_predictions.csv", usecols=["y_true", "y_pred"]
        )

        # El modelo serializado también es parte de la evidencia: debe poder
        # cargarse, describir exactamente este pipeline y predecir con 73 inputs.
        bundle = joblib.load(out_dir / MODEL_FILENAME)
        if not isinstance(bundle, dict):
            return False
        if (
            bundle.get("dataset") != dataset
            or bundle.get("feature_count") != n_features
            or list(bundle.get("features") or []) != list(features)
            or list(bundle.get("classes") or []) != class_order
            or bundle.get("feature_set_version") != FEATURE_SET_VERSION
            or bundle.get("label_schema_version") != _dataset_label_schema_version(dataset)
            or bundle.get("pipeline_run_version") != PIPELINE_RUN_VERSION
        ):
            return False
        fitted_model = bundle.get("model")
        fitted_encoder = bundle.get("label_encoder")
        if fitted_model is None or fitted_encoder is None:
            return False
        if int(getattr(fitted_model, "n_features_in_", -1)) != n_features:
            return False
        if [str(value) for value in getattr(fitted_encoder, "classes_", [])] != class_order:
            return False
        model_classes = np.asarray(getattr(fitted_model, "classes_", []))
        if not np.array_equal(model_classes, np.arange(n_classes)):
            return False
        smoke_x = np.zeros((1, n_features), dtype=np.float32)
        smoke_pred = np.asarray(fitted_model.predict(smoke_x))
        smoke_proba = np.asarray(fitted_model.predict_proba(smoke_x))
        if (
            smoke_pred.shape != (1,)
            or smoke_proba.shape != (1, n_classes)
            or not np.isfinite(smoke_proba).all()
            or int(smoke_pred[0]) not in range(n_classes)
        ):
            return False

        if len(global_fi) != n_features or set(global_fi["feature"].astype(str)) != expected_features:
            return False
        if len(per_class_metrics) != n_classes or set(per_class_metrics["class"].astype(str)) != expected_classes:
            return False
        if len(per_class_fi) != n_classes * n_features:
            return False
        if set(per_class_fi["class"].astype(str)) != expected_classes:
            return False
        for class_name in expected_classes:
            rows = per_class_fi[per_class_fi["class"].astype(str) == class_name]
            if len(rows) != n_features:
                return False
            if set(rows["feature"].astype(str)) != expected_features:
                return False
            ranks = set(pd.to_numeric(rows["rank_in_class"], errors="coerce").dropna().astype(int))
            if ranks != set(range(1, n_features + 1)):
                return False
        if len(predictions) != int(expected_test_rows):
            return False
        if set(predictions["y_true"].astype(str)) != expected_classes:
            return False
        if not set(predictions["y_pred"].astype(str)).issubset(expected_classes):
            return False

        # Las métricas declaradas se reconstruyen desde las predicciones. Esto
        # impide que JSON/CSV consistentes entre sí pero inflados pasen la guarda.
        y_true_labels = predictions["y_true"].astype(str).to_numpy()
        y_pred_labels = predictions["y_pred"].astype(str).to_numpy()
        observed_supports = {
            str(key): int(value)
            for key, value in pd.Series(y_true_labels).value_counts().items()
        }
        if observed_supports != _dataset_profile(dataset)["expected_test_counts"]:
            return False
        reconstructed_accuracy = float(accuracy_score(y_true_labels, y_pred_labels))
        reconstructed_f1_macro = float(f1_score(
            y_true_labels,
            y_pred_labels,
            labels=class_order,
            average="macro",
            zero_division=0,
        ))
        test_metrics = metrics_payload.get("test_metrics") or {}

        def finite_close(left: Any, right: Any, *, atol: float = 1e-12) -> bool:
            try:
                left_f = float(left)
                right_f = float(right)
            except (TypeError, ValueError):
                return False
            return bool(
                np.isfinite(left_f)
                and np.isfinite(right_f)
                and math.isclose(left_f, right_f, rel_tol=0.0, abs_tol=atol)
            )

        if (
            metrics_payload.get("dataset") != dataset
            or int(metrics_payload.get("feature_count", -1)) != n_features
            or list(metrics_payload.get("features_used") or []) != list(features)
            or list(metrics_payload.get("classes") or []) != class_order
            or int(metrics_payload.get("test_rows", -1)) != int(expected_test_rows)
            or metrics_payload.get("test_class_counts") != observed_supports
            or not finite_close(test_metrics.get("accuracy"), reconstructed_accuracy)
            or not finite_close(test_metrics.get("f1_macro"), reconstructed_f1_macro)
        ):
            return False

        reconstructed_cm = confusion_matrix(
            y_true_labels, y_pred_labels, labels=class_order
        ).astype(int)
        payload_cm = np.asarray(test_metrics.get("confusion_matrix"), dtype=float)
        csv_cm = pd.read_csv(out_dir / "confusion_matrix_test.csv", index_col=0)
        if (
            payload_cm.shape != (n_classes, n_classes)
            or not np.array_equal(payload_cm, reconstructed_cm)
            or list(csv_cm.index.astype(str)) != class_order
            or list(csv_cm.columns.astype(str)) != class_order
            or not np.array_equal(csv_cm.to_numpy(dtype=float), reconstructed_cm)
        ):
            return False

        # Verifica los campos operativos del informe por clase contra la misma
        # matriz, no sólo que haya una fila con cada nombre.
        required_per_class_columns = {
            "class", "class_index", "support", "pred_support", "tp", "fp",
            "fn", "tn", "precision", "recall_tpr", "f1",
        }
        if not required_per_class_columns.issubset(per_class_metrics.columns):
            return False
        indexed_per_class = per_class_metrics.set_index(
            per_class_metrics["class"].astype(str), drop=False
        )
        for class_index, class_name in enumerate(class_order):
            row = indexed_per_class.loc[class_name]
            if isinstance(row, pd.DataFrame):
                return False
            tp = int(reconstructed_cm[class_index, class_index])
            support = int(reconstructed_cm[class_index, :].sum())
            pred_support = int(reconstructed_cm[:, class_index].sum())
            fn = support - tp
            fp = pred_support - tp
            tn = int(expected_test_rows) - tp - fn - fp
            precision = float(precision_score(
                y_true_labels,
                y_pred_labels,
                labels=[class_name],
                average=None,
                zero_division=0,
            )[0])
            recall = float(recall_score(
                y_true_labels,
                y_pred_labels,
                labels=[class_name],
                average=None,
                zero_division=0,
            )[0])
            f1_value = float(f1_score(
                y_true_labels,
                y_pred_labels,
                labels=[class_name],
                average=None,
                zero_division=0,
            )[0])
            if (
                not finite_close(row["class_index"], class_index, atol=0.0)
                or not finite_close(row["support"], support, atol=0.0)
                or not finite_close(row["pred_support"], pred_support, atol=0.0)
                or not finite_close(row["tp"], tp, atol=0.0)
                or not finite_close(row["fp"], fp, atol=0.0)
                or not finite_close(row["fn"], fn, atol=0.0)
                or not finite_close(row["tn"], tn, atol=0.0)
                or not finite_close(row["precision"], precision)
                or not finite_close(row["recall_tpr"], recall)
                or not finite_close(row["f1"], f1_value)
            ):
                return False

        report_json = _load_json_file(out_dir / "classification_report_test.json")
        report_csv = pd.read_csv(
            out_dir / "classification_report_test.csv", index_col=0
        )
        for class_index, class_name in enumerate(class_order):
            report_row = report_json.get(class_name) or {}
            csv_row = report_csv.loc[class_name]
            if (
                not finite_close(report_row.get("support"), observed_supports[class_name])
                or not finite_close(report_row.get("precision"), indexed_per_class.loc[class_name, "precision"])
                or not finite_close(report_row.get("recall"), indexed_per_class.loc[class_name, "recall_tpr"])
                or not finite_close(report_row.get("f1-score"), indexed_per_class.loc[class_name, "f1"])
                or not finite_close(csv_row.get("support"), observed_supports[class_name])
                or not finite_close(csv_row.get("precision"), report_row.get("precision"))
                or not finite_close(csv_row.get("recall"), report_row.get("recall"))
                or not finite_close(csv_row.get("f1-score"), report_row.get("f1-score"))
            ):
                return False

        if dataset == "egipcios":
            benchmark = _load_json_file(out_dir / "benchmark_vs_wamm.json")
            derived_beats_accuracy = reconstructed_accuracy > WAMM_PAPER_ACCURACY
            derived_beats_f1 = reconstructed_f1_macro > WAMM_PAPER_F1_MACRO
            if (
                benchmark != (metrics_payload.get("benchmark_vs_wamm") or {})
                or not finite_close(benchmark.get("our_accuracy"), reconstructed_accuracy)
                or not finite_close(benchmark.get("our_f1_macro"), reconstructed_f1_macro)
                or benchmark.get("beats_paper_accuracy") is not derived_beats_accuracy
                or benchmark.get("beats_paper_f1_macro") is not derived_beats_f1
                or benchmark.get("beats_paper_both_primary_metrics") is not (
                    derived_beats_accuracy and derived_beats_f1
                )
            ):
                return False

        per_class_errors = fi_meta.get("per_class_errors") or {}
        binary_errors = fi_meta.get("binary_errors") or {}
        if int(fi_meta.get("n_features", -1)) != n_features:
            return False
        if int(fi_meta.get("fi_rows_effective", 0)) <= 0:
            return False
        if fi_meta.get("permutation_error"):
            return False
        if set(per_class_errors) != expected_classes or any(per_class_errors.values()):
            return False
        if set(binary_errors) != {"binary_attack_f1", "binary_normal_f1"} or any(binary_errors.values()):
            return False

        required_operational = (
            "latency_ms_pipeline_online_p50",
            "latency_ms_pipeline_online_p95",
            "latency_ms_pipeline_online_p99",
            "throughput_req_s_pipeline_online",
            "throughput_req_s_pipeline_batch",
            "throughput_req_s_inference_batch",
        )
        if operational.get("available") is not True or int(operational.get("measured_rows", 0)) <= 0:
            return False
        for key in required_operational:
            value = float(operational.get(key, float("nan")))
            if not np.isfinite(value) or value <= 0.0:
                return False

        if require_loss_curve:
            loss_meta = _load_json_file(out_dir / "loss_curve_metadata.json")
            loss_rows = pd.read_csv(out_dir / "loss_curve_histgb.csv")
            if loss_meta.get("available") is not True or loss_meta.get("staged_error"):
                return False
            if int(loss_meta.get("n_curve_rows", 0)) <= 0:
                return False
            if len(loss_rows) != int(loss_meta.get("n_curve_rows", -1)):
                return False
        return True
    except Exception:
        return False


def _fingerprints_match(observed: Any, expected: Sequence[Dict[str, Any]]) -> bool:
    if not isinstance(observed, list) or len(observed) != len(expected):
        return False
    keys = ("path", "size_bytes", "sha256")
    if not all(isinstance(item, dict) for item in observed):
        return False
    if not all(isinstance(item, dict) for item in expected):
        return False
    return all(
        all(left.get(key) == right.get(key) for key in keys)
        for left, right in zip(observed, expected)
    )


def _dataset_is_complete(
    out_dir: Path,
    dataset: str,
    input_fingerprints: Sequence[Dict[str, Any]],
    methodology_config_sha256: str,
) -> bool:
    profile = _dataset_profile(dataset)
    schema_version = str(profile["label_schema_version"])
    dataset_label_mode = str(profile["dataset_label_mode"])
    expected_classes = tuple(profile["expected_classes"])
    expected_test_counts = dict(profile["expected_test_counts"])
    required = _dataset_required_artifacts(out_dir, dataset=dataset)
    if not all(path.is_file() and path.stat().st_size > 0 for path in required):
        return False
    try:
        payload = _load_json_file(out_dir / METRICS_FILENAME)
        status = _load_json_file(out_dir / "run_status.json")
        config = _load_json_file(out_dir / "run_config.json")
        validation = _load_json_file(out_dir / "label_schema_validation.json")
        selection = _load_json_file(out_dir / "model_selection_inner_validation.json")
        manifest = pd.read_csv(out_dir / "feature_selection_manifest.csv")
        observed_powers = sorted(
            round(float(row.get("weight_power")), 8)
            for row in (payload.get("model_selection_candidates") or [])
            if isinstance(row, dict) and row.get("weight_power") is not None
        )
        manifest_ok = bool(
            len(manifest) == len(ALL_CANDIDATE_FEATURES)
            and "selected" in manifest.columns
            and int(manifest["selected"].astype(str).str.lower().isin({"true", "1"}).sum()) == len(FIXED_FEATURES)
            and int((manifest["status"].astype(str) == "dropped").sum()) == len(FI_DROPPED_FEATURES)
        )
        return bool(
            payload.get("dataset") == dataset
            and payload.get("dataset_label_mode") == dataset_label_mode
            and payload.get("label_schema_version") == schema_version
            and payload.get("feature_set_version") == FEATURE_SET_VERSION
            and payload.get("pipeline_run_version") == PIPELINE_RUN_VERSION
            and payload.get("methodology_config_sha256") == methodology_config_sha256
            and int(payload.get("feature_count", -1)) == len(FIXED_FEATURES)
            and int(payload.get("n_classes", -1)) == len(expected_classes)
            and tuple(sorted(payload.get("classes") or [])) == expected_classes
            and int(payload.get("test_rows", -1)) == sum(expected_test_counts.values())
            and payload.get("test_class_counts") == expected_test_counts
            and payload.get("model_selection_scope") == "outer_train_only"
            and observed_powers == sorted(MODEL_WEIGHT_POWERS)
            and _fingerprints_match(payload.get("input_fingerprints"), input_fingerprints)
            and bool(payload.get("test_metrics"))
            and status.get("status") == "complete"
            and status.get("feature_set_version") == FEATURE_SET_VERSION
            and status.get("label_schema_version") == schema_version
            and status.get("pipeline_run_version") == PIPELINE_RUN_VERSION
            and status.get("methodology_config_sha256") == methodology_config_sha256
            and _fingerprints_match(status.get("input_fingerprints"), input_fingerprints)
            and config.get("feature_set_version") == FEATURE_SET_VERSION
            and config.get("label_schema_version") == schema_version
            and config.get("pipeline_run_version") == PIPELINE_RUN_VERSION
            and config.get("methodology_config_sha256") == methodology_config_sha256
            and _fingerprints_match(config.get("input_fingerprints"), input_fingerprints)
            and validation.get("status") == "valid"
            and validation.get("dataset") == dataset
            and validation.get("label_schema_version") == schema_version
            and validation.get("strict_exact_distribution") is True
            and int(validation.get("rows", -1)) == sum(profile["expected_full_counts"].values())
            and int(validation.get("n_classes", -1)) == len(expected_classes)
            and validation.get("canonical_label_counts") == profile["expected_full_counts"]
            and tuple(sorted(validation.get("classes") or [])) == expected_classes
            and selection.get("selection_scope") == "outer_train_only"
            and selection.get("outer_test_touched") is False
            and manifest_ok
            and _analysis_artifacts_valid(
                out_dir,
                dataset,
                expected_classes,
                FIXED_FEATURES,
                expected_test_rows=sum(expected_test_counts.values()),
                require_loss_curve=True,
            )
        )
    except Exception:
        return False


def _load_processed_for_resume(
    out_dir: Path,
    df_filtered: pd.DataFrame,
    dataset: str,
) -> Optional[Tuple[pd.DataFrame, Path, float]]:
    config_path = out_dir / "run_config.json"
    if not config_path.exists():
        return None
    try:
        config = _load_json_file(config_path)
        if config.get("feature_set_version") != FEATURE_SET_VERSION:
            return None
        if config.get("label_schema_version") != _dataset_label_schema_version(dataset):
            return None
        if config.get("pipeline_run_version") != PIPELINE_RUN_VERSION:
            return None
        extraction_report = _load_json_file(out_dir / "feature_extraction_report.json")
        expected_signature = extraction_report.get("input_signature")
        if not isinstance(expected_signature, dict):
            return None
    except Exception:
        return None

    current_rows = zip(
        df_filtered[RAW_METHOD_COL].fillna("").astype(str),
        df_filtered[RAW_URI_COL].fillna("").astype(str),
        df_filtered[RAW_HEADERS_JSON_COL].fillna("{}").astype(str),
        df_filtered[RAW_BODY_COL].fillna("").astype(str),
    )
    if _feature_checkpoint_signature(df_filtered, current_rows) != expected_signature:
        print("[WARN] Processed cache no coincide exactamente con las requests actuales; recalculo features.", flush=True)
        return None

    candidates = [
        out_dir / PROCESSED_PARQUET_FILENAME,
        out_dir / PROCESSED_CSV_FILENAME,
    ]
    for path in candidates:
        if not path.exists() or path.stat().st_size <= 0:
            continue
        try:
            cached = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path, low_memory=False)
            required_cols = set(FIXED_FEATURES) | {
                RAW_METHOD_COL,
                RAW_URI_COL,
                RAW_BODY_COL,
                RAW_HEADERS_JSON_COL,
                LABEL_COL,
            }
            if len(cached) != len(df_filtered) or not required_cols.issubset(cached.columns):
                continue
            positions = sorted(set(i for i in [0, len(cached) // 2, len(cached) - 1] if 0 <= i < len(cached)))
            cached_ids = cached.get("sample_id", pd.Series(np.arange(len(cached)))).astype(str)
            current_ids = df_filtered.get("sample_id", pd.Series(np.arange(len(df_filtered)))).astype(str)
            if any(cached_ids.iloc[i] != current_ids.iloc[i] for i in positions):
                continue
            if not cached[LABEL_COL].astype(str).equals(df_filtered[LABEL_COL].astype(str).reset_index(drop=True)):
                continue
            return cached.reset_index(drop=True), path, float(config.get("feature_extraction_seconds", 0.0) or 0.0)
        except Exception as e:
            print(f"[WARN] No pude reutilizar processed cache {path}: {type(e).__name__}: {e}", flush=True)
    return None


def _remove_feature_checkpoint(report: Dict[str, Any]) -> None:
    for key in ["checkpoint_matrix", "checkpoint_state"]:
        raw = report.get(key)
        if not raw:
            continue
        try:
            Path(str(raw)).unlink(missing_ok=True)
        except Exception as e:
            print(f"[WARN] No pude eliminar {raw}: {type(e).__name__}: {e}", flush=True)


def _write_run_config(args: argparse.Namespace, dataset: str, inputs: Sequence[str], out_dir: Path, extra: Dict[str, Any]) -> None:
    feature_definitions_frame().to_csv(out_dir / "features_used.csv", index=False)
    feature_selection_manifest_frame().to_csv(out_dir / "feature_selection_manifest.csv", index=False)
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
        "pipeline_run_version": PIPELINE_RUN_VERSION,
        "original_candidate_feature_count": len(ALL_84_CANDIDATE_FEATURES),
        "paper_gap_feature_count": len(PAPER_GAP_FEATURES),
        "paper_gap_features": PAPER_GAP_FEATURES,
        "contextual_http_feature_count": len(CONTEXTUAL_HTTP_FEATURES),
        "contextual_http_features": CONTEXTUAL_HTTP_FEATURES,
        "all_candidate_feature_count": len(ALL_CANDIDATE_FEATURES),
        "dropped_feature_count": len(FI_DROPPED_FEATURES),
        "dropped_features": FI_DROPPED_FEATURES,
        "feature_selection_rule": (
            "Se retienen features con importancia positiva global o por clase en la corrida previa; "
            "se conserva shell_operator_command_pair_count como excepción para la nueva frontera "
            "CAPEC-88/CAPEC-248; se añaden word_count y cuatro separadores compactos. Sin TF-IDF."
        ),
        "feature_selection_manifest": str(out_dir / "feature_selection_manifest.csv"),
        "label_schema_version": _dataset_label_schema_version(dataset),
        "methodology_sources": {
            "wamm_paper": WAMM_PAPER_URL,
            "wamm_pattern_reference": WAMM_PATTERN_REFERENCE_URL,
        },
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

    configured_items = args.inputs or getattr(args, f"{dataset}_inputs", None) or []
    configured_inputs = _collect_inputs(configured_items, dataset=dataset)
    if not configured_inputs:
        raise SystemExit(f"No se encontraron inputs para {dataset}.")
    input_fingerprints = _input_file_fingerprints(configured_inputs)
    methodology_configuration = _methodology_configuration(args, dataset)
    methodology_config_sha256 = _methodology_configuration_sha256(methodology_configuration)

    if bool(getattr(args, "skip_complete", False)) and _dataset_is_complete(
        out_dir, dataset, input_fingerprints, methodology_config_sha256
    ):
        payload = _load_json_file(out_dir / METRICS_FILENAME)
        print(f"[SKIP] {dataset} ya está completo y validado: {out_dir}", flush=True)
        return payload

    _save_json(
        {
            "status": "running",
            "started_at": _now_iso(),
            "dataset": dataset,
            "feature_set_version": FEATURE_SET_VERSION,
            "label_schema_version": _dataset_label_schema_version(dataset),
            "pipeline_run_version": PIPELINE_RUN_VERSION,
            "methodology_config_sha256": methodology_config_sha256,
            "input_fingerprints": input_fingerprints,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
        out_dir / "run_status.json",
    )

    print("================================================================", flush=True)
    print(f"[START] {dataset} compact73 unified HistGradientBoosting @ {_now_iso()}", flush=True)
    print(f"[OUT] {out_dir}", flush=True)
    print("================================================================", flush=True)

    gpu_visible = _gpu_visible_auto()
    if args.use_gpu in {"on", "auto"}:
        msg = "visible" if gpu_visible else "no visible"
        print(f"[INFO] GPU {msg}; HistGradientBoostingClassifier es CPU/OpenMP, no CUDA. Se usará CPU.", flush=True)

    t_run0 = time.perf_counter()
    df_raw, inputs = load_dataset_for_run(args, dataset)
    if [str(Path(p).resolve()) for p in inputs] != [
        str(Path(p).resolve()) for p in configured_inputs
    ]:
        raise RuntimeError("Los inputs resueltos cambiaron entre fingerprint y carga.")
    load_seconds = time.perf_counter() - t_run0
    print(f"[INFO] Rows loaded before filters: {len(df_raw)}", flush=True)
    df_filtered, filtering_report = apply_label_filters(df_raw, args, dataset, out_dir)
    print(f"[INFO] Rows after filters: {len(df_filtered)}", flush=True)
    post_filter_validation = _validate_dataset_schema(
        df_filtered,
        dataset,
        require_exact_distribution=int(args.sample_n or 0) == 0,
    )
    post_filter_validation["stage"] = "after_filters"
    _save_json(post_filter_validation, out_dir / "label_schema_validation.json")
    _dataset_label_mapping_frame(dataset).to_csv(out_dir / "label_mapping.csv", index=False)

    resumed_processed = _load_processed_for_resume(out_dir, df_filtered, dataset) if bool(getattr(args, "resume", False)) else None
    feature_checkpoint_report: Dict[str, Any]
    if resumed_processed is not None:
        df, processed_path, feature_extraction_seconds = resumed_processed
        feature_checkpoint_report = {
            "processed_cache_reused": True,
            "processed_cache_path": str(processed_path),
            "rows": int(len(df)),
        }
        print(f"[RESUME] Reutilizo features procesadas completas: {processed_path}", flush=True)
    else:
        checkpoint_dir = Path(args.definitivo_dir) / "_feature_checkpoints" / dataset
        df, feature_extraction_seconds, feature_checkpoint_report = build_feature_frame(
            df_filtered,
            workers=int(args.feature_workers),
            chunksize=int(args.feature_chunksize),
            checkpoint_dir=checkpoint_dir,
            checkpoint_rows=int(args.feature_checkpoint_rows),
            resume=bool(getattr(args, "resume", False)),
            row_timeout_seconds=float(args.feature_row_timeout_seconds),
        )
        processed_path = _save_processed(df, out_dir / PROCESSED_PARQUET_FILENAME)
        _save_json(
            {
                "status": "processed_cache_ready",
                "created_at": _now_iso(),
                "processed_path": str(processed_path),
                "feature_set_version": FEATURE_SET_VERSION,
                "label_schema_version": _dataset_label_schema_version(dataset),
                "pipeline_run_version": PIPELINE_RUN_VERSION,
                **feature_checkpoint_report,
            },
            out_dir / "feature_extraction_report.json",
        )

    label_dist = df[LABEL_COL].astype(str).value_counts().rename_axis("label").reset_index(name="count")
    label_dist.to_csv(out_dir / "label_distribution.csv", index=False)
    _write_run_config(args, dataset, inputs, out_dir, {
        "load_seconds": float(load_seconds),
        "feature_extraction_seconds": float(feature_extraction_seconds),
        "feature_checkpoint": feature_checkpoint_report,
        "processed_path": str(processed_path),
        "filtering_report_path": str(out_dir / "dataset_filtering_report.json"),
        "input_fingerprints": input_fingerprints,
        "methodology_configuration": methodology_configuration,
        "methodology_config_sha256": methodology_config_sha256,
    })
    # El checkpoint se elimina sólo después de que processed + reporte + config
    # quedaron escritos. Así nunca existe una ventana sin ruta de reanudación.
    _remove_feature_checkpoint(feature_checkpoint_report)

    X = df[FIXED_FEATURES].fillna(0).astype(np.float32).to_numpy()
    y_labels = df[LABEL_COL].astype(str).to_numpy()
    le = LabelEncoder()
    y = le.fit_transform(y_labels)
    classes = [str(c) for c in le.classes_]
    if len(classes) < 2:
        raise SystemExit(f"Need at least 2 classes to train {dataset}.")
    expected_profile = _dataset_profile(dataset)
    expected_classes = tuple(expected_profile["expected_classes"])
    if tuple(sorted(classes)) != expected_classes:
        raise RuntimeError(
            f"El entrenamiento no conserva las clases exactas de {dataset}: {classes!r}"
        )

    train_idx, test_idx, stratified = split_indices(y_labels, float(args.test_size), int(args.seed))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    df_test_raw = df.iloc[test_idx].copy()
    observed_test_counts = {
        str(k): int(v) for k, v in pd.Series(y_labels[test_idx]).value_counts().items()
    }
    expected_test_counts = dict(expected_profile["expected_test_counts"])
    if observed_test_counts != expected_test_counts:
        raise RuntimeError(
            f"El split de {dataset} no coincide con los soportes validados: "
            f"observado={observed_test_counts!r}; esperado={expected_test_counts!r}."
        )

    model_selection_rows: List[Dict[str, Any]] = []
    model_selection_seconds = 0.0
    if bool(args.model_selection):
        winner, model_selection_rows, model_selection_seconds = select_model_on_inner_validation(
            args, X_train, y_train, out_dir
        )
        weight_mode = str(winner["weight_mode"])
        selected_candidate_name = str(winner["candidate"])
        model = _histgb_model_from_params(args, dict(winner["params"]))
    else:
        weight_mode = str(args.sample_weight_mode or default_weight_mode_for_dataset(dataset))
        selected_candidate_name = "fixed_cli_configuration"
        model = make_histgb_model(args)
    sw = make_sample_weight_vector(y_train, mode=weight_mode, clip=float(args.sample_weight_clip))

    t0 = time.perf_counter()
    cpu0 = time.process_time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
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
    benchmark_wamm: Dict[str, Any] = {}
    if dataset == "egipcios":
        benchmark_wamm = {
            "comparison_scope": "same_DS_Augmented_v2_native9_classes_and_exact_test_supports",
            "paper_model": "WAMM_XGBoost",
            "our_model": "HistGradientBoostingClassifier",
            "paper_accuracy": WAMM_PAPER_ACCURACY,
            "our_accuracy": float(metrics["accuracy"]),
            "accuracy_delta_percentage_points": 100.0 * (float(metrics["accuracy"]) - WAMM_PAPER_ACCURACY),
            "paper_f1_macro": WAMM_PAPER_F1_MACRO,
            "our_f1_macro": float(metrics["f1_macro"]),
            "f1_macro_delta_percentage_points": 100.0 * (float(metrics["f1_macro"]) - WAMM_PAPER_F1_MACRO),
            "beats_paper_accuracy": bool(float(metrics["accuracy"]) > WAMM_PAPER_ACCURACY),
            "beats_paper_f1_macro": bool(float(metrics["f1_macro"]) > WAMM_PAPER_F1_MACRO),
        }
        benchmark_wamm["beats_paper_both_primary_metrics"] = bool(
            benchmark_wamm["beats_paper_accuracy"] and benchmark_wamm["beats_paper_f1_macro"]
        )
        _save_json(benchmark_wamm, out_dir / "benchmark_vs_wamm.json")
        pd.DataFrame([
            {"metric": "accuracy", "ours": benchmark_wamm["our_accuracy"], "paper": WAMM_PAPER_ACCURACY,
             "delta_percentage_points": benchmark_wamm["accuracy_delta_percentage_points"], "beats_paper": benchmark_wamm["beats_paper_accuracy"]},
            {"metric": "f1_macro", "ours": benchmark_wamm["our_f1_macro"], "paper": WAMM_PAPER_F1_MACRO,
             "delta_percentage_points": benchmark_wamm["f1_macro_delta_percentage_points"], "beats_paper": benchmark_wamm["beats_paper_f1_macro"]},
        ]).to_csv(out_dir / "benchmark_vs_wamm.csv", index=False)
        normal_idx = classes.index("NORMAL")
        block_rate_rows: List[Dict[str, Any]] = []
        for class_idx, class_name in enumerate(classes):
            if class_name == "NORMAL":
                continue
            mask = y_test == class_idx
            ours_rate = float(np.mean(y_pred[mask] != normal_idx)) if np.any(mask) else float("nan")
            paper_rate = WAMM_PAPER_BLOCK_RATE.get(class_name)
            block_rate_rows.append({
                "class": class_name, "support_test": int(np.sum(mask)),
                "ours_block_rate": ours_rate, "paper_wamm_block_rate": paper_rate,
                "delta_percentage_points": 100.0 * (ours_rate - float(paper_rate)) if paper_rate is not None else None,
            })
        pd.DataFrame(block_rate_rows).to_csv(out_dir / "benchmark_wamm_per_class_block_rate.csv", index=False)

    loss_curve_paths, loss_curve_summary = compute_and_save_loss_curves(
        model, X_train, y_train, X_test, y_test, classes, args, out_dir, dataset
    )

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

    model_path = out_dir / MODEL_FILENAME
    joblib.dump({
        "task": "multiclass",
        "dataset": dataset,
        "model": model,
        "label_encoder": le,
        "features": FIXED_FEATURES,
        "feature_count": len(FIXED_FEATURES),
        "feature_policy": FEATURE_POLICY_NAME,
        "feature_set_version": FEATURE_SET_VERSION,
        "pipeline_run_version": PIPELINE_RUN_VERSION,
        "methodology_config_sha256": methodology_config_sha256,
        "label_schema_version": _dataset_label_schema_version(dataset),
        "label_specs": _dataset_profile(dataset),
        "classes": classes,
        "selected_model": f"hist_gradient_boosting__{selected_candidate_name}",
        "selected_model_family": "hist_gradient_boosting",
        "selected_candidate": selected_candidate_name,
        "model_selection_scope": "outer_train_only",
        "model_selection_candidates": model_selection_rows,
        "weight_mode": weight_mode,
        "input_fingerprints": input_fingerprints,
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
    selected_model = f"hist_gradient_boosting__{selected_candidate_name}"
    metrics_payload: Dict[str, Any] = {
        "created_at": _now_iso(),
        "task": "multiclass",
        "dataset": dataset,
        "dataset_label_mode": str(_dataset_profile(dataset)["dataset_label_mode"]),
        "label_schema_version": _dataset_label_schema_version(dataset),
        "methodology_sources": ({
            "wamm_paper": WAMM_PAPER_URL,
            "wamm_pattern_reference": WAMM_PATTERN_REFERENCE_URL,
        } if dataset == "egipcios" else {}),
        "feature_policy": FEATURE_POLICY_NAME,
        "feature_count": len(FIXED_FEATURES),
        "features_used": FIXED_FEATURES,
        "feature_set_version": FEATURE_SET_VERSION,
        "pipeline_run_version": PIPELINE_RUN_VERSION,
        "input_fingerprints": input_fingerprints,
        "methodology_config_sha256": methodology_config_sha256,
        "selected_model": selected_model,
        "selected_model_family": "hist_gradient_boosting",
        "model_backend": "sklearn.ensemble.HistGradientBoostingClassifier",
        "n_iter_actual": _safe_float(getattr(model, "n_iter_", None)),
        "max_iter_configured": _safe_float(getattr(model, "max_iter", None)),
        "gpu_requested": args.use_gpu,
        "gpu_visible": bool(gpu_visible),
        "gpu_used": False,
        "gpu_note": "HistGradientBoostingClassifier de scikit-learn no tiene backend CUDA; se acelera con CPU/OpenMP.",
        "weight_mode": weight_mode,
        "selected_candidate": selected_candidate_name,
        "model_selection_scope": "outer_train_only",
        "model_selection_candidates": model_selection_rows,
        "sample_weight_clip": float(args.sample_weight_clip),
        "classes": classes,
        "n_classes": int(len(classes)),
        "label_distribution": label_dist.to_dict(orient="records"),
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "test_size": float(args.test_size),
        "split_stratified": bool(stratified),
        "test_class_counts": observed_test_counts,
        "expected_test_class_counts": expected_test_counts,
        "seed": int(args.seed),
        "timing": {
            "load_seconds": float(load_seconds),
            "feature_extraction_seconds": float(feature_extraction_seconds),
            "model_selection_seconds": float(model_selection_seconds),
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
        "benchmark_vs_wamm": benchmark_wamm,
        "binary_normal_vs_attack_metrics": binary_metrics,
        "operational_metrics": operational_metrics,
        "loss_curve": loss_curve_summary,
        "loss_curve_config": {
            "loss_curve": bool(args.loss_curve),
            "loss_curve_max_rows": int(args.loss_curve_max_rows),
            "loss_curve_min_rows_per_class": int(args.loss_curve_min_rows_per_class),
            "loss_curve_every_n_iter": int(args.loss_curve_every_n_iter),
            "epoch_definition": "HistGradientBoostingClassifier: epoch = boosting iteration",
        },
        "feature_extraction_config": {
            "resume_enabled": bool(getattr(args, "resume", False)),
            "checkpoint_rows": int(args.feature_checkpoint_rows),
            "row_timeout_seconds": float(args.feature_row_timeout_seconds),
            "resumed_rows": int(feature_checkpoint_report.get("resumed_rows", 0) or 0),
            "processed_cache_reused": bool(feature_checkpoint_report.get("processed_cache_reused", False)),
        },
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
            "metrics": str(out_dir / METRICS_FILENAME),
            "metrics_flat": str(out_dir / "metrics_flat.csv"),
            "confusion_matrix": str(out_dir / "confusion_matrix_test.csv"),
            "confusion_matrix_normalized_true": str(out_dir / "confusion_matrix_test_normalized_true.csv"),
            "classification_report_csv": str(out_dir / "classification_report_test.csv"),
            "classification_report_json": str(out_dir / "classification_report_test.json"),
            "per_class_ovr_metrics": str(out_dir / "per_class_ovr_metrics.csv"),
            "binary_normal_vs_attack_metrics": str(out_dir / "binary_normal_vs_attack_metrics.json"),
            "predictions": str(out_dir / "test_predictions.csv"),
            "operational_metrics": str(out_dir / "operational_metrics.json"),
            **({
                "benchmark_vs_wamm_json": str(out_dir / "benchmark_vs_wamm.json"),
                "benchmark_vs_wamm_csv": str(out_dir / "benchmark_vs_wamm.csv"),
                "benchmark_wamm_per_class_block_rate": str(out_dir / "benchmark_wamm_per_class_block_rate.csv"),
            } if dataset == "egipcios" else {}),
            "features_used": str(out_dir / "features_used.csv"),
            "feature_selection_manifest": str(out_dir / "feature_selection_manifest.csv"),
            "label_mapping": str(out_dir / "label_mapping.csv"),
            "label_validation": str(out_dir / "label_schema_validation.json"),
            "model_selection_json": str(out_dir / "model_selection_inner_validation.json"),
            "model_selection_csv": str(out_dir / "model_selection_inner_validation.csv"),
            **loss_curve_paths,
            **fi_paths,
        },
    }
    _save_json(metrics_payload, out_dir / METRICS_FILENAME)

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
        out_dir / METRICS_FILENAME,
        out_dir / "metrics_flat.csv",
        out_dir / "selected_model_summary.txt",
        out_dir / "confusion_matrix_test.csv",
        out_dir / "classification_report_test.csv",
        out_dir / "per_class_ovr_metrics.csv",
        out_dir / "test_predictions.csv",
        out_dir / "loss_curve_by_epoch.csv",
        out_dir / "loss_curve_histgb.csv",
        out_dir / "error_curve_mse_rmse_by_epoch.csv",
        out_dir / "loss_curve_log_loss.png",
        out_dir / "loss_curve_histgb.png",
        out_dir / "loss_curve_mse.png",
        out_dir / "loss_curve_rmse.png",
        out_dir / "prediction_error_curve_histgb.png",
        out_dir / "loss_curve_metadata.json",
        out_dir / "feature_importance.csv",
        out_dir / "operational_metrics.json",
    ]:
        if Path(p).exists():
            print(f"  {p}", flush=True)
    required_final = _dataset_required_artifacts(
        out_dir,
        dataset=dataset,
        require_loss_curve=bool(args.loss_curve),
        require_fi_per_class=bool(args.fi_per_class),
    )
    missing_final = [str(path) for path in required_final if not path.is_file() or path.stat().st_size <= 0]
    if missing_final:
        raise RuntimeError("La corrida terminó sin todos los artefactos obligatorios: " + ", ".join(missing_final))
    if not _analysis_artifacts_valid(
        out_dir,
        dataset,
        classes,
        FIXED_FEATURES,
        expected_test_rows=len(y_test),
        require_loss_curve=bool(args.loss_curve),
    ):
        raise RuntimeError(
            "Los artefactos analíticos existen pero su contenido está incompleto o es inconsistente "
            "(métricas por clase, feature importance, operación, predicciones o curvas)."
        )
    _save_json(
        {
            "status": "complete",
            "completed_at": _now_iso(),
            "dataset": dataset,
            "feature_set_version": FEATURE_SET_VERSION,
            "label_schema_version": _dataset_label_schema_version(dataset),
            "pipeline_run_version": PIPELINE_RUN_VERSION,
            "methodology_config_sha256": methodology_config_sha256,
            "input_fingerprints": input_fingerprints,
            "metrics": str(out_dir / METRICS_FILENAME),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
        out_dir / "run_status.json",
    )
    return metrics_payload


def flatten_metrics_for_comparison(m: Dict[str, Any]) -> Dict[str, Any]:
    tm = m.get("test_metrics") or {}
    bm = m.get("binary_normal_vs_attack_metrics") or {}
    om = m.get("operational_metrics") or {}
    timing = m.get("timing") or {}
    res = m.get("resource_usage") or {}
    lc = m.get("loss_curve") or {}
    return {
        "dataset": m.get("dataset"),
        "task": m.get("task"),
        "feature_set_version": m.get("feature_set_version"),
        "selected_model": m.get("selected_model"),
        "selected_model_family": m.get("selected_model_family"),
        "weight_mode": m.get("weight_mode"),
        "train_rows": m.get("train_rows"),
        "test_rows": m.get("test_rows"),
        "n_classes": m.get("n_classes"),
        "n_iter_actual": m.get("n_iter_actual"),
        "loss_curve_available": lc.get("available"),
        "loss_curve_epochs_recorded": lc.get("n_epochs_recorded"),
        "loss_curve_final_test_log_loss": lc.get("final_test_log_loss"),
        "loss_curve_final_test_mse": lc.get("final_test_mse"),
        "loss_curve_final_test_rmse": lc.get("final_test_rmse"),
        "loss_curve_final_test_error_rate": lc.get("final_test_error_rate"),
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
    lc = m.get("loss_curve") or {}
    wb = m.get("benchmark_vs_wamm") or {}
    lines = [
        "THREE_DATASETS_UNIFIED_HISTGB_COMPACT73",
        f"dataset={m.get('dataset')}",
        f"task={m.get('task')}",
        f"selected_model={m.get('selected_model')}",
        f"selected_model_family={m.get('selected_model_family')}",
        f"weight_mode={m.get('weight_mode')}",
        f"n_iter_actual={m.get('n_iter_actual')}",
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
        f"wamm_paper_accuracy={wb.get('paper_accuracy')}",
        f"wamm_paper_f1_macro={wb.get('paper_f1_macro')}",
        f"beats_wamm_accuracy={wb.get('beats_paper_accuracy')}",
        f"beats_wamm_f1_macro={wb.get('beats_paper_f1_macro')}",
        f"beats_wamm_both={wb.get('beats_paper_both_primary_metrics')}",
        f"loss_curve_available={lc.get('available')}",
        f"loss_curve_epochs_recorded={lc.get('n_epochs_recorded')}",
        f"loss_curve_final_test_log_loss={lc.get('final_test_log_loss')}",
        f"loss_curve_final_test_mse={lc.get('final_test_mse')}",
        f"loss_curve_final_test_rmse={lc.get('final_test_rmse')}",
        f"loss_curve_final_test_error_rate={lc.get('final_test_error_rate')}",
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
    merged: Dict[str, Dict[str, Any]] = {}
    for payload in payloads:
        dataset = str(payload.get("dataset") or "").strip().lower()
        if dataset:
            merged[dataset] = payload
    for dataset in ("torpeda", "harvard", "egipcios"):
        if dataset in merged:
            continue
        path = root / dataset / METRICS_FILENAME
        if path.is_file():
            candidate = _load_json_file(path)
            if candidate.get("test_metrics"):
                merged[dataset] = candidate
    ordered_payloads = [merged[name] for name in ("torpeda", "harvard", "egipcios") if name in merged]

    rows = [flatten_metrics_for_comparison(p) for p in ordered_payloads]
    df = pd.DataFrame(rows)
    comparison_csv = root / "comparison_integrated_3datasets_compact73.csv"
    comparison_json = root / "comparison_integrated_3datasets_compact73.json"
    df.to_csv(comparison_csv, index=False)
    _save_json({
        "created_at": _now_iso(),
        "task": "multiclass",
        "model_family": "hist_gradient_boosting",
        "feature_count": len(FIXED_FEATURES),
        "features": FIXED_FEATURES,
        "datasets": [p.get("dataset") for p in ordered_payloads],
        "rows": rows,
        "artifacts": {
            "comparison_csv": str(comparison_csv),
            "comparison_json": str(comparison_json),
        },
    }, comparison_json)
    _save_json({
        "created_at": _now_iso(),
        "root": str(Path(args.definitivo_dir)),
        "generated_task": "multiclase",
        "label_schema_version": LABEL_SCHEMA_VERSION,
        "structure": {
            "multiclase": {
                p.get("dataset"): (p.get("artifacts") or {}) for p in ordered_payloads
            },
        },
    }, Path(args.definitivo_dir) / "manifest_3datasetsMulticlase3.json")
    print(f"[OK] Aggregate comparison: {comparison_csv}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Pipeline unificado compact73: TorpEda + Harvard + Egipcios.")
    ap.add_argument("--dataset", default="all", choices=["torpeda", "harvard", "egipcios", "both", "all", "three", "3datasets"], help="Default all: ejecuta los tres datasets con el mismo pipeline.")
    ap.add_argument("--project-dir", default=DEFAULT_PROJECT_DIR)
    ap.add_argument("--definitivo-dir", "--result-dir", dest="definitivo_dir", default=None, help=f"Default: PROJECT_DIR/{DEFAULT_RESULT_DIR_NAME}")
    ap.add_argument("--inputs", nargs="*", default=None, help="Inputs para un dataset único. TorpEda: XML. Harvard: CSV/TSV/GZ.")
    ap.add_argument("--torpeda-inputs", nargs="*", default=None, help="Inputs TorpEda cuando --dataset both.")
    ap.add_argument("--harvard-inputs", nargs="*", default=None, help="Inputs Harvard/SR-BH cuando --dataset both/all.")
    ap.add_argument("--egipcios-inputs", nargs="*", default=None, help="Inputs Egipcios CSV/TSV cuando --dataset all/egipcios.")
    ap.add_argument("--output-dir", default=None, help="Sólo para un dataset; sobreescribe su subdirectorio.")

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
        default="top10-request-groups",
        choices=["top10-request-groups", "top10-request-group", "request-groups", "request-group", "top14-request-groups", "top14-request-group", "top10-operational", "top10-operational-groups", "top10-operational-group", "top14-combo", "top14-combos", "top14-combination", "top14-combinations", "exact-top14-combo", "exact-top14-combos", "optimized-family", "optimized_family", "optimized", "coarse", "coarse-family", "coarse_family", "family", "mapped", "attack-family", "attack_families", "native", "capec", "srbh", "legacy-common-anomalous", "legacy_common_anomalous"],
    )
    ap.add_argument("--harvard-multiclass-strategy", default="severity", choices=["severity", "first"])

    # Egipcios / DS_Augmented_v2_csv CSV: col1=request HTTP completa, col2=label textual.
    ap.add_argument("--egipcios-sep", default="auto")
    ap.add_argument("--egipcios-request-col", default=EGIPCIOS_DEFAULT_REQUEST_COL)
    ap.add_argument("--egipcios-label-col", default=EGIPCIOS_DEFAULT_LABEL_COL)
    ap.add_argument("--egipcios-normal-regex", default=EGIPCIOS_DEFAULT_NORMAL_REGEX)
    ap.add_argument("--egipcios-min-class-count", type=int, default=2)
    ap.add_argument("--egipcios-strict-paper-dataset", action=argparse.BooleanOptionalAction, default=True, help="Exige 704.665 filas y la distribución exacta de las nueve clases.")

    # Train/test.
    ap.add_argument("--test-size", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--histgb-preset", default="max", choices=["fast", "strong", "max"])
    ap.add_argument("--sample-weight-mode", default="", help="Sólo se usa si se desactiva la selección interna.")
    ap.add_argument("--sample-weight-clip", type=float, default=20.0)
    ap.add_argument("--model-selection", "--egipcios-model-selection", dest="model_selection", action=argparse.BooleanOptionalAction, default=True, help="Elige la potencia de pesos dentro de outer-train para los tres datasets.")
    ap.add_argument("--model-selection-max-train-rows", type=int, default=250000, help="Cap estratificado por candidato; 0 usa todo inner-train.")
    ap.add_argument("--model-selection-validation-size", type=float, default=0.10)
    ap.add_argument("--loss-curve", action=argparse.BooleanOptionalAction, default=True, help="Guarda curva por época/iteración con log-loss, MSE, RMSE y error de predicción.")
    ap.add_argument("--loss-curve-max-rows", type=int, default=50000, help="Máximo de filas por split para curvas. 0=usar todo train/test.")
    ap.add_argument("--loss-curve-min-rows-per-class", type=int, default=200, help="Mínimo por clase al muestrear la curva.")
    ap.add_argument("--loss-curve-every-n-iter", type=int, default=1, help="Guardar cada N épocas/iteraciones. 1=por cada época.")
    ap.add_argument("--use-gpu", default="auto", choices=["auto", "on", "off"], help="Se registra pero HistGB de sklearn no usa CUDA.")

    # Paralelismo/operativas.
    ap.add_argument("--feature-workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    ap.add_argument("--feature-chunksize", type=int, default=256)
    ap.add_argument("--feature-checkpoint-rows", type=int, default=10000, help="Filas por checkpoint atómico de extracción.")
    ap.add_argument("--feature-row-timeout-seconds", type=float, default=30.0, help="Fusible por request; 0 lo desactiva.")
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

    ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False, help="Reutiliza checkpoints/features válidos de una corrida interrumpida.")
    ap.add_argument("--skip-complete", action=argparse.BooleanOptionalAction, default=False, help="No recalcula datasets cuyos artefactos finales estén completos.")
    ap.add_argument("--clean-output-dir", action="store_true")
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args(argv)

    if args.definitivo_dir is None:
        args.definitivo_dir = str(Path(args.project_dir) / DEFAULT_RESULT_DIR_NAME)
    if args.resume and args.clean_output_dir:
        raise SystemExit("--resume y --clean-output-dir son incompatibles: limpiar borraría el estado que se quiere reanudar.")
    if int(args.feature_checkpoint_rows) < 1:
        raise SystemExit("--feature-checkpoint-rows debe ser >= 1.")
    if float(args.feature_row_timeout_seconds) < 0:
        raise SystemExit("--feature-row-timeout-seconds debe ser >= 0.")
    if args.dataset in {"both", "all", "three", "3datasets"} and args.output_dir:
        raise SystemExit("--output-dir sólo se usa con un dataset individual.")
    if args.dataset in {"both", "all", "three", "3datasets"} and args.inputs:
        raise SystemExit("--inputs sólo se usa con un dataset individual; use --torpeda-inputs/--harvard-inputs/--egipcios-inputs.")
    if not args.check_only:
        forbidden = []
        if int(args.sample_n or 0) != 0:
            forbidden.append("--sample-n debe ser 0")
        if int(args.max_normal_rows or 0) != 0:
            forbidden.append("--max-normal-rows debe ser 0")
        if str(args.keep_labels_regex or "").strip():
            forbidden.append("--keep-labels-regex debe estar vacío")
        if str(args.drop_labels_regex or "").strip():
            forbidden.append("--drop-labels-regex debe estar vacío")
        if not bool(args.egipcios_strict_paper_dataset):
            forbidden.append("--egipcios-strict-paper-dataset debe permanecer activo")
        if not bool(args.model_selection):
            forbidden.append("--model-selection debe permanecer activo")
        if not bool(args.loss_curve):
            forbidden.append("--loss-curve debe permanecer activo")
        if not bool(args.fi_per_class):
            forbidden.append("--fi-per-class debe permanecer activo")
        if not bool(args.fi_binary_normal_attack):
            forbidden.append("--fi-binary-normal-attack debe permanecer activo")
        if not bool(args.torpeda_only_common_labels):
            forbidden.append("--torpeda-only-common-labels debe permanecer activo")
        if str(args.harvard_label_mode) != "top10-request-groups":
            forbidden.append("--harvard-label-mode debe ser top10-request-groups")
        if not math.isclose(float(args.test_size), 0.20, rel_tol=0.0, abs_tol=1e-12):
            forbidden.append("--test-size debe ser 0.20")
        if int(args.seed) != 42:
            forbidden.append("--seed debe ser 42")
        if forbidden:
            raise SystemExit("Corrida fuera del perfil unificado reproducible: " + "; ".join(forbidden))
    if int(args.model_selection_max_train_rows) < 0:
        raise SystemExit("--model-selection-max-train-rows debe ser >= 0")
    if not 0.02 <= float(args.model_selection_validation_size) <= 0.30:
        raise SystemExit("--model-selection-validation-size debe estar entre 0.02 y 0.30")
    if not args.torpeda_inputs:
        args.torpeda_inputs = [os.environ.get("TORPEDA_RAW_DIR", str(Path(args.project_dir) / "data" / "raw" / "torpeda"))]
    if not args.harvard_inputs:
        args.harvard_inputs = [os.environ.get("HARVARD_RAW_DIR", str(Path(args.project_dir) / "data" / "raw" / "harvard"))]
    if not args.egipcios_inputs:
        args.egipcios_inputs = [os.environ.get("EGIPCIOS_RAW_DIR", str(Path(args.project_dir) / "egipcios" / "DS_Augmented_v2_csv" / "combined_data.csv"))]
    return args


def dependency_check(require_matplotlib: bool = False) -> Dict[str, Any]:
    import importlib.util
    required = ["numpy", "pandas", "sklearn", "joblib", "tqdm"]
    if require_matplotlib:
        required.append("matplotlib")
    missing = [p for p in required if importlib.util.find_spec(p) is None]
    parquet = importlib.util.find_spec("pyarrow") is not None or importlib.util.find_spec("fastparquet") is not None
    matplotlib_available = importlib.util.find_spec("matplotlib") is not None
    info = {"missing": missing, "parquet_available": bool(parquet), "matplotlib_available": bool(matplotlib_available), "python": sys.executable}
    if missing:
        raise SystemExit("Faltan dependencias base: " + ", ".join(missing))
    if not parquet:
        print("[WARN] pyarrow/fastparquet no detectado: processed se guardará como CSV fallback.", flush=True)
    return info


def feature_regex_self_check(max_seconds: float = 5.0) -> Dict[str, Any]:
    """Regresión rápida contra los payloads que antes bloqueaban un worker."""
    started = time.perf_counter()
    stress_cases = [
        ("ssti_quotes", PATTERNS["ssti_polyglot"], "'" * 10000),
        ("format_zeros", PATTERNS["format_string"], "%" + "0" * 10000),
        ("ldap_unclosed", PATTERNS["ldap"], "(a=" * 10000),
        ("ifs_unclosed", PATTERNS["shell_operator"], "${IFS" * 10000),
    ]
    counts = {name: _count_patterns(text, patterns) for name, patterns, text in stress_cases}
    positives = {
        "ssti_polyglot": _count_patterns("'\"`'\"`{{7*7}}", PATTERNS["ssti_polyglot"]),
        "format_string": _count_patterns("%08x %lld %*.*f", PATTERNS["format_string"]),
        "ldap": _count_patterns("(&(uid=foo)(cn=bar))", PATTERNS["ldap"]),
        "ifs": _count_patterns("${IFS} ${IFS%?}", PATTERNS["shell_operator"]),
    }
    elapsed = time.perf_counter() - started
    missing = [name for name, count in positives.items() if int(count) <= 0]
    if missing:
        raise RuntimeError("Regex self-check perdió detecciones: " + ", ".join(missing))
    if elapsed > float(max_seconds):
        raise RuntimeError(f"Regex self-check demasiado lento: {elapsed:.3f}s > {max_seconds:.3f}s")
    return {"elapsed_seconds": elapsed, "stress_counts": counts, "positive_counts": positives}


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    dep = dependency_check(require_matplotlib=bool(args.loss_curve))
    Path(args.definitivo_dir).mkdir(parents=True, exist_ok=True)
    _save_json({"created_at": _now_iso(), "dependency_check": dep, "args": vars(args)}, Path(args.definitivo_dir) / "last_invocation.json")

    if args.check_only:
        regex_check = feature_regex_self_check()
        label_check = {
            "capec88": _normalize_egipcios_label("88 - OS Command Injection"),
            "capec248": _normalize_egipcios_label("248 - Command Injection"),
            "capec1336": _normalize_egipcios_label("1336 - SSTI"),
        }
        if label_check["capec88"] == label_check["capec248"]:
            raise RuntimeError("Self-check inválido: CAPEC-88 y CAPEC-248 quedaron fusionadas.")
        manifest = feature_selection_manifest_frame()
        if int(manifest["selected"].sum()) != 73:
            raise RuntimeError("Self-check inválido: el vector final no tiene 73 features.")
        schema_check = {
            ds: {
                "classes": len(_dataset_profile(ds)["expected_classes"]),
                "full_rows": sum(_dataset_profile(ds)["expected_full_counts"].values()),
                "test_rows": sum(_dataset_profile(ds)["expected_test_counts"].values()),
            }
            for ds in ("torpeda", "harvard", "egipcios")
        }
        print(f"[OK] Regex safety self-check: {regex_check['elapsed_seconds']:.4f}s", flush=True)
        print(f"[OK] Label self-check: {label_check}", flush=True)
        print(f"[OK] Dataset schema self-check: {schema_check}", flush=True)
        print("[OK] Feature self-check: 73 seleccionadas; CAPEC-88/CAPEC-248 separadas. No entreno.", flush=True)
        return 0

    if args.dataset == "both":
        datasets = ["torpeda", "harvard"]
    elif args.dataset in {"all", "three", "3datasets"}:
        datasets = ["torpeda", "harvard", "egipcios"]
    else:
        datasets = [args.dataset]
    payloads: List[Dict[str, Any]] = []
    for ds in datasets:
        try:
            payload = train_eval_one_dataset(args, ds)
            payloads.append(payload)
        except BaseException as e:
            out_dir = _dataset_out_dir(args, ds)
            out_dir.mkdir(parents=True, exist_ok=True)
            try:
                _save_json(
                    {
                        "status": "failed",
                        "failed_at": _now_iso(),
                        "dataset": ds,
                        "feature_set_version": FEATURE_SET_VERSION,
                        "label_schema_version": _dataset_label_schema_version(ds),
                        "pipeline_run_version": PIPELINE_RUN_VERSION,
                        "error_type": type(e).__name__,
                        "error": str(e),
                        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                    },
                    out_dir / "run_status.json",
                )
            except Exception:
                pass
            raise
    write_aggregate_outputs(args, payloads)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
