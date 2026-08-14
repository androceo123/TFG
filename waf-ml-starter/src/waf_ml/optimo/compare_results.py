"""compare_results.py

Compara las metricas de dos experimentos OCSVM sobre el dataset CSIC:
  - Experimento BASE:           dataset completo  (resultsOptimo/csic/oneclass/)
  - Experimento SIN_REGISTRO:   sin aplicacion registro (resultsOptimo/csic_sin_registro/oneclass/)

Uso:
    python src/waf_ml/optimo/compare_results.py
    python src/waf_ml/optimo/compare_results.py --base resultsOptimo/csic/oneclass/metrics.json \
        --exp resultsOptimo/csic_sin_registro/oneclass/metrics.json \
        --out resultsOptimo/comparacion_csic_vs_sin_registro.json

Salidas:
    - Tabla comparativa en consola
    - JSON con diferencias absolutas y relativas
    - CSV con tabla de comparacion
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Metricas de interes para la comparacion (en orden de importancia para el TFG)
# ---------------------------------------------------------------------------
METRICS_OF_INTEREST: List[Tuple[str, str]] = [
    ("f1",                  "F1 Score"),
    ("balanced_accuracy",   "Balanced Accuracy"),
    ("precision",           "Precision"),
    ("recall",              "Recall (TPR)"),
    ("mcc",                 "MCC"),
    ("roc_auc",             "ROC-AUC"),
    ("pr_auc",              "PR-AUC"),
    ("accuracy",            "Accuracy"),
    ("specificity",         "Specificity (TNR)"),
    ("fpr",                 "FPR"),
    ("fnr",                 "FNR"),
]

# Metricas de confusion matrix
CM_KEYS = ["tn", "fp", "fn", "tp"]

# Metricas de benchmark de latencia
BENCHMARK_METRICS: List[Tuple[str, str]] = [
    ("mean_ms",  "Latencia media (ms)"),
    ("p50_ms",   "Latencia p50 (ms)"),
    ("p95_ms",   "Latencia p95 (ms)"),
    ("p99_ms",   "Latencia p99 (ms)"),
]


def _load_json(path: str) -> Optional[Dict]:
    p = Path(path)
    if not p.exists():
        print(f"[WARN] Archivo no encontrado: {path}")
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _get_nested(d: Dict, *keys, default=None):
    """Navega por claves anidadas de forma segura."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k, default)
        if cur is None:
            return default
    return cur


def _fmt(v) -> str:
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return f"{v:.6f}"
    return str(v)


def _delta_str(base, exp) -> str:
    """Muestra la diferencia y la flecha de mejora/empeora."""
    if base is None or exp is None:
        return "N/A"
    diff = exp - base
    arrow = "+" if diff >= 0 else ""
    pct = (diff / abs(base) * 100) if base != 0 else float("inf")
    pct_str = f"({arrow}{pct:.2f}%)" if abs(pct) != float("inf") else ""
    return f"{arrow}{diff:.6f} {pct_str}"


def _improvement_symbol(key: str, base, exp) -> str:
    """Determina si el cambio es una mejora (↑), empeora (↓) o neutro (=)."""
    if base is None or exp is None:
        return " "
    diff = exp - base
    if abs(diff) < 1e-6:
        return "="
    # Para FPR y FNR, bajar es mejor
    lower_is_better = key in {"fpr", "fnr", "false_reject_rate"}
    if lower_is_better:
        return "↑ MEJORA" if diff < 0 else "↓ EMPEORA"
    return "↑ MEJORA" if diff > 0 else "↓ EMPEORA"


def compare_metrics(base_path: str, exp_path: str, out_json: Optional[str], out_csv: Optional[str]) -> None:
    base = _load_json(base_path)
    exp = _load_json(exp_path)

    if base is None or exp is None:
        print("[ERROR] No se pueden comparar: uno o ambos archivos de metricas no existen.")
        print("        Asegurate de haber ejecutado ambos experimentos primero.")
        return

    # Extraer metricas del nivel correcto (puede estar bajo 'metrics' o directo)
    def _metrics_dict(d: Dict) -> Dict:
        # Los JSON de ocsvmOptimo tienen las metricas directamente o bajo una clave 'eval'
        for key in ("eval", "metrics", "test"):
            if key in d and isinstance(d[key], dict):
                sub = d[key]
                # Verificar que tiene metricas reales
                if any(k in sub for k in ("f1", "accuracy", "roc_auc")):
                    return sub
        return d

    base_m = _metrics_dict(base)
    exp_m = _metrics_dict(exp)

    # -----------------------------------------------------------------------
    # Tabla de comparacion principal
    # -----------------------------------------------------------------------
    col_w = 22
    print()
    print("=" * 90)
    print("  COMPARACION DE RESULTADOS: CSIC COMPLETO vs CSIC SIN APLICACION REGISTRO")
    print("=" * 90)
    print(f"  BASE (completo):       {base_path}")
    print(f"  EXPERIMENTO (filtrado): {exp_path}")
    print("=" * 90)
    print(f"  {'Metrica':<30} {'BASE':>{col_w}} {'SIN_REGISTRO':>{col_w}} {'Diferencia':>{col_w}} {'Resultado'}")
    print("-" * 90)

    rows_for_export = []

    for key, label in METRICS_OF_INTEREST:
        b_val = base_m.get(key)
        e_val = exp_m.get(key)
        symbol = _improvement_symbol(key, b_val, e_val)
        delta = _delta_str(b_val, e_val)
        print(f"  {label:<30} {_fmt(b_val):>{col_w}} {_fmt(e_val):>{col_w}} {delta:>{col_w}} {symbol}")
        rows_for_export.append({
            "metrica": key,
            "etiqueta": label,
            "base": b_val,
            "sin_registro": e_val,
            "diferencia_absoluta": (e_val - b_val) if (b_val is not None and e_val is not None) else None,
            "resultado": symbol,
        })

    print("-" * 90)

    # Confusion matrix
    base_cm = base_m.get("confusion_matrix", {}) or {}
    exp_cm = exp_m.get("confusion_matrix", {}) or {}
    print(f"\n  {'Confusion Matrix':<30} {'BASE':>{col_w}} {'SIN_REGISTRO':>{col_w}}")
    print(f"  {'-'*70}")
    for k in CM_KEYS:
        print(f"  {k.upper():<30} {str(base_cm.get(k, 'N/A')):>{col_w}} {str(exp_cm.get(k, 'N/A')):>{col_w}}")

    # Info del dataset
    base_n = base_m.get("n")
    exp_n = exp_m.get("n")
    print(f"\n  {'N (filas evaluacion)':<30} {_fmt(base_n):>{col_w}} {_fmt(exp_n):>{col_w}}")

    # Benchmark de latencia (si existe)
    base_bench_path = base_path.replace("metrics.json", "benchmark.json")
    exp_bench_path = exp_path.replace("metrics.json", "benchmark.json")
    base_bench = _load_json(base_bench_path)
    exp_bench = _load_json(exp_bench_path)

    if base_bench or exp_bench:
        print(f"\n  {'Benchmark de latencia':<30} {'BASE':>{col_w}} {'SIN_REGISTRO':>{col_w}} {'Diferencia':>{col_w}}")
        print(f"  {'-'*90}")
        for bkey, blabel in BENCHMARK_METRICS:
            b_lat = _get_nested(base_bench or {}, "latency_single_row", bkey) if base_bench else None
            e_lat = _get_nested(exp_bench or {}, "latency_single_row", bkey) if exp_bench else None
            delta = _delta_str(b_lat, e_lat)
            print(f"  {blabel:<30} {_fmt(b_lat):>{col_w}} {_fmt(e_lat):>{col_w}} {delta:>{col_w}}")

    print("=" * 90)
    print()

    # -----------------------------------------------------------------------
    # Resumen ejecutivo
    # -----------------------------------------------------------------------
    mejoras = [r for r in rows_for_export if "MEJORA" in str(r.get("resultado", ""))]
    empeoradas = [r for r in rows_for_export if "EMPEORA" in str(r.get("resultado", ""))]

    print("  RESUMEN EJECUTIVO")
    print(f"  Metricas que MEJORAN al eliminar 'registro' ({len(mejoras)}):")
    for r in mejoras:
        print(f"    - {r['etiqueta']}: {_fmt(r['base'])} -> {_fmt(r['sin_registro'])} ({r['resultado']})")

    print(f"\n  Metricas que EMPEORAN al eliminar 'registro' ({len(empeoradas)}):")
    for r in empeoradas:
        print(f"    - {r['etiqueta']}: {_fmt(r['base'])} -> {_fmt(r['sin_registro'])} ({r['resultado']})")

    neutral = len(rows_for_export) - len(mejoras) - len(empeoradas)
    print(f"\n  Metricas sin cambio significativo: {neutral}")
    print()

    # -----------------------------------------------------------------------
    # Guardar JSON de comparacion
    # -----------------------------------------------------------------------
    if out_json:
        export = {
            "experimentos": {
                "base": base_path,
                "sin_registro": exp_path,
            },
            "metricas": rows_for_export,
            "resumen": {
                "n_mejoras": len(mejoras),
                "n_empeoradas": len(empeoradas),
                "n_neutras": neutral,
                "mejoras": [r["etiqueta"] for r in mejoras],
                "empeoradas": [r["etiqueta"] for r in empeoradas],
            },
            "confusion_matrix": {
                "base": base_cm,
                "sin_registro": exp_cm,
            },
        }
        os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(export, f, ensure_ascii=False, indent=2, default=str)
        print(f"  Comparacion guardada en: {out_json}")

    # -----------------------------------------------------------------------
    # Guardar CSV de comparacion
    # -----------------------------------------------------------------------
    if out_csv:
        import csv
        os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["metrica", "etiqueta", "base", "sin_registro", "diferencia_absoluta", "resultado"])
            writer.writeheader()
            writer.writerows(rows_for_export)
        print(f"  CSV guardado en: {out_csv}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Compara metricas OCSVM CSIC completo vs sin aplicacion registro.")
    ap.add_argument(
        "--base",
        default="resultsOptimo/csic/oneclass/metrics.json",
        help="JSON de metricas del experimento BASE (dataset completo).",
    )
    ap.add_argument(
        "--exp",
        default="resultsOptimo/csic_sin_registro/oneclass/metrics.json",
        help="JSON de metricas del experimento SIN REGISTRO.",
    )
    ap.add_argument(
        "--out",
        default="resultsOptimo/comparacion_csic_vs_sin_registro.json",
        help="Ruta de salida para el JSON de comparacion.",
    )
    ap.add_argument(
        "--out-csv",
        default="resultsOptimo/comparacion_csic_vs_sin_registro.csv",
        help="Ruta de salida para el CSV de comparacion.",
    )
    args = ap.parse_args()
    compare_metrics(args.base, args.exp, args.out, args.out_csv)


if __name__ == "__main__":
    main()
