"""generar_reporte_pdf.py

Genera un reporte PDF con los resultados del experimento OCSVM
sobre el dataset CSIC (con o sin aplicacion registro).

Uso:
    python3.11 src/waf_ml/optimo/generar_reporte_pdf.py \
        --metrics  resultsOptimo/csic_sin_registro/oneclass_test1000/metrics.json \
        --benchmark resultsOptimo/csic_sin_registro/oneclass_test1000/benchmark.json \
        --fi       resultsOptimo/csic_sin_registro/oneclass_test1000/feature_importance_shap.csv \
        --search   resultsOptimo/csic_sin_registro/oneclass_test1000/search.csv \
        --out      resultsOptimo/reporte_csic_sin_registro_1000.pdf \
        --titulo   "Experimento CSIC sin Registro - Prueba 1000 filas"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(path: Optional[str]):
    if not path or not Path(path).exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_csv(path: Optional[str]):
    if not HAS_PANDAS or not path or not Path(path).exists():
        return None
    return pd.read_csv(path)


def _safe(d: dict, *keys, default="N/A"):
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k, default)
        if cur is None:
            return default
    if isinstance(cur, float):
        return f"{cur:.4f}"
    return str(cur)


def _color(val, key):
    """Devuelve color segun si la metrica es buena o mala."""
    try:
        v = float(val)
    except Exception:
        return "black"
    if key in ("fpr", "fnr"):
        return "#c0392b" if v > 0.3 else "#27ae60" if v < 0.1 else "#e67e22"
    return "#27ae60" if v >= 0.8 else "#e67e22" if v >= 0.5 else "#c0392b"


# ---------------------------------------------------------------------------
# Paginas del PDF
# ---------------------------------------------------------------------------

def _page_portada(pdf: PdfPages, titulo: str, meta: dict) -> None:
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor("#1a1a2e")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("#1a1a2e")
    ax.axis("off")

    # Franja superior
    ax.add_patch(plt.Rectangle((0, 0.78), 1, 0.22, transform=ax.transAxes,
                                color="#16213e", zorder=1))

    ax.text(0.5, 0.88, "TFG — Web Application Firewall con ML",
            ha="center", va="center", fontsize=14, color="#a8dadc",
            fontweight="bold", transform=ax.transAxes, zorder=2)

    ax.text(0.5, 0.60, titulo,
            ha="center", va="center", fontsize=20, color="white",
            fontweight="bold", transform=ax.transAxes, wrap=True, zorder=2)

    # Linea divisoria
    ax.axhline(0.50, color="#a8dadc", linewidth=1.5, xmin=0.1, xmax=0.9)

    info_lines = [
        f"Modelo:       One-Class SVM  (SGDOneClassSVM + Nystroem RBF)",
        f"Dataset:      CSIC 2010  —  {meta.get('dataset', 'csic_features_sin_registro')}",
        f"Filas eval.:  {meta.get('n_eval', 'N/A')}",
        f"Aceleración:  GPU  (RAPIDS cuml.accel)",
        f"Fecha:        {datetime.now().strftime('%d/%m/%Y %H:%M')}",
    ]
    for i, line in enumerate(info_lines):
        ax.text(0.15, 0.44 - i * 0.07, line,
                ha="left", va="center", fontsize=12, color="#e0e0e0",
                fontfamily="monospace", transform=ax.transAxes)

    ax.text(0.5, 0.05,
            "Experimento solicitado por el Prof. Cristian Cappo\n"
            "Objetivo: evaluar impacto de eliminar la aplicacion 'registro' del dataset CSIC",
            ha="center", va="center", fontsize=10, color="#a8dadc",
            style="italic", transform=ax.transAxes)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _page_metricas(pdf: PdfPages, metrics: dict, titulo: str) -> None:
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.suptitle(f"Métricas de Efectividad — {titulo}",
                 fontsize=14, fontweight="bold", y=0.97)

    # Extraer bloque de metricas
    m = metrics
    for key in ("eval", "metrics", "test"):
        if key in metrics and isinstance(metrics[key], dict):
            if any(k in metrics[key] for k in ("f1", "accuracy")):
                m = metrics[key]
                break

    # ---- Tabla principal ----
    METRICAS = [
        ("F1 Score",            m.get("f1")),
        ("Balanced Accuracy",   m.get("balanced_accuracy")),
        ("Precision",           m.get("precision")),
        ("Recall (TPR)",        m.get("recall")),
        ("Specificity (TNR)",   m.get("specificity")),
        ("MCC",                 m.get("mcc")),
        ("ROC-AUC",             m.get("roc_auc")),
        ("PR-AUC",              m.get("pr_auc")),
        ("FPR",                 m.get("fpr")),
        ("FNR",                 m.get("fnr")),
        ("Accuracy",            m.get("accuracy")),
    ]

    ax_table = fig.add_axes([0.05, 0.35, 0.42, 0.55])
    ax_table.axis("off")

    col_labels = ["Métrica", "Valor"]
    table_data = []
    cell_colors = []
    for label, val in METRICAS:
        fval = f"{float(val):.4f}" if val is not None else "N/A"
        table_data.append([label, fval])
        key = label.lower().replace(" ", "_").split("(")[0].strip()
        c = _color(fval, key)
        cell_colors.append(["#f5f5f5", c])

    tbl = ax_table.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        cellColours=cell_colors,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 1.6)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")

    # ---- Confusion Matrix ----
    cm = m.get("confusion_matrix", {}) or {}
    tn = cm.get("tn", 0) or 0
    fp = cm.get("fp", 0) or 0
    fn = cm.get("fn", 0) or 0
    tp = cm.get("tp", 0) or 0
    n_total = tn + fp + fn + tp

    ax_cm = fig.add_axes([0.55, 0.42, 0.38, 0.48])
    cm_matrix = np.array([[tn, fp], [fn, tp]])
    im = ax_cm.imshow(cm_matrix, cmap="Blues", aspect="auto")
    ax_cm.set_xticks([0, 1])
    ax_cm.set_yticks([0, 1])
    ax_cm.set_xticklabels(["Pred: Normal", "Pred: Ataque"], fontsize=9)
    ax_cm.set_yticklabels(["Real: Normal", "Real: Ataque"], fontsize=9)
    ax_cm.set_title("Matriz de Confusión", fontsize=11, fontweight="bold", pad=8)
    for i in range(2):
        for j in range(2):
            val = cm_matrix[i, j]
            pct = f"\n({val/n_total*100:.1f}%)" if n_total > 0 else ""
            ax_cm.text(j, i, f"{val}{pct}", ha="center", va="center",
                       fontsize=11, fontweight="bold",
                       color="white" if cm_matrix[i, j] > cm_matrix.max() * 0.5 else "black")
    plt.colorbar(im, ax=ax_cm, shrink=0.8)

    # ---- Grafico de barras horizontal ----
    ax_bar = fig.add_axes([0.05, 0.05, 0.88, 0.25])
    bar_metrics = [(l, v) for l, v in METRICAS if v is not None and l not in ("FPR", "FNR", "Accuracy")]
    bar_labels = [l for l, _ in bar_metrics]
    bar_vals = [float(v) if v is not None else 0 for _, v in bar_metrics]
    colors = ["#27ae60" if v >= 0.8 else "#e67e22" if v >= 0.5 else "#c0392b" for v in bar_vals]
    bars = ax_bar.barh(bar_labels, bar_vals, color=colors, edgecolor="white", height=0.6)
    ax_bar.set_xlim(0, 1.05)
    ax_bar.axvline(0.8, color="gray", linestyle="--", linewidth=1, alpha=0.7, label="Umbral 0.8")
    ax_bar.set_xlabel("Valor", fontsize=9)
    ax_bar.legend(fontsize=8)
    for bar, val in zip(bars, bar_vals):
        ax_bar.text(min(val + 0.02, 1.0), bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", fontsize=8)

    # Info N
    fig.text(0.55, 0.37, f"N total evaluación: {n_total}  |  Normales: {tn+fp}  |  Ataques: {fn+tp}",
             ha="center", fontsize=9, color="#555")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _page_feature_importance(pdf: PdfPages, fi_df, titulo: str) -> None:
    if fi_df is None or len(fi_df) == 0:
        return

    # Detectar columna de importancia
    imp_col = None
    for c in ("shap_mean_abs", "importance", "mean_importance", "mean_abs_shap"):
        if c in fi_df.columns:
            imp_col = c
            break
    feat_col = None
    for c in ("feature", "Feature", "feature_name"):
        if c in fi_df.columns:
            feat_col = c
            break

    if imp_col is None or feat_col is None:
        return

    df = fi_df[[feat_col, imp_col]].dropna().sort_values(imp_col, ascending=True).tail(20)

    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    fig.suptitle(f"Importancia de Features (SHAP) — {titulo}",
                 fontsize=14, fontweight="bold")

    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(df)))
    bars = ax.barh(df[feat_col], df[imp_col], color=colors, edgecolor="white")
    ax.set_xlabel("SHAP Mean |value|", fontsize=11)
    ax.set_title("Top features por importancia media absoluta SHAP", fontsize=10, color="#555")

    for bar, val in zip(bars, df[imp_col]):
        ax.text(val + df[imp_col].max() * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.5f}", va="center", fontsize=8)

    ax.set_xlim(0, df[imp_col].max() * 1.2)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _page_benchmark(pdf: PdfPages, benchmark: dict, titulo: str) -> None:
    if not benchmark:
        return

    fig = plt.figure(figsize=(11.69, 8.27))
    fig.suptitle(f"Benchmark de Rendimiento — {titulo}",
                 fontsize=14, fontweight="bold", y=0.98)

    # Tabla de metricas de rendimiento
    ax_t = fig.add_axes([0.05, 0.45, 0.55, 0.48])
    ax_t.axis("off")

    lat = benchmark.get("latency_total") or benchmark.get("latency_single_row") or {}
    bench_rows = [
        ["Throughput (req/s)",    f"{benchmark.get('throughput_req_per_sec', 'N/A'):.2f}" if isinstance(benchmark.get('throughput_req_per_sec'), float) else "N/A"],
        ["Latencia media (ms)",   f"{lat.get('mean_ms', 'N/A'):.3f}" if isinstance(lat.get('mean_ms'), float) else "N/A"],
        ["Latencia p50 (ms)",     f"{lat.get('p50_ms', 'N/A'):.3f}" if isinstance(lat.get('p50_ms'), float) else "N/A"],
        ["Latencia p95 (ms)",     f"{lat.get('p95_ms', 'N/A'):.3f}" if isinstance(lat.get('p95_ms'), float) else "N/A"],
        ["Latencia p99 (ms)",     f"{lat.get('p99_ms', 'N/A'):.3f}" if isinstance(lat.get('p99_ms'), float) else "N/A"],
        ["Train time (s)",        f"{benchmark.get('train_time_seconds', 'N/A'):.3f}" if isinstance(benchmark.get('train_time_seconds'), float) else "N/A"],
        ["Tamaño modelo (bytes)", str(benchmark.get("model_size_bytes", "N/A"))],
        ["Failure rate",          f"{benchmark.get('failure_rate', 'N/A'):.4f}" if isinstance(benchmark.get('failure_rate'), float) else "N/A"],
        ["CPU util. %",           f"{benchmark.get('cpu_util_pct', 'N/A'):.1f}" if isinstance(benchmark.get('cpu_util_pct'), float) else "N/A"],
        ["Peak RSS (MB)",         f"{benchmark.get('peak_rss_mb', 'N/A'):.1f}" if isinstance(benchmark.get('peak_rss_mb'), float) else "N/A"],
    ]

    tbl = ax_t.table(
        cellText=bench_rows,
        colLabels=["Métrica", "Valor"],
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 1.7)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#ecf0f1")

    # Grafico de latencias
    ax_bar = fig.add_axes([0.65, 0.45, 0.30, 0.48])
    lat_labels = ["mean", "p50", "p95", "p99"]
    lat_keys   = ["mean_ms", "p50_ms", "p95_ms", "p99_ms"]
    lat_vals = [lat.get(k) for k in lat_keys]
    valid = [(l, v) for l, v in zip(lat_labels, lat_vals) if isinstance(v, (int, float))]
    if valid:
        ll, lv = zip(*valid)
        ax_bar.bar(ll, lv, color=["#3498db", "#2ecc71", "#f39c12", "#e74c3c"], edgecolor="white")
        ax_bar.set_ylabel("ms", fontsize=10)
        ax_bar.set_title("Latencia por percentil", fontsize=10)
        for i, (l, v) in enumerate(zip(ll, lv)):
            ax_bar.text(i, v + max(lv) * 0.02, f"{v:.2f}", ha="center", fontsize=9)

    # Modo y filas del benchmark
    mode = benchmark.get("mode", "N/A")
    rows_bench = benchmark.get("rows_sampled", "N/A")
    fig.text(0.05, 0.42, f"Modo: {mode}  |  Filas muestreadas: {rows_bench}",
             fontsize=9, color="#555")

    # Nota
    fig.text(0.05, 0.08,
             "Nota: El benchmark mide la latencia de inferencia por petición individual.\n"
             "Throughput y latencias son aproximados (entorno de prueba en nodo HPC compartido).\n"
             "Para un WAF en producción, los valores reales dependen del hardware dedicado.",
             fontsize=9, color="#777", style="italic")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _page_tuning(pdf: PdfPages, search_df, titulo: str) -> None:
    if search_df is None or len(search_df) == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(11.69, 8.27))
    fig.suptitle(f"Búsqueda de Hiperparámetros — {titulo}",
                 fontsize=14, fontweight="bold")

    score_col = None
    for c in ("score", "cv_score", "val_score", "normal_acceptance"):
        if c in search_df.columns:
            score_col = c
            break

    if score_col:
        df_sorted = search_df.sort_values(score_col, ascending=False).reset_index(drop=True)

        # Barras de candidatos
        ax = axes[0]
        colors = ["#27ae60" if i == 0 else "#3498db" for i in range(len(df_sorted))]
        ax.bar(range(len(df_sorted)), df_sorted[score_col], color=colors, edgecolor="white")
        ax.set_xlabel("Candidato (ordenado por score)", fontsize=10)
        ax.set_ylabel(score_col, fontsize=10)
        ax.set_title("Score por candidato", fontsize=11)
        if len(df_sorted) > 0:
            ax.axhline(df_sorted[score_col].iloc[0], color="#e74c3c",
                       linestyle="--", linewidth=1.5, label=f"Mejor: {df_sorted[score_col].iloc[0]:.4f}")
            ax.legend(fontsize=9)

        # Tabla top 5
        ax2 = axes[1]
        ax2.axis("off")
        top5 = df_sorted.head(5)
        cols_show = [c for c in ["nu", "gamma", "n_components", "max_iter", score_col] if c in top5.columns]
        tbl = ax2.table(
            cellText=top5[cols_show].round(5).values.tolist(),
            colLabels=cols_show,
            cellLoc="center",
            loc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1.1, 1.6)
        for (r, c_), cell in tbl.get_celld().items():
            if r == 0:
                cell.set_facecolor("#2c3e50")
                cell.set_text_props(color="white", fontweight="bold")
            elif r == 1:
                cell.set_facecolor("#d5f5e3")
        ax2.set_title("Top 5 candidatos", fontsize=11, fontweight="bold", pad=20)
    else:
        for ax in axes:
            ax.text(0.5, 0.5, "Datos de tuning no disponibles",
                    ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _page_conclusiones(pdf: PdfPages, metrics: dict, titulo: str) -> None:
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor("#fafafa")

    m = metrics
    for key in ("eval", "metrics", "test"):
        if key in metrics and isinstance(metrics[key], dict):
            if any(k in metrics[key] for k in ("f1", "accuracy")):
                m = metrics[key]
                break

    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    ax.text(0.5, 0.93, "Conclusiones del Experimento",
            ha="center", va="center", fontsize=16, fontweight="bold",
            color="#2c3e50", transform=ax.transAxes)
    ax.axhline(0.88, color="#2c3e50", linewidth=1, xmin=0.1, xmax=0.9)

    roc = m.get("roc_auc")
    pr  = m.get("pr_auc")
    f1  = m.get("f1")
    ba  = m.get("balanced_accuracy")
    cm  = m.get("confusion_matrix", {}) or {}

    observaciones = [
        f"• ROC-AUC = {float(roc):.4f}  — el modelo tiene buena capacidad discriminativa." if roc else "",
        f"• PR-AUC  = {float(pr):.4f}  — adecuado para datasets desbalanceados." if pr else "",
        f"• F1 = {float(f1):.4f} | Balanced Accuracy = {float(ba):.4f}" if f1 is not None and ba is not None else "",
        "",
        "• ADVERTENCIA: Esta es una prueba con solo 1000 filas muestreadas.",
        "  Con tan pocas muestras el modelo puede no aprender la frontera correctamente.",
        f"  → TN={cm.get('tn',0)}  FP={cm.get('fp',0)}  FN={cm.get('fn',0)}  TP={cm.get('tp',0)}",
        "",
        "• El filtrado de la aplicacion 'registro' elimino 10.201 filas del dataset original,",
        "  pasando de 97.065 a 86.864 filas totales.",
        "",
        "• Para obtener resultados representativos y comparables se recomienda:",
        "  → Ejecutar el experimento completo con el dataset sin registro (86.864 filas)",
        "  → Comparar con el baseline sobre el dataset original (97.065 filas)",
        "  → Evaluar si la eliminacion de 'registro' mejora F1, Balanced Accuracy y MCC.",
    ]

    y = 0.82
    for line in observaciones:
        if not line:
            y -= 0.025
            continue
        color = "#c0392b" if "ADVERTENCIA" in line else "#2c3e50"
        size = 11 if not line.startswith("  ") else 10
        ax.text(0.08, y, line, ha="left", va="top", fontsize=size,
                color=color, transform=ax.transAxes)
        y -= 0.055

    ax.add_patch(plt.Rectangle((0.06, 0.05), 0.88, 0.12,
                                transform=ax.transAxes, color="#d5e8d4",
                                linewidth=1.5, edgecolor="#82b366"))
    ax.text(0.5, 0.115,
            "Solicitud al Profesor: autorizar ejecucion del experimento completo\n"
            "con el dataset CSIC sin aplicacion 'registro' (86.864 filas, GPU HPC)",
            ha="center", va="center", fontsize=11, fontweight="bold",
            color="#2c3e50", transform=ax.transAxes)

    ax.text(0.5, 0.02, f"Generado: {datetime.now().strftime('%d/%m/%Y %H:%M')}",
            ha="center", fontsize=8, color="#aaa", transform=ax.transAxes)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Genera reporte PDF de resultados OCSVM.")
    ap.add_argument("--metrics",   default=None)
    ap.add_argument("--benchmark", default=None)
    ap.add_argument("--fi",        default=None, help="CSV feature importance")
    ap.add_argument("--search",    default=None, help="CSV resultados tuning")
    ap.add_argument("--out",       default="reporte_ocsvm.pdf")
    ap.add_argument("--titulo",    default="Experimento OCSVM CSIC")
    args = ap.parse_args()

    metrics   = _load_json(args.metrics)
    benchmark = _load_json(args.benchmark)
    fi_df     = _load_csv(args.fi)
    search_df = _load_csv(args.search)

    # Extraer meta info
    m = metrics
    for key in ("eval", "metrics", "test"):
        if key in metrics and isinstance(metrics[key], dict):
            if any(k in metrics[key] for k in ("f1", "accuracy", "n")):
                m = metrics[key]
                break
    meta = {
        "n_eval": m.get("n", "N/A"),
        "dataset": str(args.metrics or ""),
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(args.out) as pdf:
        print(f"[PDF] Generando portada...")
        _page_portada(pdf, args.titulo, meta)

        print(f"[PDF] Generando pagina de metricas...")
        _page_metricas(pdf, metrics, args.titulo)

        if fi_df is not None:
            print(f"[PDF] Generando pagina de feature importance...")
            _page_feature_importance(pdf, fi_df, args.titulo)

        if benchmark:
            print(f"[PDF] Generando pagina de benchmark...")
            _page_benchmark(pdf, benchmark, args.titulo)

        if search_df is not None:
            print(f"[PDF] Generando pagina de tuning...")
            _page_tuning(pdf, search_df, args.titulo)

        print(f"[PDF] Generando pagina de conclusiones...")
        _page_conclusiones(pdf, metrics, args.titulo)

        d = pdf.infodict()
        d["Title"] = args.titulo
        d["Author"] = "TFG WAF-ML — Juan & Andres"
        d["Subject"] = "Resultados experimento OCSVM CSIC sin registro"
        d["CreationDate"] = datetime.now()

    print(f"\n[OK] PDF generado: {args.out}")


if __name__ == "__main__":
    main()
