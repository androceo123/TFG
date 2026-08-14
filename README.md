# Executables: execution and reproduction guide

This directory contains the experiment launchers and Python programs used during the development of the thesis **“Detección y clasificación de ataques en tráfico HTTP mediante enfoques binario, one-class, multiclase y multietiqueta.”** This guide explains how to run every file in the directory, where inputs must be placed, which artifacts should be produced, and which files correspond to the final results reported in the thesis.

The most important distinction is:

| Thesis branch | Thesis reference run(s) | Model | Executable in this directory |
|---|---:|---|---|
| Supervised binary | B1/B2 | Random Forest | **Not present.** The binary launchers referenced by the thesis are not included in `executables/`. |
| One-class detection | O3 | Isolation Forest | `run_csic_oneclass_gpu_optimo.sh` + `csic_oneclass_gpu_optimo.py` |
| Multiclass classification | M9 | HistGradientBoostingClassifier | `3datasetsMulticlase3.sh` + `3datasetsMulticlase3.py` |
| Multilabel classification | L2 | XGBoost One-vs-Rest | `multietiquetaCorrida2.sh` + `multietiquetaCorrida2.py` |

`Harvard` and `SR-BH 2020` refer to the same dataset in the code and thesis. A binary NORMAL-versus-attack report produced inside a multiclass or multilabel program is an auxiliary reduction; it is **not** the separate supervised Random Forest branch B1/B2.

## Reproducibility scope

The current `implementacion` branch was inspected at commit:

```text
2a8ca53ce6d1cfe75b820845aa57e1894f392d9e
```

The six final O3/M9/L2 files in this directory have the same SHA-256 values recorded in Table D.2 of the thesis:

| File | SHA-256 |
|---|---|
| `3datasetsMulticlase3.sh` | `f1fd7dd76d1edd0d6d1be7826c6561729f8691856b7922cb538775224ff53b7c` |
| `3datasetsMulticlase3.py` | `c930f2a2b7a02d054c11d6a2f6693fec915b238fd9513fea25923dcc4b47efb4` |
| `multietiquetaCorrida2.sh` | `ef2891d4cfc1e87626df220661c990b8fe6dab09ed3202ca3695e9f3e810da08` |
| `multietiquetaCorrida2.py` | `1120c44776ebd75ad65c98fa2c69a11a4ed086c9f6a6a81d69d2e085c0619ecb` |
| `run_csic_oneclass_gpu_optimo.sh` | `faf96529bc11a72ba39cb7fc49d37879b36863f253db9bb2669f3702c43f99b1` |
| `csic_oneclass_gpu_optimo.py` | `e8cc64fe34d06aa68eec61bc36864df13ef9cb76b8260e4db6e44c090ba65e40` |

This establishes file-level alignment with the preserved thesis material. It does **not** establish that the current Git commit was the commit used by the historical June/July 2026 jobs. The exact historical commit was not recorded, and the thesis explicitly warns against attributing the metrics retrospectively to the current repository state.

Expect a new run to produce the same **structure, protocol, feature schema, class taxonomy, and output types**. Exact numerical or bit-for-bit equality can still be affected by dataset identity, library versions, backend, CPU/GPU choice, thread scheduling, and the historical environment.

The output lists below describe what the current programs are designed to create in a **fresh run**. They are not inventories of the surviving historical evidence: the preserved M9 package lacked its models, predictions, and importance files; the preserved L2 package lacked its model, predictions, split indices, and complete trial history; and the preserved O3 package lacked the original job 2469 log and metric files.

## 1. Prerequisites

### 1.1 Platform

- Linux and Bash.
- Slurm for the supplied `.sh` batch launchers.
- Python 3.11.7 is documented for M9, L1, and L2. O3's effective Python and scikit-learn versions were not preserved. `pyproject.toml` declares Python 3.10 or newer.
- Sufficient storage for raw data, processed caches, models, predictions, and checkpoints.
- `sha256sum` and `flock` for M9.

The shell files are stored with mode `0644`. Slurm can read them with `sbatch`, but login-node or local helpers must be invoked as `bash executables/<script>.sh` unless executable permission is added separately.

### 1.2 Clone and create the environment

```bash
git clone --branch implementacion --single-branch https://github.com/androceo123/TFG.git
cd TFG/waf-ml-starter

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` does not currently include two dependencies used by the final executables. Install the versions fixed in the thesis environment table:

```bash
python -m pip install xgboost==3.2.0 matplotlib==3.10.9
```

Optional variants may also require:

- LightGBM and CatBoost for the multifamily multilabel search;
- RAPIDS cuML and CuPy for GPU Isolation Forest;
- PyTorch for the optional one-class autoencoder.

These optional packages are not required for the reported O3 result, whose final backend was scikit-learn Isolation Forest on CPU. M9 also uses CPU: scikit-learn's HistGradientBoostingClassifier has no CUDA backend.

Verify the base environment:

```bash
.venv/bin/python - <<'PY'
import importlib

for name in ("numpy", "pandas", "sklearn", "joblib", "tqdm", "pyarrow", "psutil", "matplotlib", "xgboost"):
    module = importlib.import_module(name)
    print(name, getattr(module, "__version__", "installed"))
PY
```

## 2. Required data layout

No training dataset is committed to this branch. Populate the following locations before submitting jobs:

```text
waf-ml-starter/
├── data/
│   └── raw/
│       ├── csic/
│       │   ├── normalTrafficTraining.txt
│       │   ├── normalTrafficTest.txt
│       │   └── anomalousTrafficTest.txt
│       ├── harvard/
│       │   └── data_capec_multilabel.csv     # exact preserved input for aligned M9/L2 runs
│       └── torpeda/
│           └── *.xml
└── egipcios/
    └── DS_Augmented_v2_csv/
        └── combined_data.csv
```

Important data rules:

- M9 requires TorpEda, SR-BH 2020, and DS-Augmented-v2 in the same run.
- A thesis-aligned TorpEda run uses exactly `allAnomalies1.xml`, `allAnomalies2.xml`, `allAttacks1.xml` through `allAttacks5.xml`, and `allNormals1.xml`.
- L2 requires the native SR-BH CAPEC columns, including `000 - Normal`. Its reported NORMAL decision is derived from the absence of every modeled CAPEC label; it is not a thirteenth independently trained classifier.
- Several Harvard loaders concatenate **every** CSV, TSV, CSV.GZ, or TSV.GZ found under the input directory. Do not place exports, duplicate copies, or generated results inside `data/raw/harvard/`.
- L2 fingerprints the canonicalized method, URI, protocol, merged request-header map, and body with BLAKE2b-128, unions the labels of each group with logical OR, and keeps each group in only one split. M9 does not use this group split; it uses a stratified 80/20 split within each dataset.
- The O3 program evaluates ground truth only as normal versus anomalous. Its attack-family reports are heuristic pattern summaries, not authoritative CSIC multiclass labels.

Compatible CSV/TSV/GZ tables can be used for custom or historical experiments, but they are not the preserved M9/L2 input unless their content matches `data_capec_multilabel.csv`. Useful historical sanity checks are:

| Run | Population expected from the preserved inputs |
|---|---|
| O3 | 36,000 normal training requests; validation 18,000 normal + 12,532 anomalous; final holdout 18,000 normal + 12,533 anomalous |
| M9 TorpEda | 74,133 records: 59,306 development + 14,827 test |
| M9 SR-BH 2020 | 907,815 observed rows; the publication states 907,814 |
| M9 DS-Augmented-v2 | 704,665 records: 563,732 development + 140,933 test |
| L2 | 542,990 consolidated request groups: 347,513 train + 43,439 validation + 43,440 threshold adjustment + 108,598 test |

For a defensible reproduction, preserve the input hashes alongside the run configuration. Table D.3 of the thesis records `9c73c90ce6564ae48b14f7179cd864d037a6a130ef69c68c1626ec5d7ce4a910` for SR-BH's `data_capec_multilabel.csv` and `20725ad5c44bad2596820f0156d14106e44174890880840a274429336ca3b647` for DS-Augmented-v2's `combined_data.csv`, plus individual hashes for the eight TorpEda XML files.

Do not redistribute CSIC, TorpEda, or DS-Augmented-v2 until their licenses have been verified. SR-BH 2020 is published as CC0 1.0, but potentially sensitive HTTP fields should still be kept access-restricted.

## 3. Current directory layout and launcher compatibility

Most launchers were written when their Python companion was located directly in the `waf-ml-starter` root. The current branch stores both files under `executables/`. Define these variables once:

```bash
export PROJECT_DIR="$(pwd -P)"
export EXEC_DIR="$PROJECT_DIR/executables"
export PYTHON="$PROJECT_DIR/.venv/bin/python"
```

The `normal` partition, `c2`/`c3` node names, and GPU resource syntax below match the author's Slurm cluster. Adapt or remove those selectors on a different cluster. The historical wall times are observations from that environment, not guaranteed runtimes or minimum resource requirements.

For launchers that honor `TRAIN_PY`, the following helper submits the current companion path and overrides historical absolute Slurm log/working-directory directives:

```bash
submit_pair() {
  local launcher="$1"
  local program="$2"
  shift 2

  sbatch "$@" \
    --chdir="$PROJECT_DIR" \
    --output="$PROJECT_DIR/slurm-%x-%j.out" \
    --error="$PROJECT_DIR/slurm-%x-%j.err" \
    --export="ALL,PROJECT_DIR=$PROJECT_DIR,TRAIN_PY=$EXEC_DIR/$program" \
    "$EXEC_DIR/$launcher"
}
```

Example:

```bash
submit_pair 10claseResults.sh 10claseResults.py --partition=normal
```

For a no-training preflight, most paired launchers support:

```bash
PROJECT_DIR="$PROJECT_DIR" \
TRAIN_PY="$EXEC_DIR/10claseResults.py" \
CHECK_ONLY=1 \
bash "$EXEC_DIR/10claseResults.sh"
```

The final M9 launcher is the exception: it intentionally hardcodes `$PROJECT_DIR/3datasetsMulticlase3.py` and verifies its hash. Its section below shows the required compatibility symlink.

On a non-Slurm Linux host, a paired launcher can also be invoked with the same `PROJECT_DIR`/`TRAIN_PY` overrides and `bash` instead of `sbatch`; all `#SBATCH` resource directives are then ignored. Use `CHECK_ONLY=1` first and do not start a full dataset run unless the host has enough memory, CPU, and disk capacity.

## 4. Final thesis runs

### 4.1 M9 - final multiclass run

M9 trains one independent HistGradientBoosting model per dataset using the same ordered set of 73 request-only features. It does not train one cross-dataset model.

Create the root-level compatibility symlink expected by the frozen launcher:

```bash
cd "$PROJECT_DIR"
if [[ ! -e 3datasetsMulticlase3.py && ! -L 3datasetsMulticlase3.py ]]; then
  ln -s executables/3datasetsMulticlase3.py 3datasetsMulticlase3.py
fi
sha256sum 3datasetsMulticlase3.py
```

The printed hash must be `c930f2a2b7a02d054c11d6a2f6693fec915b238fd9513fea25923dcc4b47efb4`.

Preflight without training:

```bash
PROJECT_DIR="$PROJECT_DIR" CHECK_ONLY=1 bash "$EXEC_DIR/3datasetsMulticlase3.sh"
```

Submit with the historical allocation and node used by the reported run:

```bash
sbatch --partition=normal --nodelist=c3 \
  --chdir="$PROJECT_DIR" \
  --output="$PROJECT_DIR/slurm-%x-%j.out" \
  --error="$PROJECT_DIR/slurm-%x-%j.err" \
  --export="ALL,PROJECT_DIR=$PROJECT_DIR" \
  "$EXEC_DIR/3datasetsMulticlase3.sh"
```

Expected output root:

```text
3datasetsMulticlase3/
├── informe_integrado_3datasets.md
├── manifest_integrated_3datasets.json
├── manifest_3datasetsMulticlase3.json
└── multiclase/
    ├── comparison_integrated_3datasets.csv
    ├── comparison_integrated_3datasets.json
    ├── comparison_integrated_3datasets_compact73.csv
    ├── comparison_integrated_3datasets_compact73.json
    ├── performance_regression_unified73.csv
    ├── performance_regression_unified73.json
    ├── torpeda/
    ├── harvard/
    └── egipcios/
```

Each dataset directory should contain `model_multiclass_histgb_compact73_unified_inner_weight_selection.joblib`, `metrics_multiclass_histgb_compact73_unified_inner_weight_selection.json`, `processed_compact73_unified_inner_weight_selection.parquet` (or a CSV fallback), a run configuration, confusion data, predictions, feature importances, loss-curve artifacts, `feature_selection_manifest.csv`, `label_schema_validation.json`, model-selection evidence, and `operational_metrics.json`. M9 is resume-safe by default (`RESUME=1`, `SKIP_COMPLETE=1`) and uses a lock to prevent concurrent writers.

Reference results from the thesis:

| Dataset | Accuracy | Macro F1 | MCC | Historical Slurm wall time |
|---|---:|---:|---:|---:|
| TorpEda | 99.8853% | 99.9221% | 0.99808 | included in the unified run |
| SR-BH 2020 | 98.0103% | 97.2135% | 0.96554 | included in the unified run |
| DS-Augmented-v2 | 99.4281% | 83.8175% | 0.98766 | included in the unified run |
| Unified M9 job | - | - | - | about 6 h 03 min |

Slurm accounting records `06:03:24`; the thesis narrative differs by one second and states `6 h 03 min 23 s`.

The final validator also checks expected official class sets, row counts, feature hashes, artifacts, and regression gates. A modified dataset can complete model training but fail those final historical-alignment guards.

### 4.2 L2 - final multilabel run

L2 uses 90 request-only features and an independent XGBoost classifier for each CAPEC label. It consolidates records whose canonicalized method, URI, protocol, merged request headers, and body produce the same fingerprint before the split, then uses separate training, validation, threshold-adjustment, and final-test groups in effective proportions 64/8/8/20. Equal fingerprints define the experiment's grouping key; they do not prove byte-for-byte or semantic identity.

Create the Slurm log directory before submission because Slurm opens its output files before the script can create directories:

```bash
mkdir -p "$PROJECT_DIR/multietiquetaCorrida2/logs"
```

Preflight:

```bash
PROJECT_DIR="$PROJECT_DIR" \
TRAIN_PY="$EXEC_DIR/multietiquetaCorrida2.py" \
CHECK_ONLY=1 \
bash "$EXEC_DIR/multietiquetaCorrida2.sh"
```

Submit using the documented CPU configuration:

```bash
sbatch --partition=normal --nodelist=c3 \
  --chdir="$PROJECT_DIR" \
  --output="$PROJECT_DIR/multietiquetaCorrida2/logs/slurm-%x-%j.out" \
  --error="$PROJECT_DIR/multietiquetaCorrida2/logs/slurm-%x-%j.err" \
  --export="ALL,PROJECT_DIR=$PROJECT_DIR,TRAIN_PY=$EXEC_DIR/multietiquetaCorrida2.py" \
  "$EXEC_DIR/multietiquetaCorrida2.sh"
```

Expected principal artifacts under `multietiquetaCorrida2/harvard/`:

```text
metrics_multilabel_xgb_request90.json
model_multilabel_xgb_request90.joblib
processed_request90_multilabel.parquet
request_duplicate_groups.csv
request_deduplication_audit.json
per_label_metrics.csv
target_attainment.csv
target_attainment_f1_85_normal_99.csv
test_metrics_multilabel.json
xgb_best_params_by_label.csv
xgb_tuning_trials_by_label.csv
tuning_metrics_multilabel.json
calibration_metrics_multilabel.json
thresholds_calibration.csv
threshold_joint_optimization_calibration.json
operational_metrics.json
run_config.json
```

The processed table can fall back to `processed_request90_multilabel.csv` if no Parquet engine is available. Checkpoints are written under `multietiquetaCorrida2/checkpoints/`. The launcher resumes and skips completed work by default. L2 validates input, preprocessing, fingerprint, and feature-schema signatures before cache reuse; `FORCE_REPROCESS=1` explicitly ignores an otherwise compatible processed cache, while checkpoint compatibility is checked separately.

The documented launcher overrides the Python default and writes to `multietiquetaCorrida2/harvard/`. Direct execution of `multietiquetaCorrida2.py` without `--output-dir` instead uses `multietiquetaCorrida2/multietiqueta/harvard/`.

Reference L2 results:

| Metric | Thesis value |
|---|---:|
| Test units | 108,598 consolidated request groups |
| Exact match | 96.920% |
| Hamming loss | 0.276% |
| Micro F1 | 97.604% |
| Macro F1 | 93.940% |
| Macro PR-AUC | 96.260% |
| Historical Slurm wall time | 00:39:10 |

Ten of twelve labels exceeded 85% F1. CAPEC-153 reached 80.86% and CAPEC-272 reached 58.04%. NORMAL recall on the final test was 98.9933%, three correct groups short of the 99% target; the README must not round that result up to a successful target.

### 4.3 O3 - final one-class run

O3 trains only on normal CSIC traffic. It makes a stratified 50/50 split of the official normal and anomalous test files, uses one half to choose the Isolation Forest configuration and threshold, and evaluates the other half as the reserved final holdout. The additional metric over all 61,065 official evaluation rows includes the selection half, so it is complementary rather than a second independent test.

Inspect the Python interface:

```bash
"$PYTHON" "$EXEC_DIR/csic_oneclass_gpu_optimo.py" --help
```

Submit:

```bash
sbatch --partition=normal \
  --chdir="$PROJECT_DIR" \
  --output="$PROJECT_DIR/slurm-csic-oneclass-%j.out" \
  --error="$PROJECT_DIR/slurm-csic-oneclass-%j.err" \
  --export="ALL,PROJECT_DIR=$PROJECT_DIR,PY_SCRIPT=$EXEC_DIR/csic_oneclass_gpu_optimo.py,MODEL=sklearn_iforest,GPU_MODE=off" \
  "$EXEC_DIR/run_csic_oneclass_gpu_optimo.sh"
```

The explicit `MODEL=sklearn_iforest,GPU_MODE=off` settings align the model backend with the reported O3 result. The launcher still requests one GPU through its historical `#SBATCH` directive, even though this CPU backend does not use it. A GPU allocation or visible `nvidia-smi` output does not prove that the selected model used the GPU.

Expected outputs under `resultsOptimo/csic/oneclass_gpu_optimo/`:

```text
summary.md
metrics_test_holdout.json
metrics_official_full.json
search.csv
best_validation_metrics.json
feature_importance_permutation.csv
feature_importance_top.png
predictions_test_holdout.csv
predictions_official_full.csv
attack_family_report_official_full.csv
model.joblib
model_meta.json
run_info.json
```

The launcher always passes `--force-reprocess`, so it recreates `data/processed/csic_oneclass_gpu_optimo/`. It does not delete raw CSIC files.

Reference O3 results:

| F1 | Recall | FPR | MCC | PR-AUC | ROC-AUC | Historical Slurm wall time |
|---:|---:|---:|---:|---:|---:|---:|
| 0.70091 | 0.72353 | 0.23744 | 0.48208 | 0.78169 | 0.83423 | 00:26:20 |

The 23.744% false-positive rate is too high for autonomous blocking and is reported in the thesis as a limitation.

## 5. Timing and operational evidence

The programs separate parameter/model selection, final construction, and application where the pipeline supports that distinction:

| Run | Parameter optimization / selection | Model construction | Application and operational metrics |
|---|---|---|---|
| M9 | `model_selection_seconds` and candidate `fit_seconds` | `t_build_train_seconds` | `t_apply_inference_batch_seconds`; p50/p95/p99 and throughput in `operational_metrics.json` |
| L2 | `train_select_seconds`; per-label `fit_seconds` in tuning trials | `final_refit_seconds` when applicable; L2 is launched with no refit after thresholds | `t_apply_inference_batch_seconds`; online/batch latency and throughput in `operational_metrics.json` |
| O3 | `search.csv` contains candidate `fit_sec` and `score_sec` | final refit is logged; selected validation configuration in `best_validation_metrics.json` | evaluation `score_sec`; total `elapsed_sec` in `run_info.json` |

These measurements cover local feature extraction/model inference as implemented by the scripts. They do not include network transport, TLS, sustained concurrency, queues, a production WAF rule engine, or blocking decisions.

## 6. Historical multiclass sequence M1-M9

The thesis uses the following mapping. M1-M8 are exploratory because test results informed features and taxonomies; some transitions also changed multiple factors. M9 is the final development configuration, but not an independent confirmation because its 73-feature representation inherits that history.

| ID | Launcher and Python | Main change | Default output root | Submit from the current layout |
|---:|---|---|---|---|
| M1 | `run_DEFINITIVO_multiclass_histgb_fixed30_arandu_safe.sh` + `train_waf_multiclass_DEFINITIVO_histgb_fixed30.py` | Fixed 30-feature HistGB baseline | `DEFINITIVO/` | `submit_pair run_DEFINITIVO_multiclass_histgb_fixed30_arandu_safe.sh train_waf_multiclass_DEFINITIVO_histgb_fixed30.py --partition=normal` |
| M2 | `pipeline14classes.sh` + `pipeline14classes.py` | Harvard top-14, fixed 30 features | `pipeline14classesResult/` | `submit_pair pipeline14classes.sh pipeline14classes.py --partition=normal` |
| M3 | `corrida3mejoresResultadosPosibles.sh` + `.py` | 84 optimized request features | `corrida3mejoresResultadosPosibles/` | `submit_pair corrida3mejoresResultadosPosibles.sh corrida3mejoresResultadosPosibles.py --partition=normal` |
| M4 | `corrida3mejoresResultadosPosibles_featuresUtiles58.sh` + `.py` | Pruned useful58 feature set | `corrida3mejoresResultadosPosibles_featuresUtiles58/` | `submit_pair corrida3mejoresResultadosPosibles_featuresUtiles58.sh corrida3mejoresResultadosPosibles_featuresUtiles58.py --partition=normal` |
| M5 | `11claseResults.sh` + `11claseResults.py` | Harvard 11 operational groups | `11claseResults/` | `submit_pair 11claseResults.sh 11claseResults.py --partition=normal` |
| M6 | `10claseResults.sh` + `10claseResults.py` | Harvard final 10-class reduction in this development stage | `10claseResults/` | `submit_pair 10claseResults.sh 10claseResults.py --partition=normal` |
| M7 | `3datasetsMulticlase.sh` + `.py` | Adds DS-Augmented-v2, 58 features; DS has 8 classes because Command Injection and OS Command Injection are merged | `3datasetsMulticlase/` | `submit_pair 3datasetsMulticlase.sh 3datasetsMulticlase.py --partition=normal` |
| M8 | `3datasetsMulticlase2.sh` + `.py` | 72 features; retains the 8-class DS taxonomy | `3datasetsMulticlase2/` | `submit_pair 3datasetsMulticlase2.sh 3datasetsMulticlase2.py --partition=normal --nodelist=c2` |
| M9 | `3datasetsMulticlase3.sh` + `.py` | Final 73-feature flow; DS has 9 classes because Command Injection and OS Command Injection are separated | `3datasetsMulticlase3/` | Use the dedicated M9 command above. |

Most M1-M8 launchers default to `CLEAN_OUTPUT_DIR=1` and do not resume. A rerun can replace the generated result tree. Use a copied output location where the launcher supports it, or archive the prior result tree before resubmission.

## 7. Multilabel development launchers

### 7.1 L1 fixed30 XGBoost family

The following five launchers all invoke `train_waf_multilabel_DEFINITIVO_harvard_xgboost_none_fixed30_gpu_metrics.py` and default to the same output/checkpoint tree: `DEFINITIVO/multietiqueta/multiquetaSalidaXG/`.

| Launcher | Requested resources | Correct use |
|---|---|---|
| `run_DEFINITIVO_multilabel_harvard_xgboost_none_fixed30_arandu_nodo3_gpu_metrics.sh` | 32 CPU, all node memory, 1 GPU, exclusive, 3 days | GPU variant. Submit with `submit_pair <launcher> train_waf_multilabel_DEFINITIVO_harvard_xgboost_none_fixed30_gpu_metrics.py --partition=normal --gres=gpu:1`. |
| `run_DEFINITIVO_multilabel_harvard_xgboost_none_fixed30_arandu_c3_cpu_tmp.sh` | 32 CPU, all memory, exclusive, no GPU | Export `USE_GPU=off`; submit on `c3`. |
| `run_DEFINITIVO_multilabel_harvard_xgboost_none_fixed30_arandu_c3_8cpu_tmp.sh` | 8 CPU, 32 GiB, no GPU | Functionally duplicates the c3 resume resource profile; export `USE_GPU=off`. |
| `run_resume_multilabel_xgb_cpu_c3.sh` | 8 CPU, 32 GiB, no GPU | Export `USE_GPU=off FORCE_REPROCESS=0`; submit with `--nodelist=c3`. |
| `run_resume_multilabel_xgb_cpu_c2.sh` | 8 CPU, 32 GiB, no GPU | Export `USE_GPU=off FORCE_REPROCESS=0`; the filename does not pin the node, so submit with `--nodelist=c2`. |

Example of a safe CPU resume:

```bash
export USE_GPU=off
export FORCE_REPROCESS=0
submit_pair \
  run_resume_multilabel_xgb_cpu_c2.sh \
  train_waf_multilabel_DEFINITIVO_harvard_xgboost_none_fixed30_gpu_metrics.py \
  --partition=normal --nodelist=c2
```

Warnings:

- Despite `cpu` in several filenames, all five scripts default to `USE_GPU=on` unless overridden.
- Despite `resume` in two filenames, they default to `FORCE_REPROCESS=1`; set it to `0` for a true checkpoint resume.
- Do not run two variants concurrently with the same output/checkpoint directories. They have no lock and can overwrite or corrupt shared artifacts.
- The thesis L1 chain used CPU (`USE_GPU=off`) and resumed checkpoints after a 72-hour timeout. The GPU-named launchers should not be cited as proof that historical L1 used a GPU.

### 7.2 Multifamily fixed30 search

`run_multilabel_harvard_multifamily_resume_resultsOptimo_arandu_safe_fixed30.sh` points by default to a filename that is not present. The available companion omits `DEFINITIVO` from its name, so pass it explicitly:

```bash
mkdir -p "$PROJECT_DIR/resultsOptimo/slurm"
submit_pair \
  run_multilabel_harvard_multifamily_resume_resultsOptimo_arandu_safe_fixed30.sh \
  train_waf_multilabel_multifamily_resume_resultsOptimo_fixed30.py \
  --partition=normal
```

Expected root: `resultsOptimo/multietiqueta/harvard/`. The program compares available One-vs-Rest families and resumes processed data/checkpoints. XGBoost is required by the launcher; LightGBM and CatBoost are optional and are skipped when absent.

This is a development/model-search pipeline, not L2. It uses 30 features and does not implement L2's request fingerprint consolidation and group-aware split.

## 8. Additional multiclass model-search launchers

### Harvard v12 all-model search

```bash
submit_pair \
  run_harvard_multiclass_fixed30_allTorpedaModels_noFI_sbatch_v12.sh \
  train_waf_multiclass_fixed30_resultsOptimo_sbatch_v12_noFI_allTorpedaModelsHarvard.py \
  --partition=normal --gres=gpu:1
```

Expected root: `resultsOptimo/harvard/multiclass/`. It compares XGBoost, two-stage XGBoost, HistGradientBoosting, ExtraTrees, and logistic regression candidates; feature importance is disabled by this launcher.

Candidate-score resume matches candidate names, not a complete input/configuration fingerprint. For a fresh auditable run after changing data or arguments, use a new `OUT_DIR` or set `CLEAN_OUTPUT_DIR=1` after preserving prior results.

### TorpEda v10 launcher - incomplete in this branch

`run_torpeda_multiclass_fixed30_optimo_sbatch_v10_normal_sharedgpu.sh` expects:

```text
train_waf_multiclass_fixed30_resultsOptimo_sbatch_v10.py
```

That Python file is absent from this repository. The launcher therefore cannot be reproduced from the current checkout. Do not silently substitute the available v12 Harvard program: its interface and experiment are not established as equivalent. If the exact v10 file is recovered, pass its path through `TRAIN_PY` and archive its hash with the result.

The launcher also always appends `--clean-output-dir`; `CLEAN_OUTPUT_DIR=0` does not prevent it from clearing the guarded TorpEda output tree.

## 9. Audit and helper scripts

### 9.1 Duplicate-content audit for SR-BH 2020

Run locally from the project root:

```bash
PYTHON_BIN="$PYTHON" \
bash "$EXEC_DIR/duplicadosHarvard.sh" \
  data/raw/harvard/data_capec_multilabel.csv \
  duplicadosHarvard/harvard_duplicados.csv
```

Expected outputs:

```text
duplicadosHarvard/harvard_duplicados.csv
duplicadosHarvard/harvard_duplicados.xlsx
```

The audit includes the first occurrence and every additional occurrence of repeated HTTP content. It does not delete, consolidate, or change source rows. Set `STRICT_OFFICIAL=1` to enforce the official source hash/schema/counts, and `FORCE=0` to protect an existing derived output.

To submit this particular wrapper through Slurm without a compatibility symlink, submit it from `executables/`, because the spooled batch copy otherwise cannot resolve its Python companion:

```bash
cd "$EXEC_DIR"
sbatch --chdir="$EXEC_DIR" duplicadosHarvard.sh \
  ../data/raw/harvard/data_capec_multilabel.csv \
  ../duplicadosHarvard/harvard_duplicados.csv
cd "$PROJECT_DIR"
```

### 9.2 Slurm evidence package

Run on the cluster login node, not as a training job:

```bash
cd "$PROJECT_DIR"
SLURM_AUDIT_PROJECT_ROOT="$PROJECT_DIR" \
bash "$EXEC_DIR/extraer_evidencia_tesis_slurm.sh"
```

Expected outputs:

```text
evidencia_tesis_slurm_<timestamp>/
evidencia_tesis_slurm_<timestamp>.tar.gz
evidencia_tesis_slurm_<timestamp>.tar.gz.sha256
```

The script queries historical Slurm accounting and captures a current node/environment inventory. Current hardware, packages, and Git state do not prove the environment of historical jobs. Many individual collection commands are best-effort, so inspect the package logs even if the top-level script completes.

### 9.3 Best-node submit helper for M6

Preview only:

```bash
TRAIN_PY="$EXEC_DIR/10claseResults.py" \
JOB_SCRIPT="$EXEC_DIR/10claseResults.sh" \
EXTRA_SBATCH_ARGS="--chdir=$PROJECT_DIR --output=$PROJECT_DIR/slurm-%x-%j.out --error=$PROJECT_DIR/slurm-%x-%j.err" \
DRY_RUN=1 \
bash "$EXEC_DIR/submit_10claseResults_best_node.sh"
```

Submit:

```bash
TRAIN_PY="$EXEC_DIR/10claseResults.py" \
JOB_SCRIPT="$EXEC_DIR/10claseResults.sh" \
EXTRA_SBATCH_ARGS="--chdir=$PROJECT_DIR --output=$PROJECT_DIR/slurm-%x-%j.out --error=$PROJECT_DIR/slurm-%x-%j.err" \
bash "$EXEC_DIR/submit_10claseResults_best_node.sh"
```

The helper selects a node from a momentary CPU/memory/load snapshot. It does not reserve that capacity before calling `sbatch`. If it selects a node, Slurm may queue the job until that node is available; only the no-fit fallback submits without `--nodelist` and lets Slurm choose a node.

## 10. Complete Python file catalog

All 15 Python files parse successfully. Use `"$PYTHON" "$EXEC_DIR/<file>.py" --help` to inspect the full CLI; for thesis-style runs, prefer the paired launcher because it fixes resources and arguments.

| Python file | Purpose and expected result |
|---|---|
| `10claseResults.py` | M6 multiclass HistGB, 58 features, TorpEda + Harvard 10-class reduction; writes models, metrics, loss curves, FI, and operational artifacts under `10claseResults/`. |
| `11claseResults.py` | M5 multiclass HistGB, 58 features, older Harvard 11-group taxonomy; writes under `11claseResults/`. |
| `3datasetsMulticlase.py` | M7 three-dataset multiclass run with 58 features; writes under `3datasetsMulticlase/`. |
| `3datasetsMulticlase2.py` | M8 three-dataset multiclass run with 72 features; writes under `3datasetsMulticlase2/`. |
| `3datasetsMulticlase3.py` | M9 final three-dataset multiclass run with frozen ordered compact73 schema; writes under `3datasetsMulticlase3/`. |
| `corrida3mejoresResultadosPosibles.py` | M3 multiclass development run with 84 features; writes under `corrida3mejoresResultadosPosibles/`. |
| `corrida3mejoresResultadosPosibles_featuresUtiles58.py` | M4 pruned 58-feature successor; writes under its same-named result root. |
| `csic_oneclass_gpu_optimo.py` | O3 CSIC one-class Isolation Forest pipeline; writes processed CSIC data and `resultsOptimo/csic/oneclass_gpu_optimo/`. |
| `duplicadosHarvard.py` | Read-only SR-BH repeated-content audit; writes a technical CSV and review XLSX. |
| `multietiquetaCorrida2.py` | L2 final request90 XGBoost OVR pipeline with fingerprint consolidation and group split; writes under `multietiquetaCorrida2/`. |
| `pipeline14classes.py` | M2 fixed30/top-14 HistGB stage; writes under `pipeline14classesResult/`. |
| `train_waf_multiclass_DEFINITIVO_histgb_fixed30.py` | M1 fixed30 HistGB stage for TorpEda and Harvard; writes under `DEFINITIVO/multiclase/`. |
| `train_waf_multiclass_fixed30_resultsOptimo_sbatch_v12_noFI_allTorpedaModelsHarvard.py` | Harvard fixed30 candidate-family comparison used by the v12 launcher; writes under `resultsOptimo/harvard/multiclass/`. |
| `train_waf_multilabel_DEFINITIVO_harvard_xgboost_none_fixed30_gpu_metrics.py` | L1-era fixed30 XGBoost OVR program used by the five GPU/CPU/resume wrappers; writes under `DEFINITIVO/multietiqueta/multiquetaSalidaXG/`. |
| `train_waf_multilabel_multifamily_resume_resultsOptimo_fixed30.py` | Fixed30 multifamily One-vs-Rest search with checkpoints; writes under `resultsOptimo/multietiqueta/harvard/`. |

## 11. Complete shell file catalog

All 22 shell files pass `bash -n`. The table gives the intended entry point for each file.

| Shell file | How to execute / status |
|---|---|
| `10claseResults.sh` | `submit_pair 10claseResults.sh 10claseResults.py --partition=normal`; historical M6, cleans output by default. |
| `11claseResults.sh` | `submit_pair 11claseResults.sh 11claseResults.py --partition=normal`; historical M5, cleans output by default. |
| `3datasetsMulticlase.sh` | `submit_pair 3datasetsMulticlase.sh 3datasetsMulticlase.py --partition=normal`; historical M7. |
| `3datasetsMulticlase2.sh` | `submit_pair 3datasetsMulticlase2.sh 3datasetsMulticlase2.py --partition=normal --nodelist=c2`; historical M8. |
| `3datasetsMulticlase3.sh` | Use the dedicated M9 symlink/preflight/submission commands; final M9. |
| `corrida3mejoresResultadosPosibles.sh` | `submit_pair corrida3mejoresResultadosPosibles.sh corrida3mejoresResultadosPosibles.py --partition=normal`; historical M3. |
| `corrida3mejoresResultadosPosibles_featuresUtiles58.sh` | `submit_pair corrida3mejoresResultadosPosibles_featuresUtiles58.sh corrida3mejoresResultadosPosibles_featuresUtiles58.py --partition=normal`; historical M4. |
| `duplicadosHarvard.sh` | Run with `bash` locally or submit from `executables/` as shown above; produces duplicate audit CSV/XLSX. |
| `extraer_evidencia_tesis_slurm.sh` | Run with `bash` on the login node; produces a timestamped evidence directory, tarball, and checksum. |
| `multietiquetaCorrida2.sh` | Use the dedicated L2 preflight/submission commands; final L2. |
| `pipeline14classes.sh` | `submit_pair pipeline14classes.sh pipeline14classes.py --partition=normal`; historical M2. |
| `run_DEFINITIVO_multiclass_histgb_fixed30_arandu_safe.sh` | `submit_pair run_DEFINITIVO_multiclass_histgb_fixed30_arandu_safe.sh train_waf_multiclass_DEFINITIVO_histgb_fixed30.py --partition=normal`; historical M1. |
| `run_DEFINITIVO_multilabel_harvard_xgboost_none_fixed30_arandu_c3_8cpu_tmp.sh` | L1 resource variant; export `USE_GPU=off`, select a unique output or shared resume target, then use `submit_pair`. |
| `run_DEFINITIVO_multilabel_harvard_xgboost_none_fixed30_arandu_c3_cpu_tmp.sh` | L1 CPU resource variant; export `USE_GPU=off`, then use `submit_pair ... --partition=normal --nodelist=c3`. |
| `run_DEFINITIVO_multilabel_harvard_xgboost_none_fixed30_arandu_nodo3_gpu_metrics.sh` | L1 GPU resource variant; use `submit_pair ... --partition=normal --gres=gpu:1`. |
| `run_csic_oneclass_gpu_optimo.sh` | Use the dedicated O3 command with `PY_SCRIPT`; final O3. |
| `run_harvard_multiclass_fixed30_allTorpedaModels_noFI_sbatch_v12.sh` | Use the v12 all-model command above; development search, not M9. |
| `run_multilabel_harvard_multifamily_resume_resultsOptimo_arandu_safe_fixed30.sh` | Override its outdated default companion name with the available multifamily Python; development search, not L2. |
| `run_resume_multilabel_xgb_cpu_c2.sh` | L1 resume variant; export `USE_GPU=off FORCE_REPROCESS=0` and pass `--nodelist=c2`. |
| `run_resume_multilabel_xgb_cpu_c3.sh` | L1 resume variant; export `USE_GPU=off FORCE_REPROCESS=0` and pass `--nodelist=c3`. |
| `run_torpeda_multiclass_fixed30_optimo_sbatch_v10_normal_sharedgpu.sh` | **Blocked:** its exact v10 Python companion is absent. Do not substitute v12 without a separate validation. |
| `submit_10claseResults_best_node.sh` | Run with `bash` on the login node; use `DRY_RUN=1` first. It submits `10claseResults.sh`. |

## 12. Monitoring and validation

After `sbatch` returns a job ID:

```bash
squeue -j JOB_ID
sacct -j JOB_ID --format=JobID,JobName,Partition,State,ExitCode,Elapsed,NodeList,AllocCPUS,ReqMem,MaxRSS
scontrol show job JOB_ID -o | tr ' ' '\n' | grep -E '^(StdOut|StdErr)='
# Then use the reported path, for example:
tail -f /absolute/path/from/StdOut
```

Several launchers also create internal `.out` and `.err` files with `tee`; inspect those in the run's `logs/` directory as well.

Before accepting a run:

1. Confirm `State=COMPLETED` and `ExitCode=0:0`.
2. Read both Slurm and internal `.out`/`.err` logs.
3. Verify `run_config.json`, feature count and ordered feature hash where available.
4. Verify dataset paths, sizes, and SHA-256 values.
5. Confirm split sizes and class/label supports before comparing metrics.
6. Confirm the actual backend (`gpu_used`, model class, and package version); requested hardware is not proof of use.
7. Preserve code, launcher, inputs or hashes, taxonomy, configuration, metrics, predictions, model, importance files, timing artifacts, and the Slurm job record together.

## 13. Known limitations of this directory

- It does not contain the historical Random Forest B1/B2 launchers, so it is not a complete executable reproduction of all four thesis branches.
- The historical O1/O2 one-class batch bodies and program provenance are unavailable; only O3 is represented by a current launcher/program pair.
- The exact Git commit used for each historical metric was not recorded.
- The datasets are not included.
- Several launchers retain historical absolute paths and root-level companion assumptions.
- The TorpEda v10 launcher has no companion Python in the branch.
- Some CPU-named L1 wrappers default to GPU mode, and some resume-named wrappers default to reprocessing.
- M1-M8 are exploratory because test results informed variables and taxonomies; some transitions also changed multiple factors. M9 remains a final development configuration rather than an independent confirmation.
- Operational measurements are local pipeline measurements, not production WAF benchmarks.
