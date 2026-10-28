# Execution and reproduction guide

This guide covers the two experiment directories preserved in this branch:

- `waf-ml-starter/jobs/` contains the supervised binary work and auxiliary diagnostic jobs added from the second thesis workstream.
- `waf-ml-starter/executables/` contains the final O3 one-class pair, the multiclass history M1-M9, the multilabel history L1-L2, and their audit helpers.

It explains how to submit every shell file, where inputs must be placed, what each program is expected to produce, and which files correspond to the reference results in the thesis.

The primary thesis mapping is:

| Thesis branch | Thesis reference run(s) | Model | Current entry point(s) |
|---|---:|---|---|
| Supervised binary ablation | B1 | Random Forest | `jobs/run_csic_random_forest.sh`, `run_csic_rf_25feat.sh`, `run_csic_rf_34feat.sh`, and `run_csic_rf_top10.sh` |
| Supervised binary by corpus | B2 | Random Forest | B1 RF-57 for CSIC, plus `jobs/run_torpeda_binary_rf.sh`, `run_harvard_binary_rf.sh`, and `run_compare_binary.sh` |
| One-class detection | O3 | Isolation Forest | `executables/run_csic_oneclass_gpu_optimo.sh` + `csic_oneclass_gpu_optimo.py` |
| Multiclass classification | M9 | HistGradientBoostingClassifier | `executables/3datasetsMulticlase3.sh` + `3datasetsMulticlase3.py` |
| Multilabel classification | L2 | XGBoost One-vs-Rest | `executables/multietiquetaCorrida2.sh` + `multietiquetaCorrida2.py` |

`Harvard` and `SR-BH 2020` refer to the same dataset in the code and thesis. B2 is not transfer learning: it trains and evaluates a new RF inside each corpus. The OCSVM jobs in `jobs/` are auxiliary diagnostics, not O1/O2/O3 from the thesis, which use Isolation Forest. Likewise, `jobs/run_torpeda_rf_multiclass.sh` is an exploratory RF run and is not part of M1-M9.

## Reproducibility scope

The current `implementacion` branch was inspected at commit:

```text
58901587b5eec8df3f72d5aadf38f1c94b568eee
```

The six final O3/M9/L2 files in `executables/` have the same SHA-256 values recorded in Table D.2 of the thesis:

| File | SHA-256 |
|---|---|
| `3datasetsMulticlase3.sh` | `f1fd7dd76d1edd0d6d1be7826c6561729f8691856b7922cb538775224ff53b7c` |
| `3datasetsMulticlase3.py` | `c930f2a2b7a02d054c11d6a2f6693fec915b238fd9513fea25923dcc4b47efb4` |
| `multietiquetaCorrida2.sh` | `ef2891d4cfc1e87626df220661c990b8fe6dab09ed3202ca3695e9f3e810da08` |
| `multietiquetaCorrida2.py` | `1120c44776ebd75ad65c98fa2c69a11a4ed086c9f6a6a81d69d2e085c0619ecb` |
| `run_csic_oneclass_gpu_optimo.sh` | `faf96529bc11a72ba39cb7fc49d37879b36863f253db9bb2669f3702c43f99b1` |
| `csic_oneclass_gpu_optimo.py` | `e8cc64fe34d06aa68eec61bc36864df13ef9cb76b8260e4db6e44c090ba65e40` |

This establishes file-level alignment for O3/M9/L2. It does **not** establish that the current Git commit was used by the historical June/July 2026 jobs. The exact historical commit was not recorded, and the thesis explicitly warns against attributing the metrics retrospectively to the current repository state.

The current B1/B2 files match the experiment names, RF settings, and result structure described by the binary reports, but their hashes do not appear in the thesis. Historical B1/B2 jobs ran from `/home_data/aroman/TFG-JA/waf-ml-starter`; their original spooled batch bodies, logs, and complete RF outputs were not recovered. The current copies therefore support a new execution, not a claim of byte-identical historical reproduction.

Expect a new run to produce the same **structure, protocol, feature schema, class taxonomy, and output types**. Exact numerical or bit-for-bit equality can still be affected by dataset identity, library versions, backend, CPU/GPU choice, thread scheduling, and the historical environment.

The output lists below describe what the current programs are designed to create in a **fresh run**. They are not inventories of the surviving historical evidence: the preserved B1/B2 evidence lacked the original batch bodies and complete RF result packages; M9 lacked its models, predictions, and importance files; L2 lacked its model, predictions, split indices, and complete trial history; and O3 lacked the original job 2469 log and metric files.

## 1. Prerequisites

### 1.1 Platform

- Linux and Bash.
- Slurm for the supplied `.sh` batch launchers.
- Python 3.11.7 is documented for M9, L1, and L2. B1/B2 effective versions and O3's effective Python/scikit-learn versions were not preserved. `pyproject.toml` declares Python 3.10 or newer, while the current `jobs/` scripts explicitly call `.venv/bin/python3.11`.
- Sufficient storage for raw data, processed caches, models, predictions, and checkpoints.
- `sha256sum` and `flock` for M9.

The shell files are stored with mode `0644`. Slurm can read them with `sbatch`. Login-node helpers in `executables/` must be invoked as `bash executables/<script>.sh` unless executable permission is added separately. Do not run the training files in `jobs/` with plain `bash`: they rely on `SLURM_SUBMIT_DIR`, `srun`, and the resources assigned by Slurm.

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
- RAPIDS cuML and CuPy for optional GPU Isolation Forest;
- PyTorch for the optional one-class autoencoder.
- RAPIDS `cuml.accel`, CuPy, and SHAP for the GPU OCSVM diagnostic jobs in `jobs/`.

The CPU RF jobs for B1/B2 are covered by the base requirements. RAPIDS installation is platform- and CUDA-specific and is intentionally not given as a generic `pip` command here. The optional packages are not required for the reported O3 result, whose final backend was scikit-learn Isolation Forest on CPU. M9 also uses CPU: scikit-learn's HistGradientBoostingClassifier has no CUDA backend.

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

No training dataset, processed table, result, or virtual environment is committed to this branch. Create and populate the following locations before submitting jobs:

```text
waf-ml-starter/
|-- data/
|   |-- raw/
|   |   |-- csic/
|   |   |   |-- normalTrafficTraining.txt
|   |   |   |-- normalTrafficTest.txt
|   |   |   `-- anomalousTrafficTest.txt
|   |   |-- harvard/
|   |   |   `-- data_capec_multilabel.csv
|   |   `-- torpeda/
|   |       `-- *.xml
|   |-- processed/
|   |   |-- csic/csic_features.parquet
|   |   |-- harvard/harvard.parquet
|   |   `-- torpeda/torpeda_features.parquet
|   `-- tmp/
|       `-- csic_sin_registro_v2.parquet
`-- egipcios/
    `-- DS_Augmented_v2_csv/
        `-- combined_data.csv
```

Important data rules:

- M9 requires TorpEda, SR-BH 2020, and DS-Augmented-v2 in the same run.
- A thesis-aligned TorpEda run uses exactly `allAnomalies1.xml`, `allAnomalies2.xml`, `allAttacks1.xml` through `allAttacks5.xml`, and `allNormals1.xml`.
- B1 uses the current CSIC subset whose URI does not contain `registro`: 86,864 rows in the historical report, with 66,000 normal and 20,864 attack rows. B2 reuses that RF-57 result for its CSIC row.
- The B2 TorpEda and SR-BH jobs rebuild the same nominal RF-57 representation from their processed tables. CSIC passes parsed headers to the feature extractor, while the current TorpEda/SR-BH rebuilds do not; `req_content_length` may therefore use a real header in CSIC and fall back to body length in the other two corpora.
- L2 requires the native SR-BH CAPEC columns, including `000 - Normal`. Its reported NORMAL decision is derived from the absence of every modeled CAPEC label; it is not a thirteenth independently trained classifier.
- Several Harvard loaders concatenate **every** CSV, TSV, CSV.GZ, or TSV.GZ found under the input directory. Do not place exports, duplicate copies, or generated results inside `data/raw/harvard/`.
- L2 fingerprints the canonicalized method, URI, protocol, merged request-header map, and body with BLAKE2b-128, unions the labels of each group with logical OR, and keeps each group in only one split. M9 does not use this group split; it uses a stratified 80/20 split within each dataset.
- The O3 program evaluates ground truth only as normal versus anomalous. Its attack-family reports are heuristic pattern summaries, not authoritative CSIC multiclass labels.

Compatible CSV/TSV/GZ tables can be used for custom or historical experiments, but they are not the preserved M9/L2 input unless their content matches `data_capec_multilabel.csv`. Useful historical sanity checks are:

| Run | Population expected from the preserved inputs |
|---|---|
| B1 CSIC without `registro` | 86,864 rows: 66,000 normal + 20,864 attack |
| B2 TorpEda | 74,133 rows: 8,363 normal + 65,770 attack |
| B2 SR-BH 2020 | 907,815 observed rows: 525,195 normal + 382,620 attack |
| O3 | 36,000 normal training requests; validation 18,000 normal + 12,532 anomalous; final holdout 18,000 normal + 12,533 anomalous |
| M9 TorpEda | 74,133 records: 59,306 development + 14,827 test |
| M9 SR-BH 2020 | 907,815 observed rows; the publication states 907,814 |
| M9 DS-Augmented-v2 | 704,665 records: 563,732 development + 140,933 test |
| L2 | 542,990 consolidated request groups: 347,513 train + 43,439 validation + 43,440 threshold adjustment + 108,598 test |

For a defensible reproduction, preserve the input hashes alongside the run configuration. Table D.3 of the thesis records `9c73c90ce6564ae48b14f7179cd864d037a6a130ef69c68c1626ec5d7ce4a910` for SR-BH's `data_capec_multilabel.csv` and `20725ad5c44bad2596820f0156d14106e44174890880840a274429336ca3b647` for DS-Augmented-v2's `combined_data.csv`, plus individual hashes for the eight TorpEda XML files. Those hashes support M9/L2 input identification; they do not by themselves prove which copies were used by historical B1/B2.

Do not redistribute CSIC, TorpEda, or DS-Augmented-v2 until their licenses have been verified. SR-BH 2020 is published as CC0 1.0, but potentially sensitive HTTP fields should still be kept access-restricted.

### 2.1 Build the processed inputs used by `jobs/`

Run these commands from `waf-ml-starter/` after placing the raw files:

```bash
mkdir -p \
  data/raw/csic data/raw/torpeda data/raw/harvard \
  data/processed/csic data/processed/torpeda data/processed/harvard \
  data/tmp results resultsOptimo

PYTHONPATH=src .venv/bin/python3.11 -m waf_ml.scripts.build_features_csic \
  --inputs data/raw/csic \
  --output data/processed/csic/csic_features.parquet \
  --out-format parquet \
  --use-headers

PYTHONPATH=src .venv/bin/python3.11 -m waf_ml.scripts.build_features_torpeda \
  --inputs data/raw/torpeda \
  --output data/processed/torpeda/torpeda_features.parquet \
  --out-format parquet

PYTHONPATH=src .venv/bin/python3.11 -m waf_ml.scripts.build_features \
  --inputs data/raw/harvard \
  --output data/processed/harvard/harvard.parquet \
  --out-format parquet
```

The Harvard builder reads every compatible table below its input path. Keep only the intended source file there. Record counts and hashes after building; a successful conversion is not proof that the historical inputs were used.

## 3. Current directory layout and launcher compatibility

Most launchers were written when their Python companion was located directly in the `waf-ml-starter` root. The current branch stores both files under `executables/`. Define these variables once:

```bash
export PROJECT_DIR="$(pwd -P)"
export EXEC_DIR="$PROJECT_DIR/executables"
export JOB_DIR="$PROJECT_DIR/jobs"
export PYTHON="$PROJECT_DIR/.venv/bin/python"
```

The `normal` partition, `c2`/`c3` node names, and GPU resource syntax below match the author's Slurm cluster. Adapt or remove those selectors on a different cluster. The historical wall times are observations from that environment, not guaranteed runtimes or minimum resource requirements.

Submit 19 of the 21 files in `jobs/` from `waf-ml-starter/`:

```bash
cd "$PROJECT_DIR"
sbatch "$JOB_DIR/NAME.sh"
```

Two legacy OCSVM wrappers add `waf-ml-starter` to `SLURM_SUBMIT_DIR` themselves and must instead be submitted from the repository root:

```bash
cd "$PROJECT_DIR/.."
sbatch waf-ml-starter/jobs/run_csic_ocsvm.sh
sbatch waf-ml-starter/jobs/run_csic_ocsvm_1000.sh
cd "$PROJECT_DIR"
```

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

## 4. Principal thesis reference runs

### 4.1 B1/B2 - supervised binary Random Forest

B1 is a feature ablation on CSIC. B2 reuses B1's RF-57 result for CSIC and applies the same nominal RF-57 pipeline independently inside TorpEda and SR-BH 2020. All current RF launchers use binary labels (`0=normal`, `1=attack`), a stratified 80/20 row split, seed 42, 500 trees, `class_weight=balanced`, and CPU execution. They sweep thresholds from 0.05 to 0.95 in steps of 0.01.

These are **exploratory** results: the threshold maximizing F1 is selected on the same 20% test partition later used to report the metrics. The split also does not keep equal HTTP content in one partition.

#### Prepare the CSIC input for B1

The four B1 jobs require `data/tmp/csic_sin_registro_v2.parquet`. The supplied preparation job creates it and then runs an auxiliary GPU OCSVM:

```bash
cd "$PROJECT_DIR"
sbatch "$JOB_DIR/run_csic_global_v2.sh"
```

That job requires RAPIDS/SHAP. Only its preprocessing step must successfully create the temporary Parquet before the RF jobs; its later auxiliary OCSVM can fail after that file has already been written. Check both the Slurm state and the file itself. If only the CPU RF branch is needed, and `data/processed/csic/csic_features.parquet` was built with the current command in Section 2.1, create the same filtered 57-feature table without launching OCSVM:

```bash
cd "$PROJECT_DIR"
PYTHONPATH=src "$PYTHON" - <<'PY'
from pathlib import Path
import pandas as pd
from waf_ml.features.http_features import extract_http_features

source = Path("data/processed/csic/csic_features.parquet")
target = Path("data/tmp/csic_sin_registro_v2.parquet")

frame = pd.read_parquet(source)
frame = frame[
    ~frame["request_http_request"].str.contains("registro", case=False, na=False)
].reset_index(drop=True)

features = list(
    extract_http_features(method="", uri="", headers={}, body=b"").keys()
)
if len(features) != 57:
    raise SystemExit(f"Expected 57 current extractor features, found {len(features)}")

labels = [c for c in frame.columns if c.startswith("label_")]
metadata = [
    c for c in ("source_file", "dataset_name", "split", "label_type_raw")
    if c in frame.columns and c not in labels
]

target.parent.mkdir(parents=True, exist_ok=True)
frame[labels + metadata + features].to_parquet(target, index=False)
print(target, "rows=", len(frame), "features=", len(features))
PY
```

The historical sanity check is `rows=86864 features=57`, with 66,000 normal and 20,864 attack rows. Stop if the counts, labels, or feature count differ and determine why before comparing with the thesis.

#### Submit B1 and B2

Run from `waf-ml-starter/` after all three processed tables and the CSIC temporary table exist:

```bash
cd "$PROJECT_DIR"

# B1: CSIC feature ablation
rf57_job=$(sbatch --parsable "$JOB_DIR/run_csic_random_forest.sh")
sbatch "$JOB_DIR/run_csic_rf_25feat.sh"
sbatch "$JOB_DIR/run_csic_rf_34feat.sh"
sbatch --dependency="afterok:$rf57_job" "$JOB_DIR/run_csic_rf_top10.sh"

# B2: separate RF-57 training/evaluation inside the other two corpora
torpeda_job=$(sbatch --parsable "$JOB_DIR/run_torpeda_binary_rf.sh")
harvard_job=$(sbatch --parsable "$JOB_DIR/run_harvard_binary_rf.sh")

# Aggregate only after all three B2 result sources exist
sbatch \
  --dependency="afterok:$rf57_job:$torpeda_job:$harvard_job" \
  "$JOB_DIR/run_compare_binary.sh"
```

`run_csic_rf_34feat.sh` is a historical filename. Its list is 25 base features plus 8 entropy features, so the book correctly calls the representation **RF-33**. Some current metadata still says 34. RF-10-Gini depends on the newly generated RF-57 importance ranking; the thesis reference list was `uri_entropy`, `uri_pct_non_alnum_ratio`, `uri_len`, `path_depth`, `body_encoded`, `req_content_length`, `encoded`, `body_len`, `body_char_dist_i0`, and `body_pct_alpha`.

#### Expected B1 artifacts

RF-57 writes:

```text
resultsOptimo/csic_sin_registro/
|-- rf_supervised/
|   |-- model.joblib
|   |-- pred.csv
|   |-- metrics.json
|   `-- feature_importance_rf.csv
`-- rf_supervised_threshold/
    |-- threshold_sweep.csv
    `-- metrics.json
```

RF-25, RF-33, and RF-10-Gini use the analogous roots `rf_25feat*`, `rf_34feat*`, and `rf_top10*`; RF-10-Gini also writes `selected_features.csv`. The CSIC RF-57 model does not save an ordered feature manifest and relies on the input Parquet column order. Verify that its log reports exactly 57 features and preserve the ordered columns with the model.

Reference B1 values from Table 8.1 of the thesis:

| Representation | F1 | Recall | Precision | FPR | Balanced accuracy | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| RF-57 | 0.947 | 0.919 | 0.975 | 0.007 | 0.956 | 0.9971 |
| RF-33 | 0.935 | 0.905 | 0.966 | 0.010 | 0.948 | 0.9958 |
| RF-10-Gini | 0.914 | 0.863 | 0.971 | 0.008 | 0.927 | 0.9856 |
| RF-25 | 0.904 | 0.861 | 0.951 | 0.014 | 0.924 | 0.9816 |

#### Expected B2 artifacts

For `torpeda` and `harvard`, `train_binary_rf.py` writes:

```text
results/<dataset>/binary/
|-- model.joblib
|-- metrics.json
|-- pred.csv
|-- confusion_matrix.csv
|-- classification_report.txt
|-- feature_importance_rf.csv
|-- features_used.csv
`-- model_config.json

results/<dataset>/binary_threshold/
|-- metrics.json
|-- threshold_sweep.csv
|-- confusion_matrix.csv
`-- classification_report.txt
```

Feature rebuilding also overwrites `data/tmp/torpeda_binary_v2.parquet` or `data/tmp/harvard_binary_v2.parquet`. The aggregation job creates `results/comparison_binary_rf.csv`. It skips a corpus whose `pred.csv` is missing and silently falls back to threshold 0.50 when its `threshold_sweep.csv` is missing. Before aggregation, require all three prediction files and all three sweep files; afterward, require exactly the three expected dataset rows.

Reference B2 values from Table 8.3 of the thesis:

| Dataset | Threshold | F1 | FPR | FNR | MCC |
|---|---:|---:|---:|---:|---:|
| TorpEda | Optimized on test | 0.9989 | 0.0012 | 0.0021 | 0.9900 |
| SR-BH 2020 | 0.44 on test | 0.9507 | 0.0342 | 0.0514 | 0.9149 |
| CSIC 2010 | Optimized on test | 0.9467 | 0.0073 | N/D | N/D |

Do not infer the missing CSIC FNR or MCC from another similar run. For SR-BH, changing the threshold from 0.50 to 0.44 raised F1 only from 0.9504 to 0.9507 and raised FPR from 0.0301 to 0.0342. These are historical reference values, not guarantees for the current commit.

All `jobs/` launchers use fixed output paths, overwrite prior artifacts, and have no lock. Do not run two copies of the same job concurrently; archive a complete result package before rerunning.

### 4.2 M9 - final multiclass run

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
|-- informe_integrado_3datasets.md
|-- manifest_integrated_3datasets.json
|-- manifest_3datasetsMulticlase3.json
`-- multiclase/
    |-- comparison_integrated_3datasets.csv
    |-- comparison_integrated_3datasets.json
    |-- comparison_integrated_3datasets_compact73.csv
    |-- comparison_integrated_3datasets_compact73.json
    |-- performance_regression_unified73.csv
    |-- performance_regression_unified73.json
    |-- torpeda/
    |-- harvard/
    `-- egipcios/
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

### 4.3 L2 - final multilabel run

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

### 4.4 O3 - final one-class run

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
| B1/B2 | `threshold_sweep.csv`; selection is performed on the reported test and is exploratory | RF fit time is logged; the reusable B2 trainer records `train_time_s` | B2 records `predict_time_s` and writes `pred.csv`; these jobs do not implement an end-to-end production WAF benchmark |
| M9 | `model_selection_seconds` and candidate `fit_seconds` | `t_build_train_seconds` | `t_apply_inference_batch_seconds`; p50/p95/p99 and throughput in `operational_metrics.json` |
| L2 | `train_select_seconds`; per-label `fit_seconds` in tuning trials | `final_refit_seconds` when applicable; L2 is launched with no refit after thresholds | `t_apply_inference_batch_seconds`; online/batch latency and throughput in `operational_metrics.json` |
| O3 | `search.csv` contains candidate `fit_sec` and `score_sec` | final refit is logged; selected validation configuration in `best_validation_metrics.json` | evaluation `score_sec`; total `elapsed_sec` in `run_info.json` |

These measurements cover local feature extraction/model inference as implemented by the scripts. They do not include network transport, TLS, sustained concurrency, queues, a production WAF rule engine, or blocking decisions.

Historical Slurm accounting for the partner's binary branch reported:

| Evidence | Allocation | Wall / MaxRSS | Alignment note |
|---|---|---|---|
| B1 jobs 2476, 2496-2498 on c2 | 8 CPU, 16 GiB, no GPU each | 15/14/14/14 s; 0.920/0.727/0.946/0.957 GiB | Strong for RF-25 and RF-10; RF-57 is probable by chronology. Job 2497 says `rf_34feat`; RF-33 equivalence comes from the 25+8 count. |
| B2 TorpEda 2528 on c2 | 8 CPU, 16 GiB, no GPU | 00:00:25 / 0.872 GiB | Exact functional mapping. |
| B2 SR-BH 2529 on c2 | 8 CPU, 32 GiB, no GPU | 00:02:57 / 9.312 GiB | Exact functional mapping. |
| B2 comparison 2530 on c1 | 2 CPU, 8 GiB, no GPU | 00:00:04 / 0.269 GiB | Aggregation only; it does not train. |

These are historical observations, not promised runtimes for the current code or hardware.

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

### 10.1 Programs in `executables/`

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

### 10.2 Supporting code used by `jobs/`

Run module CLIs from `waf-ml-starter/` with `PYTHONPATH=src "$PYTHON" -m <module> ...`. The main partner components are:

| Source file | Purpose / status |
|---|---|
| `src/waf_ml/data/csic_loader.py` | Parses the three CSIC HTTP text files and builds binary/multiclass metadata. |
| `src/waf_ml/data/torpeda_loader.py` | Parses TorpEda XML requests and attack names. |
| `src/waf_ml/data/srbh_loader.py` | Loads SR-BH tables, identifies CAPEC columns, and derives task labels. |
| `src/waf_ml/features/http_features.py` | Current shared HTTP feature extractor; it returns 57 features. |
| `src/waf_ml/scripts/build_features_csic.py` | Builds `data/processed/csic/csic_features.parquet`; use `--use-headers` for the current CSIC path. |
| `src/waf_ml/scripts/build_features_torpeda.py` | Builds `data/processed/torpeda/torpeda_features.parquet` from XML. |
| `src/waf_ml/scripts/build_features.py` | Builds the SR-BH processed table from compatible CSV/TSV/GZ inputs. |
| `src/waf_ml/scripts/train_binary_rf.py` | Reusable B2 RF trainer for TorpEda/SR-BH; creates base and threshold-optimized result directories. Its `--feat-set 34` option actually selects 33 variables. |
| `src/waf_ml/optimo/ocsvmOptimo.py` | GPU-accelerated diagnostic OCSVM used by many auxiliary jobs; currently consumes only its fixed 25-feature list even when a job comment says 34/51/57. |
| `src/waf_ml/scripts/train_supervised.py` | Logistic Regression baseline used by auxiliary jobs; currently consumes a fixed 25-feature list. |
| `src/waf_ml/optimo/compare_results.py` | Compares two OCSVM metric packages and writes JSON/CSV summaries. |
| `src/waf_ml/optimo/generar_reporte_pdf.py` | Creates an optional PDF report from OCSVM metrics, benchmark, importance, and search files. |
| `src/waf_ml/scripts/train_ocsvm_updated.py` | Stand-alone scalable one-class prototype; no current `jobs/` launcher points to it. |
| `src/waf_ml/scripts/train_multilabel_lr_harvard_thesis.py` | Stand-alone SR-BH multilabel Logistic Regression prototype; it is not L1 or L2. |

## 11. Complete shell file catalog

### 11.1 Files in `executables/`

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

### 11.2 Files in `jobs/`

All 21 jobs pass `bash -n`, and their 34 embedded Python blocks parse. Except for the two explicitly marked special-CWD wrappers, execute each as:

```bash
cd "$PROJECT_DIR"
sbatch "$JOB_DIR/<file>.sh"
```

The requested resources are historical launcher defaults. Output paths are fixed and are overwritten on rerun.

| Job | Purpose, prerequisite, and expected result |
|---|---|
| `run_compare_binary.sh` | B2 aggregation; 2 CPU, 8 GiB, 10 min. Reads CSIC/TorpEda/SR-BH predictions and sweeps, then overwrites `results/comparison_binary_rf.csv`. It skips missing corpora and falls back to threshold 0.50 if a sweep is absent; verify three rows. |
| `run_csic_global_v2.sh` | Prepares `data/tmp/csic_sin_registro_v2.parquet`, then runs an auxiliary GPU OCSVM; 4 CPU, 32 GiB, 1 GPU, 2 h. Also writes `resultsOptimo/csic_sin_registro/nu05_v2/*`. Its comment says 51 features, but the current extractor creates 57 and the current OCSVM consumes only 25. |
| `run_csic_logreg_nobalanced.sh` | Intended unbalanced LogReg comparison and threshold sweep; 8 CPU, 16 GiB, 30 min. Writes `supervised_nobalanced*`. Current warning: it omits `--class-weight none`, so `train_supervised.py` keeps its `balanced` default; it also uses only 25 features, not the 57 stated in the comment. Do not interpret it as the advertised no-balance run without correcting and recording the code change. |
| `run_csic_logreg_threshold.sh` | Does not retrain. Sweeps the probabilities from `supervised_binary/pred.csv` and overwrites `supervised_binary_threshold/{threshold_sweep.csv,metrics.json}`; 4 CPU, 8 GiB, 15 min. Run `run_csic_supervised_binary.sh` first. |
| `run_csic_nicoRalf_nu.sh` | Four auxiliary GPU OCSVM variants (`nu=0.05`, `0.10`, `0.01`, and `mean_score` tuning); 4 CPU, 24 GiB, 1 GPU, 3 h. Writes four `nicoRalf_*` result directories with model, metrics, predictions, SHAP, benchmark, and one search table. |
| `run_csic_nicoRalf_nu_1000.sh` | Fast 1,000-row diagnostic for `nu=0.05/0.10`; 4 CPU, 16 GiB, 1 GPU, 30 min. Overwrites `data/tmp/csic_nicoRalf_1000.parquet` and two test result directories. |
| `run_csic_ocsvm.sh` | Auxiliary full-CSIC GPU OCSVM with tuning; 4 CPU, 24 GiB, 1 GPU, 3 days. Writes `resultsOptimo/csic/oneclass/` model, metrics, predictions, search, SHAP, and benchmark. **Special CWD:** submit from the repository root as shown in Section 3. |
| `run_csic_ocsvm_1000.sh` | 1,000-row version of the preceding OCSVM; 4 CPU, 24 GiB, 1 GPU, 1 h. Overwrites its sample and `oneclass_test1000/*`. **Special CWD:** submit from the repository root. |
| `run_csic_random_forest.sh` | B1 RF-57 and the CSIC source for B2; 8 CPU, 16 GiB, 30 min. Requires `csic_sin_registro_v2.parquet`; writes `rf_supervised/` and `rf_supervised_threshold/`. |
| `run_csic_rf_25feat.sh` | B1 RF-25; 8 CPU, 16 GiB, 20 min. Writes `rf_25feat/` and `rf_25feat_threshold/`. |
| `run_csic_rf_34feat.sh` | B1 RF-33 despite its historical filename; 8 CPU, 16 GiB, 20 min. Writes `rf_34feat*`; threshold metadata incorrectly retains `n_features=34`. |
| `run_csic_rf_top10.sh` | B1 RF-10-Gini; 8 CPU, 16 GiB, 20 min. Requires RF-57 importance first; writes `rf_top10*` plus `selected_features.csv`. |
| `run_csic_sin_registro.sh` | Auxiliary GPU OCSVM after removing URIs containing `registro`; 4 CPU, 24 GiB, 1 GPU, 3 days. Reuses an existing filtered Parquet solely by existence, then writes `csic_sin_registro/oneclass/*`; delete/archive stale derived data deliberately before a changed-input run. |
| `run_csic_sin_registro_1000.sh` | 1,000-row filtered OCSVM diagnostic; 4 CPU, 24 GiB, 1 GPU, 1 h. Rebuilds its temporary sample and overwrites `oneclass_test1000/*`. |
| `run_csic_solo_registro.sh` | Auxiliary OCSVM restricted to requests containing `registro`; 4 CPU, 16 GiB, 1 GPU, 1 h. Writes `data/tmp/csic_solo_registro.parquet` and `resultsOptimo/csic_solo_registro/nu05/*`. |
| `run_csic_solo_registro_entropy.sh` | Historical entropy-labelled OCSVM variant; 4 CPU, 16 GiB, 1 GPU, 1 h. Current extraction creates 57 columns but the OCSVM consumes only 25, so it is not the claimed 34-feature ablation. |
| `run_csic_solo_registro_v2.sh` | Historical char-distribution OCSVM variant; 4 CPU, 16 GiB, 1 GPU, 1 h. It also extracts 57 but trains on the OCSVM's fixed 25; it is not the claimed 51-feature ablation. |
| `run_csic_supervised_binary.sh` | Auxiliary balanced Logistic Regression; 8 CPU, 16 GiB, 30 min. Writes model, coefficients, metrics, predictions, and benchmark under `supervised_binary/`; current trainer uses 25 rather than the 57 claimed in the comment. |
| `run_harvard_binary_rf.sh` | B2 SR-BH RF-57; 8 CPU, 32 GiB, 1 h 30 min. Requires `harvard.parquet`; rebuilds `harvard_binary_v2.parquet` and writes `results/harvard/binary*`. |
| `run_torpeda_binary_rf.sh` | B2 TorpEda RF-57; 8 CPU, 16 GiB, 30 min. Requires `torpeda_features.parquet`; rebuilds `torpeda_binary_v2.parquet` and writes `results/torpeda/binary*`. |
| `run_torpeda_rf_multiclass.sh` | Exploratory TorpEda RF multiclass, not M1-M9; 8 CPU, 16 GiB, 30 min. Trains 25/33/57 variants and writes model/predictions/importance/metrics under `resultsOptimo/torpeda/rf_multiclass_*`. If `data/tmp/torpeda_v2.parquet` is absent it requires raw XML; it does not save the label encoder or ordered feature manifest. |

The auxiliary OCSVM/LogReg jobs are preserved for traceability, but their current feature-count mismatches mean that comments and filenames are not sufficient evidence of what was fitted. Inspect the actual program, input columns, and logged feature list before using their results.

The accompanying partner reports are `ablacion_features_csic.md` (B1), `experimentos_binario_cross_dataset.md` (B2), `resultados_pi1_csic.md` (CSIC comparison), `experimento_sin_registro.md`, `recalibracion_parametros.md`, and `protocolo_experimental_v1.md`. They preserve development context. Where an early report says RF-34, 51 features, or "cross-dataset generalization," use the thesis's audited interpretation above: RF-33, verify the actual current feature list, and separate training/evaluation within each corpus.

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
7. For B1/B2, confirm the logged feature count, remember that threshold selection used test, and verify `comparison_binary_rf.csv` contains exactly CSIC, TorpEda, and SR-BH.
8. Preserve code, launcher, inputs or hashes, taxonomy, configuration, metrics, predictions, model, ordered features, importance files, timing artifacts, and the Slurm job record together.

## 13. Known limitations of this directory

- Current B1/B2 launchers are present, but their historical byte identity, logs, package versions, and complete result packages were not recovered; the current commit must not be presented as the proven producer of the thesis metrics.
- B1/B2 choose their optimized threshold on the reported test and use row splits without fingerprint grouping, so their values are exploratory rather than independent confirmation.
- The historical O1/O2 one-class batch bodies and program provenance are unavailable; only O3 is represented by a current launcher/program pair.
- The exact Git commit used for each historical metric was not recorded.
- The datasets, processed Parquet files, virtual environment, and generated results are not included.
- `jobs/` uses fixed output paths, no locks, and no general resume protocol; reruns overwrite artifacts and concurrent copies can collide.
- Several auxiliary OCSVM/LogReg job comments claim 34/51/57 features while the current trainers consume only 25; the nominal no-balance LogReg job also retains the balanced default.
- The current CSIC RF-57 model lacks an ordered feature manifest, while the B2 TorpEda/SR-BH trainer does save `features_used.csv`.
- Several launchers retain historical absolute paths and root-level companion assumptions.
- The TorpEda v10 launcher has no companion Python in the branch.
- Some CPU-named L1 wrappers default to GPU mode, and some resume-named wrappers default to reprocessing.
- M1-M8 are exploratory because test results informed variables and taxonomies; some transitions also changed multiple factors. M9 remains a final development configuration rather than an independent confirmation.
- Operational measurements are local pipeline measurements, not production WAF benchmarks.
