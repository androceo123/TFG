# WAF-ML Starter (SR-BH 2020 / HTTP traffic)
This repo is a starter scaffold to implement the TFG proposal:
- Parse HTTP requests
- Extract HTTP-oriented features (URI / query / encoding / suspicious tokens / method / content-length)
- Train baseline models (One-Class SVM + supervised baselines)
- Evaluate binary / multiclass / multilabel settings
- Run a tiny WAF-like reverse proxy in passive or active mode

## Quickstart
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt

# 1) Build features from a CSV (you will download SR-BH 2020 separately)
python -m waf_ml.scripts.build_features --input data/raw/srbh.csv --output data/processed/features.parquet

# 2) Train OCSVM (binary anomaly detection)
python -m waf_ml.scripts.train_ocsvm --data data/processed/features.parquet --label-col label_binary --out models/ocsvm.joblib

# 3) Train multiclass or multilabel baseline
python -m waf_ml.scripts.train_supervised --data data/processed/features.parquet --task multiclass --label-col label_multiclass --out models/svm_multiclass.joblib
python -m waf_ml.scripts.train_supervised --data data/processed/features.parquet --task multilabel --label-col label_multilabel --out models/ovr_lr_multilabel.joblib

# 4) Run proxy (passive by default)
python -m waf_ml.waf.proxy --config configs/config.example.yaml
```
> Notes:
> - You must adapt column mapping in `waf_ml/data/srbh_loader.py` after you inspect the SR-BH CSV headers.
> - `label_binary`, `label_multiclass`, `label_multilabel` are example columns you create in your processed dataset.

## Repo layout
- `src/waf_ml/features/http_features.py`: feature extraction
- `src/waf_ml/data/srbh_loader.py`: loader + label normalization stubs
- `src/waf_ml/scripts/*`: CLI scripts to build features/train/eval
- `src/waf_ml/waf/proxy.py`: minimal FastAPI reverse proxy (passive/active)
