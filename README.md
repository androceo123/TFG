--virtual enviroment for dependencies--

python3 -m venv .venv

source .venv/bin/activate

python -m pip install --upgrade pip

pip install -r requirements.txt

--generate proccessed dataset(size of 50000, not the whole dataset)--

python -m waf_ml.scripts.build_features --input data/raw/data_capec_multilabel.csv --output data/processed/features.parquet --sep ',' --sample-n 50000

--see the proccessed dataset--
python -c "import pandas as pd; df=pd.read_parquet('data/processed/features.parquet'); df.to_csv('data/processed/features_full.csv', index=False)"

--train the OCSVM model with that processed dataset
python -m waf_ml.scripts.train_ocsvm --data data/processed/features.parquet --label-col label_binary --out models/ocsvm.joblib

--train multiclass baseline--
python -m waf_ml.scripts.train_supervised --data data/processed/features.parquet --task multiclass --label-col label_multiclass --out models/lr_multiclass.joblib

--train multilabel training--
python -m waf_ml.scripts.train_supervised --data data/processed/features.parquet --task multilabel --label-col label_multilabel --out models/ovr_multilabel.joblib
