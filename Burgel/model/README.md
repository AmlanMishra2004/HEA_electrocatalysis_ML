# OER Feature Engineering + Modeling Pipeline

This folder contains a full pipeline to engineer HEA physics-guided descriptors and train a model for OER target prediction (`eta_at_10mA`) from the `15453660` dataset.

## What It Does
- Loads and aligns:
  - `Composition_Dataset/*.csv`
  - `XRD_Dataset/*.csv`
  - `LSV_Dataset/*_OER.csv`
- Aligns rows by `(sample_id, index)` where composition row index = XRD column = OER column.
- Engineers features:
  - composition weighted means and disorder descriptors
  - electronic proxies
  - OER-relevant physics proxies
  - pairwise interaction terms
  - XRD geometric descriptors
- Builds target `eta_at_10mA` from each OER curve.
- Splits data 70:10:20 (train/val/test) with seed 42.
- Trains:
  - preferred: `xgboost.XGBRegressor`
  - fallback: numpy ridge regression if XGBoost is unavailable
- Saves dataset, splits, model, metrics, feature importance, and predictions.

## Environment Setup
Create an environment inside `model/.venv`:

```bash
cd /home/amlan/energy3/Burgel/model
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If package installation fails (e.g., restricted network), the pipeline still runs using numpy/scipy fallback training.

## Run
From project root:

```bash
python3 /home/amlan/energy3/Burgel/model/run_pipeline.py
```

Optional flags:

```bash
python3 /home/amlan/energy3/Burgel/model/run_pipeline.py \
  --data-dir /home/amlan/energy3/Burgel/15453660 \
  --artifacts-dir /home/amlan/energy3/Burgel/model/artifacts \
  --properties /home/amlan/energy3/Burgel/model/element_properties.json \
  --seed 42
```

## Outputs
Generated in `model/artifacts/`:
- `dataset_features.csv`
- `splits.json`
- `metrics.json`
- `feature_importance.csv`
- `predictions_train.csv`
- `predictions_val.csv`
- `predictions_test.csv`
- `run_summary.json`
- model file:
  - `model_xgb.json` (if XGBoost used), or
  - `model_ridge.npz` (fallback)

## Notes
- `N` in `ML_1...EDX.csv` is mapped to `Ni`.
- OER target is overpotential at 10 mA/cm²:
  - interpolate potential at `j=0.01 A/cm²`
  - compute `eta = E - 1.23 V`
- Electrochemical conditions like pH/electrolyte/scan-rate are not present in source CSVs and are not included.
