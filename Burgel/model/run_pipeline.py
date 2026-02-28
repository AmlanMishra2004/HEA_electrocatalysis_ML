#!/usr/bin/env python3
import argparse
import csv
import json
import os
from collections import Counter
from typing import Dict, List

import numpy as np

from src.config import (
    DEFAULT_ARTIFACTS_DIR,
    DEFAULT_DATA_DIR,
    DEFAULT_PROPERTIES_PATH,
    MAX_VALID_ETA,
    MIN_VALID_ETA,
    RANDOM_SEED,
    TRAIN_RATIO,
    VAL_RATIO,
)
from src.features import composition_features, load_element_properties, xrd_features
from src.io_data import (
    discover_sample_files,
    load_composition,
    load_oer,
    load_xrd,
    validate_alignment,
)
from src.target import eta_at_target_current
from src.train import build_split_data, make_splits, train_model


def _write_csv(path: str, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="OER feature engineering and model training pipeline")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--artifacts-dir", default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--properties", default=DEFAULT_PROPERTIES_PATH)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    os.makedirs(args.artifacts_dir, exist_ok=True)

    sample_files = discover_sample_files(args.data_dir)
    element_props = load_element_properties(args.properties)

    records: List[Dict[str, object]] = []
    dropped_by_reason: Counter = Counter()

    for sample in sorted(sample_files.keys(), key=lambda s: int(s.split("_")[1])):
        files = sample_files[sample]

        comp_idx, comp_rows = load_composition(files["composition"], map_n_to_ni=True)
        twotheta, xrd_col_ids, xrd_mat = load_xrd(files["xrd"])
        potential_mv, oer_col_ids, oer_mat = load_oer(files["oer"])

        common_ids = validate_alignment(sample, comp_idx, xrd_col_ids, oer_col_ids)

        comp_map = {idx: row for idx, row in zip(comp_idx, comp_rows)}
        xrd_id_to_pos = {cid: i for i, cid in enumerate(xrd_col_ids)}
        oer_id_to_pos = {cid: i for i, cid in enumerate(oer_col_ids)}

        for idx in common_ids:
            comp = comp_map[idx]
            x_curve = xrd_mat[:, xrd_id_to_pos[idx]]
            o_curve = oer_mat[:, oer_id_to_pos[idx]]

            eta, info = eta_at_target_current(potential_mv, o_curve, smooth=True)
            if eta is None:
                dropped_by_reason[info.get("reason", "unknown")] += 1
                continue
            if not (MIN_VALID_ETA <= eta <= MAX_VALID_ETA):
                dropped_by_reason["eta_out_of_range"] += 1
                continue

            comp_feat = composition_features(comp, element_props)
            xrd_feat = xrd_features(twotheta, x_curve)
            if not comp_feat or not xrd_feat:
                dropped_by_reason["empty_feature_block"] += 1
                continue

            row: Dict[str, object] = {
                "row_id": f"{sample}_{idx}",
                "sample_id": sample,
                "index": idx,
                "target_eta_at_10mA": float(eta),
            }
            row.update(comp_feat)
            row.update(xrd_feat)
            records.append(row)

    if not records:
        raise RuntimeError("No valid rows produced after filtering. Check target crossing and parsing.")

    peak_positions = np.array([float(r["xrd_peak_position_2theta"]) for r in records], dtype=float)
    peak_ref = float(np.median(peak_positions[np.isfinite(peak_positions)]))
    for r in records:
        r["xrd_peak_shift_vs_ref"] = float(r["xrd_peak_position_2theta"]) - peak_ref

    meta_cols = ["row_id", "sample_id", "index", "target_eta_at_10mA"]
    feature_names = [
        k
        for k in records[0].keys()
        if k not in meta_cols
    ]

    clean_records: List[Dict[str, object]] = []
    for r in records:
        vals = np.array([float(r[f]) for f in feature_names] + [float(r["target_eta_at_10mA"])], dtype=float)
        if np.all(np.isfinite(vals)):
            clean_records.append(r)
        else:
            dropped_by_reason["non_finite_features"] += 1

    if not clean_records:
        raise RuntimeError("All rows removed due to non-finite feature values.")

    X = np.array([[float(r[f]) for f in feature_names] for r in clean_records], dtype=float)
    y = np.array([float(r["target_eta_at_10mA"]) for r in clean_records], dtype=float)

    idx_train, idx_val, idx_test = make_splits(len(clean_records), TRAIN_RATIO, VAL_RATIO, args.seed)
    split = build_split_data(X, y, idx_train, idx_val, idx_test)

    train_out = train_model(split, feature_names, args.artifacts_dir)

    split_labels = np.array(["" for _ in range(len(clean_records))], dtype=object)
    split_labels[idx_train] = "train"
    split_labels[idx_val] = "val"
    split_labels[idx_test] = "test"

    dataset_rows: List[Dict[str, object]] = []
    for i, r in enumerate(clean_records):
        out = dict(r)
        out["split"] = split_labels[i]
        dataset_rows.append(out)

    dataset_path = os.path.join(args.artifacts_dir, "dataset_features.csv")
    dataset_fieldnames = list(dataset_rows[0].keys())
    _write_csv(dataset_path, dataset_rows, dataset_fieldnames)

    splits = {
        "seed": args.seed,
        "counts": {
            "total": len(clean_records),
            "train": int(len(idx_train)),
            "val": int(len(idx_val)),
            "test": int(len(idx_test)),
        },
        "row_ids": {
            "train": [clean_records[i]["row_id"] for i in idx_train],
            "val": [clean_records[i]["row_id"] for i in idx_val],
            "test": [clean_records[i]["row_id"] for i in idx_test],
        },
    }
    with open(os.path.join(args.artifacts_dir, "splits.json"), "w", encoding="utf-8") as fh:
        json.dump(splits, fh, indent=2)

    importance_path = os.path.join(args.artifacts_dir, "feature_importance.csv")
    _write_csv(importance_path, train_out["importance_rows"], ["feature", "importance"])

    pred_map = {
        "train": (idx_train, train_out["pred_train"]),
        "val": (idx_val, train_out["pred_val"]),
        "test": (idx_test, train_out["pred_test"]),
    }
    for name, (idxs, preds) in pred_map.items():
        rows = []
        for k, ridx in enumerate(idxs):
            src = clean_records[int(ridx)]
            rows.append(
                {
                    "row_id": src["row_id"],
                    "sample_id": src["sample_id"],
                    "index": src["index"],
                    "y_true": float(src["target_eta_at_10mA"]),
                    "y_pred": float(preds[k]),
                }
            )
        _write_csv(
            os.path.join(args.artifacts_dir, f"predictions_{name}.csv"),
            rows,
            ["row_id", "sample_id", "index", "y_true", "y_pred"],
        )

    summary = {
        "n_samples_valid": len(clean_records),
        "n_samples_raw": len(records),
        "target": "eta_at_10mA",
        "xrd_peak_reference_2theta": peak_ref,
        "dropped_by_reason": dict(dropped_by_reason),
        "model_type": train_out["model_type"],
        "model_path": train_out["model_path"],
    }
    with open(os.path.join(args.artifacts_dir, "run_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print("Pipeline complete")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
