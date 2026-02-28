import csv
import glob
import os
import re
from typing import Dict, List, Tuple

import numpy as np

SAMPLE_RE = re.compile(r"^(ML_\d+)")


class DataAlignmentError(RuntimeError):
    pass


def _sample_key_from_path(path: str) -> str:
    base = os.path.basename(path)
    m = SAMPLE_RE.match(base)
    if not m:
        raise ValueError(f"Could not parse sample id from filename: {base}")
    return m.group(1)


def discover_sample_files(data_dir: str) -> Dict[str, Dict[str, str]]:
    comp_files = glob.glob(os.path.join(data_dir, "Composition_Dataset", "*.csv"))
    xrd_files = glob.glob(os.path.join(data_dir, "XRD_Dataset", "*.csv"))
    oer_files = glob.glob(os.path.join(data_dir, "LSV_Dataset", "*_OER.csv"))

    mapping: Dict[str, Dict[str, str]] = {}

    for path in comp_files:
        sample = _sample_key_from_path(path)
        mapping.setdefault(sample, {})["composition"] = path
    for path in xrd_files:
        sample = _sample_key_from_path(path)
        mapping.setdefault(sample, {})["xrd"] = path
    for path in oer_files:
        sample = _sample_key_from_path(path)
        mapping.setdefault(sample, {})["oer"] = path

    missing = [
        (sample, sorted(set(["composition", "xrd", "oer"]) - set(paths.keys())))
        for sample, paths in sorted(mapping.items())
        if set(paths.keys()) != {"composition", "xrd", "oer"}
    ]
    if missing:
        raise DataAlignmentError(f"Missing modality files for samples: {missing}")

    return mapping


def load_composition(path: str, map_n_to_ni: bool = True) -> Tuple[List[int], List[Dict[str, float]]]:
    indices: List[int] = []
    rows: List[Dict[str, float]] = []

    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        headers = [h.strip() for h in reader.fieldnames or []]
        if "Index" not in headers:
            raise DataAlignmentError(f"No 'Index' column in composition file: {path}")

        element_headers = [h for h in headers if h != "Index"]
        for raw in reader:
            idx = int(float(raw["Index"]))
            fractions: Dict[str, float] = {}
            for e in element_headers:
                e_clean = e.strip()
                if map_n_to_ni and e_clean == "N":
                    e_clean = "Ni"
                val = raw.get(e, "").strip()
                if val == "":
                    continue
                fractions[e_clean] = float(val)
            indices.append(idx)
            rows.append(fractions)

    return indices, rows


def _load_curve_file(path: str) -> Tuple[np.ndarray, List[int], np.ndarray]:
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        if len(header) < 2:
            raise DataAlignmentError(f"Unexpected curve CSV header format: {path}")

        axis_name = header[0].strip()
        col_ids = [int(float(c)) for c in header[1:]]

        axis_vals: List[float] = []
        table_rows: List[List[float]] = []
        for row in reader:
            if not row:
                continue
            axis_vals.append(float(row[0]))
            vals: List[float] = []
            for token in row[1:]:
                token = token.strip()
                vals.append(float(token) if token != "" else np.nan)
            table_rows.append(vals)

    matrix = np.array(table_rows, dtype=float)  # shape: n_axis x n_cols
    axis = np.array(axis_vals, dtype=float)

    if matrix.shape[1] != len(col_ids):
        raise DataAlignmentError(
            f"Column count mismatch in {path}: matrix {matrix.shape[1]} vs header {len(col_ids)}"
        )

    _ = axis_name
    return axis, col_ids, matrix


def load_xrd(path: str) -> Tuple[np.ndarray, List[int], np.ndarray]:
    return _load_curve_file(path)


def load_oer(path: str) -> Tuple[np.ndarray, List[int], np.ndarray]:
    return _load_curve_file(path)


def validate_alignment(
    sample: str,
    comp_indices: List[int],
    xrd_col_ids: List[int],
    oer_col_ids: List[int],
) -> List[int]:
    set_comp = set(comp_indices)
    set_xrd = set(xrd_col_ids)
    set_oer = set(oer_col_ids)

    if not (set_comp == set_xrd == set_oer):
        raise DataAlignmentError(
            f"Index mismatch for {sample}: comp={len(set_comp)}, xrd={len(set_xrd)}, oer={len(set_oer)}"
        )

    common = sorted(set_comp)
    if len(common) == 0:
        raise DataAlignmentError(f"No aligned indices for sample {sample}")

    return common
