import json
import math
from typing import Dict, List, Tuple

import numpy as np
from scipy.signal import find_peaks

from .config import (
    DEFAULT_MILLER_FCC,
    MASTER_ELEMENTS,
    NOBLE_ELEMENTS,
    NON_NOBLE_ELEMENTS,
    R_GAS_CONSTANT,
    SCHERRER_K,
    XRD_WAVELENGTH_ANGSTROM,
)


def load_element_properties(path: str) -> Dict[str, Dict[str, float]]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _normalize_fractions(comp: Dict[str, float]) -> Dict[str, float]:
    total = sum(max(v, 0.0) for v in comp.values())
    if total <= 0:
        return {}
    return {k: max(v, 0.0) / total for k, v in comp.items() if max(v, 0.0) > 0}


def _weighted_mean(fractions: Dict[str, float], prop: Dict[str, float]) -> float:
    return sum(x * prop[e] for e, x in fractions.items())


def composition_features(
    comp: Dict[str, float],
    props: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    x = _normalize_fractions(comp)
    if not x:
        return {}

    radius = {e: props[e]["atomic_radius"] for e in x}
    chi = {e: props[e]["electronegativity"] for e in x}
    d_e = {e: props[e]["d_electrons"] for e in x}

    weighted_props = [
        "electronegativity",
        "atomic_radius",
        "d_electrons",
        "valence_electrons",
        "first_ionization_energy",
        "electron_affinity",
        "work_function",
        "cohesive_energy",
        "bulk_modulus",
        "standard_reduction_potential",
        "group_number",
        "surface_energy",
        "oxide_formation_enthalpy",
        "oxygen_affinity",
        "mo_bond_strength_proxy",
        "oxidation_tendency",
    ]

    feats: Dict[str, float] = {}
    for p in weighted_props:
        feats[f"mean_{p}"] = _weighted_mean(x, {e: props[e][p] for e in x})

    r_bar = _weighted_mean(x, radius)
    size_term = sum(x[e] * (1.0 - radius[e] / r_bar) ** 2 for e in x)
    feats["size_mismatch_delta"] = math.sqrt(max(size_term, 0.0))

    mix_entropy = -R_GAS_CONSTANT * sum(xi * math.log(xi) for xi in x.values() if xi > 0)
    feats["mixing_entropy"] = mix_entropy

    chi_bar = _weighted_mean(x, chi)
    feats["electronegativity_variance"] = sum(x[e] * (chi[e] - chi_bar) ** 2 for e in x)

    feats["vec"] = _weighted_mean(x, {e: props[e]["valence_electrons"] for e in x})

    d_bar = _weighted_mean(x, d_e)
    feats["d_electron_variance"] = sum(x[e] * (d_e[e] - d_bar) ** 2 for e in x)

    present = list(x.keys())
    max_dchi = 0.0
    max_dr = 0.0
    for i in range(len(present)):
        for j in range(i + 1, len(present)):
            ei, ej = present[i], present[j]
            max_dchi = max(max_dchi, abs(chi[ei] - chi[ej]))
            max_dr = max(max_dr, abs(radius[ei] - radius[ej]))
    feats["max_electronegativity_difference"] = max_dchi
    feats["max_radius_difference"] = max_dr

    for i in range(len(MASTER_ELEMENTS)):
        for j in range(i + 1, len(MASTER_ELEMENTS)):
            ei, ej = MASTER_ELEMENTS[i], MASTER_ELEMENTS[j]
            xi, xj = x.get(ei, 0.0), x.get(ej, 0.0)
            feats[f"pair_{ei}_{ej}"] = xi * xj

    feats["noble_fraction"] = sum(x.get(e, 0.0) for e in NOBLE_ELEMENTS)
    feats["non_noble_fraction"] = sum(x.get(e, 0.0) for e in NON_NOBLE_ELEMENTS)

    for e in MASTER_ELEMENTS:
        feats[f"frac_{e}"] = x.get(e, 0.0)

    return feats


def _interp_x(x0: float, y0: float, x1: float, y1: float, y_target: float) -> float:
    if y1 == y0:
        return 0.5 * (x0 + x1)
    return x0 + (y_target - y0) * (x1 - x0) / (y1 - y0)


def _fwhm(theta: np.ndarray, intensity: np.ndarray, peak_idx: int) -> float:
    i_peak = intensity[peak_idx]
    half = i_peak / 2.0

    left = peak_idx
    while left > 0 and intensity[left] > half:
        left -= 1
    right = peak_idx
    n = len(intensity)
    while right < n - 1 and intensity[right] > half:
        right += 1

    if left == peak_idx or right == peak_idx or left == 0 or right == n - 1:
        return float("nan")

    left_theta = _interp_x(theta[left], intensity[left], theta[left + 1], intensity[left + 1], half)
    right_theta = _interp_x(theta[right - 1], intensity[right - 1], theta[right], intensity[right], half)
    return right_theta - left_theta


def _top3_peaks(intensity: np.ndarray) -> List[int]:
    peaks, _ = find_peaks(intensity)
    if len(peaks) == 0:
        return [int(np.argmax(intensity))]
    order = np.argsort(intensity[peaks])[::-1]
    top = peaks[order][:3]
    return [int(i) for i in top]


def xrd_features(
    twotheta_deg: np.ndarray,
    intensity_raw: np.ndarray,
    hkl: Tuple[int, int, int] = DEFAULT_MILLER_FCC,
) -> Dict[str, float]:
    intensity = np.array(intensity_raw, dtype=float)

    finite = np.isfinite(twotheta_deg) & np.isfinite(intensity)
    theta = twotheta_deg[finite]
    y = intensity[finite]
    if len(theta) < 5:
        return {}

    peak_idx = int(np.argmax(y))
    peak_pos = float(theta[peak_idx])
    peak_int = float(y[peak_idx])

    top = _top3_peaks(y)
    i1 = float(y[top[0]]) if len(top) > 0 else peak_int
    i2 = float(y[top[1]]) if len(top) > 1 else float("nan")
    i3 = float(y[top[2]]) if len(top) > 2 else float("nan")

    fwhm_deg = _fwhm(theta, y, peak_idx)

    theta_rad = math.radians(peak_pos / 2.0)
    if math.sin(theta_rad) > 0:
        d_hkl = XRD_WAVELENGTH_ANGSTROM / (2.0 * math.sin(theta_rad))
        h2k2l2 = hkl[0] ** 2 + hkl[1] ** 2 + hkl[2] ** 2
        lattice_a = d_hkl * math.sqrt(h2k2l2)
    else:
        lattice_a = float("nan")

    beta_rad = math.radians(fwhm_deg) if np.isfinite(fwhm_deg) else float("nan")
    if np.isfinite(beta_rad) and beta_rad > 0 and math.cos(theta_rad) > 0:
        crystallite_size = (SCHERRER_K * XRD_WAVELENGTH_ANGSTROM) / (beta_rad * math.cos(theta_rad))
    else:
        crystallite_size = float("nan")

    feats = {
        "xrd_peak_position_2theta": peak_pos,
        "xrd_peak_intensity": peak_int,
        "xrd_fwhm_deg": fwhm_deg,
        "xrd_i2_i1": i2 / i1 if i1 != 0 and np.isfinite(i2) else float("nan"),
        "xrd_i3_i1": i3 / i1 if i1 != 0 and np.isfinite(i3) else float("nan"),
        "xrd_lattice_parameter_a": lattice_a,
        "xrd_crystallite_size": crystallite_size,
    }
    return feats
