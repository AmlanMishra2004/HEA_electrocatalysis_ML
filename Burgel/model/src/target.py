from typing import Dict, Optional, Tuple

import numpy as np
from scipy.signal import savgol_filter

from .config import OER_REFERENCE_POTENTIAL_V, SAVGOL_POLYORDER, SAVGOL_WINDOW, TARGET_CURRENT_A_CM2


def _smooth_signal(y: np.ndarray) -> np.ndarray:
    n = len(y)
    window = min(SAVGOL_WINDOW, n if n % 2 == 1 else n - 1)
    if window < 5:
        return y
    poly = min(SAVGOL_POLYORDER, window - 1)
    try:
        return savgol_filter(y, window_length=window, polyorder=poly)
    except Exception:
        return y


def eta_at_target_current(
    potential_mv: np.ndarray,
    current_a: np.ndarray,
    target_current_a_cm2: float = TARGET_CURRENT_A_CM2,
    reference_v: float = OER_REFERENCE_POTENTIAL_V,
    smooth: bool = True,
) -> Tuple[Optional[float], Dict[str, str]]:
    finite = np.isfinite(potential_mv) & np.isfinite(current_a)
    if np.count_nonzero(finite) < 3:
        return None, {"reason": "insufficient_finite_points"}

    e_v = potential_mv[finite] / 1000.0
    j = current_a[finite]

    if smooth:
        j_eval = _smooth_signal(j)
    else:
        j_eval = j

    crossing_idx = None
    for i in range(len(j_eval) - 1):
        j0, j1 = j_eval[i], j_eval[i + 1]
        e0, e1 = e_v[i], e_v[i + 1]
        if e1 <= e0:
            continue
        if j1 <= j0:
            continue
        if j0 <= target_current_a_cm2 <= j1:
            crossing_idx = i
            break

    if crossing_idx is None:
        for i in range(len(j_eval) - 1):
            j0, j1 = j_eval[i], j_eval[i + 1]
            e0, e1 = e_v[i], e_v[i + 1]
            if e1 <= e0:
                continue
            if min(j0, j1) <= target_current_a_cm2 <= max(j0, j1):
                crossing_idx = i
                break

    if crossing_idx is None:
        return None, {"reason": "no_target_crossing"}

    i = crossing_idx
    j0, j1 = j_eval[i], j_eval[i + 1]
    e0, e1 = e_v[i], e_v[i + 1]

    if j1 == j0:
        e_cross = 0.5 * (e0 + e1)
    else:
        e_cross = e0 + (target_current_a_cm2 - j0) * (e1 - e0) / (j1 - j0)

    eta = e_cross - reference_v
    if not np.isfinite(eta):
        return None, {"reason": "non_finite_eta"}

    return float(eta), {"reason": "ok"}
