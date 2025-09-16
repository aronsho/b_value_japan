# ========= IMPORTS =========
import numpy as np
import pandas as pd
from pathlib import Path

from seismostats.analysis import (
    b_significant_1D,
    estimate_mc_maxc,
    BPositiveBValueEstimator,
)
from functions.main_functions import find_sequences, load_catalog


# ========= CONFIG =========
CAT_DIR = Path("data/catalogs")
SHAPE_DIR = Path("data/shape_japan")
RESULT_DIR = Path("results/b_significant")

MAGNITUDE_THRESHOLD = 6.0      # mainshock magnitude threshold
MC_FIXED = 0.7                 # fixed magnitude of completeness
DELTA_M = 0.1                  # magnitude binning width
CORRECTION_FACTOR = 0.2        # correction factor for Mc (maxc)

RELATION = "surface"           # rupture length relation type
DAYS_AFTER = 100               # time window after main event
DAYS_BEFORE = 10 * 365         # time window before main event
DIMENSION = 3                  # 2D or 3D
RADIUS_FAR = 2.0               # radius multiplier for "far"

N_MS = np.arange(150, 250, 50)  # no of magnitudes used per b-value estimate
P_THRESHOLD = 0.05             # significance threshold


# ========= HELPERS =========
def get_histograms(
    seqs,
    main_idx,
    n_ms: np.ndarray,
    STAI_cutoff: pd.Timedelta,
    sort_parameter: str,
) -> np.ndarray:
    """
    Compute p-values for each sequence, for different sample sizes (n_ms).
    """
    p = np.zeros((len(seqs), len(n_ms)))

    for ii, sequence in enumerate(seqs):
        # estimate Mc
        mc, _ = estimate_mc_maxc(
            sequence.magnitude,
            fmd_bin=DELTA_M,
            correction_factor=CORRECTION_FACTOR,
        )

        # sort sequence
        sequence = sequence.sort_values(sort_parameter)

        # main event
        main_event = cat_close.loc[main_idx[ii]]

        # remove aftershocks inside cutoff
        idx_after = sequence[sequence["time"] >
                             main_event["time"] + STAI_cutoff].index
        idx_before = sequence[sequence["time"] < main_event["time"]].index
        sequence = sequence.loc[np.concatenate([idx_before, idx_after])]

        # cutoff below Mc
        mags = sequence.magnitude.values
        times = sequence.time.values
        idx = mags > mc - DELTA_M / 2
        mags, times = mags[idx], times[idx]

        # estimate p-values for different n_m
        for jj, n_m in enumerate(n_ms):
            if len(mags) < n_m * 10:
                p[ii, jj] = np.nan
                continue
            p_val, _, _, _ = b_significant_1D(
                mags, mc, DELTA_M, times, n_m, method=BPositiveBValueEstimator
            )
            p[ii, jj] = p_val

    return p


def run_with_cutoff(STAI_cutoff_days: int) -> None:
    cutoff = pd.Timedelta(days=STAI_cutoff_days)

    seqs, main_idx, _ = find_sequences(
        cat_close,
        cat_400km,
        magnitude_threshold=MAGNITUDE_THRESHOLD,
        relation=RELATION,
        days_after=pd.Timedelta(days=DAYS_AFTER),
        days_before=pd.Timedelta(days=DAYS_BEFORE),
        exclude_aftershocks=cutoff,
        dimension=DIMENSION,
        radius_far=RADIUS_FAR,
    )

    # time-sorted histograms
    p = get_histograms(seqs, main_idx, N_MS, cutoff, sort_parameter="time")
    np.savetxt(
        RESULT_DIR / f"p_values_time_{STAI_cutoff_days}dcutoff.csv",
        p,
        delimiter=",",
    )

    # space-sorted histograms
    p = get_histograms(
        seqs, main_idx, N_MS, cutoff, sort_parameter="distance_to_main"
    )
    np.savetxt(
        RESULT_DIR / f"p_values_space_{STAI_cutoff_days}dcutoff.csv",
        p,
        delimiter=",",
    )


# ========= MAIN =========
if __name__ == "__main__":
    # load catalogs
    fname_close = "df_japan_buffered_catalog_40km_3D.csv"
    fname_far = "df_japan_buffered_catalog_400km_3D.csv"
    cat_close = load_catalog(fname_close, MC_FIXED -
                             CORRECTION_FACTOR, DELTA_M, CAT_DIR)
    cat_400km = load_catalog(fname_far, MC_FIXED -
                             CORRECTION_FACTOR, DELTA_M, CAT_DIR)

    # run for multiple aftershock cutoffs
    for exclude_aftershock_days in [0, 1, 2]:
        run_with_cutoff(exclude_aftershock_days)
