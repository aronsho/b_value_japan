# python 2_b_significant.py

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

# ======== SPECIFY PARAMETERS ===
RESULT_DIR = Path("results/b_significant")

# single value
P_THRESHOLD = 0.05  # significance threshold

# multiple values
EXCLUDE_AFTERSHOCK_DAYS = [0, 1, 2]  # no of days after mainshock to exclude
N_MS = np.arange(150, 250, 50)  # no of magnitudes used for b-value estimation

# ======== LOAD PARAMETERS ======
DIR = Path("data")
variables_df = pd.read_csv(DIR / "variables.csv")
variables = variables_df.to_dict(orient="records")[0]

SHAPE_DIR = Path(variables["SHAPE_DIR"])
CAT_DIR = Path(variables["CAT_DIR"])

# b-val estimation
MC_FIXED = variables["MC_FIXED"]
CORRECTION_FACTOR = variables["CORRECTION_FACTOR"]
DELTA_M = variables["DELTA_M"]
DMC = variables["DMC"]  # check if used

# sequences
DIMENSION = variables["DIMENSION"]
MAGNITUDE_THRESHOLD = variables["MAGNITUDE_THRESHOLD"]
RUPTURE_RELATION = variables["RUPTURE_RELATION"]
DAYS_AFTER = variables["DAYS_AFTER"]
DAYS_BEFORE = variables["DAYS_BEFORE"]
RADIUS_FAR = variables["RADIUS_FAR"]
MIN_N_SEQ = variables["MIN_N_SEQ"]

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
        relation=RUPTURE_RELATION,
        days_after=pd.Timedelta(days=DAYS_AFTER),
        days_before=pd.Timedelta(days=DAYS_BEFORE),
        exclude_aftershocks=cutoff,
        dimension=DIMENSION,
        radius_far=RADIUS_FAR,
        min_n_seq=MIN_N_SEQ,
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
    for exclude_aftershock_days in EXCLUDE_AFTERSHOCK_DAYS:
        run_with_cutoff(exclude_aftershock_days)
