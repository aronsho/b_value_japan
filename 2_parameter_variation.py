# sbatch --array=0-10 --mem-per-cpu=4000 --wrap="python 2_b_significant.py"


# ========= IMPORTS =========
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from seismostats import Catalog
from seismostats.analysis import (
    BPositiveBValueEstimator,
    ClassicBValueEstimator,
    estimate_mc_maxc,
)

from functions.main_functions import find_sequences, load_catalog

# ======== SPECIFY PARAMETERS ===
# single value
RESULT_DIR = Path("results/map")

# multiple values
MAGNITUDE_THRESHOLDS = [6.0]
B_METHODS = ["global", "local"]
RUPTURE_RELATIONS = ["surface"]
DAYS_AFTER = [100]
DISTANCES_TO_COAST = [40]
DIMENSIONS = [3]
EXCLUDE_AFTERSHOCKS_DAYS = [1]

# ======== LOAD PARAMETERS ======
DIR = Path("data")
variables_df = pd.read_csv(DIR / "variables.csv")
variables = variables_df.to_dict(orient="records")[0]

CAT_DIR = Path(variables["CAT_DIR"])

# b-val estimation, catalog in general
MC_FIXED = variables["MC_FIXED"]
CORRECTION_FACTOR = variables["CORRECTION_FACTOR"]
DELTA_M = variables["DELTA_M"]
DMC = variables["DMC"]
MIN_N_M = variables["MIN_N_M"]

# for sequeneces
DAYS_BEFORE = variables["DAYS_BEFORE"]
RADIUS_FAR = variables["RADIUS_FAR"]
RADIUS_CLOSE = variables["RADIUS_CLOSE"]
MIN_N_SEQ = variables["MIN_N_SEQ"]

# ========= HELPERS =========


def estimate_b_values(sequences: list[pd.DataFrame],
                      main_indices: list[int],
                      cat: Catalog,
                      b_method: str,
                      delta_m: float,
                      correction_factor: float,
                      radius_close: float,
                      n_check: int) -> pd.DataFrame:
    """Estimate b-values for all sequences and return as DataFrame."""
    n_check = n_check if b_method == "global" else 2 * n_check
    estimator = BPositiveBValueEstimator()

    columns = [
        'b_sequence', 'std_sequence', 'p_l_sequence',
        'b_close_after', 'std_close_after', 'p_l_close_after',
        'b_close_before', 'std_close_before', 'p_l_close_before',
        'b_far_after', 'std_far_after', 'p_l_far_after',
        'b_far_before', 'std_far_before', 'p_l_far_before',
        'b_before', 'std_before', 'p_l_before',
        'b_after', 'std_after', 'p_l_after',
        'b_close', 'std_close', 'p_l_close',
        'b_far', 'std_far', 'p_l_far',
        'b_before1', 'std_before1', 'p_l_before1',
        'b_before2', 'std_before2', 'p_l_before2',
        'b_before1_close', 'std_before1_close', 'p_l_before1_close',
        'b_before2_close', 'std_before2_close', 'p_l_before2_close'
    ]
    results = []

    for seq, idx in zip(sequences, main_indices):
        main = cat.loc[idx]

        # estimate Mc
        mc, _ = estimate_mc_maxc(
            seq.magnitude, fmd_bin=delta_m, correction_factor=correction_factor
        )

        # initial b-value estimation
        estimator.calculate(seq.magnitude, mc=mc,
                            delta_m=delta_m, times=seq.time, dmc=DMC)
        if b_method == "global":
            mags, times = estimator.magnitudes, estimator.times
            distances = seq["distance_to_main"].values[estimator.idx]
            estimator2 = ClassicBValueEstimator()
        elif b_method == "local":
            mask = seq["magnitude"] >= mc
            mags = seq["magnitude"].values[mask]
            times = seq["time"].values[mask]
            distances = seq["distance_to_main"].values[mask]
            estimator2 = BPositiveBValueEstimator()

        def calc(sub_mags, sub_times) -> tuple[float, float, float]:
            if len(sub_mags) < n_check:
                return np.nan, np.nan, np.nan
            if b_method == "global":
                estimator2.calculate(sub_mags, mc=DMC, delta_m=delta_m)
            elif b_method == "local":
                estimator2.calculate(sub_mags, mc=mc,
                                     delta_m=delta_m, dmc=DMC,
                                     times=sub_times)
            return (
                estimator2.b_value,
                estimator2.std,
                estimator2.p_lilliefors()
            )

        # precompute masks
        close = distances <= radius_close * main.rupture_length
        before = times < main["time"]
        after = times > main["time"]
        close_before = close & before
        close_after = close & after
        far = ~close
        far_before = far & before
        far_after = far & after

        # estimate and store results
        b_sequence, std_sequence, p_l_sequence = calc(mags, times)
        b_close, std_close, p_l_close = calc(mags[close], times[close])
        b_far, std_far, p_l_far = calc(mags[far], times[far])
        b_before, std_before, p_l_before = calc(mags[before], times[before])
        b_after, std_after, p_l_after = calc(mags[after], times[after])
        b_close_before, std_close_before, p_l_close_before = calc(
            mags[close_before], times[close_before])
        b_close_after, std_close_after, p_l_close_after = calc(
            mags[close_after], times[close_after])
        b_far_before, std_far_before, p_l_far_before = calc(
            mags[far_before], times[far_before])
        b_far_after, std_far_after, p_l_far_after = calc(
            mags[far_after], times[far_after])

        # before 1 and before 2
        n_before = np.sum(before)
        b_before1, std_before1, p_l_before1 = calc(
            mags[before][:n_before//2], times[before][:n_before//2])
        b_before2, std_before2, p_l_before2 = calc(
            mags[before][n_before//2:], times[before][n_before//2:])

        n_close_before = np.sum(close_before)
        b_before1_close, std_before1_close, p_l_before1_close = calc(
            mags[close_before][:n_close_before//2],
            times[close_before][:n_close_before//2])
        b_before2_close, std_before2_close, p_l_before2_close = calc(
            mags[close_before][n_close_before//2:],
            times[close_before][n_close_before//2:])

        results.append({
            'b_sequence': b_sequence,
            'std_sequence': std_sequence,
            'p_l_sequence': p_l_sequence,
            'b_close_after': b_close_after,
            'std_close_after': std_close_after,
            'p_l_close_after': p_l_close_after,
            'b_close_before': b_close_before,
            'std_close_before': std_close_before,
            'p_l_close_before': p_l_close_before,
            'b_far_after': b_far_after,
            'std_far_after': std_far_after,
            'p_l_far_after': p_l_far_after,
            'b_far_before': b_far_before,
            'std_far_before': std_far_before,
            'p_l_far_before': p_l_far_before,
            'b_before': b_before,
            'std_before': std_before,
            'p_l_before': p_l_before,
            'b_after': b_after,
            'std_after': std_after,
            'p_l_after': p_l_after,
            'b_close': b_close,
            'std_close': std_close,
            'p_l_close': p_l_close,
            'b_far': b_far,
            'std_far': std_far,
            'p_l_far': p_l_far,
            'b_before1': b_before1,
            'std_before1': std_before1,
            'p_l_before1': p_l_before1,
            'b_before2': b_before2,
            'std_before2': std_before2,
            'p_l_before2': p_l_before2,
            'b_before1_close': b_before1_close,
            'std_before1_close': std_before1_close,
            'p_l_before1_close': p_l_before1_close,
            'b_before2_close': b_before2_close,
            'std_before2_close': std_before2_close,
            'p_l_before2_close': p_l_before2_close
        })

    df_b_values = pd.DataFrame(results, index=main_indices, columns=columns)
    return df_b_values

# ========= MAIN =========


def main() -> None:
    param_grid = product(
        MAGNITUDE_THRESHOLDS,
        B_METHODS,
        RUPTURE_RELATIONS,
        DAYS_AFTER,
        DISTANCES_TO_COAST,
        DIMENSIONS,
        EXCLUDE_AFTERSHOCKS_DAYS,
    )
    param_combinations = list(param_grid)
    print(f"{len(param_combinations)} parameter combinations found.")

    for ii, params in enumerate(param_combinations, 1):
        (mag_thr, b_method, relation,
         days_after, dist_coast, dim, excl_days) = params
        print(
            f"Processing comb. {ii} of {len(param_combinations)}: {params}")

        # load catalogs
        print('Loading catalogs...')
        fname_close = f"df_japan_buffered_catalog_{dist_coast}km_{dim}D.csv"
        fname_far = f"df_japan_buffered_catalog_400km_{dim}D.csv"
        cat_close = load_catalog(
            fname_close, MC_FIXED - CORRECTION_FACTOR, DELTA_M, CAT_DIR)
        cat_far = load_catalog(fname_far, MC_FIXED -
                               CORRECTION_FACTOR, DELTA_M, CAT_DIR)

        # find sequences
        print('Finding sequences...')
        seqs, main_idx, cat_close = find_sequences(
            cat_close, cat_far,
            magnitude_threshold=mag_thr,
            relation=relation,
            days_after=pd.Timedelta(days=days_after),
            days_before=pd.Timedelta(days=DAYS_BEFORE),
            exclude_aftershocks=pd.Timedelta(days=excl_days),
            dimension=dim,
            radius_far=RADIUS_FAR,
            min_n_seq=MIN_N_SEQ
        )
        print(f"  {len(seqs)} sequences found.")

        # estimate b-values
        print('Estimating b-values...')
        df_b = estimate_b_values(
            seqs,
            main_idx,
            cat_close,
            b_method,
            delta_m=DELTA_M,
            correction_factor=CORRECTION_FACTOR,
            radius_close=RADIUS_CLOSE,
            n_check=MIN_N_M
        )

        # save
        save_name = (
            f"df_b_values_{mag_thr}M_{b_method}_{relation}_"
            f"{days_after}days_{dist_coast}km_{dim}D_{excl_days}days.csv"
        )
        out_path = RESULT_DIR / save_name
        df_b.to_csv(out_path)
        print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
