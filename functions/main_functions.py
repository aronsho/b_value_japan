import numpy as np
import pandas as pd
from seismostats import Catalog

from functions.general_functions import dist_to_ref


def load_catalog(
        fname: str, mc: float, delta_m: float, cat_dir: str) -> Catalog:
    """Load, filter, and return a Catalog from CSV."""
    df = pd.read_csv(cat_dir / fname, index_col=0)
    df["time"] = pd.to_datetime(df["time"], format="mixed")
    cat = Catalog(df)
    cat.delta_m = delta_m
    cat.mc = mc
    cat = cat[cat["magnitude"] >= mc]

    return cat


def rupture_length(magnitude: np.ndarray, relation: str) -> np.ndarray:
    """Return rupture length [km] using Wells & Coppersmith relations."""
    if relation == "surface":
        a, b = -3.22, 0.69
    elif relation == "subsurface":
        a, b = -2.44, 0.59
    else:
        raise ValueError(f"Unknown rupture relation: {relation}")
    return 10 ** (a + b * magnitude)


def distance_series(cat_like: pd.DataFrame,
                    main: pd.DataFrame,
                    dimension: int) -> np.ndarray:
    """Vectorized distance from every event in `cat_like` to `main` event."""
    if dimension == 3:
        return dist_to_ref(
            cat_like["x"], main["x"],
            cat_like["y"], main["y"],
            cat_like["z"], main["z"])
    if dimension == 2:
        # type: ignore[arg-type]
        return dist_to_ref(cat_like["x"], main["x"], cat_like["y"], main["y"])
    raise ValueError("DIMENSION must be 2 or 3.")


def find_sequences(cat_close: Catalog,
                   cat_far: Catalog,
                   magnitude_threshold: float,
                   relation: str,
                   days_after: pd.Timedelta,
                   days_before: pd.Timedelta,
                   exclude_aftershocks: pd.Timedelta,
                   dimension: int,
                   radius_far: float,
                   min_n_seq: int,
                   post_include_aftershocks: bool = False,
                   ) -> tuple[list[pd.DataFrame], list[int], Catalog]:
    """
    Identify large-event sequences in catalog.
    Returns (list of sequences, list of main event indices, updated cat_close).
    """

    # select large events
    large_far = cat_far[cat_far["magnitude"] >= magnitude_threshold].copy()
    mask_close = cat_close["magnitude"] >= magnitude_threshold
    large_close = cat_close[mask_close].copy()

    # add rupture lengths
    large_far["rupture_length"] = rupture_length(
        large_far["magnitude"].values, relation)
    rupt_len = rupture_length(large_close["magnitude"].values, relation)
    large_close["rupture_length"] = rupt_len
    cat_close.loc[mask_close, "rupture_length"] = rupt_len

    # find sequences
    sequences, main_indices = [], []
    for idx, main in large_close.iterrows():
        start = main["time"] - days_before
        stop = main["time"] + days_after

        # avoid overlap with other large events
        for jj in large_far.index:
            if jj == idx:
                continue
            other = large_far.loc[jj]

            dist = distance_series(other, main, dimension)
            if dist < radius_far * (
                    main["rupture_length"] + other["rupture_length"]):
                if other["time"] > main["time"]:
                    stop = min(stop, other["time"])
                elif other["time"] < main["time"]:
                    start = max(start, other["time"] + days_after)

        if start > main["time"]:
            continue

        # select nearby events
        dist_all = distance_series(cat_close, main, dimension)

        # mask for sequence
        mask = (dist_all <= radius_far * main["rupture_length"]) & (
            (cat_close["time"] > start) & (cat_close["time"] < stop)
        )
        seq = cat_close[mask].copy()
        seq["distance_to_main"] = dist_all[mask]

        # drop the main event and aftershocks
        mask = (seq["time"] < main["time"]) | (
            seq["time"] > main["time"] + exclude_aftershocks)
        seq_loop = seq[mask]

        # only keep sequences with > min_n_seq events
        if len(seq_loop) > min_n_seq:
            if post_include_aftershocks:
                sequences.append(seq)
            else:
                sequences.append(seq_loop)
            main_indices.append(idx)

    return sequences, main_indices, cat_close
