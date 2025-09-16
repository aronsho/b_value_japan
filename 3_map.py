# ========= IMPORTS =========
import numpy as np
import pandas as pd
from pathlib import Path

from seismostats.analysis import estimate_mc_maxc, BPositiveBValueEstimator
from functions.main_functions import find_sequences, load_catalog
from functions.space_map import mac_space


# ======== SPECIFY PARAMETERS ===
# single value
RESULT_DIR = Path("results/map")
N_REALIZATIONS = 2
NS = np.array([100, 200, 400, 800, 1600, 3200, 6400,
              12800, 25600, 51200])  # number of tiles

# ======== LOAD PARAMETERS ======
DIR = Path("data")
variables_df = pd.read_csv(DIR / "variables.csv")
variables = variables_df.to_dict(orient="records")[0]

SHAPE_DIR = Path(variables["SHAPE_DIR"])
CAT_DIR = Path(variables["CAT_DIR"])

# transformation to local variables
p1 = np.array(eval(variables["p1"]))
p2 = np.array(eval(variables["p2"]))
EPSG_GEOGRAPHIC = variables["EPSG_GEOGRAPHIC"]
EPSG_JAPAN_M = variables["EPSG_JAPAN_M"]

# b-val estimation
MC_FIXED = variables["MC_FIXED"]
CORRECTION_FACTOR = variables["CORRECTION_FACTOR"]
DELTA_M = variables["DELTA_M"]
DMC = variables["DMC"]

# sequences
MAGNITUDE_THRESHOLD = variables["MAGNITUDE_THRESHOLD"]
RUPTURE_RELATION = variables["RUPTURE_RELATION"]
DIMENSION = variables["DIMENSION"]
DAYS_AFTER = variables["DAYS_AFTER"]
DAYS_BEFORE = variables["DAYS_BEFORE"]
RADIUS_FAR = variables["RADIUS_FAR"]
EXCLUDE_AFTERSHOCK_DAYS = variables["EXCLUDE_AFTERSHOCK_DAYS"]
MIN_N_SEQ = variables["MIN_N_SEQ"]

# ========= HELPERS =========


def estimate_mc(magnitudes, delta_m):
    """Maximum curvature as Mc method."""
    mc, _ = estimate_mc_maxc(magnitudes, delta_m, CORRECTION_FACTOR)
    return mc


# ========= MAIN =========
if __name__ == "__main__":
    # --- Load catalogs ---
    print('Loading catalogs...')
    fname_close = "df_japan_buffered_catalog_40km_3D.csv"
    fname_far = "df_japan_buffered_catalog_400km_3D.csv"
    cat_close = load_catalog(fname_close, MC_FIXED -
                             CORRECTION_FACTOR, DELTA_M, CAT_DIR)
    cat_far = load_catalog(fname_far, MC_FIXED -
                           CORRECTION_FACTOR, DELTA_M, CAT_DIR)

    # --- Find sequences ---
    print('Finding sequences...')
    seqs, main_idx, _ = find_sequences(
        cat_close,
        cat_far,
        magnitude_threshold=MAGNITUDE_THRESHOLD,
        relation=RUPTURE_RELATION,
        days_after=pd.Timedelta(days=DAYS_AFTER),
        days_before=pd.Timedelta(days=DAYS_BEFORE),
        exclude_aftershocks=pd.Timedelta(days=EXCLUDE_AFTERSHOCK_DAYS),
        dimension=DIMENSION,
        radius_far=RADIUS_FAR,
        min_n_seq=MIN_N_SEQ,
        post_include_aftershocks=True,
    )

    # --- Filter catalog ---
    df_large_close = cat_close.loc[main_idx]
    all_sequences = pd.concat(seqs + [df_large_close])
    filtered_df = cat_close.drop(all_sequences.index, errors="ignore")
    filtered_df.mc = min(filtered_df.magnitude)

    # coordinates and limits
    grid = np.array(
        [df_large_close["x"], df_large_close["y"], -df_large_close["z"]])
    coords = [filtered_df.x.values,
              filtered_df.y.values, -filtered_df.z.values]
    limits = [
        [coords[0].min(), coords[0].max()],
        [coords[1].min(), coords[1].max()],
        [coords[2].min(), coords[2].max()],
    ]

    # global volume and length scales (once)
    volume = (
        (limits[0][1] - limits[0][0])
        * (limits[1][1] - limits[1][0])
        * (limits[2][1] - limits[2][0])
    )
    length_scales = (volume / NS * 3 / (4 * np.pi)) ** (1 / 3)

    # --- Run mac_space for each grid size ---
    mac, mu_mac, std_mac = [], [], []
    print('Estimating maps...')
    for n in NS:
        print('current number of tiles:', n)
        b_avg, b_std, mac_spatial, mu_mac_spatial, std_mac_spatial = mac_space(
            coords=coords,
            mags=filtered_df.magnitude,
            delta_m=filtered_df.delta_m,
            mc=filtered_df.mc,
            times=filtered_df.time,
            limits=limits,
            n_space=n,
            n_realizations=N_REALIZATIONS,
            eval_coords=grid,
            min_num=25,
            method=BPositiveBValueEstimator,
            mc_method=estimate_mc,
            transform=True,
            voronoi_method="random",
            dmc=DMC,
        )

        # save b-value maps as a DataFrame
        b_df = pd.DataFrame({
            "b_avg": b_avg,
            "b_std": b_std
        })
        b_df.to_csv(RESULT_DIR / f"b_values_{n}.csv", index=False)

        # collect mac arrays
        mac.append(mac_spatial)
        mu_mac.append(mu_mac_spatial)
        std_mac.append(std_mac_spatial)

    # --- Save summary DataFrame ---
    df = pd.DataFrame({
        "n_space": NS,
        "mc": filtered_df.mc,
        "volume": volume,
        "length_scale": length_scales,
        "mac_spatial": np.array(mac),
        "mu_mac_spatial": np.array(mu_mac),
        "std_mac_spatial": np.array(std_mac),
    })

    out_file = RESULT_DIR / "macs_volume_lengthscale.csv"
    df.to_csv(out_file, index=False)
