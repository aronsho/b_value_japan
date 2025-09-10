# ========= IMPORTS =========
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
from seismostats import Catalog
from seismostats.utils import CoordinateTransformer, cat_intersect_polygon

# own functions
from functions.transformation_functions import transform_and_rotate


# ========= CONFIG =========
DIMENSION = 3                 # 2 or 3
BUFFER_M = 30_000            # 30 km
EPSG_GEOGRAPHIC = 4326       # WGS84
EPSG_JAPAN_M = 6677          # Japan (meters)

SHAPE_DIR = Path("data/shape_japan")
CAT_DIR = Path("data/catalogs")


# ========= HELPERS =========
def load_clean_catalog(csv_path: Path) -> Catalog:
    # === GET EARTHQUAKE CATALOG (kept close to original) ===
    df = pd.read_csv(csv_path, parse_dates=["time"])

    # convert time to datetime (mixed formats)
    df["time"] = pd.to_datetime(df["time"], format="mixed")

    # exclude rows with missing magnitude
    df = df[~df["magnitude"].isna()]

    # only include event_type='earthquake'
    df = df[df["event_type"] == "earthquake"]

    # drop rows with missing depth
    df = df[~df["depth"].isna()]

    # create catalog object
    cat = Catalog(df)

    # set binning width
    cat.bin_magnitudes(0.1, inplace=True)

    # only consider events from 2000 onwards
    cat = cat[cat["time"] >= "2000-01-01"]

    # only consider events with depth <= 150 km
    cat = cat[cat["depth"] <= 150]

    return cat


def load_japan_polygon() -> gpd.GeoDataFrame:
    """
    Build a single polygon (GeoDataFrame) for mainland Japan intersected
    with an approximate CSV polygon mask.
    """
    # Mainland Japan (exclude Okinawa), then dissolve to one geometry
    gdf = gpd.read_file(SHAPE_DIR / "jp.shp")
    gdf = gdf[gdf["name"] != "Okinawa"]
    # union_all for modern GeoPandas; fallback to unary_union if needed
    combined = gdf.geometry.union_all()
    combined_gdf = gpd.GeoDataFrame(geometry=[combined], crs=gdf.crs)

    # CSV polygon mask
    df_lonlat = pd.read_csv(SHAPE_DIR / "polygon.csv")
    polygon_approx = Polygon(zip(df_lonlat["lon"], df_lonlat["lat"]))
    polygon_gdf = gpd.GeoDataFrame(geometry=[polygon_approx], crs=gdf.crs)

    # Intersection
    return gpd.overlay(combined_gdf, polygon_gdf, how="intersection")


def buffered_polygon_vertices_latlon(
    poly_gdf: gpd.GeoDataFrame,
    buffer_m: float,
    epsg_src: int,
    epsg_metric: int,
) -> np.ndarray:
    """
    Buffer a polygon (meters), return exterior vertices as [[lat, lon], ...].
    """
    # project to metric CRS, buffer, then back
    poly_metric = poly_gdf.to_crs(epsg=epsg_metric)
    buffered = poly_metric.buffer(buffer_m).to_crs(epsg=epsg_src)

    # Convert exterior coords to [[lat, lon], ...]
    exterior = np.asarray(list(buffered.geometry.iloc[0].exterior.coords))
    # exterior is [[lon, lat], ...] -> swap to [[lat, lon], ...]
    return exterior[:, [1, 0]]


def transform_catalog_3d(cat: Catalog) -> None:
    """
    In-place: compute local (x,y,z) via custom transform_and_rotate around two
    reference points near Kyoto. No distortion here, as no projection is used.
    """
    lats = cat["latitude"].values
    lons = cat["longitude"].values
    depths = cat["depth"].values

    p1 = np.array([35.018970, 135.769601, 0])  # reference point 1
    p2 = np.array([36.018970, 135.769601, 0])  # reference point 2

    cart_coords, _ = transform_and_rotate(p1, p2, lats, lons, depths)
    cat["y"] = cart_coords[0, :]
    cat["x"] = cart_coords[1, :]
    cat["z"] = cart_coords[2, :]


def transform_catalog_2d(cat: Catalog) -> None:
    """
    In-place: compute local 2D (x,y) with CoordinateTransformer using
    reference at (lon=135.769601, lat=35.018970). This uses a projection
    that introduces some distortion. It is used for the 2D case only.
    """
    ct_ref = CoordinateTransformer(EPSG_GEOGRAPHIC)
    ref_easting, ref_northing, _ = ct_ref.to_local_coords(
        135.769601, 35.018970, 0)
    ct = CoordinateTransformer(EPSG_GEOGRAPHIC, ref_easting, ref_northing)

    x, y, z = ct.to_local_coords(
        cat["longitude"].values,
        cat["latitude"].values,
        cat["depth"].values,
    )

    cat["x"], cat["y"], cat["z"] = x * 100, y * 100, z


def save_catalog(
        cat: Catalog,
        buffer_m: float,
        dim: int, out_dir: Path) -> Path:
    fname = f"df_japan_buffered_catalog_{int(buffer_m/1000)}km_{dim}D.csv"
    out_path = out_dir / fname
    cat.to_csv(out_path)
    return out_path


# ========= MAIN =========
def main() -> None:
    # 1) Catalog
    cat = load_clean_catalog(CAT_DIR / "all_JMA.csv")

    # 2) Geo mask (Japan ∩ approx polygon), buffer, intersect
    japan_poly = load_japan_polygon()
    verts_latlon = buffered_polygon_vertices_latlon(
        japan_poly,
        buffer_m=BUFFER_M,
        epsg_src=japan_poly.crs.to_epsg() or EPSG_GEOGRAPHIC,
        epsg_metric=EPSG_JAPAN_M,
    )
    cat = cat_intersect_polygon(cat, verts_latlon)

    # 3) Transform coordinates
    if DIMENSION == 2:
        transform_catalog_2d(cat)
    elif DIMENSION == 3:
        transform_catalog_3d(cat)
    else:
        raise ValueError("DIMENSION must be 2 or 3.")

    # 4) Save
    out_path = save_catalog(cat, BUFFER_M, DIMENSION, CAT_DIR)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
