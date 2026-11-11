"""Build the unified CAMELS-AUS v2 array.

Source (public, no registration): Zenodo doi:10.5281/zenodo.12575680
   03_streamflow.zip, 04_attributes.zip, 02_location_boundary_area.zip,
   05_hydrometeorology.zip  ->  unzip all four into one directory.

Output schema matches prep_us.py exactly, so the two datasets are
interchangeable downstream.
"""
import argparse
import glob
import os
import struct

import numpy as np


def read_wide(path):
    """Read a CAMELS-AUS timeseries CSV -> (dates, station_ids, arr[T,B])."""
    import pandas as pd
    df = pd.read_csv(path, low_memory=False)
    low = [str(c).strip().lower() for c in df.columns[:3]]
    if low[:3] == ["year", "month", "day"]:
        dates = pd.to_datetime(dict(year=df.iloc[:, 0], month=df.iloc[:, 1],
                                    day=df.iloc[:, 2]), errors="coerce"
                               ).values.astype("datetime64[D]")
        stations = [str(c) for c in df.columns[3:]]
        arr = df.iloc[:, 3:].apply(pd.to_numeric, errors="coerce").values.astype(np.float32)
        return dates, stations, arr
    first = str(df.columns[0])
    # case 1: first column holds dates
    d = pd.to_datetime(df.iloc[:, 0], errors="coerce", dayfirst=False)
    if d.notna().mean() > 0.9:
        dates = d.values.astype("datetime64[D]")
        stations = [str(c) for c in df.columns[1:]]
        arr = df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").values.astype(np.float32)
        return dates, stations, arr
    # case 2: first column holds station ids, dates are in the header
    dates = pd.to_datetime(pd.Index(df.columns[1:]), errors="coerce").values.astype("datetime64[D]")
    stations = [str(x) for x in df.iloc[:, 0]]
    arr = df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").values.astype(np.float32).T
    return dates, stations, arr


def daylength(lat_deg, doy):
    """Astronomical day length in seconds."""
    phi = np.radians(lat_deg)
    delta = 0.409 * np.sin(2 * np.pi * doy / 365.0 - 1.39)
    x = np.clip(-np.tan(phi) * np.tan(delta), -1, 1)
    ws = np.arccos(x)
    return 24.0 / np.pi * ws * 3600.0


def shp_centroids(path):
    import shapefile
    r = shapefile.Reader(path)
    out = {}
    for sr in r.iterShapeRecords():
        pts = sr.shape.points
        if not pts:
            continue
        out[str(sr.record["station_id"] if "station_id" in sr.record.as_dict()
                else sr.record[0])] = (np.mean([p[1] for p in pts]),
                                       np.mean([p[0] for p in pts]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="directory holding the unzipped CAMELS-AUS v2 archives")
    ap.add_argument("--out", default="data/aus.npz")
    ap.add_argument("--start", default="1980-10-01")
    ap.add_argument("--end", default="2014-09-30")
    args = ap.parse_args()

    R = args.root
    files = {
        "prcp": "precipitation_SILO.csv",
        "tmax": "tmax_SILO.csv",
        "tmin": "tmin_SILO.csv",
        "pet": "et_short_crop_SILO.csv",
        "srad": "radiation_SILO.csv",
        "vp": "vp_SILO.csv",
    }
    flows = None
    common = None
    stations = None
    dyn = {}
    for key, fn in files.items():
        hits = glob.glob(os.path.join(R, "**", fn), recursive=True)
        if not hits:
            print("MISSING", fn)
            continue
        dates, sts, arr = read_wide(hits[0])
        print("%-5s %s  dates %d  stations %d" % (key, fn, len(dates), len(sts)))
        dyn[key] = (dates, sts, arr)
        common = set(sts) if common is None else (common & set(sts))

    fhits = glob.glob(os.path.join(R, "**", "streamflow_mmd.csv"), recursive=True)
    dates_f, sts_f, arr_f = read_wide(fhits[0])
    print("flow  %d dates  %d stations" % (len(dates_f), len(sts_f)))
    common = common & set(sts_f)
    stations = [s for s in dyn["prcp"][1] if s in common]
    print("common stations:", len(stations))

    win = (dates_f >= np.datetime64(args.start)) & (dates_f <= np.datetime64(args.end))
    dts = dates_f[win]
    T = len(dts)
    print("window", dts[0], "->", dts[-1], T, "days")

    def align(dates, sts, arr):
        idx_d = {d: i for i, d in enumerate(dates)}
        rows = np.full((len(stations), T), np.nan, dtype=np.float32)
        idx_s = {s: j for j, s in enumerate(sts)}
        for i, s in enumerate(stations):
            if s not in idx_s:
                continue
            col = arr[:, idx_s[s]]
            for t, d in enumerate(dts):
                j = idx_d.get(d)
                if j is not None:
                    rows[i, t] = col[j]
        return rows

    prcp = align(*(dyn["prcp"]))
    tmax = align(*(dyn["tmax"]))
    tmin = align(*(dyn["tmin"]))
    pet = align(*(dyn["pet"]))
    srad = align(*(dyn["srad"]))
    vp = align(*(dyn["vp"]))
    flow = align(dates_f, sts_f, arr_f)
    tavg = (tmax + tmin) / 2.0

    # day length from catchment centroid
    cent = {}
    shps = glob.glob(os.path.join(R, "**", "*.shp"), recursive=True)
    if shps:
        try:
            cent = shp_centroids(shps[0])
            print("centroids:", len(cent))
        except Exception as e:
            print("shapefile failed:", e)
    doy = np.array([np.datetime64(d, "D").astype("datetime64[D]").astype(object).timetuple().tm_yday
                    for d in dts])
    lat = np.array([cent.get(s, (np.nan, np.nan))[0] for s in stations])
    if np.isnan(lat).all():
        lat = np.full(len(stations), -25.0)
    dayl = np.array([daylength(la, doy) for la in lat], dtype=np.float32)

    X = np.stack([prcp, tavg, pet, dayl, srad, vp], axis=2)
    Y = flow

    # attributes
    arows = {}
    for f in glob.glob(os.path.join(R, "**", "CatchmentAttributes_*.csv"), recursive=True):
        with open(f, "r", encoding="utf-8-sig", errors="replace") as fh:
            head = fh.readline().strip().split(",")
            for line in fh:
                parts = line.rstrip("\n").split(",")
                if not parts:
                    continue
                arows.setdefault(parts[0], {})
                for k, v in zip(head[1:], parts[1:]):
                    try:
                        arows[parts[0]][k] = float(v)
                    except ValueError:
                        pass
    cols = sorted({k for v in arows.values() for k in v})
    A = np.full((len(stations), len(cols)), np.nan, dtype=np.float32)
    for i, s in enumerate(stations):
        for j, c in enumerate(cols):
            if c in arows.get(s, {}):
                A[i, j] = arows[s][c]
    print("attributes:", A.shape)

    keep = np.isfinite(flow).sum(1) > 2000
    print("stations with enough flow:", int(keep.sum()))
    X, Y, A, stations = X[keep], Y[keep], A[keep], [s for s, k in zip(stations, keep) if k]

    aridity = np.nanmean(X[:, :, 2], 1) / np.maximum(np.nanmean(X[:, :, 0], 1), 1e-3)
    e = np.quantile(aridity, [1 / 3, 2 / 3])
    cell = np.where(aridity <= e[0], "wet", np.where(aridity <= e[1], "mid", "dry"))
    print("aridity terciles %.3f / %.3f" % (e[0], e[1]),
          {c: int((cell == c).sum()) for c in set(cell)})

    np.savez_compressed(args.out, X=X, Y=Y, A=A,
                        ids=np.array(stations), cell=cell, dates=dts)
    print("saved", args.out, X.shape, Y.shape, A.shape)


if __name__ == "__main__":
    main()
