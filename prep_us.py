"""Build the unified CAMELS-US array from the MHPI release.

Source (public, no registration):
    https://mhpi-spatial.s3.us-east-2.amazonaws.com/mhpi-release/data/1-camels.zip
    -> unzip gives 1-camels/camels_daymetv2 (pickle) and 1-camels/gage_id.npy

Output npz keys
---------------
X     (B, T, 6)  [prcp, tavg, pet, dayl, srad, vp]
Y     (B, T)     streamflow in mm/day
A     (B, P)     static attributes
ids   (B,)
cell  (B,)      environment label, aridity tercile x snowfall median
dates (T,)
"""
import argparse
import datetime
import pickle

import numpy as np

AREA_COL = 11          # km^2, verified against CAMELS area_gages2
CFS_TO_MMDAY = 2.4466  # 1 cfs over 1 km^2 = 2.4466 mm/day
START = datetime.date(1980, 10, 1)


class SafeUnpickler(pickle.Unpickler):
    """Only numpy and builtins are needed to load the array file."""

    def find_class(self, module, name):
        if module.startswith("numpy") or module == "builtins":
            try:
                return super().find_class(module, name)
            except Exception:
                return super().find_class(module.replace("_core", "core"), name)
        raise pickle.UnpicklingError("blocked %s.%s" % (module, name))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="path to 1-camels/camels_daymetv2")
    ap.add_argument("--ids", default="", help="path to 1-camels/gage_id.npy (default: next to --src)")
    ap.add_argument("--out", default="data/us.npz")
    args = ap.parse_args()

    with open(args.src, "rb") as f:
        X, Y, A = SafeUnpickler(f).load()
    ids = np.load(args.ids or (args.src.rsplit("/", 1)[0] + "/gage_id.npy"))

    Y = Y[:, :, 0]
    ymm = (Y * CFS_TO_MMDAY / A[:, AREA_COL][:, None]).astype(np.float32)
    dates = np.array([np.datetime64(START + datetime.timedelta(days=i)) for i in range(X.shape[1])])

    P, T, PET = X[:, :, 0], X[:, :, 1], X[:, :, 2]
    aridity = np.nanmean(PET, 1) / np.maximum(np.nanmean(P, 1), 1e-3)
    snow = np.nansum(P * (T < 0), 1) / np.maximum(np.nansum(P, 1), 1e-6)
    e = np.quantile(aridity, [1 / 3, 2 / 3])
    a_lab = np.where(aridity <= e[0], "wet", np.where(aridity <= e[1], "mid", "dry"))
    s_lab = np.where(snow <= np.median(snow), "low", "high")
    cell = np.array(["%s_%s" % (a, s) for a, s in zip(a_lab, s_lab)])

    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, X=X.astype(np.float32), Y=ymm, A=A.astype(np.float32),
                        ids=ids, cell=cell, dates=dates)
    print("saved", args.out, X.shape, ymm.shape, A.shape)
    print({c: int((cell == c).sum()) for c in sorted(set(cell))})


if __name__ == "__main__":
    main()
