# -*- coding: utf-8 -*-
"""Classical regionalisation baseline: k-nearest donor basins.

For every basin in the held-out environment, find the k most similar basins
(in standardised static-attribute space) among the training basins and
predict streamflow as the inverse-distance-weighted mean of their observed
hydrographs.  This is the benchmark that any "regionalisation breaks here"
claim has to beat.
"""
import argparse
import json
import os

import numpy as np


def kge(o, s):
    if len(o) < 30 or np.std(o) < 1e-6 or np.std(s) < 1e-6:
        return np.nan
    r = np.corrcoef(o, s)[0, 1]
    return 1 - np.sqrt((r - 1) ** 2 + (np.std(s) / np.std(o) - 1) ** 2 +
                       (np.mean(s) / max(abs(np.mean(o)), 1e-9) - 1) ** 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--holdout", required=True, help="environment cell to predict")
    ap.add_argument("--id_frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, nargs="+", default=[1, 5, 20])
    ap.add_argument("--min_days", type=int, default=1000)
    ap.add_argument("--out", default="runs/donor")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    d = np.load(args.npz, allow_pickle=True)
    Y, A, cell = d["Y"], d["A"], d["cell"].astype(str)

    # standardise attributes on finite values
    A = np.where(np.isfinite(A), A, np.nanmean(A, 0))
    mu, sd = np.nanmean(A, 0), np.nanstd(A, 0)
    sd[sd < 1e-6] = 1
    As = (A - mu) / sd

    rng = np.random.RandomState(args.seed)
    ood = cell == args.holdout
    seen = np.where(~ood)[0]
    id_idx = rng.choice(seen, int(len(seen) * args.id_frac), replace=False)
    ood_idx = np.where(ood)[0]

    def run(targets, donors, k):
        D = np.linalg.norm(As[targets][:, None, :] - As[donors][None, :, :], axis=2)
        order = np.argsort(D, axis=1)[:, :k]
        dd = np.take_along_axis(D, order, axis=1)
        w = 1.0 / (dd + 1e-6)
        w /= w.sum(1, keepdims=True)
        pred = np.zeros((len(targets), Y.shape[1]), dtype=np.float32)
        for i in range(len(targets)):
            stack = Y[donors[order[i]]]            # (k, T)
            stack = np.nan_to_num(stack)
            pred[i] = (w[i][:, None] * stack).sum(0)
        kr, kl = [], []
        for i, b in enumerate(targets):
            ok = np.isfinite(Y[b]) & np.isfinite(pred[i])
            if ok.sum() < args.min_days:
                continue
            o, pp = Y[b][ok], pred[i][ok]
            a1 = kge(o, pp)
            a2 = kge(np.log1p(o), np.log1p(np.maximum(pp, 0)))
            if np.isfinite(a1):
                kr.append(a1)
            if np.isfinite(a2):
                kl.append(a2)
        if not kr:
            return (np.nan, np.nan, 0)
        return (float(np.median(kr)), float(np.median(kl)), len(kr))

    res = {"holdout": args.holdout, "k": {}}
    for k in args.k:
        res["k"][k] = {"id": run(id_idx, np.setdiff1d(seen, id_idx), k),
                       "ood": run(ood_idx, seen, k)}
        a, b = res["k"][k]["id"], res["k"][k]["ood"]
        print("k=%-3d  ID raw %.3f log %.3f (n=%d) | OOD(%s) raw %.3f log %.3f (n=%d)"
              % (k, a[0], a[1], a[2], args.holdout, b[0], b[1], b[2]))
    json.dump(res, open(os.path.join(args.out, "donor_%s.json" % args.holdout), "w"), indent=1)
    print("saved", os.path.join(args.out, "donor_%s.json" % args.holdout))


if __name__ == "__main__":
    main()
