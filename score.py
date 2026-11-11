"""Aggregate run outputs into the tables and figures of the paper.

Usage
-----
python score.py --runs runs --out figures
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--donor", default="", help="directory with donor_*.json from donor.py")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows = []
    for f in glob.glob(os.path.join(args.runs, "*", "result.json")):
        r = json.load(open(f))
        idv = r.get("id", r.get("idv"))
        ood = r.get("ood")
        if not idv or not ood:
            continue
        rows.append(dict(dataset=r.get("dataset", ""), arch=r.get("arch", "lstm"),
                         holdout=r["holdout"], method=r.get("method", "erm"),
                         seed=int(r.get("seed", 0)),
                         id_log=idv[1] if len(idv) > 2 else idv[0],
                         ood_log=ood[1] if len(ood) > 2 else ood[0]))
    if not rows:
        print("no results found")
        return

    agg = defaultdict(list)
    for r in rows:
        agg[(r["dataset"], r["arch"], r["method"], r["holdout"])].append(r)

    print("%-10s %-12s %-5s %-10s %3s %8s %8s %8s" %
          ("dataset", "arch", "meth", "holdout", "n", "ID", "OOD", "gap"))
    print("-" * 76)
    for k in sorted(agg):
        v = agg[k]
        i = np.array([x["id_log"] for x in v])
        o = np.array([x["ood_log"] for x in v])
        print("%-10s %-12s %-5s %-10s %3d %8.3f %8.3f %+8.3f" %
              (k[0], k[1], k[2], k[3], len(v), i.mean(), o.mean(), (i - o).mean()))

    with open(os.path.join(args.out, "results.csv"), "w") as f:
        f.write("dataset,arch,method,holdout,seed,id_log,ood_log\n")
        for r in sorted(rows, key=lambda x: (x["dataset"], x["arch"], x["method"], x["holdout"], x["seed"])):
            f.write("%s,%s,%s,%s,%d,%.4f,%.4f\n" % (r["dataset"], r["arch"], r["method"],
                                                    r["holdout"], r["seed"],
                                                    r["id_log"], r["ood_log"]))
    print("\nwrote", os.path.join(args.out, "results.csv"))

    # ---- architecture comparison figure ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    order = {"us.npz": ["wet_low", "wet_high", "mid_low", "mid_high", "dry_low", "dry_high"],
             "aus.npz": ["wet", "mid", "dry"]}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), dpi=160,
                             gridspec_kw={"width_ratios": [6, 3]})
    for k, (ds, folds) in enumerate(order.items()):
        ax = axes[k]
        present = [f for f in folds if any(key[0] == ds and key[3] == f for key in agg)]
        if not present:
            continue
        x = np.arange(len(present))
        w = 0.36
        for j, (arch, col, lab) in enumerate([("lstm", "#12414A", "EA-LSTM"),
                                              ("transformer", "#C2703D", "Transformer")]):
            mu, sd = [], []
            for f in present:
                v = agg.get((ds, arch, "erm", f), [])
                if v:
                    g = np.array([q["id_log"] - q["ood_log"] for q in v])
                    mu.append(g.mean()); sd.append(g.std())
                else:
                    mu.append(np.nan); sd.append(np.nan)
            ax.bar(x + (j - 0.5) * w, mu, w, yerr=sd, capsize=2, color=col,
                   label=lab, error_kw=dict(lw=0.8))
        ax.axhline(0, color="#999", lw=0.8)
        ax.set_xticks(x); ax.set_xticklabels(present, fontsize=9)
        ax.set_ylim(-0.32, 0.92); ax.grid(alpha=0.2, axis="y")
        ax.set_title(ds.replace(".npz", "").upper(), fontsize=11)
        if k == 0:
            ax.set_ylabel("environment-shift gap (median log-KGE)", fontsize=9.5)
            ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle("Environment-shift gap by environment and model class", fontsize=11.5)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "gap_by_environment.png"))
    fig.savefig(os.path.join(args.out, "gap_by_environment.pdf"))
    print("wrote", os.path.join(args.out, "gap_by_environment.png"))


if __name__ == "__main__":
    main()
