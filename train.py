"""Leave-one-environment-out training and evaluation.

Usage
-----
python train.py --npz data/us.npz --holdout dry_high --method erm \
                --arch lstm --seed 1 --epochs 60 --out runs/dry_high_erm_s1
"""
import argparse
import json
import os

import numpy as np
import torch

from models import build_model


def kge(o, s):
    if len(o) < 30 or np.std(o) < 1e-6 or np.std(s) < 1e-6:
        return np.nan
    r = np.corrcoef(o, s)[0, 1]
    return 1 - np.sqrt((r - 1) ** 2 + (np.std(s) / np.std(o) - 1) ** 2 +
                       (np.mean(s) / max(abs(np.mean(o)), 1e-9) - 1) ** 2)


def sample_nse(pred, target, mask):
    """Per-window Nash-Sutcliffe error. The denominator floor keeps flat
    windows (zero-flow periods) from dominating the gradient."""
    n = mask.sum(1).clamp(min=1.0)
    mean = (target * mask).sum(1) / n
    num = (((pred - target) ** 2) * mask).sum(1)
    den = (((target - mean[:, None]) ** 2) * mask).sum(1)
    return torch.clamp(num / den.clamp(min=5.0), max=20.0)


def split_indices(cell, holdout, id_frac, seed):
    rng = np.random.RandomState(seed)
    ood = cell == holdout
    seen = np.where(~ood)[0]
    id_idx = rng.choice(seen, int(len(seen) * id_frac), replace=False)
    return np.setdiff1d(seen, id_idx), id_idx, np.where(ood)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="prepared dataset (us.npz / aus.npz)")
    ap.add_argument("--holdout", required=True, help="environment to hold out")
    ap.add_argument("--method", default="erm", choices=["erm", "dro", "vrex"])
    ap.add_argument("--arch", default="lstm", choices=["lstm", "transformer"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seq", type=int, default=270)
    ap.add_argument("--warmup", type=int, default=60, help="days at the start of each window excluded from the loss")
    ap.add_argument("--per_group", type=int, default=30)
    ap.add_argument("--id_frac", type=float, default=0.15)
    ap.add_argument("--beta", type=float, default=1.0, help="V-REx penalty")
    ap.add_argument("--eta", type=float, default=0.05, help="GroupDRO step size")
    ap.add_argument("--min_days", type=int, default=1000,
                    help="minimum valid days required to score a basin")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    d = np.load(args.npz, allow_pickle=True)
    X = d["X"].astype(np.float32)
    Y = d["Y"].astype(np.float32)
    A = d["A"].astype(np.float32)
    cell = d["cell"].astype(str)
    print("data", X.shape, "| cells", {c: int((cell == args.holdout).sum()) for c in [args.holdout]})

    tr_idx, id_idx, ood_idx = split_indices(cell, args.holdout, args.id_frac, args.seed)
    print("train %d | ID %d | OOD %d" % (len(tr_idx), len(id_idx), len(ood_idx)))

    # scalers from training basins only -- no leakage from the held-out environment
    d_mean = np.nanmean(X[tr_idx], axis=(0, 1))
    d_std = np.nanstd(X[tr_idx], axis=(0, 1))
    d_std[d_std < 1e-6] = 1.0
    Xn = np.nan_to_num(np.clip((X - d_mean) / d_std, -6, 6)).astype(np.float32)

    a_mean = np.nanmean(A[tr_idx], 0)
    a_std = np.nanstd(A[tr_idx], 0)
    a_std[a_std < 1e-6] = 1.0
    S = np.clip((np.where(np.isfinite(A), A, a_mean) - a_mean) / a_std, -6, 6).astype(np.float32)

    y_mean = np.nanmean(Y[tr_idx])
    y_std = np.nanstd(Y[tr_idx])
    Yn = np.nan_to_num((Y - y_mean) / y_std).astype(np.float32)

    Xt = torch.from_numpy(Xn).to(dev)
    St = torch.from_numpy(S).to(dev)
    Yt = torch.from_numpy(Yn).to(dev)
    Mt = torch.from_numpy(np.isfinite(Y).astype(np.float32)).to(dev)

    T = X.shape[1]
    starts = list(range(0, T - args.seq + 1, args.seq))
    groups = sorted(set(cell[tr_idx]))
    win = {g: np.array([(b, s) for b in tr_idx if cell[b] == g for s in starts]) for g in groups}
    for g in groups:
        print("  group %-9s basins %3d windows %5d" % (g, sum(1 for b in tr_idx if cell[b] == g), len(win[g])))

    model = build_model(args.arch, X.shape[2], S.shape[1], args.hidden, args.dropout).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    q = torch.ones(len(groups), device=dev) / len(groups)
    steps = max(1, min(len(win[g]) for g in groups) // args.per_group)

    for ep in range(args.epochs):
        model.train()
        tot = 0.0
        for st in range(steps):
            bs, ss = [], []
            for g in groups:
                n = len(win[g])
                for k in np.random.choice(n, args.per_group, replace=n < args.per_group):
                    bs.append(win[g][k][0]); ss.append(win[g][k][1])
            sel = np.stack([np.array(bs), np.array(ss)], 1)
            b = torch.from_numpy(sel[:, 0]).long().to(dev)
            xb = torch.stack([Xt[b_, s_:s_ + args.seq] for b_, s_ in sel])
            yb = torch.stack([Yt[b_, s_:s_ + args.seq] for b_, s_ in sel])
            mb = torch.stack([Mt[b_, s_:s_ + args.seq] for b_, s_ in sel]).clone()
            mb[:, :args.warmup] = 0
            pred, _ = model(xb, St[b])
            per = sample_nse(pred, yb, mb)
            ng = len(bs) // len(groups)
            risks = torch.stack([per[i * ng:(i + 1) * ng].mean() for i in range(len(groups))])
            if args.method == "erm":
                loss = per.mean()
            elif args.method == "vrex":
                loss = risks.mean() + args.beta * risks.var()
            else:
                with torch.no_grad():
                    q = q * torch.exp(args.eta * risks.detach())
                    q = q / q.sum()
                loss = (q * risks).sum()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item()
        sched.step()
        if (ep + 1) % 10 == 0:
            print("epoch %02d loss %.4f" % (ep + 1, tot / steps), flush=True)

    model.eval()
    pall = np.full(Y.shape, np.nan, dtype=np.float32)
    with torch.no_grad():
        for s0 in range(0, T - args.seq, 30):
            p, _ = model(Xt[:, s0:s0 + args.seq], St)
            pall[:, s0 + args.seq - 30:s0 + args.seq] = p[:, -30:].cpu().numpy()
    pall = pall * y_std + y_mean

    def score(idx):
        kr, kl = [], []
        for b in idx:
            ok = np.isfinite(Y[b]) & np.isfinite(pall[b])
            if ok.sum() < args.min_days:
                continue
            o, p = Y[b][ok], pall[b][ok]
            a1 = kge(o, p)
            a2 = kge(np.log1p(o), np.log1p(np.maximum(p, 0)))
            if np.isfinite(a1):
                kr.append(a1)
            if np.isfinite(a2):
                kl.append(a2)
        return (float(np.median(kr)) if kr else np.nan,
                float(np.median(kl)) if kl else np.nan, len(kr))

    res = dict(dataset=os.path.basename(args.npz), arch=args.arch,
               holdout=args.holdout, method=args.method, seed=args.seed,
               id=score(id_idx), ood=score(ood_idx))
    print("[%s|%s|%s|s%d] ID log %.3f | OOD log %.3f | gap %+.3f"
          % (res["dataset"], args.arch, args.method, args.seed,
             res["id"][1], res["ood"][1], res["id"][1] - res["ood"][1]))
    json.dump(res, open(os.path.join(args.out, "result.json"), "w"), indent=1)
    np.save(os.path.join(args.out, "preds.npy"), pall)


if __name__ == "__main__":
    main()
