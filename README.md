# Aridity limits the transferability of streamflow models

Code and instructions to reproduce the experiments in:

> **Aridity limits the transferability of streamflow models across climate environments:
> a two-continent benchmark of 1,232 catchments**
> En Hu, *Journal of Hydroinformatics* (submitted)

The study holds out entire climate environments, trains one model on the rest, and
measures what the holdout costs. It is run twice — CAMELS-US (671 catchments) and
CAMELS-AUS v2 (561) — with two model classes (EA-LSTM and a Transformer), three
domain-generalization objectives, and a classical donor-basin baseline.

---

## What is here

| File | Purpose |
|---|---|
| `prep_us.py` | build `us.npz` from the public CAMELS-US release |
| `prep_aus.py` | build `aus.npz` from the public CAMELS-AUS v2 release |
| `models.py` | EA-LSTM and Transformer encoder |
| `train.py` | leave-one-environment-out training and evaluation |
| `donor.py` | k-nearest donor-basin regionalization baseline |
| `score.py` | aggregate runs into tables and the main figure |
| `run_example.sh` | a single end-to-end run (~10 min) |

---

## Data

Both datasets are public and require no registration.

**CAMELS-US** — the arrays in this study were built from the MHPI release, which
bundles the daily forcing, streamflow and attributes with consistent dates:

```
https://mhpi-spatial.s3.us-east-2.amazonaws.com/mhpi-release/data/1-camels.zip   (247 MB)
```

Unzip and keep `1-camels/camels_daymetv2` and `1-camels/gage_id.npy`.
The original CAMELS-US data (Newman et al., 2015; Addor et al., 2017) is available
from NCAR; the MHPI archive is a repackaging of it.

**CAMELS-AUS v2** — Zenodo, doi:10.5281/zenodo.12575680. Download and unzip
`03_streamflow.zip`, `04_attributes.zip`, `02_location_boundary_area.zip` and
`05_hydrometeorology.zip` into one directory.

Neither dataset is redistributed here.

---

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.8+ and PyTorch 2.0+ (CUDA optional; the models are small enough for CPU,
though the full grid is much faster on a GPU).

---

## Quick start

```bash
# 1. build the arrays (once)
python prep_us.py  --src /path/to/1-camels/camels_daymetv2 --out data/us.npz
python prep_aus.py --root /path/to/camels_aus --out data/aus.npz

# 2. one run: hold out the arid snow-dominated environment, EA-LSTM, ERM, seed 1
python train.py --npz data/us.npz --holdout dry_high --method erm --arch lstm \
                --seed 1 --epochs 60 --out runs/dry_high_erm_s1

# 3. collect
python score.py --runs runs --out figures
```

A single run takes about 10 minutes on one RTX 4090 and prints, for example:

```
[us.npz|lstm|erm|s1] ID log 0.736 | OOD log 0.310 | gap +0.426
```

---

## Full reproduction

Every configuration in the paper is a call of the form

```bash
python train.py --npz data/<us|aus>.npz --holdout <env> --method <erm|dro|vrex> \
                --arch <lstm|transformer> --seed <0|1|2> --epochs 60 --out runs/<tag>
```

The grid is:

| Axis | Values | Count |
|---|---|---|
| Dataset | US (6 environments), AUS (3) | 9 |
| Objective | ERM, GroupDRO, V-REx | 3 |
| Architecture | EA-LSTM, Transformer | 2 |
| Seed | 0, 1, 2 | 3 |

**Main grid** (LSTM × 3 objectives × 3 seeds × 9 environments) = 81 runs.
**Architecture check** (Transformer × ERM × 3 seeds × 9 environments) = 27 runs.
**Donor baseline** is deterministic: `python donor.py --npz data/us.npz --holdout dry_high --k 1 5 20`.
**Half-environment control** uses `--ood_frac` (see the paper's Methods).

Runs parallelize trivially; the full grid is ~110 GPU-hours but all runs are
independent.

---

## Expected results (EA-LSTM, ERM, three seeds, median log-KGE)

| Environment | US gap | AUS gap |
|---|---|---|
| humid | −0.05 to −0.06 | −0.11 |
| sub-humid | −0.01 to +0.02 | +0.08 |
| arid (snow-free) | **+0.28** | **+0.36** |
| arid (snow-dominated) | **+0.49** | — |

The Transformer reproduces the same ordering (Table 5 of the paper).

---

## Notes on the setup

1. **No leakage across environments.** Input and target scalers are computed from
   training basins only; nothing from the held-out environment enters the model.
2. **Log-KGE, not KGE.** Arid catchments are intermittent (15% zero-flow days in the
   driest US environment). Log transformation makes the metric comparable across
   environments. Note that the mean-flow benchmark under KGE is 1 − √2 ≈ −0.41,
   not 0 (Knoben et al., 2019).
3. **Targets in mm/day.** Streamflow is converted from cfs using catchment area
   (`AREA_COL = 11` in the attribute matrix), so catchments of different size are
   comparable without per-basin normalisation.
4. **Deterministic given the seed.** `torch.manual_seed` and `np.random.seed` are set
   at entry; the only source of run-to-run variation is GPU nondeterminism in cuDNN.

---

## Citation

If you use this code, please cite the paper above. The datasets should be cited as:

- Newman, A. J. et al. (2015) *Hydrol. Earth Syst. Sci.* 19, 209–223.
- Addor, N. et al. (2017) *Hydrol. Earth Syst. Sci.* 21, 5293–5313.
- Fowler, K. J. A., Zhang, Z. & Hou, X. (2025) *Earth Syst. Sci. Data* 17, 4079–4095.

---

## License

MIT — see `LICENSE`.
