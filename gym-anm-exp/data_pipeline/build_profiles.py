"""Build train/validation/test profile banks for ANM6 from real 15-minute data.

Sources (already resampled to 15 min with convert_dataset_15min.py):
- loads: PQSf three-phase active power of ipin04, ipin07, ipin02 (kW);
- PV: PVGIS Padova 2005-2023, 1 kWp (W);
- wind: T1 SCADA 2018, 3.6 MW turbine (kW).

The sources cover different periods, so they cannot be joined on timestamps.
Every split has its own pool of whole days for each source and an episode day
combines one day drawn from each pool (loads are drawn per node). Days never
cross splits, so there is no leakage, but correlations between weather and
consumption are not preserved.

Each split is saved as one scenario of consecutive days, in the format read by
rl_agents/profiles.py (shape (1, days * 96, devices)), plus a JSON manifest.

Usage: python -m data_pipeline.build_profiles [--datasets DIR] [--out DIR]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

STEPS_PER_DAY = 96
SPLITS = ("train", "validation", "test")
SPLIT_FRACTIONS = (0.70, 0.15, 0.15)
SPLIT_DAYS = {"train": 365, "validation": 60, "test": 60}
MONTHS = (7, 8, 9)  # season of the load measurements, also used for PV and wind

# Gym-ANM device -> (source node, peak MW). The train 99th percentile of each
# node is mapped to the peak; the devices' limits are 10, 30 and 30 MW.
LOADS = {1: ("ipin04", 5.0), 3: ("ipin07", 20.0), 5: ("ipin02", 25.0)}
PV_DEVICE, PV_MW = 2, 30.0
WIND_DEVICE, WIND_MW = 4, 50.0
WIND_RATED_KW = 3600.0


def whole_days(series, valid=None):
    """Return {date: 96 values} for complete days whose values are all valid."""
    days = {}
    for date, values in series.groupby(series.index.date):
        if len(values) != STEPS_PER_DAY or values.isna().any():
            continue
        if valid is not None and not valid(values):
            continue
        days[date] = values.to_numpy(dtype=np.float64)
    return days


def split_random(dates, rng):
    """Assign whole days to splits at random (for short series)."""
    dates = sorted(dates)
    order = rng.permutation(len(dates))
    n_val = max(1, round(SPLIT_FRACTIONS[1] * len(dates)))
    n_test = max(1, round(SPLIT_FRACTIONS[2] * len(dates)))
    test = [dates[i] for i in order[:n_test]]
    val = [dates[i] for i in order[n_test:n_test + n_val]]
    train = [dates[i] for i in order[n_test + n_val:]]
    return {"train": sorted(train), "validation": sorted(val), "test": sorted(test)}


def read_load(datasets, node):
    path = datasets / "PQSf" / node / f"{node}_3-phase  active pow_15min.csv"
    series = pd.read_csv(path, index_col=0, parse_dates=True)["value"]
    # Zeros are meter outages (e.g. ipin04 on 10-24 Aug), not real consumption.
    return path, whole_days(series, valid=lambda v: bool((v > 0.01).all()))


def read_pv(datasets):
    path = datasets / "PVGIS" / "pvgis_padova_2005_2023_1kwp_15min.csv"
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    series = (frame["P"] / 1000.0).clip(lower=0.0)  # W of a 1 kWp plant -> capacity factor
    series = series[series.index.month.isin(MONTHS)]
    # PVGIS starts at 00:15, so the first day of each year is incomplete and dropped.
    return path, whole_days(series)


def read_wind(datasets):
    path = datasets / "T1_15min.csv"
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    # Available power: the measured output can be curtailed, the power curve gives
    # what the wind allowed, so the larger of the two is used.
    available = np.maximum(frame["LV ActivePower (kW)"], frame["Theoretical_Power_Curve (KWh)"])
    series = (available.clip(0.0, WIND_RATED_KW) / WIND_RATED_KW)
    series = series[series.index.month.isin(MONTHS)]
    return path, whole_days(series)


def build(datasets, out_dir, seed):
    rng = np.random.default_rng(seed)
    manifest = {"seed": seed, "months": list(MONTHS), "split_days": SPLIT_DAYS,
                "sources": {}, "devices": {}, "splits": {}}
    pools = {}

    for dev_id, (node, peak_mw) in LOADS.items():
        path, days = read_load(datasets, node)
        split = split_random(days, rng)
        train_values = np.concatenate([days[d] for d in split["train"]])
        scale = peak_mw / np.percentile(train_values, 99)  # kW -> MW at the device size
        pools[("load", dev_id)] = {s: [-np.minimum(days[d] * scale, peak_mw) for d in split[s]]
                                   for s in SPLITS}
        manifest["sources"][node] = str(path)
        manifest["devices"][f"load:{dev_id}"] = {"source": node, "peak_mw": peak_mw,
                                                 "kw_to_mw_scale": scale,
                                                 "days": {s: [str(d) for d in split[s]] for s in SPLITS}}

    path, days = read_pv(datasets)
    years = {"train": range(2005, 2019), "validation": range(2019, 2022), "test": range(2022, 2024)}
    split = {s: sorted(d for d in days if d.year in years[s]) for s in SPLITS}
    pools[("gen", PV_DEVICE)] = {s: [days[d] * PV_MW for d in split[s]] for s in SPLITS}
    manifest["sources"]["pv"] = str(path)
    manifest["devices"][f"generation:{PV_DEVICE}"] = {
        "source": "PVGIS", "peak_mw": PV_MW,
        "years": {s: [years[s].start, years[s].stop - 1] for s in SPLITS}}

    path, days = read_wind(datasets)
    split = split_random(days, rng)
    pools[("gen", WIND_DEVICE)] = {s: [days[d] * WIND_MW for d in split[s]] for s in SPLITS}
    manifest["sources"]["wind"] = str(path)
    manifest["devices"][f"generation:{WIND_DEVICE}"] = {
        "source": "T1", "peak_mw": WIND_MW,
        "days": {s: [str(d) for d in split[s]] for s in SPLITS}}

    load_ids = sorted(LOADS)
    gen_ids = [PV_DEVICE, WIND_DEVICE]
    out_dir.mkdir(parents=True, exist_ok=True)
    for s in SPLITS:
        n_days = SPLIT_DAYS[s]

        def draw(key):
            pool = pools[key][s]
            idx = rng.integers(0, len(pool), size=n_days)
            return np.concatenate([pool[i] for i in idx]), idx.tolist()

        loads, gens, chosen = [], [], {}
        for dev_id in load_ids:
            values, chosen[f"load:{dev_id}"] = draw(("load", dev_id))
            loads.append(values)
        for dev_id in gen_ids:
            values, chosen[f"generation:{dev_id}"] = draw(("gen", dev_id))
            gens.append(values)

        path = out_dir / f"profiles_{s}.npz"
        np.savez(path,
                 load_p_mw=np.stack(loads, axis=1)[None],
                 gen_p_max_mw=np.stack(gens, axis=1)[None],
                 load_device_ids=np.array(load_ids),
                 gen_device_ids=np.array(gen_ids))
        manifest["splits"][s] = {"file": str(path), "days": n_days,
                                 "pool_sizes": {f"{k[0]}:{k[1]}": len(pools[k][s]) for k in pools},
                                 "pool_index_per_day": chosen}
        print(f"{s}: {n_days} days -> {path}")

    with (out_dir / "profiles.manifest.json").open("w") as file:
        json.dump(manifest, file, indent=1)


def main():
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", type=Path, default=root / "Datasets")
    parser.add_argument("--out", type=Path, default=root / "Datasets" / "processed")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    build(args.datasets, args.out, args.seed)


if __name__ == "__main__":
    main()
