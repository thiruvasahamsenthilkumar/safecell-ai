"""
Step 1: load the NASA PCoE battery data and compute 4 features per discharge cycle.

Run:  python src/data.py
Out:  results/features.csv, results/soh_vs_cycle.png, results/voltage_curves.png
"""
import os
import numpy as np
import pandas as pd
import scipy.io
import matplotlib
matplotlib.use("Agg")  # save plots to files, no window
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, "data", "raw")
RESULTS = os.path.join(ROOT, "results")

BATTERIES = ["B0005", "B0006", "B0007", "B0018"]
RATED_CAPACITY_AH = 2.0  # from the dataset README: "rated capacity (2Ahr)"

# Feature window: every discharge passes from 3.8 V down to 3.5 V
V_START = 3.8
V_END = 3.5
FEATURES = ["dt_window", "temp_start", "temp_rise", "v_drop_60s"]


def load_discharges(name):
    """Return a list of discharge cycles for one battery.

    The .mat file is nested: mat[name]['cycle'][0][k] is one operation
    (charge / discharge / impedance). We keep only 'discharge'.
    """
    mat = scipy.io.loadmat(os.path.join(RAW_DIR, name + ".mat"))
    ops = mat[name][0, 0]["cycle"][0]
    cycles = []
    for op in ops:
        if op["type"][0] != "discharge":
            continue
        d = op["data"][0, 0]
        cycles.append({
            "t": d["Time"][0].astype(float),               # seconds from cycle start
            "v": d["Voltage_measured"][0].astype(float),   # volts
            "i": d["Current_measured"][0].astype(float),   # amps (negative = discharge)
            "T": d["Temperature_measured"][0].astype(float),  # deg C
            "capacity": float(d["Capacity"][0][0]) if d["Capacity"].size else np.nan,
        })
    return cycles


def compute_features(t, v, T):
    """Compute the 4 features from one discharge cycle.

    Uses the FIRST sample at or below each voltage level, so the ESP32 can
    do exactly the same thing sample by sample (no look-ahead needed).
    Returns a dict, or None if the cycle is broken.
    """
    # v_drop_60s: voltage at t=0 minus voltage at the first sample with t >= 60 s
    idx60 = np.where(t >= 60.0)[0]
    if len(idx60) == 0:
        return None
    v_drop_60s = v[0] - v[idx60[0]]

    # Window start: first sample at or below 3.8 V
    below_start = np.where(v <= V_START)[0]
    if len(below_start) == 0:
        return None
    i0 = below_start[0]

    # Window end: first sample at or below 3.5 V, after the window start
    below_end = np.where(v[i0:] <= V_END)[0]
    if len(below_end) == 0:
        return None
    i1 = i0 + below_end[0]

    return {
        "dt_window": t[i1] - t[i0],      # shrinks as the cell ages
        "temp_start": T[i0],
        "temp_rise": T[i1] - T[i0],
        "v_drop_60s": v_drop_60s,        # bigger drop = higher internal resistance
    }


def is_valid_cycle(c):
    """Basic sanity checks; returns (ok, reason)."""
    if not np.isfinite(c["capacity"]) or c["capacity"] <= 0:
        return False, "bad capacity"
    if len(c["t"]) < 10:
        return False, "too few samples"
    if np.any(np.diff(c["t"]) <= 0):
        return False, "time not increasing"
    return True, ""


def build_features():
    rows = []
    for name in BATTERIES:
        cycles = load_discharges(name)
        dropped = {}
        for k, c in enumerate(cycles):
            ok, reason = is_valid_cycle(c)
            feats = compute_features(c["t"], c["v"], c["T"]) if ok else None
            if ok and feats is None:
                reason = "window 3.8->3.5 V not found"
            if feats is None:
                dropped[reason] = dropped.get(reason, 0) + 1
                continue
            rows.append({"battery": name, "cycle": k + 1, **feats,
                         "capacity_ah": c["capacity"],
                         "soh": 100.0 * c["capacity"] / RATED_CAPACITY_AH})
        kept = sum(r["battery"] == name for r in rows)
        print(f"{name}: {len(cycles)} discharge cycles, kept {kept}, dropped {dropped or 0}")
    return pd.DataFrame(rows)


def plot_voltage_curves():
    """Show early/mid/late discharge curves so we can see the 3.8->3.5 V window."""
    fig, axes = plt.subplots(1, len(BATTERIES), figsize=(16, 4), sharey=True)
    for ax, name in zip(axes, BATTERIES):
        cycles = load_discharges(name)
        for k in [0, len(cycles) // 2, len(cycles) - 1]:
            ax.plot(cycles[k]["t"], cycles[k]["v"], label=f"cycle {k + 1}")
        ax.axhline(V_START, color="gray", ls="--")
        ax.axhline(V_END, color="gray", ls="--")
        ax.set_title(name)
        ax.set_xlabel("time (s)")
        ax.legend()
    axes[0].set_ylabel("voltage (V)")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "voltage_curves.png"), dpi=120)


def plot_soh(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, g in df.groupby("battery"):
        ax.plot(g["cycle"], g["soh"], ".-", label=name)
    ax.set_xlabel("discharge cycle")
    ax.set_ylabel("SoH (%)")
    ax.set_title("SoH vs cycle (capacity / 2.0 Ah)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "soh_vs_cycle.png"), dpi=120)


if __name__ == "__main__":
    os.makedirs(RESULTS, exist_ok=True)
    plot_voltage_curves()
    df = build_features()
    df.to_csv(os.path.join(RESULTS, "features.csv"), index=False)
    plot_soh(df)
    print(f"Saved {len(df)} rows to results/features.csv")
    print(df.groupby("battery")[FEATURES + ["soh"]].agg(["min", "max"]).round(3).T)
