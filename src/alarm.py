"""
Step 3: overheating alarm (simple and predictable on purpose).

Alarm if:
  (a) dT/dt (3-sample moving average) > threshold for 3 samples in a row, or
  (b) temperature > 60 C (hard limit).
Threshold = 99.9th percentile of dT/dt on the training batteries' normal data x 1.5.

The dataset has no real fires, so we test with synthetic faults: a heating
ramp added to held-out cycles from a random start time.

Run:  python src/alarm.py
Out:  results/alarm_results.csv, results/models/alarm_threshold.json
"""
import os
import json
import numpy as np
import pandas as pd

from data import RESULTS, BATTERIES, load_discharges, is_valid_cycle

PERCENTILE = 99.9
SAFETY_FACTOR = 1.5
CONSECUTIVE = 3          # samples in a row above threshold
TEMP_LIMIT_C = 60.0      # hard limit
SLOPES = [0.01, 0.02, 0.05, 0.1]   # synthetic heating ramps, C/s
FIRMWARE_TEST_BATTERY = "B0005"
SEED = 0


def dtdt_ma(t, T):
    """dT/dt per sample, smoothed with a 3-sample moving average.

    Value k uses rates between samples k-3..k, so it needs k >= 3.
    Returns an array the same length as t, with NaN where not defined yet.
    """
    rate = np.full(len(t), np.nan)
    rate[1:] = np.diff(T) / np.diff(t)
    ma = np.full(len(t), np.nan)
    for k in range(3, len(t)):
        ma[k] = (rate[k] + rate[k - 1] + rate[k - 2]) / 3.0
    return ma


def run_alarm(t, T, threshold):
    """Stream the cycle sample by sample, exactly like the ESP32 does.

    Returns (index of first alarm or None, reason string).
    """
    ma = dtdt_ma(t, T)
    above = 0
    for k in range(len(t)):
        if T[k] > TEMP_LIMIT_C:
            return k, "temp_limit"
        if not np.isnan(ma[k]) and ma[k] > threshold:
            above += 1
            if above >= CONSECUTIVE:
                return k, "dTdt"
        else:
            above = 0
    return None, ""


def add_heating_ramp(t, T, slope, t0):
    """Fake fault: temperature rises by an extra `slope` C/s after time t0."""
    return T + slope * np.clip(t - t0, 0, None)


def learn_threshold(cycles):
    """99.9th percentile of normal dT/dt, times a 1.5 safety margin."""
    all_ma = np.concatenate([dtdt_ma(c["t"], c["T"]) for c in cycles])
    p = np.nanpercentile(all_ma, PERCENTILE)
    return p * SAFETY_FACTOR, p


def main():
    rng = np.random.default_rng(SEED)
    data = {b: [c for c in load_discharges(b) if is_valid_cycle(c)[0]] for b in BATTERIES}
    rows, fold_info = [], []

    # Leave-one-battery-out, same as the SoH model
    for test_bat in BATTERIES:
        train_cycles = [c for b in BATTERIES if b != test_bat for c in data[b]]
        thr, p999 = learn_threshold(train_cycles)
        test_cycles = data[test_bat]

        # False alarms on clean held-out cycles (target 0)
        false_alarms = sum(run_alarm(c["t"], c["T"], thr)[0] is not None for c in test_cycles)
        fold_info.append({"test_battery": test_bat, "p99_9": p999, "threshold": thr,
                          "clean_cycles": len(test_cycles), "false_alarms": false_alarms})
        print(f"held-out {test_bat}: threshold {thr:.4f} C/s, "
              f"false alarms {false_alarms}/{len(test_cycles)}")

        if test_bat == FIRMWARE_TEST_BATTERY:
            with open(os.path.join(RESULTS, "models", "alarm_threshold.json"), "w") as f:
                json.dump({"threshold_c_per_s": thr, "p99_9": p999,
                           "train_batteries": [b for b in BATTERIES if b != test_bat]}, f, indent=2)

        # Inject faults into every held-out cycle, for every slope
        for c in test_cycles:
            t0 = rng.uniform(0.1, 0.6) * c["t"][-1]   # random ramp start time
            for slope in SLOPES:
                T_fault = add_heating_ramp(c["t"], c["T"], slope, t0)
                k, reason = run_alarm(c["t"], T_fault, thr)
                detected = k is not None and c["t"][k] >= t0
                rows.append({"test_battery": test_bat, "slope": slope,
                             "detected": detected,
                             "early_alarm": k is not None and c["t"][k] < t0,
                             "time_to_alarm_s": c["t"][k] - t0 if detected else np.nan,
                             "reason": reason if detected else ""})

    df = pd.DataFrame(rows)
    folds = pd.DataFrame(fold_info)
    summary = df.groupby("slope").agg(
        cycles=("detected", "size"),
        detection_rate_pct=("detected", lambda s: 100 * s.mean()),
        avg_time_to_alarm_s=("time_to_alarm_s", "mean"),
        by_dTdt=("reason", lambda s: (s == "dTdt").sum()),
        by_temp_limit=("reason", lambda s: (s == "temp_limit").sum()),
        alarm_before_fault=("early_alarm", "sum"),
    ).reset_index()
    summary["false_alarms_clean"] = folds["false_alarms"].sum()
    summary["clean_cycles"] = folds["clean_cycles"].sum()
    summary.to_csv(os.path.join(RESULTS, "alarm_results.csv"), index=False, float_format="%.2f")
    folds.to_csv(os.path.join(RESULTS, "alarm_thresholds.csv"), index=False, float_format="%.4f")
    print(summary.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
