"""Context-aware ML anomaly detection for BESS rack cell-voltage deviation.

Idea
----
A fixed rule (SOC >= 10 % and deviation >= 50 mV) ignores operating context. Cell
voltage spread naturally changes with SOC, load and temperature, so we learn the
*expected* spread for each operating condition and flag where the measured spread
is clearly higher than expected.

Validation
----------
Leave-one-day-out: every day is scored by a model trained only on the other days,
so a day can never "learn itself" as normal.

Run from the repo root:
    python anomaly_detection/detect.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import LeaveOneGroupOut

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "BESS_data_task2.csv"
OUT = ROOT / "anomaly_detection" / "outputs"

FEATURES = ["SOC", "rack_current", "abscur_mean_30min", "rack_max_temp"]
RESIDUAL_K = 4.0         # threshold = max(K * robust sigma, MIN_EXCESS_MV)
MIN_EXCESS_MV = 10.0     # never flag less than 10 mV above expected
EVENT_GAP_S = 300        # flagged points closer than 5 min form one event
MIN_EVENT_POINTS = 3     # ignore single-sample blips
RULE_SOC, RULE_DEV = 10, 50


def load() -> pd.DataFrame:
    df = pd.read_csv(DATA)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601", utc=True)
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    return df.rename(columns={"rack_voltage_deviation_mV": "dev_mV"})


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.set_index("timestamp")
    d["abscur_mean_30min"] = d["rack_current"].abs().rolling("30min").mean()
    # sign convention in this dataset: negative current = charging (SOC rises)
    d["state"] = np.select([d["rack_current"] < -2, d["rack_current"] > 2], ["charge", "discharge"], "idle")
    d["day"] = d.index.date
    return d.reset_index()


def expected_deviation(df: pd.DataFrame) -> np.ndarray:
    """Out-of-fold prediction: each day predicted by a model that never saw it."""
    oof = np.zeros(len(df))
    for tr, te in LeaveOneGroupOut().split(df, groups=df["day"]):
        m = HistGradientBoostingRegressor(
            loss="absolute_error", max_iter=200, learning_rate=0.05,
            max_depth=4, min_samples_leaf=1000, random_state=0,
        )
        m.fit(df.iloc[tr][FEATURES], df.iloc[tr]["dev_mV"])
        oof[te] = m.predict(df.iloc[te][FEATURES])
    return oof


def soc_state_baseline(df: pd.DataFrame) -> np.ndarray:
    """Naive comparison: median spread per SOC bin x state, from the other days."""
    socbin = pd.cut(df["SOC"], [0, 5, 10, 20, 40, 60, 80, 100]).astype(str)
    out = np.zeros(len(df))
    for tr, te in LeaveOneGroupOut().split(df, groups=df["day"]):
        tr_df = df.iloc[tr].assign(socbin=socbin.iloc[tr])
        med = tr_df.groupby(["socbin", "state"])["dev_mV"].median()
        keys = pd.MultiIndex.from_arrays([socbin.iloc[te], df.iloc[te]["state"]])
        out[te] = med.reindex(keys).fillna(tr_df["dev_mV"].median()).to_numpy()
    return out


def group_events(df: pd.DataFrame, flag_col: str) -> pd.DataFrame:
    f = df[df[flag_col]].copy()
    if f.empty:
        return pd.DataFrame()
    f["event_id"] = (f["timestamp"].diff().dt.total_seconds().fillna(1e9) > EVENT_GAP_S).cumsum()
    ev = f.groupby("event_id").agg(
        start=("timestamp", "min"), end=("timestamp", "max"), points=("timestamp", "size"),
        max_dev_mV=("dev_mV", "max"), expected_median_mV=("expected_mV", "median"),
        max_excess_mV=("residual_mV", "max"), soc_min=("SOC", "min"), soc_max=("SOC", "max"),
        current_min=("rack_current", "min"), current_max=("rack_current", "max"),
        main_state=("state", lambda s: s.mode().iat[0]),
    )
    ev = ev[ev["points"] >= MIN_EVENT_POINTS].reset_index(drop=True)
    ev["duration_min"] = (ev["end"] - ev["start"]).dt.total_seconds().div(60).round(1)
    num = ev.select_dtypes("number").columns
    ev[num] = ev[num].round(1)
    return ev


def main() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    df = add_features(load())

    df["expected_mV"] = expected_deviation(df)
    df["baseline_mV"] = soc_state_baseline(df)
    df["residual_mV"] = df["dev_mV"] - df["expected_mV"]

    r = df["residual_mV"]
    sigma = 1.4826 * (r - r.median()).abs().median()
    threshold = max(RESIDUAL_K * sigma, MIN_EXCESS_MV)
    df["ml_anomaly"] = r > threshold
    df["rule_warning"] = (df["SOC"] >= RULE_SOC) & (df["dev_mV"] >= RULE_DEV)

    events_ml = group_events(df, "ml_anomaly")
    events_rule = group_events(df, "rule_warning")

    daily = df.groupby("day").agg(
        soc_min=("SOC", "min"), soc_max=("SOC", "max"),
        measured_median_mV=("dev_mV", "median"), expected_median_mV=("expected_mV", "median"),
        median_excess_mV=("residual_mV", "median"),
        pct_points_flagged_ml=("ml_anomaly", "mean"), pct_points_rule=("rule_warning", "mean"),
    )
    daily[["pct_points_flagged_ml", "pct_points_rule"]] *= 100
    daily = daily.round(1)

    n_rule, n_ml = int(df["rule_warning"].sum()), int(df["ml_anomaly"].sum())
    both = int((df["rule_warning"] & df["ml_anomaly"]).sum())
    summary = {
        "rows": len(df),
        "period": f"{df.timestamp.min():%Y-%m-%d} to {df.timestamp.max():%Y-%m-%d}",
        "model_mae_mV": round((df["dev_mV"] - df["expected_mV"]).abs().mean(), 2),
        "model_median_abs_error_mV": round((df["dev_mV"] - df["expected_mV"]).abs().median(), 2),
        "baseline_soc_state_mae_mV": round((df["dev_mV"] - df["baseline_mV"]).abs().mean(), 2),
        "robust_sigma_mV": round(sigma, 2),
        "threshold_excess_mV": round(threshold, 1),
        "rule_points": n_rule,
        "ml_points": n_ml,
        "rule_events": len(events_rule),
        "ml_events": len(events_ml),
        "rule_points_not_flagged_by_ml_pct": round(100 * (n_rule - both) / max(n_rule, 1), 1),
        "ml_points_missed_by_rule": n_ml - both,
        "ml_points_missed_by_rule_soc_below_10": int((df["ml_anomaly"] & ~df["rule_warning"] & (df["SOC"] < RULE_SOC)).sum()),
    }

    df.drop(columns=["day"]).to_parquet(OUT / "scored.parquet", index=False)
    events_ml.to_csv(OUT / "ml_anomaly_events.csv", index=False)
    events_rule.to_csv(OUT / "rule_warning_events.csv", index=False)
    daily.to_csv(OUT / "daily_summary.csv")
    pd.Series(summary).to_csv(OUT / "summary.csv", header=["value"])
    return summary


if __name__ == "__main__":
    for k, v in main().items():
        print(f"{k:42s} {v}")
